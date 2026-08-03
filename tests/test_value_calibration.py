"""Provisional-value calibration: semantics, validation, and wiring."""

from __future__ import annotations

import argparse
import json

import pytest

from auctionlab.experiments.run_config import (
    add_calibration_fields,
    build_run_config_document,
    calibration_summary_fields,
    format_run_config,
    write_run_config_json,
)
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmInferredXorProxy, apply_value_calibration
from auctionlab.llm.schemas import LlmInterestMap
from auctionlab.llm.value_calibration import (
    NO_CALIBRATION,
    CalibrationConfigError,
    ValueCalibration,
    legacy_calibration,
    load_calibration_config,
    resolve_cli_calibration,
    write_calibration_config,
)


ITEM_DESCRIPTIONS = {"A": "item a", "B": "item b", "C": "item c", "D": "item d"}


# ---------------------------------------------------------------------------
# Core semantics
# ---------------------------------------------------------------------------

class TestFamilySemantics:
    def test_none_family_is_exact_identity(self):
        calibration = ValueCalibration(family="none")
        for value in (0.0, 1.0, 137.5, 1e6):
            for size in range(1, 8):
                assert calibration.apply(value, size) == value

    def test_none_family_ignores_budget_cap(self):
        calibration = ValueCalibration(family="none", budget_cap=True)
        assert calibration.apply(9999.0, 4, disclosed_budget=10.0) == 9999.0

    def test_uniform_applies_scale_only(self):
        calibration = ValueCalibration(
            family="uniform", scale=1.75, budget_cap=False
        )
        assert calibration.apply(100.0, 1) == pytest.approx(175.0)
        assert calibration.apply(100.0, 9) == pytest.approx(175.0)

    def test_exponential_applies_scale_and_size_factor(self):
        calibration = ValueCalibration(
            family="exponential",
            scale=2.0,
            size_gamma=0.5,
            size_threshold=3,
            budget_cap=False,
        )
        # At or below the threshold the size factor is exactly 1.
        assert calibration.apply(100.0, 3) == pytest.approx(200.0)
        assert calibration.apply(100.0, 4) == pytest.approx(100.0)
        assert calibration.apply(100.0, 5) == pytest.approx(50.0)

    def test_scale_above_one_is_accepted(self):
        # The estimator under-estimates, so the useful correction inflates.
        calibration = ValueCalibration(family="uniform", scale=1.83)
        assert calibration.scale == pytest.approx(1.83)
        assert calibration.apply(100.0, 2) == pytest.approx(183.0)

    def test_negative_results_are_clamped(self):
        calibration = ValueCalibration(family="uniform", scale=2.0)
        assert calibration.apply(-5.0, 1) == 0.0


class TestBudgetCap:
    def test_budget_cap_binds(self):
        calibration = ValueCalibration(family="uniform", scale=3.0, budget_cap=True)
        assert calibration.apply(500.0, 2, disclosed_budget=900.0) == 900.0

    def test_budget_cap_does_not_bind_below_budget(self):
        calibration = ValueCalibration(family="uniform", scale=1.2, budget_cap=True)
        assert calibration.apply(500.0, 2, disclosed_budget=900.0) == pytest.approx(600.0)

    def test_budget_cap_off_ignores_budget(self):
        calibration = ValueCalibration(family="uniform", scale=3.0, budget_cap=False)
        assert calibration.apply(500.0, 2, disclosed_budget=900.0) == pytest.approx(1500.0)

    @pytest.mark.parametrize("budget", [None, 0.0, -10.0])
    def test_missing_or_nonpositive_budget_never_caps(self, budget):
        calibration = ValueCalibration(family="uniform", scale=3.0, budget_cap=True)
        assert calibration.apply(500.0, 2, disclosed_budget=budget) == pytest.approx(1500.0)


class TestValidation:
    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_scale_rejected(self, scale):
        with pytest.raises(CalibrationConfigError, match="scale"):
            ValueCalibration(family="uniform", scale=scale)

    @pytest.mark.parametrize("gamma", [0.0, -0.5, float("inf")])
    def test_invalid_size_gamma_rejected(self, gamma):
        with pytest.raises(CalibrationConfigError, match="size_gamma"):
            ValueCalibration(family="exponential", size_gamma=gamma)

    @pytest.mark.parametrize("threshold", [-1, 2.5, "3", True])
    def test_invalid_size_threshold_rejected(self, threshold):
        with pytest.raises(CalibrationConfigError, match="size_threshold"):
            ValueCalibration(family="exponential", size_threshold=threshold)

    def test_zero_size_threshold_allowed(self):
        assert ValueCalibration(family="exponential", size_threshold=0).size_threshold == 0

    def test_unknown_family_rejected(self):
        with pytest.raises(CalibrationConfigError, match="family"):
            ValueCalibration(family="quadratic")

    def test_none_family_with_scale_is_rejected_as_inert(self):
        with pytest.raises(CalibrationConfigError, match="ignores scale"):
            ValueCalibration(family="none", scale=1.8)

    def test_uniform_family_with_gamma_is_rejected_as_inert(self):
        with pytest.raises(CalibrationConfigError, match="ignores size_gamma"):
            ValueCalibration(family="uniform", scale=1.8, size_gamma=0.9)


class TestSerialisation:
    def test_round_trip_through_json(self, tmp_path):
        original = ValueCalibration(
            family="exponential",
            scale=1.83,
            size_gamma=0.961,
            size_threshold=3,
            budget_cap=True,
            provenance={"fitted_by": "test"},
        )
        path = write_calibration_config(original, tmp_path / "cal.json")
        loaded = load_calibration_config(path)
        assert loaded.family == original.family
        assert loaded.scale == pytest.approx(original.scale)
        assert loaded.size_gamma == pytest.approx(original.size_gamma)
        assert loaded.budget_cap is True
        assert loaded.provenance["fitted_by"] == "test"
        assert loaded.source_path == str(path)

    def test_hash_ignores_provenance_and_path(self, tmp_path):
        base = ValueCalibration(family="uniform", scale=1.5)
        annotated = base.with_provenance(created_at="whenever", note="hello")
        assert annotated.config_hash() == base.config_hash()

    def test_hash_tracks_effective_parameters(self):
        a = ValueCalibration(family="uniform", scale=1.5)
        b = ValueCalibration(family="uniform", scale=1.6)
        assert a.config_hash() != b.config_hash()

    def test_unknown_config_key_rejected(self, tmp_path):
        path = tmp_path / "cal.json"
        path.write_text(
            json.dumps(
                {"schema_version": "1", "family": "uniform", "scail": 1.5}
            )
        )
        with pytest.raises(CalibrationConfigError, match="unknown calibration config keys"):
            load_calibration_config(path)

    def test_unsupported_schema_version_rejected(self, tmp_path):
        path = tmp_path / "cal.json"
        path.write_text(json.dumps({"schema_version": "9", "family": "none"}))
        with pytest.raises(CalibrationConfigError, match="schema_version"):
            load_calibration_config(path)


class TestLegacyTranslation:
    def test_discount_inferred_false_is_identity(self):
        calibration = legacy_calibration(discount_inferred=False, epsilon=0.5)
        assert calibration.is_identity
        assert calibration.apply(100.0, 5) == 100.0

    def test_matches_historical_transform(self):
        for epsilon, gamma, k0, size in (
            (0.75, 1.0, 3, 2),
            (0.5, 0.9, 1, 4),
            (1.0, 0.95, 3, 6),
        ):
            expected = 100.0 * epsilon * gamma ** max(0, size - k0)
            assert apply_value_calibration(
                100.0,
                size,
                discount_inferred=True,
                epsilon=epsilon,
                size_discount_family="exponential",
                size_discount_k0=k0,
                size_discount_gamma=gamma,
            ) == pytest.approx(expected)

    def test_compat_wrapper_still_validates(self):
        with pytest.raises(ValueError, match="size_discount_gamma"):
            apply_value_calibration(
                1.0, 1, discount_inferred=True, size_discount_gamma=0.0
            )


# ---------------------------------------------------------------------------
# CLI resolution
# ---------------------------------------------------------------------------

def _args(**overrides):
    base = dict(
        pv_calibration_config=None,
        discount_inferred=False,
        epsilon=1.0,
        size_discount_family=None,
        size_discount_k0=3,
        size_discount_gamma=1.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestCliResolution:
    def test_no_flags_means_raw(self):
        calibration = resolve_cli_calibration(_args(), set(), warn=lambda _m: None)
        assert calibration.is_identity
        assert calibration is NO_CALIBRATION

    def test_config_file_is_loaded(self, tmp_path):
        path = write_calibration_config(
            ValueCalibration(family="uniform", scale=1.5), tmp_path / "c.json"
        )
        calibration = resolve_cli_calibration(
            _args(pv_calibration_config=path), set(), warn=lambda _m: None
        )
        assert calibration.scale == pytest.approx(1.5)

    def test_config_plus_legacy_flag_is_rejected(self, tmp_path):
        path = write_calibration_config(
            ValueCalibration(family="uniform", scale=1.5), tmp_path / "c.json"
        )
        with pytest.raises(CalibrationConfigError, match="cannot be combined"):
            resolve_cli_calibration(
                _args(pv_calibration_config=path, epsilon=0.8, discount_inferred=True),
                {"pv_calibration_config", "epsilon", "discount_inferred"},
                warn=lambda _m: None,
            )

    def test_legacy_values_without_gate_is_an_error_not_a_no_op(self):
        with pytest.raises(CalibrationConfigError, match="silently do"):
            resolve_cli_calibration(
                _args(epsilon=0.8), {"epsilon"}, warn=lambda _m: None
            )

    def test_size_discount_gamma_without_family_is_an_error(self):
        with pytest.raises(CalibrationConfigError, match="size-discount-family"):
            resolve_cli_calibration(
                _args(discount_inferred=True, size_discount_gamma=0.9),
                {"discount_inferred", "size_discount_gamma"},
                warn=lambda _m: None,
            )

    def test_size_discount_k0_without_family_is_an_error(self):
        with pytest.raises(CalibrationConfigError, match="size-discount-family"):
            resolve_cli_calibration(
                _args(discount_inferred=True, size_discount_k0=2),
                {"discount_inferred", "size_discount_k0"},
                warn=lambda _m: None,
            )

    def test_legacy_epsilon_above_one_is_rejected_with_guidance(self):
        with pytest.raises(CalibrationConfigError, match="pv-calibration-config"):
            resolve_cli_calibration(
                _args(discount_inferred=True, epsilon=1.5),
                {"discount_inferred", "epsilon"},
                warn=lambda _m: None,
            )

    def test_legacy_use_emits_deprecation_warning(self):
        warnings: list[str] = []
        calibration = resolve_cli_calibration(
            _args(discount_inferred=True, epsilon=0.8),
            {"discount_inferred", "epsilon"},
            warn=warnings.append,
        )
        assert warnings and "deprecated" in warnings[0]
        assert calibration.family == "uniform"
        assert calibration.scale == pytest.approx(0.8)

    def test_legacy_exponential_translation(self):
        calibration = resolve_cli_calibration(
            _args(
                discount_inferred=True,
                epsilon=0.9,
                size_discount_family="exponential",
                size_discount_gamma=0.95,
                size_discount_k0=2,
            ),
            {
                "discount_inferred",
                "epsilon",
                "size_discount_family",
                "size_discount_gamma",
                "size_discount_k0",
            },
            warn=lambda _m: None,
        )
        assert calibration.family == "exponential"
        assert calibration.size_gamma == pytest.approx(0.95)
        assert calibration.size_threshold == 2
        # Legacy never capped at the disclosed budget; the translation must
        # not silently add a cap that changes historical results.
        assert calibration.budget_cap is False


# ---------------------------------------------------------------------------
# Proxy integration
# ---------------------------------------------------------------------------

def _make_proxy(calibration=None, *, ground_truth=None, **proxy_kwargs):
    person = LlmPersonSimulator(
        bidder_id="b1",
        scenario_description="scenario",
        person_seed="seed",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=MockLlmClient(['{"bundle_value": 400}'] * 20),
        ground_truth_valuations=ground_truth,
    )
    return LlmInferredXorProxy(
        bidder_id="b1", person=person, calibration=calibration, **proxy_kwargs
    )


def _interest_map(budget_hint=None):
    return LlmInterestMap(
        interested_items=["A", "B", "C", "D"],
        excluded_items=[],
        complementary_groups=[],
        substitute_groups=[],
        budget_hint=budget_hint,
        reasoning="test",
    )


def _atom_value(proxy, bundle):
    return next(
        atom.value for atom in proxy._cached_bid.atoms if atom.bundle == bundle
    )


class TestProxyCalibration:
    def test_replay_applies_configured_calibration(self):
        proxy = _make_proxy(ValueCalibration(family="uniform", scale=1.5))
        proxy.replay_elicitation(
            nl_question="q",
            nl_answer="a",
            provisional_raw_values={frozenset({"A"}): 100.0},
        )
        assert _atom_value(proxy, frozenset({"A"})) == pytest.approx(150.0)

    def test_replay_uses_disclosed_budget_from_interest_map(self):
        proxy = _make_proxy(
            ValueCalibration(family="uniform", scale=3.0, budget_cap=True)
        )
        proxy.replay_elicitation(
            nl_question="q",
            nl_answer="a",
            interest_map=_interest_map(budget_hint=250.0),
            provisional_raw_values={frozenset({"A", "B"}): 100.0},
        )
        assert _atom_value(proxy, frozenset({"A", "B"})) == pytest.approx(250.0)

    def test_explicit_calibration_overrides_discount_inferred_gate(self):
        proxy = _make_proxy(ValueCalibration(family="uniform", scale=2.0))
        proxy.replay_elicitation(
            nl_question="q",
            nl_answer="a",
            provisional_raw_values={frozenset({"A"}): 100.0},
            discount_inferred=False,
        )
        assert _atom_value(proxy, frozenset({"A"})) == pytest.approx(200.0)

    def test_same_raw_values_replay_under_several_calibrations(self):
        """One frozen pack, several calibrations, zero extra LLM calls."""
        raw = {
            frozenset({"A"}): 100.0,
            frozenset({"A", "B"}): 180.0,
            frozenset({"A", "B", "C", "D"}): 300.0,
        }
        configs = {
            "raw": ValueCalibration(family="none"),
            "uniform": ValueCalibration(
                family="uniform", scale=1.5, budget_cap=False
            ),
            "exponential": ValueCalibration(
                family="exponential",
                scale=1.5,
                size_gamma=0.9,
                size_threshold=3,
                budget_cap=False,
            ),
        }
        results = {}
        clients = {}
        for name, calibration in configs.items():
            proxy = _make_proxy(calibration)
            proxy.replay_elicitation(
                nl_question="q", nl_answer="a", provisional_raw_values=dict(raw)
            )
            results[name] = {
                bundle: _atom_value(proxy, bundle) for bundle in raw
            }
            clients[name] = proxy.person.client

        assert results["raw"][frozenset({"A"})] == pytest.approx(100.0)
        assert results["uniform"][frozenset({"A"})] == pytest.approx(150.0)
        assert results["exponential"][
            frozenset({"A", "B", "C", "D"})
        ] == pytest.approx(300.0 * 1.5 * 0.9)
        # Replay is the whole point: not one model call was issued.
        assert all(client.calls == [] for client in clients.values())

    def test_size_discount_cannot_push_a_superset_below_its_subset(self):
        """XOR monotonicity repair still wins over an aggressive size gamma."""
        proxy = _make_proxy(
            ValueCalibration(
                family="exponential",
                scale=1.5,
                size_gamma=0.5,
                size_threshold=3,
                budget_cap=False,
            )
        )
        proxy.replay_elicitation(
            nl_question="q",
            nl_answer="a",
            provisional_raw_values={
                frozenset({"A", "B"}): 180.0,
                frozenset({"A", "B", "C", "D"}): 300.0,
            },
        )
        # 300 * 1.5 * 0.5 = 225 would sit below the 270 subset atom, so the
        # superset is raised rather than left inconsistent.
        assert _atom_value(proxy, frozenset({"A", "B"})) == pytest.approx(270.0)
        assert _atom_value(
            proxy, frozenset({"A", "B", "C", "D"})
        ) == pytest.approx(270.0)

    def test_exact_deterministic_refinement_is_never_calibrated(self):
        bundle = frozenset({"A", "B"})
        proxy = _make_proxy(
            ValueCalibration(family="uniform", scale=2.0, budget_cap=False),
            ground_truth={bundle: 777.0, frozenset({"A"}): 100.0},
        )
        proxy.replay_elicitation(
            nl_question="q",
            nl_answer="a",
            provisional_raw_values={bundle: 100.0},
        )
        assert _atom_value(proxy, bundle) == pytest.approx(200.0)

        refined = proxy.refine_bundle_value(bundle, reason="test")
        assert refined == pytest.approx(777.0)
        assert _atom_value(proxy, bundle) == pytest.approx(777.0)

    def test_legacy_fields_still_work_without_calibration(self):
        proxy = _make_proxy(None, epsilon=0.5)
        proxy.replay_elicitation(
            nl_question="q",
            nl_answer="a",
            provisional_raw_values={frozenset({"A"}): 100.0},
            discount_inferred=True,
        )
        assert _atom_value(proxy, frozenset({"A"})) == pytest.approx(50.0)

    def test_disclosed_budget_reads_interest_map_not_ground_truth(self):
        proxy = _make_proxy(
            ValueCalibration(family="uniform", scale=1.0),
            ground_truth={frozenset({"A"}): 5000.0},
        )
        assert proxy.disclosed_budget() is None
        proxy.interest_map = _interest_map(budget_hint=42.0)
        assert proxy.disclosed_budget() == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Run auditability
# ---------------------------------------------------------------------------

class TestRunAuditability:
    def test_summary_fields_cover_every_reported_parameter(self):
        calibration = ValueCalibration(
            family="exponential", scale=1.8, size_gamma=0.96, size_threshold=3
        )
        fields = calibration_summary_fields(calibration)
        assert fields["pv_calibration_family"] == "exponential"
        assert fields["pv_calibration_scale"] == pytest.approx(1.8)
        assert fields["pv_calibration_size_gamma"] == pytest.approx(0.96)
        assert fields["pv_calibration_size_threshold"] == 3
        assert fields["pv_calibration_budget_cap"] is True
        assert fields["pv_calibration_config_hash"] == calibration.config_hash()

    def test_none_calibration_is_reported_not_omitted(self):
        fields = calibration_summary_fields(None)
        assert fields["pv_calibration_family"] == "none"
        assert fields["pv_calibration_config_hash"]

    def test_rows_are_stamped_in_place(self):
        rows = [{"scenario": "s", "efficiency": 1.0}]
        add_calibration_fields(rows, ValueCalibration(family="uniform", scale=1.4))
        assert rows[0]["pv_calibration_scale"] == pytest.approx(1.4)
        assert rows[0]["efficiency"] == 1.0

    def test_header_reports_effective_calibration(self):
        calibration = ValueCalibration(
            family="exponential", scale=1.8, size_gamma=0.96
        )
        lines = format_run_config(argparse.Namespace(), [], calibration=calibration)
        text = "\n".join(lines)
        assert "pv_calibration_family     exponential" in text
        assert "1.8" in text
        assert "0.96" in text
        assert calibration.short_hash() in text

    def test_stamped_columns_survive_into_a_result_csv(self, tmp_path):
        import csv

        from auctionlab.experiments.export import write_csv
        from auctionlab.experiments.run_config import CALIBRATION_CSV_FIELDS

        calibration = ValueCalibration(
            family="exponential", scale=1.83, size_gamma=0.961
        )
        rows = [
            {"scenario": "s1", "arm": "proxy_sealed", "efficiency": 0.97},
            {"scenario": "s1", "arm": "proxy_clock_k1", "efficiency": 0.94},
        ]
        add_calibration_fields(rows, calibration)
        path = tmp_path / "detailed.csv"
        write_csv(rows, path)

        with path.open() as handle:
            written = list(csv.DictReader(handle))
        assert len(written) == 2
        for field in CALIBRATION_CSV_FIELDS:
            assert field in written[0]
        assert written[0]["pv_calibration_family"] == "exponential"
        assert written[0]["pv_calibration_config_hash"] == calibration.config_hash()

    def test_scalability_fields_include_the_calibration_columns(self):
        from auctionlab.experiments.scalability_analysis import (
            SCALABILITY_FIELDS,
        )

        for field in (
            "pv_calibration_family",
            "pv_calibration_scale",
            "pv_calibration_size_gamma",
            "pv_calibration_size_threshold",
            "pv_calibration_budget_cap",
            "pv_calibration_config_hash",
        ):
            assert field in SCALABILITY_FIELDS

    def test_run_config_json_records_resolved_calibration(self, tmp_path):
        calibration = ValueCalibration(family="uniform", scale=1.75)
        document = build_run_config_document(
            argparse.Namespace(log_dir=str(tmp_path)),
            calibration=calibration,
            scenarios=[],
        )
        path = write_run_config_json(tmp_path / "run_config.json", document)
        written = json.loads(path.read_text())
        assert written["pv_calibration"]["family"] == "uniform"
        assert written["pv_calibration"]["scale"] == pytest.approx(1.75)
        assert written["pv_calibration_config_hash"] == calibration.config_hash()
