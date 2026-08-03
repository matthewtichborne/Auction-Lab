"""CLI-parsing tests for ``examples/run_live_llm_curated_batch.py``.

Only exercises ``parse_args()`` (imported directly from the script by file
path) -- no live LLM/API call, no auction mechanism invoked. Covers the
``--size-discount-family``/``--size-discount-k0``/``--size-discount-gamma``
flags added alongside ``--epsilon``/``--discount-inferred``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from auctionlab.instances.structured_spec import (
    make_pc_build_scenario_from_spec,
)
from scripts.validate_single_bidder_elicitation import (
    provisional_value_accuracy,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "run_live_llm_curated_batch.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_run_live_llm_curated_batch_cli_test_module", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_module = _load_module()
parse_args = _module.parse_args


def test_live_person_truth_matches_positive_disclosure_not_classification(
    monkeypatch,
):
    scenario = make_pc_build_scenario_from_spec(
        "scenarios/pc_build_v3/pc_build_population_16x16.json",
        num_goods=8,
        num_bidders=8,
        seed=0,
        selection_policy="coverage_stratified",
    )
    monkeypatch.setattr(
        _module,
        "make_live_client",
        lambda **_kwargs: object(),
    )

    persons = _module.make_live_persons_for_scenario(
        scenario,
        model="person-model",
        provider="openai",
        base_url=None,
        api_key=None,
        temperature=0,
        max_tokens=100,
        timeout=10,
        logger=None,
        max_parse_retries=0,
    )

    assert persons["overclocker"].expected_interested_items == {
        "CPU_HI",
        "CPU_MID",
        "MB_STD",
        "RAM_32",
        "RAM_64",
        "SSD_2TB",
    }
    assert "GPU_VALUE" not in persons[
        "overclocker"
    ].expected_interested_items
    assert "SSD_4TB" not in persons[
        "overclocker"
    ].expected_interested_items


def _parse(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_live_llm_curated_batch.py", *argv])
    return parse_args()


class TestSizeDiscountFlags:
    def test_defaults_preserve_epsilon_only_behaviour(self, monkeypatch):
        args = _parse([], monkeypatch)
        assert args.size_discount_family is None
        assert args.size_discount_k0 == 3
        assert args.size_discount_gamma == 1.0

    def test_accepts_exponential_family_and_custom_k0_gamma(self, monkeypatch):
        args = _parse(
            [
                "--discount-inferred",
                "--size-discount-family", "exponential",
                "--size-discount-k0", "2",
                "--size-discount-gamma", "0.9",
            ],
            monkeypatch,
        )
        assert args.size_discount_family == "exponential"
        assert args.size_discount_k0 == 2
        assert args.size_discount_gamma == 0.9

    def test_rejects_unknown_size_discount_family(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse(["--size-discount-family", "linear"], monkeypatch)

    def test_epsilon_and_discount_inferred_only_still_parse(self, monkeypatch):
        """Existing commands that only use --epsilon/--discount-inferred
        (no size-discount flags) must keep working unchanged."""
        args = _parse(["--epsilon", "0.8", "--discount-inferred"], monkeypatch)
        assert args.epsilon == 0.8
        assert args.discount_inferred is True
        assert args.size_discount_family is None


class TestEventPolicyFlag:
    def test_defaults_to_custom_for_backward_compatibility(self, monkeypatch):
        args = _parse([], monkeypatch)
        assert args.event_policy == "custom"

    def test_accepts_recommended_policy(self, monkeypatch):
        args = _parse(["--event-policy", "recommended"], monkeypatch)
        assert args.event_policy == "recommended"


class TestPvCalibrationConfigFlag:
    """``--pv-calibration-config`` and its interaction with the legacy flags.

    Resolution (not just parsing) is exercised here because the whole point
    of the new flag is that misuse fails loudly instead of silently doing
    nothing.
    """

    def test_defaults_to_no_calibration(self, monkeypatch):
        args = _parse([], monkeypatch)
        assert args.pv_calibration_config is None
        calibration = _module.resolve_cli_calibration(
            args, set(), warn=lambda _m: None
        )
        assert calibration.is_identity

    def test_accepts_a_config_path(self, monkeypatch, tmp_path):
        from auctionlab.llm.value_calibration import (
            ValueCalibration,
            write_calibration_config,
        )

        path = write_calibration_config(
            ValueCalibration(family="uniform", scale=1.75),
            tmp_path / "cal.json",
        )
        args = _parse(["--pv-calibration-config", str(path)], monkeypatch)
        assert args.pv_calibration_config == Path(path)
        calibration = _module.resolve_cli_calibration(
            args, {"pv_calibration_config"}, warn=lambda _m: None
        )
        assert calibration.scale == pytest.approx(1.75)

    def test_config_plus_legacy_flags_is_rejected(self, monkeypatch, tmp_path):
        from auctionlab.llm.value_calibration import (
            CalibrationConfigError,
            ValueCalibration,
            write_calibration_config,
        )

        path = write_calibration_config(
            ValueCalibration(family="uniform", scale=1.75),
            tmp_path / "cal.json",
        )
        args = _parse(
            [
                "--pv-calibration-config", str(path),
                "--discount-inferred",
                "--epsilon", "0.8",
            ],
            monkeypatch,
        )
        with pytest.raises(CalibrationConfigError, match="cannot be combined"):
            _module.resolve_cli_calibration(
                args,
                {"pv_calibration_config", "discount_inferred", "epsilon"},
                warn=lambda _m: None,
            )

    def test_legacy_gamma_without_discount_inferred_now_fails(self, monkeypatch):
        """The exact no-op shape that used to run silently."""
        from auctionlab.llm.value_calibration import CalibrationConfigError

        args = _parse(
            [
                "--epsilon", "1",
                "--size-discount-family", "exponential",
                "--size-discount-gamma", "0.9",
            ],
            monkeypatch,
        )
        with pytest.raises(CalibrationConfigError, match="silently do"):
            _module.resolve_cli_calibration(
                args,
                {"epsilon", "size_discount_family", "size_discount_gamma"},
                warn=lambda _m: None,
            )

    def test_help_lists_the_preferred_flag(self, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["run_live_llm_curated_batch.py", "--help"]
        )
        with pytest.raises(SystemExit):
            parse_args()
        help_text = capsys.readouterr().out
        assert "--pv-calibration-config" in help_text
        assert "DEPRECATED" in help_text


class TestLlmRoleFlags:
    @pytest.mark.parametrize("provider", ["openai", "anthropic"])
    def test_accepts_first_class_additional_providers(
        self, monkeypatch, provider
    ):
        args = _parse(["--provider", provider], monkeypatch)
        assert args.provider == provider

    def test_role_overrides_default_to_legacy_global_values(self, monkeypatch):
        args = _parse(
            ["--provider", "gemini", "--model", "global-model"],
            monkeypatch,
        )
        assert args.person_provider is None
        assert args.proxy_provider is None

        _module.resolve_llm_role_args(args)

        assert (args.person_provider, args.person_model) == (
            "gemini",
            "global-model",
        )
        assert (args.proxy_provider, args.proxy_model) == (
            "gemini",
            "global-model",
        )

    def test_person_and_proxy_can_use_distinct_providers_and_models(
        self, monkeypatch
    ):
        args = _parse(
            [
                "--person-provider", "gemini",
                "--person-model", "person-model",
                "--person-temperature", "0.1",
                "--proxy-provider", "groq",
                "--proxy-model", "proxy-model",
                "--proxy-temperature", "0.2",
            ],
            monkeypatch,
        )
        _module.resolve_llm_role_args(args)

        assert (args.person_provider, args.person_model, args.person_temperature) == (
            "gemini",
            "person-model",
            0.1,
        )
        assert (args.proxy_provider, args.proxy_model, args.proxy_temperature) == (
            "groq",
            "proxy-model",
            0.2,
        )

    @pytest.mark.parametrize(
        "model",
        ["gpt-5-mini-2025-08-07", "gpt-5.6-sol", "gpt-5.6-terra"],
    )
    def test_gpt5_live_clients_omit_temperature(self, model):
        client = _module._build_raw_live_client(
            model=model,
            provider="openai",
            base_url=None,
            api_key="test-key",
            temperature=0.0,
            max_tokens=800,
            timeout=120,
        )

        assert client.temperature is None
        assert (
            _module.effective_model_temperature("openai", model, 0.0)
            is None
        )

    @pytest.mark.parametrize(
        "model",
        ["gemini-3.5-flash-lite", "gemini-3.6-flash"],
    )
    def test_new_gemini_models_record_omitted_temperature(self, model):
        assert (
            _module.effective_model_temperature("gemini", model, 0.0)
            is None
        )

    def test_supported_temperature_is_preserved_in_provenance(self):
        assert (
            _module.effective_model_temperature(
                "anthropic", "claude-haiku-4-5-20251001", 0.0
            )
            == 0.0
        )


class TestOpeningQuestionPolicy:
    def test_canonical_is_default(self, monkeypatch):
        args = _parse([], monkeypatch)

        assert args.opening_question_policy == "canonical"
        assert args.opening_question is None

    def test_proxy_generated_and_explicit_question_parse(self, monkeypatch):
        generated = _parse(
            ["--opening-question-policy", "proxy_generated"],
            monkeypatch,
        )
        explicit = _parse(
            ["--opening-question", "What fits your needs?"],
            monkeypatch,
        )

        assert generated.opening_question_policy == "proxy_generated"
        assert explicit.opening_question == "What fits your needs?"


class TestInitialElicitationFlags:
    def test_provisional_values_imply_interest_map_and_opening_question(
        self, monkeypatch
    ):
        args = _parse(["--use-provisional-valuations"], monkeypatch)

        _module.resolve_initial_elicitation_flags(args)

        assert args.use_provisional_valuations is True
        assert args.use_interest_map is True
        assert args.ask_initial_question is True


class TestPersonQueryMode:
    def test_deterministic_is_the_default(self, monkeypatch):
        args = _parse([], monkeypatch)
        assert args.person_query_mode == "deterministic"
        assert args.ground_truth_queries is False
        _module.resolve_person_query_mode(args)
        assert args.ground_truth_queries is True

    def test_llm_query_mode_remains_available_as_robustness_treatment(
        self, monkeypatch
    ):
        args = _parse(["--person-query-mode", "llm"], monkeypatch)
        assert args.person_query_mode == "llm"
        _module.resolve_person_query_mode(args)
        assert args.ground_truth_queries is False

    def test_legacy_ground_truth_flag_still_parses(self, monkeypatch):
        args = _parse(["--ground-truth-queries"], monkeypatch)
        assert args.ground_truth_queries is True
        _module.resolve_person_query_mode(args)
        assert args.person_query_mode == "deterministic"


class TestFrozenElicitationFlags:
    def test_frozen_pack_flags_default_off(self, monkeypatch):
        args = _parse([], monkeypatch)
        assert args.elicitation_pack is None
        assert args.disclosure_pack is None
        assert args.write_elicitation_pack is None
        assert args.prepare_elicitation_only is False

    def test_accepts_prepare_and_replay_paths(self, monkeypatch):
        prepared = _parse(
            [
                "--write-elicitation-pack", "pack.json",
                "--prepare-elicitation-only",
            ],
            monkeypatch,
        )
        assert prepared.write_elicitation_pack == Path("pack.json")
        assert prepared.prepare_elicitation_only is True

        replay = _parse(
            ["--elicitation-pack", "pack.json"],
            monkeypatch,
        )
        assert replay.elicitation_pack == Path("pack.json")

    def test_accepts_fixed_disclosure_source(self, monkeypatch):
        args = _parse(
            [
                "--disclosure-pack", "source.json",
                "--write-elicitation-pack", "alternative.json",
            ],
            monkeypatch,
        )
        assert args.disclosure_pack == Path("source.json")
        assert args.write_elicitation_pack == Path("alternative.json")


class TestPvChunkingFlags:
    """--pv-chunk-size / --pv-failure-policy CLI flags (PV chunking + PV
    failure policy). No live LLM/API calls -- parse_args() only."""

    def test_pv_chunk_size_defaults_to_zero_disabled(self, monkeypatch):
        args = _parse([], monkeypatch)
        assert args.pv_chunk_size == 0

    def test_pv_chunk_size_accepts_explicit_value(self, monkeypatch):
        args = _parse(["--pv-chunk-size", "50"], monkeypatch)
        assert args.pv_chunk_size == 50

    def test_pv_failure_policy_defaults_to_raise(self, monkeypatch):
        args = _parse([], monkeypatch)
        assert args.pv_failure_policy == "raise"

    def test_pv_failure_policy_accepts_zero(self, monkeypatch):
        args = _parse(["--pv-failure-policy", "zero"], monkeypatch)
        assert args.pv_failure_policy == "zero"

    def test_pv_failure_policy_rejects_invalid_choice(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse(["--pv-failure-policy", "mass_vq"], monkeypatch)

    def test_pv_chunk_size_rejects_non_integer(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse(["--pv-chunk-size", "not-a-number"], monkeypatch)

    def test_existing_commands_without_pv_chunk_size_still_work(self, monkeypatch):
        """Regression: pre-chunking invocations must keep parsing unchanged."""
        args = _parse(
            [
                "--use-provisional-valuations",
                "--use-interest-map",
                "--pv-max-tokens",
                "6000",
            ],
            monkeypatch,
        )
        assert args.pv_chunk_size == 0
        assert args.pv_failure_policy == "raise"
        assert args.pv_max_tokens == 6000

    def test_help_lists_pv_chunk_size_and_pv_failure_policy(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_live_llm_curated_batch.py", "--help"])
        with pytest.raises(SystemExit):
            parse_args()
        help_text = capsys.readouterr().out
        assert "--pv-chunk-size" in help_text
        assert "--pv-failure-policy" in help_text


class TestInterestMapFailurePolicy:
    def test_defaults_to_raise(self, monkeypatch):
        assert _parse([], monkeypatch).interest_map_failure_policy == "raise"

    def test_accepts_all_items_debug_fallback(self, monkeypatch):
        args = _parse(
            ["--interest-map-failure-policy", "all_items"],
            monkeypatch,
        )
        assert args.interest_map_failure_policy == "all_items"

    def test_rejects_invalid_policy(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse(["--interest-map-failure-policy", "continue"], monkeypatch)


class TestStructuredResponseTokenBudgets:
    def test_defaults_are_large_enough_for_nl_and_interest_map(self, monkeypatch):
        args = _parse([], monkeypatch)
        assert args.max_tokens == 300
        assert args.person_nl_max_tokens == 1500
        assert args.interest_map_max_tokens == 1500

    def test_accepts_independent_budgets(self, monkeypatch):
        args = _parse(
            [
                "--max-tokens", "600",
                "--person-nl-max-tokens", "1800",
                "--interest-map-max-tokens", "2000",
            ],
            monkeypatch,
        )
        assert args.max_tokens == 600
        assert args.person_nl_max_tokens == 1800
        assert args.interest_map_max_tokens == 2000


class TestScenarioSelectionPolicy:
    def test_defaults_to_legacy_prefix(self, monkeypatch):
        assert _parse([], monkeypatch).selection_policy == "prefix"

    @pytest.mark.parametrize("policy", ["seeded_sample", "stratified"])
    def test_accepts_seed_sensitive_policy(self, monkeypatch, policy):
        args = _parse(["--selection-policy", policy], monkeypatch)
        assert args.selection_policy == policy
def test_single_bidder_pv_accuracy_reports_welfare_and_scale():
    truth = {
        frozenset({"A"}): 10.0,
        frozenset({"B"}): 20.0,
        frozenset({"A", "B"}): 30.0,
    }
    predicted = {
        frozenset({"A"}): 9.0,
        frozenset({"B"}): 50.0,
        frozenset({"A", "B"}): 25.0,
    }

    result = provisional_value_accuracy(predicted, truth, top_k=2)

    assert result["predicted_best_bundle"] == ["B"]
    assert result["single_bidder_welfare_efficiency"] == pytest.approx(2 / 3)
    assert result["predicted_max_to_true_max_ratio"] == pytest.approx(5 / 3)
    assert result["candidate_true_top_k_recall"] == pytest.approx(1.0)
    assert result["candidate_top_k_overlap_count"] == 2
    assert result["candidate_top_k_tie_aware_hit_rate"] == pytest.approx(1.0)
    assert "global_true_top_k_recall" not in result


def test_single_bidder_pv_accuracy_is_tie_aware_within_candidate_support():
    truth = {
        frozenset({"A"}): 10.0,
        frozenset({"B"}): 10.0,
        frozenset({"C"}): 10.0,
        frozenset({"D"}): 1.0,
        frozenset({"OUTSIDE"}): 100.0,
    }
    predicted = {
        frozenset({"A"}): 8.0,
        frozenset({"B"}): 7.0,
        frozenset({"C"}): 9.0,
        frozenset({"D"}): 20.0,
    }

    result = provisional_value_accuracy(predicted, truth, top_k=2)

    assert result["candidate_true_top_k_recall"] == pytest.approx(0.0)
    assert result["candidate_top_k_tie_aware_hit_rate"] == pytest.approx(0.5)
    assert result["candidate_true_top_tie_set_size"] == 3
    assert result["candidate_true_top_k_boundary_value"] == 10.0
