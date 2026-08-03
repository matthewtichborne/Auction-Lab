"""Tests for auctionlab.experiments.run_config: policies, warnings, header
formatting, and refinement-record CSV helpers.

Also covers:
- _collect_arm_stats / _collect_initial_stats helpers in the runner script
- refinement_records_to_rows output shape
"""

from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest

from auctionlab.experiments.run_config import (
    build_run_config_document,
    config_warnings,
    explicitly_set_args,
    event_policy_summary_fields,
    format_run_config,
    refinement_cap_display,
    refinement_records_to_rows,
    resolve_event_policy,
)
from auctionlab.solvers.wdp_ilp import WdpResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal args namespace with sensible defaults."""
    defaults = dict(
        provider="ollama",
        model="llama3.1:8b",
        scenario=["pc_build"],
        num_goods=6,
        num_bidders=6,
        scenario_seed=0,
        seed_type="structured",
        proxy_type="llm",
        ask_initial_question=False,
        use_interest_map=False,
        interest_map_failure_policy="raise",
        use_provisional_valuations=False,
        max_candidate_bundles=None,
        pv_max_tokens=1500,
        max_bundle_size=2,
        top_k=[1],
        max_rounds=20,
        sealed_elicitation_rounds=0,
        sealed_feedback_rule="none",
        sealed_stopping_rule="fixed_rounds",
        elicited_clock=False,
        max_refinement_queries_per_bidder=0,
        skip_baselines=False,
        ground_truth_queries=False,
        log_dir="outputs/llm_runs/curated_batch",
        event_policy="custom",
        event_incumbent_verification=True,
        event_pivotal_challengers=False,
        event_scarcity_fallbacks=False,
        event_large_correction_followup=False,
        event_correction_threshold=0.25,
        event_gate_near_zero_surplus=False,
        event_terminal_regret_audit=False,
        sealed_loser_challenger_policy="off",
        clock_top_k_frontier_policy="off",
        clock_allocation_counterfactual_frontier=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# A. Config warnings
# ---------------------------------------------------------------------------

class TestConfigWarnings:
    def test_no_warnings_for_clean_config(self):
        args = _make_args(
            sealed_elicitation_rounds=3,
            sealed_feedback_rule="competitive",
            elicited_clock=True,
            max_refinement_queries_per_bidder=3,
        )
        assert config_warnings(args) == []

    def test_warning_sealed_rounds_with_none_feedback(self):
        args = _make_args(
            sealed_elicitation_rounds=3,
            sealed_feedback_rule="none",
        )
        warnings = config_warnings(args)
        assert len(warnings) == 1
        assert "feedback_rule" in warnings[0]
        assert "none" in warnings[0]
        assert "WARNING" in warnings[0]

    def test_no_warning_sealed_rounds_zero_with_none_feedback(self):
        # rounds=0 means sealed elicitation is disabled — no warning needed
        args = _make_args(sealed_elicitation_rounds=0, sealed_feedback_rule="none")
        warnings = config_warnings(args)
        assert not any("sealed" in w for w in warnings)

    def test_note_elicited_clock_unlimited_refinements(self):
        args = _make_args(
            elicited_clock=True,
            max_refinement_queries_per_bidder=0,
        )
        warnings = config_warnings(args)
        assert any("per-bidder refinement cap: none" in w for w in warnings)

    def test_no_note_elicited_clock_with_positive_cap(self):
        args = _make_args(
            elicited_clock=True,
            max_refinement_queries_per_bidder=3,
        )
        warnings = config_warnings(args)
        assert not any("unlimited" in w for w in warnings)

    def test_multiple_warnings_accumulate(self):
        args = _make_args(
            sealed_elicitation_rounds=3,
            sealed_feedback_rule="none",
            elicited_clock=True,
            max_refinement_queries_per_bidder=0,
        )
        warnings = config_warnings(args)
        assert len(warnings) == 2


class TestEventPolicy:
    def test_custom_preserves_shared_large_correction_flag(self):
        args = _make_args(event_large_correction_followup=True)
        resolved = resolve_event_policy(args)

        assert resolved["name"] == "custom"
        assert resolved["sealed"]["large_correction_followup"] is True
        assert resolved["clock"]["large_correction_followup"] is True

    def test_recommended_resolves_mechanism_specific_design(self):
        args = _make_args(event_policy="recommended")
        resolved = resolve_event_policy(args)

        assert args.sealed_feedback_rule == "competitive"
        assert args.sealed_loser_challenger_policy == "off"
        assert args.clock_allocation_counterfactual_frontier is True
        assert args.clock_top_k_frontier_policy == "allocation_pivotal"
        assert resolved["sealed"]["incumbent_verification"] is True
        assert resolved["sealed"]["scarcity_fallbacks"] is True
        assert resolved["sealed"]["large_correction_followup"] is True
        assert resolved["clock"]["scarcity_fallbacks"] is True
        assert resolved["clock"]["large_correction_followup"] is False
        assert resolved["clock"]["terminal_stability_audit"] is True
        assert resolved["clock"]["additional_pivotal_challengers"] is False
        assert resolved["clock"]["gate_near_zero_surplus"] is False
        assert resolved["clock"]["terminal_regret_audit"] is False

    def test_recommended_rejects_granular_override(self):
        args = _make_args(event_policy="recommended")
        with pytest.raises(ValueError, match="fixed specification"):
            resolve_event_policy(args, {"event_scarcity_fallbacks"})


    def test_final_v3_uses_revealed_winner_sandwich_clock(self):
        args = _make_args(event_policy="final-v3")
        resolved = resolve_event_policy(args)

        assert resolved["sealed"]["feedback_rule"] == "competitive"
        assert resolved["sealed"]["scarcity_fallbacks"] is True
        clock = resolved["clock"]
        assert clock["framework"] == "frontier_v1"
        assert clock["frontier_vcg_single_pass"] is True
        assert clock["frontier_vcg_revealed_only"] is True
        assert clock["frontier_winner_closure"] is True
        assert clock["frontier_staged_revealed_vcg_closure"] is True
        assert clock["frontier_vcg_witness_verification"] is False

    def test_summary_fields_record_asymmetric_large_correction(self):
        args = _make_args(event_policy="recommended")
        resolve_event_policy(args)
        fields = event_policy_summary_fields(args)

        assert fields["event_policy"] == "recommended"
        assert fields["sealed_event_large_correction_followup"] is True
        assert fields["clock_event_large_correction_followup"] is False

    def test_run_config_document_records_fully_resolved_policy(self):
        args = _make_args(event_policy="recommended")
        document = build_run_config_document(
            args,
            calibration=None,
            scenarios=[],
        )

        assert document["event_policy"]["name"] == "recommended"
        assert document["event_policy"]["sealed"][
            "large_correction_followup"
        ] is True
        assert document["event_policy"]["clock"][
            "large_correction_followup"
        ] is False
        assert document["event_policy"]["clock"][
            "terminal_regret_audit"
        ] is False



# ---------------------------------------------------------------------------
# C. Header formatting
# ---------------------------------------------------------------------------

class TestFormatRunConfig:
    def _mock_scenarios(self):
        """Return a minimal mock scenario list for formatting tests."""
        import types

        s = types.SimpleNamespace()
        s.name = "pc_build_6x6_calibrated"
        s.metadata = {"num_goods": 6, "num_bidders": 6, "scenario_seed": 0}
        instance = types.SimpleNamespace()
        instance.items = list("ABCDEF")
        instance.bidder_ids = [f"b{i}" for i in range(6)]
        s.instance = instance
        return [s]

    def test_format_run_config_returns_list_of_strings(self):
        args = _make_args()
        lines = format_run_config(args, self._mock_scenarios())
        assert isinstance(lines, list)
        assert all(isinstance(l, str) for l in lines)

    def test_header_includes_provider_and_model(self):
        args = _make_args(provider="gemini", model="gemini-2.0-flash-lite")
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "gemini" in combined
        assert "gemini-2.0-flash-lite" in combined

    def test_header_includes_scenario_name(self):
        args = _make_args()
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "pc_build_6x6_calibrated" in combined

    def test_header_includes_interest_map_failure_policy(self):
        args = _make_args(interest_map_failure_policy="raise")
        combined = "\n".join(format_run_config(args, self._mock_scenarios()))
        assert "interest_map_failure_policy  raise" in combined

    def test_header_shows_num_goods_and_bidders(self):
        args = _make_args()
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "6" in combined

    def test_header_shows_proxy_sealed_disabled_when_rounds_zero(self):
        args = _make_args(sealed_elicitation_rounds=0)
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "proxy sealed" in combined
        assert "disabled" in combined

    def test_header_shows_proxy_sealed_config_when_enabled(self):
        args = _make_args(
            sealed_elicitation_rounds=3,
            sealed_feedback_rule="competitive",
            max_refinement_queries_per_bidder=3,
        )
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "competitive" in combined
        assert "rounds=3" in combined
        assert "stopping_rule=fixed_rounds" in combined

    def test_header_shows_adaptive_sealed_max_rounds(self):
        args = _make_args(
            sealed_elicitation_rounds=50,
            sealed_feedback_rule="competitive",
            sealed_stopping_rule="no_new_refinements",
        )
        combined = "\n".join(
            format_run_config(args, self._mock_scenarios())
        )
        assert "max_rounds=50" in combined
        assert "stopping_rule=no_new_refinements" in combined

    def test_header_shows_per_bidder_cap_none_when_zero(self):
        args = _make_args(
            sealed_elicitation_rounds=3,
            sealed_feedback_rule="competitive",
            max_refinement_queries_per_bidder=0,
        )
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "per-bidder refinement cap: none" in combined

    def test_header_shows_per_bidder_cap_value_when_positive(self):
        args = _make_args(
            elicited_clock=True,
            max_refinement_queries_per_bidder=3,
        )
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "per-bidder refinement cap: 3" in combined

    def test_header_shows_global_cap_none_when_zero(self):
        args = _make_args(
            elicited_clock=True,
            max_total_refinement_queries=0,
        )
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "global refinement safety cap: none" in combined

    def test_header_shows_global_cap_value_when_positive(self):
        args = _make_args(
            elicited_clock=True,
            max_total_refinement_queries=200,
        )
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "global refinement safety cap: 200" in combined

    def test_header_shows_both_caps_for_robustness_style_arm(self):
        # A positive per-bidder cap alongside the shared global safety cap.
        args = _make_args(
            elicited_clock=True,
            max_refinement_queries_per_bidder=3,
            max_total_refinement_queries=200,
        )
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "per-bidder refinement cap: 3" in combined
        assert "global refinement safety cap: 200" in combined

    def test_header_no_longer_shows_stale_field_name_or_unlimited_text(self):
        args = _make_args(
            elicited_clock=True,
            max_refinement_queries_per_bidder=0,
            max_total_refinement_queries=200,
        )
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "max_refinement_queries_per_bidder=0" not in combined
        assert "unlimited" not in combined

    def test_header_shows_person_query_mode(self):
        args = _make_args(person_query_mode="deterministic")
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "person query mode" in combined
        assert "deterministic" in combined


# ---------------------------------------------------------------------------
# C2. refinement_cap_display
# ---------------------------------------------------------------------------

class TestRefinementCapDisplay:
    def test_both_zero_display_as_none(self):
        args = _make_args(max_refinement_queries_per_bidder=0, max_total_refinement_queries=0)
        assert refinement_cap_display(args) == ("none", "none")

    def test_positive_per_bidder_displays_as_value(self):
        args = _make_args(max_refinement_queries_per_bidder=3, max_total_refinement_queries=0)
        assert refinement_cap_display(args) == ("3", "none")

    def test_positive_global_displays_as_value(self):
        args = _make_args(max_refinement_queries_per_bidder=0, max_total_refinement_queries=200)
        assert refinement_cap_display(args) == ("none", "200")

    def test_both_positive(self):
        args = _make_args(max_refinement_queries_per_bidder=3, max_total_refinement_queries=200)
        assert refinement_cap_display(args) == ("3", "200")

    def test_missing_attrs_default_to_none(self):
        args = argparse.Namespace()
        assert refinement_cap_display(args) == ("none", "none")


# ---------------------------------------------------------------------------
# D. Refinement record CSV helpers
# ---------------------------------------------------------------------------

class _FakeRefinementRecord:
    def __init__(self, bidder_id, bundle, old_value, new_value, event_type="allocated_bundle",
                 round_idx=0, reason="test reason", mechanism="proxy_sealed_vcg",
                 query_text=None, response_summary=None):
        self.bidder_id = bidder_id
        self.bundle = bundle
        self.old_value = old_value
        self.new_value = new_value
        self.event_type = event_type
        self.round_idx = round_idx
        self.reason = reason
        self.mechanism = mechanism
        self.query_text = query_text
        self.response_summary = response_summary


class TestRefinementRecordsToRows:
    def _make_records_by_bidder(self):
        return {
            "budget_gamer": [
                _FakeRefinementRecord(
                    bidder_id="budget_gamer",
                    bundle=frozenset({"CPU_BUDGET", "GPU_MID"}),
                    old_value=375.0,
                    new_value=1220.0,
                    event_type="allocated_bundle",
                    round_idx=1,
                    reason="sealed round 1: provisionally allocated",
                )
            ],
            "office_upgrader": [],
        }

    def test_returns_list_of_dicts(self):
        rows = refinement_records_to_rows("pc_build_6x6", "proxy_sealed", self._make_records_by_bidder())
        assert isinstance(rows, list)
        assert len(rows) == 1  # only budget_gamer has a record

    def test_row_has_required_keys(self):
        rows = refinement_records_to_rows("pc_build_6x6", "proxy_sealed", self._make_records_by_bidder())
        row = rows[0]
        for key in (
            "scenario", "arm", "bidder_id", "event_type", "round_idx",
            "bundle", "old_value", "new_value", "value_delta", "reason",
            "query_text", "response_summary",
        ):
            assert key in row, f"missing key: {key}"

    def test_row_values_are_correct(self):
        rows = refinement_records_to_rows("pc_build_6x6", "proxy_sealed_competitive", self._make_records_by_bidder())
        row = rows[0]
        assert row["scenario"] == "pc_build_6x6"
        assert row["arm"] == "proxy_sealed_competitive"
        assert row["bidder_id"] == "budget_gamer"
        assert row["event_type"] == "allocated_bundle"
        assert row["round_idx"] == 1
        assert "{CPU_BUDGET,GPU_MID}" == row["bundle"]
        assert float(row["old_value"]) == pytest.approx(375.0)
        assert float(row["new_value"]) == pytest.approx(1220.0)
        assert float(row["value_delta"]) == pytest.approx(845.0)

    def test_empty_records_produce_empty_rows(self):
        rows = refinement_records_to_rows("pc_build_6x6", "proxy_sealed", {"bidder": []})
        assert rows == []

    def test_null_old_value_handled(self):
        rec = _FakeRefinementRecord(
            bidder_id="b1",
            bundle=frozenset({"CPU_HIGH"}),
            old_value=None,
            new_value=500.0,
        )
        rows = refinement_records_to_rows("s", "arm", {"b1": [rec]})
        assert rows[0]["old_value"] == ""
        assert rows[0]["value_delta"] == ""

    def test_multiple_bidders_sorted(self):
        records = {
            "z_bidder": [_FakeRefinementRecord("z_bidder", frozenset({"A"}), 100.0, 200.0)],
            "a_bidder": [_FakeRefinementRecord("a_bidder", frozenset({"B"}), 50.0, 150.0)],
        }
        rows = refinement_records_to_rows("s", "arm", records)
        assert rows[0]["bidder_id"] == "a_bidder"
        assert rows[1]["bidder_id"] == "z_bidder"

    def test_vcg_witness_hits_are_exported_when_outcomes_are_supplied(self):
        records = self._make_records_by_bidder()
        target = frozenset({"CPU_BUDGET", "GPU_MID"})
        rows = refinement_records_to_rows(
            "pc_build_6x6",
            "proxy_sealed",
            records,
            final_allocation={"budget_gamer": frozenset()},
            reported_vcg_counterfactuals={
                "office_upgrader": WdpResult(
                    allocation={"budget_gamer": target},
                    welfare=1220.0,
                ),
                "budget_gamer": WdpResult(
                    allocation={"office_upgrader": frozenset({"CPU_BUDGET"})},
                    welfare=200.0,
                ),
            },
            full_info_allocation={"budget_gamer": target},
            full_info_vcg_counterfactuals={
                "office_upgrader": WdpResult(
                    allocation={"budget_gamer": frozenset({"GPU_MID"})},
                    welfare=900.0,
                )
            },
        )

        row = rows[0]
        assert row["appears_in_final_allocation"] is False
        assert row["reported_vcg_counterfactual_count"] == 1
        assert row["appears_in_reported_vcg_counterfactual"] is True
        assert row["appears_in_any_reported_vcg_witness"] is True
        assert row["appears_in_full_info_allocation"] is True
        assert row["full_info_vcg_counterfactual_count"] == 0
        assert row["appears_in_any_full_info_vcg_witness"] is True


# ---------------------------------------------------------------------------
# E. Token accounting helpers (collect_arm_stats / collect_initial_stats)
# ---------------------------------------------------------------------------

class TestCollectArmStats:
    def test_collect_arm_stats_returns_expected_keys(self):
        from auctionlab.experiments.run_config import collect_arm_stats
        from auctionlab.llm.logging import CallTypeStats
        stats = {
            "value_query": CallTypeStats(calls=6, input_tokens=1000, output_tokens=200),
            "demand_query": CallTypeStats(calls=2, input_tokens=400, output_tokens=80),
        }
        result = collect_arm_stats(stats)
        assert result["vq"] == 6
        assert result["dq"] == 2
        assert result["nl"] == 0
        assert result["tok_in"] == 1400
        assert result["tok_out"] == 280

    def test_collect_initial_stats_returns_proxy_nl_counts(self):
        from auctionlab.experiments.run_config import collect_initial_stats
        from auctionlab.llm.logging import CallTypeStats
        stats = {
            "proxy_nl_gen": CallTypeStats(calls=6, input_tokens=3000, output_tokens=600),
            "proxy_interest_map": CallTypeStats(calls=6, input_tokens=2000, output_tokens=400),
            "proxy_interest_map_complement_entailment": CallTypeStats(
                calls=2, input_tokens=500, output_tokens=100
            ),
            "proxy_provisional_valuations": CallTypeStats(calls=6, input_tokens=8000, output_tokens=1600),
            "person_answer_semantic_extraction": CallTypeStats(
                calls=6, input_tokens=900, output_tokens=180
            ),
            "nl_question": CallTypeStats(
                calls=6, input_tokens=1200, output_tokens=300
            ),
        }
        result = collect_initial_stats(stats)
        assert result["vq"] == 0
        assert result["dq"] == 0
        assert result["nl"] == 20  # 6+6+2+6
        assert result["tok_in"] == 14700
        assert result["person_tok_in"] == 1200
        assert result["person_tok_out"] == 300
        assert result["proxy_tok_in"] == 13500
        assert result["proxy_tok_out"] == 2700
        assert result["verification_calls"] == 6
        assert result["verification_tok_in"] == 900
        assert "token_accounting_note" in result

# ---------------------------------------------------------------------------
# F. explicitly_set_args
# ---------------------------------------------------------------------------

class TestExplicitlySetArgs:
    def test_detects_long_flags(self):
        with patch.object(sys, "argv", ["prog", "--num-goods", "6", "--model=gemini"]):
            result = explicitly_set_args()
        assert "num_goods" in result
        assert "model" in result

    def test_does_not_include_positional(self):
        with patch.object(sys, "argv", ["prog", "pc_build"]):
            result = explicitly_set_args()
        assert "pc_build" not in result

    def test_equal_sign_style_parsed(self):
        with patch.object(sys, "argv", ["prog", "--sealed-feedback-rule=competitive"]):
            result = explicitly_set_args()
        assert "sealed_feedback_rule" in result

    def test_negative_boolean_flag_maps_to_canonical_destination(self):
        with patch.object(
            sys,
            "argv",
            ["prog", "--no-event-incumbent-verification"],
        ):
            result = explicitly_set_args()
        assert "event_incumbent_verification" in result
        assert "no_event_incumbent_verification" not in result
