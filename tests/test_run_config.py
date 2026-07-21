"""Tests for auctionlab.experiments.run_config: presets, warnings, header
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
    PRESETS,
    apply_preset,
    config_warnings,
    explicitly_set_args,
    format_run_config,
    refinement_records_to_rows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal args namespace with sensible defaults."""
    defaults = dict(
        preset=None,
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
        use_provisional_valuations=False,
        max_candidate_bundles=None,
        pv_max_tokens=1500,
        max_bundle_size=2,
        top_k=[1],
        max_rounds=20,
        sealed_elicitation_rounds=0,
        sealed_feedback_rule="none",
        elicited_clock=False,
        max_refinement_queries_per_bidder=0,
        skip_baselines=False,
        ground_truth_queries=False,
        log_dir="outputs/llm_runs/curated_batch",
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
        assert any("unlimited" in w for w in warnings)

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


# ---------------------------------------------------------------------------
# B. Preset application
# ---------------------------------------------------------------------------

class TestPreset:
    def test_preset_exists(self):
        assert "structured-6x6-diagnostic" in PRESETS

    def test_preset_contains_expected_keys(self):
        p = PRESETS["structured-6x6-diagnostic"]
        for key in (
            "scenario", "num_goods", "num_bidders", "scenario_seed",
            "seed_type", "proxy_type",
            "ask_initial_question", "use_interest_map", "use_provisional_valuations",
            "top_k", "max_bundle_size", "sealed_elicitation_rounds",
            "sealed_feedback_rule", "elicited_clock",
            "max_refinement_queries_per_bidder", "max_rounds",
        ):
            assert key in p, f"missing key: {key}"

    def test_preset_sets_6x6_defaults(self):
        p = PRESETS["structured-6x6-diagnostic"]
        assert p["num_goods"] == 6
        assert p["num_bidders"] == 6
        assert p["scenario"] == ["pc_build"]
        assert p["sealed_feedback_rule"] == "all_valued_bundles"
        assert p["sealed_elicitation_rounds"] == 3
        assert p["max_bundle_size"] == 3
        assert p["use_provisional_valuations"] is True

    def test_apply_preset_no_preset(self):
        args = _make_args(preset=None)
        applied = apply_preset(args, set())
        assert applied == []
        assert args.num_goods == 6  # default unchanged

    def test_apply_preset_sets_values(self):
        args = _make_args(preset="structured-6x6-diagnostic", num_goods=10)
        # num_goods not in explicitly_set → overwritten by preset
        applied = apply_preset(args, set())
        assert "num_goods" in applied
        assert args.num_goods == 6

    def test_apply_preset_respects_explicit_flags(self):
        args = _make_args(preset="structured-6x6-diagnostic", num_goods=10)
        # num_goods was explicitly set on the command line
        applied = apply_preset(args, {"num_goods"})
        assert "num_goods" not in applied
        assert args.num_goods == 10  # user's explicit value preserved

    def test_apply_preset_elicited_clock_set(self):
        args = _make_args(preset="structured-6x6-diagnostic", elicited_clock=False)
        apply_preset(args, set())
        assert args.elicited_clock is True

    def test_apply_preset_elicited_clock_kept_if_explicit(self):
        args = _make_args(preset="structured-6x6-diagnostic", elicited_clock=False)
        apply_preset(args, {"elicited_clock"})
        assert args.elicited_clock is False


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

    def test_header_shows_max_ref_unlimited_when_zero(self):
        args = _make_args(
            sealed_elicitation_rounds=3,
            sealed_feedback_rule="competitive",
            max_refinement_queries_per_bidder=0,
        )
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "unlimited" in combined

    def test_header_shows_ground_truth_queries(self):
        args = _make_args(ground_truth_queries=True)
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "True" in combined

    def test_header_shows_preset_name_when_set(self):
        args = _make_args(preset="structured-6x6-diagnostic")
        lines = format_run_config(args, self._mock_scenarios())
        combined = "\n".join(lines)
        assert "structured-6x6-diagnostic" in combined


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
            "proxy_provisional_valuations": CallTypeStats(calls=6, input_tokens=8000, output_tokens=1600),
        }
        result = collect_initial_stats(stats)
        assert result["vq"] == 0
        assert result["dq"] == 0
        assert result["nl"] == 18  # 6+6+6
        assert result["tok_in"] == 13000
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
