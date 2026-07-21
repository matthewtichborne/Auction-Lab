"""Tests for the experiment-grid configuration schema.

Covers parsing/validation of the JSON config format, CLI-argument
flattening, and the methodology-warning checks that flag candidate caps /
low refinement caps / max_bundle_size appearing outside an explicitly
labelled robustness arm. No live LLM/API calls.
"""

from __future__ import annotations

import json
import warnings

import pytest
from pydantic import ValidationError

from auctionlab.experiments.grid_config import (
    ExperimentArm,
    ExperimentGrid,
    SafetyLimits,
    grid_methodology_warnings,
    load_experiment_grid,
)


class TestSafetyLimitsSemantics:
    """``null``/omitted = unbounded; ``0`` is rejected on the new field
    names and accepted only as a deprecated (warned) alias spelling."""

    def test_defaults_are_null(self):
        sl = SafetyLimits()
        assert sl.per_bidder_refinement_query_limit is None
        assert sl.global_refinement_query_safety_limit is None

    def test_zero_rejected_on_new_field_name_per_bidder(self):
        with pytest.raises(ValidationError):
            SafetyLimits(per_bidder_refinement_query_limit=0)

    def test_zero_rejected_on_new_field_name_global(self):
        with pytest.raises(ValidationError):
            SafetyLimits(global_refinement_query_safety_limit=0)

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            SafetyLimits(per_bidder_refinement_query_limit=-1)
        with pytest.raises(ValidationError):
            SafetyLimits(global_refinement_query_safety_limit=-1)

    def test_positive_new_field_name_accepted(self):
        sl = SafetyLimits(
            per_bidder_refinement_query_limit=3,
            global_refinement_query_safety_limit=200,
        )
        assert sl.per_bidder_refinement_query_limit == 3
        assert sl.global_refinement_query_safety_limit == 200

    def test_legacy_alias_zero_converts_to_null_with_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sl = SafetyLimits(
                max_refinement_queries_per_bidder=0,
                max_total_refinement_queries=0,
            )
        assert sl.per_bidder_refinement_query_limit is None
        assert sl.global_refinement_query_safety_limit is None
        assert len(caught) == 2
        assert all(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_legacy_alias_positive_value_carries_over_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sl = SafetyLimits(
                max_refinement_queries_per_bidder=5,
                max_total_refinement_queries=100,
            )
        assert sl.per_bidder_refinement_query_limit == 5
        assert sl.global_refinement_query_safety_limit == 100
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_new_field_name_wins_over_legacy_alias_when_both_given(self):
        sl = SafetyLimits.model_validate({
            "per_bidder_refinement_query_limit": 7,
            "max_refinement_queries_per_bidder": 99,
        })
        assert sl.per_bidder_refinement_query_limit == 7

    def test_deprecated_property_views_reflect_null_as_zero(self):
        sl = SafetyLimits()
        assert sl.max_refinement_queries_per_bidder == 0
        assert sl.max_total_refinement_queries == 0
        sl2 = SafetyLimits(per_bidder_refinement_query_limit=4)
        assert sl2.max_refinement_queries_per_bidder == 4


def test_default_arm_has_sensible_defaults():
    arm = ExperimentArm(name="a1")
    assert arm.label == "final"
    assert arm.environment.num_goods == 8
    assert arm.proxy_candidate_policy.max_candidate_bundles is None
    assert arm.safety_limits.per_bidder_refinement_query_limit is None
    assert arm.safety_limits.global_refinement_query_safety_limit is None
    # Deprecated 0-means-unlimited views still work for old code.
    assert arm.safety_limits.max_refinement_queries_per_bidder == 0
    assert arm.safety_limits.max_total_refinement_queries == 0


def test_rejects_invalid_feedback_rule():
    with pytest.raises(ValidationError):
        ExperimentArm(name="a1", sealed_treatment={"sealed_feedback_rule": "bogus"})


def test_rejects_invalid_proxy_type():
    with pytest.raises(ValidationError):
        ExperimentArm(name="a1", proxy_candidate_policy={"proxy_type": "bogus"})


def test_duplicate_arm_names_rejected():
    arm = ExperimentArm(name="dup")
    with pytest.raises(ValidationError, match="duplicate arm names"):
        ExperimentGrid(grid_name="g", grid_type="final", arms=[arm, arm.model_copy()])


class TestToCliArgs:
    def test_flattens_all_categories(self):
        arm = ExperimentArm(
            name="a1",
            environment={"num_goods": 6, "num_bidders": 6, "scenario_seed": 3},
            proxy_candidate_policy={
                "use_interest_map": True,
                "use_provisional_valuations": True,
                "max_candidate_bundles": 20,
            },
            valuation_calibration={"discount_inferred": True},
            sealed_treatment={"sealed_feedback_rule": "competitive", "sealed_elicitation_rounds": 2},
            clock_treatment={"top_k": [1, 2, 3]},
            safety_limits={"global_refinement_query_safety_limit": 500},
            elicited_clock=True,
        )
        args = arm.to_cli_args()

        assert "--num-goods" in args and args[args.index("--num-goods") + 1] == "6"
        assert "--use-interest-map" in args
        assert "--use-provisional-valuations" in args
        assert args[args.index("--max-candidate-bundles") + 1] == "20"
        assert "--discount-inferred" in args
        assert args[args.index("--sealed-feedback-rule") + 1] == "competitive"
        assert args[args.index("--sealed-elicitation-rounds") + 1] == "2"
        assert args[args.index("--top-k") + 1 : args.index("--top-k") + 4] == ["1", "2", "3"]
        assert args[args.index("--max-total-refinement-queries") + 1] == "500"
        assert "--elicited-clock" in args

    def test_omits_unset_optional_flags(self):
        arm = ExperimentArm(name="a1")
        args = arm.to_cli_args()
        assert "--max-candidate-bundles" not in args
        assert "--max-bundle-size" not in args
        assert "--ask-initial-question" not in args
        assert "--elicited-clock" not in args
        # The core ask this schema exists to prevent: a null/omitted limit
        # must never surface as an ambiguous literal "0" on the command line.
        assert "--max-refinement-queries-per-bidder" not in args
        assert "--max-total-refinement-queries" not in args

    def test_null_per_bidder_limit_emits_no_flag(self):
        arm = ExperimentArm(
            name="a1", safety_limits={"per_bidder_refinement_query_limit": None}
        )
        assert "--max-refinement-queries-per-bidder" not in arm.to_cli_args()

    def test_positive_per_bidder_limit_emits_legacy_flag_with_value(self):
        arm = ExperimentArm(
            name="a1", safety_limits={"per_bidder_refinement_query_limit": 3}
        )
        args = arm.to_cli_args()
        assert args[args.index("--max-refinement-queries-per-bidder") + 1] == "3"

    def test_null_global_safety_limit_emits_no_flag(self):
        arm = ExperimentArm(
            name="a1", safety_limits={"global_refinement_query_safety_limit": None}
        )
        assert "--max-total-refinement-queries" not in arm.to_cli_args()

    def test_positive_global_safety_limit_emits_legacy_flag_with_value(self):
        arm = ExperimentArm(
            name="a1", safety_limits={"global_refinement_query_safety_limit": 200}
        )
        args = arm.to_cli_args()
        assert args[args.index("--max-total-refinement-queries") + 1] == "200"

    def test_hybrid_flags_only_present_for_hybrid_proxy_type(self):
        arm = ExperimentArm(name="a1", proxy_candidate_policy={"proxy_type": "llm"})
        assert "--hybrid-alpha" not in arm.to_cli_args()

        hybrid_arm = ExperimentArm(name="a2", proxy_candidate_policy={"proxy_type": "hybrid"})
        args = hybrid_arm.to_cli_args()
        assert "--hybrid-alpha" in args
        assert "--hybrid-delta" in args

    def test_extra_cli_args_escape_hatch(self):
        arm = ExperimentArm(
            name="a1",
            extra_cli_args={"verbose": True, "est_tok_per_vq": 150, "skip_this": False},
        )
        args = arm.to_cli_args()
        assert "--verbose" in args
        assert args[args.index("--est-tok-per-vq") + 1] == "150"
        assert "--skip-this" not in args


class TestLoadExperimentGrid:
    @pytest.mark.parametrize(
        "path",
        [
            "configs/value_calibration_example.json",
        ],
    )
    def test_value_calibration_config_is_not_a_grid(self, path):
        # Sanity check: the value-calibration config has a different shape
        # and must NOT be parseable as an ExperimentGrid.
        with pytest.raises(Exception):
            load_experiment_grid(path)

    @pytest.mark.parametrize(
        "path",
        [
            "configs/auction_development_grid_example.json",
            "configs/final_experiment_grid_example.json",
            "configs/final_experiment_grid_example_8x8.json",
            "configs/final_experiment_grid_example_10x10.json",
        ],
    )
    def test_example_grid_configs_parse_and_have_no_methodology_warnings(self, path):
        grid = load_experiment_grid(path)
        assert len(grid.arms) > 0
        assert grid_methodology_warnings(grid) == []

    def test_final_grids_cover_the_expected_sealed_treatment_sweep(self):
        expected_sealed_arms = {
            "final_sealed_none_r0",
            "final_sealed_allocated_bundle_r1",
            "final_sealed_allocated_bundle_r3",
            "final_sealed_competitive_r3",
            "final_sealed_all_provisional_r3",
            "final_sealed_all_provisional_r5",
            "final_sealed_all_valued_bundles_r3",
        }
        expected_clock_arms = {"final_clock_topk1", "final_clock_topk2", "final_clock_topk3"}

        for path, (num_goods, num_bidders) in {
            "configs/final_experiment_grid_example.json": (6, 6),
            "configs/final_experiment_grid_example_8x8.json": (8, 8),
            "configs/final_experiment_grid_example_10x10.json": (10, 10),
        }.items():
            grid = load_experiment_grid(path)
            arm_names = {arm.name for arm in grid.arms}
            assert expected_sealed_arms <= arm_names
            assert expected_clock_arms <= arm_names

            for arm in grid.arms:
                assert arm.environment.num_goods == num_goods
                assert arm.environment.num_bidders == num_bidders
                # Implementation budget: generous, not tuned for welfare.
                assert arm.proxy_candidate_policy.pv_max_tokens >= 6000
                # No candidate cap / per-bidder limit outside robustness.
                assert arm.proxy_candidate_policy.max_candidate_bundles is None
                assert arm.safety_limits.per_bidder_refinement_query_limit is None
                # A high, explicit (not zero) global safety backstop.
                assert arm.safety_limits.max_total_refinement_queries > 0

    def test_development_grid_uses_high_total_cap_not_zero(self):
        grid = load_experiment_grid("configs/auction_development_grid_example.json")
        for arm in grid.arms:
            assert arm.safety_limits.global_refinement_query_safety_limit >= 200
            assert arm.proxy_candidate_policy.pv_max_tokens >= 6000
            # Only the explicitly labelled query-budget robustness arm may
            # set a per-bidder limit, and it must be a positive value, not 0.
            if arm.label == "robustness" and "query_budget" in arm.name:
                assert arm.safety_limits.per_bidder_refinement_query_limit == 3
            else:
                assert arm.safety_limits.per_bidder_refinement_query_limit is None

    def test_no_example_config_contains_a_literal_zero_refinement_limit(self):
        """The exact regression this rename exists to prevent: no arm in
        any example config should carry a literal 0 for either refinement
        limit -- unset is null, and a deliberate cap is a positive int."""
        for path in [
            "configs/auction_development_grid_example.json",
            "configs/final_experiment_grid_example.json",
            "configs/final_experiment_grid_example_8x8.json",
            "configs/final_experiment_grid_example_10x10.json",
        ]:
            grid = load_experiment_grid(path)
            for arm in grid.arms:
                assert arm.safety_limits.per_bidder_refinement_query_limit != 0
                assert arm.safety_limits.global_refinement_query_safety_limit != 0

    def test_dry_run_output_for_development_grid_never_shows_zero_per_bidder_flag(
        self, tmp_path, capsys
    ):
        from scripts.run_proxy_parameter_grid import dry_run

        grid = load_experiment_grid("configs/auction_development_grid_example.json")
        dry_run(grid, tmp_path)
        output = capsys.readouterr().out
        assert "--max-refinement-queries-per-bidder 0" not in output

    def test_example_configs_use_a_consistent_provider_model_pairing(self):
        for path in [
            "configs/auction_development_grid_example.json",
            "configs/final_experiment_grid_example.json",
            "configs/final_experiment_grid_example_8x8.json",
            "configs/final_experiment_grid_example_10x10.json",
        ]:
            grid = load_experiment_grid(path)
            for arm in grid.arms:
                assert arm.provider == "gemini"
                assert "gemini" in arm.model

    def test_load_rejects_malformed_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            load_experiment_grid(bad)

    def test_load_rejects_schema_violation(self, tmp_path):
        bad = tmp_path / "bad_grid.json"
        bad.write_text(json.dumps({"grid_name": "g", "grid_type": "not_a_real_type", "arms": []}))
        with pytest.raises(ValidationError):
            load_experiment_grid(bad)


class TestMethodologyWarnings:
    def _grid(self, arm_kwargs, *, grid_type="final"):
        arm = ExperimentArm(name="a1", **arm_kwargs)
        return ExperimentGrid(grid_name="g", grid_type=grid_type, arms=[arm])

    def test_warns_on_candidate_cap_outside_robustness(self):
        grid = self._grid({
            "label": "final",
            "proxy_candidate_policy": {"max_candidate_bundles": 10},
        })
        warnings = grid_methodology_warnings(grid)
        assert any("max_candidate_bundles" in w for w in warnings)

    def test_no_warning_when_candidate_cap_labelled_robustness(self):
        grid = self._grid({
            "label": "robustness",
            "proxy_candidate_policy": {"max_candidate_bundles": 10},
        })
        assert grid_methodology_warnings(grid) == []

    def test_warns_on_max_bundle_size_outside_robustness(self):
        grid = self._grid({
            "label": "development",
            "proxy_candidate_policy": {"max_bundle_size": 2},
        })
        warnings = grid_methodology_warnings(grid)
        assert any("max_bundle_size" in w for w in warnings)

    def test_warns_on_low_per_bidder_refinement_cap_outside_robustness(self):
        grid = self._grid({
            "label": "final",
            "safety_limits": {"per_bidder_refinement_query_limit": 3},
        })
        warnings = grid_methodology_warnings(grid)
        assert any("per_bidder_refinement_query_limit" in w for w in warnings)

    def test_no_warning_for_high_total_refinement_cap(self):
        # A high global safety cap is fine anywhere -- only the per-bidder
        # cap and candidate/bundle-size caps are flagged.
        grid = self._grid({
            "label": "final",
            "safety_limits": {"global_refinement_query_safety_limit": 10_000},
        })
        assert grid_methodology_warnings(grid) == []

    def test_warns_on_sealed_rounds_with_no_feedback_rule(self):
        grid = self._grid({
            "label": "final",
            "sealed_treatment": {"sealed_elicitation_rounds": 3, "sealed_feedback_rule": "none"},
        })
        warnings = grid_methodology_warnings(grid)
        assert any("no-op" in w for w in warnings)

    def test_clean_final_grid_has_no_warnings(self):
        grid = self._grid({"label": "final"})
        assert grid_methodology_warnings(grid) == []
