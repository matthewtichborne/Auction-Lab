"""Tests for scripts/validate_scenario_spec.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_scenario_spec import (
    aggregate_metrics_across_seeds,
    build_multi_seed_report,
    build_validation_report,
    compute_mechanism_benchmark_diagnostics,
    default_acceptance_thresholds,
    evaluate_acceptance,
    load_thresholds,
    main as validate_main,
)

V0_SPEC_PATH = Path("scenarios/pc_build_v1/pc_build_profiles_v0_manual.json")


@pytest.fixture(scope="module")
def v0_spec_path(tmp_path_factory) -> Path:
    if V0_SPEC_PATH.exists():
        return V0_SPEC_PATH
    from auctionlab.instances.scenario_spec import write_scenario_profile_spec
    from scripts.export_current_pc_build_profiles import build_manual_spec

    spec = build_manual_spec(seed=0)
    path = tmp_path_factory.mktemp("spec") / "pc_build_profiles_v0_manual.json"
    write_scenario_profile_spec(spec, path)
    return path


def test_validation_report_top_level_keys(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(4, 4), (6, 6)])
    assert set(report) == {
        "spec_path",
        "selection_policy",
        "scenario_seed",
        "schema_validation",
        "profile_summary",
        "economic_diagnostics",
        "acceptance",
    }
    assert report["schema_validation"]["valid"] is True
    assert report["schema_validation"]["errors"] == []


def test_validation_report_profile_summary_fields(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(4, 4)])
    summary = report["profile_summary"]
    expected_keys = {
        "num_goods",
        "num_bidders",
        "average_base_value",
        "min_base_value",
        "max_base_value",
        "average_complement_groups_per_bidder",
        "average_substitute_groups_per_bidder",
        "average_budget_low",
        "average_budget_high",
        "num_bidders_with_saturation",
        "num_bidders_with_budget_cap",
    }
    assert expected_keys <= set(summary)
    assert summary["num_goods"] == 10
    assert summary["num_bidders"] == 10


def test_validation_report_economic_diagnostics_fields(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(4, 4), (6, 6)])
    diag = report["economic_diagnostics"]
    assert set(diag) == {"4x4", "6x6"}
    for size_report in diag.values():
        assert size_report.get("skipped") is not True
        expected_keys = {
            "full_valuation_table_size_per_bidder",
            "global_optimum_welfare",
            "num_winners",
            "winner_bundle_sizes",
            "largest_winner_welfare_share",
            "is_degenerate_single_winner",
            "welfare_hhi",
            "num_items_allocated",
            "num_items_unallocated",
            "share_items_allocated",
            "mean_winner_bundle_size",
            "max_winner_bundle_size",
            "free_disposal_repair_count_total",
            "free_disposal_repair_share_total",
            "free_disposal_repair_mean_magnitude",
            "free_disposal_repair_max_magnitude",
            "free_disposal_repair_count_per_bidder",
            "free_disposal_repair_share_per_bidder",
            "mean_top_valued_bundle_size",
            "num_bidders_with_grand_bundle_as_unique_top",
            "share_bidders_with_grand_bundle_as_unique_top",
            "complement_groups_available",
            "substitute_groups_available",
            "num_bidders_with_saturation_binding",
            "num_bidders_with_budget_cap_binding",
            "saturation_budget_binding_per_bidder",
            "mean_complement_contribution_share",
            "max_complement_contribution_share",
            "num_bidders_with_complement_groups",
            "mean_substitute_backup_factor",
            "min_substitute_backup_factor",
            "max_substitute_backup_factor",
            "num_strong_substitute_groups",
            "num_weak_substitute_groups",
        }
        assert expected_keys <= set(size_report)
        assert size_report["global_optimum_welfare"] > 0
        assert "mechanism_benchmark" not in size_report


def test_validation_report_skips_size_larger_than_spec(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(4, 4), (50, 50)])
    assert report["economic_diagnostics"]["50x50"]["skipped"] is True


def test_validation_report_is_json_serializable(v0_spec_path, tmp_path):
    report = build_validation_report(v0_spec_path, sizes=[(4, 4)])
    path = tmp_path / "report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    with open(path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded == report


def test_validation_report_invalid_spec_short_circuits(tmp_path):
    bad = {"schema_version": "wrong", "domain": "pc_build", "goods": [], "bidder_profiles": []}
    path = tmp_path / "bad_spec.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    report = build_validation_report(path, sizes=[(4, 4)])
    assert report["schema_validation"]["valid"] is False
    assert "profile_summary" not in report
    assert "economic_diagnostics" not in report
    assert report["acceptance"]["passed"] is False
    assert report["acceptance"]["checks"] == [
        {"size": None, "name": "schema_valid", "passed": False, "actual": False, "threshold": True}
    ]


def test_committed_validation_report_matches_current_spec_if_present():
    report_path = Path("scenarios/pc_build_v1/validation_report_v0_manual.json")
    if not report_path.exists() or not V0_SPEC_PATH.exists():
        pytest.skip("committed v0 spec/report not present in this checkout")
    with open(report_path, "r", encoding="utf-8") as f:
        committed = json.load(f)
    assert committed["schema_validation"]["valid"] is True
    assert committed["acceptance"]["passed"] is True


# ---------------------------------------------------------------------------
# Repair share / magnitude
# ---------------------------------------------------------------------------

def test_repair_share_and_magnitude_present_and_consistent(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(6, 6)])
    diag = report["economic_diagnostics"]["6x6"]
    assert 0.0 <= diag["free_disposal_repair_share_total"] <= 1.0
    assert diag["free_disposal_repair_share_total"] == pytest.approx(
        diag["free_disposal_repair_count_total"] / (diag["full_valuation_table_size_per_bidder"] * 6)
    )
    for bidder_id, share in diag["free_disposal_repair_share_per_bidder"].items():
        assert 0.0 <= share <= 1.0
    if diag["free_disposal_repair_count_total"] > 0:
        assert diag["free_disposal_repair_mean_magnitude"] > 0.0
        assert diag["free_disposal_repair_max_magnitude"] >= diag["free_disposal_repair_mean_magnitude"]


# ---------------------------------------------------------------------------
# Saturation / budget-cap binding
# ---------------------------------------------------------------------------

def test_saturation_binding_detected_for_saturated_bidder(v0_spec_path):
    """budget_gamer has saturation_start set in the v0 manual spec and should
    show up as saturation-binding once enough goods are selected."""
    report = build_validation_report(v0_spec_path, sizes=[(8, 8)])
    diag = report["economic_diagnostics"]["8x8"]
    per_bidder = diag["saturation_budget_binding_per_bidder"]
    assert "budget_gamer" in per_bidder
    assert per_bidder["budget_gamer"]["has_saturation"] is True
    assert diag["num_bidders_with_saturation_binding"] >= 1


def test_no_budget_cap_binding_when_no_bidder_has_a_cap(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(8, 8)])
    diag = report["economic_diagnostics"]["8x8"]
    assert diag["num_bidders_with_budget_cap_binding"] == 0
    for stats in diag["saturation_budget_binding_per_bidder"].values():
        assert stats["has_budget_cap"] is False
        assert stats["budget_cap_binding"] is False


# ---------------------------------------------------------------------------
# Complement contribution
# ---------------------------------------------------------------------------

def test_complement_contribution_share_in_unit_interval(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(6, 6)])
    diag = report["economic_diagnostics"]["6x6"]
    assert 0.0 <= diag["mean_complement_contribution_share"] <= 1.0
    assert 0.0 <= diag["max_complement_contribution_share"] <= 1.0
    assert diag["max_complement_contribution_share"] >= diag["mean_complement_contribution_share"]
    assert diag["num_bidders_with_complement_groups"] >= 1


# ---------------------------------------------------------------------------
# Substitute strength
# ---------------------------------------------------------------------------

def test_substitute_strength_fields(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(6, 6)])
    diag = report["economic_diagnostics"]["6x6"]
    assert diag["substitute_groups_available"] > 0
    assert 0.0 <= diag["min_substitute_backup_factor"] <= diag["max_substitute_backup_factor"] <= 1.0
    assert (
        diag["min_substitute_backup_factor"]
        <= diag["mean_substitute_backup_factor"]
        <= diag["max_substitute_backup_factor"]
    )
    assert (
        diag["num_strong_substitute_groups"] + diag["num_weak_substitute_groups"]
        <= diag["substitute_groups_available"]
    )


def test_reseller_backup_factors_are_weak_at_8x8(v0_spec_path):
    """pc_reseller's substitute groups all use backup_factor >= 0.80, which
    should be classified as 'weak' (near-independent) substitution."""
    report = build_validation_report(v0_spec_path, sizes=[(8, 8)])
    diag = report["economic_diagnostics"]["8x8"]
    assert diag["num_weak_substitute_groups"] >= 1


# ---------------------------------------------------------------------------
# Allocation concentration
# ---------------------------------------------------------------------------

def test_allocation_concentration_fields(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(4, 4), (8, 8)])
    single_winner = report["economic_diagnostics"]["4x4"]
    assert single_winner["welfare_hhi"] == pytest.approx(1.0)
    assert single_winner["num_items_allocated"] == 4
    assert single_winner["share_items_allocated"] == pytest.approx(1.0)

    multi_winner = report["economic_diagnostics"]["8x8"]
    assert 0.0 < multi_winner["welfare_hhi"] < 1.0
    assert multi_winner["num_items_allocated"] + multi_winner["num_items_unallocated"] == 8
    assert multi_winner["mean_winner_bundle_size"] <= multi_winner["max_winner_bundle_size"]


# ---------------------------------------------------------------------------
# Mechanism benchmark (deterministic, no LLM calls)
# ---------------------------------------------------------------------------

def test_mechanism_benchmark_present_when_requested(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(4, 4)], mechanism_benchmark=True)
    mb = report["economic_diagnostics"]["4x4"]["mechanism_benchmark"]
    assert set(mb) == {"sealed_vcg", "clock"}
    assert set(mb["clock"]) == {"k1", "k2", "k3"}
    assert mb["sealed_vcg"]["welfare"] == report["economic_diagnostics"]["4x4"]["global_optimum_welfare"]
    assert mb["sealed_vcg"]["efficiency"] == pytest.approx(1.0)
    for k_result in mb["clock"].values():
        assert k_result["efficiency"] == pytest.approx(1.0)
        assert k_result["converged"] is True
        assert k_result["num_rounds"] > 0


def test_mechanism_benchmark_is_deterministic(v0_spec_path):
    from auctionlab.instances.structured_spec import make_pc_build_scenario_from_spec

    scenario = make_pc_build_scenario_from_spec(v0_spec_path, num_goods=4, num_bidders=4)
    mb1 = compute_mechanism_benchmark_diagnostics(scenario.instance, 1.0, clock_max_rounds=40)
    mb2 = compute_mechanism_benchmark_diagnostics(scenario.instance, 1.0, clock_max_rounds=40)
    assert mb1 == mb2


def test_mechanism_benchmark_no_llm_import_at_module_level():
    """The validate_scenario_spec module must not import any LLM client code."""
    import scripts.validate_scenario_spec as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "auctionlab.llm" not in source


# ---------------------------------------------------------------------------
# Acceptance criteria / configurable thresholds
# ---------------------------------------------------------------------------

def test_default_acceptance_thresholds_shape():
    thresholds = default_acceptance_thresholds()
    assert thresholds["min_bidders_for_concentration_checks"] == 6
    assert 0.0 < thresholds["max_largest_winner_welfare_share"] <= 1.0


def test_evaluate_acceptance_passes_on_lenient_thresholds(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(4, 4), (6, 6), (8, 8), (10, 10)])
    assert report["acceptance"]["passed"] is True
    assert all(c["passed"] for c in report["acceptance"]["checks"])


def test_evaluate_acceptance_fails_with_strict_thresholds(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(6, 6)])
    strict = default_acceptance_thresholds()
    strict["max_largest_winner_welfare_share"] = 0.01
    result = evaluate_acceptance(report, strict)
    assert result["passed"] is False
    failing = [c for c in result["checks"] if not c["passed"]]
    assert any(c["name"] == "largest_winner_welfare_share" for c in failing)


def test_evaluate_acceptance_skips_concentration_checks_for_small_sizes(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(4, 4)])
    strict = default_acceptance_thresholds()
    strict["max_largest_winner_welfare_share"] = 0.01
    result = evaluate_acceptance(report, strict)
    # 4x4 has only 4 bidders < min_bidders_for_concentration_checks (6), so
    # the largest-winner check should not even run for it.
    assert not any(c["name"] == "largest_winner_welfare_share" for c in result["checks"])
    assert result["passed"] is True


def test_load_thresholds_merges_overrides(tmp_path):
    override_path = tmp_path / "thresholds.json"
    override_path.write_text(json.dumps({"max_largest_winner_welfare_share": 0.5}), encoding="utf-8")
    thresholds = load_thresholds(override_path)
    assert thresholds["max_largest_winner_welfare_share"] == 0.5
    # unspecified keys keep their defaults
    assert thresholds["min_bidders_for_concentration_checks"] == 6


def test_load_thresholds_defaults_when_no_path():
    assert load_thresholds(None) == default_acceptance_thresholds()


def test_cli_thresholds_override_flows_into_report(v0_spec_path, tmp_path):
    override_path = tmp_path / "thresholds.json"
    override_path.write_text(json.dumps({"max_largest_winner_welfare_share": 0.01}), encoding="utf-8")
    thresholds = load_thresholds(override_path)
    report = build_validation_report(v0_spec_path, sizes=[(6, 6)], thresholds=thresholds)
    assert report["acceptance"]["passed"] is False


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------

def test_build_validation_report_records_selection_policy_and_seed(v0_spec_path):
    report = build_validation_report(
        v0_spec_path, sizes=[(4, 4)], seed=7, selection_policy="seeded_sample"
    )
    assert report["selection_policy"] == "seeded_sample"
    assert report["scenario_seed"] == 7


def test_build_validation_report_stratified_selection_runs(v0_spec_path):
    report = build_validation_report(
        v0_spec_path, sizes=[(6, 6)], seed=0, selection_policy="stratified"
    )
    assert report["schema_validation"]["valid"] is True
    assert report["economic_diagnostics"]["6x6"]["global_optimum_welfare"] > 0


def test_build_validation_report_unknown_selection_policy_raises(v0_spec_path):
    with pytest.raises(ValueError):
        build_validation_report(v0_spec_path, sizes=[(4, 4)], selection_policy="bogus")


def test_cli_selection_policy_and_seed_flow_through(v0_spec_path, tmp_path, monkeypatch):
    output_path = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_scenario_spec.py",
            str(v0_spec_path),
            "--output",
            str(output_path),
            "--sizes",
            "6x6",
            "--selection-policy",
            "stratified",
            "--seed",
            "3",
        ],
    )
    validate_main()
    with open(output_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["selection_policy"] == "stratified"
    assert report["scenario_seed"] == 3


# ---------------------------------------------------------------------------
# Multi-seed validation
# ---------------------------------------------------------------------------

def test_aggregate_metrics_across_seeds_computes_mean_min_max():
    fake_per_seed = {
        "0": {"economic_diagnostics": {"4x4": {"global_optimum_welfare": 100.0, "num_winners": 1}}},
        "1": {"economic_diagnostics": {"4x4": {"global_optimum_welfare": 300.0, "num_winners": 3}}},
        "2": {"economic_diagnostics": {"4x4": {"global_optimum_welfare": 200.0, "num_winners": 2}}},
    }
    aggregate = aggregate_metrics_across_seeds(fake_per_seed)
    welfare_stats = aggregate["4x4"]["global_optimum_welfare"]
    assert welfare_stats["mean"] == pytest.approx(200.0)
    assert welfare_stats["min"] == 100.0
    assert welfare_stats["max"] == 300.0
    assert welfare_stats["values_by_seed"] == {"0": 100.0, "1": 300.0, "2": 200.0}


def test_aggregate_metrics_across_seeds_handles_all_skipped_size():
    fake_per_seed = {
        "0": {"economic_diagnostics": {"50x50": {"skipped": True, "reason": "too big"}}},
        "1": {"economic_diagnostics": {"50x50": {"skipped": True, "reason": "too big"}}},
    }
    aggregate = aggregate_metrics_across_seeds(fake_per_seed)
    assert aggregate["50x50"] == {"skipped": True}


def test_aggregate_metrics_across_seeds_handles_partially_skipped_size():
    fake_per_seed = {
        "0": {"economic_diagnostics": {"10x10": {"skipped": True}}},
        "1": {"economic_diagnostics": {"10x10": {"global_optimum_welfare": 500.0}}},
    }
    aggregate = aggregate_metrics_across_seeds(fake_per_seed)
    assert aggregate["10x10"]["global_optimum_welfare"]["values_by_seed"] == {"1": 500.0}


def test_build_multi_seed_report_shape(v0_spec_path):
    report = build_multi_seed_report(v0_spec_path, sizes=[(4, 4), (6, 6)], seeds=[0, 1, 2])
    assert set(report) == {
        "spec_path", "selection_policy", "seeds", "all_seeds_schema_valid",
        "all_seeds_acceptance_passed", "per_seed", "aggregate",
    }
    assert report["seeds"] == [0, 1, 2]
    assert set(report["per_seed"]) == {"0", "1", "2"}
    for seed_report in report["per_seed"].values():
        assert seed_report["schema_validation"]["valid"] is True
    assert set(report["aggregate"]) == {"4x4", "6x6"}


def test_build_multi_seed_report_prefix_is_seed_invariant(v0_spec_path):
    """Under prefix selection, seed doesn't affect what's chosen, so every
    seed's diagnostics should be identical -- min == mean == max."""
    report = build_multi_seed_report(
        v0_spec_path, sizes=[(6, 6)], seeds=[0, 1, 2], selection_policy="prefix"
    )
    welfare_stats = report["aggregate"]["6x6"]["global_optimum_welfare"]
    assert welfare_stats["min"] == welfare_stats["max"] == pytest.approx(welfare_stats["mean"])


def test_build_multi_seed_report_stratified_varies_across_seeds():
    """Stratified selection on a real generated spec should show *some*
    variance in at least one metric across seeds (not seed-invariant, unlike
    prefix)."""
    v2_path = Path("scenarios/pc_build_v1/pc_build_profiles_v2_gemini_trial.json")
    if not v2_path.exists():
        pytest.skip("v2 gemini trial spec not present in this checkout")
    report = build_multi_seed_report(
        v2_path, sizes=[(6, 6)], seeds=[0, 1, 2, 3], selection_policy="stratified"
    )
    welfare_stats = report["aggregate"]["6x6"]["global_optimum_welfare"]
    assert welfare_stats["min"] < welfare_stats["max"]


def test_build_multi_seed_report_empty_seeds_rejected(v0_spec_path):
    with pytest.raises(ValueError):
        build_multi_seed_report(v0_spec_path, sizes=[(4, 4)], seeds=[])


def test_build_multi_seed_report_json_serializable(v0_spec_path, tmp_path):
    report = build_multi_seed_report(v0_spec_path, sizes=[(4, 4)], seeds=[0, 1])
    path = tmp_path / "multiseed.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    with open(path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded == report


def test_cli_seeds_flag_writes_multi_seed_report(v0_spec_path, tmp_path, monkeypatch):
    output_path = tmp_path / "multiseed.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_scenario_spec.py",
            str(v0_spec_path),
            "--output",
            str(output_path),
            "--sizes",
            "4x4",
            "--seeds",
            "0",
            "1",
        ],
    )
    validate_main()
    with open(output_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert set(report["per_seed"]) == {"0", "1"}
    assert "all_seeds_schema_valid" in report
