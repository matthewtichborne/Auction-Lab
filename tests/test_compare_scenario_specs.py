"""Tests for scripts/compare_scenario_specs.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auctionlab.instances.scenario_spec import (
    load_scenario_profile_spec,
    scenario_profile_spec_to_dict,
    write_scenario_profile_spec,
)
from scripts.compare_scenario_specs import compare_reports, main as compare_main
from scripts.validate_scenario_spec import build_validation_report

V0_SPEC_PATH = Path("scenarios/pc_build_v1/pc_build_profiles_v0_manual.json")


@pytest.fixture(scope="module")
def v0_spec_path(tmp_path_factory) -> Path:
    if V0_SPEC_PATH.exists():
        return V0_SPEC_PATH
    from scripts.export_current_pc_build_profiles import build_manual_spec

    spec = build_manual_spec(seed=0)
    path = tmp_path_factory.mktemp("spec") / "pc_build_profiles_v0_manual.json"
    write_scenario_profile_spec(spec, path)
    return path


@pytest.fixture
def perturbed_spec_path(v0_spec_path, tmp_path) -> Path:
    """A copy of the v0 spec with one bidder's base values scaled way up."""
    spec = load_scenario_profile_spec(v0_spec_path)
    data = scenario_profile_spec_to_dict(spec)
    first_bidder = data["bidder_profiles"][0]
    first_bidder["base_values"] = {k: v * 5 for k, v in first_bidder["base_values"].items()}
    path = tmp_path / "perturbed_spec.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def test_compare_identical_specs_has_no_significant_changes(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(4, 4), (6, 6)])
    comparison = compare_reports(report, report, tolerance=0.1)
    assert comparison["significant_changes"] == []
    for entry in comparison["profile_summary_diff"].values():
        assert entry["changed"] is False


def test_compare_detects_significant_welfare_change(v0_spec_path, perturbed_spec_path):
    report_a = build_validation_report(v0_spec_path, sizes=[(6, 6)])
    report_b = build_validation_report(perturbed_spec_path, sizes=[(6, 6)])
    comparison = compare_reports(report_a, report_b, tolerance=0.1)

    assert len(comparison["significant_changes"]) > 0
    fields_changed = {c["field"] for c in comparison["significant_changes"]}
    assert "global_optimum_welfare" in fields_changed or "max_base_value" in comparison["profile_summary_diff"]


def test_compare_diff_entries_have_delta_and_pct_change(v0_spec_path, perturbed_spec_path):
    report_a = build_validation_report(v0_spec_path, sizes=[(6, 6)])
    report_b = build_validation_report(perturbed_spec_path, sizes=[(6, 6)])
    comparison = compare_reports(report_a, report_b, tolerance=0.1)

    entry = comparison["economic_diagnostics_diff"]["6x6"]["global_optimum_welfare"]
    assert entry["b"] >= entry["a"]
    assert entry["delta"] == pytest.approx(entry["b"] - entry["a"])
    assert entry["pct_change"] == pytest.approx(entry["delta"] / entry["a"])


def test_compare_skips_nested_dict_fields(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(6, 6)])
    comparison = compare_reports(report, report, tolerance=0.1)
    diff_fields = set(comparison["economic_diagnostics_diff"]["6x6"])
    assert "winner_bundle_sizes" not in diff_fields
    assert "saturation_budget_binding_per_bidder" not in diff_fields
    assert "global_optimum_welfare" in diff_fields


def test_compare_handles_skipped_sizes(v0_spec_path):
    report = build_validation_report(v0_spec_path, sizes=[(50, 50)])
    comparison = compare_reports(report, report, tolerance=0.1)
    assert comparison["economic_diagnostics_diff"]["50x50"] == {
        "skipped_a": True,
        "skipped_b": True,
    }


def test_compare_handles_invalid_report():
    invalid_report = {
        "spec_path": "bad.json",
        "schema_validation": {"valid": False, "errors": ["boom"]},
    }
    valid_report = {
        "spec_path": "good.json",
        "schema_validation": {"valid": True, "errors": []},
        "profile_summary": {"num_goods": 4},
        "economic_diagnostics": {},
    }
    comparison = compare_reports(invalid_report, valid_report)
    assert comparison["profile_summary_diff"] == {}
    assert comparison["significant_changes"] == []


def test_compare_scenario_specs_cli_writes_output(v0_spec_path, perturbed_spec_path, tmp_path, monkeypatch):
    output_path = tmp_path / "diff.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_scenario_specs.py",
            str(v0_spec_path),
            str(perturbed_spec_path),
            "--output",
            str(output_path),
            "--sizes",
            "6x6",
        ],
    )
    compare_main()
    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "significant_changes" in data


def test_compare_scenario_specs_cli_fail_on_significant_change(
    v0_spec_path, perturbed_spec_path, tmp_path, monkeypatch
):
    output_path = tmp_path / "diff.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_scenario_specs.py",
            str(v0_spec_path),
            str(perturbed_spec_path),
            "--output",
            str(output_path),
            "--sizes",
            "6x6",
            "--fail-on-significant-change",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        compare_main()
    assert exc_info.value.code == 1


def test_compare_scenario_specs_cli_exits_zero_for_identical_specs(v0_spec_path, tmp_path, monkeypatch):
    output_path = tmp_path / "diff.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_scenario_specs.py",
            str(v0_spec_path),
            str(v0_spec_path),
            "--output",
            str(output_path),
            "--sizes",
            "6x6",
            "--fail-on-significant-change",
        ],
    )
    compare_main()  # should not raise SystemExit
    assert output_path.exists()
