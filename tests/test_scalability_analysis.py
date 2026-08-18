"""Offline aggregation of scalability results.

Covers person-side metrics computed from run summaries, reporting of a
missing summary rather than silent omission, inclusion of frozen opening-answer
tokens in the person-side totals, and addition of the shared anchor to each
scaling series.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from auctionlab.experiments.scalability_analysis import (
    _aggregate_metric,
    load_scalability_results,
    write_scalability_tables,
)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _completed_case(root: Path, name: str) -> Path:
    case = root / "seed_0" / name
    _write_csv(
        case / "curated_run_summary.csv",
        [
            "scenario",
            "arm",
            "efficiency",
            "true_welfare",
            "full_info_welfare",
            "vq",
            "dq",
            "nl",
            "tok_in",
            "tok_out",
            "revenue",
            "surplus",
        ],
        [
            {
                "scenario": "fixture",
                "arm": "shared initial (nl+im+pv)",
                "tok_in": 1000,
                "tok_out": 200,
            },
            {
                "scenario": "fixture",
                "arm": "proxy sealed",
                "efficiency": 0.9,
                "true_welfare": 900,
                "full_info_welfare": 1000,
                "vq": 7,
                "dq": 2,
                "nl": 1,
                "tok_in": 300,
                "tok_out": 40,
                "revenue": 450,
                "surplus": 450,
            },
        ],
    )
    _write_csv(
        case / "curated_sealed_proxy_elicited.csv",
        ["full_info_revenue", "proxy_revenue"],
        [{"full_info_revenue": 500, "proxy_revenue": 450}],
    )
    _write_csv(
        case / "curated_pv_candidate_bundle_stats.csv",
        [
            "candidate_bundles_generated",
            "candidate_bundles_sent_to_pv",
            "candidate_bundles_truncated",
        ],
        [
            {
                "candidate_bundles_generated": 12,
                "candidate_bundles_sent_to_pv": 10,
                "candidate_bundles_truncated": 2,
            },
            {
                "candidate_bundles_generated": 8,
                "candidate_bundles_sent_to_pv": 8,
                "candidate_bundles_truncated": 0,
            },
        ],
    )
    return case


def test_load_scalability_results_computes_person_side_metrics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sealed_r3"
    _completed_case(root, "goods_4x8")

    cases, incomplete = load_scalability_results(root)

    assert incomplete == []
    values = cases[0].values
    assert values["seed"] == "0"
    assert values["series"] == "goods"
    assert values["x_value"] == 4
    assert values["person_queries"] == 10
    assert values["person_tokens_total"] == 340
    assert values["shared_proxy_tokens_in"] == 1000
    assert values["efficiency_pct"] == pytest.approx(90)
    assert values["welfare_loss"] == pytest.approx(100)
    assert values["revenue_difference_pct"] == pytest.approx(-10)
    assert values["candidate_bundles_generated"] == 20
    assert values["candidate_bundles_sent_to_pv"] == 18


def test_load_scalability_results_reports_missing_summary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sealed_r3"
    failed = root / "seed_0" / "bidders_8x10"
    failed.mkdir(parents=True)
    (failed / "calls.jsonl").write_text("{}\n", encoding="utf-8")

    cases, incomplete = load_scalability_results(root)

    assert cases == []
    assert incomplete[0]["case"] == "bidders_8x10"
    assert incomplete[0]["reason"] == "missing curated_run_summary.csv"


def test_person_tokens_include_frozen_opening_answer(tmp_path: Path) -> None:
    root = tmp_path / "sealed_r3"
    case = _completed_case(root, "goods_4x8")
    rows = list(csv.DictReader(
        (case / "curated_run_summary.csv").open(
            newline="", encoding="utf-8"
        )
    ))
    rows[0]["person_tok_in"] = "700"
    rows[0]["person_tok_out"] = "120"
    rows[0]["proxy_tok_in"] = "800"
    rows[0]["proxy_tok_out"] = "160"
    _write_csv(
        case / "curated_run_summary.csv",
        list(rows[0].keys()),
        rows,
    )

    cases, incomplete = load_scalability_results(root)

    assert incomplete == []
    values = cases[0].values
    assert values["person_tokens_in"] == 1000
    assert values["person_tokens_out"] == 160
    assert values["person_tokens_total"] == 1160
    assert values["shared_proxy_tokens_in"] == 800
    assert values["shared_proxy_tokens_out"] == 160


def test_anchor_is_added_to_each_scaling_series(tmp_path: Path) -> None:
    root = tmp_path / "sealed_r3"
    _completed_case(root, "anchor_8x8")
    _completed_case(root, "goods_4x8")

    cases, _ = load_scalability_results(root)
    points = _aggregate_metric(cases, "person_queries")

    assert [point[0] for point in points["goods"]] == [4, 8]
    assert [point[0] for point in points["bidders"]] == [8]
    assert [point[0] for point in points["joint"]] == [8]


def test_write_scalability_tables_writes_both_reports(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sealed_r3"
    _completed_case(root, "joint_4x4")
    cases, incomplete = load_scalability_results(root)

    metrics_path, incomplete_path = write_scalability_tables(
        tmp_path / "analysis",
        cases,
        incomplete,
    )

    assert metrics_path.exists()
    assert incomplete_path.exists()
    assert len(_read_rows(metrics_path)) == 1
    assert _read_rows(incomplete_path) == []


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestCalibrationPassthrough:
    """Aggregated rows must carry the calibration each case ran under.

    Pooling cases run under different calibrations would silently mix
    treatments, so the calibration travels with every row rather than living
    only in each case directory.
    """

    def _case_with_calibration(self, root: Path, name: str) -> Path:
        case = _completed_case(root, name)
        path = case / "curated_run_summary.csv"
        rows = _read_rows(path)
        extra = {
            "pv_calibration_family": "exponential",
            "pv_calibration_scale": "1.83",
            "pv_calibration_size_gamma": "0.961",
            "pv_calibration_size_threshold": "3",
            "pv_calibration_budget_cap": "True",
            "pv_calibration_config_hash": "deadbeef",
        }
        for row in rows:
            row.update(extra)
        _write_csv(path, list(rows[0]), rows)
        return case

    def test_columns_reach_the_metrics_table(self, tmp_path):
        root = tmp_path / "sealed_r3"
        self._case_with_calibration(root, "joint_4x4")
        cases, _ = load_scalability_results(root)
        metrics_path, _ = write_scalability_tables(
            tmp_path / "analysis", cases, []
        )
        row = _read_rows(metrics_path)[0]
        assert row["pv_calibration_family"] == "exponential"
        assert row["pv_calibration_scale"] == "1.83"
        assert row["pv_calibration_config_hash"] == "deadbeef"

    def test_older_summaries_without_the_columns_still_load(self, tmp_path):
        root = tmp_path / "sealed_r3"
        _completed_case(root, "joint_4x4")
        cases, incomplete = load_scalability_results(root)
        assert not incomplete
        assert cases[0].values["pv_calibration_family"] == ""
