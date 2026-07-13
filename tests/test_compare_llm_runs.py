from __future__ import annotations

import csv
from pathlib import Path

import pytest

from auctionlab.experiments.llm_analysis import (
    COMPARISON_FIELDS,
    compare_llm_runs,
    join_run_rows,
)
from auctionlab.experiments.llm_analysis_io import (
    load_csv_by_key,
    parse_labeled_path,
)


KEY = (
    "creative_studio_4x4",
    "implicit",
    "sealed_llm_proxy_vcg",
    "",
)


def summary_row(
    *,
    mechanism: str = "sealed_llm_proxy_vcg",
    top_k: str = "",
) -> dict[str, str]:
    return {
        "scenario": "creative_studio_4x4",
        "seed_type": "implicit",
        "mechanism": mechanism,
        "top_k": top_k,
        "efficiency": "0.8",
        "true_welfare": "80.0",
        "reported_welfare": "100.0",
        "full_info_welfare": "100.0",
        "query_count": "20",
        "rounds": "" if not top_k else "5",
        "allocation_match": "False",
        "welfare_match": "False",
        "inferred_bids": "",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_parse_labeled_path_with_label():
    label, path = parse_labeled_path("anchored=outputs/run")

    assert label == "anchored"
    assert path == Path("outputs/run")


def test_parse_labeled_path_infers_directory_name():
    label, path = parse_labeled_path("outputs/llm_runs/anchored")

    assert label == "anchored"
    assert path == Path("outputs/llm_runs/anchored")


def test_join_run_rows_adds_diagnostics_and_derived_fields():
    value_row = {
        "mae": "12.0",
        "rmse": "15.0",
        "mean_signed_error": "-2.0",
        "overreport_rate": "0.4",
        "underreport_rate": "0.6",
        "max_absolute_error": "30.0",
        "mean_absolute_relative_error": "0.2",
    }
    allocation_row = {
        "changed_bidder_count": "3",
        "changed_bidder_rate": "0.75",
        "welfare_loss": "20.0",
        "positive_delta_count": "2",
        "negative_delta_count": "1",
        "zero_delta_count": "1",
    }

    rows = join_run_rows(
        "anchored",
        {KEY: summary_row()},
        {KEY: value_row},
        {KEY: allocation_row},
    )

    assert list(rows[0]) == COMPARISON_FIELDS
    assert rows[0]["reported_welfare_gap"] == "-20.0"
    assert rows[0]["reported_welfare_ratio"] == "1.25"
    assert rows[0]["value_mae"] == "12.0"
    assert rows[0]["value_rmse"] == "15.0"
    assert rows[0]["changed_bidder_count"] == "3"
    assert rows[0]["welfare_loss"] == "20.0"


def test_join_run_rows_leaves_missing_diagnostics_blank():
    row = join_run_rows("baseline", {KEY: summary_row()})[0]

    assert row["value_mae"] == ""
    assert row["value_rmse"] == ""
    assert row["changed_bidder_count"] == ""
    assert row["welfare_loss"] == ""


def test_join_run_rows_uses_unique_seed_type_fallback():
    summary = summary_row()
    summary["seed_type"] = "explicit"
    summary_key = (
        "creative_studio_4x4",
        "explicit",
        "sealed_llm_proxy_vcg",
        "",
    )
    diagnostic = {"mae": "12.0"}

    row = join_run_rows(
        "anchored",
        {summary_key: summary},
        {KEY: diagnostic},
    )[0]

    assert row["seed_type"] == "explicit"
    assert row["value_mae"] == "12.0"


def test_compare_llm_runs_sorts_mechanism_top_k_and_label(tmp_path):
    run_b = tmp_path / "run_b"
    run_a = tmp_path / "run_a"
    rows = [
        summary_row(mechanism="clock_llm_proxy_vcg", top_k="2"),
        summary_row(mechanism="sealed_llm_proxy_vcg", top_k=""),
        summary_row(mechanism="clock_llm_proxy_vcg", top_k="1"),
    ]
    write_csv(run_b / "curated_summary.csv", rows)
    write_csv(run_a / "curated_summary.csv", rows)

    combined = compare_llm_runs(
        [("b", run_b), ("a", run_a)]
    )

    assert [
        (row["mechanism"], row["top_k"], row["run_label"])
        for row in combined
    ] == [
        ("sealed_llm_proxy_vcg", "", "a"),
        ("sealed_llm_proxy_vcg", "", "b"),
        ("clock_llm_proxy_vcg", "1", "a"),
        ("clock_llm_proxy_vcg", "1", "b"),
        ("clock_llm_proxy_vcg", "2", "a"),
        ("clock_llm_proxy_vcg", "2", "b"),
    ]


def test_load_csv_by_key_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "summary.csv"
    row = summary_row()
    write_csv(path, [row, row])

    with pytest.raises(ValueError, match="Duplicate join key"):
        load_csv_by_key(path)
