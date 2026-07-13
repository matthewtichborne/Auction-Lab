from __future__ import annotations

import csv

from auctionlab.experiments.export import batch_summaries_to_rows, write_csv
from auctionlab.experiments.runner import BatchSummary


EXPECTED_KEYS = {
    "top_k",
    "n_instances",
    "sealed_avg_welfare",
    "clock_avg_welfare",
    "clock_avg_efficiency",
    "sealed_avg_revenue",
    "clock_avg_revenue",
    "clock_avg_rounds",
    "sealed_avg_query_count",
    "clock_avg_query_count",
    "allocation_match_rate",
    "welfare_match_rate",
}


def make_summary(multiplier: float) -> BatchSummary:
    return BatchSummary(
        n_instances=2,
        sealed_avg_welfare=10.0,
        clock_avg_welfare=9.0 * multiplier,
        clock_avg_efficiency=0.9 * multiplier,
        sealed_avg_revenue=5.0,
        clock_avg_revenue=4.0 * multiplier,
        clock_avg_rounds=3.0,
        sealed_avg_query_count=2.0,
        clock_avg_query_count=6.0,
        allocation_match_rate=0.5,
        welfare_match_rate=0.5,
    )


def test_batch_summaries_to_rows_sorts_top_k_and_has_expected_keys():
    rows = batch_summaries_to_rows(
        {
            3: make_summary(1.0),
            1: make_summary(0.8),
            2: make_summary(0.9),
        }
    )

    assert [row["top_k"] for row in rows] == [1, 2, 3]
    assert set(rows[0]) == EXPECTED_KEYS


def test_write_csv_writes_summary_rows(tmp_path):
    rows = batch_summaries_to_rows({1: make_summary(1.0)})
    output_path = tmp_path / "summary.csv"

    write_csv(rows, output_path)

    with output_path.open(newline="") as f:
        loaded = list(csv.DictReader(f))

    assert loaded[0]["top_k"] == "1"
    assert set(loaded[0]) == EXPECTED_KEYS
