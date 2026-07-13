from __future__ import annotations

from pathlib import Path

import pytest

from auctionlab.experiments.llm_analysis import (
    AGGREGATE_FIELDS,
    SUMMARY_FIELDS,
    build_aggregate_rows,
    build_summary_rows,
    clock_row_to_summary,
    sealed_row_to_summary,
)
from auctionlab.experiments.llm_analysis_io import (
    extract_top_k_from_filename,
)


def make_sealed_row(instance_name: str) -> dict[str, str]:
    return {
        "instance_name": instance_name,
        "efficiency": "0.9",
        "llm_proxy_true_welfare": "90.0",
        "llm_proxy_reported_welfare": "95.0",
        "full_info_welfare": "100.0",
        "llm_proxy_query_count": "4",
        "allocation_match": "False",
        "welfare_match": "False",
        "llm_proxy_reported_bids": "i1={[A]:10.0}",
    }


def make_clock_row(instance_name: str) -> dict[str, str]:
    return {
        "instance_name": instance_name,
        "efficiency": "1.0",
        "clock_llm_true_welfare": "100.0",
        "clock_llm_reported_welfare": "105.0",
        "full_info_welfare": "100.0",
        "clock_llm_query_count": "8",
        "clock_rounds": "3",
        "allocation_match": "True",
        "welfare_match": "True",
        "clock_llm_reported_bids": "i1={[A]:10.0}",
    }


def test_extract_top_k_from_filename():
    path = Path("curated_clock_llm_comparison_top_12.csv")

    assert extract_top_k_from_filename(path) == 12


def test_extract_top_k_rejects_unexpected_filename():
    with pytest.raises(ValueError, match="Cannot extract top_k"):
        extract_top_k_from_filename(Path("clock_top_2.csv"))


def test_sealed_row_to_summary_infers_seed_type():
    summary = sealed_row_to_summary(
        make_sealed_row("substitutes_implicit")
    )

    assert list(summary) == SUMMARY_FIELDS
    assert summary["scenario"] == "substitutes_implicit"
    assert summary["seed_type"] == "implicit"
    assert summary["mechanism"] == "sealed_llm_proxy_vcg"
    assert summary["top_k"] == ""
    assert summary["true_welfare"] == "90.0"
    assert summary["reported_welfare"] == "95.0"
    assert summary["query_count"] == "4"
    assert summary["rounds"] == ""


def test_clock_row_to_summary_maps_clock_fields():
    summary = clock_row_to_summary(
        make_clock_row("substitutes"),
        top_k=2,
    )

    assert list(summary) == SUMMARY_FIELDS
    assert summary["seed_type"] == "explicit"
    assert summary["mechanism"] == "clock_llm_proxy_vcg"
    assert summary["top_k"] == "2"
    assert summary["true_welfare"] == "100.0"
    assert summary["reported_welfare"] == "105.0"
    assert summary["query_count"] == "8"
    assert summary["rounds"] == "3"


def test_build_summary_rows_orders_scenario_mechanism_and_top_k():
    rows = build_summary_rows(
        [
            make_sealed_row("zeta"),
            make_sealed_row("alpha_implicit"),
        ],
        {
            2: [
                make_clock_row("zeta"),
                make_clock_row("alpha_implicit"),
            ],
            1: [
                make_clock_row("zeta"),
                make_clock_row("alpha_implicit"),
            ],
        },
    )

    assert [
        (row["scenario"], row["mechanism"], row["top_k"])
        for row in rows
    ] == [
        ("alpha_implicit", "sealed_llm_proxy_vcg", ""),
        ("alpha_implicit", "clock_llm_proxy_vcg", "1"),
        ("alpha_implicit", "clock_llm_proxy_vcg", "2"),
        ("zeta", "sealed_llm_proxy_vcg", ""),
        ("zeta", "clock_llm_proxy_vcg", "1"),
        ("zeta", "clock_llm_proxy_vcg", "2"),
    ]


def test_build_aggregate_rows_groups_and_computes_averages():
    rows = [
        {
            **sealed_row_to_summary(make_sealed_row("alpha_implicit")),
            "efficiency": "1.0",
            "true_welfare": "100.0",
            "reported_welfare": "110.0",
            "full_info_welfare": "100.0",
            "query_count": "4",
            "allocation_match": "True",
            "welfare_match": "True",
        },
        {
            **sealed_row_to_summary(make_sealed_row("beta_implicit")),
            "efficiency": "0.8",
            "true_welfare": "80.0",
            "reported_welfare": "90.0",
            "full_info_welfare": "100.0",
            "query_count": "6",
            "allocation_match": "False",
            "welfare_match": "False",
        },
        {
            **clock_row_to_summary(
                make_clock_row("alpha_implicit"),
                top_k=1,
            ),
            "rounds": "3",
        },
        {
            **clock_row_to_summary(
                make_clock_row("beta_implicit"),
                top_k=1,
            ),
            "rounds": "5",
            "allocation_match": "False",
        },
        clock_row_to_summary(
            make_clock_row("alpha_implicit"),
            top_k=2,
        ),
    ]

    aggregate = build_aggregate_rows(rows)

    assert all(list(row) == AGGREGATE_FIELDS for row in aggregate)
    sealed = aggregate[0]
    clock_top_1 = aggregate[1]
    clock_top_2 = aggregate[2]

    assert sealed["top_k"] == ""
    assert sealed["n"] == "2"
    assert sealed["avg_efficiency"] == "0.9"
    assert sealed["avg_true_welfare"] == "90.0"
    assert sealed["avg_reported_welfare"] == "100.0"
    assert sealed["avg_full_info_welfare"] == "100.0"
    assert sealed["avg_query_count"] == "5.0"
    assert sealed["avg_rounds"] == ""
    assert sealed["allocation_match_rate"] == "0.5"
    assert sealed["welfare_match_rate"] == "0.5"

    assert clock_top_1["top_k"] == "1"
    assert clock_top_1["n"] == "2"
    assert clock_top_1["avg_rounds"] == "4.0"
    assert clock_top_1["allocation_match_rate"] == "0.5"
    assert clock_top_1["welfare_match_rate"] == "1.0"

    assert clock_top_2["top_k"] == "2"
    assert clock_top_2["n"] == "1"
