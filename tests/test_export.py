from __future__ import annotations

import csv

from auctionlab.auctions.clock import ClockConfig
from auctionlab.experiments.export import (
    allocation_to_str,
    batch_results_to_comparison_rows,
    batch_results_to_mechanism_rows,
    write_csv,
    write_csv_variable_rows,
)
from auctionlab.experiments.runner import run_batch_experiments
from auctionlab.instances.random import make_random_xor_instance


def test_allocation_to_str_is_stable():
    allocation = {
        "i2": frozenset({"B"}),
        "i1": frozenset(),
        "i3": frozenset({"C", "A"}),
    }

    assert allocation_to_str(allocation) == "i1:[];i2:[B];i3:[A,C]"


def test_batch_results_export_rows_have_expected_shape():
    instances = []

    for seed in range(2):
        instance = make_random_xor_instance(
            n_items=4,
            n_bidders=3,
            atoms_per_bidder=4,
            max_bundle_size=2,
            min_value=1.0,
            max_value=10.0,
            seed=seed,
        )
        instances.append((f"seed_{seed}", instance))

    batch_results = run_batch_experiments(
        instances,
        clock_cfg=ClockConfig(max_rounds=30, price_step=1.0, reserve=0.0),
    )

    mechanism_rows = batch_results_to_mechanism_rows(batch_results)
    comparison_rows = batch_results_to_comparison_rows(batch_results)

    assert len(mechanism_rows) == 4
    assert len(comparison_rows) == 2

    assert set(mechanism_rows[0].keys()) == {
        "instance_name",
        "mechanism",
        "welfare",
        "revenue",
        "rounds",
        "query_count",
        "allocation",
    }

    assert set(comparison_rows[0].keys()) == {
        "instance_name",
        "clock_mechanism",
        "sealed_welfare",
        "clock_welfare",
        "efficiency",
        "sealed_revenue",
        "clock_revenue",
        "sealed_query_count",
        "clock_query_count",
        "clock_rounds",
        "allocation_match",
        "welfare_match",
        "sealed_allocation",
        "clock_allocation",
    }


def test_write_csv_writes_file(tmp_path):
    rows = [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
    ]

    output_path = tmp_path / "test.csv"

    write_csv(rows, output_path)

    with output_path.open(newline="") as f:
        reader = csv.DictReader(f)
        loaded = list(reader)

    assert loaded == [
        {"a": "1", "b": "x"},
        {"a": "2", "b": "y"},
    ]


def test_write_csv_variable_rows_unions_fieldnames(tmp_path):
    rows = [
        {"scenario": "s1", "arm": "shared initial", "tok_in": 10},
        {"scenario": "s1", "arm": "sealed", "efficiency": 0.9, "revenue": 5},
    ]
    output_path = tmp_path / "summary.csv"

    write_csv_variable_rows(rows, output_path)

    with output_path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["scenario", "arm", "tok_in", "efficiency", "revenue"]
        loaded = list(reader)

    assert loaded == [
        {"scenario": "s1", "arm": "shared initial", "tok_in": "10", "efficiency": "", "revenue": ""},
        {"scenario": "s1", "arm": "sealed", "tok_in": "", "efficiency": "0.9", "revenue": "5"},
    ]


def test_write_csv_variable_rows_empty_writes_empty_file(tmp_path):
    output_path = tmp_path / "empty.csv"
    write_csv_variable_rows([], output_path)
    assert output_path.read_text() == ""


def test_comparison_rows_include_diagnostics_when_instances_provided():
    instances = []

    for seed in range(2):
        instance = make_random_xor_instance(
            n_items=4,
            n_bidders=3,
            atoms_per_bidder=4,
            max_bundle_size=2,
            min_value=1.0,
            max_value=10.0,
            seed=seed,
        )
        instances.append((f"seed_{seed}", instance))

    instances_by_name = dict(instances)

    batch_results = run_batch_experiments(
        instances,
        clock_cfg=ClockConfig(max_rounds=30, price_step=1.0, reserve=0.0),
    )

    comparison_rows = batch_results_to_comparison_rows(
        batch_results,
        instances_by_name=instances_by_name,
    )

    assert len(comparison_rows) == 2

    assert "full_atoms_count" in comparison_rows[0]
    assert "supplementary_atoms_count" in comparison_rows[0]
    assert "supplementary_coverage_ratio" in comparison_rows[0]
    assert "sealed_winning_bundles_observed" in comparison_rows[0]
    assert "missing_sealed_winning_bundles" in comparison_rows[0]


def test_comparison_rows_select_requested_clock_mechanism():
    instances = [
        (
            "seed_0",
            make_random_xor_instance(
                n_items=4,
                n_bidders=3,
                atoms_per_bidder=4,
                max_bundle_size=2,
                min_value=1.0,
                max_value=10.0,
                seed=0,
            ),
        )
    ]

    batch_results = run_batch_experiments(
        instances,
        clock_cfg=ClockConfig(max_rounds=30, price_step=1.0, reserve=0.0),
        clock_top_k_values=[1, 2],
    )

    comparison_rows = batch_results_to_comparison_rows(
        batch_results,
        clock_mechanism="clock_supplementary_vcg_top_2",
    )

    assert comparison_rows[0]["clock_mechanism"] == (
        "clock_supplementary_vcg_top_2"
    )
