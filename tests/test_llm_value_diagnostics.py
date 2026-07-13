from __future__ import annotations

import csv
import math

import pytest

from auctionlab.experiments.llm_value_diagnostics import (
    aggregate_value_error_records,
    compute_value_error_records,
    group_value_error_records,
    parse_reported_bids,
    value_error_records_to_rows,
    write_value_error_aggregate_csv,
    write_value_error_records_csv,
)
from auctionlab.instances.base import AuctionInstance


def test_parse_reported_bids():
    parsed = parse_reported_bids(
        "bidder={[A]:10.0;[A,B]:15.0}|other={[C]:7.0}"
    )

    assert parsed == {
        "bidder": {
            frozenset({"A"}): 10.0,
            frozenset({"A", "B"}): 15.0,
        },
        "other": {
            frozenset({"C"}): 7.0,
        },
    }
    assert parse_reported_bids("") == {}


@pytest.mark.parametrize(
    "text",
    [
        "bidder=[A]:10.0",
        "bidder={[A]10.0}",
        "bidder={[A,A]:10.0}",
        "bidder={[A]:10.0}|bidder={[B]:2.0}",
    ],
)
def test_parse_reported_bids_rejects_malformed_input(text):
    with pytest.raises(ValueError):
        parse_reported_bids(text)


def test_compute_value_error_records_uses_instance_values():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1"],
        valuations={
            "i1": {
                frozenset({"A"}): 10.0,
                frozenset({"A", "B"}): 20.0,
            }
        },
    )

    records = compute_value_error_records(
        scenario="toy",
        seed_type="implicit",
        mechanism="sealed_llm_proxy_vcg",
        top_k="",
        reported_bids={
            "i1": {
                frozenset({"A"}): 12.0,
                frozenset({"A", "B"}): 15.0,
            }
        },
        instance=instance,
    )

    assert [record.bundle_label for record in records] == ["[A]", "[A,B]"]
    assert records[0].true_value == 10.0
    assert records[0].signed_error == 2.0
    assert records[0].absolute_error == 2.0
    assert records[0].relative_error == 0.2
    assert records[1].true_value == 20.0
    assert records[1].signed_error == -5.0
    assert records[1].bundle_size == 2

    rows = value_error_records_to_rows(records)
    assert rows[0]["bundle"] == "A"
    assert rows[1]["bundle_label"] == "[A,B]"


def test_aggregate_value_error_records():
    instance = AuctionInstance(
        items=["A", "B", "C"],
        bidder_ids=["i1"],
        valuations={
            "i1": {
                frozenset({"A"}): 10.0,
                frozenset({"B"}): 20.0,
            }
        },
    )
    records = compute_value_error_records(
        scenario="toy",
        seed_type="implicit",
        mechanism="sealed_llm_proxy_vcg",
        top_k="",
        reported_bids={
            "i1": {
                frozenset({"A"}): 15.0,
                frozenset({"B"}): 15.0,
                frozenset({"C"}): 0.0,
            }
        },
        instance=instance,
    )

    aggregate = aggregate_value_error_records(records)

    assert aggregate["n"] == "3"
    assert float(aggregate["mae"]) == pytest.approx(10.0 / 3.0)
    assert float(aggregate["rmse"]) == pytest.approx(
        math.sqrt(50.0 / 3.0)
    )
    assert float(aggregate["mean_signed_error"]) == 0.0
    assert float(aggregate["overreport_rate"]) == pytest.approx(1.0 / 3.0)
    assert float(aggregate["underreport_rate"]) == pytest.approx(1.0 / 3.0)
    assert float(aggregate["exact_match_rate"]) == pytest.approx(1.0 / 3.0)
    assert float(aggregate["max_absolute_error"]) == 5.0
    assert float(aggregate["mean_relative_error"]) == pytest.approx(0.125)
    assert float(aggregate["mean_absolute_relative_error"]) == pytest.approx(
        0.375
    )


def test_value_error_csv_writers(tmp_path):
    instance = AuctionInstance(
        items=["A"],
        bidder_ids=["i1"],
        valuations={"i1": {frozenset({"A"}): 10.0}},
    )
    records = compute_value_error_records(
        scenario="toy",
        seed_type="implicit",
        mechanism="sealed_llm_proxy_vcg",
        top_k="",
        reported_bids={"i1": {frozenset({"A"}): 12.0}},
        instance=instance,
    )
    records_path = tmp_path / "records.csv"
    aggregate_path = tmp_path / "aggregate.csv"

    write_value_error_records_csv(records, records_path)
    write_value_error_aggregate_csv(
        group_value_error_records(records),
        aggregate_path,
    )

    with records_path.open(newline="") as file:
        record_rows = list(csv.DictReader(file))
    with aggregate_path.open(newline="") as file:
        aggregate_rows = list(csv.DictReader(file))

    assert record_rows[0]["bundle_label"] == "[A]"
    assert record_rows[0]["absolute_error"] == "2.0"
    assert aggregate_rows[0]["scenario"] == "toy"
    assert aggregate_rows[0]["mae"] == "2.0"
