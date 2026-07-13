from __future__ import annotations

import csv

import pytest

from auctionlab.experiments.llm_allocation_diagnostics import (
    aggregate_allocation_loss_records,
    allocation_loss_records_to_rows,
    bundle_label,
    compute_allocation_loss_records,
    group_allocation_loss_records,
    parse_allocation,
    write_allocation_loss_aggregate_csv,
    write_allocation_loss_records_csv,
)
from auctionlab.instances.base import AuctionInstance


def make_instance() -> AuctionInstance:
    return AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {
                frozenset({"A"}): 10.0,
                frozenset({"A", "B"}): 15.0,
            },
            "i2": {
                frozenset({"B"}): 8.0,
            },
        },
    )


def test_bundle_label():
    assert bundle_label(frozenset()) == "[]"
    assert bundle_label(frozenset({"B", "A"})) == "[A,B]"


def test_parse_allocation_accepts_export_and_pipe_formats():
    expected = {
        "i1": frozenset({"A", "B"}),
        "i2": frozenset(),
    }

    assert parse_allocation("i1:[A,B];i2:[]") == expected
    assert parse_allocation("i1:[A,B]|i2:[]") == expected
    assert parse_allocation("") == {}


@pytest.mark.parametrize(
    "text",
    [
        "i1=A",
        "i1:[A,A]",
        "i1:[A];i1:[B]",
        ":[A]",
    ],
)
def test_parse_allocation_rejects_malformed_input(text):
    with pytest.raises(ValueError):
        parse_allocation(text)


def test_compute_allocation_loss_records_includes_empty_bundles():
    records = compute_allocation_loss_records(
        scenario="toy",
        seed_type="implicit",
        mechanism="sealed_llm_proxy_vcg",
        top_k="",
        instance=make_instance(),
        full_info_allocation={
            "i1": frozenset({"A"}),
            "i2": frozenset({"B"}),
        },
        llm_allocation={
            "i1": frozenset({"A", "B"}),
            "i2": frozenset(),
        },
    )

    assert [record.bidder_id for record in records] == ["i1", "i2"]
    assert records[0].full_info_true_value == 10.0
    assert records[0].llm_true_value == 15.0
    assert records[0].true_value_delta == 5.0
    assert records[0].changed is True
    assert records[1].llm_bundle_label == "[]"
    assert records[1].llm_true_value == 0.0
    assert records[1].true_value_delta == -8.0


def test_aggregate_allocation_loss_records():
    records = compute_allocation_loss_records(
        scenario="toy",
        seed_type="implicit",
        mechanism="clock_llm_proxy_vcg",
        top_k="1",
        instance=make_instance(),
        full_info_allocation={
            "i1": frozenset({"A"}),
            "i2": frozenset({"B"}),
        },
        llm_allocation={
            "i1": frozenset({"A", "B"}),
            "i2": frozenset(),
        },
    )

    aggregate = aggregate_allocation_loss_records(records)

    assert aggregate["n_bidders"] == "2"
    assert aggregate["changed_bidder_count"] == "2"
    assert aggregate["changed_bidder_rate"] == "1.0"
    assert aggregate["full_info_true_welfare"] == "18.0"
    assert aggregate["llm_true_welfare"] == "15.0"
    assert aggregate["welfare_loss"] == "3.0"
    assert float(aggregate["efficiency"]) == pytest.approx(15.0 / 18.0)
    assert aggregate["positive_delta_count"] == "1"
    assert aggregate["negative_delta_count"] == "1"
    assert aggregate["zero_delta_count"] == "0"


def test_allocation_loss_rows_and_csv_writers(tmp_path):
    records = compute_allocation_loss_records(
        scenario="toy",
        seed_type="implicit",
        mechanism="sealed_llm_proxy_vcg",
        top_k="",
        instance=make_instance(),
        full_info_allocation={"i1": frozenset({"A"})},
        llm_allocation={"i1": frozenset({"A"})},
    )
    rows = allocation_loss_records_to_rows(records)
    records_path = tmp_path / "records.csv"
    aggregate_path = tmp_path / "aggregate.csv"

    write_allocation_loss_records_csv(records, records_path)
    write_allocation_loss_aggregate_csv(
        group_allocation_loss_records(records),
        aggregate_path,
    )

    assert rows[0]["full_info_bundle_label"] == "[A]"
    assert rows[0]["changed"] == "False"
    assert rows[1]["full_info_bundle_label"] == "[]"

    with records_path.open(newline="") as file:
        record_rows = list(csv.DictReader(file))
    with aggregate_path.open(newline="") as file:
        aggregate_rows = list(csv.DictReader(file))

    assert record_rows[0]["bidder_id"] == "i1"
    assert aggregate_rows[0]["n_bidders"] == "2"
