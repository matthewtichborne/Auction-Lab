from __future__ import annotations

import pytest

from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.experiments.llm_comparison import (
    reported_bids_to_str,
    run_batch_sealed_llm_comparisons,
    run_sealed_llm_comparison,
    sealed_llm_comparison_to_row,
    sealed_llm_comparisons_to_rows,
    xor_bid_to_str,
)
from auctionlab.experiments.llm_runner import make_llm_proxies_for_instance
from auctionlab.instances.base import AuctionInstance
from auctionlab.llm.clients import MockLlmClient


CANDIDATE_BUNDLES = [
    frozenset({"A"}),
    frozenset({"B"}),
]


@pytest.fixture
def comparison_instance() -> AuctionInstance:
    return AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {
                frozenset({"A"}): 10.0,
                frozenset({"B"}): 1.0,
            },
            "i2": {
                frozenset({"A"}): 2.0,
                frozenset({"B"}): 9.0,
            },
        },
    )


def make_proxies(
    instance: AuctionInstance,
    *,
    i1_values: tuple[float, float] = (10.0, 1.0),
    i2_values: tuple[float, float] = (2.0, 9.0),
):
    clients = {
        "i1": MockLlmClient(
            [
                f'{{"bundle_value": {i1_values[0]}}}',
                f'{{"bundle_value": {i1_values[1]}}}',
            ]
        ),
        "i2": MockLlmClient(
            [
                f'{{"bundle_value": {i2_values[0]}}}',
                f'{{"bundle_value": {i2_values[1]}}}',
            ]
        ),
    }
    return make_llm_proxies_for_instance(
        instance=instance,
        scenario_description="A two-item auction.",
        person_seeds={
            "i1": "Prefers A.",
            "i2": "Prefers B.",
        },
        item_descriptions={
            "A": "Item A",
            "B": "Item B",
        },
        clients=clients,
        epsilon=1.0,
    )


def test_xor_bid_to_str_is_stable():
    bid = XorBid(
        bidder_id="i1",
        atoms=[
            XorAtomicBid(bundle=frozenset({"B", "A"}), value=15.0),
            XorAtomicBid(bundle=frozenset({"A"}), value=10.0),
        ],
    )

    assert xor_bid_to_str(bid) == "[A]:10.0;[A,B]:15.0"


def test_reported_bids_to_str_is_stable():
    bids = {
        "i2": XorBid(
            bidder_id="i2",
            atoms=[
                XorAtomicBid(bundle=frozenset({"B"}), value=9.0),
            ],
        ),
        "i1": XorBid(
            bidder_id="i1",
            atoms=[
                XorAtomicBid(bundle=frozenset({"A"}), value=10.0),
                XorAtomicBid(
                    bundle=frozenset({"A", "B"}),
                    value=15.0,
                ),
            ],
        ),
    }

    assert reported_bids_to_str(bids) == (
        "i1={[A]:10.0;[A,B]:15.0}|i2={[B]:9.0}"
    )


def test_sealed_llm_comparison_matches_full_information(comparison_instance):
    result = run_sealed_llm_comparison(
        instance=comparison_instance,
        instance_name="matching",
        proxies=make_proxies(comparison_instance),
        candidate_bundles=CANDIDATE_BUNDLES,
    )
    row = sealed_llm_comparison_to_row(result)

    expected_allocation = {
        "i1": frozenset({"A"}),
        "i2": frozenset({"B"}),
    }
    assert result.full_info.allocation == expected_allocation
    assert result.llm_proxy.allocation == expected_allocation
    assert row["efficiency"] == 1.0
    assert row["allocation_match"] is True
    assert row["welfare_match"] is True
    assert row["candidate_bundle_count"] == 4
    assert row["llm_proxy_query_count"] == 4
    assert "[A]:10.0" in row["llm_proxy_reported_bids"]
    assert "[B]:9.0" in row["llm_proxy_reported_bids"]


def test_sealed_llm_comparison_separates_reported_and_true_welfare(
    comparison_instance,
):
    result = run_sealed_llm_comparison(
        instance=comparison_instance,
        instance_name="overreported",
        proxies=make_proxies(
            comparison_instance,
            i1_values=(1000.0, 1.0),
            i2_values=(2.0, 900.0),
        ),
        candidate_bundles=CANDIDATE_BUNDLES,
    )

    row = sealed_llm_comparison_to_row(result)

    assert row["llm_proxy_reported_welfare"] == 1900.0
    assert row["llm_proxy_true_welfare"] == 19.0
    assert row["efficiency"] == 1.0
    assert row["welfare_match"] is True


def test_sealed_llm_comparison_detects_degraded_proxy(comparison_instance):
    result = run_sealed_llm_comparison(
        instance=comparison_instance,
        instance_name="degraded",
        proxies=make_proxies(
            comparison_instance,
            i1_values=(1.0, 8.0),
            i2_values=(7.0, 1.0),
        ),
        candidate_bundles=CANDIDATE_BUNDLES,
    )
    row = sealed_llm_comparison_to_row(result)

    assert row["allocation_match"] is False
    assert row["llm_proxy_true_welfare"] < row["full_info_welfare"]
    assert row["efficiency"] < 1.0


def test_batch_comparison_and_row_conversion(comparison_instance):
    results = run_batch_sealed_llm_comparisons(
        [
            (
                "first",
                comparison_instance,
                make_proxies(comparison_instance),
                CANDIDATE_BUNDLES,
            ),
            (
                "second",
                comparison_instance,
                make_proxies(comparison_instance),
                CANDIDATE_BUNDLES,
            ),
        ]
    )
    rows = sealed_llm_comparisons_to_rows(results)

    assert [result.instance_name for result in results] == ["first", "second"]
    assert [row["instance_name"] for row in rows] == ["first", "second"]


def test_epsilon_by_bidder_is_stable(comparison_instance):
    result = run_sealed_llm_comparison(
        instance=comparison_instance,
        instance_name="stable_epsilon",
        proxies=make_proxies(comparison_instance),
        candidate_bundles=CANDIDATE_BUNDLES,
    )

    row = sealed_llm_comparison_to_row(result)

    assert row["epsilon_by_bidder"] == "i1:1.0;i2:1.0"
