"""The deterministic full-information proxy.

This proxy answers exact value queries from the hidden table and stands in
for a person during the frozen experiments. Covers its initial-bid modes,
that refinement installs the true value and updates query statistics, and
that query accounting stays consistent.
"""

from __future__ import annotations

import pytest

from auctionlab.bids.xor import XorAtomicBid
from auctionlab.instances.base import AuctionInstance
from auctionlab.proxies.base import ElicitationEvent
from auctionlab.proxies.full_info import FullInfoAuctionProxy


def make_instance() -> AuctionInstance:
    return AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1"],
        valuations={
            "i1": {
                frozenset({"A"}): 10.0,
                frozenset({"B"}): 5.0,
                frozenset({"A", "B"}): 12.0,
            },
        },
    )


def test_empty_initial_bid_starts_with_no_atoms():
    proxy = FullInfoAuctionProxy(
        bidder_id="i1",
        instance=make_instance(),
        initial="empty",
    )

    assert proxy.current_bid().atoms == []
    assert proxy.submit_bid().atoms == []
    assert proxy.stats().value_queries == 0


def test_all_atoms_initial_bid_matches_instance_valuations():
    instance = make_instance()
    proxy = FullInfoAuctionProxy(
        bidder_id="i1",
        instance=instance,
        initial="all_atoms",
    )

    bid = proxy.current_bid()

    assert sorted(bid.atoms, key=lambda a: sorted(a.bundle)) == sorted(
        [
            XorAtomicBid(bundle=bundle, value=value)
            for bundle, value in instance.valuations["i1"].items()
        ],
        key=lambda a: sorted(a.bundle),
    )


def test_selected_initial_bid_queries_true_values():
    proxy = FullInfoAuctionProxy(
        bidder_id="i1",
        instance=make_instance(),
        initial="selected",
        initial_bundles=[frozenset({"A"})],
    )

    bid = proxy.current_bid()

    assert bid.atoms == [XorAtomicBid(bundle=frozenset({"A"}), value=10.0)]
    assert proxy.stats().value_queries == 1


def test_refine_adds_true_value_atom_and_updates_stats():
    proxy = FullInfoAuctionProxy(
        bidder_id="i1",
        instance=make_instance(),
        initial="empty",
    )

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="probe",
            bidder_id="i1",
            bundle=frozenset({"B"}),
        )
    )

    assert proxy.current_bid().atoms == [
        XorAtomicBid(bundle=frozenset({"B"}), value=5.0)
    ]
    assert proxy.stats().refinement_queries == 1
    assert proxy.stats().value_queries == 1


def test_refine_replaces_existing_atom():
    instance = make_instance()
    proxy = FullInfoAuctionProxy(
        bidder_id="i1",
        instance=instance,
        initial="all_atoms",
    )

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="probe",
            bidder_id="i1",
            bundle=frozenset({"A"}),
        )
    )

    atoms_for_a = [
        atom for atom in proxy.current_bid().atoms if atom.bundle == frozenset({"A"})
    ]
    assert len(atoms_for_a) == 1
    assert atoms_for_a[0].value == 10.0
    assert proxy.stats().refinement_queries == 1


def test_refine_records_old_and_new_value():
    proxy = FullInfoAuctionProxy(
        bidder_id="i1",
        instance=make_instance(),
        initial="empty",
    )

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="probe",
            bidder_id="i1",
            bundle=frozenset({"B"}),
            reason="initial probe",
            round_idx=0,
        )
    )

    records = proxy.refinement_records()
    assert len(records) == 1
    record = records[0]
    assert record.bidder_id == "i1"
    assert record.mechanism == "test"
    assert record.event_type == "probe"
    assert record.round_idx == 0
    assert record.bundle == frozenset({"B"})
    assert record.old_value is None
    assert record.new_value == 5.0
    assert record.reason == "initial probe"

    # current_bid() reflects the refined value.
    assert proxy.current_bid().value_of(frozenset({"B"})) == 5.0
    assert proxy.submit_bid().value_of(frozenset({"B"})) == 5.0


def test_refine_unchanged_value_is_still_recorded():
    instance = make_instance()
    proxy = FullInfoAuctionProxy(
        bidder_id="i1",
        instance=instance,
        initial="all_atoms",
    )

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="probe",
            bidder_id="i1",
            bundle=frozenset({"A"}),
        )
    )

    records = proxy.refinement_records()
    assert len(records) == 1
    assert records[0].old_value == 10.0
    assert records[0].new_value == 10.0


def test_refine_ignores_empty_bundle():
    proxy = FullInfoAuctionProxy(
        bidder_id="i1",
        instance=make_instance(),
        initial="empty",
    )

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="probe",
            bidder_id="i1",
            bundle=frozenset(),
        )
    )

    assert proxy.current_bid().atoms == []
    assert proxy.stats().refinement_queries == 0


def test_demand_at_prices_returns_max_surplus_bundle():
    proxy = FullInfoAuctionProxy(
        bidder_id="i1",
        instance=make_instance(),
        initial="empty",
    )

    response = proxy.demand_at_prices({"A": 3.0, "B": 3.0}, round_idx=0, top_k=2)

    # surplus: A -> 7, B -> 2, AB -> 6; best is A.
    assert response.primary_bundle == frozenset({"A"})
    # supplementary_atoms is the bidder's entire valuation table, not capped
    # by top_k, so the final WDP+VCG resolution sees everything they value.
    assert {atom.bundle for atom in response.supplementary_atoms} == {
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"A", "B"}),
    }
    assert proxy.stats().demand_queries == 1


def test_demand_at_prices_returns_none_when_no_positive_surplus():
    proxy = FullInfoAuctionProxy(
        bidder_id="i1",
        instance=make_instance(),
        initial="empty",
    )

    response = proxy.demand_at_prices({"A": 100.0, "B": 100.0}, round_idx=0, top_k=1)

    assert response.primary_bundle is None
    # supplementary_atoms still returns the full valuation table even with
    # no positive-surplus primary demand: the final WDP doesn't depend on
    # current prices, only on what's known.
    assert {atom.bundle for atom in response.supplementary_atoms} == {
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"A", "B"}),
    }


def test_demand_at_prices_rejects_invalid_top_k():
    proxy = FullInfoAuctionProxy(
        bidder_id="i1",
        instance=make_instance(),
        initial="empty",
    )

    with pytest.raises(ValueError, match="top_k"):
        proxy.demand_at_prices({"A": 0.0, "B": 0.0}, round_idx=0, top_k=0)


def test_invalid_initial_mode_rejected():
    with pytest.raises(ValueError, match="initial"):
        FullInfoAuctionProxy(
            bidder_id="i1",
            instance=make_instance(),
            initial="bogus",
        )
