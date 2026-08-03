from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.experiments.event_policy import (
    best_neighbour_bundle,
    best_scarcity_avoiding_bundle,
    contested_goods_from_bundles,
    correction_fraction,
)
from auctionlab.proxies.base import RefinementRecord


def _bid() -> XorBid:
    return XorBid(
        bidder_id="b",
        atoms=[
            XorAtomicBid(frozenset({"A", "B"}), 100.0),
            XorAtomicBid(frozenset({"A"}), 80.0),
            XorAtomicBid(frozenset({"C"}), 70.0),
        ],
    )


def test_contested_goods_require_two_bundles():
    assert contested_goods_from_bundles([
        frozenset({"A", "B"}),
        frozenset({"A", "C"}),
        frozenset({"D"}),
    ]) == {"A"}


def test_scarcity_fallback_strictly_reduces_exposure():
    assert best_scarcity_avoiding_bundle(
        _bid(),
        frozenset({"A", "B"}),
        {"A", "B"},
    ) == frozenset({"C"})


def test_neighbour_prefers_small_symmetric_difference_then_value():
    assert best_neighbour_bundle(
        _bid(), frozenset({"A", "B"})
    ) == frozenset({"A"})


def test_correction_fraction_is_symmetric_and_bounded():
    record = RefinementRecord(
        bidder_id="b",
        mechanism="sealed",
        event_type="test",
        round_idx=0,
        bundle=frozenset({"A"}),
        old_value=50.0,
        new_value=100.0,
        reason=None,
    )
    assert correction_fraction(record) == 0.5
