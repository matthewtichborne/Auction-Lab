"""Winner determination over XOR bids.

Covers the welfare-maximising allocation itself and the second-stage
tie-break that prefers allocations serving more bidders, including that the
tie-break never trades away welfare and leaves a unique optimum untouched.
"""

from __future__ import annotations

from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.solvers.wdp_ilp import solve_wdp_xor_ilp, solve_wdp_xor_ilp_max_winners


def test_wdp_finds_welfare_maximising_allocation(
    toy_items,
    toy_bids,
    expected_toy_allocation,
):
    result = solve_wdp_xor_ilp(toy_items, toy_bids)

    assert result.allocation == expected_toy_allocation
    assert result.welfare == 23.0


def _tied_welfare_bids() -> list[XorBid]:
    """A genuine welfare tie between a 1-winner and a 2-winner allocation.

    bidder1 values {A,B} together at 20; bidder2/bidder3 each value their
    own singleton at 10. {bidder1: {A,B}} and {bidder2: {A}, bidder3: {B}}
    both achieve welfare 20.
    """
    return [
        XorBid(
            bidder_id="bidder1",
            atoms=[XorAtomicBid(bundle=frozenset({"A", "B"}), value=20.0)],
        ),
        XorBid(
            bidder_id="bidder2",
            atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=10.0)],
        ),
        XorBid(
            bidder_id="bidder3",
            atoms=[XorAtomicBid(bundle=frozenset({"B"}), value=10.0)],
        ),
    ]


def test_max_winners_breaks_welfare_tie_in_favor_of_more_winners():
    items = ["A", "B"]
    bids = _tied_welfare_bids()

    stage1 = solve_wdp_xor_ilp(items, bids)
    assert stage1.welfare == 20.0

    stage2 = solve_wdp_xor_ilp_max_winners(items, bids, min_welfare=stage1.welfare)

    assert stage2.welfare == 20.0
    assert stage2.allocation == {
        "bidder1": frozenset(),
        "bidder2": frozenset({"A"}),
        "bidder3": frozenset({"B"}),
    }


def test_max_winners_respects_welfare_floor():
    items = ["A", "B"]
    bids = _tied_welfare_bids()

    stage1 = solve_wdp_xor_ilp(items, bids)
    stage2 = solve_wdp_xor_ilp_max_winners(items, bids, min_welfare=stage1.welfare)

    # Stage 2 never sacrifices welfare to win more bidders.
    assert stage2.welfare >= stage1.welfare - 1e-6


def test_max_winners_matches_unique_optimum_when_no_tie_exists(
    toy_items,
    toy_bids,
    expected_toy_allocation,
):
    stage1 = solve_wdp_xor_ilp(toy_items, toy_bids)
    stage2 = solve_wdp_xor_ilp_max_winners(
        toy_items, toy_bids, min_welfare=stage1.welfare
    )

    assert stage2.welfare == stage1.welfare
    assert stage2.allocation == expected_toy_allocation


def test_wdp_tie_result_is_independent_of_atom_order():
    items = ["A", "B"]
    atoms = [
        XorAtomicBid(bundle=frozenset({"A"}), value=10.0),
        XorAtomicBid(bundle=frozenset({"B"}), value=10.0),
    ]
    bids_forward = [XorBid(bidder_id="bidder", atoms=atoms)]
    bids_reverse = [XorBid(bidder_id="bidder", atoms=list(reversed(atoms)))]

    forward = solve_wdp_xor_ilp(items, bids_forward)
    reverse = solve_wdp_xor_ilp(items, bids_reverse)

    assert forward == reverse
