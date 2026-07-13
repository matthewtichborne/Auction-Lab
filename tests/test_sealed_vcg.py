from __future__ import annotations

from auctionlab.auctions.sealed_vcg import run_sealed_xor_vcg


def test_sealed_vcg_allocation_welfare_and_payments(
    toy_items,
    toy_bids,
    expected_toy_allocation,
):
    outcome = run_sealed_xor_vcg(toy_items, toy_bids)

    assert outcome.allocation == expected_toy_allocation
    assert outcome.welfare == 23.0

    assert outcome.payments == {
        "i1": 0.0,
        "i2": 1.0,
        "i3": 13.0,
    }