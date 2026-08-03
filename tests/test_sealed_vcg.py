from __future__ import annotations

from auctionlab.auctions.sealed_vcg import run_sealed_xor_vcg
from auctionlab.payments import vcg


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

    assert set(outcome.vcg_counterfactuals) == {"i1", "i2", "i3"}
    assert outcome.vcg_counterfactuals["i2"].welfare == 15.0
    assert outcome.vcg_counterfactuals["i2"].allocation["i1"] == frozenset(
        {"A", "B"}
    )
    assert outcome.vcg_counterfactuals["i3"].welfare == 22.0


def test_vcg_witness_logging_does_not_add_wdp_solves(
    toy_items,
    toy_bids,
    monkeypatch,
):
    original = vcg.solve_wdp_xor_ilp
    calls = 0

    def counted_solve(items, bids):
        nonlocal calls
        calls += 1
        return original(items, bids)

    monkeypatch.setattr(vcg, "solve_wdp_xor_ilp", counted_solve)

    outcome = run_sealed_xor_vcg(toy_items, toy_bids)

    # One counterfactual solve per bidder. The full-allocation solve occurs
    # in sealed_vcg and is supplied to the payment routine, so witness
    # retention causes no additional solves.
    assert calls == len(toy_bids)
    assert len(outcome.vcg_counterfactuals) == len(toy_bids)
