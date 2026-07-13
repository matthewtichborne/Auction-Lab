from __future__ import annotations

from auctionlab.auction_types import Bundle
from auctionlab.auctions.ceca import (
    CecaConfig,
    build_lindahl_prices,
    finalize_ceca,
    finalize_ceca_pay_as_bid,
    finalize_ceca_vcg,
    run_ceca,
)
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.instances.base import CecaStepResponse
from auctionlab.payments.vcg import compute_vcg_payments


def make_deterministic_oracle(true_values: dict[str, dict[Bundle, float]]):
    """A CecaStepOracle that always reports the bidder's true-surplus-maximizing
    bundle among their known true-valued bundles plus the current allocation."""

    def oracle(
        bidder_id: str,
        prices,
        current_bundle: Bundle,
        round_idx: int,
    ) -> CecaStepResponse:
        candidates = set(true_values[bidder_id]) | {current_bundle}
        best_bundle: Bundle | None = None
        best_surplus = float("-inf")
        for bundle in candidates:
            value = true_values[bidder_id].get(bundle, 0.0)
            price = prices(bundle)
            surplus = value - price
            if surplus > best_surplus:
                best_surplus = surplus
                best_bundle = bundle

        if best_bundle == current_bundle:
            return CecaStepResponse(satisfied=True, demanded_bundle=None, value=None)

        value = true_values[bidder_id].get(best_bundle, 0.0)
        return CecaStepResponse(
            satisfied=False, demanded_bundle=best_bundle, value=value
        )

    return oracle


def test_run_ceca_converges_to_competitive_equilibrium():
    items = ["A", "B"]
    bidder_ids = ["i1", "i2"]
    true_values = {
        "i1": {frozenset({"A"}): 10.0},
        "i2": {frozenset({"B"}): 8.0},
    }
    oracle = make_deterministic_oracle(true_values)
    cfg = CecaConfig(max_rounds=20)

    state = run_ceca(items, bidder_ids, oracle, cfg)

    assert state.converged is True
    assert state.allocation == {"i1": frozenset({"A"}), "i2": frozenset({"B"})}
    assert state.manifest_bids["i1"].value_of(frozenset({"A"})) == 10.0
    assert state.manifest_bids["i2"].value_of(frozenset({"B"})) == 8.0


def test_run_ceca_halts_at_max_rounds_and_flags_non_convergence():
    items = ["A"]
    bidder_ids = ["i1"]

    def never_satisfied_oracle(bidder_id, prices, current_bundle, round_idx):
        return CecaStepResponse(
            satisfied=False, demanded_bundle=frozenset({"A"}), value=10.0
        )

    cfg = CecaConfig(max_rounds=3)
    state = run_ceca(items, bidder_ids, never_satisfied_oracle, cfg)

    assert state.converged is False
    assert state.round_idx == 2
    assert len(state.history) == 3


def _tied_welfare_manifest() -> dict[str, XorBid]:
    """Same tie shape as test_wdp.py: {bidder1:{A,B}}=20 ties
    {bidder2:{A},bidder3:{B}}=10+10."""
    return {
        "bidder1": XorBid(
            bidder_id="bidder1",
            atoms=[XorAtomicBid(bundle=frozenset({"A", "B"}), value=20.0)],
        ),
        "bidder2": XorBid(
            bidder_id="bidder2",
            atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=10.0)],
        ),
        "bidder3": XorBid(
            bidder_id="bidder3",
            atoms=[XorAtomicBid(bundle=frozenset({"B"}), value=10.0)],
        ),
    }


def test_finalize_ceca_picks_broader_allocation_on_welfare_tie():
    items = ["A", "B"]
    manifest_bids = _tied_welfare_manifest()

    stage1, stage2 = finalize_ceca(items, manifest_bids)

    assert stage1.welfare == 20.0
    assert stage2.welfare == 20.0
    assert stage2.allocation == {
        "bidder1": frozenset(),
        "bidder2": frozenset({"A"}),
        "bidder3": frozenset({"B"}),
    }


def test_finalize_ceca_pay_as_bid_charges_own_reported_value():
    items = ["A", "B"]
    manifest_bids = _tied_welfare_manifest()

    outcome = finalize_ceca_pay_as_bid(items, manifest_bids)

    assert outcome.welfare == 20.0
    for bidder_id, bid in manifest_bids.items():
        expected = bid.value_of(outcome.allocation.get(bidder_id, frozenset()))
        assert outcome.payments[bidder_id] == expected


def test_finalize_ceca_vcg_matches_compute_vcg_payments_directly():
    items = ["A", "B"]
    manifest_bids = _tied_welfare_manifest()

    outcome = finalize_ceca_vcg(items, manifest_bids)

    _stage1, stage2 = finalize_ceca(items, manifest_bids)
    expected_payments = compute_vcg_payments(
        items, list(manifest_bids.values()), full=stage2
    )

    assert outcome.payments == expected_payments


def test_build_lindahl_prices_uses_xor_induction():
    bid = XorBid(
        bidder_id="i1",
        atoms=[
            XorAtomicBid(bundle=frozenset({"A"}), value=10.0),
            XorAtomicBid(bundle=frozenset({"A", "B"}), value=25.0),
        ],
    )

    prices = build_lindahl_prices(bid)

    # Manifest atoms priced at their own value.
    assert prices(frozenset({"A"})) == 10.0
    assert prices(frozenset({"A", "B"})) == 25.0
    # Superset of {A}: priced at max manifest atom ⊆ it.
    # {A,B,C} contains {A} (10) and {A,B} (25) -> 25.
    assert prices(frozenset({"A", "B", "C"})) == 25.0
    # {B,C} contains no manifest atom -> 0.
    assert prices(frozenset({"B", "C"})) == 0.0
    # {B} alone contains no manifest atom -> 0.
    assert prices(frozenset({"B"})) == 0.0
