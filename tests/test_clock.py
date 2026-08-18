"""Ascending clock mechanics and the demand oracle.

Covers supplementary bid bookkeeping (a later report for a bundle replaces an
earlier one, even downwards), deterministic tie-breaking towards the smaller
bundle, and that the oracle returns a bidder's full valuation table
irrespective of the top-k reporting limit.
"""

from __future__ import annotations

import pytest

from auctionlab.auctions.clock import (
    ClockConfig,
    ClockState,
    finalize_from_supplementary_vcg,
    record_supplementary_bids,
    run_ascending_clock_with_supplementary,
)
from auctionlab.bids.xor import XorAtomicBid
from auctionlab.instances.base import make_demand_oracle


def test_record_supplementary_bids_overwrites_with_latest_even_if_lower():
    # Regression guard: an earlier (inflated) provisional value must not
    # survive a later round's genuine downward correction. Real-world case:
    # PV overestimates a bundle, refinement later corrects it down -- the
    # corrected value must win, not the historical maximum.
    state = ClockState(
        round_idx=0, prices={}, history=[], supplementary={"b1": []}
    )

    record_supplementary_bids(
        state,
        {"b1": [XorAtomicBid(bundle=frozenset({"A"}), value=2800.0)]},
    )
    record_supplementary_bids(
        state,
        {"b1": [XorAtomicBid(bundle=frozenset({"A"}), value=1700.0)]},
    )

    assert state.supplementary["b1"] == [
        XorAtomicBid(bundle=frozenset({"A"}), value=1700.0)
    ]


def test_toy_oracle_tie_breaks_to_smaller_bundle_for_i1(toy_instance):
    demand_oracle = make_demand_oracle(toy_instance)

    prices = {"A": 7.0, "B": 5.0, "C": 2.0}

    response = demand_oracle("i1", prices)

    assert response.primary_bundle == frozenset({"A"})
    # supplementary_atoms is i1's entire valuation table regardless of
    # prices or top_k, so the final WDP+VCG resolution always sees every
    # bundle this bidder is known to value.
    assert XorAtomicBid(bundle=frozenset({"A"}), value=10.0) in (
        response.supplementary_atoms
    )


def test_toy_oracle_returns_full_valuation_table_regardless_of_top_k(
    toy_instance,
):
    demand_oracle = make_demand_oracle(toy_instance, top_k=1)

    response = demand_oracle("i1", {"A": 0.0, "B": 0.0, "C": 0.0})

    assert response.primary_bundle == frozenset({"A", "B"})
    assert set(response.supplementary_atoms) == {
        XorAtomicBid(bundle=frozenset({"A", "B"}), value=15.0),
        XorAtomicBid(bundle=frozenset({"A"}), value=10.0),
    }


def test_demand_oracle_rejects_non_positive_top_k(toy_instance):
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        make_demand_oracle(toy_instance, top_k=0)


def test_top_k_includes_multiple_bundles_in_primary_demand(toy_instance):
    # i2 has two disjoint singleton bundles: {B}: 9.0, {C}: 7.0.
    demand_oracle = make_demand_oracle(toy_instance, top_k=2)

    response = demand_oracle("i2", {"A": 0.0, "B": 0.0, "C": 0.0})

    # Best-surplus bundle is still just {B} for display/tie-break purposes...
    assert response.primary_bundle == frozenset({"B"})
    # ...but with top_k=2, both of i2's positive-surplus bundles count as
    # primary demand, since both fit within the top-2 ranking.
    assert set(response.primary_bundles) == {
        frozenset({"B"}),
        frozenset({"C"}),
    }


def test_top_k_one_reports_only_the_single_best_bundle(toy_instance):
    demand_oracle = make_demand_oracle(toy_instance, top_k=1)

    response = demand_oracle("i2", {"A": 0.0, "B": 0.0, "C": 0.0})

    assert response.primary_bundles == [frozenset({"B"})]


def test_top_k_can_change_clock_price_dynamics(toy_instance):
    """top_k now controls how many bundles drive excess demand per round.

    i2's two disjoint bundles ({B}, {C}) both contribute excess demand
    simultaneously under top_k=2 instead of one at a time under top_k=1, so
    the resulting price paths are allowed to diverge. supplementary_atoms
    is unaffected either way: it always returns each bidder's full known
    bid regardless of top_k.
    """
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)

    state_top_1, _ = run_ascending_clock_with_supplementary(
        items=toy_instance.items,
        bidder_ids=toy_instance.bidder_ids,
        demand_oracle=make_demand_oracle(toy_instance, top_k=1),
        cfg=cfg,
    )
    state_top_2, _ = run_ascending_clock_with_supplementary(
        items=toy_instance.items,
        bidder_ids=toy_instance.bidder_ids,
        demand_oracle=make_demand_oracle(toy_instance, top_k=2),
        cfg=cfg,
    )

    assert state_top_1.prices != state_top_2.prices
    assert len(state_top_2.supplementary["i1"]) == len(
        state_top_1.supplementary["i1"]
    )


def test_clock_supplementary_vcg_matches_expected_outcome(
    toy_instance,
    expected_toy_allocation,
):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)

    demand_oracle = make_demand_oracle(toy_instance)

    state, provisional = run_ascending_clock_with_supplementary(
        items=toy_instance.items,
        bidder_ids=toy_instance.bidder_ids,
        demand_oracle=demand_oracle,
        cfg=cfg,
    )

    outcome = finalize_from_supplementary_vcg(
        toy_instance.items,
        state.supplementary,
    )

    assert len(state.history) == 11
    assert state.prices == {"A": 10.0, "B": 5.0, "C": 3.0}

    assert provisional == expected_toy_allocation
    assert outcome.allocation == expected_toy_allocation
    assert outcome.welfare == 23.0
    assert outcome.payments == {
        "i1": 0.0,
        "i2": 1.0,
        "i3": 13.0,
    }
    assert set(outcome.vcg_counterfactuals) == {"i1", "i2", "i3"}
    assert outcome.vcg_counterfactuals["i3"].welfare == 22.0
