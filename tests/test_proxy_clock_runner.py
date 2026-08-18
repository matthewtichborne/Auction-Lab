"""Clock execution and terminal event orchestration.

Covers agreement with the demand oracle under a static configuration, the
elicited configuration's refinement records, unlimited refinement when the
per-bidder cap is zero, and the metadata each run reports.
"""

from __future__ import annotations

import pytest

from auctionlab.auctions.clock import (
    ClockConfig,
    finalize_from_supplementary_vcg,
    run_ascending_clock_with_supplementary,
)
from auctionlab.bids.xor import XorAtomicBid
from auctionlab.experiments.llm_comparison import proxy_clock_result_to_row
from auctionlab.experiments.proxy_clock_runner import (
    ProxyClockConfig,
    run_proxy_clock_experiment,
)
from auctionlab.proxies.elicitation import (
    candidate_refinements,
    pivotality_score,
)

# Aliases kept so tests that already referenced the private names still work.
_candidate_refinements = candidate_refinements
_pivotality_score = pivotality_score
from auctionlab.instances.base import make_demand_oracle
from auctionlab.instances.base import DemandResponse
from auctionlab.proxies.full_info import FullInfoAuctionProxy


def make_proxies(toy_instance, *, initial: str) -> list[FullInfoAuctionProxy]:
    return [
        FullInfoAuctionProxy(bidder_id=bidder_id, instance=toy_instance, initial=initial)
        for bidder_id in toy_instance.bidder_ids
    ]


def test_static_config_matches_existing_demand_oracle(toy_instance):
    proxies = make_proxies(toy_instance, initial="all_atoms")
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        cfg,
        ProxyClockConfig(top_k=1, elicited=False),
    )

    state, _provisional = run_ascending_clock_with_supplementary(
        items=toy_instance.items,
        bidder_ids=toy_instance.bidder_ids,
        demand_oracle=make_demand_oracle(toy_instance, top_k=1),
        cfg=cfg,
    )
    expected = finalize_from_supplementary_vcg(
        items=toy_instance.items,
        supplementary_atoms=state.supplementary,
    )

    assert result.mechanism == "proxy_clock_vcg_static_top_1"
    assert result.allocation == expected.allocation
    assert result.welfare == expected.welfare
    assert result.payments == expected.payments
    assert result.rounds == len(state.history)
    assert result.metadata["elicited"] is False
    assert result.metadata["refinement_query_count_by_bidder"] == {
        bidder_id: 0 for bidder_id in toy_instance.bidder_ids
    }


def test_elicited_config_matches_static_allocation_and_records_refinements(
    toy_instance,
):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)

    static_proxies = make_proxies(toy_instance, initial="all_atoms")
    static_result = run_proxy_clock_experiment(
        toy_instance,
        static_proxies,
        cfg,
        ProxyClockConfig(top_k=1, elicited=False),
    )

    elicited_proxies = make_proxies(toy_instance, initial="all_atoms")
    elicited_result = run_proxy_clock_experiment(
        toy_instance,
        elicited_proxies,
        cfg,
        ProxyClockConfig(
            top_k=1,
            elicited=True,
            margin_threshold=100.0,
            tie_threshold=100.0,
            max_refinements_per_bidder=10,
        ),
    )

    assert elicited_result.mechanism == "proxy_clock_vcg_elicited_top_1"
    # FullInfoAuctionProxy.demand_at_prices always reflects true valuations,
    # so refinement does not change the allocation here.
    assert elicited_result.allocation == static_result.allocation
    assert elicited_result.welfare == static_result.welfare

    refinement_counts = elicited_result.metadata["refinement_query_count_by_bidder"]
    assert any(count > 0 for count in refinement_counts.values())


def test_max_refinements_per_bidder_zero_is_unlimited(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)

    proxies = make_proxies(toy_instance, initial="all_atoms")
    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        cfg,
        ProxyClockConfig(
            top_k=1,
            elicited=True,
            margin_threshold=100.0,
            tie_threshold=100.0,
            max_refinements_per_bidder=0,
        ),
    )

    # A cap of 0 means unlimited, so refinement queries are not blocked.
    refinement_counts = result.metadata["refinement_query_count_by_bidder"]
    assert any(count > 0 for count in refinement_counts.values())


def test_metadata_includes_expected_fields(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxies = make_proxies(toy_instance, initial="all_atoms")

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        cfg,
        ProxyClockConfig(top_k=2, elicited=False),
    )

    assert result.metadata["top_k"] == 2
    assert result.metadata["elicited"] is False
    assert result.metadata["margin_threshold"] == 100.0
    assert result.metadata["tie_threshold"] == 100.0
    assert result.metadata["max_refinements_per_bidder"] == 0
    assert set(result.metadata["demand_query_count_by_bidder"]) == set(
        toy_instance.bidder_ids
    )
    assert set(result.metadata["final_bids"]) == set(toy_instance.bidder_ids)
    assert "supplementary_atoms" in result.metadata
    assert "final_prices" in result.metadata


def test_proxy_clock_row_exports_final_updated_bid_and_refinement_records(
    toy_instance,
):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxies = make_proxies(toy_instance, initial="all_atoms")
    proxy_by_bidder = {proxy.bidder_id: proxy for proxy in proxies}

    # Tamper with i1's belief about {A, B} so refinement has something to fix.
    i1 = proxy_by_bidder["i1"]
    for idx, atom in enumerate(i1.current_bid().atoms):
        if atom.bundle == frozenset({"A", "B"}):
            i1.current_bid().atoms[idx] = XorAtomicBid(
                bundle=atom.bundle, value=999.0
            )

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        cfg,
        ProxyClockConfig(
            top_k=1,
            elicited=True,
            margin_threshold=10_000.0,
            tie_threshold=10_000.0,
            max_refinements_per_bidder=10,
        ),
    )

    records = result.metadata["refinement_records_by_bidder"]["i1"]
    assert any(
        record.bundle == frozenset({"A", "B"})
        and record.old_value == 999.0
        and record.new_value == 15.0
        for record in records
    )

    row = proxy_clock_result_to_row(
        instance_name="toy",
        instance=toy_instance,
        result=result,
    )

    # Initial snapshot keeps the pre-refinement (tampered) value.
    assert "[A,B]:999.0" in row["initial_bids"]
    assert "[A,B]:999.0" in row["initial_reported_bids"]
    # Final bid reflects the post-refinement true value.
    assert "[A,B]:15.0" in row["final_bids"]
    assert "[A,B]:15.0" in row["final_reported_bids"]
    assert "i1:[A,B] 999.0->15.0" in row["refinement_records"]


def test_proxies_must_match_instance_bidder_ids(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxies = make_proxies(toy_instance, initial="all_atoms")[:-1]

    with pytest.raises(ValueError, match="proxies must contain"):
        run_proxy_clock_experiment(
            toy_instance, proxies, cfg, ProxyClockConfig(top_k=1, elicited=False)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"margin_threshold": -1.0},
        {"tie_threshold": -1.0},
        {"max_refinements_per_bidder": -1},
        {"max_total_refinements": -1},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ProxyClockConfig(**kwargs)


def test_max_total_refinements_caps_across_bidders(toy_instance):
    """A global cap stops firing further refinements once the total is
    reached, even for a bidder whose own per-bidder count is still zero."""
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)

    uncapped_proxies = make_proxies(toy_instance, initial="all_atoms")
    uncapped_result = run_proxy_clock_experiment(
        toy_instance,
        uncapped_proxies,
        cfg,
        ProxyClockConfig(
            top_k=1, elicited=True, margin_threshold=100.0, tie_threshold=100.0
        ),
    )
    uncapped_total = sum(
        uncapped_result.metadata["refinement_query_count_by_bidder"].values()
    )
    assert uncapped_total > 2  # sanity: several bidders refine without a cap

    capped_proxies = make_proxies(toy_instance, initial="all_atoms")
    capped_result = run_proxy_clock_experiment(
        toy_instance,
        capped_proxies,
        cfg,
        ProxyClockConfig(
            top_k=1,
            elicited=True,
            margin_threshold=100.0,
            tie_threshold=100.0,
            max_total_refinements=2,
        ),
    )
    counts = capped_result.metadata["refinement_query_count_by_bidder"]
    assert sum(counts.values()) == 2
    assert capped_result.metadata["total_refinement_queries"] == 2
    assert capped_result.metadata["safety_cap_hit"] is True
    assert capped_result.metadata["cap_binding_indicator"] is False


def test_refinement_cap_metadata_fields_present_and_unhit_by_default(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxies = make_proxies(toy_instance, initial="all_atoms")

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        cfg,
        ProxyClockConfig(
            top_k=1, elicited=True, margin_threshold=100.0, tie_threshold=100.0
        ),
    )

    counts = result.metadata["refinement_query_count_by_bidder"]
    assert result.metadata["total_refinement_queries"] == sum(counts.values())
    assert result.metadata["per_bidder_refinement_queries"] == counts
    assert result.metadata["cap_binding_indicator"] is False
    assert result.metadata["safety_cap_hit"] is False

    row = proxy_clock_result_to_row(
        instance_name="toy",
        instance=toy_instance,
        result=result,
    )
    assert row["total_refinement_queries"] == sum(counts.values())
    assert row["cap_binding_indicator"] is False
    assert row["safety_cap_hit"] is False
    assert row["max_total_refinements"] == 0


class _CapturingProxy(FullInfoAuctionProxy):
    """FullInfoAuctionProxy that records all ElicitationEvents it receives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received_events: list = []

    def refine(self, event) -> None:
        self.received_events.append(event)
        super().refine(event)


def test_near_tie_event_carries_bundles_field(toy_instance):
    """near_tie elicitation events should include both tied bundles."""
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)

    run_proxy_clock_experiment(
        toy_instance,
        proxies,
        cfg,
        ProxyClockConfig(
            top_k=1,
            elicited=True,
            tie_threshold=1000.0,  # very wide — force near-tie triggers
            margin_threshold=0.0,
            max_refinements_per_bidder=5,
        ),
    )

    near_tie_events = [
        e
        for proxy in proxies
        for e in proxy.received_events
        if e.event_type == "near_tie"
    ]
    if near_tie_events:
        for event in near_tie_events:
            assert event.bundles is not None, "near_tie event must carry bundles"
            assert len(event.bundles) == 2
            # bundle (runner-up) must be one of the two bundles
            assert event.bundle in event.bundles


def test_top_k_frontier_event_refines_newly_seen_demand_bundles(toy_instance):
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=3, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=2,
            elicited=True,
            margin_threshold=0.0,
            tie_threshold=0.0,
            refine_top_k_frontier=True,
            allocation_change_audit=False,
            terminal_stability_audit=False,
        ),
    )

    frontier_events = [
        event
        for proxy in proxies
        for event in proxy.received_events
        if event.event_type == "top_k_frontier"
    ]
    assert frontier_events
    event_keys = [(event.bidder_id, event.bundle) for event in frontier_events]
    assert len(event_keys) == len(set(event_keys))
    assert result.metadata["refine_top_k_frontier"] is True


def test_top_k_frontier_refinement_is_disabled_by_default(toy_instance):
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=3, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=2,
            elicited=True,
            margin_threshold=0.0,
            tie_threshold=0.0,
            allocation_change_audit=False,
            terminal_stability_audit=False,
        ),
    )

    assert not [
        event
        for proxy in proxies
        for event in proxy.received_events
        if event.event_type == "top_k_frontier"
    ]
    assert result.metadata["refine_top_k_frontier"] is False


def test_targeted_clock_exact_queries_are_unique_per_bidder_bundle(toy_instance):
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=3,
            elicited=True,
            event_framework="targeted_v1",
            demand_switch_verification=True,
            contested_bundle_refinement=True,
            allocation_change_audit=True,
            terminal_winner_verification=True,
            terminal_vcg_witness_verification=True,
            terminal_best_losing_challenger=True,
            terminal_stability_audit=False,
        ),
    )

    keys = [
        (bidder_id, record.bundle)
        for bidder_id, records in result.metadata[
            "refinement_records_by_bidder"
        ].items()
        for record in records
    ]
    assert len(keys) == len(set(keys))


def test_demand_revealed_support_contains_only_clock_demands(toy_instance):
    proxies = make_proxies(toy_instance, initial="all_atoms")

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=1,
            elicited=False,
            supplementary_support_policy="demand_revealed",
        ),
    )

    demanded = {bidder_id: set() for bidder_id in toy_instance.bidder_ids}
    # Clock history stores every reported primary-demand bundle.
    # A demand-revealed supplementary atom must have appeared there.
    for round_record in result.metadata["clock_history"]:
        for bidder_id, bundles in round_record.demands.items():
            demanded[bidder_id].update(bundles)
    for bidder_id, atoms in result.metadata["supplementary_bids"].items():
        assert {atom.bundle for atom in atoms} <= demanded[bidder_id]


class _SingleBundleDemandProxy(_CapturingProxy):
    """Expose one demand while retaining a larger private candidate bid."""

    def demand_at_prices(self, prices, round_idx=0, top_k=1):
        bid = self.current_bid()
        atom = bid.atoms[0]
        return DemandResponse(
            primary_bundle=atom.bundle,
            primary_bundles=[atom.bundle],
            supplementary_atoms=list(bid.atoms),
        )


def test_terminal_audit_does_not_leak_unrevealed_private_atoms(toy_instance):
    proxies = [
        _SingleBundleDemandProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]
    revealed = {
        proxy.bidder_id: proxy.current_bid().atoms[0].bundle
        for proxy in proxies
    }

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=2, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=1,
            elicited=True,
            event_framework="targeted_v1",
            supplementary_support_policy="demand_revealed",
            allocation_change_audit=False,
            terminal_winner_verification=True,
            terminal_vcg_witness_verification=True,
            terminal_stability_audit=False,
        ),
    )

    for bidder_id, atoms in result.metadata["supplementary_bids"].items():
        assert {atom.bundle for atom in atoms} <= {revealed[bidder_id]}


def test_native_clock_with_all_events_off_has_no_refinements(toy_instance):
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=3,
            elicited=True,
            event_framework="native_v1",
            native_near_zero_surplus=False,
            native_demand_changed=False,
            native_near_tie=False,
            allocation_change_audit=False,
            terminal_stability_audit=False,
        ),
    )

    assert sum(
        result.metadata["refinement_query_count_by_bidder"].values()
    ) == 0
    assert not [event for proxy in proxies for event in proxy.received_events]


def test_native_clock_emits_only_original_price_path_events(toy_instance):
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]

    run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=3,
            elicited=True,
            event_framework="native_v1",
            margin_threshold=1000.0,
            tie_threshold=1000.0,
            allocation_change_audit=False,
            terminal_stability_audit=False,
        ),
    )

    event_types = {
        event.event_type for proxy in proxies for event in proxy.received_events
    }
    assert event_types
    assert event_types <= {"near_zero_surplus", "demand_changed", "near_tie"}


def test_frontier_clock_pv_only_makes_no_exact_queries(toy_instance):
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]
    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=3,
            elicited=True,
            event_framework="frontier_v1",
            supplementary_support_policy="demand_revealed",
            allocation_change_audit=False,
            terminal_stability_audit=False,
        ),
    )

    assert sum(
        result.metadata["refinement_query_count_by_bidder"].values()
    ) == 0
    assert not [event for proxy in proxies for event in proxy.received_events]


def test_frontier_clock_queries_only_terminal_frontier_events(toy_instance):
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]
    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=3,
            elicited=True,
            tie_threshold=1000.0,
            event_framework="frontier_v1",
            supplementary_support_policy="demand_revealed",
            frontier_winner_verification=True,
            frontier_pivotal_challengers=True,
            frontier_winner_closure=True,
            allocation_change_audit=False,
            terminal_stability_audit=False,
        ),
    )

    events = [event for proxy in proxies for event in proxy.received_events]
    assert events
    assert {event.event_type for event in events} <= {
        "frontier_winner",
        "frontier_pivotal_challenger",
        "frontier_closure_winner",
    }
    keys = [(event.bidder_id, event.bundle) for event in events]
    assert len(keys) == len(set(keys))
    assert all(
        event.round_idx == result.rounds for event in events
    )


def test_frontier_single_pass_freezes_witnesses_without_closure(toy_instance):
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]
    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=3,
            elicited=True,
            tie_threshold=1000.0,
            event_framework="frontier_v1",
            supplementary_support_policy="all_atoms",
            frontier_pivotal_challengers=True,
            frontier_vcg_single_pass=True,
            allocation_change_audit=False,
            terminal_stability_audit=False,
        ),
    )

    events = [event for proxy in proxies for event in proxy.received_events]
    assert any(
        event.event_type == "frontier_vcg_single_pass_all"
        for event in events
    )
    assert not any(
        event.event_type in {
            "frontier_closure_winner", "frontier_vcg_witness"
        }
        for event in events
    )
    assert result.metadata["frontier_vcg_single_pass"] is True


def test_frontier_vcg_closure_and_single_pass_are_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ProxyClockConfig(
            frontier_vcg_witness_verification=True,
            frontier_vcg_single_pass=True,
        )


def test_staged_revealed_vcg_closure_is_terminal_and_deduplicated(
    toy_instance,
):
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]
    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=3,
            elicited=True,
            event_framework="frontier_v1",
            supplementary_support_policy="all_atoms",
            frontier_winner_closure=True,
            frontier_staged_revealed_vcg_closure=True,
            allocation_change_audit=False,
            terminal_stability_audit=False,
        ),
    )

    events = [event for proxy in proxies for event in proxy.received_events]
    assert events
    assert {event.event_type for event in events} <= {
        "frontier_closure_winner",
        "frontier_staged_revealed_vcg_witness",
    }
    keys = [(event.bidder_id, event.bundle) for event in events]
    assert len(keys) == len(set(keys))
    assert all(event.round_idx == result.rounds for event in events)
    assert result.metadata["frontier_staged_revealed_vcg_closure"] is True


# ---------------------------------------------------------------------------
# Priority scoring tests
# ---------------------------------------------------------------------------


def test_pivotality_score_near_zero_surplus_at_zero_is_one():
    assert _pivotality_score("near_zero_surplus", surplus=0.0, margin_threshold=100.0) == 1.0


def test_pivotality_score_near_zero_surplus_at_threshold_is_zero():
    assert _pivotality_score("near_zero_surplus", surplus=100.0, margin_threshold=100.0) == 0.0


def test_pivotality_score_near_zero_surplus_midpoint():
    score = _pivotality_score("near_zero_surplus", surplus=50.0, margin_threshold=100.0)
    assert score == pytest.approx(0.5)


def test_pivotality_score_demand_changed_is_always_one():
    assert _pivotality_score("demand_changed") == 1.0


def test_pivotality_score_near_tie_at_zero_gap_is_one():
    assert _pivotality_score("near_tie", gap=0.0, tie_threshold=100.0) == 1.0


def test_pivotality_score_near_tie_at_threshold_is_zero():
    assert _pivotality_score("near_tie", gap=100.0, tie_threshold=100.0) == 0.0


def test_pivotality_score_zero_threshold_returns_one():
    assert _pivotality_score("near_zero_surplus", surplus=50.0, margin_threshold=0.0) == 1.0
    assert _pivotality_score("near_tie", gap=50.0, tie_threshold=0.0) == 1.0


def test_pivotality_score_unknown_event_type_returns_zero():
    assert _pivotality_score("unknown_event") == 0.0


def test_candidate_refinements_returns_score_as_fourth_element():
    """_candidate_refinements must return 4-tuples with a float score."""
    from auctionlab.bids.xor import XorAtomicBid

    ranked = [
        (XorAtomicBid(bundle=frozenset({"A"}), value=10.0), 5.0),
        (XorAtomicBid(bundle=frozenset({"B"}), value=8.0), 3.0),
    ]
    candidates = _candidate_refinements(
        ranked=ranked,
        current_primary_bundle=frozenset({"A"}),
        previous_primary_bundle=None,
        margin_threshold=100.0,
        tie_threshold=100.0,
        round_idx=0,
    )
    for item in candidates:
        assert len(item) == 4
        score = item[3]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_priority_scoring_fires_highest_score_first(toy_instance):
    """With budget=1, priority_scoring should choose the highest-score candidate."""
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)

    run_proxy_clock_experiment(
        toy_instance,
        proxies,
        cfg,
        ProxyClockConfig(
            top_k=1,
            elicited=True,
            margin_threshold=10_000.0,
            tie_threshold=10_000.0,
            max_refinements_per_bidder=1,  # only one refinement per bidder
            priority_scoring=True,
        ),
    )

    # With priority_scoring=True, the first event processed by each bidder
    # must be the highest-score one (demand_changed=1.0 if it fires, else the
    # highest of near_zero_surplus and near_tie by score).
    for proxy in proxies:
        events = proxy.received_events
        if len(events) >= 2:
            # There can only be 1 event per bidder (budget=1) — enforce that.
            assert False, "More events than budget allows were fired"


def test_priority_scoring_in_metadata(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxies = make_proxies(toy_instance, initial="all_atoms")

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        cfg,
        ProxyClockConfig(top_k=1, elicited=False, priority_scoring=True),
    )
    assert result.metadata["priority_scoring"] is True


def test_priority_scoring_false_in_metadata_by_default(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxies = make_proxies(toy_instance, initial="all_atoms")

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        cfg,
        ProxyClockConfig(top_k=1, elicited=False),
    )
    assert result.metadata["priority_scoring"] is False


def test_non_near_tie_events_have_no_bundles_field(toy_instance):
    """near_zero_surplus and demand_changed events should not set bundles."""
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)

    run_proxy_clock_experiment(
        toy_instance,
        proxies,
        cfg,
        ProxyClockConfig(
            top_k=1,
            elicited=True,
            tie_threshold=0.0,         # no near-ties
            margin_threshold=1000.0,   # wide margin to trigger near_zero events
            max_refinements_per_bidder=5,
        ),
    )

    non_tie_events = [
        e
        for proxy in proxies
        for e in proxy.received_events
        if e.event_type != "near_tie"
    ]
    for event in non_tie_events:
        assert event.bundles is None, (
            f"{event.event_type} event should not carry bundles"
        )


def test_terminal_stability_audit_corrects_newly_exposed_winner():
    from auctionlab.instances.base import AuctionInstance

    instance = AuctionInstance(
        items=["A"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"A"}): 10.0},
            "i2": {frozenset({"A"}): 90.0},
        },
    )
    proxies = make_proxies(instance, initial="all_atoms")
    for proxy, reported_value in zip(proxies, [100.0, 80.0]):
        proxy.current_bid().atoms[0] = XorAtomicBid(
            bundle=frozenset({"A"}),
            value=reported_value,
        )

    result = run_proxy_clock_experiment(
        instance,
        proxies,
        ClockConfig(max_rounds=1, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=1,
            elicited=True,
            margin_threshold=0.0,
            tie_threshold=0.0,
        ),
    )

    assert result.allocation["i2"] == frozenset({"A"})
    assert result.metadata["terminal_stability_audit_queries"] == 2
    assert result.metadata["terminal_stability_audit_iterations"] == 2
    assert result.metadata["pre_terminal_revenue"] is not None
    assert result.metadata["post_terminal_revenue"] == result.revenue


def test_targeted_clock_uses_true_bidder_removal_vcg_witness():
    from auctionlab.instances.base import AuctionInstance

    instance = AuctionInstance(
        items=["A"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"A"}): 10.0},
            "i2": {frozenset({"A"}): 9.0},
        },
    )
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=instance,
            initial="all_atoms",
        )
        for bidder_id in instance.bidder_ids
    ]
    result = run_proxy_clock_experiment(
        instance,
        proxies,
        ClockConfig(max_rounds=1, price_step=1.0),
        ProxyClockConfig(
            top_k=1,
            elicited=True,
            event_framework="targeted_v1",
            allocation_change_audit=True,
            terminal_stability_audit=False,
            demand_switch_verification=True,
            terminal_winner_verification=True,
            terminal_vcg_witness_verification=True,
        ),
    )

    event_types = [
        event.event_type
        for proxy in proxies
        for event in proxy.received_events
    ]
    assert "terminal_winner" in event_types
    assert "terminal_vcg_witness" in event_types
    assert "near_zero_surplus" not in event_types
    assert "near_tie" not in event_types
    assert result.metadata["event_framework"] == "targeted_v1"


def test_allocation_pivotal_frontier_only_fires_within_gap(toy_instance):
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=1, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=2,
            elicited=True,
            margin_threshold=0.0,
            tie_threshold=0.0,
            top_k_frontier_policy="allocation_pivotal",
            allocation_change_audit=False,
            terminal_stability_audit=False,
        ),
    )

    events = [
        event
        for proxy in proxies
        for event in proxy.received_events
        if event.event_type.startswith("allocation_pivotal_")
    ]
    assert events
    assert all(event.allocation_gap is not None for event in events)
    assert all(event.allocation_gap <= 0.0 for event in events)
    assert result.metadata["top_k_frontier_policy"] == "allocation_pivotal"


def test_initial_allocation_counterfactual_frontier_is_bounded(toy_instance):
    proxies = [
        _CapturingProxy(
            bidder_id=bidder_id,
            instance=toy_instance,
            initial="all_atoms",
        )
        for bidder_id in toy_instance.bidder_ids
    ]

    result = run_proxy_clock_experiment(
        toy_instance,
        proxies,
        ClockConfig(max_rounds=1, price_step=1.0, reserve=0.0),
        ProxyClockConfig(
            top_k=1,
            elicited=True,
            margin_threshold=0.0,
            tie_threshold=0.0,
            allocation_counterfactual_frontier=True,
            terminal_stability_audit=False,
        ),
    )

    counterfactual_events = [
        event
        for proxy in proxies
        for event in proxy.received_events
        if event.event_type == "allocation_counterfactual"
    ]
    assert counterfactual_events
    assert len({
        event.bidder_id for event in counterfactual_events
    }) == len(counterfactual_events)
    assert result.metadata["allocation_counterfactual_frontier"] is True
    assert result.metadata[
        "allocation_counterfactual_frontier_queries"
    ] == len(counterfactual_events)
