from __future__ import annotations

from auctionlab.auctions.clock import ClockConfig
from auctionlab.bids.xor import XorAtomicBid
from auctionlab.experiments.proxy_clock_runner import ProxyClockConfig, run_proxy_clock_experiment
from auctionlab.experiments.proxy_sealed_runner import (
    ProxySealedConfig,
    run_proxy_sealed_vcg_experiment,
)
from auctionlab.experiments.runner import MechanismResult
from auctionlab.instances.base import AuctionInstance
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.proxies.base import (
    ClockAuctionProxy,
    ElicitationEvent,
    SealedAuctionProxy,
)
from auctionlab.proxies.baselines.dnf_learning import DnfLearningProxy


ITEM_DESCRIPTIONS = {"A": "Item A", "B": "Item B"}


def make_proxy(responses: list[str]) -> DnfLearningProxy:
    person = LlmPersonSimulator(
        bidder_id="i1",
        scenario_description="A test auction.",
        person_seed="Values A highly; B is redundant given A.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=MockLlmClient(responses),
    )
    return DnfLearningProxy(bidder_id="i1", person=person, items=["A", "B"])


def test_satisfies_proxy_protocols():
    proxy = make_proxy([])

    assert isinstance(proxy, ClockAuctionProxy)
    assert isinstance(proxy, SealedAuctionProxy)


def test_initial_bid_is_empty_and_grand_bundle_atoms():
    proxy = make_proxy([])

    bid = proxy.current_bid()

    assert bid.atoms == [
        XorAtomicBid(bundle=frozenset(), value=0.0),
        XorAtomicBid(bundle=frozenset({"A", "B"}), value=0.0),
    ]
    assert proxy.stats().value_queries == 0


def test_refine_satisfied_updates_bid_to_cost():
    # satisfied=True is a revealed lower bound: value(bundle) >= cost.
    # The proxy should raise the atom to the bundle cost when cost > current bid.
    proxy = make_proxy(['{"satisfied": true, "preferred_bundle": null}'])

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
            prices={"A": 1.0, "B": 1.0},
        )
    )

    assert proxy.current_bid().atoms == [
        XorAtomicBid(bundle=frozenset(), value=0.0),
        XorAtomicBid(bundle=frozenset({"A", "B"}), value=2.0),
    ]
    records = proxy.refinement_records()
    assert len(records) == 1
    assert records[0].bundle == frozenset({"A", "B"})
    assert records[0].old_value is None
    assert records[0].new_value == 2.0
    assert proxy.stats().refinement_queries == 1


def test_refine_satisfied_at_zero_price_leaves_bid_unchanged():
    # satisfied=True at zero price: cost == 0, so no lower-bound update is possible.
    proxy = make_proxy(['{"satisfied": true, "preferred_bundle": null}'])

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
            prices={"A": 0.0, "B": 0.0},
        )
    )

    assert proxy.current_bid().atoms == [
        XorAtomicBid(bundle=frozenset(), value=0.0),
        XorAtomicBid(bundle=frozenset({"A", "B"}), value=0.0),
    ]
    assert proxy.refinement_records() == []
    assert proxy.stats().refinement_queries == 1


def test_refine_unsatisfied_learns_minimal_atomic_bundle():
    proxy = make_proxy(
        [
            '{"satisfied": false, "preferred_bundle": null}',
            '{"bundle_value": 10}',  # value_query({A, B})
            '{"bundle_value": 5}',   # value_query({B}) -- removing A changes value
            '{"bundle_value": 10}',  # value_query({A}) -- removing B keeps value
        ]
    )

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
            prices={"A": 1.0, "B": 1.0},
        )
    )

    bid = proxy.current_bid()
    assert XorAtomicBid(bundle=frozenset({"A"}), value=10.0) in bid.atoms
    assert proxy.stats().value_queries == 3
    records = proxy.refinement_records()
    assert len(records) == 1
    assert records[0].bundle == frozenset({"A"})
    assert records[0].old_value is None
    assert records[0].new_value == 10.0


def test_demand_at_prices_ranks_atoms_by_surplus():
    proxy = make_proxy(
        [
            '{"satisfied": false, "preferred_bundle": null}',
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"bundle_value": 10}',
        ]
    )
    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
            prices={"A": 1.0, "B": 1.0},
        )
    )

    response = proxy.demand_at_prices({"A": 1.0, "B": 1.0}, round_idx=0, top_k=1)

    assert response.primary_bundle == frozenset({"A"})
    # demand_queries = 1 elicitation demand check (in refine) + 1 clock demand
    assert proxy.stats().demand_queries == 2


def test_receive_provisional_feedback_forwards_to_refine():
    proxy = make_proxy(['{"satisfied": true, "preferred_bundle": null}'])

    proxy.receive_provisional_feedback(
        ElicitationEvent(
            mechanism="test",
            event_type="allocated_bundle",
            bidder_id="i1",
            bundle=frozenset({"A"}),
            prices={"A": 1.0, "B": 1.0},
            allocated_bundle=frozenset({"A"}),
        )
    )

    assert proxy.stats().refinement_queries == 1


def test_refine_ignores_event_without_bundle():
    proxy = make_proxy([])

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="happy",
            bidder_id="i1",
            bundle=None,
        )
    )

    assert proxy.stats().refinement_queries == 0


# ---------------------------------------------------------------------------
# Fix 1: event.bundles multi-bundle support
# ---------------------------------------------------------------------------


def test_refine_processes_all_bundles_in_event_bundles():
    """When event.bundles is set, every non-empty bundle should be refined."""
    proxy = make_proxy(
        [
            # bundle A: unsatisfied → learn {A} value
            '{"satisfied": false, "preferred_bundle": null}',
            '{"bundle_value": 8}',   # value_query({A})
            # bundle B: unsatisfied → learn {B} value
            '{"satisfied": false, "preferred_bundle": null}',
            '{"bundle_value": 6}',   # value_query({B})
        ]
    )

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_tie",
            bidder_id="i1",
            bundle=frozenset({"B"}),
            bundles=(frozenset({"A"}), frozenset({"B"})),
            prices={"A": 1.0, "B": 1.0},
        )
    )

    bundles_learned = {atom.bundle for atom in proxy.current_bid().atoms}
    assert frozenset({"A"}) in bundles_learned
    assert frozenset({"B"}) in bundles_learned
    assert proxy.stats().refinement_queries == 2


def test_refine_single_bundle_event_unchanged():
    """Without event.bundles, behaviour is identical to the original."""
    proxy = make_proxy(
        [
            '{"satisfied": false, "preferred_bundle": null}',
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"bundle_value": 10}',
        ]
    )

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
            prices={"A": 1.0, "B": 1.0},
        )
    )

    assert proxy.stats().refinement_queries == 1
    assert XorAtomicBid(bundle=frozenset({"A"}), value=10.0) in proxy.current_bid().atoms


# ---------------------------------------------------------------------------
# Fix 2: floating-point tolerance in _learn_atomic_bundle
# ---------------------------------------------------------------------------


def test_learn_atomic_bundle_removes_item_when_value_within_tolerance():
    """A near-equal candidate value (within tolerance) should allow shrinking."""
    proxy = make_proxy(
        [
            '{"bundle_value": 10.0}',         # value_query({A, B})
            '{"bundle_value": 9.9999999999}',  # value_query({B}) ≈ 10 within 1e-9
        ]
    )
    proxy.tolerance = 1e-6  # widen so the near-equal value is accepted

    learned_bundle, value = proxy._learn_atomic_bundle(frozenset({"A", "B"}))

    # B alone has ≈ same value as {A,B}, so A should be removed
    assert learned_bundle == frozenset({"B"})


def test_learn_atomic_bundle_keeps_item_when_value_outside_tolerance():
    """A value difference exceeding tolerance must prevent shrinking."""
    proxy = make_proxy(
        [
            '{"bundle_value": 10.0}',
            '{"bundle_value": 9.0}',   # value_query({B}): differs by 1.0 > 1e-9
            '{"bundle_value": 10.0}',  # value_query({A}): same value, A can be shrunk out
        ]
    )

    learned_bundle, value = proxy._learn_atomic_bundle(frozenset({"A", "B"}))

    # B alone differs by 1.0; A alone equals bundle value; result: {A}
    assert learned_bundle == frozenset({"A"})
    assert value == 10.0


# ---------------------------------------------------------------------------
# Fix 3: elicitation demand queries counted in demand_queries
# ---------------------------------------------------------------------------


def test_refine_demand_check_increments_demand_queries():
    """The satisfaction demand query inside refine should count in demand_queries."""
    proxy = make_proxy(['{"satisfied": true, "preferred_bundle": null}'])

    assert proxy.stats().demand_queries == 0
    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="i1",
            bundle=frozenset({"A"}),
            prices={"A": 1.0, "B": 1.0},
        )
    )

    assert proxy.stats().demand_queries == 1


def test_refine_without_prices_does_not_increment_demand_queries():
    """Sealed-bid events (no prices) should not issue or count demand queries."""
    proxy = make_proxy(
        [
            '{"bundle_value": 10}',   # value_query({A, B})
            '{"bundle_value": 10}',   # value_query({B}) — same value, shrink
        ]
    )

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="allocated_bundle",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
        )
    )

    assert proxy.stats().demand_queries == 0
    assert proxy.stats().value_queries == 2


def test_demand_queries_separates_clock_and_elicitation():
    """clock demand (demand_at_prices) and elicitation demand (refine) both count."""
    proxy = make_proxy(
        [
            '{"satisfied": false, "preferred_bundle": null}',
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"bundle_value": 10}',
        ]
    )

    proxy.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
            prices={"A": 1.0, "B": 1.0},
        )
    )
    proxy.demand_at_prices({"A": 1.0, "B": 1.0})

    assert proxy.stats().demand_queries == 2  # 1 elicitation + 1 clock


def make_instance() -> AuctionInstance:
    return AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1"],
        valuations={"i1": {frozenset({"A", "B"}): 10.0}},
    )


def test_runs_as_static_proxy_in_proxy_clock_experiment():
    proxy = make_proxy([])

    result = run_proxy_clock_experiment(
        make_instance(),
        [proxy],
        ClockConfig(),
        ProxyClockConfig(elicited=False),
    )

    assert isinstance(result, MechanismResult)
    assert result.mechanism.startswith("proxy_clock_vcg_static")
    assert proxy.stats().value_queries == 0
    assert proxy.stats().refinement_queries == 0


def test_runs_as_static_proxy_in_proxy_sealed_experiment():
    proxy = make_proxy([])

    result = run_proxy_sealed_vcg_experiment(
        make_instance(),
        [proxy],
        ProxySealedConfig(),
    )

    assert isinstance(result, MechanismResult)
    assert result.mechanism == "proxy_sealed_vcg_static"
    assert proxy.stats().refinement_queries == 0
