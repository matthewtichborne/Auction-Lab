from __future__ import annotations

from auctionlab.bids.xor import XorAtomicBid
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmAuctionProxyAdapter, LlmInferredXorProxy
from auctionlab.proxies.base import ElicitationEvent


ITEM_DESCRIPTIONS = {"A": "Item A", "B": "Item B"}


def make_adapter(
    responses: list[str],
    *,
    epsilon: float = 1.0,
    refinement_strategy: str = "value_query",
    candidate_bundles: list | None = None,
) -> LlmAuctionProxyAdapter:
    client = MockLlmClient(responses)
    person = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A test auction.",
        person_seed="Values useful combinations.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
    )
    proxy = LlmInferredXorProxy(bidder_id="bidder_1", person=person, epsilon=epsilon)
    return LlmAuctionProxyAdapter(
        bidder_id="bidder_1",
        proxy=proxy,
        candidate_bundles=(
            candidate_bundles
            if candidate_bundles is not None
            else [frozenset({"A"}), frozenset({"B"})]
        ),
        refinement_strategy=refinement_strategy,
    )


def test_submit_bid_infers_initial_bid_and_records_value_queries():
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
        ]
    )

    bid = adapter.submit_bid()

    assert sorted(bid.atoms, key=lambda a: sorted(a.bundle)) == [
        XorAtomicBid(bundle=frozenset({"A"}), value=10.0),
        XorAtomicBid(bundle=frozenset({"B"}), value=5.0),
    ]
    assert adapter.stats().value_queries == 2
    assert len(adapter.proxy.person.client.calls) == 2


def test_current_bid_is_cached_across_calls():
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
        ]
    )

    first = adapter.current_bid()
    second = adapter.current_bid()

    assert first is second
    assert len(adapter.proxy.person.client.calls) == 2


def test_demand_at_prices_uses_current_bid_and_counts_demand_queries():
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
        ]
    )

    response = adapter.demand_at_prices({"A": 1.0, "B": 1.0}, round_idx=0, top_k=1)

    assert response.primary_bundle == frozenset({"A"})
    assert adapter.stats().demand_queries == 1
    # Demand is answered from the cached bid; no extra LLM calls.
    assert len(adapter.proxy.person.client.calls) == 2


def test_refine_sends_value_query_and_updates_bid_and_stats():
    # {A, B} is not in the initial candidate set → refinement triggers a real LLM call.
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"bundle_value": 8}',
        ],
        epsilon=0.5,
    )

    adapter.current_bid()
    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="bidder_1",
            bundle=frozenset({"A", "B"}),
        )
    )

    bid = adapter.current_bid()
    refined = next(atom for atom in bid.atoms if atom.bundle == frozenset({"A", "B"}))
    # epsilon=0.5 → raw 8.0 discounted to 4.0, but {A}:5.0 (stored) is a
    # subset of {A,B} and 5 > 4, so the superset atom is raised to 5.0.
    assert refined.value == 5.0
    assert adapter.stats().refinement_queries == 1
    assert len(adapter.proxy.person.client.calls) == 3


def test_refine_records_old_and_new_value():
    # {A, B} is a new bundle — not in initial set, so old_value is None.
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"bundle_value": 8}',
        ],
        epsilon=0.5,
    )

    adapter.current_bid()
    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="bidder_1",
            bundle=frozenset({"A", "B"}),
            reason="near_zero_surplus",
            round_idx=2,
        )
    )

    records = adapter.refinement_records()
    assert len(records) == 1
    record = records[0]
    assert record.bidder_id == "bidder_1"
    assert record.mechanism == "test"
    assert record.event_type == "near_zero_surplus"
    assert record.round_idx == 2
    assert record.bundle == frozenset({"A", "B"})
    assert record.old_value is None  # not in initial bid
    # epsilon=0.5 → raw 8.0 discounted to 4.0, but {A}:5.0 (stored) is a
    # subset of {A,B}, so the superset atom is raised to 5.0.
    assert record.new_value == 5.0
    assert record.reason == "near_zero_surplus"

    # current_bid() and submit_bid() both reflect the raised value.
    bid = adapter.current_bid()
    refined = next(atom for atom in bid.atoms if atom.bundle == frozenset({"A", "B"}))
    assert refined.value == 5.0
    submitted = adapter.submit_bid()
    submitted_refined = next(
        atom for atom in submitted.atoms if atom.bundle == frozenset({"A", "B"})
    )
    assert submitted_refined.value == 5.0


def test_refine_known_bundle_is_noop():
    # Bundles already in the initial candidate set are not re-queried; the LLM
    # is not called and no refinement record is created.
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
        ],
    )

    adapter.current_bid()
    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="bidder_1",
            bundle=frozenset({"B"}),
        )
    )

    records = adapter.refinement_records()
    assert len(records) == 0
    assert adapter.stats().refinement_queries == 0
    assert len(adapter.proxy.person.client.calls) == 2  # no extra LLM call
    bid = adapter.current_bid()
    b_atom = next(atom for atom in bid.atoms if atom.bundle == frozenset({"B"}))
    assert b_atom.value == 5.0  # unchanged


def test_refine_ignores_event_without_bundle():
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
        ]
    )
    adapter.current_bid()

    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="happy",
            bidder_id="bidder_1",
            bundle=None,
        )
    )

    assert adapter.stats().refinement_queries == 0
    assert len(adapter.proxy.person.client.calls) == 2


def test_refine_demand_query_strategy_satisfied_skips_record():
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"satisfied": true, "preferred_bundle": null}',
        ],
        refinement_strategy="demand_query",
    )
    adapter.current_bid()

    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="bidder_1",
            bundle=frozenset({"B"}),
            prices={"A": 1.0, "B": 1.0},
        )
    )

    assert adapter.refinement_records() == []
    assert adapter.stats().refinement_queries == 1


def test_refine_demand_query_strategy_revalues_preferred_bundle():
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"satisfied": false, "preferred_bundle": ["A"]}',
            '{"bundle_value": 20}',
        ],
        refinement_strategy="demand_query",
    )
    adapter.current_bid()

    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="bidder_1",
            bundle=frozenset({"B"}),
            prices={"A": 1.0, "B": 1.0},
        )
    )

    records = adapter.refinement_records()
    assert len(records) == 1
    assert records[0].bundle == frozenset({"A"})
    assert records[0].old_value == 10.0
    assert records[0].new_value == 20.0

    bid = adapter.current_bid()
    refined = next(atom for atom in bid.atoms if atom.bundle == frozenset({"A"}))
    assert refined.value == 20.0


def test_refine_demand_query_strategy_without_prices_falls_back_to_value_query():
    # Without prices the demand-query path falls back to a value query.
    # Use a new bundle ({A, B}) so the fallback actually fires.
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"bundle_value": 8}',
        ],
        refinement_strategy="demand_query",
    )
    adapter.current_bid()

    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="allocated_bundle",
            bidder_id="bidder_1",
            bundle=frozenset({"A", "B"}),
        )
    )

    records = adapter.refinement_records()
    assert len(records) == 1
    assert records[0].bundle == frozenset({"A", "B"})
    # {A}:10 is a subset of {A,B}:8 → superset raised to 10 (free disposal).
    assert records[0].new_value == 10.0


def test_receive_provisional_feedback_forwards_to_refine():
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"bundle_value": 12}',
        ]
    )
    adapter.current_bid()

    adapter.receive_provisional_feedback(
        ElicitationEvent(
            mechanism="test",
            event_type="allocated_bundle",
            bidder_id="bidder_1",
            bundle=frozenset({"A", "B"}),  # new bundle — triggers real LLM call
        )
    )

    assert adapter.stats().refinement_queries == 1


def test_refine_populates_query_text_and_response_summary_in_record():
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"bundle_value": 8, "reasoning_summary": "reconsidered near-zero"}',
        ],
        epsilon=1.0,
    )
    adapter.current_bid()

    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="bidder_1",
            bundle=frozenset({"A", "B"}),  # new bundle
            reason="near-zero surplus at round 3",
        )
    )

    record = adapter.refinement_records()[0]
    assert record.query_text is not None
    assert "PROPOSED_BUNDLE_ITEM_IDS" in record.query_text
    assert record.response_summary == "reconsidered near-zero"


def test_refine_demand_query_populates_query_text_in_record():
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"satisfied": false, "preferred_bundle": null}',
            '{"bundle_value": 3, "reasoning_summary": "lower than price"}',
        ],
        epsilon=1.0,
        refinement_strategy="demand_query",
    )
    adapter.current_bid()

    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="bidder_1",
            bundle=frozenset({"B"}),
            prices={"A": 1.0, "B": 4.0},
            reason="near-zero surplus",
        )
    )

    record = adapter.refinement_records()[0]
    # Fallback to value_query when demand query has no alternative;
    # query_text should be the value query prompt.
    assert record.query_text is not None
    assert record.response_summary == "lower than price"


def test_refine_multi_bundle_event_refines_each_bundle():
    # Use candidate_bundles=[] so all bundles are new and both get refined.
    adapter = make_adapter(
        [
            '{"bundle_value": 12}',
            '{"bundle_value": 7}',
        ],
        epsilon=1.0,
        candidate_bundles=[],
    )
    adapter.current_bid()

    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_tie",
            bidder_id="bidder_1",
            bundle=frozenset({"B"}),
            bundles=(frozenset({"A"}), frozenset({"B"})),
            reason="top bundles nearly tied",
        )
    )

    records = adapter.refinement_records()
    assert len(records) == 2
    refined_bundles = {r.bundle for r in records}
    assert frozenset({"A"}) in refined_bundles
    assert frozenset({"B"}) in refined_bundles


def test_refine_multi_bundle_event_ignores_empty_bundles():
    # {A, B} is new; frozenset() is empty → only {A, B} gets refined.
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"bundle_value": 8}',
        ],
        epsilon=1.0,
    )
    adapter.current_bid()

    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_tie",
            bidder_id="bidder_1",
            bundle=frozenset({"A", "B"}),
            bundles=(frozenset({"A", "B"}), frozenset()),  # second bundle is empty
        )
    )

    records = adapter.refinement_records()
    assert len(records) == 1
    assert records[0].bundle == frozenset({"A", "B"})


def test_refine_single_bundle_still_works_without_bundles_field():
    # {A, B} is a new bundle; no bundles field → single-bundle path.
    adapter = make_adapter(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
            '{"bundle_value": 8}',
        ],
        epsilon=1.0,
    )
    adapter.current_bid()

    adapter.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="bidder_1",
            bundle=frozenset({"A", "B"}),
            # bundles not set — backward-compatible single-bundle path
        )
    )

    records = adapter.refinement_records()
    assert len(records) == 1
    assert records[0].bundle == frozenset({"A", "B"})
