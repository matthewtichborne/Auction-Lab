"""Tests for the proxy elicitation event API (ProxyElicitationEvent / ProxyResponse).

Covers:
  A. Event API basics: construction, unknown-type rejection.
  B. Backward compatibility: old methods still work after handle_event is added.
  C. submit_bid / SUBMIT_BID event produce the same XOR bid.
  D. Clock demand via old method and CLOCK_PRICES event match.
  E. Optional interest map: proxy works with interest_map=None.
  F. Initialisation events on LlmInferredXorProxy.handle_event.
"""

from __future__ import annotations

import pytest

from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmAuctionProxyAdapter, LlmInferredXorProxy
from auctionlab.proxies.base import ElicitationEvent
from auctionlab.proxies.events import (
    CLOCK_PRICES,
    GENERATE_CANDIDATE_BUNDLES,
    INFER_INTEREST_MAP,
    INFER_PROVISIONAL_VALUES,
    INITIAL_PREFERENCE_QUESTION,
    SEALED_FEEDBACK,
    SUBMIT_BID,
    ProxyElicitationEvent,
    ProxyEventLogEntry,
    ProxyResponse,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

ITEM_DESCRIPTIONS = {"A": "Item A", "B": "Item B"}


def make_adapter(
    responses: list[str],
    *,
    epsilon: float = 1.0,
    candidate_bundles: list | None = None,
) -> LlmAuctionProxyAdapter:
    client = MockLlmClient(responses)
    person = LlmPersonSimulator(
        bidder_id="b1",
        scenario_description="A test auction.",
        person_seed="Values useful combinations.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
    )
    proxy = LlmInferredXorProxy(bidder_id="b1", person=person, epsilon=epsilon)
    return LlmAuctionProxyAdapter(
        bidder_id="b1",
        proxy=proxy,
        candidate_bundles=(
            candidate_bundles if candidate_bundles is not None
            else [frozenset({"A"}), frozenset({"B"})]
        ),
    )


def make_inner_proxy(responses: list[str]) -> LlmInferredXorProxy:
    client = MockLlmClient(responses)
    person = LlmPersonSimulator(
        bidder_id="b1",
        scenario_description="A test auction.",
        person_seed="Values useful combinations.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
    )
    return LlmInferredXorProxy(bidder_id="b1", person=person)


def price_fn(d: dict):
    return lambda b: d.get(frozenset(b) if not isinstance(b, frozenset) else b, 0.0)


# ---------------------------------------------------------------------------
# A. Event API basics
# ---------------------------------------------------------------------------

def test_proxy_elicitation_event_constructs():
    event = ProxyElicitationEvent(event_type=SUBMIT_BID, bidder_id="b1")
    assert event.event_type == SUBMIT_BID
    assert event.bidder_id == "b1"
    assert event.mechanism is None
    assert event.round_idx is None
    assert event.payload == {}


def test_proxy_elicitation_event_with_payload():
    event = ProxyElicitationEvent(
        event_type=CLOCK_PRICES,
        bidder_id="b1",
        mechanism="clock",
        round_idx=3,
        payload={"prices": {"A": 5.0}, "top_k": 2},
    )
    assert event.mechanism == "clock"
    assert event.round_idx == 3
    assert event.payload["top_k"] == 2


def test_proxy_response_constructs():
    resp = ProxyResponse(response_type="xor_bid", payload={"bid": None})
    assert resp.response_type == "xor_bid"
    assert resp.records == []
    assert resp.state_delta == {}


def test_proxy_event_log_entry_constructs():
    entry = ProxyEventLogEntry(
        event_type=SUBMIT_BID,
        bidder_id="b1",
        mechanism=None,
        round_idx=None,
        response_type="xor_bid",
        records_count=0,
    )
    assert entry.event_type == SUBMIT_BID
    assert entry.records_count == 0


def test_handle_event_unknown_type_raises():
    adapter = make_adapter(['{"bundle_value": 10}', '{"bundle_value": 5}'])
    with pytest.raises(ValueError, match="Unknown proxy event type"):
        adapter.handle_event(ProxyElicitationEvent(event_type="nonsense", bidder_id="b1"))


def test_handle_event_unknown_type_raises_on_inner_proxy():
    proxy = make_inner_proxy(['{"bundle_value": 10}'])
    with pytest.raises(ValueError, match="Unknown proxy event type"):
        proxy.handle_event(ProxyElicitationEvent(event_type="bad_event", bidder_id="b1"))


def test_inner_proxy_rejects_mechanism_events():
    proxy = make_inner_proxy(['{"bundle_value": 10}'])
    with pytest.raises(ValueError, match="mechanism event"):
        proxy.handle_event(ProxyElicitationEvent(event_type=SUBMIT_BID, bidder_id="b1"))


# ---------------------------------------------------------------------------
# B. Backward compatibility: old methods unaffected
# ---------------------------------------------------------------------------

def test_old_submit_bid_still_works():
    adapter = make_adapter(['{"bundle_value": 10}', '{"bundle_value": 5}'])
    bid = adapter.submit_bid()
    assert len(bid.atoms) == 2
    assert adapter.stats().value_queries == 2


def test_old_demand_at_prices_still_works():
    adapter = make_adapter(['{"bundle_value": 10}', '{"bundle_value": 5}'])
    resp = adapter.demand_at_prices({"A": 1.0, "B": 1.0}, round_idx=0, top_k=1)
    assert resp.primary_bundle == frozenset({"A"})


def test_old_refine_still_works():
    adapter = make_adapter([
        '{"bundle_value": 10}',
        '{"bundle_value": 5}',
        '{"bundle_value": 8}',
    ])
    adapter.current_bid()
    adapter.refine(ElicitationEvent(
        mechanism="test", event_type="near_zero_surplus",
        bidder_id="b1", bundle=frozenset({"A", "B"}),
    ))
    assert adapter.stats().refinement_queries == 1


# ---------------------------------------------------------------------------
# C. submit_bid via old method and SUBMIT_BID event produce the same bid
# ---------------------------------------------------------------------------

def test_submit_bid_event_matches_old_method():
    # Two independent adapters with identical mock responses.
    adapter_old = make_adapter(['{"bundle_value": 10}', '{"bundle_value": 5}'])
    adapter_new = make_adapter(['{"bundle_value": 10}', '{"bundle_value": 5}'])

    bid_old = adapter_old.submit_bid()
    resp = adapter_new.handle_event(ProxyElicitationEvent(
        event_type=SUBMIT_BID, bidder_id="b1", mechanism="sealed",
    ))
    bid_new = resp.payload["bid"]

    assert resp.response_type == "xor_bid"
    assert sorted(bid_old.atoms, key=lambda a: sorted(a.bundle)) == \
           sorted(bid_new.atoms, key=lambda a: sorted(a.bundle))


def test_submit_bid_event_logs_to_event_log():
    adapter = make_adapter(['{"bundle_value": 10}', '{"bundle_value": 5}'])
    assert len(adapter.proxy.event_log) == 0
    adapter.handle_event(ProxyElicitationEvent(event_type=SUBMIT_BID, bidder_id="b1"))
    assert len(adapter.proxy.event_log) == 1
    assert adapter.proxy.event_log[0].event_type == SUBMIT_BID
    assert adapter.proxy.event_log[0].response_type == "xor_bid"


# ---------------------------------------------------------------------------
# D. Clock demand via old method and CLOCK_PRICES event match
# ---------------------------------------------------------------------------

def test_clock_prices_event_matches_old_demand_at_prices():
    adapter_old = make_adapter(['{"bundle_value": 10}', '{"bundle_value": 5}'])
    adapter_new = make_adapter(['{"bundle_value": 10}', '{"bundle_value": 5}'])

    prices = {"A": 1.0, "B": 1.0}
    resp_old = adapter_old.demand_at_prices(prices, round_idx=0, top_k=1)
    resp_ev = adapter_new.handle_event(ProxyElicitationEvent(
        event_type=CLOCK_PRICES,
        bidder_id="b1",
        mechanism="clock",
        round_idx=0,
        payload={"prices": prices, "top_k": 1},
    ))
    resp_new = resp_ev.payload["demand_response"]

    assert resp_ev.response_type == "clock_demand"
    assert resp_new.primary_bundle == resp_old.primary_bundle


def test_clock_prices_event_increments_demand_query_count():
    adapter = make_adapter(['{"bundle_value": 10}', '{"bundle_value": 5}'])
    adapter.handle_event(ProxyElicitationEvent(
        event_type=CLOCK_PRICES,
        bidder_id="b1",
        payload={"prices": {"A": 1.0, "B": 1.0}},
    ))
    assert adapter.stats().demand_queries == 1


# ---------------------------------------------------------------------------
# E. Optional interest map: proxy tolerates interest_map=None
# ---------------------------------------------------------------------------

def test_proxy_works_without_interest_map():
    adapter = make_adapter(['{"bundle_value": 10}', '{"bundle_value": 5}'])
    assert adapter.proxy.interest_map is None
    bid = adapter.current_bid()
    assert len(bid.atoms) == 2


def test_proxy_candidate_bundles_without_interest_map():
    # candidate_bundles can be populated without an interest map.
    adapter = make_adapter(
        ['{"bundle_value": 10}', '{"bundle_value": 5}'],
        candidate_bundles=[frozenset({"A"}), frozenset({"B"})],
    )
    assert adapter.proxy.interest_map is None
    assert len(adapter.candidate_bundles) == 2
    bid = adapter.current_bid()
    assert len(bid.atoms) == 2


# ---------------------------------------------------------------------------
# F. Initialisation events on LlmInferredXorProxy.handle_event
# ---------------------------------------------------------------------------

def test_initial_preference_question_event_on_inner_proxy():
    """INITIAL_PREFERENCE_QUESTION event calls ask_initial_question and logs it."""
    responses = [
        '{"question": "What items do you want?"}',          # proxy nl-gen
        '{"answer": "I want item A most."}',                 # person nl answer
    ]
    client = MockLlmClient(responses)
    person = LlmPersonSimulator(
        bidder_id="b1",
        scenario_description="Test.",
        person_seed="I want A.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
    )
    proxy = LlmInferredXorProxy(bidder_id="b1", person=person)
    assert proxy.nl_transcript == []

    resp = proxy.handle_event(ProxyElicitationEvent(
        event_type=INITIAL_PREFERENCE_QUESTION,
        bidder_id="b1",
        mechanism="init",
    ))

    assert resp.response_type == "natural_language_answer"
    assert len(proxy.nl_transcript) == 1
    assert len(proxy.event_log) == 1
    assert proxy.event_log[0].event_type == INITIAL_PREFERENCE_QUESTION


def test_generate_candidate_bundles_event_updates_adapter_list():
    """GENERATE_CANDIDATE_BUNDLES event via adapter updates candidate_bundles."""
    # Set up proxy with interest_map pre-populated.
    from auctionlab.llm.schemas import LlmInterestMap
    responses = [
        '{"question": "What items?"}',
        "I want A.",
        '{"bundle_value": 10}',
        '{"bundle_value": 5}',
    ]
    client = MockLlmClient(responses)
    person = LlmPersonSimulator(
        bidder_id="b1",
        scenario_description="Test.",
        person_seed="I want A.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
    )
    proxy = LlmInferredXorProxy(bidder_id="b1", person=person)
    # Pre-populate interest map to skip LLM call for derive_interest_map.
    proxy.interest_map = LlmInterestMap(
        interested_items=["A"],
        excluded_items=["B"],
        complementary_groups=[],
        substitute_groups=[],
        reasoning="test",
    )
    proxy.knowledge_base.add_qa("What items?", "I want A.")

    adapter = LlmAuctionProxyAdapter(
        bidder_id="b1", proxy=proxy,
        candidate_bundles=[],  # starts empty
    )
    assert adapter.candidate_bundles == []

    resp = adapter.handle_event(ProxyElicitationEvent(
        event_type=GENERATE_CANDIDATE_BUNDLES,
        bidder_id="b1",
        mechanism="init",
        payload={"all_items": ["A", "B"]},
    ))

    assert resp.response_type == "candidate_bundles"
    assert len(resp.payload["candidate_bundles"]) > 0
    # Adapter's candidate_bundles should be updated.
    assert adapter.candidate_bundles == resp.payload["candidate_bundles"]


def test_sealed_feedback_event_routes_to_refine():
    """SEALED_FEEDBACK event with an ElicitationEvent calls refine() internally."""
    adapter = make_adapter([
        '{"bundle_value": 10}',
        '{"bundle_value": 5}',
        '{"bundle_value": 8}',
    ])
    adapter.current_bid()

    el_event = ElicitationEvent(
        mechanism="sealed",
        event_type="allocated_bundle",
        bidder_id="b1",
        bundle=frozenset({"A", "B"}),  # new bundle → triggers LLM call
    )
    resp = adapter.handle_event(ProxyElicitationEvent(
        event_type=SEALED_FEEDBACK,
        bidder_id="b1",
        mechanism="sealed",
        payload={"elicitation_event": el_event},
    ))

    assert resp.response_type == "refinement"
    assert len(resp.records) == 1
    assert resp.records[0].bundle == frozenset({"A", "B"})


def test_event_log_accumulates_across_multiple_events():
    adapter = make_adapter([
        '{"bundle_value": 10}',
        '{"bundle_value": 5}',
    ])
    adapter.handle_event(ProxyElicitationEvent(event_type=SUBMIT_BID, bidder_id="b1"))
    adapter.handle_event(ProxyElicitationEvent(
        event_type=CLOCK_PRICES,
        bidder_id="b1",
        payload={"prices": {"A": 1.0, "B": 1.0}},
    ))
    # Both events append to proxy.event_log.
    assert len(adapter.proxy.event_log) == 2
    assert adapter.proxy.event_log[0].event_type == SUBMIT_BID
    assert adapter.proxy.event_log[1].event_type == CLOCK_PRICES
