from __future__ import annotations

import pytest

from auctionlab.bids.xor import XorAtomicBid
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmAuctionProxyAdapter, LlmInferredXorProxy
from auctionlab.proxies.base import ClockAuctionProxy, ElicitationEvent, SealedAuctionProxy
from auctionlab.proxies.baselines.dnf_learning import DnfLearningProxy
from auctionlab.proxies.baselines.hybrid import HybridProxy


ITEM_DESCRIPTIONS = {"A": "Item A", "B": "Item B"}


def make_hybrid(
    llm_responses: list[str],
    dnf_responses: list[str],
    *,
    alpha: int = 1,
    delta: float = 0.5,
) -> tuple[HybridProxy, MockLlmClient, MockLlmClient]:
    llm_client = MockLlmClient(llm_responses)
    llm_person = LlmPersonSimulator(
        bidder_id="i1",
        scenario_description="A test auction.",
        person_seed="Values A highly; B somewhat.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=llm_client,
    )
    llm_proxy = LlmAuctionProxyAdapter(
        bidder_id="i1",
        proxy=LlmInferredXorProxy(bidder_id="i1", person=llm_person, epsilon=1.0),
        candidate_bundles=[frozenset({"A"}), frozenset({"B"})],
    )

    dnf_client = MockLlmClient(dnf_responses)
    dnf_person = LlmPersonSimulator(
        bidder_id="i1",
        scenario_description="A test auction.",
        person_seed="Values A highly; B somewhat.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=dnf_client,
    )
    dnf_proxy = DnfLearningProxy(bidder_id="i1", person=dnf_person, items=["A", "B"])

    hybrid = HybridProxy(
        bidder_id="i1",
        llm_proxy=llm_proxy,
        dnf_proxy=dnf_proxy,
        alpha=alpha,
        delta=delta,
    )
    return hybrid, llm_client, dnf_client


AB_LEARN_RESPONSES = [
    '{"bundle_value": 20}',  # value_query({A, B})
    '{"bundle_value": 6}',   # value_query({B}) -- removing A changes value
    '{"bundle_value": 20}',  # value_query({A}) -- removing B keeps value
]


def test_satisfies_proxy_protocols():
    hybrid, _, _ = make_hybrid([], [])

    assert isinstance(hybrid, ClockAuctionProxy)
    assert isinstance(hybrid, SealedAuctionProxy)


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError, match="alpha"):
        make_hybrid([], [], alpha=0)


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 1.5])
def test_invalid_delta_rejected(delta):
    with pytest.raises(ValueError, match="delta"):
        make_hybrid([], [], delta=delta)


def test_current_bid_combines_dnf_and_llm_atoms():
    hybrid, llm_client, dnf_client = make_hybrid(
        [
            '{"bundle_value": 10}',  # value_query({A})
            '{"bundle_value": 6}',   # value_query({B})
        ],
        [],
    )

    bid = hybrid.current_bid()

    assert XorAtomicBid(bundle=frozenset(), value=0.0) in bid.atoms
    assert XorAtomicBid(bundle=frozenset({"A", "B"}), value=0.0) in bid.atoms
    assert XorAtomicBid(bundle=frozenset({"A"}), value=10.0) in bid.atoms
    assert XorAtomicBid(bundle=frozenset({"B"}), value=6.0) in bid.atoms
    assert len(llm_client.calls) == 2
    assert len(dnf_client.calls) == 0


def test_early_refine_of_known_bundle_is_noop():
    # Bundles already valued in the initial phase should not be re-queried;
    # the LLM call is skipped and the existing value is kept.
    hybrid, llm_client, dnf_client = make_hybrid(
        [
            '{"bundle_value": 10}',  # value_query({A})
            '{"bundle_value": 6}',   # value_query({B})
        ],
        [],
        alpha=1,
    )
    hybrid.current_bid()

    hybrid.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="i1",
            bundle=frozenset({"A"}),
        )
    )

    assert len(llm_client.calls) == 2  # no re-query issued
    assert len(dnf_client.calls) == 0
    bid = hybrid.current_bid()
    assert XorAtomicBid(bundle=frozenset({"A"}), value=10.0) in bid.atoms


def test_early_refine_of_new_bundle_delegates_to_llm():
    # Bundles NOT in the initial candidate set should still trigger an LLM call
    # when the hybrid is in its early (LLM) phase.
    hybrid, llm_client, dnf_client = make_hybrid(
        [
            '{"bundle_value": 10}',  # value_query({A})
            '{"bundle_value": 6}',   # value_query({B})
            '{"bundle_value": 12}',  # refine_bundle_value({A, B}) — new bundle
        ],
        [],
        alpha=2,
    )
    hybrid.current_bid()

    hybrid.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
        )
    )

    assert len(llm_client.calls) == 3  # new bundle triggers LLM call
    assert len(dnf_client.calls) == 0
    # The LLM proxy's internal bid is updated; the hybrid bid shows DNF's 0.0
    # for {A, B} since DNF atoms take precedence in current_bid().
    llm_bid = hybrid.llm_proxy.current_bid()
    assert XorAtomicBid(bundle=frozenset({"A", "B"}), value=12.0) in llm_bid.atoms


def test_post_switch_refine_delegates_to_dnf_and_decays_llm_atoms():
    hybrid, llm_client, dnf_client = make_hybrid(
        [
            '{"bundle_value": 10}',  # value_query({A})
            '{"bundle_value": 6}',   # value_query({B})
        ],
        AB_LEARN_RESPONSES + AB_LEARN_RESPONSES + AB_LEARN_RESPONSES,
        alpha=1,
        delta=0.5,
    )
    hybrid.current_bid()

    # First call (index 0 < alpha): {A} is already valued → no-op, but _calls
    # still increments so the switch fires on the next call.
    hybrid.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="i1",
            bundle=frozenset({"A"}),
        )
    )
    assert len(dnf_client.calls) == 0

    # Second call (index 1 == alpha): first post-switch call, gamma stays 1.0.
    hybrid.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_tie",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
        )
    )
    assert len(dnf_client.calls) == 3
    bid = hybrid.current_bid()
    dnf_bundles = {atom.bundle for atom in hybrid.dnf_proxy.current_bid().atoms}
    assert frozenset({"A"}) in dnf_bundles
    b_atom = next(atom for atom in bid.atoms if atom.bundle == frozenset({"B"}))
    assert b_atom.value == 6.0  # gamma_factor == 1.0

    # Third call (index 2 > alpha): gamma_factor *= delta.
    hybrid.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_tie",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
        )
    )
    bid = hybrid.current_bid()
    b_atom = next(atom for atom in bid.atoms if atom.bundle == frozenset({"B"}))
    assert b_atom.value == 3.0  # gamma_factor == 0.5

    # Fourth call: gamma_factor *= delta again.
    hybrid.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_tie",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
        )
    )
    bid = hybrid.current_bid()
    b_atom = next(atom for atom in bid.atoms if atom.bundle == frozenset({"B"}))
    assert b_atom.value == 1.5  # gamma_factor == 0.25


def test_stats_and_refinement_records_merge_both_delegates():
    hybrid, _, _ = make_hybrid(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 6}',
        ],
        AB_LEARN_RESPONSES,
        alpha=1,
    )
    hybrid.current_bid()
    # First refine: {A} is already valued → no-op (LLM not called, no record).
    hybrid.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_zero_surplus",
            bidder_id="i1",
            bundle=frozenset({"A"}),
        )
    )
    # Second refine: past alpha threshold → DNF handles {A, B}.
    hybrid.refine(
        ElicitationEvent(
            mechanism="test",
            event_type="near_tie",
            bidder_id="i1",
            bundle=frozenset({"A", "B"}),
        )
    )

    stats = hybrid.stats()
    assert stats.refinement_queries == 1  # only the DNF refinement counted
    assert stats.value_queries == 2 + 3  # llm initial inference + dnf learning

    records = hybrid.refinement_records()
    assert len(records) == 1
