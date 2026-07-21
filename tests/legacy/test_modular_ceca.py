"""Tests for the modular LLM proxy as the primary CECA architecture.

Covers:
- LlmAuctionProxyAdapter satisfies CecaAuctionProxy protocol
- CECA runner works with modular LLM proxy via ground-truth bypass
- CECA demanded bundle is upserted into the shared current bid
- RefinementRecord added for unsatisfied CECA demand (event_type="unsatisfied_demand")
- replay_elicitation() provides identical informational starting state across arms
- proxy_architecture metadata field is "modular_llm" for LlmAuctionProxyAdapter
- Legacy vd1/vd2/nvd still satisfy CecaAuctionProxy and run without error
"""

from __future__ import annotations

import json
from itertools import combinations

import pytest

from auctionlab.auctions.ceca import CecaConfig
from auctionlab.experiments.proxy_ceca_runner import (
    ProxyCecaConfig,
    run_proxy_ceca_experiment,
)
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmAuctionProxyAdapter, LlmInferredXorProxy
from auctionlab.proxies.base import CecaAuctionProxy
from auctionlab.proxies.baselines.llm_ceca import NvdCecaProxy, Vd1CecaProxy, Vd2CecaProxy

ITEMS = ["A", "B", "C"]
ITEM_DESCRIPTIONS = {"A": "Item A", "B": "Item B", "C": "Item C"}
SCENARIO = "Test combinatorial auction."

TRUE_VALS = {
    frozenset({"A"}): 10.0,
    frozenset({"B"}): 5.0,
    frozenset({"C"}): 3.0,
    frozenset({"A", "B"}): 20.0,
    frozenset({"A", "C"}): 12.0,
    frozenset({"B", "C"}): 7.0,
    frozenset({"A", "B", "C"}): 22.0,
}


def _all_bundles(items: list[str]) -> list[frozenset]:
    return [
        frozenset(combo)
        for size in range(1, len(items) + 1)
        for combo in combinations(items, size)
    ]


def _make_gt_person(bidder_id: str = "i1", gt: dict | None = None) -> LlmPersonSimulator:
    return LlmPersonSimulator(
        bidder_id=bidder_id,
        scenario_description=SCENARIO,
        person_seed="(ground-truth)",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=MockLlmClient([]),
        ground_truth_valuations=gt or TRUE_VALS,
    )


def _make_adapter(
    bidder_id: str = "i1",
    candidate_bundles: list | None = None,
    gt: dict | None = None,
    epsilon: float = 1.0,
) -> LlmAuctionProxyAdapter:
    proxy = LlmInferredXorProxy(
        bidder_id=bidder_id,
        person=_make_gt_person(bidder_id, gt),
        epsilon=epsilon,
    )
    return LlmAuctionProxyAdapter(
        bidder_id=bidder_id,
        proxy=proxy,
        candidate_bundles=candidate_bundles if candidate_bundles is not None
        else [frozenset({"A"}), frozenset({"B"}), frozenset({"C"})],
        discount_inferred=False,
    )


def _make_toy_adapters(toy_instance) -> list[LlmAuctionProxyAdapter]:
    """Build one LlmAuctionProxyAdapter per bidder using the toy instance's ground truth."""
    adapters = []
    all_b = _all_bundles(list(toy_instance.items))
    for bidder_id in toy_instance.bidder_ids:
        gt = {b: toy_instance.value_of(bidder_id, b) for b in all_b}
        proxy = LlmInferredXorProxy(
            bidder_id=bidder_id,
            person=LlmPersonSimulator(
                bidder_id=bidder_id,
                scenario_description=SCENARIO,
                person_seed="(ground-truth)",
                item_descriptions={item: item for item in toy_instance.items},
                client=MockLlmClient([]),
                ground_truth_valuations=gt,
            ),
            epsilon=1.0,
        )
        adapters.append(
            LlmAuctionProxyAdapter(
                bidder_id=bidder_id,
                proxy=proxy,
                candidate_bundles=all_b,
                discount_inferred=False,
            )
        )
    return adapters


# --------------------------------------------------------------------------- #
# 1. Protocol conformance                                                      #
# --------------------------------------------------------------------------- #

def test_modular_proxy_satisfies_ceca_auction_proxy_protocol():
    adapter = _make_adapter()
    assert isinstance(adapter, CecaAuctionProxy)


# --------------------------------------------------------------------------- #
# 2. CECA runner accepts modular LLM proxy and returns expected fields         #
# --------------------------------------------------------------------------- #

def test_ceca_runner_with_modular_llm_proxy(toy_instance):
    proxies = _make_toy_adapters(toy_instance)
    result = run_proxy_ceca_experiment(
        toy_instance,
        proxies,
        CecaConfig(max_rounds=20),
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )
    assert result.mechanism == "proxy_ceca_pay_as_bid"
    assert result.welfare >= 0.0
    assert isinstance(result.metadata["converged"], bool)
    assert set(result.metadata["demand_query_count_by_bidder"]) == set(
        toy_instance.bidder_ids
    )


# --------------------------------------------------------------------------- #
# 3. proxy_architecture metadata                                               #
# --------------------------------------------------------------------------- #

def test_ceca_runner_records_proxy_architecture_modular_llm(toy_instance):
    proxies = _make_toy_adapters(toy_instance)
    result = run_proxy_ceca_experiment(
        toy_instance,
        proxies,
        CecaConfig(max_rounds=10),
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )
    assert result.metadata["proxy_architecture"] == "modular_llm"


# --------------------------------------------------------------------------- #
# 4. CECA demanded bundle is upserted into the shared current bid             #
# --------------------------------------------------------------------------- #

def test_ceca_demanded_bundle_upserted_into_current_bid():
    adapter = _make_adapter(
        candidate_bundles=[frozenset({"A"}), frozenset({"B"}), frozenset({"C"})],
    )
    adapter.current_bid()  # pre-populate: A=10, B=5, C=3

    # At Lindahl price=0 for all bundles, the cached bid already has the best
    # surplus at {A}=10.  The empty bundle triggers the "best != current"
    # branch and a ceca_demand_query to the person; {A,B,C}=22 beats all at
    # price=0 → unsatisfied, demanded bundle should be added to the bid.
    response = adapter.ceca_step(
        prices=lambda b: 0.0,
        current_bundle=frozenset(),
        round_idx=0,
    )

    assert not response.satisfied
    assert response.demanded_bundle is not None

    bid = adapter.current_bid()
    demanded_bundles = {a.bundle for a in bid.atoms}
    assert response.demanded_bundle in demanded_bundles, (
        "demanded bundle must be present in the current bid after ceca_step"
    )
    demanded_atom = next(a for a in bid.atoms if a.bundle == response.demanded_bundle)
    assert demanded_atom.value == pytest.approx(response.value)


def test_ceca_satisfied_step_does_not_change_bid():
    adapter = _make_adapter(
        candidate_bundles=[frozenset({"A"}), frozenset({"B"}), frozenset({"C"})],
    )
    adapter.current_bid()

    bid_atoms_before = {a.bundle: a.value for a in adapter.current_bid().atoms}

    # {A} priced at exactly its value → current_bundle == best cached → satisfied.
    response = adapter.ceca_step(
        prices=lambda b: sum(
            {frozenset({"A"}): 10.0, frozenset({"B"}): 5.0, frozenset({"C"}): 3.0}.get(
                frozenset({item}), 0.0
            )
            for item in b
        ),
        current_bundle=frozenset({"A"}),
        round_idx=0,
    )

    bid_atoms_after = {a.bundle: a.value for a in adapter.current_bid().atoms}
    assert response.satisfied is True
    assert bid_atoms_before == bid_atoms_after


# --------------------------------------------------------------------------- #
# 5. RefinementRecord for unsatisfied CECA demand                             #
# --------------------------------------------------------------------------- #

def test_ceca_step_records_unsatisfied_demand_refinement_record():
    adapter = _make_adapter(
        candidate_bundles=[frozenset({"A"}), frozenset({"B"}), frozenset({"C"})],
    )
    adapter.current_bid()

    response = adapter.ceca_step(
        prices=lambda b: 0.0,
        current_bundle=frozenset(),
        round_idx=3,
    )

    if not response.satisfied:
        records = adapter.refinement_records()
        assert len(records) >= 1
        r = records[0]
        assert r.event_type == "unsatisfied_demand"
        assert r.round_idx == 3
        assert r.bundle == response.demanded_bundle
        assert r.new_value == pytest.approx(response.value)
        assert r.bidder_id == "i1"
        assert "lindahl_price" in (r.reason or "")


def test_ceca_step_satisfied_adds_no_refinement_record():
    adapter = _make_adapter(
        candidate_bundles=[frozenset({"A"}), frozenset({"B"}), frozenset({"C"})],
    )
    adapter.current_bid()

    # Price {A} at its full value → satisfied immediately.
    response = adapter.ceca_step(
        prices=lambda b: TRUE_VALS.get(b, 0.0),
        current_bundle=frozenset({"A"}),
        round_idx=0,
    )

    assert response.satisfied is True
    assert adapter.refinement_records() == []


# --------------------------------------------------------------------------- #
# 6. replay_elicitation() shares NL+PV state across sealed, clock, CECA arms  #
# --------------------------------------------------------------------------- #

def _make_proxy_with_replay(*, pv_values: dict) -> LlmInferredXorProxy:
    proxy = LlmInferredXorProxy(
        bidder_id="i1",
        person=_make_gt_person(),
        epsilon=1.0,
    )
    proxy.replay_elicitation(
        nl_question="Which items interest you most?",
        nl_answer="I'm most interested in A and B together.",
        interest_map=None,
        provisional_raw_values=pv_values,
        discount_inferred=False,
    )
    return proxy


def test_replay_elicitation_identical_starting_state_across_arms():
    pv_values = {
        frozenset({"A"}): 10.0,
        frozenset({"B"}): 5.0,
        frozenset({"A", "B"}): 20.0,
    }

    proxy_sealed = _make_proxy_with_replay(pv_values=pv_values)
    proxy_clock = _make_proxy_with_replay(pv_values=pv_values)
    proxy_ceca = _make_proxy_with_replay(pv_values=pv_values)

    # All three should have the same cached atoms from PV.
    for px in (proxy_sealed, proxy_clock, proxy_ceca):
        assert px._cached_bid is not None
        assert {a.bundle for a in px._cached_bid.atoms} == set(pv_values.keys())

    # NL transcript is the same across all arms.
    expected_transcript = [("Which items interest you most?", "I'm most interested in A and B together.")]
    assert proxy_sealed.nl_transcript == expected_transcript
    assert proxy_clock.nl_transcript == expected_transcript
    assert proxy_ceca.nl_transcript == expected_transcript


def test_replay_elicitation_arms_have_independent_mutable_state():
    pv_values = {
        frozenset({"A"}): 10.0,
        frozenset({"B"}): 5.0,
    }

    proxy_a = _make_proxy_with_replay(pv_values=pv_values)
    proxy_b = _make_proxy_with_replay(pv_values=pv_values)

    # Mutating one proxy's cached bid does not affect the other.
    proxy_a._cached_bid.atoms[0] = proxy_a._cached_bid.atoms[0].__class__(
        bundle=proxy_a._cached_bid.atoms[0].bundle,
        value=999.0,
    )
    b_values = {a.bundle: a.value for a in proxy_b._cached_bid.atoms}
    assert 999.0 not in b_values.values()


# --------------------------------------------------------------------------- #
# 7. Legacy vd1/vd2/nvd still satisfy CecaAuctionProxy and run without error  #
# --------------------------------------------------------------------------- #

def _make_vd1() -> Vd1CecaProxy:
    return Vd1CecaProxy(
        bidder_id="i1",
        person=_make_gt_person(),
        items=ITEMS,
        proxy_client=MockLlmClient([json.dumps({"action": "satisfied"})]),
        scenario_description=SCENARIO,
        item_descriptions=ITEM_DESCRIPTIONS,
    )


def _make_vd2() -> Vd2CecaProxy:
    gamma_resp = json.dumps({
        "estimates": [
            {"bundle": ["A"], "estimated_value": 10.0},
            {"bundle": ["B"], "estimated_value": 5.0},
            {"bundle": ["C"], "estimated_value": 3.0},
        ]
    })
    action_resp = json.dumps({"action": "satisfied"})
    from auctionlab.proxies.baselines.llm_ceca import SizeLimitedScope
    return Vd2CecaProxy(
        bidder_id="i1",
        person=_make_gt_person(),
        items=ITEMS,
        proxy_client=MockLlmClient([gamma_resp, action_resp]),
        scenario_description=SCENARIO,
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle_scope=SizeLimitedScope(max_size=1),
    )


def _make_nvd() -> NvdCecaProxy:
    nl_q = json.dumps({"question": "What do you want?"})
    gamma_resp = json.dumps({
        "estimates": [
            {"bundle": ["A"], "estimated_value": 10.0},
            {"bundle": ["B"], "estimated_value": 5.0},
            {"bundle": ["C"], "estimated_value": 3.0},
        ]
    })
    action_resp = json.dumps({"action": "satisfied"})
    person_answer = json.dumps({"answer": "I want A most."})
    from auctionlab.proxies.baselines.llm_ceca import SizeLimitedScope

    person = LlmPersonSimulator(
        bidder_id="i1",
        scenario_description=SCENARIO,
        person_seed="(ground-truth)",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=MockLlmClient([person_answer]),
        ground_truth_valuations=TRUE_VALS,
    )
    return NvdCecaProxy(
        bidder_id="i1",
        person=person,
        items=ITEMS,
        proxy_client=MockLlmClient([nl_q, gamma_resp, action_resp]),
        scenario_description=SCENARIO,
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle_scope=SizeLimitedScope(max_size=1),
        num_nl_questions=1,
    )


def test_legacy_vd1_satisfies_ceca_proxy_protocol():
    assert isinstance(_make_vd1(), CecaAuctionProxy)


def test_legacy_vd2_satisfies_ceca_proxy_protocol():
    assert isinstance(_make_vd2(), CecaAuctionProxy)


def test_legacy_nvd_satisfies_ceca_proxy_protocol():
    assert isinstance(_make_nvd(), CecaAuctionProxy)


def test_legacy_vd1_ceca_step_runs_without_error():
    proxy = _make_vd1()
    result = proxy.ceca_step(lambda b: 4.0 * len(b), frozenset({"A", "B", "C"}), round_idx=0)
    assert isinstance(result.satisfied, bool)


def test_legacy_vd2_ceca_step_runs_without_error():
    proxy = _make_vd2()
    result = proxy.ceca_step(lambda b: 4.0 * len(b), frozenset({"A", "B", "C"}), round_idx=0)
    assert isinstance(result.satisfied, bool)


def test_legacy_nvd_ceca_step_runs_without_error():
    proxy = _make_nvd()
    result = proxy.ceca_step(lambda b: 4.0 * len(b), frozenset({"A", "B", "C"}), round_idx=0)
    assert isinstance(result.satisfied, bool)


# --------------------------------------------------------------------------- #
# 8. proxy_architecture for non-LLM proxy types                               #
# --------------------------------------------------------------------------- #

def test_full_info_proxy_architecture_in_ceca_metadata(toy_instance):
    from auctionlab.proxies.full_info import FullInfoAuctionProxy
    proxies = [
        FullInfoAuctionProxy(bidder_id=bid, instance=toy_instance, initial="empty")
        for bid in toy_instance.bidder_ids
    ]
    result = run_proxy_ceca_experiment(
        toy_instance,
        proxies,
        CecaConfig(max_rounds=10),
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )
    assert result.metadata["proxy_architecture"] == "full_info"


def test_dnf_proxy_architecture_in_ceca_metadata(toy_instance):
    from auctionlab.llm.person_simulator import LlmPersonSimulator
    from auctionlab.proxies.baselines.dnf_learning import DnfLearningProxy
    proxies = [
        DnfLearningProxy(
            bidder_id=bid,
            person=LlmPersonSimulator(
                bidder_id=bid,
                scenario_description=SCENARIO,
                person_seed="gt",
                item_descriptions={item: item for item in toy_instance.items},
                client=MockLlmClient([]),
                ground_truth_valuations={
                    b: toy_instance.value_of(bid, b)
                    for b in _all_bundles(list(toy_instance.items))
                },
            ),
            items=list(toy_instance.items),
        )
        for bid in toy_instance.bidder_ids
    ]
    result = run_proxy_ceca_experiment(
        toy_instance,
        proxies,
        CecaConfig(max_rounds=10),
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )
    assert result.metadata["proxy_architecture"] == "dnf"
