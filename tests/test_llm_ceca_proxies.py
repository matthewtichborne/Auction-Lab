"""Tests for the ωvd1, ωvd2, and ωnvd LLM CECA proxy implementations."""

from __future__ import annotations

import json

from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.proxies.base import CecaAuctionProxy
from auctionlab.proxies.baselines.llm_ceca import (
    AllBundlesScope,
    BundleScope,
    InterestMapScope,
    NvdCecaProxy,
    SizeLimitedScope,
    Vd1CecaProxy,
    Vd2CecaProxy,
)

ITEMS = ["A", "B", "C"]
ITEM_DESCRIPTIONS = {"A": "Item A", "B": "Item B", "C": "Item C"}
SCENARIO = "A test combinatorial auction with three items."

# True valuations: bidder wants {A,B} as a set, B alone is worth less.
TRUE_VALS = {
    frozenset({"A"}): 10.0,
    frozenset({"B"}): 5.0,
    frozenset({"C"}): 3.0,
    frozenset({"A", "B"}): 20.0,
    frozenset({"A", "C"}): 12.0,
    frozenset({"B", "C"}): 7.0,
    frozenset({"A", "B", "C"}): 22.0,
}


def _prices(bundle):
    """Simple flat price function: $4 per item in the bundle."""
    return 4.0 * len(bundle)


def _make_person(proxy_client=None):
    return LlmPersonSimulator(
        bidder_id="i1",
        scenario_description=SCENARIO,
        person_seed="(ground-truth)",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=proxy_client or MockLlmClient([]),
        ground_truth_valuations=TRUE_VALS,
    )


def _make_vd1(proxy_responses: list[str]) -> Vd1CecaProxy:
    return Vd1CecaProxy(
        bidder_id="i1",
        person=_make_person(),
        items=ITEMS,
        proxy_client=MockLlmClient(proxy_responses),
        scenario_description=SCENARIO,
        item_descriptions=ITEM_DESCRIPTIONS,
    )


# --------------------------------------------------------------------------- #
# BundleScope tests                                                            #
# --------------------------------------------------------------------------- #

def test_all_bundles_scope_enumerates_all_nonempty_subsets():
    scope = AllBundlesScope()
    bundles = scope.bundles(["A", "B"])
    assert frozenset({"A"}) in bundles
    assert frozenset({"B"}) in bundles
    assert frozenset({"A", "B"}) in bundles
    assert frozenset() not in bundles
    assert len(bundles) == 3


def test_size_limited_scope_caps_at_max_size():
    scope = SizeLimitedScope(max_size=1)
    bundles = scope.bundles(["A", "B", "C"])
    assert all(len(b) == 1 for b in bundles)
    assert len(bundles) == 3

    scope2 = SizeLimitedScope(max_size=2)
    bundles2 = scope2.bundles(["A", "B", "C"])
    assert all(len(b) <= 2 for b in bundles2)
    assert len(bundles2) == 6  # 3 singletons + 3 pairs


def test_bundle_scope_protocol_satisfied():
    assert isinstance(AllBundlesScope(), BundleScope)
    assert isinstance(SizeLimitedScope(), BundleScope)


# --------------------------------------------------------------------------- #
# Vd1CecaProxy: protocol conformance                                          #
# --------------------------------------------------------------------------- #

def test_vd1_satisfies_ceca_proxy_protocol():
    proxy = _make_vd1([])
    assert isinstance(proxy, CecaAuctionProxy)


def test_vd1_initial_bid_is_empty():
    proxy = _make_vd1([])
    assert proxy.current_bid().atoms == []
    assert proxy.stats().value_queries == 0


# --------------------------------------------------------------------------- #
# Vd1CecaProxy: satisfied action                                              #
# --------------------------------------------------------------------------- #

def test_vd1_ceca_step_satisfied_action_returns_satisfied():
    action_response = json.dumps({"action": "satisfied", "reasoning": "happy"})
    proxy = _make_vd1([action_response])
    # current bundle = {A}, prices = $4/item → value=10, surplus=6
    result = proxy.ceca_step(_prices, frozenset({"A"}), round_idx=0)

    assert result.satisfied is True
    assert result.demanded_bundle is None
    # One value_query for current bundle
    assert proxy.stats().value_queries == 1


# --------------------------------------------------------------------------- #
# Vd1CecaProxy: value_query action                                            #
# --------------------------------------------------------------------------- #

def test_vd1_ceca_step_value_query_higher_surplus_returns_unsatisfied():
    # current = {A}: value=10, price=4, surplus=6
    # query {A,B}: value=20, price=8, surplus=12  (>6 → unsatisfied)
    action_response = json.dumps({"action": "value_query", "bundle": ["A", "B"]})
    proxy = _make_vd1([action_response])

    result = proxy.ceca_step(_prices, frozenset({"A"}), round_idx=0)

    assert result.satisfied is False
    assert result.demanded_bundle == frozenset({"A", "B"})
    assert result.value == 20.0
    assert proxy.stats().value_queries == 2  # current + target
    # Atom should be upserted
    bid = proxy.current_bid()
    assert any(a.bundle == frozenset({"A", "B"}) and a.value == 20.0 for a in bid.atoms)


def test_vd1_ceca_step_value_query_lower_surplus_returns_satisfied():
    # current = {A,B}: value=20, price=8, surplus=12
    # query {C}: value=3, price=4, surplus=-1  (<12 → satisfied)
    action_response = json.dumps({"action": "value_query", "bundle": ["C"]})
    proxy = _make_vd1([action_response])

    result = proxy.ceca_step(_prices, frozenset({"A", "B"}), round_idx=0)

    assert result.satisfied is True
    assert proxy.stats().value_queries == 2  # current + queried


# --------------------------------------------------------------------------- #
# Vd1CecaProxy: demand_query action                                           #
# --------------------------------------------------------------------------- #

def test_vd1_ceca_step_demand_query_unsatisfied_learns_atom():
    # current = {} (empty), manifest is empty so all bundles have price 0.
    # GT picks {A,B,C}: value=22, surplus=22 (highest of all bundles at price 0).
    action_response = json.dumps({"action": "demand_query"})
    proxy = _make_vd1([action_response])

    result = proxy.ceca_step(_prices, frozenset(), round_idx=0)

    assert result.satisfied is False
    assert result.demanded_bundle == frozenset({"A", "B", "C"})
    assert result.value == 22.0
    assert proxy.stats().demand_queries == 1
    assert proxy.stats().value_queries == 1  # value_query for preferred (empty skips)


def test_vd1_ceca_step_demand_query_satisfied_when_current_is_optimal():
    # current = {A,B,C}: value=22 at price=0 (empty manifest) dominates all other bundles.
    # GT: surplus({A,B,C})=22 beats all others → satisfied.
    action_response = json.dumps({"action": "demand_query"})
    proxy = _make_vd1([action_response])

    result = proxy.ceca_step(_prices, frozenset({"A", "B", "C"}), round_idx=0)

    assert result.satisfied is True
    assert proxy.stats().demand_queries == 1


# --------------------------------------------------------------------------- #
# Vd1CecaProxy: invalid item IDs in value_query fall back to demand_query     #
# --------------------------------------------------------------------------- #

def test_vd1_invalid_bundle_ids_in_value_query_fall_back_to_demand():
    # The proxy returns a bundle with unknown item IDs → falls back to demand_query.
    # current = {A,B,C}: value=22, price=0 (empty manifest) → dominates all → satisfied.
    action_response = json.dumps({"action": "value_query", "bundle": ["Z", "UNKNOWN"]})
    proxy = _make_vd1([action_response])

    result = proxy.ceca_step(_prices, frozenset({"A", "B", "C"}), round_idx=0)

    assert result.satisfied is True
    assert proxy.stats().demand_queries == 1


# --------------------------------------------------------------------------- #
# Vd2CecaProxy: gamma inference                                               #
# --------------------------------------------------------------------------- #

def _make_vd2(proxy_responses: list[str], **kwargs) -> Vd2CecaProxy:
    return Vd2CecaProxy(
        bidder_id="i1",
        person=_make_person(),
        items=ITEMS,
        proxy_client=MockLlmClient(proxy_responses),
        scenario_description=SCENARIO,
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle_scope=SizeLimitedScope(max_size=1),
        **kwargs,
    )


def test_vd2_satisfies_ceca_proxy_protocol():
    proxy = _make_vd2([])
    assert isinstance(proxy, CecaAuctionProxy)


def test_vd2_refreshes_gamma_before_first_step():
    gamma_response = json.dumps({
        "estimates": [
            {"bundle": ["A"], "estimated_value": 10.0},
            {"bundle": ["B"], "estimated_value": 5.0},
            {"bundle": ["C"], "estimated_value": 3.0},
        ],
        "reasoning": "Based on transcript.",
    })
    action_response = json.dumps({"action": "satisfied"})
    proxy = _make_vd2([gamma_response, action_response])

    proxy.ceca_step(_prices, frozenset({"A", "B"}), round_idx=0)

    assert proxy._gamma_cache[frozenset({"A"})] == 10.0
    assert proxy._gamma_cache[frozenset({"B"})] == 5.0
    assert proxy._gamma_cache[frozenset({"C"})] == 3.0


def test_vd2_gamma_not_refreshed_until_interval():
    gamma_response = json.dumps({
        "estimates": [
            {"bundle": ["A"], "estimated_value": 8.0},
            {"bundle": ["B"], "estimated_value": 4.0},
            {"bundle": ["C"], "estimated_value": 2.0},
        ],
    })
    action_response_1 = json.dumps({"action": "satisfied"})
    action_response_2 = json.dumps({"action": "satisfied"})
    # gamma_refresh_every=2 → refresh at rounds 0, 2, 4, ...
    proxy = _make_vd2(
        [gamma_response, action_response_1, action_response_2],
        gamma_refresh_every=2,
    )

    proxy.ceca_step(_prices, frozenset({"A"}), round_idx=0)  # refresh + action
    proxy.ceca_step(_prices, frozenset({"A"}), round_idx=1)  # no refresh, only action

    # MockLlmClient should have consumed: gamma + 2 actions = 3 responses
    assert len(proxy.proxy_client.prompts) == 3


def test_vd2_gamma_disabled_when_refresh_every_zero():
    action_response = json.dumps({"action": "satisfied"})
    proxy = _make_vd2([action_response], gamma_refresh_every=0)

    proxy.ceca_step(_prices, frozenset({"A"}), round_idx=0)

    assert proxy._gamma_cache == {}
    assert len(proxy.proxy_client.prompts) == 1  # only action choice


# --------------------------------------------------------------------------- #
# NvdCecaProxy: NL questions before first CECA step                           #
# --------------------------------------------------------------------------- #

def _make_nvd(
    proxy_responses: list[str],
    person_responses: list[str] | None = None,
    num_nl: int = 1,
) -> NvdCecaProxy:
    return NvdCecaProxy(
        bidder_id="i1",
        person=_make_person(MockLlmClient(person_responses or [])),
        items=ITEMS,
        proxy_client=MockLlmClient(proxy_responses),
        scenario_description=SCENARIO,
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle_scope=SizeLimitedScope(max_size=1),
        num_nl_questions=num_nl,
    )


def test_nvd_satisfies_ceca_proxy_protocol():
    proxy = _make_nvd([])
    assert isinstance(proxy, CecaAuctionProxy)


_PERSON_ANSWER = json.dumps({"answer": "I am most interested in item A."})


def test_nvd_asks_nl_question_then_refreshes_gamma_then_action():
    # Proxy LLM sequence: generate question → γ inference → action choice (3 calls)
    # Person sequence: answer the question (1 call via person client)
    nl_q_response = json.dumps({"question": "What items interest you most?"})
    gamma_response = json.dumps({
        "estimates": [
            {"bundle": ["A"], "estimated_value": 10.0},
            {"bundle": ["B"], "estimated_value": 5.0},
            {"bundle": ["C"], "estimated_value": 3.0},
        ],
    })
    action_response = json.dumps({"action": "satisfied"})

    proxy = _make_nvd(
        proxy_responses=[nl_q_response, gamma_response, action_response],
        person_responses=[_PERSON_ANSWER],
    )
    proxy.ceca_step(_prices, frozenset({"A", "B", "C"}), round_idx=0)

    assert len(proxy.proxy_client.prompts) == 3
    assert len(proxy.nl_transcript) == 1
    assert proxy._nl_questions_done is True
    assert proxy._gamma_cache[frozenset({"A"})] == 10.0
    assert proxy.stats().nl_queries == 1


def test_nvd_nl_questions_asked_only_once():
    nl_q = json.dumps({"question": "What do you want?"})
    gamma = json.dumps({"estimates": [
        {"bundle": ["A"], "estimated_value": 9.0},
        {"bundle": ["B"], "estimated_value": 4.0},
        {"bundle": ["C"], "estimated_value": 2.0},
    ]})
    action1 = json.dumps({"action": "satisfied"})
    action2 = json.dumps({"action": "satisfied"})

    proxy = _make_nvd(
        proxy_responses=[nl_q, gamma, action1, action2],
        person_responses=[_PERSON_ANSWER],
        num_nl=1,
    )
    proxy.gamma_refresh_every = 0  # disable periodic refresh; seeding in _ask_nl_questions still fires

    proxy.ceca_step(_prices, frozenset({"A", "B", "C"}), round_idx=0)  # nl_q + gamma + action
    proxy.ceca_step(_prices, frozenset({"A", "B", "C"}), round_idx=1)  # action only

    assert len(proxy.proxy_client.prompts) == 4
    assert proxy.stats().nl_queries == 1


def test_nvd_multiple_nl_questions():
    # 3 NL questions: 3 proxy calls (gen) + 3 person calls (answers)
    # γ refresh: 1 proxy call; action: 1 proxy call → 5 proxy total
    proxy_responses = [
        json.dumps({"question": f"Question {i + 1}?"}) for i in range(3)
    ] + [
        json.dumps({"estimates": [
            {"bundle": ["A"], "estimated_value": 10.0},
            {"bundle": ["B"], "estimated_value": 5.0},
            {"bundle": ["C"], "estimated_value": 3.0},
        ]}),
        json.dumps({"action": "satisfied"}),
    ]
    person_responses = [_PERSON_ANSWER] * 3

    proxy = _make_nvd(
        proxy_responses=proxy_responses,
        person_responses=person_responses,
        num_nl=3,
    )
    proxy.ceca_step(_prices, frozenset({"A", "B", "C"}), round_idx=0)

    assert proxy.stats().nl_queries == 3
    assert len(proxy.nl_transcript) == 3
    assert len(proxy.proxy_client.prompts) == 5
