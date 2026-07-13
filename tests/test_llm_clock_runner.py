from __future__ import annotations

import pytest

from auctionlab.auctions.clock import ClockConfig
from auctionlab.experiments.llm_comparison import (
    clock_llm_comparison_to_row,
    run_clock_llm_comparison,
)
from auctionlab.experiments.llm_runner import (
    make_llm_proxies_for_instance,
    run_clock_llm_proxy_experiment,
)
from auctionlab.instances.base import AuctionInstance
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmInferredXorProxy


def make_single_proxy(
    responses: list[str],
    *,
    items: tuple[str, ...] = ("A", "B"),
) -> tuple[LlmInferredXorProxy, MockLlmClient]:
    client = MockLlmClient(responses)
    person = LlmPersonSimulator(
        bidder_id="i1",
        scenario_description="A test auction.",
        person_seed="Has deterministic preferences.",
        item_descriptions={
            item: f"Item {item}"
            for item in items
        },
        client=client,
    )
    return (
        LlmInferredXorProxy(
            bidder_id="i1",
            person=person,
            epsilon=1.0,
        ),
        client,
    )


def make_clock_setup():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"A"}): 10.0},
            "i2": {frozenset({"B"}): 9.0},
        },
    )
    clients = {
        "i1": MockLlmClient(
            [
                '{"bundle_value": 10}',
                '{"bundle_value": 1}',
            ]
        ),
        "i2": MockLlmClient(
            [
                '{"bundle_value": 2}',
                '{"bundle_value": 9}',
            ]
        ),
    }
    proxies = make_llm_proxies_for_instance(
        instance=instance,
        scenario_description="A two-item auction.",
        person_seeds={
            "i1": "Prefers A.",
            "i2": "Prefers B.",
        },
        item_descriptions={
            "A": "Item A",
            "B": "Item B",
        },
        clients=clients,
        epsilon=1.0,
    )
    candidates = [
        frozenset({"A"}),
        frozenset({"B"}),
    ]
    return instance, proxies, clients, candidates


def test_cached_bid_avoids_repeated_llm_calls():
    proxy, client = make_single_proxy(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
        ]
    )
    candidates = [
        frozenset({"A"}),
        frozenset({"B"}),
    ]

    proxy.infer_cached_xor_bid(candidates)
    proxy.clock_demand_from_cached_bid({"A": 0.0, "B": 0.0})
    proxy.clock_demand_from_cached_bid({"A": 1.0, "B": 1.0})
    proxy.infer_cached_xor_bid(candidates)

    assert len(client.calls) == 2


def test_clock_demand_from_cached_bid_selects_best_surplus():
    proxy, _client = make_single_proxy(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
        ]
    )
    proxy.infer_cached_xor_bid(
        [
            frozenset({"A"}),
            frozenset({"B"}),
        ]
    )

    response = proxy.clock_demand_from_cached_bid(
        {"A": 3.0, "B": 0.0}
    )

    assert response.primary_bundle == frozenset({"A"})


def test_clock_demand_from_cached_bid_returns_full_bid_regardless_of_top_k():
    proxy, _client = make_single_proxy(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 8}',
            '{"bundle_value": 6}',
        ],
        items=("A", "B", "C"),
    )
    proxy.infer_cached_xor_bid(
        [
            frozenset({"A"}),
            frozenset({"B"}),
            frozenset({"C"}),
        ]
    )

    response = proxy.clock_demand_from_cached_bid(
        {"A": 0.0, "B": 0.0, "C": 0.0},
        top_k=2,
    )

    # supplementary_atoms is the entire cached bid, not capped by top_k, so
    # the final WDP+VCG resolution sees everything this proxy has elicited.
    assert {atom.bundle for atom in response.supplementary_atoms} == {
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
    }


def test_clock_demand_from_cached_bid_returns_full_bid_without_surplus():
    proxy, _client = make_single_proxy(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 5}',
        ]
    )
    proxy.infer_cached_xor_bid(
        [
            frozenset({"A"}),
            frozenset({"B"}),
        ]
    )

    response = proxy.clock_demand_from_cached_bid(
        {"A": 10.0, "B": 5.0}
    )

    assert response.primary_bundle is None
    # supplementary_atoms still returns the full cached bid even with no
    # positive-surplus primary demand at these prices.
    assert {atom.bundle for atom in response.supplementary_atoms} == {
        frozenset({"A"}),
        frozenset({"B"}),
    }


def test_run_clock_llm_proxy_experiment():
    instance, proxies, clients, candidates = make_clock_setup()
    result = run_clock_llm_proxy_experiment(
        instance,
        proxies,
        candidates,
        ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        top_k=1,
    )

    assert result.mechanism == "clock_llm_proxy_vcg_top_1"
    assert result.welfare >= 0.0
    assert result.rounds is not None
    assert result.query_count == 4 + result.rounds * 2
    assert result.metadata["candidate_bundle_count"] == 4
    assert result.metadata["candidate_bundle_count_by_bidder"] == {
        "i1": 2,
        "i2": 2,
    }
    assert result.metadata["targeted_candidate_bundles"] is False
    assert result.metadata["use_anchor_values"] is True
    assert result.metadata["elicited"] is False
    assert result.metadata["refinement_query_count_by_bidder"] == {
        "i1": 0,
        "i2": 0,
    }
    assert result.metadata["top_k"] == 1
    assert sum(len(client.calls) for client in clients.values()) == 4


def test_run_elicited_clock_refines_and_counts_queries():
    instance = AuctionInstance(
        items=["A"],
        bidder_ids=["i1"],
        valuations={
            "i1": {frozenset({"A"}): 20.0},
        },
    )
    client = MockLlmClient(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 20}',
        ]
    )
    proxies = make_llm_proxies_for_instance(
        instance=instance,
        scenario_description="A one-item auction.",
        person_seeds={"i1": "Values A."},
        item_descriptions={"A": "Item A"},
        clients={"i1": client},
        epsilon=1.0,
    )

    result = run_clock_llm_proxy_experiment(
        instance,
        proxies,
        [frozenset({"A"})],
        ClockConfig(max_rounds=2, price_step=1.0, reserve=9.0),
        elicited=True,
        margin_threshold=2.0,
        tie_threshold=0.0,
        max_refinement_queries_per_bidder=1,
    )

    assert result.mechanism == "elicited_clock_llm_proxy_vcg_top_1"
    assert result.rounds == 1
    assert result.query_count == 3
    assert result.metadata["elicited"] is True
    assert result.metadata["margin_threshold"] == 2.0
    assert result.metadata["tie_threshold"] == 0.0
    assert result.metadata["max_refinement_queries_per_bidder"] == 1
    assert result.metadata["refinement_query_count_by_bidder"] == {
        "i1": 1,
    }
    assert len(client.calls) == 2


def test_clock_llm_proxy_experiment_uses_bidder_specific_candidates():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"A"}): 10.0},
            "i2": {frozenset({"B"}): 9.0},
        },
    )
    clients = {
        "i1": MockLlmClient(['{"bundle_value": 10}']),
        "i2": MockLlmClient(
            [
                '{"bundle_value": 2}',
                '{"bundle_value": 9}',
            ]
        ),
    }
    proxies = make_llm_proxies_for_instance(
        instance=instance,
        scenario_description="A two-item auction.",
        person_seeds={"i1": "Prefers A.", "i2": "Prefers B."},
        item_descriptions={"A": "Item A", "B": "Item B"},
        clients=clients,
        epsilon=1.0,
    )

    result = run_clock_llm_proxy_experiment(
        instance,
        proxies,
        cfg=ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        candidate_bundles_by_bidder={
            "i1": [frozenset({"A"})],
            "i2": [
                frozenset({"A"}),
                frozenset({"B"}),
            ],
        },
        top_k=1,
    )

    assert result.rounds is not None
    assert result.query_count == 3 + result.rounds * 2
    assert result.metadata["candidate_bundle_count"] == 3
    assert result.metadata["candidate_bundle_count_by_bidder"] == {
        "i1": 1,
        "i2": 2,
    }
    assert result.metadata["targeted_candidate_bundles"] is True
    assert sum(len(client.calls) for client in clients.values()) == 3


def test_clock_llm_proxy_experiment_can_disable_anchor_values():
    instance, proxies, clients, _candidates = make_clock_setup()
    candidates = [
        frozenset({"A"}),
        frozenset({"A", "B"}),
    ]

    result = run_clock_llm_proxy_experiment(
        instance,
        proxies,
        candidates,
        ClockConfig(max_rounds=2, price_step=1.0, reserve=0.0),
        top_k=1,
        use_anchor_values=False,
    )

    assert result.rounds is not None
    assert result.query_count == 4 + result.rounds * 2
    assert result.metadata["use_anchor_values"] is False
    assert all(
        "ANCHOR_VALUES" not in prompt
        for client in clients.values()
        for prompt in client.calls
    )


def test_run_clock_llm_comparison_row_has_expected_fields():
    instance, proxies, _clients, candidates = make_clock_setup()
    comparison = run_clock_llm_comparison(
        instance=instance,
        instance_name="clock_llm",
        proxies=proxies,
        candidate_bundles=candidates,
        cfg=ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        top_k=1,
    )

    row = clock_llm_comparison_to_row(comparison)

    assert "efficiency" in row
    assert "clock_llm_reported_welfare" in row
    assert "clock_llm_true_welfare" in row
    assert "clock_llm_reported_bids" in row
    assert "clock_rounds" in row
    assert row["top_k"] == 1
    assert row["refinement_strategy"] == "value_query"
    assert row["efficiency"] == (
        row["clock_llm_true_welfare"] / row["full_info_welfare"]
    )
