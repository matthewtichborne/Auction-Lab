from __future__ import annotations

import pytest

from auctionlab.experiments.llm_runner import (
    make_llm_proxies_for_instance,
    run_sealed_llm_proxy_experiment,
)
from auctionlab.instances.base import AuctionInstance
from auctionlab.llm.clients import MockLlmClient


@pytest.fixture
def two_bidder_instance() -> AuctionInstance:
    return AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"A"}): 10.0},
            "i2": {frozenset({"B"}): 9.0},
        },
    )


def make_proxies(
    instance: AuctionInstance,
    *,
    epsilon: float = 1.0,
):
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
            "i1": "Strongly prefers A.",
            "i2": "Strongly prefers B.",
        },
        item_descriptions={
            "A": "Item A",
            "B": "Item B",
        },
        clients=clients,
        epsilon=epsilon,
    )
    return proxies


def test_sealed_llm_proxy_experiment_returns_expected_outcome(
    two_bidder_instance,
):
    proxies = make_proxies(two_bidder_instance)

    result = run_sealed_llm_proxy_experiment(
        two_bidder_instance,
        proxies,
        [
            frozenset({"A"}),
            frozenset({"B"}),
        ],
    )

    assert result.mechanism == "sealed_llm_proxy_vcg"
    assert result.allocation == {
        "i1": frozenset({"A"}),
        "i2": frozenset({"B"}),
    }
    assert result.welfare == 19.0
    assert result.query_count == 4
    assert result.metadata["candidate_bundle_count"] == 4
    assert result.metadata["candidate_bundle_count_by_bidder"] == {
        "i1": 2,
        "i2": 2,
    }
    assert result.metadata["targeted_candidate_bundles"] is False
    assert result.metadata["use_anchor_values"] is True
    assert list(result.metadata["reported_bids"]) == ["i1", "i2"]
    assert result.metadata["reported_bids"]["i1"].bidder_id == "i1"


def test_sealed_llm_proxy_experiment_applies_discount(two_bidder_instance):
    result = run_sealed_llm_proxy_experiment(
        two_bidder_instance,
        make_proxies(two_bidder_instance, epsilon=0.5),
        [
            frozenset({"A"}),
            frozenset({"B"}),
        ],
    )

    assert result.welfare == 9.5
    assert result.metadata["discount_inferred"] is True


def test_sealed_llm_proxy_experiment_can_ignore_epsilon(two_bidder_instance):
    result = run_sealed_llm_proxy_experiment(
        two_bidder_instance,
        make_proxies(two_bidder_instance, epsilon=0.5),
        [
            frozenset({"A"}),
            frozenset({"B"}),
        ],
        discount_inferred=False,
    )

    assert result.welfare == 19.0
    assert result.metadata["discount_inferred"] is False


def test_sealed_llm_proxy_experiment_uses_bidder_specific_candidates(
    two_bidder_instance,
):
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
        instance=two_bidder_instance,
        scenario_description="A two-item auction.",
        person_seeds={"i1": "Prefers A.", "i2": "Prefers B."},
        item_descriptions={"A": "Item A", "B": "Item B"},
        clients=clients,
        epsilon=1.0,
    )

    result = run_sealed_llm_proxy_experiment(
        two_bidder_instance,
        proxies,
        candidate_bundles_by_bidder={
            "i1": [frozenset({"A"})],
            "i2": [
                frozenset({"A"}),
                frozenset({"B"}),
            ],
        },
    )

    assert result.query_count == 3
    assert result.metadata["candidate_bundle_count"] == 3
    assert result.metadata["candidate_bundle_count_by_bidder"] == {
        "i1": 1,
        "i2": 2,
    }
    assert result.metadata["targeted_candidate_bundles"] is True
    assert len(clients["i1"].calls) == 1
    assert len(clients["i2"].calls) == 2


def test_sealed_llm_proxy_experiment_can_disable_anchor_values(
    two_bidder_instance,
):
    clients = {
        "i1": MockLlmClient(
            [
                '{"bundle_value": 10}',
                '{"bundle_value": 15}',
            ]
        ),
        "i2": MockLlmClient(
            [
                '{"bundle_value": 9}',
                '{"bundle_value": 12}',
            ]
        ),
    }
    proxies = make_llm_proxies_for_instance(
        instance=two_bidder_instance,
        scenario_description="A two-item auction.",
        person_seeds={"i1": "Prefers A.", "i2": "Prefers B."},
        item_descriptions={"A": "Item A", "B": "Item B"},
        clients=clients,
        epsilon=1.0,
    )
    candidates = [
        frozenset({"A"}),
        frozenset({"A", "B"}),
    ]

    result = run_sealed_llm_proxy_experiment(
        two_bidder_instance,
        proxies,
        candidates,
        use_anchor_values=False,
    )

    assert result.query_count == 4
    assert result.metadata["use_anchor_values"] is False
    assert all(
        "ANCHOR_VALUES" not in prompt
        for client in clients.values()
        for prompt in client.calls
    )


def test_sealed_llm_proxy_experiment_rejects_missing_proxy(
    two_bidder_instance,
):
    proxies = make_proxies(two_bidder_instance)
    del proxies["i2"]

    with pytest.raises(ValueError, match=r"missing=\['i2'\]"):
        run_sealed_llm_proxy_experiment(
            two_bidder_instance,
            proxies,
            [frozenset({"A"})],
        )


def test_sealed_llm_proxy_experiment_rejects_extra_proxy(
    two_bidder_instance,
):
    proxies = make_proxies(two_bidder_instance)
    proxies["unknown"] = proxies["i1"]

    with pytest.raises(ValueError, match=r"extra=\['unknown'\]"):
        run_sealed_llm_proxy_experiment(
            two_bidder_instance,
            proxies,
            [frozenset({"A"})],
        )


def test_make_llm_proxies_for_instance_creates_one_proxy_per_bidder(
    two_bidder_instance,
):
    proxies = make_proxies(two_bidder_instance, epsilon=0.8)

    assert list(proxies) == ["i1", "i2"]
    assert proxies["i1"].bidder_id == "i1"
    assert proxies["i1"].person.bidder_id == "i1"
    assert proxies["i2"].bidder_id == "i2"
    assert proxies["i2"].person.bidder_id == "i2"
    assert proxies["i1"].epsilon == 0.8


@pytest.mark.parametrize("missing_mapping", ["person_seeds", "clients"])
def test_make_llm_proxies_for_instance_validates_missing_inputs(
    two_bidder_instance,
    missing_mapping,
):
    person_seeds = {
        "i1": "Prefers A.",
        "i2": "Prefers B.",
    }
    clients = {
        "i1": MockLlmClient([]),
        "i2": MockLlmClient([]),
    }

    if missing_mapping == "person_seeds":
        del person_seeds["i2"]
    else:
        del clients["i2"]

    with pytest.raises(ValueError, match=missing_mapping):
        make_llm_proxies_for_instance(
            instance=two_bidder_instance,
            scenario_description="A two-item auction.",
            person_seeds=person_seeds,
            item_descriptions={
                "A": "Item A",
                "B": "Item B",
            },
            clients=clients,
        )
