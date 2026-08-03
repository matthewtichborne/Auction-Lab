from __future__ import annotations

import pytest

from auctionlab.auction_types import Bundle
from auctionlab.bids.xor import XorAtomicBid
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmInferredXorProxy


ITEM_DESCRIPTIONS = {
    "A": "Item A",
    "B": "Item B",
    "C": "Item C",
}


def make_proxy(
    responses: list[str],
    *,
    epsilon: float = 0.75,
    size_discount_family: str | None = None,
    size_discount_k0: int = 3,
    size_discount_gamma: float = 1.0,
) -> LlmInferredXorProxy:
    person = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A test auction.",
        person_seed="Values useful combinations.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=MockLlmClient(responses),
    )
    return LlmInferredXorProxy(
        bidder_id="bidder_1",
        person=person,
        epsilon=epsilon,
        size_discount_family=size_discount_family,
        size_discount_k0=size_discount_k0,
        size_discount_gamma=size_discount_gamma,
    )


def test_infer_xor_bid_discounts_values_and_records_transcript():
    proxy = make_proxy(
        [
            '{"bundle_value": 100, "confidence": 0.9, '
            '"reasoning_summary": "A"}',
            '{"bundle_value": 200, "confidence": 0.9, '
            '"reasoning_summary": "AB"}',
        ]
    )

    bid = proxy.infer_xor_bid(
        [
            frozenset({"A"}),
            frozenset({"A", "B"}),
        ]
    )

    assert bid.bidder_id == "bidder_1"
    assert bid.atoms == [
        XorAtomicBid(bundle=frozenset({"A"}), value=75.0),
        XorAtomicBid(bundle=frozenset({"A", "B"}), value=150.0),
    ]
    assert len(proxy.transcript) == 2
    assert proxy.transcript[0].kind == "value_query"


def test_deterministic_refinement_replaces_scaled_pv_with_unscaled_truth():
    bundle = frozenset({"A", "B", "C", "D"})
    person = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A test auction.",
        person_seed="Values the complete set.",
        item_descriptions={item: item for item in bundle},
        client=MockLlmClient([]),
        ground_truth_valuations={bundle: 1000.0},
    )
    proxy = LlmInferredXorProxy(
        bidder_id="bidder_1",
        person=person,
        epsilon=1.0,
        size_discount_family="exponential",
        size_discount_k0=3,
        size_discount_gamma=0.9,
    )
    proxy.replay_elicitation(
        nl_question="What do you want?",
        nl_answer="The complete set.",
        provisional_raw_values={bundle: 1000.0},
        discount_inferred=True,
    )

    assert proxy._cached_bid is not None
    assert proxy._cached_bid.value_of(bundle) == pytest.approx(900.0)
    refined = proxy.refine_bundle_value(bundle, "provisional allocation")

    assert refined == pytest.approx(1000.0)
    assert proxy._cached_bid.value_of(bundle) == pytest.approx(1000.0)


def test_set_provisional_bid_raises_superset_below_subset():
    proxy = make_proxy([])

    # {A,B} is a strict superset of {A}, but PV reported {A} as worth more.
    # Free disposal requires u({A,B}) >= u({A}), so the superset atom is
    # raised to match the subset -- the subset value is preserved.
    proxy.set_provisional_bid(
        {
            frozenset({"A"}): 50.0,
            frozenset({"A", "B"}): 30.0,
        }
    )

    values = {atom.bundle: atom.value for atom in proxy._cached_bid.atoms}
    assert values[frozenset({"A"})] == 50.0
    assert values[frozenset({"A", "B"})] == 50.0


def test_refine_bundle_value_raises_superset_below_subset():
    # When a subset is refined to a higher value than a cached superset,
    # the superset is raised to preserve free disposal.  The returned value
    # is the refined subset's own value (1750), not the old superset value.
    proxy = make_proxy(['{"bundle_value": 1750}'], epsilon=1.0)
    proxy.set_provisional_bid(
        {
            frozenset({"A", "B", "C"}): 1700.0,
        }
    )

    reported = proxy.refine_bundle_value(frozenset({"A", "B"}), "test refinement")

    assert reported == 1750.0
    values = {atom.bundle: atom.value for atom in proxy._cached_bid.atoms}
    assert values[frozenset({"A", "B"})] == 1750.0
    assert values[frozenset({"A", "B", "C"})] == 1750.0


def test_refine_bundle_value_does_not_clamp_consistent_values():
    proxy = make_proxy(['{"bundle_value": 1450}'], epsilon=1.0)
    proxy.set_provisional_bid(
        {
            frozenset({"A", "B", "C"}): 1700.0,
        }
    )

    reported = proxy.refine_bundle_value(frozenset({"A", "B"}), "test refinement")

    assert reported == 1450.0
    values = {atom.bundle: atom.value for atom in proxy._cached_bid.atoms}
    assert values[frozenset({"A", "B"})] == 1450.0
    assert values[frozenset({"A", "B", "C"})] == 1700.0


def test_infer_xor_bid_can_leave_values_undiscounted():
    proxy = make_proxy(['{"bundle_value": 100}'])

    bid = proxy.infer_xor_bid(
        [frozenset({"A"})],
        discount_inferred=False,
    )

    assert bid.atoms == [
        XorAtomicBid(bundle=frozenset({"A"}), value=100.0)
    ]


def test_infer_xor_bid_skips_empty_bundle():
    proxy = make_proxy(['{"bundle_value": 100}'])

    bid = proxy.infer_xor_bid(
        [
            frozenset(),
            frozenset({"A"}),
        ]
    )

    assert bid.atoms == [
        XorAtomicBid(bundle=frozenset({"A"}), value=75.0)
    ]
    assert len(proxy.person.client.calls) == 1


@pytest.mark.parametrize("epsilon", [0.0, -0.1, 1.1])
def test_proxy_rejects_invalid_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        make_proxy([], epsilon=epsilon)


# ---------------------------------------------------------------------------
# Exponential size discount (adjusted = raw * epsilon * gamma**max(0, |B|-k0))
# ---------------------------------------------------------------------------

class TestExponentialSizeDiscount:
    def test_infer_xor_bid_applies_discount_above_k0(self):
        proxy = make_proxy(
            ['{"bundle_value": 100}', '{"bundle_value": 100}'],
            epsilon=1.0,
            size_discount_family="exponential",
            size_discount_k0=1,
            size_discount_gamma=0.9,
        )

        # {A} (size 1) is not a subset of {B, C} (size 2), so free-disposal
        # monotonicity repair cannot interfere with either value.
        bid = proxy.infer_xor_bid([frozenset({"A"}), frozenset({"B", "C"})])

        values = {atom.bundle: atom.value for atom in bid.atoms}
        assert values[frozenset({"A"})] == 100.0
        assert values[frozenset({"B", "C"})] == pytest.approx(100.0 * 0.9)

    def test_size_discount_noop_when_family_none(self):
        proxy = make_proxy(['{"bundle_value": 100}'], epsilon=1.0, size_discount_family=None)
        bid = proxy.infer_xor_bid([frozenset({"B", "C"})])
        assert bid.atoms == [XorAtomicBid(bundle=frozenset({"B", "C"}), value=100.0)]

    def test_replay_elicitation_composes_epsilon_then_size_discount(self):
        proxy = make_proxy(
            [],
            epsilon=0.5,
            size_discount_family="exponential",
            size_discount_k0=1,
            size_discount_gamma=0.9,
        )
        proxy.replay_elicitation(
            nl_question="Q",
            nl_answer="A",
            provisional_raw_values={
                frozenset({"A"}): 100.0,
                frozenset({"B", "C"}): 100.0,
            },
            discount_inferred=True,
        )

        values = {atom.bundle: atom.value for atom in proxy._cached_bid.atoms}
        # size 1 <= k0=1: epsilon only.
        assert values[frozenset({"A"})] == 50.0
        # size 2 > k0=1: epsilon * gamma**(2-1).
        assert values[frozenset({"B", "C"})] == pytest.approx(100.0 * 0.5 * 0.9)

    def test_replay_elicitation_ignores_size_discount_when_discount_inferred_false(self):
        proxy = make_proxy(
            [],
            epsilon=0.5,
            size_discount_family="exponential",
            size_discount_k0=0,
            size_discount_gamma=0.1,
        )
        proxy.replay_elicitation(
            nl_question="Q",
            nl_answer="A",
            provisional_raw_values={frozenset({"A", "B", "C"}): 100.0},
            discount_inferred=False,
        )
        values = {atom.bundle: atom.value for atom in proxy._cached_bid.atoms}
        assert values[frozenset({"A", "B", "C"})] == 100.0

    @pytest.mark.parametrize("gamma", [0.0, -0.1])
    def test_proxy_rejects_invalid_size_discount_gamma(self, gamma):
        with pytest.raises(ValueError, match="size_discount_gamma"):
            make_proxy([], size_discount_family="exponential", size_discount_gamma=gamma)

    def test_proxy_rejects_unknown_size_discount_family(self):
        with pytest.raises(ValueError, match="size_discount_family"):
            make_proxy([], size_discount_family="linear")


def test_infer_values_returns_bundle_value_mapping():
    proxy = make_proxy(
        [
            '{"bundle_value": 40}',
            '{"bundle_value": 80}',
        ],
        epsilon=0.5,
    )

    values = proxy.infer_values(
        [
            frozenset({"A"}),
            frozenset({"A", "B"}),
        ]
    )

    assert values == {
        frozenset({"A"}): 20.0,
        frozenset({"A", "B"}): 40.0,
    }


def test_inference_queries_singletons_first_and_uses_raw_anchors():
    proxy = make_proxy(
        [
            '{"bundle_value": 100}',
            '{"bundle_value": 80}',
            '{"bundle_value": 60}',
            '{"bundle_value": 220}',
            '{"bundle_value": 300}',
        ],
        epsilon=0.5,
    )

    values = proxy.infer_values(
        [
            frozenset({"A", "B", "C"}),
            frozenset({"A", "B"}),
            frozenset({"C"}),
            frozenset({"B"}),
            frozenset({"A"}),
        ]
    )
    prompts = proxy.person.client.calls

    assert "PROPOSED_BUNDLE_ITEM_IDS = ['A']" in prompts[0]
    assert "PROPOSED_BUNDLE_ITEM_IDS = ['B']" in prompts[1]
    assert "PROPOSED_BUNDLE_ITEM_IDS = ['C']" in prompts[2]
    assert "PROPOSED_BUNDLE_ITEM_IDS = ['A', 'B']" in prompts[3]
    assert "PROPOSED_BUNDLE_ITEM_IDS = ['A', 'B', 'C']" in prompts[4]
    assert "ANCHOR_VALUES" not in prompts[0]
    assert "- [A]: 100.0" in prompts[3]
    assert "- [B]: 80.0" in prompts[3]
    assert "- [C]: 60.0" not in prompts[3]
    assert "- [A]: 100.0" in prompts[4]
    assert "- [B]: 80.0" in prompts[4]
    assert "- [C]: 60.0" in prompts[4]
    assert values == {
        frozenset({"A"}): 50.0,
        frozenset({"B"}): 40.0,
        frozenset({"C"}): 30.0,
        frozenset({"A", "B"}): 110.0,
        frozenset({"A", "B", "C"}): 150.0,
    }
    assert len(prompts) == 5
    assert proxy.transcript[3].kind == "value_query_with_anchors"
    assert "[A]: 100.0" in proxy.transcript[3].content
    assert "[B]: 80.0" in proxy.transcript[3].content


def test_inference_can_disable_anchor_values_without_changing_query_order():
    proxy = make_proxy(
        [
            '{"bundle_value": 100}',
            '{"bundle_value": 80}',
            '{"bundle_value": 220}',
        ],
        epsilon=1.0,
    )

    values = proxy.infer_values(
        [
            frozenset({"A", "B"}),
            frozenset({"B"}),
            frozenset({"A"}),
        ],
        use_anchor_values=False,
    )
    prompts = proxy.person.client.calls

    assert "PROPOSED_BUNDLE_ITEM_IDS = ['A']" in prompts[0]
    assert "PROPOSED_BUNDLE_ITEM_IDS = ['B']" in prompts[1]
    assert "PROPOSED_BUNDLE_ITEM_IDS = ['A', 'B']" in prompts[2]
    assert all("ANCHOR_VALUES" not in prompt for prompt in prompts)
    assert values == {
        frozenset({"A"}): 100.0,
        frozenset({"B"}): 80.0,
        frozenset({"A", "B"}): 220.0,
    }
    assert len(prompts) == 3
    assert all(entry.kind == "value_query" for entry in proxy.transcript)


def test_infer_xor_bid_returns_empty_atoms_for_no_candidates():
    proxy = make_proxy([])

    bid = proxy.infer_xor_bid([])

    assert bid.bidder_id == "bidder_1"
    assert bid.atoms == []
    assert proxy.transcript == []


def test_refine_bundle_value_updates_cached_atom_with_one_discount():
    proxy = make_proxy(
        [
            '{"bundle_value": 100}',
            '{"bundle_value": 200}',
            '{"bundle_value": 300}',
        ],
        epsilon=0.5,
    )
    cached_bid = proxy.infer_cached_xor_bid(
        [
            frozenset({"A"}),
            frozenset({"A", "B"}),
        ]
    )

    reported_value = proxy.refine_bundle_value(
        frozenset({"A", "B"}),
        "near-zero surplus",
    )

    assert reported_value == 150.0
    assert cached_bid.atoms == [
        XorAtomicBid(bundle=frozenset({"A"}), value=50.0),
        XorAtomicBid(bundle=frozenset({"A", "B"}), value=150.0),
    ]
    assert "- [A]: 100.0" in proxy.person.client.calls[-1]
    assert proxy.refinement_query_count == 1
    assert proxy.refined_bundles == {frozenset({"A", "B"})}
    assert proxy.transcript[-1].kind == "refinement_value_query"


def test_refine_bundle_value_adds_missing_atom():
    proxy = make_proxy(
        [
            '{"bundle_value": 100}',
            '{"bundle_value": 250}',
        ],
        epsilon=1.0,
    )
    cached_bid = proxy.infer_cached_xor_bid([frozenset({"A"})])

    proxy.refine_bundle_value(
        frozenset({"A", "B"}),
        "new clock candidate",
    )

    assert cached_bid.atoms[-1] == XorAtomicBid(
        bundle=frozenset({"A", "B"}),
        value=250.0,
    )


def test_ranked_surplus_atoms_includes_non_positive_atoms():
    # {A,B} is intentionally valued at the same level as each singleton (not
    # below, which monotonicity enforcement would now clamp away) so this
    # stays a meaningful check of non-positive-surplus inclusion/ordering.
    proxy = make_proxy(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 10}',
            '{"bundle_value": 10}',
        ],
        epsilon=1.0,
    )
    proxy.infer_cached_xor_bid(
        [
            frozenset({"A"}),
            frozenset({"B"}),
            frozenset({"A", "B"}),
        ]
    )

    ranked = proxy.ranked_surplus_atoms({"A": 10.0, "B": 10.0})

    assert [(atom.bundle, surplus) for atom, surplus in ranked] == [
        (frozenset({"A"}), 0.0),
        (frozenset({"B"}), 0.0),
        (frozenset({"A", "B"}), -10.0),
    ]


def test_clock_refinement_triggers_near_zero_and_respects_budget():
    proxy = make_proxy(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 9.5}',
            '{"bundle_value": 20}',
        ],
        epsilon=1.0,
    )
    proxy.infer_cached_xor_bid(
        [
            frozenset({"A"}),
            frozenset({"B"}),
        ]
    )
    proxy.reset_refinement_state()

    response = proxy.clock_demand_with_refinement(
        prices={"A": 9.0, "B": 9.0},
        top_k=1,
        round_idx=0,
        previous_primary_bundle=None,
        margin_threshold=2.0,
        tie_threshold=2.0,
        max_refinement_queries=1,
    )

    assert response.primary_bundle == frozenset({"A"})
    assert proxy.refinement_query_count == 1
    assert len(proxy.person.client.calls) == 3


def test_clock_refinement_triggers_when_primary_demand_changes():
    proxy = make_proxy(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 9}',
            '{"bundle_value": 12}',
        ],
        epsilon=1.0,
    )
    proxy.infer_cached_xor_bid(
        [
            frozenset({"A"}),
            frozenset({"B"}),
        ]
    )
    proxy.reset_refinement_state()

    response = proxy.clock_demand_with_refinement(
        prices={"A": 3.0, "B": 0.0},
        top_k=1,
        round_idx=1,
        previous_primary_bundle=frozenset({"A"}),
        margin_threshold=0.0,
        tie_threshold=0.0,
        max_refinement_queries=1,
    )

    assert response.primary_bundle == frozenset({"A"})
    assert proxy.refinement_query_count == 1
    assert proxy.refined_bundles == {frozenset({"A"})}
    assert "primary demand changed" in proxy.transcript[-1].content


def test_clock_demand_with_refinement_priority_scoring_respects_budget():
    """With budget=1 and priority_scoring=True, only one refinement fires."""
    proxy = make_proxy(
        [
            '{"bundle_value": 10}',   # initial A
            '{"bundle_value": 9.5}',  # initial B
            '{"bundle_value": 12}',   # refinement (whichever wins priority)
        ],
        epsilon=1.0,
    )
    proxy.infer_cached_xor_bid([frozenset({"A"}), frozenset({"B"})])
    proxy.reset_refinement_state()

    # Both bundles trigger near_zero_surplus (prices below values but within
    # margin); budget=1 means only one refinement should fire.
    proxy.clock_demand_with_refinement(
        prices={"A": 9.0, "B": 9.0},
        top_k=1,
        round_idx=0,
        previous_primary_bundle=None,
        margin_threshold=2.0,
        tie_threshold=2.0,
        max_refinement_queries=1,
        priority_scoring=True,
    )

    assert proxy.refinement_query_count == 1


def test_clock_demand_with_refinement_uses_clock_round_prefix_in_reason():
    """Reason strings should use the 'clock round N:' prefix from the shared module."""
    proxy = make_proxy(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 12}',
        ],
        epsilon=1.0,
    )
    proxy.infer_cached_xor_bid([frozenset({"A"})])
    proxy.reset_refinement_state()

    proxy.clock_demand_with_refinement(
        prices={"A": 9.5},
        top_k=1,
        round_idx=7,
        previous_primary_bundle=None,
        margin_threshold=1.0,
        tie_threshold=0.0,
        max_refinement_queries=1,
    )

    assert "clock round 7" in proxy.transcript[-1].content


def test_ask_initial_question_records_nl_transcript_and_feeds_inference():
    proxy = make_proxy(
        [
            '{"question": "What will you use these items for?"}',
            '{"answer": "Mostly for sketching on the go."}',
            '{"bundle_value": 100}',
        ]
    )

    proxy.ask_initial_question()

    assert proxy.nl_transcript == [
        ("What will you use these items for?", "Mostly for sketching on the go.")
    ]
    assert [entry.kind for entry in proxy.transcript[:2]] == [
        "nl_question",
        "nl_answer",
    ]

    proxy.infer_xor_bid([frozenset({"A"})])
    value_query_prompt = proxy.person.client.calls[-1]

    assert "PRIOR_PREFERENCE_QA:" in value_query_prompt
    assert "What will you use these items for?" in value_query_prompt
    assert "Mostly for sketching on the go." in value_query_prompt


def test_refine_via_demand_query_satisfied_leaves_bid_unchanged():
    proxy = make_proxy(
        [
            '{"bundle_value": 100}',
            '{"satisfied": true, "preferred_bundle": null}',
        ],
        epsilon=1.0,
    )
    cached_bid = proxy.infer_cached_xor_bid([frozenset({"A"})])

    revalued_bundle, value = proxy.refine_via_demand_query(
        frozenset({"A"}),
        prices={"A": 50.0},
        reason="near-zero surplus",
    )

    assert revalued_bundle is None
    assert value is None
    assert cached_bid.atoms == [XorAtomicBid(bundle=frozenset({"A"}), value=100.0)]
    assert proxy.refinement_query_count == 1
    assert frozenset({"A"}) in proxy.refined_bundles
    assert proxy.transcript[-1].kind == "demand_query"
    assert "satisfied=True" in proxy.transcript[-1].content


def test_refine_via_demand_query_unsatisfied_with_alternative_revalues_alternative():
    proxy = make_proxy(
        [
            '{"bundle_value": 100}',
            '{"satisfied": false, "preferred_bundle": ["B"]}',
            '{"bundle_value": 250}',
        ],
        epsilon=1.0,
    )
    cached_bid = proxy.infer_cached_xor_bid([frozenset({"A"})])

    revalued_bundle, value = proxy.refine_via_demand_query(
        frozenset({"A"}),
        prices={"A": 50.0},
        reason="near-zero surplus",
    )

    assert revalued_bundle == frozenset({"B"})
    assert value == 250.0
    assert XorAtomicBid(bundle=frozenset({"B"}), value=250.0) in cached_bid.atoms
    assert frozenset({"A"}) in proxy.refined_bundles
    assert frozenset({"B"}) in proxy.refined_bundles
    assert proxy.transcript[-1].kind == "refinement_demand_query"


def test_refine_via_demand_query_unsatisfied_without_alternative_falls_back():
    proxy = make_proxy(
        [
            '{"bundle_value": 100}',
            '{"satisfied": false, "preferred_bundle": null}',
            '{"bundle_value": 80}',
        ],
        epsilon=1.0,
    )
    cached_bid = proxy.infer_cached_xor_bid([frozenset({"A"})])

    revalued_bundle, value = proxy.refine_via_demand_query(
        frozenset({"A"}),
        prices={"A": 50.0},
        reason="near-zero surplus",
    )

    assert revalued_bundle == frozenset({"A"})
    assert value == 80.0
    assert cached_bid.atoms == [XorAtomicBid(bundle=frozenset({"A"}), value=80.0)]
    assert proxy.transcript[-1].kind == "refinement_value_query"


def test_clock_demand_with_refinement_demand_query_strategy():
    proxy = make_proxy(
        [
            '{"bundle_value": 10}',
            '{"bundle_value": 9.5}',
            '{"satisfied": false, "preferred_bundle": null}',
            '{"bundle_value": 8}',
        ],
        epsilon=1.0,
    )
    proxy.infer_cached_xor_bid([frozenset({"A"}), frozenset({"B"})])
    proxy.reset_refinement_state()

    response = proxy.clock_demand_with_refinement(
        prices={"A": 9.0, "B": 9.0},
        top_k=1,
        round_idx=0,
        previous_primary_bundle=None,
        margin_threshold=2.0,
        tie_threshold=2.0,
        max_refinement_queries=1,
        refinement_strategy="demand_query",
    )

    assert response.primary_bundle == frozenset({"B"})
    assert proxy.transcript[-2].kind == "demand_query"
    assert proxy.transcript[-1].kind == "refinement_value_query"


def test_ask_initial_question_is_idempotent():
    proxy = make_proxy(
        [
            '{"question": "What will you use these items for?"}',
            '{"answer": "Mostly for sketching on the go."}',
        ]
    )

    proxy.ask_initial_question()
    proxy.ask_initial_question()

    assert len(proxy.nl_transcript) == 1
    assert len(proxy.person.client.calls) == 2


def test_revalue_and_upsert_atom_captures_query_text_in_transcript():
    proxy = make_proxy(
        [
            '{"bundle_value": 100}',
            '{"bundle_value": 200, "reasoning_summary": "reconsidered"}',
        ],
        epsilon=1.0,
    )
    proxy.infer_cached_xor_bid([frozenset({"A"})])

    proxy.refine_bundle_value(frozenset({"A"}), "near-zero surplus")

    refinement_entry = proxy.transcript[-1]
    assert refinement_entry.kind == "refinement_value_query"
    assert refinement_entry.query_text is not None
    assert "PROPOSED_BUNDLE_ITEM_IDS" in refinement_entry.query_text
    assert refinement_entry.response_summary == "reconsidered"


def test_proxy_last_refinement_fields_set_after_refine_bundle_value():
    proxy = make_proxy(
        [
            '{"bundle_value": 100}',
            '{"bundle_value": 200, "reasoning_summary": "bundle is valuable"}',
        ],
        epsilon=1.0,
    )
    proxy.infer_cached_xor_bid([frozenset({"A"})])
    assert proxy._last_refinement_query_text is None

    proxy.refine_bundle_value(frozenset({"A"}), "test reason")

    assert proxy._last_refinement_query_text is not None
    assert proxy._last_refinement_response_summary == "bundle is valuable"


def test_demand_query_transcript_entry_has_query_text():
    proxy = make_proxy(
        [
            '{"bundle_value": 100}',
            '{"satisfied": true, "preferred_bundle": null}',
        ],
        epsilon=1.0,
    )
    proxy.infer_cached_xor_bid([frozenset({"A"})])

    proxy.refine_via_demand_query(
        frozenset({"A"}), prices={"A": 50.0, "B": 0.0}, reason="near-zero surplus"
    )

    demand_entry = proxy.transcript[-1]
    assert demand_entry.kind == "demand_query"
    assert demand_entry.query_text is not None
    assert "satisfied=True" in demand_entry.response_summary


def test_refine_bundle_value_passes_reason_as_elicitation_context_to_prompt():
    proxy = make_proxy(
        [
            '{"bundle_value": 100}',
            '{"bundle_value": 80}',
        ],
        epsilon=1.0,
    )
    proxy.infer_cached_xor_bid([frozenset({"A"})])

    proxy.refine_bundle_value(
        frozenset({"A"}),
        "clock round 3: best surplus near zero",
    )

    refinement_prompt = proxy.transcript[-1].query_text
    assert "ELICITATION_CONTEXT:" in refinement_prompt
    assert "best surplus near zero" in refinement_prompt


def test_initial_value_query_does_not_include_elicitation_context():
    proxy = make_proxy(['{"bundle_value": 100}'], epsilon=1.0)

    proxy.infer_xor_bid([frozenset({"A"})])

    assert "ELICITATION_CONTEXT:" not in proxy.person.client.calls[0]


# ---------------------------------------------------------------------------
# SummaryKnowledgeBase integration
# ---------------------------------------------------------------------------


def _make_proxy_with_summary_kb(
    person_responses: list[str],
    summary_responses: list[str],
    *,
    epsilon: float = 1.0,
) -> LlmInferredXorProxy:
    from auctionlab.llm.knowledge import SummaryKnowledgeBase

    person = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A test auction.",
        person_seed="Values useful combinations.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=MockLlmClient(person_responses),
    )
    kb = SummaryKnowledgeBase(client=MockLlmClient(summary_responses))
    return LlmInferredXorProxy(
        bidder_id="bidder_1",
        person=person,
        epsilon=epsilon,
        knowledge_base=kb,
    )


def test_proxy_accepts_summary_knowledge_base():
    from auctionlab.llm.knowledge import KnowledgeBase, SummaryKnowledgeBase

    proxy = _make_proxy_with_summary_kb([], [])

    assert isinstance(proxy.knowledge_base, SummaryKnowledgeBase)
    assert isinstance(proxy.knowledge_base, KnowledgeBase)


def test_summary_kb_proxy_is_empty_by_default():
    proxy = _make_proxy_with_summary_kb([], [])

    assert not proxy.knowledge_base
    assert proxy.nl_transcript == []


def test_summary_kb_proxy_ask_initial_question_builds_summary():
    proxy = _make_proxy_with_summary_kb(
        person_responses=[
            '{"question": "What matters most?"}',
            '{"answer": "Speed above all."}',
        ],
        summary_responses=['{"summary": "Person values speed above all."}'],
    )

    proxy.ask_initial_question()

    assert proxy.knowledge_base
    assert proxy.knowledge_base.summary == "Person values speed above all."


def test_summary_kb_proxy_ask_initial_question_is_idempotent():
    """Calling ask_initial_question twice must not issue a second LLM call."""
    proxy = _make_proxy_with_summary_kb(
        person_responses=[
            '{"question": "Q?"}',
            '{"answer": "A."}',
        ],
        summary_responses=['{"summary": "S."}'],
    )

    proxy.ask_initial_question()
    proxy.ask_initial_question()  # second call should be a no-op

    assert proxy.knowledge_base.summary == "S."
    # Only 2 person calls: one for question generation, one for answer
    assert len(proxy.person.client.calls) == 2


def test_summary_kb_context_appears_in_value_query_prompt():
    """After ask_initial_question, PREFERENCE_SUMMARY must flow into prompts."""
    proxy = _make_proxy_with_summary_kb(
        person_responses=[
            '{"question": "Q?"}',
            '{"answer": "A."}',
            '{"bundle_value": 50}',
        ],
        summary_responses=['{"summary": "Values portability."}'],
    )

    proxy.ask_initial_question()
    proxy.infer_xor_bid([frozenset({"A"})])

    value_query_prompt = proxy.person.client.calls[-1]
    assert "PREFERENCE_SUMMARY:" in value_query_prompt
    assert "Values portability." in value_query_prompt


def test_summary_kb_does_not_write_to_nl_transcript():
    """SummaryKnowledgeBase keeps its own state; nl_transcript stays empty."""
    proxy = _make_proxy_with_summary_kb(
        person_responses=[
            '{"question": "Q?"}',
            '{"answer": "A."}',
        ],
        summary_responses=['{"summary": "S."}'],
    )

    proxy.ask_initial_question()

    assert proxy.nl_transcript == []


def test_default_proxy_still_uses_transcript_knowledge_base():
    """Omitting knowledge_base must preserve the TranscriptKnowledgeBase default."""
    from auctionlab.llm.knowledge import TranscriptKnowledgeBase

    proxy = make_proxy([])

    assert isinstance(proxy.knowledge_base, TranscriptKnowledgeBase)


def test_make_llm_proxies_for_instance_with_knowledge_base_factory(toy_instance):
    from auctionlab.experiments.llm_runner import make_llm_proxies_for_instance
    from auctionlab.llm.knowledge import SummaryKnowledgeBase

    summary_client = MockLlmClient([])

    def kb_factory(bidder_id: str) -> SummaryKnowledgeBase:
        return SummaryKnowledgeBase(client=summary_client)

    proxies = make_llm_proxies_for_instance(
        instance=toy_instance,
        scenario_description="Test auction.",
        person_seeds={bid: "seed" for bid in toy_instance.bidder_ids},
        item_descriptions={item: f"Item {item}" for item in toy_instance.items},
        clients={bid: MockLlmClient([]) for bid in toy_instance.bidder_ids},
        knowledge_base_factory=kb_factory,
    )

    for bidder_id, proxy in proxies.items():
        assert isinstance(proxy.knowledge_base, SummaryKnowledgeBase), (
            f"bidder {bidder_id} did not get SummaryKnowledgeBase"
        )


def test_make_llm_proxies_for_instance_default_still_transcript(toy_instance):
    from auctionlab.experiments.llm_runner import make_llm_proxies_for_instance
    from auctionlab.llm.knowledge import TranscriptKnowledgeBase

    proxies = make_llm_proxies_for_instance(
        instance=toy_instance,
        scenario_description="Test auction.",
        person_seeds={bid: "seed" for bid in toy_instance.bidder_ids},
        item_descriptions={item: f"Item {item}" for item in toy_instance.items},
        clients={bid: MockLlmClient([]) for bid in toy_instance.bidder_ids},
    )

    for proxy in proxies.values():
        assert isinstance(proxy.knowledge_base, TranscriptKnowledgeBase)
