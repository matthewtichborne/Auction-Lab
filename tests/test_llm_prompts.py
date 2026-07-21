from __future__ import annotations

import pytest

from auctionlab.llm.prompts import (
    build_initial_proxy_question_prompt,
    build_person_answer_prompt,
    build_provisional_valuation_prompt,
    build_value_query_prompt,
    describe_bundle,
)


ITEM_DESCRIPTIONS = {
    "IPAD": "Apple iPad",
    "PENCIL": "Apple Pencil",
}


def test_describe_bundle_sorts_items_and_includes_descriptions():
    description = describe_bundle(
        frozenset({"PENCIL", "IPAD"}),
        ITEM_DESCRIPTIONS,
    )

    assert description.splitlines() == [
        "- IPAD: Apple iPad",
        "- PENCIL: Apple Pencil",
    ]


def test_describe_bundle_handles_empty_bundle():
    description = describe_bundle(frozenset(), ITEM_DESCRIPTIONS)

    assert "empty" in description.lower() or "no items" in description.lower()


def test_describe_bundle_rejects_unknown_item():
    with pytest.raises(ValueError, match="UNKNOWN"):
        describe_bundle(frozenset({"UNKNOWN"}), ITEM_DESCRIPTIONS)


def test_value_query_prompt_contains_context_steps_and_schema():
    prompt = build_value_query_prompt(
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle=frozenset({"IPAD", "PENCIL"}),
    )

    assert "A technology auction." in prompt
    assert "Prefers portable creative tools." in prompt
    assert "IPAD" in prompt
    assert "Apple iPad" in prompt
    assert "PENCIL" in prompt
    assert "Apple Pencil" in prompt
    for step in range(1, 6):
        assert f"{step}." in prompt
    assert '"bundle_value"' in prompt
    assert '"queried_bundle"' in prompt
    assert '"confidence"' in prompt
    assert '"reasoning_summary"' in prompt
    assert "ONLY the PROPOSED_BUNDLE" in prompt
    assert "PROPOSED_BUNDLE_ITEM_IDS" in prompt
    assert "Do not return the value of a larger bundle" in prompt
    assert "Do not assume bundle values are additive" in prompt
    assert "substitutes" in prompt
    assert "redundant" in prompt
    assert "complementary" in prompt
    assert "do not simply add singleton values" in prompt
    assert "Do not include markdown fences" in prompt


def test_value_query_prompt_gates_synergy_on_named_hub_items():
    # Regression guard against inventing complementary premiums for any
    # combination of items the person merely likes individually: the prompt
    # must require the EXACT named combination, and must call out the
    # hub-item case (omitting a necessary bridge item -> value as
    # independent, near the best contained subset).
    prompt = build_value_query_prompt(
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle=frozenset({"IPAD", "PENCIL"}),
    )

    assert "EXACT" in prompt
    assert "hub" in prompt
    assert "not merely because each" in prompt
    assert "best contained subset" in prompt


def test_singleton_value_query_prompt_names_only_singleton_item():
    prompt = build_value_query_prompt(
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle=frozenset({"IPAD"}),
    )

    assert "PROPOSED_BUNDLE_ITEM_IDS = ['IPAD']" in prompt
    proposed_section = prompt.split(
        "PROPOSED_BUNDLE_ITEM_IDS = ",
        maxsplit=1,
    )[1].splitlines()[0]
    assert "PENCIL" not in proposed_section


def test_value_query_prompt_includes_sorted_anchor_values_and_instructions():
    prompt = build_value_query_prompt(
        scenario_description="A technology auction.",
        person_seed="Values useful combinations.",
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle=frozenset({"IPAD", "PENCIL"}),
        anchor_values={
            frozenset({"PENCIL"}): 150.0,
            frozenset({"IPAD"}): 500.0,
        },
    )

    assert "ANCHOR_VALUES:" in prompt
    assert prompt.index("- [IPAD]: 500.0") < prompt.index(
        "- [PENCIL]: 150.0"
    )
    assert "not hard upper bounds" in prompt
    assert "base value" in prompt
    assert "synergy adjustment" in prompt
    assert "Do not treat ANCHOR_VALUES as a cap" in prompt
    assert "complete setup" in prompt
    assert "production system" in prompt
    assert '"base_value_from_anchors"' in prompt
    assert '"synergy_adjustment"' in prompt
    assert "value it directly from the seed" in prompt


def test_value_query_prompt_omits_anchor_instructions_without_anchors():
    prompt = build_value_query_prompt(
        scenario_description="A technology auction.",
        person_seed="Values useful combinations.",
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle=frozenset({"IPAD", "PENCIL"}),
    )

    assert "ANCHOR_VALUES" not in prompt
    assert "Do not treat ANCHOR_VALUES as a cap" not in prompt
    assert '"base_value_from_anchors"' not in prompt
    assert '"synergy_adjustment"' not in prompt


def test_placeholder_prompt_builders_include_supplied_context():
    answer_prompt = build_person_answer_prompt(
        scenario_description="Scenario context",
        person_seed="Preference context",
        question="What matters most?",
    )
    proxy_prompt = build_initial_proxy_question_prompt(
        scenario_description="Scenario context",
        item_descriptions=ITEM_DESCRIPTIONS,
    )

    assert "Scenario context" in answer_prompt
    assert "Preference context" in answer_prompt
    assert "What matters most?" in answer_prompt
    assert "Scenario context" in proxy_prompt
    # The proxy's opening question must not have access to the person's
    # preference seed -- it's meeting this person for the first time.
    assert "Preference context" not in proxy_prompt
    assert "IPAD" in proxy_prompt
    assert "Apple iPad" in proxy_prompt


def test_initial_proxy_question_prompt_rejects_person_seed():
    # Regression guard: the proxy must never see the person's preference
    # seed when generating its opening question, so the parameter shouldn't
    # even exist on this signature.
    with pytest.raises(TypeError):
        build_initial_proxy_question_prompt(
            scenario_description="Scenario context",
            person_seed="SECRET_TRUE_VALUES_DO_NOT_LEAK",
            item_descriptions=ITEM_DESCRIPTIONS,
        )


def test_provisional_valuation_prompt_rejects_person_seed():
    # Regression guard: PV estimates must come only from the NL exchange,
    # never from the person's actual preference seed.
    with pytest.raises(TypeError):
        build_provisional_valuation_prompt(
            scenario_description="Scenario context",
            person_seed="SECRET_TRUE_VALUES_DO_NOT_LEAK",
            item_descriptions=ITEM_DESCRIPTIONS,
            nl_question="What do you want?",
            nl_answer="I want the iPad.",
            candidate_bundles=[frozenset({"IPAD"})],
        )


def test_provisional_valuation_prompt_reflects_only_the_nl_exchange():
    prompt = build_provisional_valuation_prompt(
        scenario_description="Scenario context",
        item_descriptions=ITEM_DESCRIPTIONS,
        nl_question="What do you want?",
        nl_answer="I want the iPad.",
        candidate_bundles=[frozenset({"IPAD"})],
    )

    assert "Scenario context" in prompt
    assert "What do you want?" in prompt
    assert "I want the iPad." in prompt
    assert "IPAD" in prompt


def test_build_value_query_prompt_includes_elicitation_context():
    prompt = build_value_query_prompt(
        scenario_description="A test auction.",
        person_seed="Prefers A.",
        item_descriptions={"A": "Item A"},
        bundle=frozenset({"A"}),
        elicitation_context="This bundle is close to being dropped at current prices.",
    )

    assert "ELICITATION_CONTEXT:" in prompt
    assert "close to being dropped" in prompt
    assert "Focus your reassessment" in prompt


def test_build_value_query_prompt_no_elicitation_context_by_default():
    prompt = build_value_query_prompt(
        scenario_description="A test auction.",
        person_seed="Prefers A.",
        item_descriptions={"A": "Item A"},
        bundle=frozenset({"A"}),
    )

    assert "ELICITATION_CONTEXT:" not in prompt
