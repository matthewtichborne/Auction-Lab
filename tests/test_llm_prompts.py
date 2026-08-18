"""Prompt construction.

Covers the canonical opening question, which must stay domain-specific and
free of numeric anchoring; the word limits that scale with catalogue size;
and the interest-map prompt, which must discipline exclusions and require
explicit evidence before a substitute group is treated as exclusive.
"""

from __future__ import annotations

import pytest

from auctionlab.llm.prompts import (
    build_interest_map_prompt,
    build_initial_proxy_question_prompt,
    build_person_answer_prompt,
    build_person_answer_verification_prompt,
    build_provisional_valuation_prompt,
    build_value_query_prompt,
    canonical_opening_question,
    describe_bundle,
    person_answer_word_limits,
)


ITEM_DESCRIPTIONS = {
    "IPAD": "Apple iPad",
    "PENCIL": "Apple Pencil",
}


def test_person_answer_word_limits_scale_but_stay_compact():
    assert person_answer_word_limits(10) == (70, 100)
    assert person_answer_word_limits(16) == (91, 130)


def test_canonical_opening_question_is_domain_specific_and_non_numeric():
    question = canonical_opening_question(domain="pc_build")

    assert "PC-component auction" in question
    assert "alternatives" in question
    assert "multiples" in question
    assert "most you would spend overall" in question
    assert "$" not in question


def test_interest_map_prompt_disciplines_exclusions_and_substitutes():
    prompt = build_interest_map_prompt(
        scenario_description="A generic equipment auction.",
        item_descriptions={"A": "Item A", "B": "Item B", "C": "Item C"},
        nl_question="What do you need?",
        nl_answer="B is my first choice, A is a backup, and I do not need C.",
    )
    assert "merely because it appears in the auction" in prompt
    assert "mentioned it negatively" in prompt
    assert "backup if B is unavailable" in prompt
    assert "substitute_groups" in prompt
    assert "do not need C" in prompt
    assert "excluded_items" in prompt
    assert "do not impose hidden domain rules" in prompt
    assert "if the price is right" in prompt
    assert "is NOT evidence of" in prompt
    assert "explicit synergy is absent" in prompt


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
        item_descriptions=ITEM_DESCRIPTIONS,
    )
    proxy_prompt = build_initial_proxy_question_prompt(
        scenario_description="Scenario context",
        item_descriptions=ITEM_DESCRIPTIONS,
    )

    assert "Scenario context" in answer_prompt
    assert "Preference context" in answer_prompt
    assert "What matters most?" in answer_prompt
    assert "Available catalogue" in answer_prompt
    assert "IPAD: Apple iPad" in answer_prompt
    assert "Scenario context" in proxy_prompt
    # The proxy's opening question must not have access to the person's
    # preference seed -- it's meeting this person for the first time.
    assert "Preference context" not in proxy_prompt
    assert "IPAD" in proxy_prompt
    assert "Apple iPad" in proxy_prompt
    assert "only the one maximum-total-willingness-to-pay figure" in answer_prompt
    assert "Do not invent or report individual item values" in answer_prompt
    assert "Do not invent substitute or complementary" in answer_prompt
    assert "Do not suggest that they want all items" in proxy_prompt
    assert "maximum total willingness to pay" in proxy_prompt
    assert "not individual item" in proxy_prompt


def test_person_answer_verifier_prompt_is_blind_to_hidden_truth():
    prompt = build_person_answer_verification_prompt(
        scenario_description="Scenario context",
        question="What do you want?",
        answer="I would like the iPad.",
        item_descriptions=ITEM_DESCRIPTIONS,
    )

    assert "Person preference seed" not in prompt
    assert "Expected positive" not in prompt
    assert "Expected substitute" not in prompt
    assert "The answer is the sole source" in prompt


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
    assert "there is no default percentage" in prompt
    assert "never as a target" in prompt
    assert '"values": [' in prompt
    assert "Do not repeat bundle" in prompt
    assert "IDs in the response" in prompt
    assert '"valuations"' not in prompt


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


def test_value_query_prompt_says_not_to_output_queried_bundle():
    prompt = build_value_query_prompt(
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle=frozenset({"IPAD", "PENCIL"}),
    )

    assert "Do NOT output a queried_bundle field" in prompt
    assert "do not include a queried_bundle" in prompt


def _response_schema_json_body(prompt: str) -> str:
    """Extract just the ``{ ... }`` JSON template following the schema header."""
    header_index = prompt.index("Return JSON only in exactly this schema")
    after_header = prompt[header_index:]
    return after_header.split("{", 1)[1].split("}", 1)[0]


def test_value_query_prompt_response_template_has_value_fields():
    prompt = build_value_query_prompt(
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle=frozenset({"IPAD", "PENCIL"}),
    )

    json_body = _response_schema_json_body(prompt)
    assert '"bundle_value"' in json_body
    assert '"confidence"' in json_body
    assert '"reasoning_summary"' in json_body


def test_value_query_prompt_response_template_omits_queried_bundle():
    prompt = build_value_query_prompt(
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle=frozenset({"IPAD", "PENCIL"}),
    )

    json_body = _response_schema_json_body(prompt)
    assert "queried_bundle" not in json_body


def test_value_query_prompt_still_says_evaluate_exact_bundle():
    prompt = build_value_query_prompt(
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle=frozenset({"IPAD", "PENCIL"}),
    )

    assert "Evaluate ONLY the exact bundle named in PROPOSED_BUNDLE_ITEM_IDS" in prompt


def test_value_query_prompt_says_assign_lower_value_not_alter_bundle():
    prompt = build_value_query_prompt(
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle=frozenset({"IPAD", "PENCIL"}),
    )

    assert "LOW\n  bundle_value -- never by trying to change the bundle." in prompt


def test_value_query_prompt_includes_bad_good_example_without_queried_bundle():
    prompt = build_value_query_prompt(
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        bundle=frozenset({"IPAD", "PENCIL"}),
    )

    assert "Bad example" in prompt
    assert '"queried_bundle": ["GPU_AI", "GPU_GAM", "RAM_64"]' in prompt
    assert "Good example" in prompt

    good_example_section = prompt.split("Good example:", maxsplit=1)[1]
    good_json_body = good_example_section.split("response: {", 1)[1].split(
        "}", 1
    )[0]
    assert "queried_bundle" not in good_json_body


def test_build_value_query_prompt_no_elicitation_context_by_default():
    prompt = build_value_query_prompt(
        scenario_description="A test auction.",
        person_seed="Prefers A.",
        item_descriptions={"A": "Item A"},
        bundle=frozenset({"A"}),
    )

    assert "ELICITATION_CONTEXT:" not in prompt
