"""Prompt builders for bidder simulation and bundle-value elicitation."""

from __future__ import annotations

from auctionlab.auction_types import Bundle, Item


def describe_bundle(
    bundle: Bundle,
    item_descriptions: dict[Item, str],
) -> str:
    """Render a bundle deterministically from known item descriptions."""
    if not bundle:
        return "Empty bundle (no items)."

    lines: list[str] = []
    for item in sorted(bundle):
        if item not in item_descriptions:
            raise ValueError(f"Missing description for item {item}")
        lines.append(f"- {item}: {item_descriptions[item]}")

    return "\n".join(lines)


def format_transcript_context(entries: list[tuple[str, str]]) -> str:
    """Render prior proxy/person NL question-answer pairs as context."""
    if not entries:
        return ""

    qa_lines = "\n".join(
        f"Q: {question}\nA: {answer}" for question, answer in entries
    )
    return f"""
PRIOR_PREFERENCE_QA:
{qa_lines}

Use PRIOR_PREFERENCE_QA as additional context about this person's
preferences when estimating the bundle value below.
"""


def build_value_query_prompt(
    *,
    scenario_description: str,
    person_seed: str,
    item_descriptions: dict[Item, str],
    bundle: Bundle,
    anchor_values: dict[Bundle, float] | None = None,
    transcript_context: str | None = None,
    elicitation_context: str | None = None,
) -> str:
    """Build a strict JSON value query for exactly one proposed bundle.

    Optional singleton anchors calibrate scale while explicitly remaining
    non-binding, allowing complements to exceed additive singleton values.
    ``transcript_context`` (see :func:`format_transcript_context`) can carry
    prior natural-language question/answer pairs that should also inform the
    estimate. ``elicitation_context`` provides the mechanism reason for why
    this specific bundle is being queried (e.g. near-zero surplus, demand
    change), which can help the person focus their reassessment.
    """
    bundle_description = describe_bundle(bundle, item_descriptions)
    proposed_item_ids = sorted(bundle)
    transcript_section = transcript_context or ""
    anchor_section = ""
    response_schema_fields = ""
    if anchor_values:
        anchor_lines = "\n".join(
            f"- [{','.join(sorted(anchor_bundle))}]: {value}"
            for anchor_bundle, value in sorted(
                anchor_values.items(),
                key=lambda item: (
                    len(item[0]),
                    tuple(sorted(item[0])),
                ),
            )
        )
        anchor_section = f"""
ANCHOR_VALUES:
{anchor_lines}

Use ANCHOR_VALUES as calibration points, not hard upper bounds.
Do not treat ANCHOR_VALUES as a cap.
For a multi-item bundle:
1. Identify the relevant singleton anchors.
2. Estimate a base value from those anchors.
3. Decide whether the seed names this EXACT combination of items (or a
   combination the proposed bundle is a subset of) as complementary,
   substitutable, redundant, or irrelevant -- not whether the items are
   each individually things the person likes.
4. Add a positive synergy adjustment only when the seed clearly implies that
   this specific combination of items works together as a system or
   complete setup.
5. Apply a negative or small adjustment when items are substitutes, redundant,
   or irrelevant.
6. Return the final bundle_value.

Strong complementarity can justify a bundle value meaningfully above the sum
of singleton anchors, but only when the seed describes this specific
combination as a complete setup, production system, workflow, or essential
combination.
Do not add large synergy for items described as not central, irrelevant, or
merely broad resale goods.
If the seed identifies one item as a necessary hub or bridge that the others
require to have value (e.g. "X is needed to use Y"), and the proposed bundle
omits that hub item, do not add synergy for the remaining items -- value the
bundle near its best singleton anchor, even though those items are each
individually things the person likes.
If the proposed bundle contains only one item, value it directly from the seed.
"""
        response_schema_fields = """  "base_value_from_anchors": <non-negative number or null>,
  "synergy_adjustment": <number or null>,
"""

    elicitation_section = ""
    if elicitation_context is not None:
        elicitation_section = f"""
ELICITATION_CONTEXT:
{elicitation_context}

The ELICITATION_CONTEXT explains why this bundle is being reconsidered.
Focus your reassessment on this specific bundle in light of the context above.
"""

    return f"""You are simulating a person's valuation in this scenario.

Scenario:
{scenario_description}

Person preference seed:
{person_seed}
{anchor_section}
{elicitation_section}
{transcript_section}

You are valuing ONLY the PROPOSED_BUNDLE below.
Do not return the value of a larger bundle, a preferred bundle, or the person's
overall maximum willingness to pay unless that is exactly the proposed bundle.

PROPOSED_BUNDLE_ITEM_IDS = {proposed_item_ids}

Proposed bundle item descriptions:
{bundle_description}

The returned queried_bundle must exactly match PROPOSED_BUNDLE_ITEM_IDS.
If the proposed bundle is a singleton, return only that singleton in
queried_bundle. If the seed mentions values for larger bundles, use them only
as context; do not copy a larger bundle's value as the singleton value.

Do not assume bundle values are additive, but do not invent synergy either.
Only apply a complementary premium when the seed names this EXACT combination
of items (or a combination the proposed bundle is fully contained in) as
complementary, a core pair, or a complete setup -- not merely because each
item individually is something the person is interested in. Items the person
likes individually are not automatically complementary with each other.
If the seed identifies one item as a necessary hub or bridge that the others
require to have value (for example, "X is needed to route/use/plug in Y"),
and the proposed bundle omits that hub item, treat the remaining items as
independent: value the bundle near its best contained subset's value, not as
a complementary cluster, even though those items are each individually things
the person likes.
For substitute or redundant items, the value of a bundle may be only slightly
higher than the best individual item.
If the seed says items are substitutes, close substitutes, redundant,
overlapping, or that the person does not need both, do not simply add singleton values.
When you are not sure whether this specific combination is complementary,
default to the value of its best contained subset rather than adding a premium.

Estimate the person's value using these five steps:
1. Check whether the person explicitly stated a value for this bundle.
2. Find the closest explicitly valued bundle or bundles that are subsets of
   the proposed bundle.
3. Identify whether the EXACT proposed combination matches, or is fully
   contained in, a complementary group the seed actually names -- not
   whether its items are each individually things the person likes.
4. If it matches a named complementary group, apply that relationship. If it
   does not -- especially if it is missing a hub item the seed identifies as
   necessary -- treat the items as independent and value the bundle near its
   best contained subset.
5. Estimate the final non-negative dollar value.

Return JSON only in exactly this schema:
{{
  "queried_bundle": ["ITEM_ID_1", "ITEM_ID_2"],
{response_schema_fields}\
  "bundle_value": <non-negative number>,
  "confidence": <number between 0 and 1 or null>,
  "reasoning_summary": "<short summary explaining why this value is for exactly the queried bundle>"
}}

Do not include markdown fences or any text outside the JSON object."""


def build_person_answer_prompt(
    *,
    scenario_description: str,
    person_seed: str,
    question: str,
) -> str:
    return f"""Answer as the simulated person.

Scenario:
{scenario_description}

Person preference seed:
{person_seed}

Question:
{question}

Return JSON only in exactly this schema:
{{
  "answer": "<your answer to the question, in your own words>"
}}

Do not include markdown fences or any text outside the JSON object."""


def build_initial_proxy_question_prompt(
    *,
    scenario_description: str,
    item_descriptions: dict[Item, str],
) -> str:
    """Build the proxy's opening preference-elicitation question.

    Deliberately excludes the person's preference seed: the proxy is
    meeting this person for the first time and must ask a question that
    works from the scenario and item catalog alone, not from private
    knowledge of the person's actual values.
    """
    item_ids = sorted(item_descriptions)
    items = "\n".join(
        f"- {item}: {description}"
        for item, description in sorted(item_descriptions.items())
    )

    return f"""Create an initial preference-elicitation question.

Scenario:
{scenario_description}

Available items ({len(item_ids)} total):
{items}

You are a bidding proxy meeting this person for the first time. Ask one
conversational question that helps you understand three things:
1. Which of the available items they actually want (versus items they can ignore).
2. Whether any items are alternatives to each other — where getting one would
   make the other much less valuable (substitutes).
3. Whether any items are strongly complementary — items they would want as a
   set rather than individually.

Do not ask about prices. Ask in a natural, friendly way. The question should
be open-ended so the person can describe their situation in their own words.

Return JSON only in exactly this schema:
{{
  "question": "<your preference-elicitation question>"
}}

Do not include markdown fences or any text outside the JSON object."""


def build_provisional_valuation_prompt(
    *,
    scenario_description: str,
    item_descriptions: dict[Item, str],
    nl_question: str,
    nl_answer: str,
    candidate_bundles: list[Bundle],
    interest_map=None,
) -> str:
    """Proxy-side prompt to estimate values for all candidate bundles at once.

    Deliberately excludes the person's preference seed: the proxy's only
    legitimate source of information about this person is what was actually
    revealed through ``nl_question``/``nl_answer`` (and the structured
    ``interest_map`` derived from that answer), not private knowledge of
    their actual values.

    ``interest_map`` is optional. When provided, complement/substitute
    structure and budget hint are surfaced to help calibrate values.
    Without it the estimate relies solely on the NL text.
    """
    items_section = "\n".join(
        f"- {item}: {description}"
        for item, description in sorted(item_descriptions.items())
    )

    bundle_lines = "\n".join(
        f"{i + 1}. {sorted(b)}"
        for i, b in enumerate(candidate_bundles)
    )

    interest_map_section = ""
    if interest_map is not None:
        parts: list[str] = []
        parts.append(
            f"Interested items: {sorted(interest_map.interested_items)}"
        )
        if interest_map.complementary_groups:
            groups = [sorted(g) for g in interest_map.complementary_groups]
            parts.append(
                f"Complementary groups (value together synergistically): {groups}\n"
                "  → A complementary group's bundle value should meaningfully exceed "
                "the sum of its singleton values."
            )
        if interest_map.substitute_groups:
            groups = [sorted(g) for g in interest_map.substitute_groups]
            parts.append(
                f"Substitute groups (person wants at most one from each): {groups}\n"
                "  → Bundles combining substitutes should be only marginally more "
                "valuable than the best single substitute."
            )
        if interest_map.budget_hint is not None:
            parts.append(
                f"Approximate total budget: {interest_map.budget_hint}\n"
                "  → The total value of all items this person could win should not "
                "far exceed their stated budget."
            )
        interest_map_section = (
            "\nStructural preferences extracted from their answer:\n"
            + "\n".join(f"- {p}" for p in parts)
            + "\n"
        )

    return f"""You are a valuation analyst for an auction.

Scenario:
{scenario_description}

Available items:
{items_section}

This person was asked: "{nl_question}"
They responded: "{nl_answer}"
{interest_map_section}
Estimate this person's monetary value for every bundle listed in BUNDLES_TO_VALUE.
Produce estimates that are internally consistent:

- A bundle's value must not exceed any strict superset bundle's value.
- Items the person has no interest in contribute near zero marginal value.
- Substitute items together add little beyond the best single substitute.
- **CRITICAL — complement bonuses**: If complementary groups are listed above,
  the VALUE OF THE FULL COMPLEMENT GROUP must be SUBSTANTIALLY greater than the
  sum of each item's standalone value.  A typical complement bonus is 50–150% on
  top of the sum of parts — not a marginal increase.  For example, if item A
  alone = 400 and item B alone = 300, the bundle {{A, B}} in a complementary
  group should be around 900–1400, NOT 700.  Larger complement groups (3–4
  items) carry proportionally larger bonuses because they enable qualitatively
  new use-cases the person explicitly desires.  Failing to include a generous
  complement bonus is the most common calibration error.

BUNDLES_TO_VALUE:
{bundle_lines}

Return JSON only in exactly this schema:
{{
  "valuations": [
    {{"bundle": ["ITEM_ID"], "value": <non-negative number>}},
    ...
  ],
  "reasoning": "<2-4 sentences: state the complement bonus you applied and why>"
}}

Include one entry per bundle in BUNDLES_TO_VALUE, in the same order.
Use only item IDs from the available items list.
Do not include markdown fences or any text outside the JSON object."""


def build_interest_map_prompt(
    *,
    scenario_description: str,
    item_descriptions: dict[Item, str],
    nl_question: str,
    nl_answer: str,
) -> str:
    """Proxy-side parsing call: extract a structured interest map from a Q/A pair.

    This prompt is never shown to the person; it is a proxy inference step that
    interprets the person's NL answer and returns an :class:`LlmInterestMap`.
    """
    item_ids = sorted(item_descriptions)
    items = "\n".join(
        f"- {item}: {description}"
        for item, description in sorted(item_descriptions.items())
    )

    return f"""Extract a structured interest map from the preference question and answer below.

Scenario:
{scenario_description}

Available items (use ONLY these item IDs exactly as written):
{items}

Question asked:
{nl_question}

Person's answer:
{nl_answer}

Instructions:
- interested_items: item IDs the person clearly wants or has positive interest in.
  Include an item if they mention wanting it, valuing it, or finding it useful.
  If the person mentions interest in ALL items or does not exclude any, list all item IDs.
- excluded_items: item IDs the person explicitly does not want or has near-zero value for.
  Only exclude if they say so explicitly — do not exclude by assumption.
- complementary_groups: lists of item IDs the person wants as a COMPLETE SET together.
  Each list is a bundle to prioritise. A group may have two or more items.
  Only include groups where the person specifically said they want those items together.
- substitute_groups: lists of item IDs where the person wants AT MOST ONE item from
  the group (e.g. "I only need one of X or Y"). Two or more items from the same
  substitute_group will never appear together in a candidate bundle.
- budget_hint: a rough total spend the person mentioned, or null.
- reasoning: a brief (2–4 sentence) explanation of why you assigned each item
  to the categories above, citing the person's words directly where possible.

Important rules:
- Use ONLY item IDs from the available list above. Do not invent item IDs.
- An item can appear in interested_items AND in a complementary_group or substitute_group.
- An item in excluded_items should NOT appear in complementary_groups or substitute_groups.
- If uncertain whether an item is interested or excluded, include it in interested_items.
- If no substitutes or complements are evident, return empty lists for those fields.

Return JSON only in exactly this schema:
{{
  "interested_items": ["ITEM_ID_1", "ITEM_ID_2"],
  "excluded_items": ["ITEM_ID_3"],
  "complementary_groups": [["ITEM_ID_1", "ITEM_ID_2"]],
  "substitute_groups": [["ITEM_ID_4", "ITEM_ID_5"]],
  "budget_hint": <number or null>,
  "reasoning": "<brief explanation>"
}}

Do not include markdown fences or any text outside the JSON object."""


def format_summary_context(
    summary: str,
    recent_qa: list[tuple[str, str]],
) -> str:
    """Render a rolling summary and recent Q/A pairs as preference context."""
    if not summary and not recent_qa:
        return ""

    parts: list[str] = []

    if summary:
        parts.append(f"PREFERENCE_SUMMARY:\n{summary}")

    if recent_qa:
        qa_lines = "\n".join(
            f"Q: {question}\nA: {answer}" for question, answer in recent_qa
        )
        parts.append(f"RECENT_QA:\n{qa_lines}")

    body = "\n\n".join(parts)
    return f"""
{body}

Use PREFERENCE_SUMMARY and RECENT_QA as additional context about this person's
preferences when estimating the bundle value below.
"""


def build_summary_update_prompt(
    *,
    current_summary: str,
    question: str,
    answer: str,
) -> str:
    """Build a prompt to update a rolling preference summary with a new Q/A pair."""
    summary_section = (
        current_summary if current_summary else "(No summary recorded yet.)"
    )
    return f"""Update the preference summary below with the new question and answer.

CURRENT_SUMMARY:
{summary_section}

NEW_QA:
Q: {question}
A: {answer}

Write an updated concise summary (2 to 5 sentences) that:
1. Preserves important preferences stated in CURRENT_SUMMARY.
2. Incorporates any new preferences, priorities, or trade-offs from NEW_QA.
3. Notes complements (items that work well together) and substitutes (items that are interchangeable).
4. Does not invent preferences not mentioned in any Q/A.

Return JSON only in exactly this schema:
{{
  "summary": "<updated preference summary>"
}}

Do not include markdown fences or any text outside the JSON object."""


def build_demand_query_prompt(
    *,
    scenario_description: str,
    person_seed: str,
    item_descriptions: dict[Item, str],
    bundle: Bundle,
    prices: dict[Item, float],
) -> str:
    """Ask whether the person is satisfied with ``bundle`` at ``prices``.

    If not satisfied, the person may report a preferred bundle (by item ID)
    instead of, or in addition to, the proposed one.
    """
    bundle_description = describe_bundle(bundle, item_descriptions)
    proposed_item_ids = sorted(bundle)
    bundle_price = sum(prices[item] for item in bundle)
    price_lines = "\n".join(
        f"- {item}: {prices[item]}" for item in sorted(bundle)
    )

    return f"""You are simulating a person's purchase decision in this scenario.

Scenario:
{scenario_description}

Person preference seed:
{person_seed}

PROPOSED_BUNDLE_ITEM_IDS = {proposed_item_ids}

Proposed bundle item descriptions:
{bundle_description}

Per-item prices for the proposed bundle:
{price_lines}

Total price of the proposed bundle: {bundle_price}

Available items (for reference, if you prefer a different bundle):
{chr(10).join(f"- {item}: {description}" for item, description in sorted(item_descriptions.items()))}

Decide whether this person would be satisfied buying exactly the proposed
bundle at the listed total price (i.e. it is worth at least the price to
them). If they would not be satisfied, report the bundle of item IDs they
would prefer instead (it may be empty, smaller, larger, or different).

Return JSON only in exactly this schema:
{{
  "satisfied": <true or false>,
  "preferred_bundle": <list of item ID strings, or null if satisfied>
}}

Do not include markdown fences or any text outside the JSON object."""


