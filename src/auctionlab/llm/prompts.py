"""Prompt builders for bidder simulation and bundle-value elicitation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from auctionlab.auction_types import Bundle, Item


CANONICAL_OPENING_QUESTION = (
    "Tell me what you're hoping to get from this auction and how the options "
    "fit your needs. Are any alternatives, useful only together, or still "
    "useful in multiples—and roughly what is the most you would spend overall?"
)


def canonical_opening_question(*, domain: str | None = None) -> str:
    """Return the versioned, scenario-level opening elicitation instrument."""
    if domain == "pc_build":
        return (
            "Tell me what you're hoping to get from this PC-component auction "
            "and how the options fit your needs. Are any alternatives, useful "
            "only together, or still useful in multiples—and roughly what is "
            "the most you would spend overall?"
        )
    return CANONICAL_OPENING_QUESTION


def person_answer_word_limits(num_goods: int) -> tuple[int, int]:
    """Return the natural-language target and enforced hard word limit."""
    if num_goods <= 0:
        raise ValueError("num_goods must be positive")
    return round(35 + 3.5 * num_goods), round(50 + 5 * num_goods)


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

CRITICAL -- the queried bundle is fixed by the auction mechanism and has
already been selected. You are evaluating it, not choosing, editing, or
reporting it back:
- Do NOT output a queried_bundle field in your JSON response. The response
  schema below has no bundle field for you to fill in.
- Do not list, rewrite, repair, complete, or otherwise restate the bundle
  anywhere in your JSON -- only the value fields below.
- Do not add missing complements, remove unwanted items, substitute a
  preferred item for a queried item, replace a weak substitute with a
  preferred one, sort/reorder, or duplicate any item ID. None of that is
  possible in the response schema below, because the bundle is not yours to
  change -- it is fixed by the caller, not by your answer.
- Evaluate ONLY the exact bundle named in PROPOSED_BUNDLE_ITEM_IDS above,
  even if it is incomplete, redundant, an unusual combination, or
  economically irrational for this person. If the bundle is unattractive,
  low-interest, or pairs poor substitutes together, reflect that with a LOW
  bundle_value -- never by trying to change the bundle.
- Discuss complementarity or substitution only inside reasoning_summary, and
  only to the extent it changes the VALUE of this exact bundle.

Bad example (do NOT do this):
  PROPOSED_BUNDLE_ITEM_IDS = ["GPU_AI", "GPU_GAM"]
  response: {{"queried_bundle": ["GPU_AI", "GPU_GAM", "RAM_64"], "bundle_value": 900}}
  -- WRONG: invented a queried_bundle field and added RAM_64, which was not
  part of the query.

Good example:
  PROPOSED_BUNDLE_ITEM_IDS = ["GPU_AI", "GPU_GAM"]
  response: {{"bundle_value": 350, "confidence": 0.7, "reasoning_summary":
  "These two GPUs are largely redundant for this person, so the pair is
  worth only slightly more than the better single GPU."}}
  -- correct: no queried_bundle field at all; the low/moderate value reflects
  the redundancy instead of altering the bundle.

If the proposed bundle is a singleton, value it directly. If the seed
mentions values for larger bundles, use them only as context; do not copy a
larger bundle's value as the singleton value.

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

Return JSON only in exactly this schema -- do not include a queried_bundle
field:
{{
{response_schema_fields}\
  "bundle_value": <non-negative number>,
  "confidence": <number between 0 and 1 or null>,
  "reasoning_summary": "<one or two sentences explaining the value of the exact queried bundle>"
}}

Do not include markdown fences or any text outside the JSON object."""


def build_person_answer_prompt(
    *,
    scenario_description: str,
    person_seed: str,
    question: str,
    item_descriptions: dict[Item, str] | None = None,
    target_words: int | None = None,
    hard_max_words: int | None = None,
) -> str:
    default_target, default_hard_max = person_answer_word_limits(
        len(item_descriptions or {}) or 1
    )
    target_words = target_words or default_target
    hard_max_words = hard_max_words or default_hard_max
    catalogue_section = ""
    if item_descriptions:
        catalogue = "\n".join(
            f"- {item}: {description}"
            for item, description in sorted(item_descriptions.items())
        )
        catalogue_section = f"""
Available catalogue:
{catalogue}
"""
    return f"""Answer as the simulated person.

Scenario:
{scenario_description}
{catalogue_section}

Person preference seed:
{person_seed}

Question:
{question}

Write naturally in the first person. Aim for about {target_words} words and
never exceed {hard_max_words} words. Prefer compact grouped statements over a
repetitive item-by-item checklist.

The preference seed is deliberately qualitative. Preserve its distinctions,
but do not make the answer more numerically detailed than the seed:
- Clearly mention every item with positive interest, including weak
  fallbacks and lower-priority alternatives. Exact item IDs are encouraged
  when they fit naturally, but conversational catalogue names are acceptable.
- You do not need to enumerate every irrelevant item. Silence about an item
  is treated as no disclosed interest; mention a salient exclusion only when
  it helps explain the person's needs.
- State only the one maximum-total-willingness-to-pay figure supplied there.
- Do not invent or report individual item values, bundle values, substitute
  percentages, complement bonuses, diminishing-returns equations, or a
  complete valuation schedule.
- Do not invent substitute or complementary relationships.
- Preserve whether related alternatives are a single-choice decision or
  whether the person can still use, deploy, or resell multiple alternatives.
  For every single-choice group, explicitly say that only/at most one member
  is needed or that extra members add no meaningful benefit. A ranking or
  fallback statement alone does not communicate this. Express the meaning
  naturally and compactly; do not repeat a prescribed sentence template for
  every group.
- State a complementary group only when the seed explicitly supplies one.
  Make clear that the complete set has extra value beyond the separate
  components. Do not describe ordinary components of a setup as complementary
  merely because they work together.
- Do not turn excluded items into conditional or "if the price is right"
  fallbacks.
- Preserve the polarity of every available item: anything described as
  lower priority, less important, a fallback, or an alternative still has
  positive interest and must not be called unwanted or excluded.

Return JSON only in exactly this schema:
{{
  "answer": "<your answer to the question, in your own words>"
}}

Do not include markdown fences or any text outside the JSON object."""


def build_person_answer_verification_prompt(
    *,
    scenario_description: str,
    question: str,
    answer: str,
    item_descriptions: Mapping[Item, str],
) -> str:
    """Build a blind semantic extraction prompt for answer verification.

    The expected environment truth is deliberately absent. This avoids a
    verifier merely agreeing with a supplied answer key; code compares the
    extraction with hidden truth only after the model call.
    """
    target_words, hard_max_words = person_answer_word_limits(
        len(item_descriptions)
    )
    catalogue = "\n".join(
        f"- {item}: {description}"
        for item, description in sorted(item_descriptions.items())
    )
    return f"""Verify a simulated person's answer against its qualitative
environment specification. You are an offline preparation-time verifier, not
the bidder proxy. Judge semantic meaning, not exact wording or sentence shape.

Scenario:
{scenario_description}

Catalogue:
{catalogue}

Question:
{question}

Answer to verify:
{answer}

Reconstruct only what this answer discloses:
- Map natural catalogue names to exact item IDs.
- positive_items includes primary choices, weak fallbacks and conditional
  alternatives, but never an item mentioned only negatively.
- excluded_items includes only explicit zero-interest or rejection claims.
  Do not infer exclusion merely from silence.
- For every substitute group, report its exact disclosed members, acquisition
  mode, a short passage of supporting evidence, and whether that mode was
  explicitly stated.
- Set mode_explicitly_stated=true for "choose_one" only when the answer says
  at most/only one member is wanted or that additional members add no useful
  value. A ranking, "fallback", "backup", "alternative", or shared function
  does not establish choose_one. Apply the same standard to
  "can_use_multiple": it requires an explicit multi-unit use, resale,
  inventory, redundancy, or similar claim. Otherwise use mode "unclear" and
  mode_explicitly_stated=false.
- Do not create a substitute group merely because two independently useful
  items have different priorities, or because one is called an "option".
  There must be evidence that the items are related alternatives for the same
  need or that one can replace the other. Language saying both are useful
  independently is evidence against grouping them as substitutes.
- A complementary group requires an explicit claim that the complete set
  provides additional utility beyond owning the members separately. Words
  such as "stack", "setup", "bundle", "ideal", or a list of desired
  components are not enough by themselves.
- budget_hint is the single overall maximum willingness to pay, if supplied.
- List any other numeric item value, bundle value, factor, bonus or valuation
  detail in other_numeric_valuation_details.
- Evidence may be a concise verbatim span or faithful short paraphrase.
- Do not guess from the scenario or catalogue. The answer is the sole source.
- Do not assess whether the answer is correct and do not invent missing
  information. A separate deterministic comparison will do that later.
- The target is about {target_words} words and the hard maximum is
  {hard_max_words}; length enforcement is performed separately by code.

Return JSON only:
{{
  "positive_items": [
    {{"item_id": "ITEM_ID", "evidence": "<support from answer>"}}
  ],
  "excluded_items": [
    {{"item_id": "ITEM_ID", "evidence": "<support from answer>"}}
  ],
  "substitute_groups": [
    {{
      "items": ["ITEM_ID_1", "ITEM_ID_2"],
      "acquisition_mode": "choose_one | can_use_multiple | unclear",
      "evidence": "<support from answer>",
      "mode_explicitly_stated": <true or false>
    }}
  ],
  "complementary_groups": [
    {{
      "items": ["ITEM_ID_1", "ITEM_ID_2"],
      "evidence": "<explicit extra joint-value claim>",
      "explicit_extra_joint_value": <true or false>
    }}
  ],
  "budget_hint": <number or null>,
  "other_numeric_valuation_details": []
}}

Do not include markdown fences or text outside the JSON object."""


def build_substitute_mode_entailment_prompt(
    *,
    answer: str,
    item_descriptions: Mapping[Item, str],
    substitute_groups: Sequence[Mapping[str, Any]],
) -> str:
    """Build a truth-blind, focused acquisition-mode entailment check."""
    catalogue = "\n".join(
        f"- {item}: {description}"
        for item, description in sorted(item_descriptions.items())
    )
    group_lines = "\n".join(
        (
            f"- items={list(group['items'])}; "
            f"proposed_mode={group['acquisition_mode']}"
        )
        for group in substitute_groups
    )
    return f"""Check whether the person's answer EXPLICITLY entails each
proposed acquisition mode. This is a narrow text-entailment task. You do not
see hidden preferences and must not infer intent from the catalogue.

Catalogue:
{catalogue}

Person's answer:
{answer}

Groups to check:
{group_lines}

Rules:
- For choose_one, entailed=true only when the answer explicitly says the
  person wants/needs at most one member, is making a single choice, or would
  gain no meaningful benefit from owning additional members of that exact
  group.
- A priority ordering, "fallback", "backup", "alternative", "if unavailable",
  or the fact that items share a function is NOT sufficient.
- A statement about one category cannot establish the mode of another
  category. "I only need one CPU" supports a CPU group, not a GPU group.
- For can_use_multiple, entailed=true only when the answer explicitly says
  multiple members remain useful, for example for resale, inventory,
  redundancy, or deployment to different systems.
- Evidence must quote or faithfully reproduce the explicit words. If those
  words are absent, set entailed=false and explain what is missing.
- Return exactly one judgment for every supplied group, using the same item
  IDs and proposed acquisition mode.

Return JSON only:
{{
  "judgments": [
    {{
      "items": ["ITEM_ID_1", "ITEM_ID_2"],
      "acquisition_mode": "choose_one | can_use_multiple",
      "entailed": <true or false>,
      "evidence": "<explicit words, or empty>",
      "reason": "<brief explanation>"
    }}
  ]
}}

Do not include markdown fences or text outside the JSON object."""


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
concise, conversational question that helps you understand four things:
1. Which of the available items they actually want (versus items they can ignore).
2. Whether any items are alternatives to each other, and for each set whether
   they would choose at most one or could still use multiple alternatives.
3. Whether any items are strongly complementary — items they would want as a
   set rather than individually.
4. Their maximum total willingness to pay in this auction.

Ask for only one overall willingness-to-pay ceiling, not individual item
prices or a complete bundle-by-bundle valuation schedule. Keep the question
natural and open-ended so the person can describe their situation in their
own words. Do not suggest that they want all items, call any item a backup,
or propose a specific bundle; those are facts the question is meant to
discover.

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
    chunk_index: int | None = None,
    chunk_count: int | None = None,
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

    ``chunk_index``/``chunk_count`` (both 0-indexed/total, only set when PV
    chunking split a large candidate set into multiple calls -- see
    :func:`~auctionlab.llm.provisional_valuations.generate_provisional_valuations_chunked`)
    annotate the prompt with which slice of the bidder's full candidate set
    this call covers. The model is deliberately NOT asked to reason across
    chunks -- each call only ever sees its own slice of bundles and values
    them independently; chunk identity is purely informational context.
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
            choose_one = [
                sorted(group.items)
                for group in interest_map.substitute_groups
                if group.acquisition_mode == "choose_one"
            ]
            multi = [
                sorted(group.items)
                for group in interest_map.substitute_groups
                if group.acquisition_mode == "can_use_multiple"
            ]
            unclear = [
                sorted(group.items)
                for group in interest_map.substitute_groups
                if group.acquisition_mode == "unclear"
            ]
            if choose_one:
                parts.append(
                    f"Choose-one substitute groups: {choose_one}\n"
                    "  → Joint bundles add negligible value beyond the best "
                    "single alternative."
                )
            if multi:
                parts.append(
                    f"Related alternatives usable in multiples: {multi}\n"
                    "  → Preserve positive value for joint ownership."
                )
            if unclear:
                parts.append(
                    f"Unclear alternative relationships: {unclear}\n"
                    "  → Value joint ownership conservatively but do not "
                    "assume exclusivity."
                )
        if interest_map.budget_hint is not None:
            parts.append(
                f"Selected-auction value/spending ceiling: "
                f"{interest_map.budget_hint}\n"
                "  → Treat this as an upper bound, not a target. Do not inflate "
                "singleton or bundle values merely to approach it."
            )
        interest_map_section = (
            "\nStructural preferences extracted from their answer:\n"
            + "\n".join(f"- {p}" for p in parts)
            + "\n"
        )

    chunk_section = ""
    if chunk_index is not None and chunk_count is not None and chunk_count > 1:
        chunk_section = f"""
NOTE: This is chunk {chunk_index + 1} of {chunk_count} of this bidder's
candidate bundles, sent as separate calls only to keep each response small.
Value only the bundles listed in BUNDLES_TO_VALUE below -- do not reference,
compare against, or try to reason about bundles from any other chunk.
"""

    return f"""You are a valuation analyst for an auction.

Scenario:
{scenario_description}

Available items:
{items_section}

This person was asked: "{nl_question}"
They responded: "{nl_answer}"
{interest_map_section}{chunk_section}
Estimate this person's monetary value for every bundle listed in BUNDLES_TO_VALUE.
Produce estimates that are internally consistent:

- A bundle's value must not exceed any strict superset bundle's value.
- Items the person has no interest in contribute near zero marginal value.
- For choose-one substitutes, joint ownership adds little beyond the best
  alternative. For can-use-multiple or unclear relationships, do not erase
  the additional value of owning multiple items.
- Apply a complement bonus only to an explicitly listed complete
  complementary group. Calibrate its size to the strength of synergy actually
  described by the person; there is no default percentage. Do not add synergy
  because several useful functional components form a generic "setup" or
  "bundle".
- Treat any stated budget or selected-auction ceiling as a hard upper bound,
  never as a target that singleton values or the grand bundle should approach.

BUNDLES_TO_VALUE:
{bundle_lines}

Return JSON only in exactly this schema:
{{
  "values": [<non-negative number>, <non-negative number>, ...],
  "reasoning": "<2-4 sentences: state the complement bonus you applied and why>"
}}

The values array must contain exactly one number per bundle in
BUNDLES_TO_VALUE, in precisely the same numbered order. Do not repeat bundle
IDs in the response. Before returning, count that the number of values exactly
matches the number of numbered bundles.
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
  Do not mark an item as interested merely because it appears in the auction
  catalogue, or merely because the bidder mentioned it negatively.
- excluded_items: every available item ID the person does not positively
  want. This is a closed-world disclosure: after accounting for structural
  group members, any catalogue item not identified with positive interest
  must be excluded, even if the answer leaves it implicit.
  Items described as irrelevant, not useful, not needed, a poor fit, too weak,
  too expensive, or as something that "doesn't cut it" should usually be
  excluded unless the bidder clearly assigns them positive fallback value.
- complementary_groups: lists of item IDs the person wants as a COMPLETE SET together.
  These are items the bidder explicitly says are worth more together than
  separately because the complete set creates additional utility. Merely
  wanting several distinct functional components, describing a "setup" or
  "bundle", or saying several items are useful is NOT evidence of
  complementarity. When explicit synergy is absent, return an empty list.
- complementary_group_evidence: one record for every proposed complementary
  group. Repeat the exact item IDs, provide supporting evidence, and set
  explicit_extra_joint_value=true only when the answer explicitly says the
  complete set adds utility beyond the separate items. A proposed group
  without a matching true evidence record will be discarded.
- substitute_groups: person-specific related alternatives. For each group return:
  - items: the related item IDs;
  - acquisition_mode:
    - "choose_one" ONLY when the answer explicitly says the person wants at
      most one, is selecting a single component, or gets no meaningful
      additional benefit from owning multiple members;
    - "can_use_multiple" when multiple members retain value for resale,
      inventory, redundancy, deployment to different systems, or multi-unit use;
    - "unclear" when the answer establishes alternatives but does not clearly
      establish whether joint ownership remains useful;
  - evidence: a short phrase grounded in the person's answer.
  - mode_explicitly_stated: true only when the cited words explicitly support
    choose_one or can_use_multiple; otherwise false.
  A preference ranking, the word "fallback", or items serving the same
  function is not by itself sufficient evidence for "choose_one".
- budget_hint: a rough total spend the person mentioned, or null.
- reasoning: a brief (1–2 sentence) explanation of why you assigned each item
  to the categories above, citing the person's words directly where possible.

Important rules:
- Use ONLY item IDs from the available list above. Do not invent item IDs.
- The union of interested_items and excluded_items must cover every
  available item exactly once. Before responding, check that the two arrays
  are disjoint: an ID must NEVER appear in both arrays.
- Use the scenario and item descriptions to interpret the natural-language answer,
  but do not impose hidden domain rules beyond what can be inferred from this text.
- An item can appear in interested_items AND in a complementary_group or substitute_group.
- An item in excluded_items should NOT appear in complementary_groups or substitute_groups.
- Every member of a complementary or substitute group must also appear in
  interested_items.
- Treat "zero value", "essentially zero", "near-zero", or willingness only
  "if the price is right" as exclusion, not positive interest.
- If no substitutes or complements are evident, return empty lists for those fields.
- Keep reasoning concise (at most two short sentences) and do not add fields
  beyond the exact schema below.
- Generic examples: if a bidder says A is a backup if B is unavailable but
  does not say whether both are useful, group A and B with mode "unclear".
  If they explicitly say they need only one, use "choose_one". If they can
  resell or deploy both, use "can_use_multiple".
  If a bidder says they do not need C, put C in excluded_items, not interested_items.
- Perform a final consistency check before returning JSON:
  1. interested_items and excluded_items have no overlap;
  2. together they contain every available ID exactly once;
  3. no excluded ID occurs in any group;
  4. "choose_one" and "can_use_multiple" are supported by explicit words in
     the person's answer, otherwise use "unclear";
  5. a complementary group is present only when the answer explicitly says
     the complete set is worth more together than separately.

Return JSON only in exactly this schema:
{{
  "interested_items": ["ITEM_ID_1", "ITEM_ID_2"],
  "excluded_items": ["ITEM_ID_3"],
  "complementary_groups": [["ITEM_ID_1", "ITEM_ID_2"]],
  "complementary_group_evidence": [
    {{
      "items": ["ITEM_ID_1", "ITEM_ID_2"],
      "evidence": "<explicit extra joint-value claim>",
      "explicit_extra_joint_value": <true or false>
    }}
  ],
  "substitute_groups": [
    {{
      "items": ["ITEM_ID_4", "ITEM_ID_5"],
      "acquisition_mode": "choose_one | can_use_multiple | unclear",
      "evidence": "<short evidence from the answer>",
      "mode_explicitly_stated": <true or false>
    }}
  ],
  "budget_hint": <number or null>,
  "reasoning": "<brief explanation>"
}}

Do not include markdown fences or any text outside the JSON object."""


def build_complement_entailment_prompt(
    *,
    item_descriptions: Mapping[Item, str],
    nl_question: str,
    nl_answer: str,
    proposed_groups: Sequence[Mapping[str, Any]],
) -> str:
    """Build a truth-blind conservative check of proposed complements."""
    catalogue = "\n".join(
        f"- {item}: {description}"
        for item, description in sorted(item_descriptions.items())
    )
    return f"""Audit proposed complementary groups using only the person's
answer. You do not know the hidden environment and must not infer from general
PC compatibility.

Catalogue:
{catalogue}

Question:
{nl_question}

Answer:
{nl_answer}

Proposed groups and extraction evidence:
{list(proposed_groups)}

For each proposed group, set entailed=true only if the answer explicitly says
that owning the complete set creates additional utility beyond owning its
members separately. A preferred list, "ideal stack", "setup", "bundle",
workflow, compatibility, or wanting all the components is not sufficient.
When uncertain, set entailed=false.

Return JSON only:
{{
  "judgments": [
    {{
      "items": ["ITEM_ID_1", "ITEM_ID_2"],
      "entailed": <true or false>,
      "evidence": "<supporting answer passage or empty>",
      "reason": "<one short sentence>"
    }}
  ]
}}

Return exactly one judgment per proposed group and no text outside the JSON."""


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


def build_late_reflection_prompt(
    *,
    scenario_description: str,
    item_descriptions: dict[Item, str],
    context: dict,
) -> str:
    """Build the late-stage reflective elicitation prompt.

    ``context`` is the structured per-bidder summary produced by
    :func:`auctionlab.llm.late_reflection.build_late_reflection_context`
    (via its ``as_prompt_dict()``), never the raw transcript. The prompt
    forces every question into an explicit pairwise or marginal comparison
    (``reflection_mode``) whose answer maps directly onto one or two
    follow-up value queries -- never a generic open-ended preference
    question -- and instructs the model to prefer distinctions that are
    still unresolved by the current allocation/demand (``resolved_hint``
    fields in ``context``, when present).
    """
    items_section = "\n".join(
        f"- {item}: {description}"
        for item, description in sorted(item_descriptions.items())
    )

    def _fmt_bundle(bundle: list[str] | None) -> str:
        return "[" + ", ".join(sorted(bundle)) + "]" if bundle else "(none)"

    lines: list[str] = []
    lines.append(f"mechanism: {context.get('mechanism')}")
    lines.append(f"round: {context.get('round_idx')}")
    if context.get("initial_nl_summary"):
        lines.append(f"initial preference exchange: {context['initial_nl_summary']}")
    if context.get("interest_map_summary"):
        lines.append(f"interest map: {context['interest_map_summary']}")
    if context.get("budget_hint") is not None:
        lines.append(f"budget hint: {context['budget_hint']}")

    top_bundles = context.get("top_reported_bundles") or []
    if top_bundles:
        bundle_lines = "; ".join(
            f"{_fmt_bundle(b)}={v}" for b, v in top_bundles
        )
        lines.append(f"current top reported bundles: {bundle_lines}")

    if context.get("allocated_bundle") is not None:
        lines.append(
            f"current provisional allocated bundle: "
            f"{_fmt_bundle(context['allocated_bundle'])}"
        )
    if context.get("demanded_bundle") is not None:
        lines.append(
            f"current demanded bundle: {_fmt_bundle(context['demanded_bundle'])}"
        )
    if context.get("current_prices"):
        price_str = ", ".join(
            f"{item}={price}"
            for item, price in sorted(context["current_prices"].items())
        )
        lines.append(f"current prices: {price_str}")

    if context.get("best_large_bundle") is not None:
        lines.append(
            f"current best/high-value large bundle: "
            f"{_fmt_bundle(context['best_large_bundle'])}"
        )
    if context.get("near_tie_bundle") is not None:
        lines.append(
            f"close substitute/near-tie bundle: "
            f"{_fmt_bundle(context['near_tie_bundle'])}"
        )
    if context.get("contested_goods"):
        lines.append(
            f"goods currently contested or recently contested: "
            f"{sorted(context['contested_goods'])}"
        )

    recent_events = context.get("recent_events") or []
    if recent_events:
        event_lines = "; ".join(
            f"{e.get('event_type')} round={e.get('round_idx')} "
            f"bundle={_fmt_bundle(e.get('bundle'))}"
            for e in recent_events
        )
        lines.append(f"recent elicitation events: {event_lines}")

    recent_refinements = context.get("recent_refinements") or []
    if recent_refinements:
        refinement_lines = "; ".join(
            f"{_fmt_bundle(r.get('bundle'))}: "
            f"{r.get('old_value')}→{r.get('new_value')} ({r.get('reason')})"
            for r in recent_refinements
        )
        lines.append(f"recent refinement records: {refinement_lines}")

    resolved_hints = context.get("resolved_hints") or []
    if resolved_hints:
        lines.append(
            "distinctions already resolved by current demand/allocation "
            "(do NOT ask about these): " + "; ".join(resolved_hints)
        )

    context_section = "\n".join(lines)

    return f"""You are a bidding proxy performing a late-stage reflective
check-in with the person you represent, shortly before the auction
finalizes.

Scenario:
{scenario_description}

Available items:
{items_section}

Everything learned about this person so far:
{context_section}

Task: identify the single most important allocation-relevant preference
uncertainty that remains unresolved, and turn it into an explicit PAIRWISE
or MARGINAL comparison whose answer can be tested by one or two follow-up
value queries -- not a generic open-ended preference question. Choose one
``reflection_mode``:

- bundle_comparison: compare two allocation-relevant bundles (current
  allocation/demand vs. the nearest rejected alternative or a near-tie
  bundle; a high-value reported bundle vs. a plausible substitute).
- marginal_item_test: test the marginal importance of one item (a core
  bundle vs. that bundle plus one more item; a large bundle vs. its core
  subset; a bundle with a contested item vs. the same bundle without it).

Rules:
- Do NOT ask about a distinction already resolved by the current demand or
  allocation (see the list above, if present), and do NOT ask a generic
  question ("what else do you care about?").
- ``followup_bundles`` must correspond directly to the comparison --
  normally exactly ``[primary_bundle, comparison_bundle]``. Never choose a
  follow-up bundle unrelated to the distinction asked in the question.

Good example: "Your current report suggests the full bundle
[CPU_HI, GPU_AI, RAM_64, SSD_2TB] is much more valuable than the core
[CPU_HI, GPU_AI, RAM_64]. Is the extra storage essential, or would the core
already satisfy most of your needs?" -> reflection_mode: bundle_comparison,
primary_bundle: [CPU_HI, GPU_AI, RAM_64], comparison_bundle:
[CPU_HI, GPU_AI, RAM_64, SSD_2TB], followup_bundles: both of the above.

Bad example: "What else do you care about?" -- generic, no comparison pair,
cannot be mapped to a follow-up value query.

Also decide whether the answer should trigger a machine-actionable
follow-up: a value_query (re-price the comparison bundles) or a
demand_query (ask whether the person is satisfied with a bundle at current
prices), or none if no follow-up is warranted.

Return JSON only in exactly this schema:
{{
  "question": "<one concise pairwise/marginal comparison question>",
  "reason": "<why this uncertainty matters for the allocation>",
  "reflection_mode": "bundle_comparison | marginal_item_test",
  "target_type": "large_bundle_check | complementarity_check | substitute_check | missing_bundle | final_allocation_acceptability | other",
  "primary_bundle": ["ITEM_ID_1", "ITEM_ID_2"],
  "comparison_bundle": ["ITEM_ID_1", "ITEM_ID_2", "ITEM_ID_3"],
  "marginal_item": "ITEM_ID_3 or null",
  "suggested_followup": "none | value_query | demand_query",
  "followup_bundles": [["ITEM_ID_1", "ITEM_ID_2"], ["ITEM_ID_1", "ITEM_ID_2", "ITEM_ID_3"]]
}}

Use only item IDs from the available items list. Return only valid JSON.
Do not wrap it in markdown. Do not include any text outside the JSON
object."""


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
