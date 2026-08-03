"""Interest map derivation and interest-map-driven candidate bundle generation.

``derive_interest_map`` is a proxy-side inference step: given a person's NL
answer to the initial preference question, it asks the proxy LLM to extract a
structured :class:`~auctionlab.llm.schemas.LlmInterestMap` capturing which
items the person is interested in, which are substitutes, and which are
complements.

``generate_candidate_bundles_from_interest_map`` converts an ``LlmInterestMap``
into a priority-ordered list of candidate bundles:

  Priority 1 — explicit complementary_groups (full sets first)
  Priority 2 — singletons of interested_items
  Priority 3 — remaining valid subsets of interested_items, ascending by size

"Valid" means: no explicitly inferred ``choose_one`` substitute group
contributes two or more members to the bundle. ``can_use_multiple`` and
``unclear`` groups are retained conservatively. An optional
``max_candidate_bundles`` count cap trims the list after prioritisation.
"""

from __future__ import annotations

import warnings
from itertools import combinations
import re
import time
from typing import Any, Literal, Mapping, Sequence

from auctionlab.auction_types import Bundle, Item
from auctionlab.llm.cache import call_client
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.logging import LlmCallLogger, LlmCallRecord, current_timestamp
from auctionlab.llm.parsing import (
    parse_complement_entailment_response,
    parse_interest_map_response,
)
from auctionlab.llm.prompts import (
    build_complement_entailment_prompt,
    build_interest_map_prompt,
)
from auctionlab.llm.schemas import (
    LlmComplementGroupEvidence,
    LlmInterestMap,
    LlmSubstituteGroup,
)


_IM_MAX_ATTEMPTS = 3
InterestMapFailurePolicy = Literal["raise", "all_items"]


class InterestMapDerivationError(RuntimeError):
    """Raised when interest-map inference exhausts its parse attempts."""

    def __init__(
        self,
        *,
        bidder_id: str | None,
        attempts: int,
        original_exception: Exception,
        scenario_id: str | None = None,
    ) -> None:
        self.bidder_id = bidder_id
        self.attempts = attempts
        self.original_exception = original_exception
        self.scenario_id = scenario_id
        message = (
            "Interest-map derivation failed "
            f"(bidder_id={bidder_id!r}, scenario_id={scenario_id!r}, "
            f"attempts={attempts}, original_exception="
            f"{type(original_exception).__name__}: {original_exception}). "
            "Inspect the raw proxy_interest_map call. Use "
            "--interest-map-failure-policy all_items only for degraded debugging."
        )
        super().__init__(message)


def _unique_sorted(items: list[Item]) -> list[Item]:
    return sorted(set(items))


def _normalise_groups(groups: list[list[Item]]) -> list[list[Item]]:
    unique = {
        tuple(sorted(set(group)))
        for group in groups
        if len(set(group)) >= 2
    }
    return [list(group) for group in sorted(unique)]


def _normalise_substitute_groups(
    groups: list[LlmSubstituteGroup],
    known_items: set[Item],
    *,
    diagnostics: list[str],
) -> list[LlmSubstituteGroup]:
    """Deduplicate groups and resolve conflicting modes conservatively."""
    grouped: dict[tuple[str, ...], list[LlmSubstituteGroup]] = {}
    for group in groups:
        items = tuple(sorted(set(group.items) & known_items))
        if len(items) < 2:
            continue
        grouped.setdefault(items, []).append(group)

    result: list[LlmSubstituteGroup] = []
    for items, duplicates in sorted(grouped.items()):
        modes = {group.acquisition_mode for group in duplicates}
        mode = next(iter(modes)) if len(modes) == 1 else "unclear"
        if len(modes) > 1:
            diagnostics.append("conflicting_substitute_modes_made_unclear")
        explicit_values = {
            group.mode_explicitly_stated for group in duplicates
        }
        mode_explicitly_stated = (
            False
            if False in explicit_values
            else True
            if True in explicit_values
            else None
        )
        if mode == "unclear" and mode_explicitly_stated is True:
            mode_explicitly_stated = False
        evidence = "; ".join(
            dict.fromkeys(group.evidence.strip() for group in duplicates)
        )
        result.append(
            LlmSubstituteGroup(
                items=list(items),
                acquisition_mode=mode,
                evidence=evidence or "No explicit evidence retained.",
                mode_explicitly_stated=mode_explicitly_stated,
            )
        )
    return result


def normalise_interest_map(
    interest_map: LlmInterestMap,
    known_items: set[Item],
    *,
    strict: bool = False,
    diagnostics: list[str] | None = None,
) -> LlmInterestMap:
    """Validate and deterministically normalise the existing simple schema.

    ``diagnostics`` is an optional output list for non-fatal repairs. It keeps
    quality information out of :class:`LlmInterestMap` itself.
    """
    flags = diagnostics if diagnostics is not None else []
    mentioned = (
        list(interest_map.interested_items)
        + list(interest_map.excluded_items)
        + [item for group in interest_map.complementary_groups for item in group]
        + [
            item
            for group in interest_map.complementary_group_evidence
            for item in group.items
        ]
        + [
            item
            for group in interest_map.substitute_groups
            for item in group.items
        ]
    )
    unknown = sorted(set(mentioned) - known_items)
    if unknown:
        if strict:
            raise ValueError(f"Interest map contains unknown item IDs: {unknown}")
        flags.append("unknown_items_removed")

    def known(items: list[Item]) -> list[Item]:
        return _unique_sorted([item for item in items if item in known_items])

    interested = set(known(interest_map.interested_items))
    excluded = set(known(interest_map.excluded_items))
    conflicts = interested & excluded
    if conflicts:
        if strict:
            raise ValueError(
                "Interest map classifies item IDs as both interested and "
                f"excluded: {sorted(conflicts)}"
            )
        interested -= conflicts
        flags.append("excluded_interested_conflict_normalised")

    complement_groups = _normalise_groups(
        [known(group) for group in interest_map.complementary_groups]
    )
    evidence_by_group: dict[
        tuple[str, ...], LlmComplementGroupEvidence
    ] = {}
    for evidence in interest_map.complementary_group_evidence:
        items = tuple(sorted(set(known(evidence.items))))
        if len(items) >= 2 and evidence.explicit_extra_joint_value:
            evidence_by_group[items] = evidence.model_copy(
                update={"items": list(items)}
            )
    substitute_groups = _normalise_substitute_groups(
        interest_map.substitute_groups,
        known_items,
        diagnostics=flags,
    )

    # Exclusion is authoritative everywhere.
    complement_groups = _normalise_groups(
        [[item for item in group if item not in excluded] for group in complement_groups]
    )
    substitute_groups = _normalise_substitute_groups(
        [
            group.model_copy(
                update={
                    "items": [
                        item for item in group.items if item not in excluded
                    ]
                }
            )
            for group in substitute_groups
        ],
        known_items,
        diagnostics=flags,
    )

    # A declared structural relationship implies relevance unless excluded.
    grouped = {
        item
        for group in complement_groups
        for item in group
    }
    grouped |= {
        item for group in substitute_groups for item in group.items
    }
    if grouped - interested:
        interested |= grouped
        flags.append("group_members_added_to_interested")

    # The opening disclosure is explicitly closed-world: every positively
    # valued item must be mentioned, so silence means no interest. Keeping
    # this deterministic avoids relying on the proxy LLM to redundantly
    # enumerate all catalogue complements in ``excluded_items``.
    unclassified = known_items - interested - excluded
    if unclassified:
        if strict:
            raise ValueError(
                "Interest map does not classify every available item ID; "
                f"missing: {sorted(unclassified)}"
            )
        excluded |= unclassified
        flags.append("unmentioned_items_added_to_excluded")

    consistent_complements: list[list[Item]] = []
    for complement in complement_groups:
        complement_set = set(complement)
        conflict = any(
            len(complement_set & set(substitute.items)) >= 2
            for substitute in substitute_groups
            if substitute.acquisition_mode == "choose_one"
        )
        if conflict:
            if strict:
                raise ValueError(
                    "Interest map has a complementary group that contains "
                    "multiple members of a substitute group: "
                    f"{complement}"
                )
            flags.append("conflicting_complement_substitute_group_dropped")
            continue
        consistent_complements.append(complement)
    consistent_evidence = [
        evidence_by_group[tuple(complement)]
        for complement in consistent_complements
        if tuple(complement) in evidence_by_group
    ]

    return LlmInterestMap(
        interested_items=sorted(interested),
        excluded_items=sorted(excluded),
        complementary_groups=consistent_complements,
        complementary_group_evidence=consistent_evidence,
        substitute_groups=substitute_groups,
        budget_hint=interest_map.budget_hint,
        reasoning=interest_map.reasoning,
    )


_CHOOSE_ONE_CUES = (
    "at most one",
    "only one",
    "choose one",
    "more than one provides no",
    "no additional value from more than one",
    "no meaningful additional benefit",
)
_CAN_USE_MULTIPLE_CUES = (
    "can use multiple",
    "could use multiple",
    "can use more than one",
    "could use more than one",
    "retain value",
    "resell",
    "inventory",
    "deploy",
    "redundancy",
)
_ALTERNATIVE_CUES = ("alternative", "fallback", "instead", "either", "if")
_COMPLEMENT_CUES = (
    "more valuable together than separately",
    "additional value together",
    "only useful together",
    "complete set",
)


def _answer_clauses(answer: str) -> list[str]:
    return [
        clause.strip().lower()
        for clause in re.split(r"(?<=[.!?;])\s+|\n+", answer)
        if clause.strip()
    ]


def _grounding_clause(
    answer: str,
    items: Sequence[Item],
) -> str | None:
    lowered_items = [str(item).lower() for item in items]
    return next(
        (
            clause
            for clause in _answer_clauses(answer)
            if all(item in clause for item in lowered_items)
        ),
        None,
    )


def acquisition_mode_is_grounded(
    answer: str,
    group: LlmSubstituteGroup,
) -> bool:
    """Return whether the disclosed text explicitly supports the group mode."""
    clause = _grounding_clause(answer, group.items)
    if clause is None:
        return False
    if group.acquisition_mode == "choose_one":
        return any(cue in clause for cue in _CHOOSE_ONE_CUES)
    if group.acquisition_mode == "can_use_multiple":
        return any(cue in clause for cue in _CAN_USE_MULTIPLE_CUES)
    return any(cue in clause for cue in _ALTERNATIVE_CUES)


def complement_group_is_grounded(
    answer: str,
    group: Sequence[Item],
) -> bool:
    """Return whether text states explicit complete-set synergy for a group."""
    clause = _grounding_clause(answer, group)
    return clause is not None and any(
        cue in clause for cue in _COMPLEMENT_CUES
    )


def validate_interest_map_semantics(
    interest_map: LlmInterestMap,
    *,
    known_items: set[Item],
    nl_answer: str,
) -> None:
    """Reject parseable maps that violate structural consistency rules.

    Natural-language grounding is intentionally not enforced with substring
    matching here. The proxy may use ordinary names and paraphrases, while
    hidden-ground-truth accuracy and the person-answer LLM verifier provide
    the semantic audit.
    """
    # Strict normalization supplies deterministic ID, coverage, overlap, and
    # complement/substitute consistency checks without silently repairing.
    normalise_interest_map(interest_map, known_items, strict=True)

    interested = set(interest_map.interested_items)
    excluded = set(interest_map.excluded_items)
    grouped = {
        item
        for group in interest_map.complementary_groups
        for item in group
    } | {
        item
        for group in interest_map.substitute_groups
        for item in group.items
    }
    inconsistent_group_items = sorted(grouped - interested)
    if inconsistent_group_items:
        raise ValueError(
            "Interest-map group members must also be in interested_items: "
            f"{inconsistent_group_items}"
        )
    excluded_group_items = sorted(grouped & excluded)
    if excluded_group_items:
        raise ValueError(
            "Excluded item IDs must not appear in structural groups: "
            f"{excluded_group_items}"
        )



def interest_map_quality_flags(
    interest_map: LlmInterestMap,
    all_items: list[Item] | set[Item],
    *,
    fallback_used: bool = False,
    normalisation_flags: list[str] | None = None,
    candidate_count_after_filter: int | None = None,
) -> list[str]:
    """Return lightweight, deterministic warning flags for an interest map."""
    flags = list(normalisation_flags or [])
    all_item_set = set(all_items)
    interested_count = len(set(interest_map.interested_items) & all_item_set)
    if all_item_set and interested_count == len(all_item_set):
        flags.append("all_items_interested")
    if interested_count > 7 or (
        len(all_item_set) >= 6 and interested_count / len(all_item_set) >= 0.75
    ):
        flags.append("many_interested_items")
    if interested_count >= 4 and not interest_map.substitute_groups:
        flags.append("many_interested_items_no_substitutes")
    if interested_count >= 4 and not any(
        group.acquisition_mode == "choose_one"
        for group in interest_map.substitute_groups
    ):
        flags.append("many_interested_items_no_choose_one_substitutes")
    if not interest_map.excluded_items:
        flags.append("no_excluded_items")
    if not interest_map.substitute_groups:
        flags.append("no_substitute_groups")
    if candidate_count_after_filter is not None and candidate_count_after_filter > 50:
        flags.append("large_candidate_support")
    if fallback_used:
        flags.append("fallback_used")
    return list(dict.fromkeys(flags))


def interest_map_candidate_counts(
    interest_map: LlmInterestMap,
    all_items: list[Item],
) -> tuple[int, int]:
    """Return subset counts before and after declared-substitute filtering."""
    normalised = normalise_interest_map(interest_map, set(all_items))
    interested = sorted(set(normalised.interested_items) & set(all_items))
    substitute_groups = [
        set(group.items) & set(interested)
        for group in normalised.substitute_groups
        if group.acquisition_mode == "choose_one"
        and len(set(group.items) & set(interested)) >= 2
    ]
    before = 2 ** len(interested) - 1
    after = 0
    for size in range(1, len(interested) + 1):
        for combo in combinations(interested, size):
            bundle = set(combo)
            if all(len(bundle & group) < 2 for group in substitute_groups):
                after += 1
    return before, after


def interest_map_accuracy(
    interest_map: LlmInterestMap,
    *,
    true_interested_items: set[Item],
    true_substitute_groups: Sequence[Mapping[str, Any]],
    true_complement_groups: Sequence[
        Sequence[Item] | Mapping[str, Any]
    ] = (),
    available_items: set[Item],
    nl_answer: str | None = None,
    singleton_values: Mapping[Item, float] | None = None,
) -> dict[str, Any]:
    """Score proxy inference against hidden truth without changing it.

    Group matching is deliberately exact by item set. The returned details
    make dangerous false exclusivity (true ``can_use_multiple`` inferred as
    ``choose_one``) directly auditable.
    """
    inferred_interested = set(interest_map.interested_items) & available_items
    true_interested = true_interested_items & available_items

    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 1.0

    true_groups: dict[tuple[str, ...], str] = {}
    for group in true_substitute_groups:
        items = tuple(sorted(set(group.get("items", ())) & available_items))
        if len(items) >= 2:
            true_groups[items] = str(group["acquisition_mode"])
    inferred_groups = {
        tuple(sorted(set(group.items) & available_items)):
        group.acquisition_mode
        for group in interest_map.substitute_groups
        if len(set(group.items) & available_items) >= 2
    }

    matched_item_sets = set(true_groups) & set(inferred_groups)
    correct_modes = {
        items
        for items in matched_item_sets
        if true_groups[items] == inferred_groups[items]
    }
    dangerous_details = []
    for inferred_items, inferred_mode in inferred_groups.items():
        if inferred_mode != "choose_one":
            continue
        inferred_set = set(inferred_items)
        for true_items, true_mode in true_groups.items():
            overlap = inferred_set & set(true_items)
            if true_mode == "can_use_multiple" and len(overlap) >= 2:
                dangerous_details.append(
                    {
                        "inferred_choose_one_group": list(inferred_items),
                        "true_can_use_multiple_group": list(true_items),
                        "conflicting_items": sorted(overlap),
                    }
                )
    dangerous_inferred_groups = {
        tuple(detail["inferred_choose_one_group"])
        for detail in dangerous_details
    }
    true_choose_one = {
        items for items, mode in true_groups.items() if mode == "choose_one"
    }
    inferred_choose_one = {
        items
        for items, mode in inferred_groups.items()
        if mode == "choose_one"
    }
    correctly_inferred_choose_one = true_choose_one & inferred_choose_one
    interested_overlap = true_interested & inferred_interested
    item_precision = _ratio(
        len(interested_overlap), len(inferred_interested)
    )
    item_recall = _ratio(
        len(interested_overlap), len(true_interested)
    )
    item_f1 = (
        2 * item_precision * item_recall / (item_precision + item_recall)
        if item_precision + item_recall
        else 0.0
    )
    true_complements: set[tuple[str, ...]] = set()
    for group in true_complement_groups:
        raw_items = (
            group.get("items", ())
            if isinstance(group, Mapping)
            else group
        )
        items = tuple(sorted(set(raw_items) & available_items))
        if len(items) >= 2:
            true_complements.add(items)
    inferred_complements = {
        tuple(sorted(set(group) & available_items))
        for group in interest_map.complementary_groups
        if len(set(group) & available_items) >= 2
    }
    matched_complements = true_complements & inferred_complements
    inferred_complement_evidence = {
        tuple(sorted(set(group.items) & available_items))
        for group in interest_map.complementary_group_evidence
        if group.explicit_extra_joint_value
    }

    def _candidate_set(
        interested_items: set[Item],
        groups: Mapping[tuple[str, ...], str],
    ) -> set[Bundle]:
        choose_one = [
            set(items)
            for items, mode in groups.items()
            if mode == "choose_one"
        ]
        ordered = sorted(interested_items)
        result: set[Bundle] = set()
        for size in range(1, len(ordered) + 1):
            for combo in combinations(ordered, size):
                bundle = frozenset(combo)
                if all(len(bundle & group) < 2 for group in choose_one):
                    result.add(bundle)
        return result

    oracle_candidates = _candidate_set(true_interested, true_groups)
    inferred_candidates = _candidate_set(
        inferred_interested, inferred_groups
    )
    matched_candidates = oracle_candidates & inferred_candidates
    missed_values = {
        item: float((singleton_values or {}).get(item, 0.0))
        for item in true_interested - inferred_interested
    }

    return {
        "item_precision": item_precision,
        "item_recall": item_recall,
        "item_f1": item_f1,
        "missed_positive_items": sorted(
            true_interested - inferred_interested
        ),
        "false_positive_items": sorted(
            inferred_interested - true_interested
        ),
        "group_item_set_precision": _ratio(
            len(matched_item_sets), len(inferred_groups)
        ),
        "group_item_set_recall": _ratio(
            len(matched_item_sets), len(true_groups)
        ),
        "mode_accuracy_on_matched_groups": (
            _ratio(len(correct_modes), len(matched_item_sets))
            if matched_item_sets
            else None
        ),
        "exact_group_and_mode_recall": _ratio(
            len(correct_modes), len(true_groups)
        ),
        "choose_one_precision": _ratio(
            len(correctly_inferred_choose_one),
            len(inferred_choose_one),
        ),
        "choose_one_recall": _ratio(
            len(correctly_inferred_choose_one),
            len(true_choose_one),
        ),
        "dangerous_false_exclusivity_count": len(
            dangerous_inferred_groups
        ),
        "dangerous_false_exclusivity_groups": [
            list(items) for items in sorted(dangerous_inferred_groups)
        ],
        "dangerous_false_exclusivity_details": dangerous_details,
        "missed_true_groups": [
            {"items": list(items), "acquisition_mode": true_groups[items]}
            for items in sorted(set(true_groups) - set(inferred_groups))
        ],
        "extra_inferred_groups": [
            {
                "items": list(items),
                "acquisition_mode": inferred_groups[items],
            }
            for items in sorted(set(inferred_groups) - set(true_groups))
        ],
        "complement_group_precision": _ratio(
            len(matched_complements), len(inferred_complements)
        ),
        "complement_group_recall": _ratio(
            len(matched_complements), len(true_complements)
        ),
        "missed_true_complement_groups": [
            list(items)
            for items in sorted(true_complements - inferred_complements)
        ],
        "extra_inferred_complement_groups": [
            list(items)
            for items in sorted(inferred_complements - true_complements)
        ],
        "substitute_evidence_coverage": _ratio(
            sum(bool(group.evidence.strip()) for group in interest_map.substitute_groups),
            len(interest_map.substitute_groups),
        ),
        "explicit_substitute_mode_coverage": _ratio(
            sum(
                group.mode_explicitly_stated is True
                for group in interest_map.substitute_groups
                if group.acquisition_mode != "unclear"
            ),
            sum(
                group.acquisition_mode != "unclear"
                for group in interest_map.substitute_groups
            ),
        ),
        "complement_evidence_coverage": _ratio(
            len(inferred_complements & inferred_complement_evidence),
            len(inferred_complements),
        ),
        "oracle_candidate_count": len(oracle_candidates),
        "inferred_candidate_count": len(inferred_candidates),
        "candidate_set_precision": _ratio(
            len(matched_candidates), len(inferred_candidates)
        ),
        "candidate_set_recall": _ratio(
            len(matched_candidates), len(oracle_candidates)
        ),
        "missed_oracle_candidate_count": len(
            oracle_candidates - inferred_candidates
        ),
        "extra_candidate_count": len(
            inferred_candidates - oracle_candidates
        ),
        "missed_positive_singleton_values": missed_values,
        "missed_positive_singleton_value_total": sum(
            missed_values.values()
        ),
        "missed_positive_singleton_value_max": max(
            missed_values.values(), default=0.0
        ),
    }


def _fallback_interest_map(item_descriptions: dict[Item, str]) -> LlmInterestMap:
    """Return a conservative fallback that treats all items as of interest."""
    return LlmInterestMap(
        interested_items=sorted(item_descriptions.keys()),
        excluded_items=[],
        complementary_groups=[],
        substitute_groups=[],
        budget_hint=None,
        reasoning=(
            "Fallback: all interest-map parse attempts failed. "
            "Assuming all items are of interest with no complement/substitute structure."
        ),
    )


def _build_interest_map_repair_prompt(
    original_prompt: str,
    raw_response: str,
    validation_error: str | None = None,
) -> str:
    """Request a compact schema-only retry after a malformed extraction."""
    error_section = (
        f"\nVALIDATION ERROR TO FIX:\n{validation_error}\n"
        if validation_error
        else ""
    )
    return f"""The previous interest-map response was empty, truncated, or invalid.
Extract the interest map from the ORIGINAL REQUEST again. Preserve weak
fallbacks as interested substitutes when appropriate. An excluded item must
not appear in interested_items or in a complement or substitute group.
Every available item must appear exactly once across interested_items and
excluded_items. Use choose_one or can_use_multiple only when the person's
answer explicitly supports that mode; otherwise use unclear. Do not infer
complementarity from ordinary parts of a setup.
{error_section}

Return compact JSON only with exactly these fields:
{{"interested_items":[],"excluded_items":[],"complementary_groups":[],
"complementary_group_evidence":[{{"items":[],"evidence":"",
"explicit_extra_joint_value":false}}],
"substitute_groups":[{{"items":[],"acquisition_mode":"unclear",
"evidence":"<words from answer>","mode_explicitly_stated":false}}],
"budget_hint":null,
"reasoning":"<one short sentence>"}}

Use only item IDs from the original request. Do not add fields, markdown
fences, or text outside the JSON object.

ORIGINAL REQUEST:
{original_prompt}

PREVIOUS INVALID RESPONSE:
{raw_response}"""


def _audit_proposed_complements(
    *,
    client: LlmClient,
    interest_map: LlmInterestMap,
    item_descriptions: Mapping[Item, str],
    nl_question: str,
    nl_answer: str,
    logger: LlmCallLogger | None,
    bidder_id: str | None,
    model_name: str | None,
    scenario_id: str | None,
) -> LlmInterestMap:
    """Truth-blind second pass; failure conservatively removes complements."""
    if not interest_map.complementary_groups:
        return interest_map
    evidence_by_items = {
        tuple(sorted(group.items)): group
        for group in interest_map.complementary_group_evidence
    }
    proposed = [
        {
            "items": sorted(group),
            "evidence": (
                evidence_by_items[tuple(sorted(group))].evidence
                if tuple(sorted(group)) in evidence_by_items
                else ""
            ),
        }
        for group in interest_map.complementary_groups
    ]
    prompt = build_complement_entailment_prompt(
        item_descriptions=item_descriptions,
        nl_question=nl_question,
        nl_answer=nl_answer,
        proposed_groups=proposed,
    )
    started = time.perf_counter()
    raw: str | None = None
    parsed = None
    error: str | None = None
    try:
        raw = call_client(
            client,
            prompt,
            call_type="interest_map_complement_entailment",
            scenario_id=scenario_id,
            bidder_id=bidder_id,
        )
        parsed = parse_complement_entailment_response(raw)
        accepted = {
            tuple(sorted(set(judgment.items)))
            for judgment in parsed.judgments
            if judgment.entailed
        }
    except Exception as exc:
        error = str(exc)
        accepted = set()
    latency = time.perf_counter() - started
    retained_groups = [
        group
        for group in interest_map.complementary_groups
        if tuple(sorted(group)) in accepted
    ]
    retained_evidence = [
        evidence
        for evidence in interest_map.complementary_group_evidence
        if tuple(sorted(evidence.items)) in accepted
    ]
    result = interest_map.model_copy(update={
        "complementary_groups": retained_groups,
        "complementary_group_evidence": retained_evidence,
    })
    if logger is not None:
        logger.log(LlmCallRecord(
            timestamp=current_timestamp(),
            bidder_id=bidder_id,
            prompt_type="proxy_interest_map_complement_entailment",
            prompt=prompt,
            raw_response=raw,
            parsed_response={
                "judgments": (
                    parsed.model_dump()["judgments"] if parsed is not None else []
                ),
                "retained_groups": retained_groups,
            },
            success=error is None,
            error=error,
            latency_seconds=latency,
            model=model_name,
            provider=getattr(client, "_auctionlab_provider", None),
            llm_role=getattr(client, "_auctionlab_llm_role", "proxy"),
            attempt=1,
            input_tokens=getattr(client, "_last_input_tokens", None),
            output_tokens=getattr(client, "_last_output_tokens", None),
            total_tokens=getattr(client, "_last_total_tokens", None),
            cache_hit=getattr(client, "_last_cache_hit", None),
            finish_reason=getattr(client, "_last_finish_reason", None),
            response_char_count=len(raw) if raw is not None else None,
        ))
    return result


def derive_interest_map(
    *,
    client: LlmClient,
    scenario_description: str,
    item_descriptions: dict[Item, str],
    nl_question: str,
    nl_answer: str,
    logger: LlmCallLogger | None = None,
    bidder_id: str | None = None,
    model_name: str | None = None,
    scenario_id: str | None = None,
    failure_policy: InterestMapFailurePolicy = "all_items",
) -> LlmInterestMap:
    """Ask the proxy LLM to extract an interest map from a recorded Q/A pair.

    This is a proxy-side inference call — the person never sees this prompt.
    ``client`` is typically the same LLM client used by the proxy (not the
    person simulator), but any :class:`~auctionlab.llm.clients.LlmClient`
    works. ``scenario_id``, if given, is threaded into cache rows only (see
    :mod:`auctionlab.llm.cache`) and has no effect unless ``client`` is a
    :class:`~auctionlab.llm.cache.CachingLlmClient`.

    Retries up to ``_IM_MAX_ATTEMPTS`` times. ``failure_policy="all_items"``
    preserves the historical degraded fallback; ``"raise"`` fails closed.
    """
    if failure_policy not in ("raise", "all_items"):
        raise ValueError(
            "failure_policy must be 'raise' or 'all_items', "
            f"got {failure_policy!r}"
        )
    original_prompt = build_interest_map_prompt(
        scenario_description=scenario_description,
        item_descriptions=item_descriptions,
        nl_question=nl_question,
        nl_answer=nl_answer,
    )
    prompt = original_prompt
    known_ids = set(item_descriptions.keys())
    label = bidder_id or "?"

    last_exc: Exception | None = None
    for attempt in range(1, _IM_MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        raw = call_client(
            client,
            prompt,
            call_type="interest_map",
            scenario_id=scenario_id,
            bidder_id=bidder_id,
            extra_key_fields=(
                {"parse_repair_attempt": attempt}
                if attempt > 1
                else None
            ),
        )
        latency = time.perf_counter() - started

        parsed_result: LlmInterestMap | None = None
        try:
            parsed_result = parse_interest_map_response(raw, known_ids)
            validate_interest_map_semantics(
                parsed_result,
                known_items=known_ids,
                nl_answer=nl_answer,
            )
            result = parsed_result
        except ValueError as exc:
            last_exc = exc
            warnings.warn(
                f"Interest map for {label}: parse failed on attempt {attempt}/{_IM_MAX_ATTEMPTS} "
                f"({exc}). "
                f"{'Retrying...' if attempt < _IM_MAX_ATTEMPTS else ('Raising.' if failure_policy == 'raise' else 'Using degraded all-items fallback.')}",
                UserWarning,
                stacklevel=2,
            )
            final_attempt = attempt == _IM_MAX_ATTEMPTS
            result = (
                _fallback_interest_map(item_descriptions)
                if final_attempt and failure_policy == "all_items"
                else None
            )
            if logger is not None:
                parsed_response = (
                    {
                        **result.model_dump(),
                        "_interest_map_diagnostics": {
                            "fallback_used": True,
                            "quality_flags": ["fallback_used"],
                            "interested_count": len(result.interested_items),
                            "excluded_count": len(result.excluded_items),
                            "substitute_group_count": len(result.substitute_groups),
                            "complement_group_count": len(result.complementary_groups),
                        },
                    }
                    if result is not None
                    else (
                        {
                            **parsed_result.model_dump(),
                            "_interest_map_diagnostics": {
                                "fallback_used": False,
                                "quality_flags": [
                                    "semantic_validation_failed"
                                ],
                            },
                        }
                        if parsed_result is not None
                        else None
                    )
                )
                logger.log(
                    LlmCallRecord(
                        timestamp=current_timestamp(),
                        bidder_id=bidder_id,
                        prompt_type="proxy_interest_map",
                        prompt=prompt,
                        raw_response=raw,
                        parsed_response=parsed_response,
                        success=False,
                        error=str(last_exc),
                        latency_seconds=latency,
                        model=model_name,
                        provider=getattr(client, "_auctionlab_provider", None),
                        llm_role=getattr(
                            client, "_auctionlab_llm_role", "proxy"
                        ),
                        attempt=attempt,
                        input_tokens=getattr(client, "_last_input_tokens", None),
                        output_tokens=getattr(client, "_last_output_tokens", None),
                        total_tokens=getattr(client, "_last_total_tokens", None),
                        cached_input_tokens=getattr(
                            client, "_last_cached_input_tokens", None
                        ),
                        cached_output_tokens=getattr(
                            client, "_last_cached_output_tokens", None
                        ),
                        cache_hit=getattr(client, "_last_cache_hit", None),
                        finish_reason=getattr(
                            client, "_last_finish_reason", None
                        ),
                        response_char_count=len(raw),
                    )
                )
            if not final_attempt:
                prompt = _build_interest_map_repair_prompt(
                    original_prompt,
                    raw,
                    str(last_exc),
                )
                continue
            if failure_policy == "raise":
                raise InterestMapDerivationError(
                    bidder_id=bidder_id,
                    attempts=_IM_MAX_ATTEMPTS,
                    original_exception=last_exc,
                    scenario_id=scenario_id,
                ) from last_exc
            return result

        normalisation_flags: list[str] = []
        result = normalise_interest_map(
            result,
            known_ids,
            diagnostics=normalisation_flags,
        )
        quality_flags = interest_map_quality_flags(
            result,
            known_ids,
            normalisation_flags=normalisation_flags,
        )
        if logger is not None:
            logger.log(
                LlmCallRecord(
                    timestamp=current_timestamp(),
                    bidder_id=bidder_id,
                    prompt_type="proxy_interest_map",
                    prompt=prompt,
                    raw_response=raw,
                    parsed_response={
                        **result.model_dump(),
                        "_interest_map_diagnostics": {
                            "fallback_used": False,
                            "quality_flags": quality_flags,
                            "interested_count": len(result.interested_items),
                            "excluded_count": len(result.excluded_items),
                            "substitute_group_count": len(result.substitute_groups),
                            "complement_group_count": len(result.complementary_groups),
                        },
                    },
                    success=True,
                    error=None,
                    latency_seconds=latency,
                    model=model_name,
                    provider=getattr(client, "_auctionlab_provider", None),
                    llm_role=getattr(
                        client, "_auctionlab_llm_role", "proxy"
                    ),
                    attempt=attempt,
                    input_tokens=getattr(client, "_last_input_tokens", None),
                    output_tokens=getattr(client, "_last_output_tokens", None),
                    total_tokens=getattr(client, "_last_total_tokens", None),
                    cached_input_tokens=getattr(
                        client, "_last_cached_input_tokens", None
                    ),
                    cached_output_tokens=getattr(
                        client, "_last_cached_output_tokens", None
                    ),
                    cache_hit=getattr(client, "_last_cache_hit", None),
                    finish_reason=getattr(client, "_last_finish_reason", None),
                    response_char_count=len(raw),
                )
            )
        return _audit_proposed_complements(
            client=client,
            interest_map=result,
            item_descriptions=item_descriptions,
            nl_question=nl_question,
            nl_answer=nl_answer,
            logger=logger,
            bidder_id=bidder_id,
            model_name=model_name,
            scenario_id=scenario_id,
        )

    raise AssertionError("interest-map retry loop exited unexpectedly")


def generate_candidate_bundles_from_interest_map(
    interest_map: LlmInterestMap,
    all_items: list[Item],
    *,
    max_candidate_bundles: int | None = None,
) -> list[Bundle]:
    """Derive a priority-ordered candidate bundle list from an interest map.

    Filtering rules:
    - Only subsets of ``interest_map.interested_items`` are considered.
    - Bundles containing two or more items from an explicitly inferred
      ``choose_one`` substitute group are excluded.
    - ``can_use_multiple`` and ``unclear`` groups do not remove bundles.

    Priority ordering (stable within each tier):
    1. Complete ``complementary_groups`` bundles (explicit structure, highest value)
    2. Singletons of all ``interested_items``
    3. All other valid subsets in ascending size order

    If ``max_candidate_bundles`` is set, the list is trimmed after sorting so
    that the experimentally cheapest cap preserves the most important structure.
    The cap can be removed for a full paper run by passing ``None``.

    Returns an empty list when ``interest_map.interested_items`` is empty.
    """
    normalisation_flags: list[str] = []
    interest_map = normalise_interest_map(
        interest_map,
        set(all_items),
        diagnostics=normalisation_flags,
    )
    interested = frozenset(interest_map.interested_items) & frozenset(all_items)

    if not interested:
        return []

    choose_one_groups = [
        frozenset(group.items) & interested
        for group in interest_map.substitute_groups
        if group.acquisition_mode == "choose_one"
    ]
    choose_one_groups = [
        group for group in choose_one_groups if len(group) >= 2
    ]

    complement_sets = {
        frozenset(g) & interested
        for g in interest_map.complementary_groups
    }
    complement_sets = {g for g in complement_sets if len(g) >= 2}

    def _is_valid(bundle: Bundle) -> bool:
        for group in choose_one_groups:
            if len(bundle & group) >= 2:
                return False
        return True

    def _priority_key(bundle: Bundle) -> tuple[int, int, tuple[str, ...]]:
        if bundle in complement_sets:
            return (0, len(bundle), tuple(sorted(bundle)))
        if len(bundle) == 1:
            return (1, 1, tuple(sorted(bundle)))
        return (2, len(bundle), tuple(sorted(bundle)))

    sorted_items = sorted(interested)
    valid_bundles: list[Bundle] = []

    for size in range(1, len(sorted_items) + 1):
        for combo in combinations(sorted_items, size):
            bundle = frozenset(combo)
            if _is_valid(bundle):
                valid_bundles.append(bundle)

    valid_bundles.sort(key=_priority_key)

    candidate_count_before_filter = 2 ** len(interested) - 1
    candidate_count_after_filter = len(valid_bundles)
    quality_flags = interest_map_quality_flags(
        interest_map,
        all_items,
        normalisation_flags=normalisation_flags,
        candidate_count_after_filter=candidate_count_after_filter,
    )
    repair_flags = {
        "unknown_items_removed",
        "excluded_interested_conflict_normalised",
        "group_members_added_to_interested",
        "conflicting_complement_substitute_group_dropped",
    } & set(quality_flags)
    if repair_flags:
        warnings.warn(
            "Interest map was normalised before candidate generation: "
            + ", ".join(sorted(repair_flags)),
            UserWarning,
            stacklevel=2,
        )
    weak_filtering = (
        candidate_count_before_filter > 0
        and candidate_count_after_filter / candidate_count_before_filter
        >= 0.7
    )
    if "many_interested_items" in quality_flags and weak_filtering:
        warnings.warn(
            "Broad/weak interest map: "
            f"interested_count={len(interested)}, "
            f"candidate_count_before_filter={candidate_count_before_filter}, "
            f"candidate_count_after_substitute_filter={candidate_count_after_filter}, "
            f"choose_one_substitute_group_count={len(choose_one_groups)}. "
            "This may indicate over-inclusion or missed substitute relations.",
            UserWarning,
            stacklevel=2,
        )
    elif "large_candidate_support" in quality_flags:
        warnings.warn(
            "Large candidate support after valid interest/substitute "
            "filtering: "
            f"interested_count={len(interested)}, "
            f"candidate_count_before_filter={candidate_count_before_filter}, "
            f"candidate_count_after_substitute_filter="
            f"{candidate_count_after_filter}, "
            f"choose_one_substitute_group_count={len(choose_one_groups)}. "
            "This is a scale warning, not evidence that the inferred "
            "interest map is weak.",
            UserWarning,
            stacklevel=2,
        )

    if max_candidate_bundles is not None:
        valid_bundles = valid_bundles[:max_candidate_bundles]

    return valid_bundles
