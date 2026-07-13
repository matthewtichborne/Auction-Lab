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

"Valid" means: no substitute_group contributes two or more members to the
bundle. An optional ``max_candidate_bundles`` count cap trims the list after
prioritisation, so the most structurally important bundles are always retained.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import time

from auctionlab.auction_types import Bundle, Item
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.logging import LlmCallLogger, LlmCallRecord, current_timestamp
from auctionlab.llm.parsing import parse_interest_map_response
from auctionlab.llm.prompts import build_interest_map_prompt
from auctionlab.llm.schemas import LlmInterestMap


_IM_MAX_ATTEMPTS = 3


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
) -> LlmInterestMap:
    """Ask the proxy LLM to extract an interest map from a recorded Q/A pair.

    This is a proxy-side inference call — the person never sees this prompt.
    ``client`` is typically the same LLM client used by the proxy (not the
    person simulator), but any :class:`~auctionlab.llm.clients.LlmClient`
    works.

    Retries up to ``_IM_MAX_ATTEMPTS`` times before falling back to a
    conservative interest map that treats all items as of interest.
    """
    prompt = build_interest_map_prompt(
        scenario_description=scenario_description,
        item_descriptions=item_descriptions,
        nl_question=nl_question,
        nl_answer=nl_answer,
    )
    known_ids = set(item_descriptions.keys())
    label = bidder_id or "?"

    last_exc: Exception | None = None
    for attempt in range(1, _IM_MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        raw = client.complete(prompt)
        latency = time.perf_counter() - started

        try:
            result = parse_interest_map_response(raw, known_ids)
        except ValueError as exc:
            last_exc = exc
            warnings.warn(
                f"Interest map for {label}: parse failed on attempt {attempt}/{_IM_MAX_ATTEMPTS} "
                f"({exc}). {'Retrying...' if attempt < _IM_MAX_ATTEMPTS else 'Using fallback.'}",
                UserWarning,
                stacklevel=2,
            )
            if attempt < _IM_MAX_ATTEMPTS:
                continue

            result = _fallback_interest_map(item_descriptions)
            if logger is not None:
                logger.log(
                    LlmCallRecord(
                        timestamp=current_timestamp(),
                        bidder_id=bidder_id,
                        prompt_type="proxy_interest_map",
                        prompt=prompt,
                        raw_response=raw,
                        parsed_response=result.model_dump(),
                        success=False,
                        error=str(last_exc),
                        latency_seconds=latency,
                        model=model_name,
                        attempt=attempt,
                        input_tokens=getattr(client, "_last_input_tokens", None),
                        output_tokens=getattr(client, "_last_output_tokens", None),
                        total_tokens=getattr(client, "_last_total_tokens", None),
                    )
                )
            return result

        if logger is not None:
            logger.log(
                LlmCallRecord(
                    timestamp=current_timestamp(),
                    bidder_id=bidder_id,
                    prompt_type="proxy_interest_map",
                    prompt=prompt,
                    raw_response=raw,
                    parsed_response=result.model_dump(),
                    success=True,
                    error=None,
                    latency_seconds=latency,
                    model=model_name,
                    attempt=attempt,
                    input_tokens=getattr(client, "_last_input_tokens", None),
                    output_tokens=getattr(client, "_last_output_tokens", None),
                    total_tokens=getattr(client, "_last_total_tokens", None),
                )
            )
        return result

    # Unreachable — loop always returns — but satisfies type checker.
    return _fallback_interest_map(item_descriptions)


def generate_candidate_bundles_from_interest_map(
    interest_map: LlmInterestMap,
    all_items: list[Item],
    *,
    max_candidate_bundles: int | None = None,
) -> list[Bundle]:
    """Derive a priority-ordered candidate bundle list from an interest map.

    Filtering rules:
    - Only subsets of ``interest_map.interested_items`` are considered.
    - Bundles containing two or more items from the same ``substitute_group``
      are excluded.

    Priority ordering (stable within each tier):
    1. Complete ``complementary_groups`` bundles (explicit structure, highest value)
    2. Singletons of all ``interested_items``
    3. All other valid subsets in ascending size order

    If ``max_candidate_bundles`` is set, the list is trimmed after sorting so
    that the experimentally cheapest cap preserves the most important structure.
    The cap can be removed for a full paper run by passing ``None``.

    Returns an empty list when ``interest_map.interested_items`` is empty.
    """
    interested = frozenset(interest_map.interested_items) & frozenset(all_items)

    if not interested:
        return []

    if len(interested) > 7:
        n_bundles = 2 ** len(interested) - 1
        warnings.warn(
            f"Interest map has {len(interested)} interested items, producing up to "
            f"{n_bundles} candidate bundles before filtering. Consider setting "
            "max_candidate_bundles to limit LLM query cost.",
            stacklevel=2,
        )

    substitute_groups = [
        frozenset(g) & interested
        for g in interest_map.substitute_groups
    ]
    substitute_groups = [g for g in substitute_groups if len(g) >= 2]

    complement_sets = {
        frozenset(g) & interested
        for g in interest_map.complementary_groups
    }
    complement_sets = {g for g in complement_sets if len(g) >= 2}

    def _is_valid(bundle: Bundle) -> bool:
        for group in substitute_groups:
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

    if max_candidate_bundles is not None:
        valid_bundles = valid_bundles[:max_candidate_bundles]

    return valid_bundles
