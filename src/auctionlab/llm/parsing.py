from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from auctionlab.auction_types import Bundle
from auctionlab.llm.schemas import (
    LlmDemandQueryResponse,
    LlmInterestMap,
    LlmNaturalLanguageAnswer,
    LlmProvisionalValuations,
    LlmProxyQuestionResponse,
    LlmSummaryUpdateResponse,
    LlmValueResponse,
)


_BUNDLE_VALUE_PATTERN = re.compile(
    r"bundle\s+value\s*(?::|is)\s*\$?\s*"
    r"(?P<value>[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))",
    flags=re.IGNORECASE,
)
# Recover bundle_value from truncated JSON (thinking tokens eat max_tokens budget,
# leaving a valid partial JSON that raw_decode can't complete).
_SATISFIED_PATTERN = re.compile(
    r'"satisfied"\s*:\s*(?P<value>true|false)',
    flags=re.IGNORECASE,
)
_BUNDLE_VALUE_KEY_PATTERN = re.compile(
    r'"bundle_value"\s*:\s*(?P<value>[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))',
)
# Secondary fallback: base_value_from_anchors appears before bundle_value in the
# prompt schema, so it may be the only numeric field present when truncation is
# severe (e.g. cut off mid-synergy_adjustment field).
_BASE_VALUE_KEY_PATTERN = re.compile(
    r'"base_value_from_anchors"\s*:\s*(?P<value>[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))',
)
_QUESTION_PATTERN = re.compile(
    r'"question"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"',
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_json_object(raw: str) -> dict[str, Any]:
    """
    Extract the first valid JSON object from plain or markdown-fenced text.
    """
    decoder = json.JSONDecoder()

    for index, character in enumerate(raw):
        if character != "{":
            continue

        try:
            parsed, _end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed

    raise ValueError("No valid JSON object found")


def parse_value_response(raw: str) -> LlmValueResponse:
    """
    Parse a structured or human-readable bundle value response.
    """
    try:
        payload = extract_json_object(raw)
    except ValueError:
        # Priority order: JSON key in truncated response → NL pattern → anchor fallback
        match = (
            _BUNDLE_VALUE_KEY_PATTERN.search(raw)
            or _BUNDLE_VALUE_PATTERN.search(raw)
            or _BASE_VALUE_KEY_PATTERN.search(raw)
        )
        if match is None:
            raise ValueError("Could not parse bundle value response") from None

        payload = {
            "bundle_value": float(match.group("value").replace(",", "")),
        }

    try:
        return LlmValueResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Bundle value must be a non-negative number") from exc


def parse_demand_query_response(raw: str) -> LlmDemandQueryResponse:
    """Parse a Q_D (demand-query) response."""
    try:
        payload = extract_json_object(raw)
    except ValueError:
        match = _SATISFIED_PATTERN.search(raw)
        if match is None:
            raise ValueError("Could not parse demand query response") from None
        payload = {"satisfied": match.group("value").lower() == "true"}

    try:
        return LlmDemandQueryResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Invalid demand query response") from exc


def parse_natural_language_response(raw: str) -> LlmNaturalLanguageAnswer:
    """Parse a Q_N (natural-language) answer response."""
    try:
        payload = extract_json_object(raw)
    except ValueError:
        raise ValueError("Could not parse natural language response") from None

    try:
        return LlmNaturalLanguageAnswer.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Invalid natural language response") from exc


def parse_proxy_question_response(raw: str) -> LlmProxyQuestionResponse:
    """Parse the proxy's initial preference-elicitation question."""
    try:
        payload = extract_json_object(raw)
    except ValueError:
        match = _QUESTION_PATTERN.search(raw)
        if match is not None:
            payload = {"question": match.group("value")}
        else:
            # Model returned plain text instead of JSON — treat the whole
            # response as the question (common when gemini omits the wrapper).
            stripped = raw.strip()
            if not stripped:
                raise ValueError("Could not parse proxy question response") from None
            payload = {"question": stripped}

    try:
        return LlmProxyQuestionResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Invalid proxy question response") from exc


def parse_summary_update_response(raw: str) -> LlmSummaryUpdateResponse:
    """Parse the rolling preference summary update from the LLM."""
    try:
        payload = extract_json_object(raw)
    except ValueError:
        raise ValueError("Could not parse summary update response") from None

    try:
        return LlmSummaryUpdateResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Invalid summary update response") from exc


def parse_interest_map_response(
    raw: str,
    known_item_ids: set[str],
) -> LlmInterestMap:
    """Parse and validate an LLM-generated interest map.

    Unknown item IDs are silently filtered from all fields so that a noisy LLM
    response cannot introduce phantom items into the candidate bundle set.
    """
    try:
        payload = extract_json_object(raw)
    except ValueError:
        raise ValueError("Could not parse interest map response") from None

    try:
        parsed = LlmInterestMap.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Invalid interest map response") from exc

    def _filter(ids: list[str]) -> list[str]:
        return [i for i in ids if i in known_item_ids]

    def _filter_group(group: list[str]) -> list[str]:
        return [i for i in group if i in known_item_ids]

    return LlmInterestMap(
        interested_items=_filter(parsed.interested_items),
        excluded_items=_filter(parsed.excluded_items),
        complementary_groups=[
            filtered
            for group in parsed.complementary_groups
            if len(filtered := _filter_group(group)) >= 2
        ],
        substitute_groups=[
            filtered
            for group in parsed.substitute_groups
            if len(filtered := _filter_group(group)) >= 2
        ],
        budget_hint=parsed.budget_hint,
        reasoning=parsed.reasoning,
    )


def parse_provisional_valuations_response(
    raw: str,
    candidate_bundles: list[Bundle],
) -> dict[Bundle, float]:
    """Parse an LLM provisional-valuation response into a bundle → value map.

    Matches each returned entry to ``candidate_bundles`` by frozenset equality.
    Unmatched entries are silently ignored; bundles missing from the response
    default to 0.0.
    """
    try:
        payload = extract_json_object(raw)
    except ValueError:
        raise ValueError("Could not parse provisional valuations response") from None

    try:
        parsed = LlmProvisionalValuations.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Invalid provisional valuations response") from exc

    lookup: dict[frozenset[str], float] = {}
    for entry in parsed.valuations:
        key = frozenset(entry.bundle)
        lookup[key] = entry.value

    return {
        bundle: lookup.get(frozenset(bundle), 0.0)
        for bundle in candidate_bundles
        if bundle  # skip empty bundle
    }


def validate_queried_bundle(
    parsed: LlmValueResponse,
    expected_bundle: Bundle,
) -> None:
    queried_bundle = parsed.queried_bundle
    if queried_bundle is None:
        return

    if any(not isinstance(item, str) for item in queried_bundle):
        raise ValueError("queried_bundle must contain only string item IDs")
    if len(set(queried_bundle)) != len(queried_bundle):
        raise ValueError("queried_bundle must not contain duplicate item IDs")

    actual = frozenset(queried_bundle)
    expected = frozenset(expected_bundle)
    if actual != expected:
        raise ValueError(
            "queried_bundle does not match expected bundle: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
