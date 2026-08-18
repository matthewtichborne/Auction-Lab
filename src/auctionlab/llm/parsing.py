"""Parsing of structured model responses.

Responses are requested as JSON but arrive as text, so extraction scans for
the first object and decodes incrementally, which tolerates markdown fences
and surrounding prose. Exactly one repair is attempted, stripping trailing
commas; anything still unparsed is treated as a failure rather than guessed
at, since a wrong guess would silently alter the elicitation treatment.
"""

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
    LlmPersonAnswerSemanticExtraction,
    LlmComplementGroupEvidence,
    LlmComplementEntailmentResponse,
    LlmCompactProvisionalValuations,
    LlmProvisionalValuations,
    LlmProxyQuestionResponse,
    LlmSubstituteGroup,
    LlmSubstituteModeEntailmentResponse,
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


_TRAILING_COMMA_PATTERN = re.compile(r",(\s*[\]}])")


def _scan_for_json_object(raw: str) -> dict[str, Any] | None:
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

    return None


def extract_json_object(raw: str) -> dict[str, Any]:
    """
    Extract the first valid JSON object from plain or markdown-fenced text.

    Markdown fences and leading/trailing prose are already tolerated: the
    scan finds the first ``{`` and ``json.JSONDecoder.raw_decode`` only
    needs the object itself to be well-formed, not the rest of the string.
    If no object parses as-is, one safe repair is attempted -- stripping
    trailing commas before a closing ``]``/``}`` (a common, low-risk model
    formatting slip that never changes bundle/field identity) -- before
    giving up.
    """
    parsed = _scan_for_json_object(raw)
    if parsed is not None:
        return parsed

    repaired = _TRAILING_COMMA_PATTERN.sub(r"\1", raw)
    if repaired != raw:
        parsed = _scan_for_json_object(repaired)
        if parsed is not None:
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


def parse_person_answer_verification(
    raw: str,
) -> LlmPersonAnswerSemanticExtraction:
    """Parse a blind preparation-time person-answer extraction."""
    try:
        payload = extract_json_object(raw)
    except ValueError:
        raise ValueError(
            "Could not parse person-answer verification"
        ) from None
    try:
        return LlmPersonAnswerSemanticExtraction.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Invalid person-answer verification") from exc


def parse_substitute_mode_entailment_response(
    raw: str,
) -> LlmSubstituteModeEntailmentResponse:
    try:
        payload = extract_json_object(raw)
    except ValueError:
        raise ValueError(
            "Could not parse substitute-mode entailment response"
        ) from None
    try:
        return LlmSubstituteModeEntailmentResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            "Invalid substitute-mode entailment response"
        ) from exc


def parse_complement_entailment_response(
    raw: str,
) -> LlmComplementEntailmentResponse:
    try:
        payload = extract_json_object(raw)
    except ValueError:
        raise ValueError(
            "Could not parse complement-entailment response"
        ) from None
    try:
        return LlmComplementEntailmentResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Invalid complement-entailment response") from exc


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

    evidence_by_items = {
        tuple(sorted(set(group.items))): group
        for group in parsed.complementary_group_evidence
    }
    accepted_complements: list[list[str]] = []
    accepted_evidence: list[LlmComplementGroupEvidence] = []
    for group in parsed.complementary_groups:
        filtered = _filter_group(group)
        if len(filtered) < 2:
            continue
        key = tuple(sorted(set(filtered)))
        evidence = evidence_by_items.get(key)
        if evidence is None or not evidence.explicit_extra_joint_value:
            continue
        accepted_complements.append(filtered)
        accepted_evidence.append(
            evidence.model_copy(update={"items": list(key)})
        )

    return LlmInterestMap(
        interested_items=_filter(parsed.interested_items),
        excluded_items=_filter(parsed.excluded_items),
        complementary_groups=accepted_complements,
        complementary_group_evidence=accepted_evidence,
        substitute_groups=[
            LlmSubstituteGroup(
                items=filtered,
                acquisition_mode=(
                    group.acquisition_mode
                    if group.mode_explicitly_stated is True
                    else "unclear"
                ),
                evidence=group.evidence,
                mode_explicitly_stated=group.mode_explicitly_stated,
            )
            for group in parsed.substitute_groups
            if len(filtered := _filter_group(group.items)) >= 2
        ],
        budget_hint=parsed.budget_hint,
        reasoning=parsed.reasoning,
    )


def parse_provisional_valuations_response(
    raw: str,
    candidate_bundles: list[Bundle],
    *,
    strict_missing: bool = False,
    bidder_id: str | None = None,
    chunk_index: int | None = None,
) -> dict[Bundle, float]:
    """Parse an LLM provisional-valuation response into a bundle → value map.

    Matches each returned entry to ``candidate_bundles`` by frozenset equality.
    By default (``strict_missing=False``, unchanged from before this
    parameter existed) unmatched entries are silently ignored and bundles
    missing from the response default to 0.0.

    ``strict_missing=True`` (used by
    :func:`~auctionlab.llm.provisional_valuations.generate_provisional_valuations_chunked`
    so a partial/truncated per-chunk response fails loudly instead of
    silently defaulting missing bundles to 0.0) raises :exc:`ValueError`
    naming ``bidder_id``, ``chunk_index`` (when given), and the missing
    bundles if any requested bundle has no matching entry in the response.
    """
    try:
        payload = extract_json_object(raw)
    except ValueError:
        raise ValueError("Could not parse provisional valuations response") from None

    if "values" in payload:
        try:
            compact = LlmCompactProvisionalValuations.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("Invalid provisional valuations response") from exc
        if len(compact.values) != len(candidate_bundles):
            raise ValueError(
                "Compact PV response value count does not match requested "
                f"bundle count: received={len(compact.values)}, "
                f"requested={len(candidate_bundles)}"
            )
        return {
            bundle: float(value)
            for bundle, value in zip(candidate_bundles, compact.values)
            if bundle
        }

    # Backward compatibility for cached artefacts and older model responses.
    try:
        parsed = LlmProvisionalValuations.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Invalid provisional valuations response") from exc

    lookup: dict[frozenset[str], float] = {}
    for entry in parsed.valuations:
        key = frozenset(entry.bundle)
        lookup[key] = entry.value

    if strict_missing:
        requested = {frozenset(bundle) for bundle in candidate_bundles if bundle}
        missing = requested - set(lookup)
        if missing:
            missing_str = ", ".join(
                "{" + ",".join(sorted(bundle)) + "}"
                for bundle in sorted(missing, key=lambda b: (len(b), sorted(b)))
            )
            chunk_str = f", chunk_index={chunk_index}" if chunk_index is not None else ""
            raise ValueError(
                f"PV response for bidder_id={bidder_id!r}{chunk_str} is missing "
                f"valuations for {len(missing)} requested bundle(s): {missing_str}"
            )

    return {
        bundle: lookup.get(frozenset(bundle), 0.0)
        for bundle in candidate_bundles
        if bundle  # skip empty bundle
    }


_RAW_EXCERPT_MAX_LEN = 500

def raw_response_excerpt(raw: str | None, max_len: int = _RAW_EXCERPT_MAX_LEN) -> str:
    """Bound a raw LLM response to a diagnostic excerpt length."""
    if not raw:
        return ""
    single_line = " ".join(raw.split())
    if len(single_line) <= max_len:
        return single_line
    return single_line[:max_len] + "…[truncated]"


def validate_queried_bundle(
    parsed: LlmValueResponse,
    expected_bundle: Bundle,
    *,
    raw_response: str | None = None,
) -> None:
    """Validate that ``parsed.queried_bundle`` exactly matches ``expected_bundle``.

    The mechanism must never silently bind a value to the wrong bundle: an
    LLM that adds a complement, drops an unwanted item, duplicates an item,
    or substitutes one item for another must fail loudly here rather than
    have its answer accepted. ``raw_response``, when supplied, is included
    (as a bounded excerpt) in the error message so a live failure can be
    inspected without guessing what the model actually returned.
    """
    queried_bundle = parsed.queried_bundle
    if queried_bundle is None:
        return

    excerpt_suffix = (
        f" | raw_response_excerpt={raw_response_excerpt(raw_response)!r}"
        if raw_response is not None
        else ""
    )

    if any(not isinstance(item, str) for item in queried_bundle):
        raise ValueError(
            "queried_bundle must contain only string item IDs" + excerpt_suffix
        )

    has_duplicates = len(set(queried_bundle)) != len(queried_bundle)
    if has_duplicates:
        raise ValueError(
            "queried_bundle must not contain duplicate item IDs: "
            f"queried_bundle={queried_bundle}" + excerpt_suffix
        )

    actual = frozenset(queried_bundle)
    expected = frozenset(expected_bundle)
    if actual != expected:
        added = sorted(actual - expected)
        removed = sorted(expected - actual)
        raise ValueError(
            "queried_bundle does not match expected bundle: "
            f"expected={sorted(expected)}, actual={sorted(actual)}, "
            f"added={added}, removed={removed}, duplicates_detected="
            f"{has_duplicates}" + excerpt_suffix
        )
