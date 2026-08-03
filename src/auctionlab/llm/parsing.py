from __future__ import annotations

import json
import re
from typing import Any, get_args

from pydantic import ValidationError

from auctionlab.auction_types import Bundle
from auctionlab.llm.schemas import (
    LateReflectionMode,
    LlmDemandQueryResponse,
    LlmInterestMap,
    LlmLateReflectionResponse,
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


class LateReflectionParseError(ValueError):
    """Raised by :func:`parse_late_reflection_response` on unrecoverable failure.

    Carries a coarse ``error_type`` (one of ``missing_required_field``,
    ``invalid_json``, ``invalid_bundle``, ``truncated_output``, ``unknown``)
    and a bounded ``raw_excerpt`` of the offending response, so callers can
    log *why* parsing failed (and what the model actually said) instead of
    just recording a generic failure -- this was the missing diagnostic in
    the live 10x10 runs where every late-reflection row failed with the same
    opaque ``"Could not parse late reflection response"`` message.
    """

    def __init__(self, message: str, *, error_type: str, raw_excerpt: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.raw_excerpt = raw_excerpt


_RAW_EXCERPT_MAX_LEN = 500


def raw_response_excerpt(raw: str | None, max_len: int = _RAW_EXCERPT_MAX_LEN) -> str:
    """Bound a raw LLM response to a CSV-friendly excerpt length."""
    if not raw:
        return ""
    single_line = " ".join(raw.split())
    if len(single_line) <= max_len:
        return single_line
    return single_line[:max_len] + "…[truncated]"


def _looks_truncated(raw: str) -> bool:
    """Heuristic: an unbalanced object brace count usually means the
    provider's max-token cap cut the response off mid-object."""
    return raw.count("{") > raw.count("}")


_LR_FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "reflection_mode": re.compile(
        r'"reflection_mode"\s*:\s*"(?P<value>[a-zA-Z_]+)"'
    ),
    "target_type": re.compile(r'"target_type"\s*:\s*"(?P<value>[a-zA-Z_]+)"'),
    "suggested_followup": re.compile(
        r'"suggested_followup"\s*:\s*"(?P<value>[a-zA-Z_]+)"'
    ),
    "marginal_item": re.compile(r'"marginal_item"\s*:\s*"(?P<value>[^"]*)"'),
    "reason": re.compile(r'"reason"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"'),
}
_LR_BUNDLE_FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "primary_bundle": re.compile(
        r'"primary_bundle"\s*:\s*(?P<value>\[[^\]]*\]|"[^"]*")'
    ),
    "comparison_bundle": re.compile(
        r'"comparison_bundle"\s*:\s*(?P<value>\[[^\]]*\]|"[^"]*")'
    ),
}


def _regex_fallback_late_reflection_payload(raw: str) -> dict[str, Any] | None:
    """Best-effort field-by-field recovery when no complete JSON object exists.

    Mirrors :func:`parse_value_response`'s truncation-recovery pattern
    (regex-extracting ``bundle_value`` from a partial JSON tail): used only
    when :func:`extract_json_object` finds no valid JSON object at all --
    typically because the provider's ``max_tokens`` cap cut the response off
    mid-object. ``question`` is normally the first field in the prompt's
    schema, so it usually survives truncation even when later fields (the
    bundle arrays especially) do not; recovering just the question keeps the
    NL exchange usable even when no structured follow-up can be derived.
    Returns ``None`` (never a partial payload) if not even a question can be
    recovered -- a reflection row with no question text is useless.
    """
    question_match = _QUESTION_PATTERN.search(raw)
    if question_match is None:
        return None

    payload: dict[str, Any] = {"question": question_match.group("value")}
    for field_name, pattern in _LR_FIELD_PATTERNS.items():
        match = pattern.search(raw)
        if match is not None:
            payload[field_name] = match.group("value")
    for field_name, pattern in _LR_BUNDLE_FIELD_PATTERNS.items():
        match = pattern.search(raw)
        if match is not None:
            payload[field_name] = match.group("value")
    return payload


def _parse_bundle_string(value: str) -> list[str]:
    """Parse a bundle given as a brace/bracket string, e.g. ``"{A,B}"``."""
    text = value.strip()
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        text = text[1:-1]
    return [
        piece
        for raw_piece in text.split(",")
        if (piece := raw_piece.strip().strip('"').strip("'"))
    ]


def _normalize_single_bundle_field(value: Any) -> list[str] | None:
    """Coerce a ``primary_bundle``/``comparison_bundle``-shaped value to
    ``list[str] | None``, tolerating a bundle given as a ``"{A,B}"`` string."""
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_bundle_string(value) or None
    if isinstance(value, list):
        items = [str(item) for item in value]
        return items or None
    return None


def _normalize_marginal_item_field(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return str(value[0]) if value else None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        text = text.strip().strip('"').strip("'")
        return text or None
    return None


def _normalize_bundle_list_field(value: Any) -> list[list[str]]:
    """Coerce a ``target_bundles``/``followup_bundles``-shaped value to
    ``list[list[str]]``.

    Tolerates two common model mistakes: the whole field given as one
    bundle-string (``"{A,B}"`` instead of ``[["A","B"]]``), and a single
    bundle given as a flat list of items (``["A","B"]`` instead of
    ``[["A","B"]]``) where each element is itself a plain item ID rather
    than an encoded multi-item string.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parsed = _parse_bundle_string(value)
        return [parsed] if parsed else []
    if not isinstance(value, list):
        return []
    if not value:
        return []

    if all(isinstance(item, str) for item in value):
        # Ambiguous: a flat list of strings. If any element itself encodes
        # multiple items, treat each element as its own bundle; otherwise
        # the whole list is almost certainly ONE bundle mistakenly given as
        # a list of bundles (the exact "single bundle where a list of
        # bundles was expected" case this normalizer exists for).
        if any(("," in item or "{" in item or "[" in item) for item in value):
            return [_parse_bundle_string(item) for item in value if _parse_bundle_string(item)]
        return [[str(item) for item in value]]

    result: list[list[str]] = []
    for item in value:
        if isinstance(item, str):
            parsed = _parse_bundle_string(item)
            if parsed:
                result.append(parsed)
        elif isinstance(item, list):
            result.append([str(x) for x in item])
        # Silently skip anything else (e.g. a nested dict) -- unsafe to guess.
    return result


_VALID_REFLECTION_MODES: frozenset[str] = frozenset(get_args(LateReflectionMode))


def _normalize_late_reflection_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Pre-validation normalization: tolerant field shapes, safe scalar defaults.

    Runs *before* pydantic validation so a model's formatting quirks (a
    bundle given as a string, an unrecognised ``reflection_mode``, a missing
    ``target_type``) degrade gracefully into a valid (if less informative)
    payload instead of raising -- the actual ``reflection_mode`` recovery
    logic runs afterwards, in :func:`_apply_late_reflection_recovery`.
    """
    payload = dict(payload)

    if "primary_bundle" in payload:
        payload["primary_bundle"] = _normalize_single_bundle_field(
            payload.get("primary_bundle")
        )
    if "comparison_bundle" in payload:
        payload["comparison_bundle"] = _normalize_single_bundle_field(
            payload.get("comparison_bundle")
        )
    if "marginal_item" in payload:
        payload["marginal_item"] = _normalize_marginal_item_field(
            payload.get("marginal_item")
        )
    payload["target_bundles"] = _normalize_bundle_list_field(
        payload.get("target_bundles")
    )
    payload["followup_bundles"] = _normalize_bundle_list_field(
        payload.get("followup_bundles")
    )

    reflection_mode = payload.get("reflection_mode")
    if not (isinstance(reflection_mode, str) and reflection_mode in _VALID_REFLECTION_MODES):
        payload["reflection_mode"] = None

    if not payload.get("target_type"):
        payload["target_type"] = "other"
    if not payload.get("suggested_followup"):
        payload["suggested_followup"] = "none"

    return payload


def _bundles_are_one_item_apart(a: list[str], b: list[str]) -> bool:
    """True if ``a``/``b`` are a strict one-item superset/subset of each other."""
    set_a, set_b = set(a), set(b)
    if set_a == set_b:
        return False
    if set_a < set_b:
        return len(set_b - set_a) == 1
    if set_b < set_a:
        return len(set_a - set_b) == 1
    return False


def _infer_reflection_mode(parsed: LlmLateReflectionResponse) -> str:
    """Recovery logic for a missing/unrecognised ``reflection_mode``.

    1. ``primary_bundle`` and ``comparison_bundle`` both present ->
       ``bundle_comparison``.
    2. Else, two bundles found in ``followup_bundles``/``target_bundles`` and
       one is a one-item extension/subset of the other -> ``marginal_item_test``.
    3. Else, two such bundles present -> ``bundle_comparison``.
    4. Else -> ``bundle_comparison`` (the permissive default).
    """
    if parsed.primary_bundle and parsed.comparison_bundle:
        return "bundle_comparison"

    candidates = parsed.followup_bundles or parsed.target_bundles
    if len(candidates) >= 2:
        first, second = candidates[0], candidates[1]
        if _bundles_are_one_item_apart(first, second):
            return "marginal_item_test"
        return "bundle_comparison"

    return "bundle_comparison"


def _apply_late_reflection_recovery(
    parsed: LlmLateReflectionResponse,
) -> LlmLateReflectionResponse:
    """Old-style-response backward compatibility + ``reflection_mode`` recovery.

    - ``primary_bundle``, if missing, is derived from ``followup_bundles[0]``
      (else ``target_bundles[0]``).
    - ``comparison_bundle``, if missing, is derived from
      ``followup_bundles[1]`` (else ``target_bundles[1]``), when a second
      bundle is available; otherwise it is left ``None`` (and
      ``comparison_pair_available`` downstream is ``False``).
    - ``reflection_mode``, if missing/unrecognised, is inferred via
      :func:`_infer_reflection_mode` and ``reflection_mode_inferred`` is set.
    - ``followup_bundles``, if still empty, is derived from
      ``primary_bundle``/``comparison_bundle`` (existing behaviour).

    This is what lets an old-style response (``question``/``target_type``/
    ``target_bundles``/``suggested_followup``/``followup_bundles`` only, no
    ``primary_bundle``/``comparison_bundle``/``reflection_mode`` at all)
    parse cleanly and still populate the pairwise columns.
    """
    # Infer reflection_mode from the model's *original* primary_bundle/
    # comparison_bundle (before the followup/target-bundle-derived fallback
    # below fills them in) -- otherwise every old-style response would
    # spuriously satisfy rule 1 ("primary_bundle and comparison_bundle
    # present") via its own derived values, and the one-item-apart
    # marginal_item_test detection in rule 2 would never fire.
    reflection_mode_inferred = parsed.reflection_mode not in _VALID_REFLECTION_MODES
    if reflection_mode_inferred:
        parsed.reflection_mode = _infer_reflection_mode(parsed)
    parsed.reflection_mode_inferred = reflection_mode_inferred

    primary_bundle = parsed.primary_bundle
    if primary_bundle is None:
        if parsed.followup_bundles:
            primary_bundle = parsed.followup_bundles[0]
        elif parsed.target_bundles:
            primary_bundle = parsed.target_bundles[0]

    comparison_bundle = parsed.comparison_bundle
    if comparison_bundle is None:
        if len(parsed.followup_bundles) > 1:
            comparison_bundle = parsed.followup_bundles[1]
        elif len(parsed.target_bundles) > 1:
            comparison_bundle = parsed.target_bundles[1]

    parsed.primary_bundle = primary_bundle
    parsed.comparison_bundle = comparison_bundle

    if not parsed.followup_bundles:
        parsed.followup_bundles = derive_late_reflection_followup_bundles(parsed)

    return parsed


def _classify_late_reflection_validation_error(exc: ValidationError) -> str:
    field_names = {
        str(error["loc"][0]) for error in exc.errors() if error.get("loc")
    }
    if "question" in field_names:
        return "missing_required_field"
    bundle_fields = {
        "primary_bundle", "comparison_bundle", "target_bundles",
        "followup_bundles", "marginal_item",
    }
    if field_names & bundle_fields:
        return "invalid_bundle"
    return "unknown"


def parse_late_reflection_response(raw: str) -> LlmLateReflectionResponse:
    """Parse a late-stage reflective elicitation response.

    "Strict prompt, forgiving parser": the prompt still asks the model for
    the full pairwise/marginal ``reflection_mode`` schema, but this parser
    tolerates markdown fences, leading/trailing prose, trailing commas,
    bundles given as ``"{A,B}"`` strings, a single bundle given as a flat
    item list, old-style (pre-pairwise) responses, and a missing/unrecognised
    ``reflection_mode`` -- see :func:`_normalize_late_reflection_payload` and
    :func:`_apply_late_reflection_recovery`. Only raises
    :exc:`LateReflectionParseError` (a :exc:`ValueError` subclass carrying
    ``error_type``/``raw_excerpt``) when no question text can be recovered
    at all, or when JSON is found but genuinely malformed -- callers should
    catch this, log ``error_type``/``raw_excerpt``, and skip the bidder
    rather than aborting the whole auction.
    """
    try:
        payload = extract_json_object(raw)
    except ValueError:
        payload = _regex_fallback_late_reflection_payload(raw)
        if payload is None:
            error_type = "truncated_output" if _looks_truncated(raw) else "invalid_json"
            raise LateReflectionParseError(
                "Could not parse late reflection response",
                error_type=error_type,
                raw_excerpt=raw_response_excerpt(raw),
            ) from None

    payload = _normalize_late_reflection_payload(payload)

    try:
        parsed = LlmLateReflectionResponse.model_validate(payload)
    except ValidationError as exc:
        raise LateReflectionParseError(
            f"Invalid late reflection response: {exc}",
            error_type=_classify_late_reflection_validation_error(exc),
            raw_excerpt=raw_response_excerpt(raw),
        ) from exc

    return _apply_late_reflection_recovery(parsed)


def derive_late_reflection_followup_bundles(
    parsed: LlmLateReflectionResponse,
) -> list[list[str]]:
    """Fill in ``followup_bundles`` from ``primary_bundle``/``comparison_bundle``.

    ``reflection_mode`` forces every question to be an explicit pairwise or
    marginal comparison, so when the model leaves ``followup_bundles`` empty
    (but did name a ``primary_bundle``/``comparison_bundle``) the comparison
    pair itself is the natural follow-up: it is exactly what the question
    asked about. Returns ``parsed.followup_bundles`` unchanged when it is
    already non-empty.
    """
    if parsed.followup_bundles:
        return parsed.followup_bundles

    derived: list[list[str]] = []
    seen: set[frozenset[str]] = set()
    for bundle in (parsed.primary_bundle, parsed.comparison_bundle):
        if not bundle:
            continue
        key = frozenset(bundle)
        if key in seen:
            # Unknown-item filtering can collapse primary_bundle and
            # comparison_bundle onto the same items; don't double-query.
            continue
        seen.add(key)
        derived.append(bundle)
    return derived


def filter_late_reflection_bundles(
    parsed: LlmLateReflectionResponse,
    known_item_ids: set[str],
) -> LlmLateReflectionResponse:
    """Drop unknown item IDs and derive missing ``followup_bundles``.

    Mirrors :func:`parse_interest_map_response`'s unknown-item filtering:
    a noisy LLM response should not introduce phantom items. Bundles that
    become empty after filtering are dropped entirely. If, after filtering,
    ``followup_bundles`` is still empty, it is derived from
    ``primary_bundle``/``comparison_bundle`` via
    :func:`derive_late_reflection_followup_bundles` -- this runs *after*
    unknown-item filtering so a derived follow-up bundle is never phantom.
    """
    def _filter_bundle(bundle: list[str] | None) -> list[str] | None:
        if bundle is None:
            return None
        filtered = [item for item in bundle if item in known_item_ids]
        return filtered or None

    def _filter_bundles(bundles: list[list[str]]) -> list[list[str]]:
        result: list[list[str]] = []
        seen: set[frozenset[str]] = set()
        for bundle in bundles:
            filtered = [item for item in bundle if item in known_item_ids]
            if not filtered:
                continue
            key = frozenset(filtered)
            if key in seen:
                # Unknown-item filtering can collapse two originally-distinct
                # bundles onto the same items -- don't double-query.
                continue
            seen.add(key)
            result.append(filtered)
        return result

    primary_bundle = _filter_bundle(parsed.primary_bundle)
    comparison_bundle = _filter_bundle(parsed.comparison_bundle)
    marginal_item = (
        parsed.marginal_item if parsed.marginal_item in known_item_ids else None
    )

    filtered = LlmLateReflectionResponse(
        question=parsed.question,
        reason=parsed.reason,
        reflection_mode=parsed.reflection_mode,
        reflection_mode_inferred=parsed.reflection_mode_inferred,
        target_type=parsed.target_type,
        primary_bundle=primary_bundle,
        comparison_bundle=comparison_bundle,
        marginal_item=marginal_item,
        target_bundles=_filter_bundles(parsed.target_bundles),
        suggested_followup=parsed.suggested_followup,
        followup_bundles=_filter_bundles(parsed.followup_bundles),
    )

    if not filtered.followup_bundles:
        filtered.followup_bundles = derive_late_reflection_followup_bundles(filtered)

    return filtered


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
