"""Prompt, parse, retry, and log LLM-backed bidder value queries."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Sequence

from auctionlab.auction_types import Bundle, Item
from auctionlab.llm.cache import bundle_key_str, call_client
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.logging import (
    LlmCallLogger,
    LlmCallRecord,
    current_timestamp,
)
from auctionlab.llm.parsing import (
    parse_demand_query_response,
    parse_natural_language_response,
    parse_person_answer_verification,
    parse_substitute_mode_entailment_response,
    parse_value_response,
    validate_queried_bundle,
)
from auctionlab.llm.prompts import (
    build_demand_query_prompt,
    build_person_answer_prompt,
    build_person_answer_verification_prompt,
    build_substitute_mode_entailment_prompt,
    build_value_query_prompt,
    person_answer_word_limits,
)
from auctionlab.llm.schemas import (
    LlmDemandQueryResponse,
    LlmPersonAnswerSemanticExtraction,
    LlmPersonAnswerVerification,
)


def _parsed_response_to_dict(parsed) -> dict:
    if hasattr(parsed, "model_dump"):
        return parsed.model_dump()
    return parsed.dict()


def _truth_group_map(
    groups: Sequence[Mapping[str, Any]],
    available_items: set[Item],
) -> dict[tuple[str, ...], str]:
    result: dict[tuple[str, ...], str] = {}
    for group in groups:
        items = tuple(
            sorted(set(group.get("items", ())) & available_items)
        )
        if len(items) >= 2:
            result[items] = str(group.get("acquisition_mode", "unclear"))
    return result


def _truth_complement_sets(
    groups: Sequence[Sequence[Item] | Mapping[str, Any]],
    available_items: set[Item],
) -> set[tuple[str, ...]]:
    result: set[tuple[str, ...]] = set()
    for group in groups:
        raw_items = group.get("items", ()) if isinstance(group, Mapping) else group
        items = tuple(sorted(set(raw_items) & available_items))
        if len(items) >= 2:
            result.add(items)
    return result


def compare_person_answer_extraction(
    extraction: LlmPersonAnswerSemanticExtraction,
    *,
    expected_interested_items: set[Item],
    expected_excluded_items: set[Item],
    expected_substitute_groups: Sequence[Mapping[str, Any]],
    expected_complement_groups: Sequence[
        Sequence[Item] | Mapping[str, Any]
    ],
    available_items: set[Item],
    expected_budget_hint: float | None,
    item_descriptions: Mapping[Item, str] | None = None,
) -> LlmPersonAnswerVerification:
    """Compare a blind semantic extraction with hidden qualitative truth."""
    positive = {
        row.item_id for row in extraction.positive_items
        if row.item_id in available_items
    }
    explicitly_excluded = {
        row.item_id for row in extraction.excluded_items
        if row.item_id in available_items
    }
    expected_positive = expected_interested_items & available_items
    expected_zero = expected_excluded_items & available_items

    missing = sorted(expected_positive - positive)
    incorrectly_excluded = sorted(expected_positive & explicitly_excluded)
    invented = sorted(positive & expected_zero)

    expected_groups = _truth_group_map(
        expected_substitute_groups, available_items
    )
    extracted_groups = {
        tuple(sorted(set(group.items) & available_items)):
        (
            group.acquisition_mode
            if group.mode_explicitly_stated
            else "unclear"
        )
        for group in extraction.substitute_groups
        if len(set(group.items) & available_items) >= 2
    }
    substitute_issues: list[str] = []
    invented_groups: list[tuple[tuple[str, ...], str]] = []
    for items, mode in sorted(expected_groups.items()):
        actual_mode = extracted_groups.get(items)
        if actual_mode is None:
            substitute_issues.append(
                f"missing {mode} substitute group {list(items)}"
            )
        elif actual_mode != mode:
            substitute_issues.append(
                f"group {list(items)} has mode {actual_mode}, expected {mode}"
            )
    for items, mode in sorted(extracted_groups.items()):
        if items not in expected_groups:
            invented_groups.append((items, mode))
            substitute_issues.append(
                f"invented {mode} substitute group {list(items)}"
            )

    expected_complements = _truth_complement_sets(
        expected_complement_groups, available_items
    )
    extracted_complements = {
        tuple(sorted(set(group.items) & available_items))
        for group in extraction.complementary_groups
        if group.explicit_extra_joint_value
        and len(set(group.items) & available_items) >= 2
    }
    complement_issues = [
        f"missing complementary group {list(items)}"
        for items in sorted(expected_complements - extracted_complements)
    ]
    complement_issues.extend(
        f"invented complementary group {list(items)}"
        for items in sorted(extracted_complements - expected_complements)
    )

    budget_preserved = True
    if expected_budget_hint is not None:
        tolerance = max(1.0, abs(expected_budget_hint) * 0.01)
        budget_preserved = (
            extraction.budget_hint is not None
            and abs(extraction.budget_hint - expected_budget_hint) <= tolerance
        )
    invented_numeric = bool(extraction.other_numeric_valuation_details)

    issues: list[str] = []
    if missing:
        issues.append(f"missing positive items: {missing}")
    if incorrectly_excluded:
        issues.append(
            f"positive items described as excluded: {incorrectly_excluded}"
        )
    if invented:
        issues.append(f"zero-interest items made positive: {invented}")
    issues.extend(substitute_issues)
    issues.extend(complement_issues)
    if not budget_preserved:
        issues.append(
            "overall budget is missing or differs from the environment"
        )
    if invented_numeric:
        issues.append(
            "answer contains additional numeric valuation detail: "
            + "; ".join(extraction.other_numeric_valuation_details)
        )

    repair_parts: list[str] = []
    def _label(item: Item) -> str:
        description = (item_descriptions or {}).get(item)
        if not description:
            return str(item)
        short_description = description.split(" for ", 1)[0].rstrip(".")
        words = short_description.split()
        if len(words) > 8:
            short_description = " ".join(words[:8])
        return f"{item} ({short_description})"

    if missing:
        repair_parts.append(
            "Explicitly describe these as positively valued priorities or "
            "fallbacks, never as unwanted: "
            + ", ".join(_label(item) for item in missing)
        )
    if incorrectly_excluded:
        repair_parts.append(
            "Do not exclude these positive items: "
            + ", ".join(_label(item) for item in incorrectly_excluded)
        )
    if invented:
        repair_parts.append(
            "Do not assign positive interest to these excluded items: "
            + ", ".join(_label(item) for item in invented)
        )
    if substitute_issues:
        for items, _mode in invented_groups:
            repair_parts.append(
                "Do not describe "
                + ", ".join(_label(item) for item in items)
                + " as alternatives or as a substitute group. Present them "
                "as separate independent interests; relative priority alone "
                "does not make items substitutes"
            )
        expected_group_instructions = [
            (
                "Describe "
                + ", ".join(_label(item) for item in items)
                + (
                    " as one choose-at-most-one alternative group; every "
                    "listed member remains positively valued"
                    if mode == "choose_one"
                    else (
                        " as one multiple-items-remain-useful alternative group"
                        if mode == "can_use_multiple"
                        else " as one alternative group with unclear multiplicity"
                    )
                )
            )
            for items, mode in sorted(expected_groups.items())
        ]
        repair_parts.extend(expected_group_instructions)
    if complement_issues:
        repair_parts.append(
            "Preserve only genuine complete-set extra value from the seed"
        )
    if not budget_preserved:
        repair_parts.append(
            f"State the overall budget as approximately ${expected_budget_hint:,.0f}"
        )
    if invented_numeric:
        repair_parts.append("Remove all numeric detail except the overall budget")

    return LlmPersonAnswerVerification(
        passed=not issues,
        missing_positive_items=missing,
        incorrectly_excluded_positive_items=incorrectly_excluded,
        invented_positive_items=invented,
        substitute_group_issues=substitute_issues,
        complement_group_issues=complement_issues,
        budget_preserved=budget_preserved,
        invented_numeric_detail=invented_numeric,
        issues=issues,
        repair_instructions=". ".join(repair_parts) + ("." if repair_parts else ""),
        semantic_extraction=_parsed_response_to_dict(extraction),
    )


def _build_value_repair_prompt(
    raw_response: str,
    expected_bundle: Bundle,
) -> str:
    """Ask the model to repair schema or bundle identity without revaluing."""
    expected_item_ids = sorted(expected_bundle)
    return f"""Repair the formatting of the previous response below.
EXPECTED_BUNDLE_ITEM_IDS = {expected_item_ids}

Your previous response must be repaired so that queried_bundle exactly equals
the expected bundle. Do not change the economic value unless the previous
value was clearly for the wrong bundle.

Return JSON only using exactly this schema:
{{
  "queried_bundle": {expected_item_ids},
  "bundle_value": <non-negative number>,
  "confidence": <number between 0 and 1 or null>,
  "reasoning_summary": "<short summary explaining why this value is for exactly the queried bundle>"
}}

Do not include markdown fences or text outside the JSON object.

Previous raw response:
{raw_response}"""


def _build_natural_language_repair_prompt(
    original_prompt: str,
    raw_response: str,
    *,
    target_words: int,
    hard_max_words: int,
    validation_error: str | None = None,
) -> str:
    """Request a compact, schema-only repair after an invalid NL answer."""
    error_section = (
        "\nMANDATORY CORRECTIONS:\n"
        "Apply every correction below even when the previous answer omitted "
        "or contradicted it.\n"
        f"{validation_error}\n"
        if validation_error
        else ""
    )
    return f"""The previous simulated-person response was empty, truncated, or invalid.
Answer the ORIGINAL REQUEST again. Preserve important preferences, including
fallbacks, alternatives, exclusions, complements, and budget constraints.
Aim for about {target_words} words and never exceed {hard_max_words} words.
State only the
single overall willingness-to-pay figure present in the qualitative seed; do
not invent individual item values or other valuation arithmetic. Preserve
every positively relevant fallback and each group's acquisition meaning, but
use compact, conversational phrasing rather than a repeated template. You do
not need to enumerate every irrelevant item. Do not call ordinary parts of a
setup complementary.
For every choose-at-most-one group, explicitly state that only one member is
needed or that owning extras from that exact group adds no meaningful benefit.
Calling an item a fallback, backup, or alternative is not sufficient.

ORIGINAL REQUEST:
{original_prompt}

PREVIOUS INVALID RESPONSE:
{raw_response}
{error_section}

Edit the previous answer so that every mandatory correction is explicit.
Return compact JSON only using exactly this schema:
{{"answer":"<complete corrected answer in the person's own words>"}}

Do not add fields, markdown fences, or text outside the JSON object."""


@dataclass
class LlmPersonSimulator:
    """Issue validated bundle-value queries on behalf of one simulated bidder."""

    bidder_id: str
    scenario_description: str
    person_seed: str
    item_descriptions: dict[Item, str]
    client: LlmClient
    nl_client: LlmClient | None = None
    verifier_client: LlmClient | None = None
    logger: LlmCallLogger | None = None
    model_name: str | None = None
    provider_name: str | None = None
    verifier_model_name: str | None = None
    verifier_provider_name: str | None = None
    max_parse_retries: int = 0
    ground_truth_valuations: dict[Bundle, float] | None = None
    verbose: bool = False
    scenario_id: str | None = None
    """Optional scenario identifier, threaded into cache rows only (see
    :mod:`auctionlab.llm.cache`); has no effect unless ``client`` is a
    :class:`~auctionlab.llm.cache.CachingLlmClient`."""
    expected_interested_items: set[Item] | None = None
    expected_excluded_items: set[Item] | None = None
    expected_substitute_groups: Sequence[Mapping[str, Any]] | None = None
    expected_complement_groups: Sequence[
        Sequence[Item] | Mapping[str, Any]
    ] | None = None
    expected_budget_hint: float | None = None
    _last_prompt: str | None = field(default=None, init=False, repr=False)
    _last_response_summary: str | None = field(default=None, init=False, repr=False)
    last_answer_verification: dict[str, Any] | None = field(
        default=None, init=False
    )
    answer_verification_history: list[dict[str, Any]] = field(
        default_factory=list, init=False
    )
    first_answer_word_count: int | None = field(default=None, init=False)
    final_answer_word_count: int | None = field(default=None, init=False)
    answer_attempt_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_parse_retries < 0:
            raise ValueError("max_parse_retries must be non-negative")

    def value_query(
        self,
        bundle: Bundle,
        anchor_values: dict[Bundle, float] | None = None,
        transcript_context: str | None = None,
        elicitation_context: str | None = None,
    ) -> float:
        """Return one parsed value, retrying only malformed responses."""
        if self.ground_truth_valuations is not None:
            bundle = frozenset(bundle)
            value = self.ground_truth_valuations.get(bundle, 0.0)
            _bundle_str = "{" + ",".join(sorted(str(i) for i in bundle)) + "}"
            if self.verbose:
                print(f"  {self.bidder_id:<12}  vq  {_bundle_str}  →  {value:.0f}  [gt]", flush=True)
            self._last_prompt = f"[ground-truth] bundle={sorted(bundle)}"
            self._last_response_summary = f"ground_truth={value}"
            if self.logger is not None:
                from auctionlab.llm.logging import LlmCallRecord, current_timestamp
                self.logger.log(LlmCallRecord(
                    timestamp=current_timestamp(),
                    bidder_id=self.bidder_id,
                    prompt_type="value_query_gt",
                    prompt=self._last_prompt,
                    raw_response=self._last_response_summary,
                    parsed_response=None,
                    success=True,
                    error=None,
                    latency_seconds=0.0,
                    model=self.model_name,
                    provider=self.provider_name,
                    llm_role="person",
                    input_tokens=0,
                    output_tokens=0,
                ))
            return value

        prompt = build_value_query_prompt(
            scenario_description=self.scenario_description,
            person_seed=self.person_seed,
            item_descriptions=self.item_descriptions,
            bundle=bundle,
            anchor_values=anchor_values,
            transcript_context=transcript_context,
            elicitation_context=elicitation_context,
        )
        self._last_prompt = prompt

        _bundle_str = "{" + ",".join(sorted(str(i) for i in bundle)) + "}"
        for attempt in range(1, self.max_parse_retries + 2):
            _retry = f"  [retry {attempt}]" if attempt > 1 else ""
            if self.verbose:
                print(
                    f"  {self.bidder_id:<12}  vq  {_bundle_str}{_retry}",
                    end="",
                    flush=True,
                )
            started = time.perf_counter()
            raw_response: str | None = None

            try:
                raw_response = call_client(
                    self.client,
                    prompt,
                    call_type="value_query",
                    scenario_id=self.scenario_id,
                    bidder_id=self.bidder_id,
                    bundle_key=bundle_key_str(bundle),
                    extra_key_fields=(
                        {"parse_repair_attempt": attempt}
                        if attempt > 1
                        else None
                    ),
                )
            except Exception as exc:
                latency = time.perf_counter() - started
                if self.verbose:
                    print(f"  →  ERROR  ({latency:.1f}s)  {exc}", flush=True)
                else:
                    print(
                        f"  {self.bidder_id:<12}  vq  {_bundle_str}{_retry}"
                        f"  ERROR  ({latency:.1f}s)  {exc}",
                        flush=True,
                    )
                self._log_attempt(
                    prompt=prompt,
                    raw_response=None,
                    parsed_response=None,
                    success=False,
                    error=str(exc),
                    latency_seconds=latency,
                    attempt=attempt,
                )
                raise

            latency = time.perf_counter() - started
            parsed = None

            try:
                parsed = parse_value_response(raw_response)
                validate_queried_bundle(parsed, bundle, raw_response=raw_response)
            except ValueError as exc:
                if self.verbose:
                    print(f"  →  parse error  ({latency:.1f}s)  {exc}", flush=True)
                else:
                    print(
                        f"  {self.bidder_id:<12}  vq  {_bundle_str}{_retry}"
                        f"  parse error  ({latency:.1f}s)  {exc}",
                        flush=True,
                    )
                self._log_attempt(
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_response=(
                        _parsed_response_to_dict(parsed)
                        if parsed is not None
                        else None
                    ),
                    success=False,
                    error=str(exc),
                    latency_seconds=latency,
                    attempt=attempt,
                )
                if attempt > self.max_parse_retries:
                    raise
                prompt = _build_value_repair_prompt(raw_response, bundle)
                continue

            if parsed.queried_bundle is None:
                # The model is no longer asked to echo the bundle (see
                # build_value_query_prompt); bind the mechanism-known bundle
                # onto the parsed result so logs/downstream code always show
                # which bundle this value is for, without ever depending on
                # the model to report it correctly.
                parsed.queried_bundle = sorted(bundle)

            self._log_attempt(
                prompt=prompt,
                raw_response=raw_response,
                parsed_response=_parsed_response_to_dict(parsed),
                success=True,
                error=None,
                latency_seconds=latency,
                attempt=attempt,
            )
            self._last_response_summary = parsed.reasoning_summary
            if self.verbose:
                print(f"  →  {parsed.bundle_value:.0f}  ({latency:.1f}s)", flush=True)
            return parsed.bundle_value

        raise RuntimeError("Value query attempts exhausted unexpectedly")

    def _log_attempt(
        self,
        *,
        prompt: str,
        raw_response: str | None,
        parsed_response,
        success: bool,
        error: str | None,
        latency_seconds: float,
        attempt: int,
        prompt_type: str = "value_query",
        client: LlmClient | None = None,
    ) -> None:
        if self.logger is None:
            return

        active_client = client or self.client
        self.logger.log(
            LlmCallRecord(
                timestamp=current_timestamp(),
                bidder_id=self.bidder_id,
                prompt_type=prompt_type,
                prompt=prompt,
                raw_response=raw_response,
                parsed_response=parsed_response,
                success=success,
                error=error,
                latency_seconds=latency_seconds,
                model=self.model_name,
                provider=getattr(
                    active_client,
                    "_auctionlab_provider",
                    self.provider_name,
                ),
                llm_role=getattr(
                    active_client,
                    "_auctionlab_llm_role",
                    "person",
                ),
                attempt=attempt,
                input_tokens=getattr(active_client, "_last_input_tokens", None),
                output_tokens=getattr(active_client, "_last_output_tokens", None),
                total_tokens=getattr(active_client, "_last_total_tokens", None),
                cached_input_tokens=getattr(
                    active_client, "_last_cached_input_tokens", None
                ),
                cached_output_tokens=getattr(
                    active_client, "_last_cached_output_tokens", None
                ),
                cache_hit=getattr(active_client, "_last_cache_hit", None),
                finish_reason=getattr(
                    active_client, "_last_finish_reason", None
                ),
                response_char_count=(
                    len(raw_response) if raw_response is not None else None
                ),
            )
        )

    def value_queries(self, bundles: list[Bundle]) -> dict[Bundle, float]:
        return {
            bundle: self.value_query(bundle)
            for bundle in bundles
        }

    def demand_query(
        self,
        bundle: Bundle,
        prices: dict[Item, float],
    ) -> LlmDemandQueryResponse:
        """Ask whether the person is satisfied with ``bundle`` at ``prices``."""
        if self.ground_truth_valuations is not None:
            bundle = frozenset(bundle)
            best_bundle = bundle
            best_surplus = self.ground_truth_valuations.get(bundle, 0.0) - sum(
                prices.get(item, 0.0) for item in bundle
            )
            for b, v in self.ground_truth_valuations.items():
                surplus = v - sum(prices.get(item, 0.0) for item in b)
                if surplus > best_surplus:
                    best_surplus = surplus
                    best_bundle = b
            satisfied = best_bundle == bundle
            preferred = None if satisfied else list(best_bundle)
            _bundle_str = "{" + ",".join(sorted(str(i) for i in bundle)) + "}"
            result_str = "sat" if satisfied else "unsat"
            if self.verbose:
                print(f"  {self.bidder_id:<12}  dq  {_bundle_str}  →  {result_str}  [gt]", flush=True)
            self._last_prompt = f"[ground-truth] bundle={sorted(bundle)}"
            self._last_response_summary = (
                f"satisfied={satisfied}; preferred_bundle={sorted(preferred or [])}"
            )
            if self.logger is not None:
                from auctionlab.llm.logging import LlmCallRecord, current_timestamp
                self.logger.log(LlmCallRecord(
                    timestamp=current_timestamp(),
                    bidder_id=self.bidder_id,
                    prompt_type="demand_query_gt",
                    prompt=self._last_prompt,
                    raw_response=self._last_response_summary,
                    parsed_response=None,
                    success=True,
                    error=None,
                    latency_seconds=0.0,
                    model=self.model_name,
                    provider=self.provider_name,
                    llm_role="person",
                    input_tokens=0,
                    output_tokens=0,
                ))
            return LlmDemandQueryResponse(satisfied=satisfied, preferred_bundle=preferred)

        prompt = build_demand_query_prompt(
            scenario_description=self.scenario_description,
            person_seed=self.person_seed,
            item_descriptions=self.item_descriptions,
            bundle=bundle,
            prices=prices,
        )
        self._last_prompt = prompt

        _bundle_str = "{" + ",".join(sorted(str(i) for i in bundle)) + "}"
        for attempt in range(1, self.max_parse_retries + 2):
            _retry = f"  [retry {attempt}]" if attempt > 1 else ""
            if self.verbose:
                print(
                    f"  {self.bidder_id:<12}  dq  {_bundle_str}{_retry}",
                    end="",
                    flush=True,
                )
            started = time.perf_counter()

            try:
                raw_response = call_client(
                    self.client,
                    prompt,
                    call_type="demand_query",
                    scenario_id=self.scenario_id,
                    bidder_id=self.bidder_id,
                    bundle_key=bundle_key_str(bundle),
                    extra_key_fields=(
                        {"parse_repair_attempt": attempt}
                        if attempt > 1
                        else None
                    ),
                )
            except Exception as exc:
                latency = time.perf_counter() - started
                if self.verbose:
                    print(f"  →  ERROR  ({latency:.1f}s)  {exc}", flush=True)
                else:
                    print(
                        f"  {self.bidder_id:<12}  dq  {_bundle_str}{_retry}"
                        f"  ERROR  ({latency:.1f}s)  {exc}",
                        flush=True,
                    )
                self._log_attempt(
                    prompt=prompt,
                    raw_response=None,
                    parsed_response=None,
                    success=False,
                    error=str(exc),
                    latency_seconds=latency,
                    attempt=attempt,
                    prompt_type="demand_query",
                )
                raise

            latency = time.perf_counter() - started

            try:
                parsed = parse_demand_query_response(raw_response)
            except ValueError as exc:
                if self.verbose:
                    print(f"  →  parse error  ({latency:.1f}s)  {exc}", flush=True)
                else:
                    print(
                        f"  {self.bidder_id:<12}  dq  {_bundle_str}{_retry}"
                        f"  parse error  ({latency:.1f}s)  {exc}",
                        flush=True,
                    )
                self._log_attempt(
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_response=None,
                    success=False,
                    error=str(exc),
                    latency_seconds=latency,
                    attempt=attempt,
                    prompt_type="demand_query",
                )
                if attempt > self.max_parse_retries:
                    raise
                continue

            self._log_attempt(
                prompt=prompt,
                raw_response=raw_response,
                parsed_response=_parsed_response_to_dict(parsed),
                success=True,
                error=None,
                latency_seconds=latency,
                attempt=attempt,
                prompt_type="demand_query",
            )
            preferred_items = sorted(parsed.preferred_bundle or [])
            self._last_response_summary = (
                f"satisfied={parsed.satisfied}; "
                f"preferred_bundle={preferred_items}"
            )
            result_str = "sat" if parsed.satisfied else "unsat"
            if self.verbose:
                print(f"  →  {result_str}  ({latency:.1f}s)", flush=True)
            return parsed

        raise RuntimeError("Demand query attempts exhausted unexpectedly")

    def _check_substitute_mode_entailment(
        self,
        *,
        answer: str,
        extraction: LlmPersonAnswerSemanticExtraction,
        person_attempt: int,
    ) -> LlmPersonAnswerSemanticExtraction:
        """Independently verify restrictive modes claimed by extraction."""
        proposed = [
            group
            for group in extraction.substitute_groups
            if group.acquisition_mode != "unclear"
            and group.mode_explicitly_stated
        ]
        if not proposed:
            return extraction

        prompt = build_substitute_mode_entailment_prompt(
            answer=answer,
            item_descriptions=self.item_descriptions,
            substitute_groups=[
                {
                    "items": group.items,
                    "acquisition_mode": group.acquisition_mode,
                }
                for group in proposed
            ],
        )
        started = time.perf_counter()
        raw_response = call_client(
            self.verifier_client,
            prompt,
            call_type="person_answer_substitute_entailment",
            scenario_id=self.scenario_id,
            bidder_id=self.bidder_id,
            extra_key_fields={"person_answer_attempt": person_attempt},
        )
        latency = time.perf_counter() - started
        try:
            parsed = parse_substitute_mode_entailment_response(raw_response)
        except ValueError as exc:
            if self.logger is not None:
                self.logger.log(LlmCallRecord(
                    timestamp=current_timestamp(),
                    bidder_id=self.bidder_id,
                    prompt_type="person_answer_substitute_entailment",
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_response=None,
                    success=False,
                    error=str(exc),
                    latency_seconds=latency,
                    model=self.verifier_model_name,
                    provider=self.verifier_provider_name,
                    llm_role="verifier",
                    attempt=person_attempt,
                    input_tokens=getattr(
                        self.verifier_client, "_last_input_tokens", None
                    ),
                    output_tokens=getattr(
                        self.verifier_client, "_last_output_tokens", None
                    ),
                    total_tokens=getattr(
                        self.verifier_client, "_last_total_tokens", None
                    ),
                    cache_hit=getattr(
                        self.verifier_client, "_last_cache_hit", None
                    ),
                ))
            raise

        judgments: dict[tuple[tuple[str, ...], str], list[bool]] = {}
        for judgment in parsed.judgments:
            key = (
                tuple(sorted(set(judgment.items))),
                judgment.acquisition_mode,
            )
            judgments.setdefault(key, []).append(judgment.entailed)

        checked_groups = []
        for group in extraction.substitute_groups:
            if (
                group.acquisition_mode == "unclear"
                or not group.mode_explicitly_stated
            ):
                checked_groups.append(group)
                continue
            key = (
                tuple(sorted(set(group.items))),
                group.acquisition_mode,
            )
            results = judgments.get(key, [])
            entailed = len(results) == 1 and results[0] is True
            checked_groups.append(
                group.model_copy(update={
                    "mode_explicitly_stated": entailed,
                    "acquisition_mode": (
                        group.acquisition_mode if entailed else "unclear"
                    ),
                })
            )

        checked = extraction.model_copy(
            update={"substitute_groups": checked_groups}
        )
        if self.logger is not None:
            self.logger.log(LlmCallRecord(
                timestamp=current_timestamp(),
                bidder_id=self.bidder_id,
                prompt_type="person_answer_substitute_entailment",
                prompt=prompt,
                raw_response=raw_response,
                parsed_response={
                    "judgments": _parsed_response_to_dict(parsed),
                    "checked_semantic_extraction": (
                        _parsed_response_to_dict(checked)
                    ),
                },
                success=True,
                error=None,
                latency_seconds=latency,
                model=self.verifier_model_name,
                provider=self.verifier_provider_name,
                llm_role="verifier",
                attempt=person_attempt,
                input_tokens=getattr(
                    self.verifier_client, "_last_input_tokens", None
                ),
                output_tokens=getattr(
                    self.verifier_client, "_last_output_tokens", None
                ),
                total_tokens=getattr(
                    self.verifier_client, "_last_total_tokens", None
                ),
                cache_hit=getattr(
                    self.verifier_client, "_last_cache_hit", None
                ),
            ))
        return checked

    def _verify_person_answer(
        self,
        *,
        question: str,
        answer: str,
        person_attempt: int,
    ) -> LlmPersonAnswerVerification | None:
        if (
            self.verifier_client is None
            or self.expected_interested_items is None
        ):
            return None

        prompt = build_person_answer_verification_prompt(
            scenario_description=self.scenario_description,
            question=question,
            answer=answer,
            item_descriptions=self.item_descriptions,
        )
        started = time.perf_counter()
        raw_response = call_client(
            self.verifier_client,
            prompt,
            call_type="person_answer_semantic_extraction",
            scenario_id=self.scenario_id,
            bidder_id=self.bidder_id,
            extra_key_fields={"person_answer_attempt": person_attempt},
        )
        latency = time.perf_counter() - started
        extraction_call_metadata = {
            "input_tokens": getattr(
                self.verifier_client, "_last_input_tokens", None
            ),
            "output_tokens": getattr(
                self.verifier_client, "_last_output_tokens", None
            ),
            "total_tokens": getattr(
                self.verifier_client, "_last_total_tokens", None
            ),
            "cache_hit": getattr(
                self.verifier_client, "_last_cache_hit", None
            ),
        }
        parsed: LlmPersonAnswerSemanticExtraction | None = None
        try:
            parsed = parse_person_answer_verification(raw_response)
        except ValueError as exc:
            if self.logger is not None:
                self.logger.log(LlmCallRecord(
                    timestamp=current_timestamp(),
                    bidder_id=self.bidder_id,
                    prompt_type="person_answer_semantic_extraction",
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_response=None,
                    success=False,
                    error=str(exc),
                    latency_seconds=latency,
                    model=self.verifier_model_name,
                    provider=self.verifier_provider_name,
                    llm_role="verifier",
                    attempt=person_attempt,
                    **extraction_call_metadata,
                ))
            raise

        verification = compare_person_answer_extraction(
            parsed,
            expected_interested_items=self.expected_interested_items,
            expected_excluded_items=self.expected_excluded_items or set(),
            expected_substitute_groups=self.expected_substitute_groups or (),
            expected_complement_groups=self.expected_complement_groups or (),
            available_items=set(self.item_descriptions),
            expected_budget_hint=self.expected_budget_hint,
            item_descriptions=self.item_descriptions,
        )
        if verification.passed:
            parsed = self._check_substitute_mode_entailment(
                answer=answer,
                extraction=parsed,
                person_attempt=person_attempt,
            )
            verification = compare_person_answer_extraction(
                parsed,
                expected_interested_items=self.expected_interested_items,
                expected_excluded_items=self.expected_excluded_items or set(),
                expected_substitute_groups=(
                    self.expected_substitute_groups or ()
                ),
                expected_complement_groups=(
                    self.expected_complement_groups or ()
                ),
                available_items=set(self.item_descriptions),
                expected_budget_hint=self.expected_budget_hint,
                item_descriptions=self.item_descriptions,
            )
        if self.logger is not None:
            self.logger.log(LlmCallRecord(
                timestamp=current_timestamp(),
                bidder_id=self.bidder_id,
                prompt_type="person_answer_semantic_extraction",
                prompt=prompt,
                raw_response=raw_response,
                parsed_response={
                    "semantic_extraction": _parsed_response_to_dict(parsed),
                    "comparison": _parsed_response_to_dict(verification),
                },
                success=True,
                error=None,
                latency_seconds=latency,
                model=self.verifier_model_name,
                provider=self.verifier_provider_name,
                llm_role="verifier",
                attempt=person_attempt,
                **extraction_call_metadata,
            ))
        return verification

    def answer_question(self, question: str) -> str:
        """Answer a free-form natural-language preference question."""
        target_words, hard_max_words = person_answer_word_limits(
            len(self.item_descriptions)
        )
        if self.expected_interested_items is not None:
            structural_group_count = len(
                self.expected_substitute_groups or ()
            ) + len(self.expected_complement_groups or ())
            complexity_target = round(
                40
                + 4 * len(self.expected_interested_items)
                + 5 * structural_group_count
            )
            target_words = max(
                target_words,
                min(hard_max_words - 5, complexity_target),
            )
        original_prompt = build_person_answer_prompt(
            scenario_description=self.scenario_description,
            person_seed=self.person_seed,
            question=question,
            item_descriptions=self.item_descriptions,
            target_words=target_words,
            hard_max_words=hard_max_words,
        )
        prompt = original_prompt
        answer_client = self.nl_client or self.client
        self.answer_attempt_count = 0
        self.first_answer_word_count = None
        self.final_answer_word_count = None
        self.last_answer_verification = None
        self.answer_verification_history = []

        for attempt in range(1, self.max_parse_retries + 2):
            self.answer_attempt_count = attempt
            _retry = f"  [retry {attempt}]" if attempt > 1 else ""
            if self.verbose:
                print(
                    f"  {self.bidder_id:<12}  nl{_retry}",
                    end="",
                    flush=True,
                )
            started = time.perf_counter()

            try:
                raw_response = call_client(
                    answer_client,
                    prompt,
                    call_type="person_nl_response",
                    scenario_id=self.scenario_id,
                    bidder_id=self.bidder_id,
                    extra_key_fields=(
                        {"parse_repair_attempt": attempt}
                        if attempt > 1
                        else None
                    ),
                )
            except Exception as exc:
                latency = time.perf_counter() - started
                if self.verbose:
                    print(f"  →  ERROR  ({latency:.1f}s)  {exc}", flush=True)
                else:
                    print(
                        f"  {self.bidder_id:<12}  nl{_retry}"
                        f"  ERROR  ({latency:.1f}s)  {exc}",
                        flush=True,
                    )
                self._log_attempt(
                    prompt=prompt,
                    raw_response=None,
                    parsed_response=None,
                    success=False,
                    error=str(exc),
                    latency_seconds=latency,
                    attempt=attempt,
                    prompt_type="nl_question",
                    client=answer_client,
                )
                raise

            latency = time.perf_counter() - started

            try:
                parsed = parse_natural_language_response(raw_response)
                word_count = len(parsed.answer.split())
                if self.first_answer_word_count is None:
                    self.first_answer_word_count = word_count
                if word_count > hard_max_words:
                    raise ValueError(
                        f"person answer has {word_count} words; hard maximum "
                        f"for {len(self.item_descriptions)} goods is "
                        f"{hard_max_words}"
                    )
                verification = self._verify_person_answer(
                    question=question,
                    answer=parsed.answer,
                    person_attempt=attempt,
                )
                if verification is not None:
                    verification_payload = _parsed_response_to_dict(verification)
                    self.last_answer_verification = verification_payload
                    self.answer_verification_history.append(verification_payload)
                    if not verification.passed:
                        details = (
                            verification.repair_instructions
                            or "; ".join(verification.issues)
                            or "semantic fidelity check failed"
                        )
                        raise ValueError(
                            "person-answer verifier rejected response: "
                            + details
                        )
            except ValueError as exc:
                # A response rejected by parsing or semantic verification is
                # not reusable. Evict it so a resumed preparation run can
                # obtain a genuinely new answer instead of replaying the same
                # invalid cached text.
                invalidate = getattr(answer_client, "invalidate_last", None)
                if callable(invalidate):
                    invalidate()
                if self.verbose:
                    print(f"  →  parse error  ({latency:.1f}s)  {exc}", flush=True)
                else:
                    print(
                        f"  {self.bidder_id:<12}  nl{_retry}"
                        f"  parse error  ({latency:.1f}s)  {exc}",
                        flush=True,
                    )
                self._log_attempt(
                    prompt=prompt,
                    raw_response=raw_response,
                    parsed_response=None,
                    success=False,
                    error=str(exc),
                    latency_seconds=latency,
                    attempt=attempt,
                    prompt_type="nl_question",
                    client=answer_client,
                )
                if attempt > self.max_parse_retries:
                    raise
                prompt = _build_natural_language_repair_prompt(
                    original_prompt,
                    raw_response,
                    target_words=target_words,
                    hard_max_words=hard_max_words,
                    validation_error=str(exc),
                )
                continue

            self._log_attempt(
                prompt=prompt,
                raw_response=raw_response,
                parsed_response=_parsed_response_to_dict(parsed),
                success=True,
                error=None,
                latency_seconds=latency,
                attempt=attempt,
                prompt_type="nl_question",
                client=answer_client,
            )
            if self.verbose:
                print(f"  →  done  ({latency:.1f}s)", flush=True)
            self.final_answer_word_count = len(parsed.answer.split())
            return parsed.answer

        raise RuntimeError("Answer-question attempts exhausted unexpectedly")
