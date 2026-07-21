"""Prompt, parse, retry, and log LLM-backed bidder value queries."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from auctionlab.auction_types import Bundle, Item
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.logging import (
    LlmCallLogger,
    LlmCallRecord,
    current_timestamp,
)
from auctionlab.llm.parsing import (
    parse_demand_query_response,
    parse_natural_language_response,
    parse_value_response,
    validate_queried_bundle,
)
from auctionlab.llm.prompts import (
    build_demand_query_prompt,
    build_person_answer_prompt,
    build_value_query_prompt,
)
from auctionlab.llm.schemas import LlmDemandQueryResponse


def _parsed_response_to_dict(parsed) -> dict:
    if hasattr(parsed, "model_dump"):
        return parsed.model_dump()
    return parsed.dict()


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


@dataclass
class LlmPersonSimulator:
    """Issue validated bundle-value queries on behalf of one simulated bidder."""

    bidder_id: str
    scenario_description: str
    person_seed: str
    item_descriptions: dict[Item, str]
    client: LlmClient
    logger: LlmCallLogger | None = None
    model_name: str | None = None
    max_parse_retries: int = 0
    ground_truth_valuations: dict[Bundle, float] | None = None
    verbose: bool = False
    _last_prompt: str | None = field(default=None, init=False, repr=False)
    _last_response_summary: str | None = field(default=None, init=False, repr=False)

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
                raw_response = self.client.complete(prompt)
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
                validate_queried_bundle(parsed, bundle)
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
    ) -> None:
        if self.logger is None:
            return

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
                attempt=attempt,
                input_tokens=getattr(self.client, "_last_input_tokens", None),
                output_tokens=getattr(self.client, "_last_output_tokens", None),
                total_tokens=getattr(self.client, "_last_total_tokens", None),
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
                raw_response = self.client.complete(prompt)
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

    def answer_question(self, question: str) -> str:
        """Answer a free-form natural-language preference question."""
        prompt = build_person_answer_prompt(
            scenario_description=self.scenario_description,
            person_seed=self.person_seed,
            question=question,
        )

        for attempt in range(1, self.max_parse_retries + 2):
            _retry = f"  [retry {attempt}]" if attempt > 1 else ""
            if self.verbose:
                print(
                    f"  {self.bidder_id:<12}  nl{_retry}",
                    end="",
                    flush=True,
                )
            started = time.perf_counter()

            try:
                raw_response = self.client.complete(prompt)
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
                )
                raise

            latency = time.perf_counter() - started

            try:
                parsed = parse_natural_language_response(raw_response)
            except ValueError as exc:
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
                    raw_response=None,
                    parsed_response=None,
                    success=False,
                    error=str(exc),
                    latency_seconds=latency,
                    attempt=attempt,
                    prompt_type="nl_question",
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
                prompt_type="nl_question",
            )
            if self.verbose:
                print(f"  →  done  ({latency:.1f}s)", flush=True)
            return parsed.answer

        raise RuntimeError("Answer-question attempts exhausted unexpectedly")
