"""ωvd1, ωvd2, ωnvd: LLM-action-choice CECA proxies (archived legacy).

This module is archived legacy code and is not part of the main
sealed/clock event-driven proxy experiments.


Three incremental LLM proxy designs for CECA, each conforming to the
:class:`~auctionlab.proxies.base.CecaAuctionProxy` and
:class:`~auctionlab.proxies.base.AuctionProxy` protocols:

- **Vd1CecaProxy** (ωvd1): At each CECA step the proxy LLM consults the
  conversation transcript and chooses one of three actions — *Satisfied*,
  *ValueQuery(bundle)*, or *DemandQuery* — without any pre-computed value
  estimates.

- **Vd2CecaProxy** (ωvd2): Extends ωvd1 with a γ inference engine. Before the
  action-choice prompt the proxy LLM estimates values for all bundles in a
  pluggable :class:`BundleScope` from the transcript, without issuing any
  person queries. These γ estimates are surfaced in the action-choice prompt
  as guidance.

- **NvdCecaProxy** (ωnvd): Extends ωvd2 by asking ``num_nl_questions``
  natural-language preference questions to the person *before* the first CECA
  step. The answers populate :attr:`nl_transcript` and immediately seed γ.

Terminology:
  γ            — proxy-side LLM inference: estimates bundle values from the
                  transcript WITHOUT issuing a person query (Q^p_V).
  Q^p_V        — value query to the person (``person.value_query``).
  Q^p_D        — demand query to the person (``person.ceca_demand_query``).
  Q^p_N        — NL question to the person (``person.answer_question``).

VALUE_QUERY semantics (Section 4): after querying bundle *b*, if
``surplus(b) > surplus(current_bundle)`` then report unsatisfied (demanded=b);
otherwise report satisfied. Both surpluses use the current Lindahl prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Protocol, runtime_checkable
import time

from pydantic import BaseModel, ValidationError

from auctionlab.auction_types import Bundle, Item
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.legacy.types import CecaStepResponse
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.logging import LlmCallLogger, LlmCallRecord, current_timestamp
from auctionlab.llm.parsing import extract_json_object, parse_proxy_question_response
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.prompts import (
    build_initial_proxy_question_prompt,
    format_transcript_context,
)
from auctionlab.llm.schemas import LlmInterestMap
from auctionlab.proxies.base import ElicitationEvent, ProxyStats, RefinementRecord


# --------------------------------------------------------------------------- #
# BundleScope: pluggable abstraction for which bundles γ covers               #
# --------------------------------------------------------------------------- #

@runtime_checkable
class BundleScope(Protocol):
    """Which bundles the γ inference engine should estimate values for.

    Implementations control cost/coverage trade-off. The modular design allows
    an :class:`InterestMapScope` to be swapped in once interest maps are
    integrated into the CECA flow.
    """

    def bundles(self, all_items: list[Item]) -> list[Bundle]:
        """Return the ordered list of bundles γ should cover."""
        ...


@dataclass(frozen=True)
class AllBundlesScope:
    """Cover all 2^n – 1 non-empty subsets. Practical for n ≤ 10."""

    def bundles(self, all_items: list[Item]) -> list[Bundle]:
        if len(all_items) > 10:
            import warnings
            warnings.warn(
                f"AllBundlesScope: {len(all_items)} items → {2**len(all_items) - 1} bundles. "
                "Consider SizeLimitedScope.",
                stacklevel=3,
            )
        result: list[Bundle] = []
        for size in range(1, len(all_items) + 1):
            for combo in combinations(sorted(all_items), size):
                result.append(frozenset(combo))
        return result


@dataclass(frozen=True)
class SizeLimitedScope:
    """Cover all subsets of size ≤ ``max_size``."""

    max_size: int = 2

    def bundles(self, all_items: list[Item]) -> list[Bundle]:
        result: list[Bundle] = []
        cap = min(self.max_size, len(all_items))
        for size in range(1, cap + 1):
            for combo in combinations(sorted(all_items), size):
                result.append(frozenset(combo))
        return result


@dataclass
class InterestMapScope:
    """Cover bundles derived from a structured interest map.

    Future integration point: once the CECA flow includes a pre-round
    interest-map elicitation step, pass the resulting
    :class:`~auctionlab.llm.schemas.LlmInterestMap` here to focus γ on the
    person's revealed areas of interest.
    """

    interest_map: LlmInterestMap

    def bundles(self, all_items: list[Item]) -> list[Bundle]:
        from auctionlab.llm.interest_map import generate_candidate_bundles_from_interest_map
        return generate_candidate_bundles_from_interest_map(self.interest_map, all_items)


# --------------------------------------------------------------------------- #
# Internal pydantic schemas                                                    #
# --------------------------------------------------------------------------- #

class _ActionChoiceResponse(BaseModel):
    action: str
    bundle: list[str] | None = None
    reasoning: str | None = None


class _GammaEstimateEntry(BaseModel):
    bundle: list[str]
    estimated_value: float


class _GammaResponse(BaseModel):
    estimates: list[_GammaEstimateEntry]
    reasoning: str | None = None


# --------------------------------------------------------------------------- #
# Prompt builders (proxy-side; never shown to the person)                     #
# --------------------------------------------------------------------------- #

def _action_choice_prompt(
    *,
    scenario_description: str,
    item_descriptions: dict[Item, str],
    current_bundle: Bundle,
    v_current: float,
    s_current: float,
    prices: Callable[[Bundle], float],
    manifest_atoms: list[XorAtomicBid],
    nl_transcript: list[tuple[str, str]],
    gamma_estimates: dict[Bundle, float],
) -> str:
    items_section = "\n".join(
        f"- {item}: {desc}" for item, desc in sorted(item_descriptions.items())
    )
    transcript_section = (
        format_transcript_context(nl_transcript).strip() if nl_transcript else "(none)"
    )
    bundle_ids = sorted(current_bundle)
    p_current = prices(current_bundle)
    atoms_lines = "\n".join(
        f"  [{','.join(sorted(a.bundle))}]: value={a.value:.2f}  price={prices(a.bundle):.2f}"
        for a in manifest_atoms
        if a.bundle
    ) or "  (none yet)"
    gamma_section = ""
    if gamma_estimates:
        lines = sorted(gamma_estimates.items(), key=lambda kv: -(kv[1] - prices(kv[0])))
        gamma_section = "\nGAMMA_ESTIMATES (proxy inference from transcript; no person queries):\n"
        gamma_section += "\n".join(
            f"  [{','.join(sorted(b))}]: ~{v:.2f}  surplus ~{v - prices(b):.2f}"
            for b, v in lines
        )

    return f"""You are a bidding proxy for a person in a combinatorial auction.

Scenario:
{scenario_description}

Available items:
{items_section}

PREFERENCE_TRANSCRIPT:
{transcript_section}

CURRENT_ALLOCATION:
  Bundle: {bundle_ids}
  Value (just queried from person): {v_current:.2f}
  Lindahl price: {p_current:.2f}
  Surplus: {s_current:.2f}

MANIFEST_ATOMS (values reported to the auction so far):
{atoms_lines}
{gamma_section}
Choose the best next action for this CECA step:

  "satisfied"    — the person is content with the current bundle at these prices
  "value_query"  — ask the person for the value of a specific bundle; choose a
                   bundle you believe may have HIGHER surplus than current
  "demand_query" — ask the person directly whether they prefer a different bundle

For "value_query", specify a non-empty list of item IDs in "bundle".
Prefer "demand_query" when you are uncertain which specific bundle to target.

Return JSON only in exactly this schema:
{{
  "action": "satisfied" | "value_query" | "demand_query",
  "bundle": ["ITEM_ID", ...],
  "reasoning": "<brief explanation>"
}}

Do not include markdown fences or any text outside the JSON object."""


def _gamma_inference_prompt(
    *,
    scenario_description: str,
    item_descriptions: dict[Item, str],
    nl_transcript: list[tuple[str, str]],
    manifest_atoms: list[XorAtomicBid],
    bundles: list[Bundle],
) -> str:
    items_section = "\n".join(
        f"- {item}: {desc}" for item, desc in sorted(item_descriptions.items())
    )
    transcript_section = (
        format_transcript_context(nl_transcript).strip() if nl_transcript else "(none so far)"
    )
    atoms_section = "\n".join(
        f"  [{','.join(sorted(a.bundle))}]: {a.value:.2f}"
        for a in manifest_atoms
        if a.bundle
    ) or "  (none yet)"
    bundles_section = "\n".join(
        f"  {i + 1}. {sorted(b)}" for i, b in enumerate(bundles)
    )

    return f"""You are a valuation analyst estimating bundle values from a preference transcript.

Scenario:
{scenario_description}

Available items:
{items_section}

PREFERENCE_TRANSCRIPT (what the person has revealed):
{transcript_section}

CONFIRMED_VALUES (values the person directly reported to the auction):
{atoms_section}

Estimate the person's value for EACH bundle in BUNDLES_TO_ESTIMATE.
Use ONLY the transcript and confirmed values above — do not invent preferences.
For bundles not directly covered, extrapolate conservatively from what is known.
Estimates must be non-negative.

BUNDLES_TO_ESTIMATE:
{bundles_section}

Return JSON only in exactly this schema:
{{
  "estimates": [
    {{"bundle": ["ITEM_ID", ...], "estimated_value": <non-negative float>}},
    ...
  ],
  "reasoning": "<2-3 sentences explaining your approach>"
}}

One entry per bundle, in the same order as BUNDLES_TO_ESTIMATE.
Do not include markdown fences or any text outside the JSON object."""


# --------------------------------------------------------------------------- #
# Parse helpers                                                                #
# --------------------------------------------------------------------------- #

def _parse_action_choice(
    raw: str, known_item_ids: set[str]
) -> tuple[str, Bundle | None]:
    try:
        payload = extract_json_object(raw)
    except ValueError as exc:
        raise ValueError(f"Could not parse action choice JSON: {exc}") from exc
    try:
        resp = _ActionChoiceResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid action choice schema: {exc}") from exc

    action = resp.action.lower().strip()
    if action not in ("satisfied", "value_query", "demand_query"):
        raise ValueError(f"Unknown action {action!r}")

    target: Bundle | None = None
    if action == "value_query":
        raw_ids = resp.bundle or []
        filtered = [item for item in raw_ids if item in known_item_ids]
        if not filtered:
            action = "demand_query"
        else:
            target = frozenset(filtered)

    return action, target


def _parse_gamma_response(
    raw: str, bundles: list[Bundle]
) -> dict[Bundle, float]:
    try:
        payload = extract_json_object(raw)
    except ValueError as exc:
        raise ValueError(f"Could not parse gamma inference JSON: {exc}") from exc
    try:
        resp = _GammaResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid gamma inference schema: {exc}") from exc

    lookup: dict[Bundle, float] = {}
    for entry in resp.estimates:
        lookup[frozenset(entry.bundle)] = max(0.0, entry.estimated_value)

    return {b: lookup.get(frozenset(b), 0.0) for b in bundles}


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _upsert_atom(bid: XorBid, bundle: Bundle, value: float) -> None:
    replacement = XorAtomicBid(bundle=bundle, value=value)
    for idx, atom in enumerate(bid.atoms):
        if atom.bundle == bundle:
            bid.atoms[idx] = replacement
            return
    bid.atoms.append(replacement)


# --------------------------------------------------------------------------- #
# ωvd1: action-choice proxy, no γ                                             #
# --------------------------------------------------------------------------- #

@dataclass
class Vd1CecaProxy:
    """ωvd1: proxy LLM chooses action (Satisfied / ValueQuery / DemandQuery).

    At each CECA step:
    1. The person is queried for the value of the current bundle (Q^p_V)
       to establish baseline surplus.
    2. The proxy LLM receives the transcript, manifest, and current surplus
       and returns an action choice.
    3. The chosen action is executed.

    VALUE_QUERY semantics: if ``surplus(queried_bundle) > surplus(current)``,
    report unsatisfied (demanding the queried bundle); otherwise satisfied.
    """

    bidder_id: str
    person: LlmPersonSimulator
    items: list[Item]
    proxy_client: LlmClient
    scenario_description: str
    item_descriptions: dict[Item, str]
    nl_transcript: list[tuple[str, str]] = field(default_factory=list)
    max_parse_retries: int = 2
    logger: LlmCallLogger | None = None

    _bid: XorBid = field(init=False, repr=False)
    _stats: ProxyStats = field(default_factory=ProxyStats, init=False, repr=False)
    _refinement_records: list[RefinementRecord] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._bid = XorBid(bidder_id=self.bidder_id, atoms=[])

    # ------------------------------------------------------------------ #
    # Template methods overridden by subclasses                            #
    # ------------------------------------------------------------------ #

    def _maybe_refresh_gamma(self, round_idx: int) -> None:
        pass

    def _maybe_ask_nl_questions(self) -> None:
        pass

    def _gamma_estimates(self) -> dict[Bundle, float]:
        return {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _value_query(self, bundle: Bundle) -> float:
        if not bundle:
            return 0.0
        self._stats.value_queries += 1
        return self.person.value_query(bundle)

    def _log_proxy_call(
        self,
        *,
        prompt: str,
        raw: str | None,
        parsed,
        success: bool,
        error: str | None,
        latency: float,
        attempt: int,
        prompt_type: str,
    ) -> None:
        if self.logger is None:
            return
        self.logger.log(
            LlmCallRecord(
                timestamp=current_timestamp(),
                bidder_id=self.bidder_id,
                prompt_type=f"proxy_{prompt_type}",
                prompt=prompt,
                raw_response=raw,
                parsed_response=parsed,
                success=success,
                error=error,
                latency_seconds=latency,
                model=getattr(self.proxy_client, "model", None),
                attempt=attempt,
                input_tokens=getattr(self.proxy_client, "_last_input_tokens", None),
                output_tokens=getattr(self.proxy_client, "_last_output_tokens", None),
                total_tokens=getattr(self.proxy_client, "_last_total_tokens", None),
            )
        )

    def _choose_action(
        self,
        current_bundle: Bundle,
        v_current: float,
        s_current: float,
        prices: Callable[[Bundle], float],
    ) -> tuple[str, Bundle | None]:
        prompt = _action_choice_prompt(
            scenario_description=self.scenario_description,
            item_descriptions=self.item_descriptions,
            current_bundle=current_bundle,
            v_current=v_current,
            s_current=s_current,
            prices=prices,
            manifest_atoms=self._bid.atoms,
            nl_transcript=self.nl_transcript,
            gamma_estimates=self._gamma_estimates(),
        )
        known_ids = set(self.item_descriptions)

        for attempt in range(1, self.max_parse_retries + 2):
            started = time.perf_counter()
            try:
                raw = self.proxy_client.complete(prompt)
            except Exception as exc:
                latency = time.perf_counter() - started
                self._log_proxy_call(
                    prompt=prompt, raw=None, parsed=None,
                    success=False, error=str(exc),
                    latency=latency, attempt=attempt, prompt_type="action_choice",
                )
                raise
            latency = time.perf_counter() - started

            try:
                action, target = _parse_action_choice(raw, known_ids)
            except ValueError as exc:
                self._log_proxy_call(
                    prompt=prompt, raw=raw, parsed=None,
                    success=False, error=str(exc),
                    latency=latency, attempt=attempt, prompt_type="action_choice",
                )
                if attempt > self.max_parse_retries:
                    raise
                continue

            self._log_proxy_call(
                prompt=prompt, raw=raw,
                parsed={"action": action, "bundle": sorted(target) if target else None},
                success=True, error=None,
                latency=latency, attempt=attempt, prompt_type="action_choice",
            )
            return action, target

        raise RuntimeError("Action choice attempts exhausted")

    # ------------------------------------------------------------------ #
    # CecaAuctionProxy interface                                           #
    # ------------------------------------------------------------------ #

    def ceca_step(
        self,
        prices: Callable[[Bundle], float],
        current_bundle: Bundle,
        round_idx: int = 0,
    ) -> CecaStepResponse:
        """CECA step: establish baseline surplus, then proxy LLM chooses action."""
        # NL questions first so their answers seed the transcript before γ refresh.
        self._maybe_ask_nl_questions()
        self._maybe_refresh_gamma(round_idx)

        current_bundle = frozenset(current_bundle)
        v_current = self._value_query(current_bundle)
        s_current = v_current - prices(current_bundle)

        action, target = self._choose_action(current_bundle, v_current, s_current, prices)

        if action == "satisfied":
            return CecaStepResponse(satisfied=True, demanded_bundle=None, value=None)

        if action == "value_query" and target:
            v_target = self._value_query(target)
            _upsert_atom(self._bid, target, v_target)
            if v_target - prices(target) > s_current:
                return CecaStepResponse(satisfied=False, demanded_bundle=target, value=v_target)
            return CecaStepResponse(satisfied=True, demanded_bundle=None, value=None)

        # demand_query
        manifest_prices = {atom.bundle: atom.value for atom in self._bid.atoms}
        response = self.person.ceca_demand_query(current_bundle, manifest_prices)
        self._stats.demand_queries += 1

        if response.satisfied:
            return CecaStepResponse(satisfied=True, demanded_bundle=None, value=None)

        preferred = (
            frozenset(response.preferred_bundle)
            if response.preferred_bundle
            else current_bundle
        )
        if not preferred:
            preferred = current_bundle

        v_preferred = self._value_query(preferred)
        _upsert_atom(self._bid, preferred, v_preferred)
        return CecaStepResponse(satisfied=False, demanded_bundle=preferred, value=v_preferred)

    # ------------------------------------------------------------------ #
    # AuctionProxy / SealedAuctionProxy interface                          #
    # ------------------------------------------------------------------ #

    def current_bid(self) -> XorBid:
        return self._bid

    def submit_bid(self) -> XorBid:
        return self._bid

    def refine(self, event: ElicitationEvent) -> None:
        pass

    def receive_provisional_feedback(self, event: ElicitationEvent) -> None:
        pass

    def stats(self) -> ProxyStats:
        return self._stats

    def refinement_records(self) -> list[RefinementRecord]:
        return list(self._refinement_records)


# --------------------------------------------------------------------------- #
# ωvd2: ωvd1 + γ inference engine over a BundleScope                         #
# --------------------------------------------------------------------------- #

@dataclass
class Vd2CecaProxy(Vd1CecaProxy):
    """ωvd2: extends ωvd1 with proxy-side γ inference.

    Before the action-choice prompt, the proxy LLM estimates values for all
    bundles in ``bundle_scope`` from the transcript, with no person queries.
    These γ estimates are surfaced in the action-choice prompt.

    ``gamma_refresh_every`` controls how often γ is recomputed (every N rounds).
    Set to 0 to disable γ refresh (γ remains empty, degenerating to ωvd1).
    Set to 1 to refresh on every round (maximum freshness, highest cost).
    """

    bundle_scope: BundleScope = field(default_factory=AllBundlesScope)
    gamma_refresh_every: int = 1

    _gamma_cache: dict[Bundle, float] = field(default_factory=dict, init=False, repr=False)
    _last_gamma_refresh_round: int = field(default=-1, init=False, repr=False)

    def _gamma_estimates(self) -> dict[Bundle, float]:
        return self._gamma_cache

    def _refresh_gamma(self) -> None:
        """Infer γ values for all scoped bundles via proxy LLM."""
        scoped_bundles = self.bundle_scope.bundles(self.items)
        if not scoped_bundles:
            return

        prompt = _gamma_inference_prompt(
            scenario_description=self.scenario_description,
            item_descriptions=self.item_descriptions,
            nl_transcript=self.nl_transcript,
            manifest_atoms=self._bid.atoms,
            bundles=scoped_bundles,
        )

        for attempt in range(1, self.max_parse_retries + 2):
            started = time.perf_counter()
            try:
                raw = self.proxy_client.complete(prompt)
            except Exception as exc:
                latency = time.perf_counter() - started
                self._log_proxy_call(
                    prompt=prompt, raw=None, parsed=None,
                    success=False, error=str(exc),
                    latency=latency, attempt=attempt, prompt_type="gamma_inference",
                )
                raise
            latency = time.perf_counter() - started

            try:
                result = _parse_gamma_response(raw, scoped_bundles)
            except ValueError as exc:
                self._log_proxy_call(
                    prompt=prompt, raw=raw, parsed=None,
                    success=False, error=str(exc),
                    latency=latency, attempt=attempt, prompt_type="gamma_inference",
                )
                if attempt > self.max_parse_retries:
                    return
                continue

            self._gamma_cache.update(result)
            self._log_proxy_call(
                prompt=prompt, raw=raw,
                parsed={"num_estimates": len(result)},
                success=True, error=None,
                latency=latency, attempt=attempt, prompt_type="gamma_inference",
            )
            return

    def _maybe_refresh_gamma(self, round_idx: int) -> None:
        if self.gamma_refresh_every <= 0:
            return
        # _last_gamma_refresh_round starts at -1; the < 0 guard ensures the
        # very first CECA step always seeds γ regardless of the interval setting.
        if self._last_gamma_refresh_round < 0 or round_idx - self._last_gamma_refresh_round >= self.gamma_refresh_every:
            self._refresh_gamma()
            self._last_gamma_refresh_round = round_idx


# --------------------------------------------------------------------------- #
# ωnvd: ωvd2 + initial NL questions to seed γ                                #
# --------------------------------------------------------------------------- #

@dataclass
class NvdCecaProxy(Vd2CecaProxy):
    """ωnvd: extends ωvd2 by asking NL questions before the first CECA step.

    ``num_nl_questions`` proxy-generated questions are posed to the person
    before the CECA loop begins. Their answers are appended to
    :attr:`nl_transcript` and γ is immediately refreshed, giving the action
    choice prompt richer context from the first round onward.
    """

    num_nl_questions: int = 1

    _nl_questions_done: bool = field(default=False, init=False, repr=False)

    def _ask_nl_questions(self) -> None:
        q_prompt = build_initial_proxy_question_prompt(
            scenario_description=self.scenario_description,
            item_descriptions=self.item_descriptions,
        )
        for _ in range(self.num_nl_questions):
            question: str | None = None

            for attempt in range(1, self.max_parse_retries + 2):
                started = time.perf_counter()
                try:
                    q_raw = self.proxy_client.complete(q_prompt)
                except Exception as exc:
                    latency = time.perf_counter() - started
                    self._log_proxy_call(
                        prompt=q_prompt, raw=None, parsed=None,
                        success=False, error=str(exc),
                        latency=latency, attempt=attempt, prompt_type="nl_question_gen",
                    )
                    raise
                latency = time.perf_counter() - started

                try:
                    parsed_q = parse_proxy_question_response(q_raw)
                    question = parsed_q.question
                except ValueError as exc:
                    self._log_proxy_call(
                        prompt=q_prompt, raw=q_raw, parsed=None,
                        success=False, error=str(exc),
                        latency=latency, attempt=attempt, prompt_type="nl_question_gen",
                    )
                    if attempt > self.max_parse_retries:
                        break
                    continue

                self._log_proxy_call(
                    prompt=q_prompt, raw=q_raw, parsed={"question": question},
                    success=True, error=None,
                    latency=latency, attempt=attempt, prompt_type="nl_question_gen",
                )
                break

            if question is None:
                continue

            self._stats.nl_queries += 1
            answer = self.person.answer_question(question)
            self.nl_transcript.append((question, answer))

    def _maybe_ask_nl_questions(self) -> None:
        if self._nl_questions_done:
            return
        self._ask_nl_questions()
        self._nl_questions_done = True
        # Seed γ immediately with the answers we just collected.
        self._refresh_gamma()
        self._last_gamma_refresh_round = 0
