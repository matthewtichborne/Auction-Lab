"""LLM-backed proxies that infer reusable XOR bids from bundle queries."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from auctionlab.auction_types import Bundle, Item
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.instances.base import DemandResponse, demand_rank_key
from auctionlab.llm.bundles import bundle_sort_key
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.interest_map import (
    InterestMapFailurePolicy,
    derive_interest_map,
    generate_candidate_bundles_from_interest_map,
)
from auctionlab.llm.knowledge import KnowledgeBase, TranscriptKnowledgeBase
from auctionlab.llm.logging import LlmCallRecord, current_timestamp
from auctionlab.llm.parsing import parse_proxy_question_response
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.prompts import build_initial_proxy_question_prompt
from auctionlab.llm.provisional_valuations import (
    PvCandidateBundleStats,
    PvChunkStats,
    compute_pv_candidate_bundle_stats,
    generate_provisional_valuations_chunked,
)
from auctionlab.llm.schemas import LlmInterestMap
from auctionlab.llm.value_calibration import (
    NO_CALIBRATION,
    ValueCalibration,
    legacy_calibration,
)
from auctionlab.proxies.base import ElicitationEvent, ProxyStats, RefinementRecord
from auctionlab.proxies.elicitation import candidate_refinements
from auctionlab.proxies.events import (
    CLOCK_DEMAND_CHANGED,
    CLOCK_NEAR_TIE,
    CLOCK_NEAR_ZERO_SURPLUS,
    CLOCK_PRICES,
    GENERATE_CANDIDATE_BUNDLES,
    INFER_INTEREST_MAP,
    INFER_PROVISIONAL_VALUES,
    INITIAL_PREFERENCE_QUESTION,
    SEALED_FEEDBACK,
    SUBMIT_BID,
    _INIT_EVENT_TYPES,
    _KNOWN_EVENT_TYPES,
    ProxyElicitationEvent,
    ProxyEventLogEntry,
    ProxyResponse,
)


_SUPPORTED_SIZE_DISCOUNT_FAMILIES = (None, "exponential")


def apply_value_calibration(
    raw_value: float,
    bundle_size: int,
    *,
    discount_inferred: bool,
    epsilon: float = 1.0,
    size_discount_family: str | None = None,
    size_discount_k0: int = 3,
    size_discount_gamma: float = 1.0,
) -> float:
    """Deprecated compatibility wrapper for the legacy epsilon transform.

    Superseded by :class:`~auctionlab.llm.value_calibration.ValueCalibration`,
    which additionally supports ``scale > 1`` (the correction the current
    estimator actually needs), an explicit ``family``, a disclosed-budget cap,
    and a stable config hash for run artefacts.

    Retained because several call sites and tests still express calibration
    through ``discount_inferred``/``epsilon``. Semantics are unchanged:
    ``discount_inferred=False`` returns the raw value untouched, so
    ``epsilon``/the size discount have **no effect whatsoever** without it.
    Delegates to
    :func:`~auctionlab.llm.value_calibration.legacy_calibration` so old and
    new paths share one implementation.

    Raises :exc:`ValueError` if ``size_discount_family`` is neither ``None``
    nor ``"exponential"``, or if ``size_discount_gamma`` is not strictly
    positive.
    """
    if size_discount_family not in _SUPPORTED_SIZE_DISCOUNT_FAMILIES:
        raise ValueError(
            "size_discount_family must be one of "
            f"{_SUPPORTED_SIZE_DISCOUNT_FAMILIES}, got {size_discount_family!r}"
        )
    if size_discount_gamma <= 0:
        raise ValueError(
            f"size_discount_gamma must be > 0, got {size_discount_gamma!r}"
        )

    return legacy_calibration(
        discount_inferred=discount_inferred,
        epsilon=epsilon,
        size_discount_family=size_discount_family,
        size_discount_k0=size_discount_k0,
        size_discount_gamma=size_discount_gamma,
    ).apply(raw_value, bundle_size)


def apply_epsilon_discount(
    raw_value: float,
    *,
    epsilon: float,
    discount_inferred: bool,
) -> float:
    """Epsilon-only special case of :func:`apply_value_calibration`."""
    return apply_value_calibration(
        raw_value, 0, discount_inferred=discount_inferred, epsilon=epsilon
    )


def enforce_atom_monotonicity(atoms: list[XorAtomicBid]) -> None:
    """Raise any superset atom whose value is below a subset atom's value.

    In XOR bids the utility function is u(b) = max_{a ⊆ b} v(a), so free
    disposal holds automatically at the utility level -- the atom values
    themselves need not be monotone.  The only genuine inconsistency is when
    a superset atom has a lower explicit value than a subset atom it contains:
    a WDP that assigns that superset atom would report a lower welfare than
    the subset atom deserves, potentially misallocating items.  Repairing
    this by *raising* the superset (not lowering the subset) preserves all
    singleton/subset information while giving the solver the correct floor.
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(atoms)):
            for j in range(len(atoms)):
                if i == j:
                    continue
                if (
                    atoms[i].bundle < atoms[j].bundle
                    and atoms[i].value > atoms[j].value
                ):
                    atoms[j] = XorAtomicBid(
                        bundle=atoms[j].bundle, value=atoms[i].value
                    )
                    changed = True


@dataclass
class TranscriptEntry:
    """Compact record of one proxy inference step."""

    kind: str
    content: str
    query_text: str | None = None
    response_summary: str | None = None


@dataclass
class LlmInferredXorProxy:
    """Infer and serve a bidder's XOR bid through an LLM person simulator.

    Candidate bundles are queried in size-first order so singleton values are
    available as optional calibration anchors for larger bundles.

    Provisional-value calibration is configured either through the preferred
    ``calibration`` field (a
    :class:`~auctionlab.llm.value_calibration.ValueCalibration`) or through
    the deprecated ``epsilon``/``size_discount_*`` fields. When
    ``calibration`` is set it wins outright and the per-call
    ``discount_inferred`` gate is ignored -- an explicit calibration is never
    silently disabled by a flag somewhere else.
    """

    bidder_id: str
    person: LlmPersonSimulator
    calibration: ValueCalibration | None = None
    epsilon: float = 0.75
    size_discount_family: str | None = None
    size_discount_k0: int = 3
    size_discount_gamma: float = 1.0
    transcript: list[TranscriptEntry] = field(default_factory=list)
    nl_transcript: list[tuple[str, str]] = field(default_factory=list)
    knowledge_base: KnowledgeBase | None = field(default=None)
    _cached_bid: XorBid | None = field(default=None, init=False, repr=False)
    _cached_discount_inferred: bool | None = field(
        default=None,
        init=False,
        repr=False,
    )
    refinement_query_count: int = field(default=0, init=False)
    pruning_query_count: int = field(default=0, init=False)
    refined_bundles: set[Bundle] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _last_refinement_query_text: str | None = field(
        default=None, init=False, repr=False
    )
    _last_refinement_response_summary: str | None = field(
        default=None, init=False, repr=False
    )
    interest_map: LlmInterestMap | None = field(default=None, init=False)
    _provisional_raw_values: dict[Bundle, float] | None = field(
        default=None, init=False, repr=False
    )
    _provisional_bundles: set[Bundle] = field(
        default_factory=set, init=False, repr=False
    )
    event_log: list[ProxyEventLogEntry] = field(
        default_factory=list, init=False, repr=False
    )
    last_pv_candidate_stats: PvCandidateBundleStats | None = field(
        default=None, init=False, repr=False
    )
    last_pv_chunk_stats: PvChunkStats | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon <= 1.0:
            raise ValueError("epsilon must be in (0, 1]")
        if self.size_discount_family not in _SUPPORTED_SIZE_DISCOUNT_FAMILIES:
            raise ValueError(
                "size_discount_family must be one of "
                f"{_SUPPORTED_SIZE_DISCOUNT_FAMILIES}, got "
                f"{self.size_discount_family!r}"
            )
        if self.size_discount_gamma <= 0:
            raise ValueError(
                f"size_discount_gamma must be > 0, got {self.size_discount_gamma!r}"
            )
        if self.knowledge_base is None:
            # Default: share nl_transcript so both stay in sync.
            self.knowledge_base = TranscriptKnowledgeBase(self.nl_transcript)

    # -- provisional-value calibration -----------------------------------

    def effective_calibration(
        self,
        discount_inferred: bool = True,
    ) -> ValueCalibration:
        """Return the calibration actually applied to inferred values.

        An explicit ``calibration`` always wins; otherwise the deprecated
        ``epsilon``/``size_discount_*`` fields are translated under the
        supplied ``discount_inferred`` gate.
        """
        if self.calibration is not None:
            return self.calibration
        return legacy_calibration(
            discount_inferred=discount_inferred,
            epsilon=self.epsilon,
            size_discount_family=self.size_discount_family,
            size_discount_k0=self.size_discount_k0,
            size_discount_gamma=self.size_discount_gamma,
        )

    def disclosed_budget(self) -> float | None:
        """The proxy-observable spending ceiling, if the person disclosed one.

        Read from the interest map's ``budget_hint``, i.e. from what the
        person actually said. Hidden ground-truth valuations are never
        consulted here.
        """
        if self.interest_map is None:
            return None
        hint = getattr(self.interest_map, "budget_hint", None)
        if hint is None:
            return None
        try:
            budget = float(hint)
        except (TypeError, ValueError):
            return None
        return budget if budget > 0 else None

    def _calibrate(
        self,
        raw_value: float,
        bundle_size: int,
        *,
        discount_inferred: bool,
    ) -> float:
        return self.effective_calibration(discount_inferred).apply(
            raw_value,
            bundle_size,
            disclosed_budget=self.disclosed_budget(),
        )

    def ask_initial_question(
        self,
        question_client: LlmClient | None = None,
        question: str | None = None,
    ) -> None:
        """Ask the person an initial open-ended preference question (ωnvd).

        The question/answer pair is recorded in ``nl_transcript`` and used as
        context for all later value inference. No-op if already asked.
        Supplying ``question`` delivers a frozen scenario-level instrument
        without a question-generation LLM call. Omitting it retains the
        legacy proxy-generated path for robustness experiments.
        """
        if self.knowledge_base:
            return

        if question is not None:
            question = question.strip()
            if not question:
                raise ValueError("opening question must be non-empty")
            answer = self.person.answer_question(question)
            self.knowledge_base.add_qa(question, answer)
            self.transcript.append(
                TranscriptEntry(kind="nl_question", content=question)
            )
            self.transcript.append(
                TranscriptEntry(kind="nl_answer", content=answer)
            )
            return

        prompt = build_initial_proxy_question_prompt(
            scenario_description=self.person.scenario_description,
            item_descriptions=self.person.item_descriptions,
        )
        client = question_client or self.person.client
        max_retries = getattr(self.person, "max_parse_retries", 0)

        question: str | None = None
        for attempt in range(1, max_retries + 2):
            _retry = f"  [retry {attempt}]" if attempt > 1 else ""
            print(
                f"  {self.bidder_id:<12}  nl-gen{_retry}",
                end="",
                flush=True,
            )
            started = time.perf_counter()

            try:
                raw_response = client.complete(prompt)
            except Exception as exc:
                latency = time.perf_counter() - started
                print(f"  →  ERROR  ({latency:.1f}s)  {exc}", flush=True)
                raise

            latency = time.perf_counter() - started

            try:
                parsed = parse_proxy_question_response(raw_response)
            except ValueError as exc:
                print(f"  →  parse error  ({latency:.1f}s)  {exc}", flush=True)
                if attempt > max_retries:
                    raise
                continue

            print(f"  →  done  ({latency:.1f}s)", flush=True)
            question = parsed.question
            if self.person.logger is not None:
                self.person.logger.log(
                    LlmCallRecord(
                        timestamp=current_timestamp(),
                        bidder_id=self.bidder_id,
                        prompt_type="proxy_nl_gen",
                        prompt=prompt,
                        raw_response=raw_response,
                        parsed_response={"question": question},
                        success=True,
                        error=None,
                        latency_seconds=latency,
                        model=getattr(
                            client,
                            "_auctionlab_model",
                            getattr(client, "model", self.person.model_name),
                        ),
                        provider=getattr(
                            client,
                            "_auctionlab_provider",
                            None,
                        ),
                        llm_role=getattr(
                            client,
                            "_auctionlab_llm_role",
                            "proxy",
                        ),
                        attempt=attempt,
                        input_tokens=getattr(client, "_last_input_tokens", None),
                        output_tokens=getattr(client, "_last_output_tokens", None),
                        total_tokens=getattr(client, "_last_total_tokens", None),
                    )
                )
            break

        answer = self.person.answer_question(question)

        self.knowledge_base.add_qa(question, answer)
        self.transcript.append(
            TranscriptEntry(kind="nl_question", content=question)
        )
        self.transcript.append(
            TranscriptEntry(kind="nl_answer", content=answer)
        )

    def build_interest_map(
        self,
        client: LlmClient | None = None,
        *,
        failure_policy: InterestMapFailurePolicy = "all_items",
    ) -> LlmInterestMap:
        """Derive and cache an interest map from the most recent NL Q/A pair.

        Must be called after :meth:`ask_initial_question`.  Uses ``client``
        for the proxy-side parsing call; falls back to the person's client.
        """
        if not self.nl_transcript:
            raise RuntimeError(
                "No NL question/answer recorded; call ask_initial_question() first."
            )
        nl_question, nl_answer = self.nl_transcript[-1]
        map_client = client or self.person.client
        self.interest_map = derive_interest_map(
            client=map_client,
            scenario_description=self.person.scenario_description,
            item_descriptions=self.person.item_descriptions,
            nl_question=nl_question,
            nl_answer=nl_answer,
            logger=self.person.logger,
            bidder_id=self.bidder_id,
            model_name=getattr(
                map_client,
                "_auctionlab_model",
                getattr(map_client, "model", self.person.model_name),
            ),
            scenario_id=self.person.scenario_id,
            failure_policy=failure_policy,
        )
        self.transcript.append(
            TranscriptEntry(
                kind="interest_map",
                content=(
                    f"interested={sorted(self.interest_map.interested_items)}; "
                    f"excluded={sorted(self.interest_map.excluded_items)}; "
                    f"complements={[sorted(g) for g in self.interest_map.complementary_groups]}; "
                    "substitutes="
                    f"{[{'items': sorted(g.items), 'mode': g.acquisition_mode} for g in self.interest_map.substitute_groups]}"
                ),
            )
        )
        return self.interest_map

    def set_provisional_bid(
        self,
        bundle_values: dict[Bundle, float],
        *,
        discount_inferred: bool = True,
    ) -> None:
        """Pre-populate the cached XOR bid from externally computed values.

        Intended for use with :meth:`build_provisional_valuations`. The values
        in ``bundle_values`` are stored as-is (already discounted by the caller
        when ``discount_inferred=True``); no LLM query is issued. An existing
        cached bid is replaced entirely.
        """
        atoms = [
            XorAtomicBid(bundle=bundle, value=value)
            for bundle, value in bundle_values.items()
            if bundle
        ]
        self._cached_bid = XorBid(bidder_id=self.bidder_id, atoms=atoms)
        self._cached_discount_inferred = discount_inferred
        self._provisional_bundles = {b for b in bundle_values if b}
        self._enforce_subset_superset_monotonicity()
        self.transcript.append(
            TranscriptEntry(
                kind="provisional_bid",
                content=(
                    f"bundles={len(atoms)}; "
                    f"discount_inferred={discount_inferred}"
                ),
            )
        )

    def _enforce_subset_superset_monotonicity(self) -> None:
        """Repair the cached bid in place; see :func:`enforce_atom_monotonicity`."""
        if self._cached_bid is None:
            return
        enforce_atom_monotonicity(self._cached_bid.atoms)

    def build_provisional_valuations(
        self,
        candidate_bundles: list[Bundle],
        *,
        client: LlmClient | None = None,
        interest_map: LlmInterestMap | None = None,
        discount_inferred: bool = True,
        max_candidate_bundles: int | None = None,
        pv_chunk_size: int | None = None,
        max_parse_retries: int = 0,
    ) -> dict[Bundle, float]:
        """Generate provisional values and pre-populate the cached XOR bid.

        Calls the LLM once (or, when ``pv_chunk_size`` is set and the
        candidate count exceeds it, once per deterministic chunk -- see
        :func:`~auctionlab.llm.provisional_valuations.generate_provisional_valuations_chunked`)
        over all ``candidate_bundles`` and stores the resulting values via
        :meth:`set_provisional_bid`. The optional ``interest_map`` enriches
        the prompt; the function is fully usable without it. Returns the raw
        value map (before epsilon discounting). Deliberately does not pass
        the person's preference seed: estimates are grounded only in the NL
        question/answer actually exchanged, not private knowledge of the
        person's true values.

        ``max_candidate_bundles``, when set, deterministically caps how many
        of ``candidate_bundles`` are actually sent to the LLM (before any
        chunking). When ``None`` (default), every candidate bundle is sent --
        there is no automatic, token-budget-derived truncation. The
        resulting counts are recorded on :attr:`last_pv_candidate_stats`;
        chunking activity is recorded separately on
        :attr:`last_pv_chunk_stats`.

        ``pv_chunk_size`` unset/0 (default), or a candidate count already at
        or below it, makes exactly one PV call -- byte-for-byte the same
        behaviour as before chunking existed. ``max_parse_retries`` retries a
        malformed response for either the single call or, independently,
        each chunk.
        """
        if not self.nl_transcript:
            raise RuntimeError(
                "No NL question/answer recorded; call ask_initial_question() first."
            )
        nl_question, nl_answer = self.nl_transcript[-1]
        pv_client = client or self.person.client

        self.last_pv_candidate_stats = compute_pv_candidate_bundle_stats(
            candidate_bundles, max_candidate_bundles
        )

        t0 = time.perf_counter()
        print(f"  {self.bidder_id:<12}  pv", end="", flush=True)

        raw_values, chunk_stats = generate_provisional_valuations_chunked(
            client=pv_client,
            scenario_description=self.person.scenario_description,
            item_descriptions=self.person.item_descriptions,
            nl_question=nl_question,
            nl_answer=nl_answer,
            candidate_bundles=candidate_bundles,
            interest_map=interest_map,
            logger=self.person.logger,
            bidder_id=self.bidder_id,
            model_name=getattr(
                pv_client,
                "_auctionlab_model",
                getattr(pv_client, "model", self.person.model_name),
            ),
            max_bundles=max_candidate_bundles,
            scenario_id=self.person.scenario_id,
            pv_chunk_size=pv_chunk_size,
            max_parse_retries=max_parse_retries,
        )
        self.last_pv_chunk_stats = chunk_stats

        latency = time.perf_counter() - t0
        chunk_suffix = (
            f"  [{chunk_stats.pv_chunks} chunks]" if chunk_stats.chunking_used else ""
        )
        print(
            f"  →  {len(raw_values)} values  ({latency:.1f}s){chunk_suffix}",
            flush=True,
        )

        self._provisional_raw_values = raw_values
        calibrated = {
            bundle: self._calibrate(
                value,
                len(bundle),
                discount_inferred=discount_inferred,
            )
            for bundle, value in raw_values.items()
        }
        self.set_provisional_bid(calibrated, discount_inferred=discount_inferred)
        return raw_values

    def replay_elicitation(
        self,
        *,
        nl_question: str,
        nl_answer: str,
        interest_map: LlmInterestMap | None = None,
        provisional_raw_values: dict[Bundle, float] | None = None,
        discount_inferred: bool = True,
    ) -> None:
        """Seed this proxy from previously-computed elicitation results.

        Replays the side effects of :meth:`ask_initial_question`,
        :meth:`build_interest_map`, and :meth:`build_provisional_valuations`
        without issuing any LLM call. Intended for runs that compare several
        mechanisms over the same bidder: the NL question, interest map, and
        bulk PV call are computed once, then replayed into one fresh proxy
        instance per mechanism arm so each arm starts from an identical
        informational baseline with independent mutable state.
        """
        self.knowledge_base.add_qa(nl_question, nl_answer)
        self.transcript.append(
            TranscriptEntry(kind="nl_question", content=nl_question)
        )
        self.transcript.append(
            TranscriptEntry(kind="nl_answer", content=nl_answer)
        )

        if interest_map is not None:
            self.interest_map = interest_map
            self.transcript.append(
                TranscriptEntry(
                    kind="interest_map",
                    content=(
                        f"interested={sorted(interest_map.interested_items)}; "
                        f"excluded={sorted(interest_map.excluded_items)}; "
                        "complements="
                        f"{[sorted(g) for g in interest_map.complementary_groups]}; "
                        "substitutes="
                        f"{[{'items': sorted(g.items), 'mode': g.acquisition_mode} for g in interest_map.substitute_groups]}"
                    ),
                )
            )

        if provisional_raw_values is not None:
            # self.interest_map is assigned above, so the disclosed-budget cap
            # sees the replayed budget hint rather than an empty proxy.
            self._provisional_raw_values = provisional_raw_values
            calibrated = {
                bundle: self._calibrate(
                    value,
                    len(bundle),
                    discount_inferred=discount_inferred,
                )
                for bundle, value in provisional_raw_values.items()
            }
            self.set_provisional_bid(calibrated, discount_inferred=discount_inferred)

    def handle_event(self, event: ProxyElicitationEvent) -> ProxyResponse:
        """Dispatch a proxy lifecycle event to the appropriate inner method.

        Handles the four initialisation event types:
        ``initial_preference_question``, ``infer_interest_map``,
        ``generate_candidate_bundles``, ``infer_provisional_values``.

        Raises :exc:`ValueError` for unknown or mechanism-only event types
        (those should be dispatched via :class:`LlmAuctionProxyAdapter`).
        """
        if event.event_type not in _KNOWN_EVENT_TYPES:
            raise ValueError(
                f"Unknown proxy event type: {event.event_type!r}. "
                f"Known types: {sorted(_KNOWN_EVENT_TYPES)}"
            )
        if event.event_type not in _INIT_EVENT_TYPES:
            raise ValueError(
                f"Event type {event.event_type!r} is a mechanism event; "
                "dispatch via LlmAuctionProxyAdapter.handle_event() instead."
            )

        p = event.payload
        state_delta: dict = {}

        if event.event_type == INITIAL_PREFERENCE_QUESTION:
            self.ask_initial_question(
                p.get("client"),
                question=p.get("question"),
            )
            nl_q, nl_a = self.nl_transcript[-1] if self.nl_transcript else ("", "")
            state_delta = {"nl_transcript_length": len(self.nl_transcript)}
            response = ProxyResponse(
                response_type="natural_language_answer",
                payload={"question": nl_q, "answer": nl_a},
                state_delta=state_delta,
            )

        elif event.event_type == INFER_INTEREST_MAP:
            im = self.build_interest_map(
                p.get("client"),
                failure_policy=p.get("failure_policy", "all_items"),
            )
            state_delta = {
                "interested_items": sorted(im.interested_items),
                "excluded_items": sorted(im.excluded_items),
            }
            response = ProxyResponse(
                response_type="interest_map",
                payload={"interest_map": im},
                state_delta=state_delta,
            )

        elif event.event_type == GENERATE_CANDIDATE_BUNDLES:
            cb = self.candidate_bundles_from_interest_map(
                p["all_items"],
                max_candidate_bundles=p.get("max_candidate_bundles"),
            )
            state_delta = {"candidate_bundle_count": len(cb)}
            response = ProxyResponse(
                response_type="candidate_bundles",
                payload={"candidate_bundles": cb},
                state_delta=state_delta,
            )

        else:  # INFER_PROVISIONAL_VALUES
            candidate_bundles = p.get("candidate_bundles", [])
            raw_values = self.build_provisional_valuations(
                candidate_bundles,
                client=p.get("client"),
                interest_map=p.get("interest_map", self.interest_map),
                discount_inferred=p.get("discount_inferred", True),
                max_candidate_bundles=p.get("max_candidate_bundles"),
                pv_chunk_size=p.get("pv_chunk_size"),
                max_parse_retries=p.get("max_parse_retries", 0),
            )
            state_delta = {"provisional_value_count": len(raw_values)}
            if self.last_pv_candidate_stats is not None:
                state_delta.update(self.last_pv_candidate_stats.as_dict())
            if self.last_pv_chunk_stats is not None:
                state_delta.update(self.last_pv_chunk_stats.as_dict())
            response = ProxyResponse(
                response_type="provisional_values",
                payload={"raw_values": raw_values},
                state_delta=state_delta,
            )

        self.event_log.append(ProxyEventLogEntry(
            event_type=event.event_type,
            bidder_id=self.bidder_id,
            mechanism=event.mechanism,
            round_idx=event.round_idx,
            response_type=response.response_type,
            records_count=len(response.records),
            state_delta=state_delta,
        ))
        return response

    def candidate_bundles_from_interest_map(
        self,
        all_items: list[Item],
        *,
        max_candidate_bundles: int | None = None,
    ) -> list[Bundle]:
        """Return priority-ordered candidate bundles derived from the interest map.

        Must call :meth:`build_interest_map` first.  Pass
        ``max_candidate_bundles=None`` for a full uncapped run.
        """
        if self.interest_map is None:
            raise RuntimeError(
                "No interest map available; call build_interest_map() first."
            )
        return generate_candidate_bundles_from_interest_map(
            self.interest_map,
            all_items,
            max_candidate_bundles=max_candidate_bundles,
        )

    def _infer_reported_values(
        self,
        candidate_bundles: list[Bundle],
        *,
        discount_inferred: bool,
        use_anchor_values: bool,
    ) -> list[tuple[Bundle, float]]:
        """Query bundles once in deterministic singleton-first order."""
        inferred: list[tuple[Bundle, float]] = []
        transcript_context = self.knowledge_base.context_for_prompt()
        ordered_bundles = sorted(
            (
                frozenset(candidate_bundle)
                for candidate_bundle in candidate_bundles
                if candidate_bundle
            ),
            key=bundle_sort_key,
        )
        singleton_values: dict[Bundle, float] = {}

        for bundle in ordered_bundles:
            anchors = (
                {
                    singleton_bundle: value
                    for singleton_bundle, value in singleton_values.items()
                    if singleton_bundle.issubset(bundle)
                }
                if use_anchor_values
                else {}
            )
            value = self.person.value_query(
                bundle,
                anchor_values=anchors or None,
                transcript_context=transcript_context or None,
            )
            if len(bundle) == 1:
                singleton_values[bundle] = value

            reported_value = self._calibrate(
                value,
                len(bundle),
                discount_inferred=discount_inferred,
            )
            inferred.append((bundle, reported_value))
            transcript_kind = (
                "value_query_with_anchors"
                if anchors
                else "value_query"
            )
            anchor_entries = [
                f"[{','.join(sorted(anchor_bundle))}]: {anchor_value}"
                for anchor_bundle, anchor_value in sorted(
                    anchors.items(),
                    key=lambda item: bundle_sort_key(item[0]),
                )
            ]
            anchor_text = (
                f"; anchors={{{', '.join(anchor_entries)}}}"
                if anchor_entries
                else ""
            )
            self.transcript.append(
                TranscriptEntry(
                    kind=transcript_kind,
                    content=(
                        f"bundle={sorted(bundle)}; value={value}; "
                        f"reported_value={reported_value}{anchor_text}"
                    ),
                )
            )

        return inferred

    def infer_values(
        self,
        candidate_bundles: list[Bundle],
        *,
        discount_inferred: bool = True,
        use_anchor_values: bool = True,
    ) -> dict[Bundle, float]:
        return dict(
            self._infer_reported_values(
                candidate_bundles,
                discount_inferred=discount_inferred,
                use_anchor_values=use_anchor_values,
            )
        )

    def infer_xor_bid(
        self,
        candidate_bundles: list[Bundle],
        *,
        discount_inferred: bool = True,
        use_anchor_values: bool = True,
    ) -> XorBid:
        inferred = self._infer_reported_values(
            candidate_bundles,
            discount_inferred=discount_inferred,
            use_anchor_values=use_anchor_values,
        )
        atoms = [
            XorAtomicBid(bundle=bundle, value=value)
            for bundle, value in inferred
        ]
        enforce_atom_monotonicity(atoms)
        return XorBid(bidder_id=self.bidder_id, atoms=atoms)

    def infer_cached_xor_bid(
        self,
        candidate_bundles: list[Bundle],
        *,
        discount_inferred: bool = True,
        use_anchor_values: bool = True,
        force_refresh: bool = False,
    ) -> XorBid:
        """Reuse an inferred bid across clock rounds to avoid new LLM calls."""
        if self._cached_bid is None or force_refresh:
            self._cached_bid = self.infer_xor_bid(
                candidate_bundles,
                discount_inferred=discount_inferred,
                use_anchor_values=use_anchor_values,
            )
            self._cached_discount_inferred = discount_inferred

        return self._cached_bid

    def reset_refinement_state(self) -> None:
        """Reset per-clock-run refinement usage without clearing the bid."""
        self.refinement_query_count = 0
        self.refined_bundles.clear()

    def refine_bundle_value(
        self,
        bundle: Bundle,
        reason: str,
        use_anchor_values: bool = True,
    ) -> float:
        """Revalue one bundle and update its atom in the cached XOR bid."""
        bundle = frozenset(bundle)
        if not bundle:
            raise ValueError("Cannot refine the empty bundle")

        return self._revalue_and_upsert_atom(
            bundle,
            reason,
            use_anchor_values=use_anchor_values,
            transcript_kind="refinement_value_query",
        )

    def _revalue_and_upsert_atom(
        self,
        bundle: Bundle,
        reason: str,
        *,
        use_anchor_values: bool,
        transcript_kind: str,
    ) -> float:
        if self._cached_bid is None:
            raise RuntimeError(
                "No cached XOR bid is available; call infer_cached_xor_bid first"
            )

        calibration = self.effective_calibration(
            bool(self._cached_discount_inferred)
        )
        anchors: dict[Bundle, float] = {}
        if use_anchor_values:
            for atom in self._cached_bid.atoms:
                if (
                    len(atom.bundle) != 1
                    or atom.bundle == bundle
                    or not atom.bundle.issubset(bundle)
                ):
                    continue
                raw_value = atom.value
                # Already-refined atoms hold exact answers and were never
                # calibrated, so only provisional atoms are inverted.
                if atom.bundle not in self.refined_bundles:
                    raw_value = calibration.invert(raw_value, len(atom.bundle))
                anchors[atom.bundle] = raw_value

        value = self.person.value_query(
            bundle,
            anchor_values=anchors or None,
            transcript_context=self.knowledge_base.context_for_prompt() or None,
            elicitation_context=reason or None,
        )
        query_text = self.person._last_prompt
        response_summary = self.person._last_response_summary
        self._last_refinement_query_text = query_text
        self._last_refinement_response_summary = response_summary

        # Exact deterministic answers are ground truth, not estimates: they
        # are reported verbatim under every calibration.
        exact_oracle_value = self.person.ground_truth_valuations is not None
        reported_value = (
            value
            if exact_oracle_value
            else calibration.apply(
                value,
                len(bundle),
                disclosed_budget=self.disclosed_budget(),
            )
        )
        replacement = XorAtomicBid(
            bundle=bundle,
            value=reported_value,
        )

        for idx, atom in enumerate(self._cached_bid.atoms):
            if atom.bundle == bundle:
                self._cached_bid.atoms[idx] = replacement
                break
        else:
            self._cached_bid.atoms.append(replacement)

        self._enforce_subset_superset_monotonicity()
        if exact_oracle_value:
            # A noisy provisional subset may currently exceed this exact
            # superset value. Free disposal is already represented by XOR
            # utility (the subset atom remains usable), so do not let the
            # monotonicity convenience repair overwrite an oracle answer.
            for idx, atom in enumerate(self._cached_bid.atoms):
                if atom.bundle == bundle:
                    self._cached_bid.atoms[idx] = replacement
                    break
        # Monotonicity repair can lift the just-refined atom (e.g. if a
        # cached subset has a higher value), so read back the final value
        # rather than trusting the raw query result.
        reported_value = next(
            atom.value
            for atom in self._cached_bid.atoms
            if atom.bundle == bundle
        )

        self._provisional_bundles.discard(bundle)
        self.refinement_query_count += 1
        self.refined_bundles.add(bundle)
        self.transcript.append(
            TranscriptEntry(
                kind=transcript_kind,
                content=(
                    f"bundle={sorted(bundle)}; reason={reason}; "
                    f"value={value}; reported_value={reported_value}"
                ),
                query_text=query_text,
                response_summary=response_summary,
            )
        )
        return reported_value

    def refine_via_demand_query(
        self,
        bundle: Bundle,
        prices: dict[Item, float],
        reason: str,
        use_anchor_values: bool = True,
    ) -> tuple[Bundle | None, float | None]:
        """Refine via a demand query (ωvd1/ωvd2 message subroutine).

        Asks whether the person is satisfied with ``bundle`` at ``prices``.
        If satisfied, the cached bid is left unchanged and ``(None, None)``
        is returned. If not, and the person reports a different preferred
        bundle, that bundle's atom is revalued and upserted instead, and
        ``(preferred_bundle, value)`` is returned. If the person reports no
        useful alternative, falls back to :meth:`refine_bundle_value` on
        ``bundle`` and returns ``(bundle, value)``.
        """
        bundle = frozenset(bundle)
        if not bundle:
            raise ValueError("Cannot refine the empty bundle")

        response = self.person.demand_query(bundle, prices)
        demand_query_text = self.person._last_prompt
        demand_response_summary = self.person._last_response_summary
        self._last_refinement_query_text = demand_query_text
        self._last_refinement_response_summary = demand_response_summary

        if response.satisfied:
            self.refinement_query_count += 1
            self.refined_bundles.add(bundle)
            # satisfied=True is a revealed lower bound: value >= bundle_price.
            # If the cached atom is below this price, raise it to the floor.
            bundle_price = sum(prices.get(item, 0.0) for item in bundle)
            if self._cached_bid is not None:
                current = next(
                    (a.value for a in self._cached_bid.atoms if a.bundle == bundle),
                    None,
                )
                if current is not None and bundle_price > current:
                    replacement = XorAtomicBid(bundle=bundle, value=bundle_price)
                    for idx, atom in enumerate(self._cached_bid.atoms):
                        if atom.bundle == bundle:
                            self._cached_bid.atoms[idx] = replacement
                            break
                    self._enforce_subset_superset_monotonicity()
                    final_value = next(
                        atom.value
                        for atom in self._cached_bid.atoms
                        if atom.bundle == bundle
                    )
                    self.transcript.append(
                        TranscriptEntry(
                            kind="demand_query",
                            content=(
                                f"bundle={sorted(bundle)}; reason={reason}; "
                                f"satisfied=True; raised_to_bundle_price={bundle_price}"
                            ),
                            query_text=demand_query_text,
                            response_summary=demand_response_summary,
                        )
                    )
                    return bundle, final_value
            self.transcript.append(
                TranscriptEntry(
                    kind="demand_query",
                    content=(
                        f"bundle={sorted(bundle)}; reason={reason}; "
                        "satisfied=True"
                    ),
                    query_text=demand_query_text,
                    response_summary=demand_response_summary,
                )
            )
            return None, None

        preferred_bundle = (
            frozenset(
                item
                for item in response.preferred_bundle
                if item in self.person.item_descriptions
            )
            if response.preferred_bundle is not None
            else None
        )

        if preferred_bundle and preferred_bundle != bundle:
            self.transcript.append(
                TranscriptEntry(
                    kind="demand_query",
                    content=(
                        f"bundle={sorted(bundle)}; reason={reason}; "
                        f"satisfied=False; preferred_bundle="
                        f"{sorted(preferred_bundle)}"
                    ),
                    query_text=demand_query_text,
                    response_summary=demand_response_summary,
                )
            )
            value = self._revalue_and_upsert_atom(
                preferred_bundle,
                reason,
                use_anchor_values=use_anchor_values,
                transcript_kind="refinement_demand_query",
            )
            self.refined_bundles.add(bundle)
            return preferred_bundle, value

        self.transcript.append(
            TranscriptEntry(
                kind="demand_query",
                content=(
                    f"bundle={sorted(bundle)}; reason={reason}; "
                    "satisfied=False; no alternative reported"
                ),
                query_text=demand_query_text,
                response_summary=demand_response_summary,
            )
        )
        value = self.refine_bundle_value(
            bundle,
            reason,
            use_anchor_values=use_anchor_values,
        )
        return bundle, value

    def ranked_surplus_atoms(
        self,
        prices: dict[Item, float],
    ) -> list[tuple[XorAtomicBid, float]]:
        """Rank every cached atom by surplus and deterministic tie-breaks."""
        if self._cached_bid is None:
            raise RuntimeError(
                "No cached XOR bid is available; call infer_cached_xor_bid first"
            )

        ranked = [
            (
                atom,
                atom.value - sum(prices[item] for item in atom.bundle),
            )
            for atom in self._cached_bid.atoms
        ]
        ranked.sort(
            key=lambda ranked_atom: demand_rank_key(
                ranked_atom[0].bundle,
                ranked_atom[0].value,
                prices,
            )
        )
        return ranked

    def clock_demand_from_cached_bid(
        self,
        prices: dict[Item, float],
        *,
        top_k: int = 1,
    ) -> DemandResponse:
        """Answer clock demand deterministically from the cached XOR bid.

        The bidder's ``top_k`` best-surplus bundles at ``prices`` are
        reported as primary demand (``primary_bundles``) and drive excess
        demand / price increases across the union of items they contain;
        ``primary_bundle`` is just the single best of those.
        ``supplementary_atoms`` always returns every atom currently in the
        cached bid (not capped by ``top_k``), so the final supplementary
        WDP+VCG resolution sees everything this proxy has ever elicited,
        regardless of whether a bundle happened to be a top-k choice at any
        specific round's prices.
        """
        if self._cached_bid is None:
            raise RuntimeError(
                "No cached XOR bid is available; call infer_cached_xor_bid first"
        )
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        candidates = [
            atom
            for atom, surplus in self.ranked_surplus_atoms(prices)
            if surplus > 0.0
        ]
        supplementary_atoms = [
            atom for atom in self._cached_bid.atoms if atom.bundle
        ]

        return DemandResponse(
            primary_bundle=candidates[0].bundle if candidates else None,
            supplementary_atoms=supplementary_atoms,
            primary_bundles=[atom.bundle for atom in candidates[:top_k]],
        )

    def clock_demand_with_refinement(
        self,
        prices: dict[Item, float],
        top_k: int,
        round_idx: int,
        previous_primary_bundle: Bundle | None,
        margin_threshold: float,
        tie_threshold: float,
        max_refinement_queries: int,
        use_anchor_values: bool = True,
        refinement_strategy: str = "value_query",
        priority_scoring: bool = False,
    ) -> DemandResponse:
        """Refine uncertain clock values, then answer from the updated bid.

        With ``priority_scoring=True``, candidates are sorted by pivotality
        score (highest first) before spending the refinement budget, matching
        the behaviour of ``proxy_clock_runner`` when
        ``ProxyClockConfig.priority_scoring=True``.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if margin_threshold < 0.0:
            raise ValueError("margin_threshold must be non-negative")
        if tie_threshold < 0.0:
            raise ValueError("tie_threshold must be non-negative")
        if max_refinement_queries < 0:
            raise ValueError("max_refinement_queries must be non-negative")
        if refinement_strategy not in ("value_query", "demand_query"):
            raise ValueError(
                "refinement_strategy must be 'value_query' or 'demand_query', "
                f"got {refinement_strategy!r}"
            )

        ranked = self.ranked_surplus_atoms(prices)
        current_primary_bundle = (
            ranked[0][0].bundle
            if ranked and ranked[0][1] > 0.0
            else None
        )

        candidates = candidate_refinements(
            ranked=ranked,
            current_primary_bundle=current_primary_bundle,
            previous_primary_bundle=previous_primary_bundle,
            margin_threshold=margin_threshold,
            tie_threshold=tie_threshold,
            round_idx=round_idx,
        )

        if priority_scoring:
            candidates = sorted(candidates, key=lambda c: c[3], reverse=True)

        seen_candidates: set[Bundle] = set()
        for bundle, _event_type, reason, _score in candidates:
            if (
                not bundle
                or bundle in seen_candidates
                or bundle in self.refined_bundles
                or self.refinement_query_count >= max_refinement_queries
            ):
                continue
            seen_candidates.add(bundle)
            if refinement_strategy == "demand_query":
                self.refine_via_demand_query(
                    bundle,
                    prices,
                    reason,
                    use_anchor_values=use_anchor_values,
                )
            else:
                self.refine_bundle_value(
                    bundle,
                    reason,
                    use_anchor_values=use_anchor_values,
                )

        return self.clock_demand_from_cached_bid(
            prices,
            top_k=top_k,
        )

@dataclass
class LlmAuctionProxyAdapter:
    """Adapt :class:`LlmInferredXorProxy` to the common proxy protocols.

    The adapter does not change any existing inference/refinement behavior;
    it simply gives the LLM proxy a uniform ``current_bid``/``submit_bid``/
    ``demand_at_prices``/``refine`` surface so that proxy-mediated sealed and
    clock runners can treat it the same as any other proxy. Static runs that
    never call ``refine``/``receive_provisional_feedback`` behave exactly as
    before.
    """

    bidder_id: str
    proxy: LlmInferredXorProxy
    candidate_bundles: list[Bundle]
    discount_inferred: bool = True
    use_anchor_values: bool = True
    refinement_strategy: str = "value_query"
    _stats: ProxyStats = field(default_factory=ProxyStats, init=False)
    _refinement_records: list[RefinementRecord] = field(
        default_factory=list, init=False, repr=False
    )

    def current_bid(self) -> XorBid:
        had_cached_bid = self.proxy._cached_bid is not None
        bid = self.proxy.infer_cached_xor_bid(
            self.candidate_bundles,
            discount_inferred=self.discount_inferred,
            use_anchor_values=self.use_anchor_values,
        )
        if not had_cached_bid:
            self._stats.value_queries += sum(
                1 for bundle in self.candidate_bundles if bundle
            )
        return bid

    def submit_bid(self) -> XorBid:
        return self.current_bid()

    def demand_at_prices(
        self,
        prices: dict[Item, float],
        round_idx: int = 0,
        top_k: int = 1,
    ) -> DemandResponse:
        self.current_bid()
        self._stats.demand_queries += 1
        return self.proxy.clock_demand_from_cached_bid(prices, top_k=top_k)

    def _refine_single_bundle(
        self,
        bundle: Bundle,
        event: ElicitationEvent,
        old_values: dict[Bundle, float],
    ) -> None:
        """Refine one bundle and append a RefinementRecord."""
        reason = event.reason or event.event_type

        # Skip value-query re-elicitation for bundles already individually
        # queried in the initial phase. The initial value (with full NL
        # transcript context) is at least as good as a context-free re-query,
        # and re-querying wastes refinement budget and adds LLM noise.
        # Provisionally-valued bundles (from a bulk PV call) are NOT skipped:
        # they benefit from focused refinement queries. Demand queries are
        # price-dependent so they always proceed.
        if self.refinement_strategy != "demand_query" or not event.prices:
            cached = self.proxy._cached_bid
            if cached is not None and any(
                a.bundle == bundle for a in cached.atoms
            ):
                if bundle not in self.proxy._provisional_bundles:
                    return

        if self.refinement_strategy == "demand_query" and event.prices:
            revalued_bundle, new_value = self.proxy.refine_via_demand_query(
                bundle,
                prices=event.prices,
                reason=reason,
                use_anchor_values=self.use_anchor_values,
            )
            self._stats.refinement_queries = self.proxy.refinement_query_count
            if revalued_bundle is None:
                return
            bundle = revalued_bundle
        else:
            new_value = self.proxy.refine_bundle_value(
                bundle,
                reason=reason,
                use_anchor_values=self.use_anchor_values,
            )
            self._stats.refinement_queries = self.proxy.refinement_query_count

        self._refinement_records.append(
            RefinementRecord(
                bidder_id=self.bidder_id,
                mechanism=event.mechanism,
                event_type=event.event_type,
                round_idx=event.round_idx,
                bundle=bundle,
                old_value=old_values.get(bundle),
                new_value=new_value,
                reason=event.reason,
                query_text=self.proxy._last_refinement_query_text,
                response_summary=self.proxy._last_refinement_response_summary,
            )
        )

    def refine(self, event: ElicitationEvent) -> None:
        # Determine which bundles to refine: use the multi-bundle field when
        # present (e.g. near-tie events), falling back to the single bundle.
        target_bundles: list[Bundle]
        if event.bundles:
            target_bundles = [frozenset(b) for b in event.bundles if b]
        elif event.bundle:
            target_bundles = [frozenset(event.bundle)]
        else:
            return

        bid = self.current_bid()
        old_values = {atom.bundle: atom.value for atom in bid.atoms}

        for bundle in target_bundles:
            if not bundle:
                continue
            self._refine_single_bundle(bundle, event, old_values)

    def receive_provisional_feedback(self, event: ElicitationEvent) -> None:
        self.refine(event)

    def handle_event(self, event: ProxyElicitationEvent) -> ProxyResponse:
        """Dispatch a :class:`~auctionlab.proxies.events.ProxyElicitationEvent`.

        Initialisation events (``initial_preference_question``,
        ``infer_interest_map``, ``generate_candidate_bundles``,
        ``infer_provisional_values``) are forwarded to the inner
        :class:`LlmInferredXorProxy`. Mechanism events (``submit_bid``,
        ``sealed_feedback``, ``clock_prices``) are handled directly by the
        adapter's existing methods so that stats and refinement records are
        updated correctly.

        Raises :exc:`ValueError` for unknown event types.
        """
        if event.event_type not in _KNOWN_EVENT_TYPES:
            raise ValueError(
                f"Unknown proxy event type: {event.event_type!r}. "
                f"Known types: {sorted(_KNOWN_EVENT_TYPES)}"
            )

        p = event.payload
        records_before = len(self._refinement_records)
        state_delta: dict = {}

        # --- Initialisation events: delegate to inner proxy ---
        if event.event_type in _INIT_EVENT_TYPES:
            # generate_candidate_bundles also updates adapter's bundle list.
            response = self.proxy.handle_event(event)
            if event.event_type == GENERATE_CANDIDATE_BUNDLES:
                self.candidate_bundles = response.payload.get(
                    "candidate_bundles", self.candidate_bundles
                )
            self._log_event(event, response)
            return response

        # --- Mechanism events ---
        if event.event_type == SUBMIT_BID:
            bid = self.current_bid()
            state_delta = {"atom_count": len(bid.atoms)}
            response = ProxyResponse(
                response_type="xor_bid",
                payload={"bid": bid},
                state_delta=state_delta,
            )

        elif event.event_type == SEALED_FEEDBACK:
            el_event = p.get("elicitation_event")
            if el_event is not None:
                self.refine(el_event)
            new_records = self._refinement_records[records_before:]
            state_delta = {"refinements": len(new_records)}
            response = ProxyResponse(
                response_type="refinement",
                records=list(new_records),
                state_delta=state_delta,
            )

        elif event.event_type in (CLOCK_PRICES, CLOCK_NEAR_TIE,
                                   CLOCK_NEAR_ZERO_SURPLUS, CLOCK_DEMAND_CHANGED):
            if event.event_type == CLOCK_PRICES:
                prices = p["prices"]
                round_idx = event.round_idx or 0
                top_k = p.get("top_k", 1)
                demand = self.demand_at_prices(prices, round_idx, top_k)
                state_delta = {
                    "primary_bundle": sorted(demand.primary_bundle) if demand.primary_bundle else None
                }
                response = ProxyResponse(
                    response_type="clock_demand",
                    payload={"demand_response": demand},
                    state_delta=state_delta,
                )
            else:
                # Near-tie / near-zero / demand-changed: treated as refinement signals.
                el_event = p.get("elicitation_event")
                if el_event is not None:
                    self.refine(el_event)
                new_records = self._refinement_records[records_before:]
                state_delta = {"refinements": len(new_records)}
                response = ProxyResponse(
                    response_type="refinement",
                    records=list(new_records),
                    state_delta=state_delta,
                )

        else:
            raise ValueError(f"Unhandled event type: {event.event_type!r}")

        self._log_event(event, response)
        return response

    def _log_event(self, event: ProxyElicitationEvent, response: ProxyResponse) -> None:
        """Append a lightweight log entry to the inner proxy's event_log."""
        self.proxy.event_log.append(ProxyEventLogEntry(
            event_type=event.event_type,
            bidder_id=self.bidder_id,
            mechanism=event.mechanism,
            round_idx=event.round_idx,
            response_type=response.response_type,
            records_count=len(response.records),
            state_delta=response.state_delta,
        ))

    def stats(self) -> ProxyStats:
        return self._stats

    def refinement_records(self) -> list[RefinementRecord]:
        return list(self._refinement_records)
