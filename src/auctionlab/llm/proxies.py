"""LLM-backed proxies that infer reusable XOR bids from bundle queries."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

from auctionlab.auction_types import Bundle, CecaBidderDiagnostic, Item
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.instances.base import CecaStepResponse, DemandResponse, demand_rank_key
from auctionlab.llm.bundles import bundle_sort_key
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.interest_map import (
    derive_interest_map,
    generate_candidate_bundles_from_interest_map,
)
from auctionlab.llm.knowledge import KnowledgeBase, TranscriptKnowledgeBase
from auctionlab.llm.logging import LlmCallRecord, current_timestamp
from auctionlab.llm.parsing import parse_proxy_question_response
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.prompts import build_initial_proxy_question_prompt
from auctionlab.llm.provisional_valuations import generate_provisional_valuations
from auctionlab.llm.schemas import LlmInterestMap
from auctionlab.proxies.base import ElicitationEvent, ProxyStats, RefinementRecord
from auctionlab.proxies.elicitation import candidate_refinements
from auctionlab.proxies.events import (
    CECA_SATISFACTION,
    CECA_DEMAND_COUNTEREXAMPLE,
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


@dataclass(frozen=True)
class CecaTrimRecord:
    """One item's trimming decision during atomic trimming."""

    item: str
    reduced_bundle: "Bundle"
    raw_value: float
    removed: bool


@dataclass(frozen=True)
class CecaTrimResult:
    """Result of one atomic trimming step."""

    raw_bundle: "Bundle"
    trimmed_bundle: "Bundle"
    raw_demanded_value: float
    trim_value_queries: int
    trim_items_removed: int
    trim_trace: tuple["CecaTrimRecord", ...]


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
    """

    bidder_id: str
    person: LlmPersonSimulator
    epsilon: float = 0.75
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
    ceca_atomic_trimming: bool = True
    ceca_trim_value_tolerance: float = 0.0
    _last_trim_result: "CecaTrimResult | None" = field(
        default=None, init=False, repr=False
    )
    # Persists discounted values for sub-bundles queried during atomic trimming
    # so repeated trim attempts on the same bundle (across CECA rounds) do not
    # re-issue value queries.  Keyed by frozenset bundle.
    _trim_value_cache: dict = field(
        default_factory=dict, init=False, repr=False
    )
    event_log: list[ProxyEventLogEntry] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon <= 1.0:
            raise ValueError("epsilon must be in (0, 1]")
        if self.knowledge_base is None:
            # Default: share nl_transcript so both stay in sync.
            self.knowledge_base = TranscriptKnowledgeBase(self.nl_transcript)

    def ask_initial_question(
        self,
        question_client: LlmClient | None = None,
    ) -> None:
        """Ask the person an initial open-ended preference question (ωnvd).

        The question/answer pair is recorded in ``nl_transcript`` and used as
        context for all later value inference. No-op if already asked. The
        question is generated without the person's preference seed: the
        proxy is meeting this person for the first time and must not have
        private knowledge of their actual values.
        """
        if self.knowledge_base:
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
                        model=self.person.model_name,
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
            model_name=self.person.model_name,
        )
        self.transcript.append(
            TranscriptEntry(
                kind="interest_map",
                content=(
                    f"interested={sorted(self.interest_map.interested_items)}; "
                    f"excluded={sorted(self.interest_map.excluded_items)}; "
                    f"complements={[sorted(g) for g in self.interest_map.complementary_groups]}; "
                    f"substitutes={[sorted(g) for g in self.interest_map.substitute_groups]}"
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
    ) -> dict[Bundle, float]:
        """Generate provisional values and pre-populate the cached XOR bid.

        Calls the LLM once with all ``candidate_bundles`` and stores the
        resulting values via :meth:`set_provisional_bid`. The optional
        ``interest_map`` enriches the prompt; the function is fully usable
        without it. Returns the raw value map (before epsilon discounting).
        Deliberately does not pass the person's preference seed: estimates
        are grounded only in the NL question/answer actually exchanged, not
        private knowledge of the person's true values.
        """
        if not self.nl_transcript:
            raise RuntimeError(
                "No NL question/answer recorded; call ask_initial_question() first."
            )
        nl_question, nl_answer = self.nl_transcript[-1]
        pv_client = client or self.person.client

        t0 = time.perf_counter()
        print(f"  {self.bidder_id:<12}  pv", end="", flush=True)

        raw_values = generate_provisional_valuations(
            client=pv_client,
            scenario_description=self.person.scenario_description,
            item_descriptions=self.person.item_descriptions,
            nl_question=nl_question,
            nl_answer=nl_answer,
            candidate_bundles=candidate_bundles,
            interest_map=interest_map,
            logger=self.person.logger,
            bidder_id=self.bidder_id,
            model_name=self.person.model_name,
            # max_bundles=None → auto-derived from client.max_tokens inside
        )

        latency = time.perf_counter() - t0
        print(
            f"  →  {len(raw_values)} values  ({latency:.1f}s)",
            flush=True,
        )

        self._provisional_raw_values = raw_values
        discounted = {
            bundle: value * self.epsilon if discount_inferred else value
            for bundle, value in raw_values.items()
        }
        self.set_provisional_bid(discounted, discount_inferred=discount_inferred)
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
                        f"{[sorted(g) for g in interest_map.substitute_groups]}"
                    ),
                )
            )

        if provisional_raw_values is not None:
            self._provisional_raw_values = provisional_raw_values
            discounted = {
                bundle: value * self.epsilon if discount_inferred else value
                for bundle, value in provisional_raw_values.items()
            }
            self.set_provisional_bid(discounted, discount_inferred=discount_inferred)

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
            self.ask_initial_question(p.get("client"))
            nl_q, nl_a = self.nl_transcript[-1] if self.nl_transcript else ("", "")
            state_delta = {"nl_transcript_length": len(self.nl_transcript)}
            response = ProxyResponse(
                response_type="natural_language_answer",
                payload={"question": nl_q, "answer": nl_a},
                state_delta=state_delta,
            )

        elif event.event_type == INFER_INTEREST_MAP:
            im = self.build_interest_map(p.get("client"))
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
            )
            state_delta = {"provisional_value_count": len(raw_values)}
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

            reported_value = value * self.epsilon if discount_inferred else value
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
                if self._cached_discount_inferred:
                    raw_value /= self.epsilon
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

        reported_value = (
            value * self.epsilon
            if self._cached_discount_inferred
            else value
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

    def _prune_demanded_bundle(
        self,
        bundle: Bundle,
        demanded_value: float,
    ) -> "CecaTrimResult":
        """CECA atomic trimming (Huang et al. DNF/proper-learning update step).

        For each item (processed against the progressively-shrinking bundle,
        not the original), value-queries the bundle without it; if removing
        that item leaves value within ``ceca_trim_value_tolerance`` of
        ``demanded_value`` (the original, fixed comparison point), the item
        is dropped permanently. Costs up to one extra value query per item in
        ``bundle``, tracked via ``pruning_query_count``.

        Returns a :class:`CecaTrimResult` with full diagnostics including the
        trimmed atom, number of queries issued, items removed, and a per-item
        trace.
        """
        pruned = bundle
        transcript_context = self.knowledge_base.context_for_prompt() or None
        trim_queries = 0
        trace: list[CecaTrimRecord] = []

        for item in sorted(bundle):
            if len(pruned) <= 1 or item not in pruned:
                continue
            candidate = pruned - {item}
            if not candidate:
                continue

            # 1) Check the proxy's cached bid atoms.
            cached = next(
                (a.value for a in self._cached_bid.atoms if a.bundle == candidate),
                None,
            )
            if cached is not None:
                value_without_item = cached
                raw_value_for_trace = cached
            elif candidate in self._trim_value_cache:
                # 2) Hit the cross-round trim cache — avoids redundant VQs.
                value_without_item = self._trim_value_cache[candidate]
                raw_value_for_trace = value_without_item
            else:
                raw_value = self.person.value_query(
                    candidate,
                    transcript_context=transcript_context,
                )
                self.pruning_query_count += 1
                trim_queries += 1
                value_without_item = (
                    raw_value * self.epsilon
                    if self._cached_discount_inferred
                    else raw_value
                )
                raw_value_for_trace = value_without_item
                self._trim_value_cache[candidate] = value_without_item
                self.transcript.append(
                    TranscriptEntry(
                        kind="ceca_trim_value_query",
                        content=(
                            f"bundle={sorted(candidate)}; "
                            f"value={value_without_item}; "
                            f"demanded_value={demanded_value}"
                        ),
                    )
                )
            removed = abs(value_without_item - demanded_value) <= self.ceca_trim_value_tolerance
            trace.append(
                CecaTrimRecord(
                    item=item,
                    reduced_bundle=candidate,
                    raw_value=raw_value_for_trace,
                    removed=removed,
                )
            )
            if removed:
                pruned = candidate

        return CecaTrimResult(
            raw_bundle=bundle,
            trimmed_bundle=pruned,
            raw_demanded_value=demanded_value,
            trim_value_queries=trim_queries,
            trim_items_removed=len(bundle) - len(pruned),
            trim_trace=tuple(trace),
        )

    def ceca_step(
        self,
        prices: Callable[[Bundle], float],
        current_bundle: Bundle,
        round_idx: int = 0,
    ) -> CecaStepResponse:
        """Respond to a CECA personalized-price step.

        First checks the proxy's own cached belief for a no-LLM-call
        shortcut: if ``current_bundle`` already maximizes XOR-induced
        surplus among cached atoms, no query is needed.  Ties are broken
        in favour of the status-quo ``current_bundle``.  Otherwise asks the
        person directly -- the cached belief can be stale -- and if the
        person also reports satisfaction, trusts them with no atom update.
        If genuinely unsatisfied, revalues and prunes the demanded bundle,
        upserts it into the cached bid (inheriting monotonicity repair via
        :meth:`_revalue_and_upsert_atom`), and reports it to the mechanism.
        """
        if self._cached_bid is None:
            raise RuntimeError(
                "No cached XOR bid is available; call infer_cached_xor_bid first"
            )

        current_bundle = frozenset(current_bundle)

        def rank_key(bundle: Bundle, value: float) -> tuple:
            surplus = value - prices(bundle)
            return (-surplus, 0 if bundle == current_bundle else 1, tuple(sorted(bundle)), -value)

        current_value = next(
            (a.value for a in self._cached_bid.atoms if a.bundle == current_bundle),
            0.0,
        )
        alloc_price = prices(current_bundle)
        alloc_utility = current_value - alloc_price

        best_bundle = current_bundle
        best_key = rank_key(current_bundle, current_value)

        for atom in self._cached_bid.atoms:
            key = rank_key(atom.bundle, atom.value)
            if key < best_key:
                best_key = key
                best_bundle = atom.bundle

        if best_bundle == current_bundle:
            diag = CecaBidderDiagnostic(
                allocated_bundle=current_bundle,
                allocated_lindahl_price=alloc_price,
                allocated_manifest_value=current_value,
                allocated_utility=alloc_utility,
                best_bundle=None,
                best_value=None,
                best_price=None,
                best_utility=None,
                utility_gap=None,
                satisfied=True,
            )
            return CecaStepResponse(satisfied=True, demanded_bundle=None, value=None, diagnostic=diag)

        manifest_prices = {atom.bundle: atom.value for atom in self._cached_bid.atoms}
        response = self.person.ceca_demand_query(current_bundle, manifest_prices)

        if response.satisfied:
            diag = CecaBidderDiagnostic(
                allocated_bundle=current_bundle,
                allocated_lindahl_price=alloc_price,
                allocated_manifest_value=current_value,
                allocated_utility=alloc_utility,
                best_bundle=None,
                best_value=None,
                best_price=None,
                best_utility=None,
                utility_gap=None,
                satisfied=True,
            )
            return CecaStepResponse(satisfied=True, demanded_bundle=None, value=None, diagnostic=diag)

        preferred_bundle = (
            frozenset(
                item
                for item in response.preferred_bundle
                if item in self.person.item_descriptions
            )
            if response.preferred_bundle is not None
            else None
        )

        if not preferred_bundle or preferred_bundle == current_bundle:
            preferred_bundle = best_bundle

        demanded_value = self._revalue_and_upsert_atom(
            preferred_bundle,
            "ceca_demand_query",
            use_anchor_values=True,
            transcript_kind="ceca_demand_query",
        )

        if self.ceca_atomic_trimming:
            trim_result = self._prune_demanded_bundle(preferred_bundle, demanded_value)
            self._last_trim_result = trim_result
            pruned_bundle = trim_result.trimmed_bundle
        else:
            self._last_trim_result = None
            pruned_bundle = preferred_bundle

        if pruned_bundle != preferred_bundle:
            # Reuse the value already queried for the unpruned bundle --
            # pruning only ever drops items that didn't raise the value, so
            # re-querying the smaller bundle would be redundant, matching
            # alpha-main's heuristic (it stores the pruned bundle at the
            # value found for the unpruned one).
            self._cached_bid.atoms = [
                atom
                for atom in self._cached_bid.atoms
                if atom.bundle != preferred_bundle
            ]
            replacement = XorAtomicBid(bundle=pruned_bundle, value=demanded_value)
            for idx, atom in enumerate(self._cached_bid.atoms):
                if atom.bundle == pruned_bundle:
                    self._cached_bid.atoms[idx] = replacement
                    break
            else:
                self._cached_bid.atoms.append(replacement)
            self._enforce_subset_superset_monotonicity()
            demanded_value = next(
                atom.value
                for atom in self._cached_bid.atoms
                if atom.bundle == pruned_bundle
            )

        best_price = prices(pruned_bundle)
        best_utility = demanded_value - best_price
        diag = CecaBidderDiagnostic(
            allocated_bundle=current_bundle,
            allocated_lindahl_price=alloc_price,
            allocated_manifest_value=current_value,
            allocated_utility=alloc_utility,
            best_bundle=pruned_bundle,
            best_value=demanded_value,
            best_price=best_price,
            best_utility=best_utility,
            utility_gap=best_utility - alloc_utility,
            satisfied=False,
        )
        return CecaStepResponse(
            satisfied=False, demanded_bundle=pruned_bundle, value=demanded_value, diagnostic=diag
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
    ceca_atomic_trimming: bool = True
    ceca_trim_value_tolerance: float = 0.0

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

    def ceca_step(
        self,
        prices: Callable[[Bundle], float],
        current_bundle: Bundle,
        round_idx: int = 0,
    ) -> CecaStepResponse:
        self.current_bid()
        self._stats.demand_queries += 1
        current_bundle = frozenset(current_bundle)
        lindahl_price = prices(current_bundle)

        # Snapshot pre-step atom values so RefinementRecord can report old_value.
        cached_before = self.proxy._cached_bid
        old_values: dict[Bundle, float] = (
            {a.bundle: a.value for a in cached_before.atoms}
            if cached_before is not None
            else {}
        )

        response = self.proxy.ceca_step(prices, current_bundle, round_idx)

        if not response.satisfied and response.demanded_bundle is not None:
            demanded = response.demanded_bundle
            new_cached = self.proxy._cached_bid
            new_value = next(
                (a.value for a in new_cached.atoms if a.bundle == demanded),
                response.value,
            ) if new_cached is not None else response.value

            self._refinement_records.append(
                RefinementRecord(
                    bidder_id=self.bidder_id,
                    mechanism="ceca",
                    event_type="unsatisfied_demand",
                    round_idx=round_idx,
                    bundle=demanded,
                    old_value=old_values.get(demanded),
                    new_value=new_value,
                    reason=f"lindahl_price={lindahl_price:.2f}",
                    query_text=self.proxy._last_refinement_query_text,
                    response_summary=self.proxy._last_refinement_response_summary,
                )
            )
            self._stats.refinement_queries = self.proxy.refinement_query_count

        return response

    @property
    def pruning_query_count(self) -> int:
        return self.proxy.pruning_query_count

    @property
    def last_trim_result(self) -> "CecaTrimResult | None":
        return getattr(self.proxy, '_last_trim_result', None)

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
        ``sealed_feedback``, ``clock_prices``, ``ceca_satisfaction``) are
        handled directly by the adapter's existing methods so that stats
        and refinement records are updated correctly.

        Raises :exc:`ValueError` for unknown event types.

        Old public methods (``submit_bid``, ``demand_at_prices``,
        ``ceca_step``, ``refine``) remain unchanged and fully functional.
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

        elif event.event_type in (CECA_SATISFACTION, CECA_DEMAND_COUNTEREXAMPLE):
            prices_fn = p["prices"]
            current_bundle = p["current_bundle"]
            round_idx = event.round_idx or 0
            ceca_resp = self.ceca_step(prices_fn, current_bundle, round_idx)
            new_records = self._refinement_records[records_before:]
            state_delta = {
                "satisfied": ceca_resp.satisfied,
                "demanded_bundle": (
                    sorted(ceca_resp.demanded_bundle)
                    if ceca_resp.demanded_bundle else None
                ),
            }
            response = ProxyResponse(
                response_type="ceca_step",
                payload={"ceca_response": ceca_resp},
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
