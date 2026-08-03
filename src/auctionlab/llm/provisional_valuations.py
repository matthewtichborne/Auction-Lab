"""Bulk provisional valuation from a single LLM call over the NL Q/A pair."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

from auctionlab.auction_types import Bundle, Item
from auctionlab.llm.cache import bundle_set_hash, call_client
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.logging import LlmCallLogger, LlmCallRecord, current_timestamp
from auctionlab.llm.parsing import parse_provisional_valuations_response
from auctionlab.llm.prompts import build_provisional_valuation_prompt
from auctionlab.llm.schemas import LlmInterestMap


_TOKENS_PER_BUNDLE_ESTIMATE = 8
"""Estimated output tokens consumed per bundle entry in the PV JSON response.

Used only to size the *informational* warning below -- it never truncates
the candidate list on its own. The compact positional response contains only
one number per bundle plus JSON punctuation; the prompt already fixes bundle
identity and order.
"""


@dataclass(frozen=True)
class PvCandidateBundleStats:
    """Bookkeeping for how many candidate bundles reached the PV call.

    Logged verbatim (as a flat dict) alongside every
    ``proxy_provisional_valuations`` call record, and also available to
    callers (e.g. :class:`~auctionlab.llm.proxies.LlmInferredXorProxy`) that
    want to report or CSV-export it without re-deriving the numbers.
    """

    candidate_bundles_generated: int
    candidate_bundles_sent_to_pv: int
    candidate_bundles_truncated: bool
    candidate_truncation_reason: str | None
    max_candidate_bundles: int | None

    def as_dict(self) -> dict:
        return {
            "candidate_bundles_generated": self.candidate_bundles_generated,
            "candidate_bundles_sent_to_pv": self.candidate_bundles_sent_to_pv,
            "candidate_bundles_truncated": self.candidate_bundles_truncated,
            "candidate_truncation_reason": self.candidate_truncation_reason,
            "max_candidate_bundles": self.max_candidate_bundles,
        }


@dataclass(frozen=True)
class PvChunkStats:
    """Bookkeeping for how a bidder's provisional valuations were chunked.

    Logged alongside every ``proxy_provisional_valuations`` call record made
    by :func:`generate_provisional_valuations_chunked`, and available to
    callers (e.g. ``examples/run_live_llm_curated_batch.py``'s pv candidate
    bundle stats CSV / run summary) that want to report chunking activity.
    ``pv_chunks`` is the number of PV LLM calls actually made for this
    bidder: 1 whenever chunking wasn't used or wasn't needed (candidate
    count at or below ``pv_chunk_size``, or ``pv_chunk_size`` unset/0),
    matching the pre-chunking behaviour exactly.
    """

    pv_chunk_size: int | None
    pv_chunks: int
    candidate_count: int
    per_chunk_bundle_counts: tuple[int, ...]
    chunking_used: bool

    def as_dict(self) -> dict:
        return {
            "pv_chunk_size": self.pv_chunk_size,
            "pv_chunks": self.pv_chunks,
            "pv_candidate_count": self.candidate_count,
            "pv_per_chunk_bundle_counts": list(self.per_chunk_bundle_counts),
            "pv_chunking_used": self.chunking_used,
        }


def chunk_candidate_bundles(
    candidate_bundles: list[Bundle],
    chunk_size: int | None,
) -> list[list[Bundle]]:
    """Deterministically split ``candidate_bundles`` into chunks of at most
    ``chunk_size``, preserving input order and bundle identity.

    Returns a single chunk (the whole list, unchanged) when ``chunk_size``
    is ``None``, ``0``, or negative, or when the list already fits within
    one chunk -- this is what makes the single-PV-call path and the chunked
    path identical whenever chunking isn't actually needed. An empty
    ``candidate_bundles`` returns an empty list of chunks (no PV call at
    all, matching :func:`generate_provisional_valuations`'s early return).
    """
    if not candidate_bundles:
        return []
    if not chunk_size or chunk_size <= 0 or len(candidate_bundles) <= chunk_size:
        return [list(candidate_bundles)]
    return [
        list(candidate_bundles[i : i + chunk_size])
        for i in range(0, len(candidate_bundles), chunk_size)
    ]


def _estimated_token_budget_capacity(max_tokens: int) -> int:
    """How many bundles a conservative reading of ``max_tokens`` could fit.

    Advisory only: used to decide whether to *warn* that the response might
    get cut off by the model's own output limit. Never used to truncate the
    candidate list -- that only ever happens via an explicit
    ``max_candidate_bundles``/``max_bundles`` cap.
    """
    # Reserve ~200 tokens for the reasoning field and JSON envelope.
    usable = max(max_tokens - 200, 1)
    return max(usable // _TOKENS_PER_BUNDLE_ESTIMATE, 1)


def compute_pv_candidate_bundle_stats(
    candidate_bundles: list[Bundle],
    max_bundles: int | None,
) -> PvCandidateBundleStats:
    """Pure truncation decision, shared by the PV call and its callers.

    Truncation happens **only** when ``max_bundles`` is explicitly set and is
    smaller than the generated count -- never automatically. This is the
    single source of truth for that decision so
    :meth:`LlmInferredXorProxy.build_provisional_valuations` can record the
    same stats the PV call itself logs, without re-implementing the logic.
    """
    generated_count = len(candidate_bundles)

    if max_bundles is not None and generated_count > max_bundles:
        return PvCandidateBundleStats(
            candidate_bundles_generated=generated_count,
            candidate_bundles_sent_to_pv=max_bundles,
            candidate_bundles_truncated=True,
            candidate_truncation_reason=(
                f"explicit max_candidate_bundles={max_bundles} < "
                f"generated count {generated_count}"
            ),
            max_candidate_bundles=max_bundles,
        )

    return PvCandidateBundleStats(
        candidate_bundles_generated=generated_count,
        candidate_bundles_sent_to_pv=generated_count,
        candidate_bundles_truncated=False,
        candidate_truncation_reason=None,
        max_candidate_bundles=max_bundles,
    )


def _resolve_candidate_bundles_for_pv(
    candidate_bundles: list[Bundle],
    max_bundles: int | None,
    *,
    client: LlmClient,
    label: str,
) -> tuple[list[Bundle], PvCandidateBundleStats]:
    """Apply the explicit ``max_bundles`` truncation decision and emit the
    informational token-budget warning -- the single source of truth shared
    by :func:`generate_provisional_valuations` and
    :func:`generate_provisional_valuations_chunked` so both apply identical
    truncation/warning semantics.
    """
    stats = compute_pv_candidate_bundle_stats(candidate_bundles, max_bundles)

    if stats.candidate_bundles_truncated:
        warnings.warn(
            f"PV for {label}: truncating {stats.candidate_bundles_generated} "
            f"candidate bundles to the explicit --max-candidate-bundles cap of "
            f"{stats.candidate_bundles_sent_to_pv} (complement-group bundles are "
            f"prioritised by candidate_bundles_from_interest_map).",
            UserWarning,
            stacklevel=3,
        )
        return candidate_bundles[: stats.candidate_bundles_sent_to_pv], stats

    client_max = getattr(client, "max_tokens", None)
    if client_max:
        estimated_capacity = _estimated_token_budget_capacity(client_max)
        if stats.candidate_bundles_generated > estimated_capacity:
            if max_bundles is None:
                reason = "no truncation applied because --max-candidate-bundles was not set"
                suggestion = (
                    "Consider raising --pv-max-tokens, or pass --max-candidate-bundles "
                    "to cap the list explicitly and reproducibly."
                )
            else:
                reason = (
                    f"no truncation applied because the explicit "
                    f"--max-candidate-bundles={max_bundles} was not exceeded"
                )
                suggestion = "Consider raising --pv-max-tokens."
            warnings.warn(
                f"PV for {label}: candidate bundle count is "
                f"{stats.candidate_bundles_generated}; {reason}. This exceeds a "
                f"conservative estimate of what fits in the model's max_tokens "
                f"budget (~{estimated_capacity} bundles), so the model's own "
                f"response may be cut off or fail to parse. {suggestion}",
                UserWarning,
                stacklevel=3,
            )

    return candidate_bundles, stats


def generate_provisional_valuations(
    *,
    client: LlmClient,
    scenario_description: str,
    item_descriptions: dict[Item, str],
    nl_question: str,
    nl_answer: str,
    candidate_bundles: list[Bundle],
    interest_map: LlmInterestMap | None = None,
    logger: LlmCallLogger | None = None,
    bidder_id: str | None = None,
    model_name: str | None = None,
    max_bundles: int | None = None,
    scenario_id: str | None = None,
    max_parse_retries: int = 0,
    strict_missing: bool = False,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> dict[Bundle, float]:
    """Call the LLM once to estimate values for all ``candidate_bundles``.

    Returns a mapping from every non-empty bundle in ``candidate_bundles``
    to a non-negative estimated value. By default (``strict_missing=False``)
    bundles absent from the LLM response default to 0.0. The optional
    ``interest_map`` enriches the prompt with complement/substitute
    structure and a budget hint; the function is fully usable without it.
    Deliberately does not take the person's preference seed -- estimates are
    grounded only in what ``nl_question``/``nl_answer`` actually revealed,
    not private knowledge of the person's true values.

    ``max_bundles``, when set, deterministically truncates
    ``candidate_bundles`` to the first ``max_bundles`` entries before
    building the prompt. Callers should ensure the highest-priority bundles
    (complement groups, singletons) appear first, which
    :meth:`LlmInferredXorProxy.candidate_bundles_from_interest_map` already
    guarantees.

    When ``max_bundles`` is ``None`` (the default), **no truncation is
    applied** -- every candidate bundle is sent to the LLM, regardless of the
    client's ``max_tokens`` setting. If the candidate count looks large
    relative to a conservative estimate of what ``max_tokens`` output tokens
    can hold, an informational warning is raised suggesting the caller raise
    ``--pv-max-tokens`` or pass ``--max-candidate-bundles`` explicitly -- the
    list itself is never silently reduced. Candidate-bundle limits are the
    caller's decision to make explicitly and reproducibly, not something this
    function infers on their behalf.

    ``max_parse_retries`` retries a malformed/unparseable response (mirroring
    :meth:`~auctionlab.llm.person_simulator.LlmPersonSimulator.value_query`'s
    retry convention) -- a call-level exception (network/timeout) is never
    retried here, only a parse failure. ``strict_missing``/``chunk_index``/
    ``chunk_count`` are consumed by
    :func:`generate_provisional_valuations_chunked`'s per-chunk calls; direct
    callers normally leave them at their defaults.
    """
    if not candidate_bundles:
        return {}

    label = bidder_id or "?"
    candidate_bundles, stats = _resolve_candidate_bundles_for_pv(
        candidate_bundles, max_bundles, client=client, label=label
    )

    prompt = build_provisional_valuation_prompt(
        scenario_description=scenario_description,
        item_descriptions=item_descriptions,
        nl_question=nl_question,
        nl_answer=nl_answer,
        candidate_bundles=candidate_bundles,
        interest_map=interest_map,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
    )
    bundle_key = bundle_set_hash(candidate_bundles)

    for attempt in range(1, max_parse_retries + 2):
        started = time.perf_counter()
        try:
            raw = call_client(
                client,
                prompt,
                call_type="provisional_valuations",
                scenario_id=scenario_id,
                bidder_id=bidder_id,
                bundle_key=bundle_key,
                extra_key_fields=(
                    {"parse_repair_attempt": attempt}
                    if attempt > 1
                    else None
                ),
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            if logger is not None:
                logger.log(
                    LlmCallRecord(
                        timestamp=current_timestamp(),
                        bidder_id=bidder_id,
                        prompt_type="proxy_provisional_valuations",
                        prompt=prompt,
                        raw_response=None,
                        parsed_response=None,
                        success=False,
                        error=str(exc),
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
                    )
                )
            raise

        latency = time.perf_counter() - started

        try:
            result = parse_provisional_valuations_response(
                raw,
                candidate_bundles,
                strict_missing=strict_missing,
                bidder_id=bidder_id,
                chunk_index=chunk_index,
            )
        except ValueError as exc:
            # Raw-response caches sit below parsing. Never retain a truncated
            # or malformed response: otherwise every parse retry (and every
            # resumed preparation run) would replay the same invalid text.
            invalidate = getattr(client, "invalidate_last", None)
            if callable(invalidate):
                invalidate()
            if logger is not None:
                logger.log(
                    LlmCallRecord(
                        timestamp=current_timestamp(),
                        bidder_id=bidder_id,
                        prompt_type="proxy_provisional_valuations",
                        prompt=prompt,
                        raw_response=raw,
                        parsed_response=None,
                        success=False,
                        error=str(exc),
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
                    )
                )
            if attempt > max_parse_retries:
                raise
            continue

        if logger is not None:
            logger.log(
                LlmCallRecord(
                    timestamp=current_timestamp(),
                    bidder_id=bidder_id,
                    prompt_type="proxy_provisional_valuations",
                    prompt=prompt,
                    raw_response=raw,
                    parsed_response={
                        "n_bundles": len(result),
                        "valuations": {
                            str(sorted(b)): v for b, v in result.items()
                        },
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        **stats.as_dict(),
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
                )
            )

        return result

    raise RuntimeError("Provisional valuation attempts exhausted unexpectedly")


def generate_provisional_valuations_chunked(
    *,
    client: LlmClient,
    scenario_description: str,
    item_descriptions: dict[Item, str],
    nl_question: str,
    nl_answer: str,
    candidate_bundles: list[Bundle],
    interest_map: LlmInterestMap | None = None,
    logger: LlmCallLogger | None = None,
    bidder_id: str | None = None,
    model_name: str | None = None,
    max_bundles: int | None = None,
    scenario_id: str | None = None,
    pv_chunk_size: int | None = None,
    max_parse_retries: int = 0,
) -> tuple[dict[Bundle, float], PvChunkStats]:
    """Generate provisional valuations for one bidder, in deterministic chunks.

    Preserves the semantics of one bulk provisional-valuation call while
    reducing parse/truncation risk on large candidate supports: instead of
    one LLM call over every candidate bundle, ``candidate_bundles`` (after
    applying the same ``max_bundles`` truncation
    :func:`generate_provisional_valuations` would) is split via
    :func:`chunk_candidate_bundles` into chunks of at most ``pv_chunk_size``,
    each valued by its own call to :func:`generate_provisional_valuations`.
    Whenever ``pv_chunk_size`` is a positive int (chunking actually
    requested), each chunk call uses ``strict_missing=True``, so a chunk
    response missing a valuation for one of its own requested bundles raises
    rather than silently defaulting to 0.0; the per-chunk results are then
    merged into one bundle -> value table.

    When ``pv_chunk_size`` is ``None``/``0``, or the (post-truncation)
    candidate count is already at or below it, exactly one chunk is made and
    this reduces to a single call identical to calling
    :func:`generate_provisional_valuations` directly -- this is what keeps
    existing (unchunked) behaviour byte-for-byte unchanged.

    Each chunk gets the same bidder/person/interest-map context but only its
    own slice of candidate bundles; the prompt is annotated with which chunk
    it is (see :func:`~auctionlab.llm.prompts.build_provisional_valuation_prompt`)
    but the model is never asked to reason across chunks. A chunk that fails
    to parse is retried per ``max_parse_retries`` (the same retry behaviour
    :func:`generate_provisional_valuations` already applies); if a chunk
    still fails after retries, the exception propagates to the caller
    unchanged -- callers implement their own PV failure policy (e.g.
    ``examples/run_live_llm_curated_batch.py``'s ``--pv-failure-policy``).

    Raises :exc:`ValueError` if two chunks ever produce a valuation for the
    same bundle (should be impossible given a correct, non-overlapping
    chunker; guarded defensively) or if a chunk is missing a valuation for
    one of its own requested bundles.
    """
    if not candidate_bundles:
        return {}, PvChunkStats(
            pv_chunk_size=pv_chunk_size,
            pv_chunks=0,
            candidate_count=0,
            per_chunk_bundle_counts=(),
            chunking_used=False,
        )

    label = bidder_id or "?"
    bundles_to_value, _stats = _resolve_candidate_bundles_for_pv(
        candidate_bundles, max_bundles, client=client, label=label
    )

    chunks = chunk_candidate_bundles(bundles_to_value, pv_chunk_size)
    chunking_used = len(chunks) > 1
    # Only enforce strict missing-bundle detection when the caller actually
    # opted into chunking (a positive pv_chunk_size) -- this is what keeps
    # pv_chunk_size unset/0 byte-for-byte identical to the pre-chunking
    # lenient (missing -> 0.0) behaviour, even though both paths may reduce
    # to a single chunk. A caller who explicitly set pv_chunk_size wants the
    # stronger merge-safety guarantee regardless of whether the candidate
    # count happened to fit in one chunk.
    chunking_requested = pv_chunk_size is not None and pv_chunk_size > 0

    merged: dict[Bundle, float] = {}
    seen: set[frozenset] = set()
    for idx, chunk in enumerate(chunks):
        chunk_values = generate_provisional_valuations(
            client=client,
            scenario_description=scenario_description,
            item_descriptions=item_descriptions,
            nl_question=nl_question,
            nl_answer=nl_answer,
            candidate_bundles=chunk,
            interest_map=interest_map,
            logger=logger,
            bidder_id=bidder_id,
            model_name=model_name,
            max_bundles=None,  # already truncated above; chunks split what remains
            scenario_id=scenario_id,
            max_parse_retries=max_parse_retries,
            strict_missing=chunking_requested,
            chunk_index=idx if chunking_used else None,
            chunk_count=len(chunks) if chunking_used else None,
        )
        for bundle, value in chunk_values.items():
            key = frozenset(bundle)
            if key in seen:
                bundle_str = "{" + ",".join(sorted(bundle)) + "}"
                raise ValueError(
                    f"PV chunking produced a duplicate valuation for "
                    f"bidder_id={label!r} at chunk_index={idx}: bundle "
                    f"{bundle_str} was already valued by an earlier chunk."
                )
            seen.add(key)
            merged[bundle] = value

    chunk_stats = PvChunkStats(
        pv_chunk_size=pv_chunk_size,
        pv_chunks=len(chunks),
        candidate_count=len(bundles_to_value),
        per_chunk_bundle_counts=tuple(len(c) for c in chunks),
        chunking_used=chunking_used,
    )
    return merged, chunk_stats
