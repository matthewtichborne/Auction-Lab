"""Bulk provisional valuation from a single LLM call over the NL Q/A pair."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

from auctionlab.auction_types import Bundle, Item
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.logging import LlmCallLogger, LlmCallRecord, current_timestamp
from auctionlab.llm.parsing import parse_provisional_valuations_response
from auctionlab.llm.prompts import build_provisional_valuation_prompt
from auctionlab.llm.schemas import LlmInterestMap


_TOKENS_PER_BUNDLE_ESTIMATE = 50
"""Estimated output tokens consumed per bundle entry in the PV JSON response.

Used only to size the *informational* warning below -- it never truncates
the candidate list on its own. Each entry serialises to roughly
``["ITEM_ID", ...]`` + a number + overhead.
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
) -> dict[Bundle, float]:
    """Call the LLM once to estimate values for all ``candidate_bundles``.

    Returns a mapping from every non-empty bundle in ``candidate_bundles``
    to a non-negative estimated value. Bundles absent from the LLM response
    default to 0.0. The optional ``interest_map`` enriches the prompt with
    complement/substitute structure and a budget hint; the function is fully
    usable without it. Deliberately does not take the person's preference
    seed -- estimates are grounded only in what ``nl_question``/``nl_answer``
    actually revealed, not private knowledge of the person's true values.

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
    """
    if not candidate_bundles:
        return {}

    stats = compute_pv_candidate_bundle_stats(candidate_bundles, max_bundles)
    label = bidder_id or "?"

    if stats.candidate_bundles_truncated:
        warnings.warn(
            f"PV for {label}: truncating {stats.candidate_bundles_generated} "
            f"candidate bundles to the explicit --max-candidate-bundles cap of "
            f"{stats.candidate_bundles_sent_to_pv} (complement-group bundles are "
            f"prioritised by candidate_bundles_from_interest_map).",
            UserWarning,
            stacklevel=2,
        )
        candidate_bundles = candidate_bundles[: stats.candidate_bundles_sent_to_pv]
    else:
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
                    stacklevel=2,
                )

    prompt = build_provisional_valuation_prompt(
        scenario_description=scenario_description,
        item_descriptions=item_descriptions,
        nl_question=nl_question,
        nl_answer=nl_answer,
        candidate_bundles=candidate_bundles,
        interest_map=interest_map,
    )
    started = time.perf_counter()
    raw = client.complete(prompt)
    latency = time.perf_counter() - started

    result = parse_provisional_valuations_response(raw, candidate_bundles)

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
                    **stats.as_dict(),
                },
                success=True,
                error=None,
                latency_seconds=latency,
                model=model_name,
                attempt=1,
                input_tokens=getattr(client, "_last_input_tokens", None),
                output_tokens=getattr(client, "_last_output_tokens", None),
                total_tokens=getattr(client, "_last_total_tokens", None),
            )
        )

    return result
