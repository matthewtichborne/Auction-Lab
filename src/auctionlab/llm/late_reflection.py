"""Late-stage reflective natural-language elicitation.

Implements the ``late_reflection`` proxy elicitation event: a targeted
natural-language check-in issued near auction termination, built from
accumulated auction context (initial NL exchange, interest map, current
top bundles, current allocation/demand, recent elicitation events and
refinements). The answer may drive a small, capped number of follow-up
value/demand queries through the *existing* VQ/DQ paths on
:class:`~auctionlab.llm.proxies.LlmInferredXorProxy` -- this module never
revises reported values directly from the natural-language answer.

Like ``near_zero_surplus``/``near_tie``/``demand_changed`` (clock) and
``allocated_bundle``/``lost_interested_bundle`` (sealed), this is one more
elicitation event in the existing event-driven framework: it is off by
default and is triggered/scoped by :class:`LateReflectionConfig`, not baked
into the mechanism runners' default behaviour.

Two independent pieces live here:

1. Pure, mechanism-agnostic building blocks -- :class:`LateReflectionConfig`,
   :func:`build_late_reflection_context`, the allocation-relevance
   selectors, and :func:`run_late_reflection_for_bidder`/
   :func:`run_late_reflection_trigger` -- that never invoke a real auction
   mechanism themselves and are fully testable with a mock LLM client.
2. Mechanism-specific *trigger timing* (when to call
   :func:`run_late_reflection_trigger`), which lives in
   :mod:`auctionlab.experiments.proxy_sealed_runner` (once, before the
   final sealed round) and :mod:`auctionlab.experiments.proxy_clock_runner`
   (once, when total positive excess demand first drops to/below
   ``near_clearing_threshold``) to avoid this module depending on the
   experiments layer.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

from auctionlab.auction_types import Bundle, Item
from auctionlab.bids.xor import XorBid
from auctionlab.instances.base import AuctionInstance
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.logging import LlmCallRecord, current_timestamp
from auctionlab.llm.parsing import (
    LateReflectionParseError,
    filter_late_reflection_bundles,
    parse_late_reflection_response,
    raw_response_excerpt,
)
from auctionlab.llm.prompts import build_late_reflection_prompt
from auctionlab.llm.proxies import LlmInferredXorProxy, TranscriptEntry
from auctionlab.payments.vcg import compute_vcg_payments
from auctionlab.solvers.wdp_ilp import solve_wdp_xor_ilp

_VALID_SCOPES = {"allocation_relevant", "all_bidders", "allocation_marginal"}
_VALID_FOLLOWUPS = {"none", "value_query", "demand_query", "mechanism_default"}

# Default K for sealed's "top losing reported bundles" marginality signal.
# Deliberately hard-coded rather than config/CLI-exposed -- see
# sealed_marginality_scores.
_TOP_LOSING_BIDDERS_FOR_MARGINAL_SCOPE = 3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LateReflectionConfig:
    """Configuration for the ``late_reflection`` elicitation event.

    ``enabled=False`` (the default) means late reflection never fires --
    every trigger call site checks this first. ``followup="mechanism_default"``
    resolves to ``value_query`` for *both* sealed and clock via
    :meth:`resolved_followup`; the resolved (or explicitly configured)
    followup type is what actually fires, independent of what the LLM's own
    ``suggested_followup`` says -- the model's suggestion is only logged
    (``suggested_followup`` vs. ``actual_followup_type`` in
    :class:`LateReflectionRecord`) for later analysis of which target
    categories are most useful.

    Clock previously defaulted ``mechanism_default`` to ``demand_query``
    (the clock already elicits price-conditioned demand, so a late
    demand_query seemed natural); a live 10x10 run showed this mostly just
    confirmed current demand near clearing and didn't improve pricing error
    or the final allocation. Since the reflection question is now always an
    explicit pairwise/marginal comparison (``reflection_mode``), a direct
    value_query over the comparison pair tests it more precisely --
    ``demand_query`` is still fully supported, just no longer the default;
    request it explicitly with ``followup="demand_query"``.
    """

    enabled: bool = False
    scope: str = "allocation_relevant"
    followup: str = "mechanism_default"
    followups_per_bidder: int = 1
    near_clearing_threshold: int = 2
    recent_window_rounds: int = 3
    # A live 10x10 run truncated the reflection response at ~296 output
    # tokens (well under the ordinary --max-tokens=300 value/demand-query
    # budget), which made every row fail to parse -- the pairwise schema's
    # JSON is noticeably larger than a bare bundle_value response. This is
    # threaded to a *separate* client built with this max_tokens value (see
    # examples/run_live_llm_curated_batch.py), not the shared VQ/DQ client.
    max_tokens: int = 1000
    # Only consulted when scope == "allocation_marginal": caps how many
    # top-scoring marginal bidders actually get queried (see
    # rank_and_select_marginal_bidders). ``None`` means no cap -- every
    # positive-score bidder is selected, matching the "None or omitted"
    # convention documented on this field (deliberately *not* the legacy
    # 0-means-unlimited convention some other CLI flags in this project
    # use: ``0`` here means "cap at zero bidders", i.e. select nobody).
    # Ignored entirely for scope in {"allocation_relevant", "all_bidders"}.
    late_reflection_max_bidders: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in _VALID_SCOPES:
            raise ValueError(
                f"scope must be one of {sorted(_VALID_SCOPES)}, got {self.scope!r}"
            )
        if self.followup not in _VALID_FOLLOWUPS:
            raise ValueError(
                f"followup must be one of {sorted(_VALID_FOLLOWUPS)}, "
                f"got {self.followup!r}"
            )
        if self.followups_per_bidder < 0:
            raise ValueError("followups_per_bidder must be non-negative")
        if self.near_clearing_threshold < 0:
            raise ValueError("near_clearing_threshold must be non-negative")
        if self.recent_window_rounds < 0:
            raise ValueError("recent_window_rounds must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if (
            self.late_reflection_max_bidders is not None
            and self.late_reflection_max_bidders < 0
        ):
            raise ValueError("late_reflection_max_bidders must be non-negative or None")

    def resolved_followup(self, mechanism: str) -> str:
        """Resolve ``mechanism_default`` into the mechanism's concrete followup type.

        ``mechanism_default`` now resolves to ``value_query`` for both
        sealed and clock (see class docstring). ``mechanism`` is kept as a
        parameter -- rather than dropping it -- so callers don't need to
        change, and so a future mechanism-specific default can be
        reintroduced without changing every call site. An explicit
        ``demand_query`` request (not ``mechanism_default``) always passes
        through unchanged; sealed auctions have no price concept, so the
        caller is responsible for the documented sealed fallback to
        ``value_query`` when ``prices`` is unavailable -- see
        :func:`run_late_reflection_for_bidder`.
        """
        if self.followup != "mechanism_default":
            return self.followup
        return "value_query"


# ---------------------------------------------------------------------------
# Structured context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LateReflectionContext:
    """Compact, structured per-bidder summary -- never the raw transcript."""

    bidder_id: str
    mechanism: str
    round_idx: int
    initial_nl_summary: str | None = None
    interest_map_summary: str | None = None
    budget_hint: float | None = None
    top_reported_bundles: tuple[tuple[Bundle, float], ...] = ()
    allocated_bundle: Bundle | None = None
    demanded_bundle: Bundle | None = None
    current_prices: dict[Item, float] | None = None
    recent_events: tuple[dict[str, Any], ...] = ()
    recent_refinements: tuple[dict[str, Any], ...] = ()
    best_large_bundle: Bundle | None = None
    near_tie_bundle: Bundle | None = None
    contested_goods: tuple[Item, ...] = ()
    resolved_hints: tuple[str, ...] = ()

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "round_idx": self.round_idx,
            "initial_nl_summary": self.initial_nl_summary,
            "interest_map_summary": self.interest_map_summary,
            "budget_hint": self.budget_hint,
            "top_reported_bundles": [
                (sorted(b), v) for b, v in self.top_reported_bundles
            ],
            "allocated_bundle": (
                sorted(self.allocated_bundle) if self.allocated_bundle else None
            ),
            "demanded_bundle": (
                sorted(self.demanded_bundle)
                if self.demanded_bundle is not None
                else None
            ),
            "current_prices": dict(self.current_prices) if self.current_prices else None,
            "recent_events": list(self.recent_events),
            "recent_refinements": list(self.recent_refinements),
            "best_large_bundle": (
                sorted(self.best_large_bundle) if self.best_large_bundle else None
            ),
            "near_tie_bundle": (
                sorted(self.near_tie_bundle) if self.near_tie_bundle else None
            ),
            "contested_goods": sorted(self.contested_goods),
            "resolved_hints": list(self.resolved_hints),
        }


def build_late_reflection_context(
    *,
    bidder_id: str,
    mechanism: str,
    round_idx: int,
    proxy: LlmInferredXorProxy,
    top_k_bundles: int = 5,
    allocated_bundle: Bundle | None = None,
    demanded_bundle: Bundle | None = None,
    current_prices: dict[Item, float] | None = None,
    recent_events: list[Any] | None = None,
    recent_refinements: list[Any] | None = None,
    contested_goods: set[Item] | None = None,
    near_tie_relative_gap: float = 0.1,
) -> LateReflectionContext:
    """Build the structured context for one bidder's late-reflection call.

    Pure and side-effect-free: reads ``proxy``'s already-accumulated state
    (NL transcript, interest map, cached bid, event log) without issuing any
    LLM call. ``recent_events``/``recent_refinements`` are duck-typed
    (``ElicitationEvent``/``RefinementRecord``-like objects exposing
    ``event_type``/``round_idx``/``bundle`` and ``old_value``/``new_value``/
    ``reason``/``bundle`` respectively) so mechanism-specific callers can
    pass their own trimmed windows without this module depending on them.
    """
    nl_summary: str | None = None
    if proxy.nl_transcript:
        question, answer = proxy.nl_transcript[-1]
        nl_summary = f"Q: {question} A: {answer}"

    interest_map_summary: str | None = None
    budget_hint: float | None = None
    if proxy.interest_map is not None:
        im = proxy.interest_map
        interest_map_summary = (
            f"interested={sorted(im.interested_items)}; "
            f"excluded={sorted(im.excluded_items)}; "
            f"complements={[sorted(g) for g in im.complementary_groups]}; "
            "substitutes="
            f"{[{'items': sorted(g.items), 'mode': g.acquisition_mode} for g in im.substitute_groups]}"
        )
        budget_hint = im.budget_hint

    top_bundles: list[tuple[Bundle, float]] = []
    best_large_bundle: Bundle | None = None
    if proxy._cached_bid is not None:
        atoms_sorted = sorted(
            (a for a in proxy._cached_bid.atoms if a.bundle),
            key=lambda a: a.value,
            reverse=True,
        )
        top_bundles = [(a.bundle, a.value) for a in atoms_sorted[:top_k_bundles]]
        large_atoms = [a for a in atoms_sorted if len(a.bundle) >= 3]
        if large_atoms:
            best_large_bundle = large_atoms[0].bundle

    near_tie_bundle: Bundle | None = None
    if len(top_bundles) >= 2:
        (_first_bundle, first_value), (second_bundle, second_value) = top_bundles[:2]
        if first_value > 0 and abs(first_value - second_value) <= (
            near_tie_relative_gap * first_value
        ):
            near_tie_bundle = second_bundle

    recent_events_ctx = tuple(
        {
            "event_type": getattr(event, "event_type", ""),
            "round_idx": getattr(event, "round_idx", None),
            "bundle": (
                sorted(getattr(event, "bundle", None) or [])
                or None
            ),
        }
        for event in (recent_events or [])
    )
    recent_refinements_ctx = tuple(
        {
            "bundle": sorted(getattr(rec, "bundle", None) or []) or None,
            "old_value": getattr(rec, "old_value", None),
            "new_value": getattr(rec, "new_value", None),
            "reason": getattr(rec, "reason", None),
        }
        for rec in (recent_refinements or [])
    )

    current_bundle = demanded_bundle if demanded_bundle is not None else allocated_bundle

    resolved_hints: list[str] = []
    if near_tie_bundle is None:
        resolved_hints.append(
            "no near-tie/substitute bundle is currently observed -- avoid "
            "asking a substitute/near-tie comparison"
        )
    if (
        best_large_bundle is not None
        and current_bundle is not None
        and best_large_bundle == current_bundle
    ):
        resolved_hints.append(
            f"[{','.join(sorted(best_large_bundle))}] is already the current "
            "bundle -- there is no smaller core subset to compare it "
            "against, so a large-bundle-vs-core-subset question is moot"
        )
    if contested_goods:
        relevant_bundle = best_large_bundle or current_bundle
        if relevant_bundle is not None and not (
            set(contested_goods) & set(relevant_bundle)
        ):
            resolved_hints.append(
                f"none of the currently contested goods {sorted(contested_goods)} "
                "appear in this bidder's current/large bundle -- do not ask "
                "about dropping a good that was never part of it"
            )

    return LateReflectionContext(
        bidder_id=bidder_id,
        mechanism=mechanism,
        round_idx=round_idx,
        initial_nl_summary=nl_summary,
        interest_map_summary=interest_map_summary,
        budget_hint=budget_hint,
        top_reported_bundles=tuple(top_bundles),
        allocated_bundle=allocated_bundle,
        demanded_bundle=demanded_bundle,
        current_prices=dict(current_prices) if current_prices else None,
        recent_events=recent_events_ctx,
        recent_refinements=recent_refinements_ctx,
        best_large_bundle=best_large_bundle,
        near_tie_bundle=near_tie_bundle,
        contested_goods=tuple(sorted(contested_goods)) if contested_goods else (),
        resolved_hints=tuple(resolved_hints),
    )


# ---------------------------------------------------------------------------
# Allocation relevance
# ---------------------------------------------------------------------------

def sealed_allocation_relevant_bidders(events: list[Any]) -> dict[str, str]:
    """Bidders who receive sealed feedback in the pre-final round.

    ``events`` is the pre-final round's provisional-feedback event list
    (``ElicitationEvent``-like objects exposing ``bidder_id``/
    ``event_type``), as already computed by the sealed runner under the
    active ``feedback_rule`` -- this already covers both currently
    provisionally-allocated bidders and losing bidders whose best/
    interested bundle receives feedback, since those are exactly the cases
    that produce an event.
    """
    relevant: dict[str, str] = {}
    for event in events:
        bidder_id = getattr(event, "bidder_id", None)
        if not bidder_id:
            continue
        reason = f"sealed_feedback:{getattr(event, 'event_type', '')}"
        relevant.setdefault(bidder_id, reason)
    return relevant


def clock_allocation_relevant_bidders(
    *,
    bidder_ids: list[str],
    recent_event_bidders: set[str],
    current_demand_by_bidder: dict[str, Bundle | None],
    recently_contested_goods: set[Item],
) -> dict[str, str]:
    """Bidders with a recent local clock event, or demand touching a
    recently (or currently) contested good.
    """
    relevant: dict[str, str] = {}
    for bidder_id in bidder_ids:
        if bidder_id in recent_event_bidders:
            relevant[bidder_id] = "recent_clock_event"
            continue
        demand = current_demand_by_bidder.get(bidder_id)
        if demand and set(demand) & recently_contested_goods:
            relevant[bidder_id] = "contested_good_in_current_demand"
    return relevant


# ---------------------------------------------------------------------------
# Allocation-marginal scoring (scope == "allocation_marginal")
# ---------------------------------------------------------------------------
#
# ``allocation_relevant`` turned out to be too broad under sealed feedback
# rules like ``all_provisional``: almost every bidder receives
# ``allocated_bundle`` or ``lost_interested_bundle`` feedback, so it
# effectively asked *every* bidder a reflection question -- a broad
# late-stage revaluation sweep, not a targeted elicitation event. A live
# 10x10 run with that scope and two follow-up VQs actually *reduced* final
# efficiency (94.1% sealed, 93.0% clock, vs. 97.9%/94.6% with no late
# reflection at all). ``allocation_marginal`` replaces the binary "did this
# bidder get any signal" test with an explicit, additive marginality score,
# then only the top-``late_reflection_max_bidders`` scorers are queried.

@dataclass(frozen=True)
class MarginalityScore:
    """One bidder's allocation-marginality score and the reasons behind it."""

    score: float
    reasons: tuple[str, ...] = ()


def sealed_marginality_scores(
    *,
    bidder_ids: list[str],
    provisional_allocation: dict[str, Bundle],
    previous_allocation: dict[str, Bundle] | None,
    bids_by_bidder: dict[str, XorBid],
    events: list[Any],
    top_losing_bidders_for_marginal_scope: int = _TOP_LOSING_BIDDERS_FOR_MARGINAL_SCOPE,
) -> dict[str, MarginalityScore]:
    """Score every sealed bidder's allocation-marginality.

    Additive scoring rules (a bidder can satisfy several at once):

    - +100 if the bidder currently receives a non-empty provisional
      allocation.
    - +100 if the bidder's allocated bundle changed between the previous and
      current sealed round (``previous_allocation`` may be ``None`` for the
      very first round, in which case this never fires).
    - +80 if the bidder is losing (no current provisional allocation) and
      their best positive-value reported bundle is among the
      ``top_losing_bidders_for_marginal_scope`` highest-valued losing
      bundles across all losing bidders.
    - +40 if the bidder has a large (>=3 items) high-value reported bundle
      that shares at least one good with the provisional allocation (i.e.
      it is contesting an item someone currently holds).
    - +20 if the bidder received sealed feedback in the current/pre-final
      round (the same signal ``allocation_relevant`` alone used to gate on).

    Deliberately does *not* select every ``lost_interested_bundle``
    recipient outright -- a losing bidder only scores unless they are
    genuinely near-winning (top-K) or otherwise high-scoring.
    """
    provisional_allocation = provisional_allocation or {}
    previous_allocation = previous_allocation or {}
    feedback_bidders = sealed_allocation_relevant_bidders(events)

    losing_best: dict[str, tuple[Bundle, float]] = {}
    for bidder_id in bidder_ids:
        if provisional_allocation.get(bidder_id, frozenset()):
            continue
        atom = _best_positive_atom(bids_by_bidder.get(bidder_id))
        if atom is not None:
            losing_best[bidder_id] = (atom.bundle, atom.value)

    losing_ranked = sorted(losing_best, key=lambda b: (-losing_best[b][1], b))
    top_losing_rank_by_bidder = {
        bidder_id: rank
        for rank, bidder_id in enumerate(
            losing_ranked[:top_losing_bidders_for_marginal_scope], start=1
        )
    }

    allocated_goods_union: set[Item] = set()
    for bundle in provisional_allocation.values():
        allocated_goods_union |= set(bundle)

    scores: dict[str, MarginalityScore] = {}
    for bidder_id in bidder_ids:
        score = 0.0
        reasons: list[str] = []

        current_bundle = provisional_allocation.get(bidder_id, frozenset())
        if current_bundle:
            score += 100
            reasons.append("currently_allocated")

        prev_bundle = previous_allocation.get(bidder_id, frozenset())
        if prev_bundle != current_bundle:
            score += 100
            reasons.append("allocation_changed_last_round")

        top_losing_rank = top_losing_rank_by_bidder.get(bidder_id)
        if top_losing_rank is not None:
            score += 80
            reasons.append(f"top_losing_bundle_rank={top_losing_rank}")

        best_large_atom = _best_large_positive_atom(bids_by_bidder.get(bidder_id))
        if best_large_atom is not None and set(best_large_atom.bundle) & allocated_goods_union:
            score += 40
            reasons.append("large_bundle_overlaps_allocation")

        if bidder_id in feedback_bidders:
            score += 20
            reasons.append("received_sealed_feedback")

        scores[bidder_id] = MarginalityScore(score=score, reasons=tuple(reasons))

    return scores


def clock_marginality_scores(
    *,
    bidder_ids: list[str],
    current_demand_by_bidder: dict[str, Bundle | None],
    positive_excess_demand_goods: set[Item],
    contested_goods: set[Item],
    recent_events_by_bidder: dict[str, list[Any]],
    old_rule_relevant_bidders: dict[str, str],
) -> dict[str, MarginalityScore]:
    """Score every clock bidder's allocation-marginality.

    Additive scoring rules:

    - +100 if the bidder's current demand contains a good with currently
      positive excess demand.
    - +80 if the bidder had a ``near_tie`` event in the recent window.
    - +70 if the bidder had a ``near_zero_surplus`` event in the recent
      window.
    - +50 if the bidder had a ``demand_changed`` event in the recent window.
    - +40 if the bidder's current demand is non-empty and overlaps the
      (recently or currently) contested goods.
    - +20 if the bidder was already allocation-relevant under the old
      (binary) ``allocation_relevant`` rule.

    ``recent_events_by_bidder`` should already be windowed to the last
    ``late_reflection_recent_window_rounds`` rounds (including the
    triggering round) -- this function does not re-window, so it naturally
    prefers recency: a stale event outside that window was never passed in
    and so cannot contribute to the score at all.
    """
    scores: dict[str, MarginalityScore] = {}
    for bidder_id in bidder_ids:
        score = 0.0
        reasons: list[str] = []

        demand = current_demand_by_bidder.get(bidder_id)
        if demand and set(demand) & positive_excess_demand_goods:
            score += 100
            reasons.append("current_demand_positive_excess_demand_good")

        event_types = {
            getattr(event, "event_type", "")
            for event in recent_events_by_bidder.get(bidder_id, [])
        }
        if "near_tie" in event_types:
            score += 80
            reasons.append("recent_near_tie")
        if "near_zero_surplus" in event_types:
            score += 70
            reasons.append("recent_near_zero_surplus")
        if "demand_changed" in event_types:
            score += 50
            reasons.append("recent_demand_changed")

        if demand and set(demand) & contested_goods:
            score += 40
            reasons.append("current_demand_contested_good")

        if bidder_id in old_rule_relevant_bidders:
            score += 20
            reasons.append("old_rule_allocation_relevant")

        scores[bidder_id] = MarginalityScore(score=score, reasons=tuple(reasons))

    return scores


def rank_and_select_marginal_bidders(
    scores: dict[str, MarginalityScore],
    max_bidders: int | None,
) -> tuple[list[tuple[str, MarginalityScore, int]], dict[str, str]]:
    """Rank every scored bidder, then select the top positive-scoring ones.

    Ranking is over *every* scored bidder (including zero/negative scores)
    so ``curated_late_reflection_candidates.csv`` can show the full picture
    -- who was considered and why they were or weren't queried. Selection
    (the second return value, in the same ``{bidder_id: reason}`` shape
    ``allocation_relevant``/``all_bidders`` already use) only ever includes
    bidders with a strictly positive score, sorted by descending score then
    ascending ``bidder_id`` (deterministic tie-break), capped at
    ``max_bidders`` (``None`` = no cap). If no bidder has a positive score,
    selection is empty and no late reflection fires.
    """
    ranked_pairs = sorted(scores.items(), key=lambda kv: (-kv[1].score, kv[0]))
    ranked = [
        (bidder_id, score, rank)
        for rank, (bidder_id, score) in enumerate(ranked_pairs, start=1)
    ]

    selected: dict[str, str] = {}
    for bidder_id, score, _rank in ranked:
        if score.score <= 0:
            continue
        if max_bidders is not None and len(selected) >= max_bidders:
            continue
        selected[bidder_id] = "; ".join(score.reasons) or "marginal"

    return ranked, selected


def _best_positive_atom(bid: XorBid | None):
    """Return the highest-value positive atom of ``bid``, if any."""
    if bid is None:
        return None
    best = None
    for atom in bid.atoms:
        if not atom.bundle or atom.value <= 0.0:
            continue
        if best is None or atom.value > best.value:
            best = atom
    return best


def _best_large_positive_atom(bid: XorBid | None):
    """Return the highest-value positive atom of size >= 3, if any.

    Matches :func:`build_late_reflection_context`'s ``best_large_bundle``
    convention: filter to large (>=3 item) atoms *first*, then take the
    best of those -- not "take the overall best atom, then check if it
    happens to be large" (a bidder's single highest-value atom is often a
    small bundle even when they also have a decent large one).
    """
    if bid is None:
        return None
    best = None
    for atom in bid.atoms:
        if not atom.bundle or atom.value <= 0.0 or len(atom.bundle) < 3:
            continue
        if best is None or atom.value > best.value:
            best = atom
    return best


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class LateReflectionRecord:
    """One row of ``curated_late_reflection_records.csv``.

    One :class:`LateReflectionRecord` is produced per (bidder, followup
    bundle) pair actually queried; a bidder with no followup (including
    ``followups_per_bidder=0`` or ``suggested_followup="none"`` with no
    forced override) still gets exactly one record with
    ``followup_bundle=None``/``actual_followup_type="none"`` so the NL
    question/answer is never lost.
    """

    scenario: str
    mechanism: str
    arm: str
    round_idx: int | None
    bidder_id: str
    trigger_reason: str
    scope_rule: str
    allocation_relevant_reason: str
    # Only populated for scope == "allocation_marginal" (None otherwise --
    # every selected bidder here was, by construction, chosen because it
    # scored positively, so marginality_selected is always True when set).
    # Non-selected candidates never get a LateReflectionRecord at all; see
    # LateReflectionCandidateRecord / curated_late_reflection_candidates.csv
    # for the full (selected + non-selected) picture.
    marginality_score: float | None = None
    marginality_rank: int | None = None
    marginality_selected: bool | None = None
    marginality_reasons: str = ""
    question: str = ""
    person_response: str = ""
    parse_success: bool = False
    parse_error_type: str = ""
    raw_reflection_response_excerpt: str = ""
    reflection_mode: str = ""
    reflection_mode_inferred: bool | None = None
    target_type: str = ""
    primary_bundle: Bundle | None = None
    comparison_bundle: Bundle | None = None
    marginal_item: str | None = None
    comparison_pair_available: bool | None = None
    target_bundles: list[Bundle] = field(default_factory=list)
    suggested_followup: str = "none"
    actual_followup_type: str = "none"
    followup_bundle: Bundle | None = None
    followup_bundle_rank: int | None = None
    old_reported_value: float | None = None
    new_reported_value: float | None = None
    true_value: float | None = None
    absolute_correction: float | None = None
    signed_correction: float | None = None
    old_abs_error: float | None = None
    new_abs_error: float | None = None
    old_signed_error: float | None = None
    new_signed_error: float | None = None
    pricing_error_improved: bool | None = None
    pair_old_abs_error_sum: float | None = None
    pair_new_abs_error_sum: float | None = None
    pair_pricing_error_improved: bool | None = None
    pair_old_signed_error_sum: float | None = None
    pair_new_signed_error_sum: float | None = None
    demand_before: Bundle | None = None
    demand_after: Bundle | None = None
    demand_changed: bool | None = None
    allocation_before: dict[str, Bundle] | None = None
    allocation_after: dict[str, Bundle] | None = None
    allocation_changed_after_reflection: bool | None = None
    true_welfare_before: float | None = None
    true_welfare_after: float | None = None
    welfare_delta_after_reflection: float | None = None
    reported_welfare_before: float | None = None
    reported_welfare_after: float | None = None
    reported_welfare_delta_after_reflection: float | None = None
    revenue_before: float | None = None
    revenue_after: float | None = None
    surplus_before: float | None = None
    surplus_after: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cache_hit: bool | None = None
    error_message: str = ""


@dataclass
class LateReflectionCandidateRecord:
    """One row of ``curated_late_reflection_candidates.csv``.

    Written only for ``scope == "allocation_marginal"`` -- one row per
    bidder *considered* for that trigger (not just the ones selected), so a
    reader can see why a bidder was or wasn't queried. Selected bidders also
    get a normal :class:`LateReflectionRecord` (or several, one per
    follow-up bundle); non-selected candidates only appear here.
    """

    scenario: str
    mechanism: str
    arm: str
    round_idx: int | None
    bidder_id: str
    scope_rule: str
    marginality_score: float
    marginality_rank: int
    marginality_selected: bool
    marginality_reasons: str
    current_allocation: Bundle | None = None
    current_demand: Bundle | None = None
    recent_events: str = ""
    best_losing_bundle: Bundle | None = None
    best_losing_bundle_reported_value: float | None = None


@dataclass(frozen=True)
class LateReflectionTriggerResult:
    """Return value of :func:`run_late_reflection_trigger`."""

    records: list[LateReflectionRecord]
    candidates: list[LateReflectionCandidateRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-bidder orchestration
# ---------------------------------------------------------------------------

def _log_late_reflection_call(
    proxy: LlmInferredXorProxy,
    *,
    prompt: str,
    raw_response: str | None,
    parsed: Any | None,
    success: bool,
    error: str | None,
    latency_seconds: float,
    client: LlmClient,
) -> None:
    """Log the reflection question-generation call to ``calls.jsonl``, if a
    logger is configured.

    Prior to this, the reflection call bypassed ``LlmPersonSimulator``'s
    normal ``_log_attempt`` path entirely (it calls the client directly, to
    support a separate ``client_override`` with its own ``max_tokens``), so
    it never appeared in ``calls.jsonl`` at all -- there was no way to see
    the raw model output behind a parse failure. ``prompt_type=
    "proxy_late_reflection"`` distinguishes this call from the ordinary
    person-facing ``value_query``/``demand_query``/``nl_question`` calls (and
    from the other proxy-side ``proxy_nl_gen``/``proxy_interest_map``/
    ``proxy_provisional_valuations`` calls) already logged under that name.
    The follow-up VQ/DQ triggered by a successful reflection reuses the
    existing ``LlmPersonSimulator.value_query``/``demand_query`` paths, which
    already log under ``prompt_type="value_query"``/``"demand_query"`` --
    this function only covers the reflection question-generation call.
    """
    logger = getattr(proxy.person, "logger", None)
    if logger is None:
        return

    parsed_response: dict[str, Any] | None = None
    if parsed is not None and hasattr(parsed, "model_dump"):
        parsed_response = parsed.model_dump()

    logger.log(LlmCallRecord(
        timestamp=current_timestamp(),
        bidder_id=proxy.bidder_id,
        prompt_type="proxy_late_reflection",
        prompt=prompt,
        raw_response=raw_response,
        parsed_response=parsed_response,
        success=success,
        error=error,
        latency_seconds=latency_seconds,
        model=getattr(
            client,
            "_auctionlab_model",
            getattr(client, "model", proxy.person.model_name),
        ),
        provider=getattr(client, "_auctionlab_provider", None),
        llm_role=getattr(client, "_auctionlab_llm_role", "proxy"),
        input_tokens=getattr(client, "_last_input_tokens", None),
        output_tokens=getattr(client, "_last_output_tokens", None),
        total_tokens=getattr(client, "_last_total_tokens", None),
    ))


def run_late_reflection_for_bidder(
    *,
    proxy: LlmInferredXorProxy,
    context: LateReflectionContext,
    config: LateReflectionConfig,
    resolved_followup: str,
    prices: dict[Item, float] | None,
    instance: AuctionInstance | None,
    bidder_id: str,
    scenario: str,
    arm: str,
    mechanism: str,
    round_idx: int | None,
    trigger_reason: str,
    scope_rule: str,
    allocation_relevant_reason: str,
    client_override: LlmClient | None = None,
) -> list[LateReflectionRecord]:
    """Run one bidder's full late-reflection flow: ask, answer, (maybe) act.

    Never raises: any failure (LLM call, parsing, person answer, follow-up
    query) is captured in the returned record's ``error_message`` so one
    bidder's failure never aborts the whole auction. Always returns at
    least one record.
    """
    base = LateReflectionRecord(
        scenario=scenario,
        mechanism=mechanism,
        arm=arm,
        round_idx=round_idx,
        bidder_id=bidder_id,
        trigger_reason=trigger_reason,
        scope_rule=scope_rule,
        allocation_relevant_reason=allocation_relevant_reason,
    )

    prompt = build_late_reflection_prompt(
        scenario_description=proxy.person.scenario_description,
        item_descriptions=proxy.person.item_descriptions,
        context=context.as_prompt_dict(),
    )
    active_client = client_override or proxy.person.client

    started = time.perf_counter()
    try:
        raw = active_client.complete(prompt)
    except Exception as exc:
        base.error_message = f"reflection LLM call failed: {exc}"
        _log_late_reflection_call(
            proxy,
            prompt=prompt,
            raw_response=None,
            parsed=None,
            success=False,
            error=str(exc),
            latency_seconds=time.perf_counter() - started,
            client=active_client,
        )
        return [base]

    latency_seconds = time.perf_counter() - started
    base.tokens_in = getattr(active_client, "_last_input_tokens", None) or 0
    base.tokens_out = getattr(active_client, "_last_output_tokens", None) or 0
    base.raw_reflection_response_excerpt = raw_response_excerpt(raw)

    try:
        parsed = parse_late_reflection_response(raw)
        parsed = filter_late_reflection_bundles(
            parsed, set(proxy.person.item_descriptions)
        )
    except ValueError as exc:
        base.error_message = f"reflection parse failed: {exc}"
        base.parse_error_type = getattr(exc, "error_type", "unknown")
        # LateReflectionParseError's own excerpt takes precedence (it is
        # built from the exact raw text that failed to parse); fall back to
        # the excerpt already captured above for any other ValueError.
        base.raw_reflection_response_excerpt = (
            getattr(exc, "raw_excerpt", None) or base.raw_reflection_response_excerpt
        )
        _log_late_reflection_call(
            proxy,
            prompt=prompt,
            raw_response=raw,
            parsed=None,
            success=False,
            error=str(exc),
            latency_seconds=latency_seconds,
            client=active_client,
        )
        return [base]

    _log_late_reflection_call(
        proxy,
        prompt=prompt,
        raw_response=raw,
        parsed=parsed,
        success=True,
        error=None,
        latency_seconds=latency_seconds,
        client=active_client,
    )

    base.parse_success = True
    base.question = parsed.question
    base.reflection_mode = parsed.reflection_mode
    base.reflection_mode_inferred = parsed.reflection_mode_inferred
    base.target_type = parsed.target_type
    base.primary_bundle = (
        frozenset(parsed.primary_bundle) if parsed.primary_bundle else None
    )
    base.comparison_bundle = (
        frozenset(parsed.comparison_bundle) if parsed.comparison_bundle else None
    )
    base.marginal_item = parsed.marginal_item
    base.comparison_pair_available = bool(
        base.primary_bundle and base.comparison_bundle
    )
    base.target_bundles = [frozenset(b) for b in parsed.target_bundles if b]
    base.suggested_followup = parsed.suggested_followup

    try:
        answer = proxy.person.answer_question(parsed.question)
    except Exception as exc:
        base.error_message = f"person answer failed: {exc}"
        return [base]

    base.person_response = answer
    proxy.knowledge_base.add_qa(parsed.question, answer)
    proxy.transcript.append(
        TranscriptEntry(kind="late_reflection_question", content=parsed.question)
    )
    proxy.transcript.append(
        TranscriptEntry(kind="late_reflection_answer", content=answer)
    )

    followup_type = "none" if config.followup == "none" else resolved_followup
    candidate_bundles = [
        frozenset(b) for b in parsed.followup_bundles if b
    ] or list(base.target_bundles)
    candidate_bundles = candidate_bundles[: config.followups_per_bidder]

    if followup_type == "none" or not candidate_bundles:
        base.actual_followup_type = "none"
        return [base]

    demand_before: Bundle | None = None
    if mechanism == "clock" and prices is not None:
        demand_before = proxy.clock_demand_from_cached_bid(
            prices, top_k=1
        ).primary_bundle

    records: list[LateReflectionRecord] = []
    for rank, bundle in enumerate(candidate_bundles, start=1):
        rec = copy.copy(base)
        rec.actual_followup_type = followup_type
        rec.followup_bundle = bundle
        rec.followup_bundle_rank = rank

        old_value = (
            next(
                (a.value for a in proxy._cached_bid.atoms if a.bundle == bundle),
                None,
            )
            if proxy._cached_bid is not None
            else None
        )
        true_value = (
            instance.value_of(bidder_id, bundle) if instance is not None else None
        )

        try:
            if followup_type == "demand_query":
                if prices is None:
                    # Sealed auctions have no per-round price concept: fall
                    # back to a direct value query (documented limitation).
                    new_value = proxy.refine_bundle_value(
                        bundle, reason="late_reflection_followup"
                    )
                    rec.actual_followup_type = "value_query"
                else:
                    revalued_bundle, new_value = proxy.refine_via_demand_query(
                        bundle, prices, reason="late_reflection_followup"
                    )
                    if revalued_bundle is not None:
                        bundle = revalued_bundle
                        rec.followup_bundle = bundle
                    if new_value is None:
                        # satisfied=True: no revaluation, demand confirmed.
                        new_value = old_value
            else:
                new_value = proxy.refine_bundle_value(
                    bundle, reason="late_reflection_followup"
                )
        except Exception as exc:
            rec.error_message = f"followup query failed: {exc}"
            records.append(rec)
            continue

        rec.old_reported_value = old_value
        rec.new_reported_value = new_value
        if true_value is not None:
            rec.true_value = true_value
            if new_value is not None:
                rec.signed_correction = new_value - (
                    old_value if old_value is not None else 0.0
                )
                rec.absolute_correction = abs(rec.signed_correction)
            if old_value is not None:
                rec.old_abs_error = abs(old_value - true_value)
                rec.old_signed_error = old_value - true_value
            if new_value is not None:
                rec.new_abs_error = abs(new_value - true_value)
                rec.new_signed_error = new_value - true_value
            if rec.old_abs_error is not None and rec.new_abs_error is not None:
                rec.pricing_error_improved = rec.new_abs_error < rec.old_abs_error

        if mechanism == "clock" and prices is not None:
            demand_after = proxy.clock_demand_from_cached_bid(
                prices, top_k=1
            ).primary_bundle
            rec.demand_before = demand_before
            rec.demand_after = demand_after
            rec.demand_changed = demand_before != demand_after

        records.append(rec)

    if not records:
        return [base]

    _stamp_pair_error_fields(
        records, base.primary_bundle, base.comparison_bundle
    )
    return records


def _stamp_pair_error_fields(
    records: list[LateReflectionRecord],
    primary_bundle: Bundle | None,
    comparison_bundle: Bundle | None,
) -> None:
    """Fill in the pairwise pricing-error summary columns, if computable.

    Only meaningful when *both* halves of the comparison pair were actually
    queried (i.e. both ``primary_bundle`` and ``comparison_bundle`` appear
    as some record's ``followup_bundle`` -- not just named in the
    question) and both have a known true value. Otherwise every
    ``pair_*`` field is left ``None`` (blank in the CSV): a single-bundle
    follow-up (``followups_per_bidder=1``) or a bidder with no ground-truth
    valuation available legitimately has no pairwise comparison to report.
    Applies the same values to every record for this bidder/trigger so the
    pairwise summary is visible regardless of which follow-up row is read.
    """
    if not primary_bundle or not comparison_bundle:
        return

    by_bundle = {r.followup_bundle: r for r in records if r.followup_bundle is not None}
    primary_rec = by_bundle.get(primary_bundle)
    comparison_rec = by_bundle.get(comparison_bundle)
    if primary_rec is None or comparison_rec is None:
        return
    if primary_rec.old_abs_error is None or comparison_rec.old_abs_error is None:
        return
    if primary_rec.new_abs_error is None or comparison_rec.new_abs_error is None:
        return

    old_sum = primary_rec.old_abs_error + comparison_rec.old_abs_error
    new_sum = primary_rec.new_abs_error + comparison_rec.new_abs_error
    old_signed_sum: float | None = None
    new_signed_sum: float | None = None
    if (
        primary_rec.old_signed_error is not None
        and comparison_rec.old_signed_error is not None
    ):
        old_signed_sum = primary_rec.old_signed_error + comparison_rec.old_signed_error
    if (
        primary_rec.new_signed_error is not None
        and comparison_rec.new_signed_error is not None
    ):
        new_signed_sum = primary_rec.new_signed_error + comparison_rec.new_signed_error

    for rec in records:
        rec.pair_old_abs_error_sum = old_sum
        rec.pair_new_abs_error_sum = new_sum
        rec.pair_pricing_error_improved = new_sum < old_sum
        rec.pair_old_signed_error_sum = old_signed_sum
        rec.pair_new_signed_error_sum = new_signed_sum


# ---------------------------------------------------------------------------
# Trigger-level orchestration (allocation/welfare before-after snapshot)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Snapshot:
    allocation: dict[str, Bundle]
    reported_welfare: float
    true_welfare: float
    revenue: float
    surplus: float


def snapshot_from_bids(
    instance: AuctionInstance,
    bids_by_bidder: dict[str, XorBid],
) -> _Snapshot:
    """Solve a WDP+VCG snapshot over each bidder's current reported bid.

    Used to compute the before/after allocation and welfare columns in
    :class:`LateReflectionRecord` -- not a live mechanism run, just a
    (cheap, deterministic) solve of the same WDP the mechanisms already use.
    """
    bids = [bids_by_bidder[bidder_id] for bidder_id in instance.bidder_ids]
    result = solve_wdp_xor_ilp(instance.items, bids)
    payments = compute_vcg_payments(instance.items, bids, result)
    revenue = sum(payments.values())
    true_welfare = sum(
        instance.value_of(bidder_id, result.allocation.get(bidder_id, frozenset()))
        for bidder_id in instance.bidder_ids
    )
    return _Snapshot(
        allocation=dict(result.allocation),
        reported_welfare=result.welfare,
        true_welfare=true_welfare,
        revenue=revenue,
        surplus=true_welfare - revenue,
    )


def _build_candidate_records(
    *,
    ranked: list[tuple[str, MarginalityScore, int]],
    selected: dict[str, str],
    scenario_name: str,
    mechanism: str,
    arm: str,
    round_idx: int | None,
    scope: str,
    allocated_bundle_by_bidder: dict[str, Bundle] | None,
    demanded_bundle_by_bidder: dict[str, Bundle | None] | None,
    recent_events_by_bidder: dict[str, list[Any]] | None,
    bids_by_bidder: dict[str, XorBid],
) -> list[LateReflectionCandidateRecord]:
    """Build one ``LateReflectionCandidateRecord`` per scored bidder."""
    candidates: list[LateReflectionCandidateRecord] = []
    for bidder_id, score, rank in ranked:
        best_losing_bundle: Bundle | None = None
        best_losing_value: float | None = None
        if mechanism == "sealed" and not (allocated_bundle_by_bidder or {}).get(bidder_id):
            atom = _best_positive_atom(bids_by_bidder.get(bidder_id))
            if atom is not None:
                best_losing_bundle = atom.bundle
                best_losing_value = atom.value

        recent_events_str = ""
        if mechanism == "clock":
            events_for_bidder = (recent_events_by_bidder or {}).get(bidder_id) or []
            recent_events_str = "; ".join(sorted({
                getattr(event, "event_type", "") for event in events_for_bidder
            }))

        candidates.append(LateReflectionCandidateRecord(
            scenario=scenario_name,
            mechanism=mechanism,
            arm=arm,
            round_idx=round_idx,
            bidder_id=bidder_id,
            scope_rule=scope,
            marginality_score=score.score,
            marginality_rank=rank,
            marginality_selected=bidder_id in selected,
            marginality_reasons="; ".join(score.reasons),
            current_allocation=(allocated_bundle_by_bidder or {}).get(bidder_id),
            current_demand=(demanded_bundle_by_bidder or {}).get(bidder_id),
            recent_events=recent_events_str,
            best_losing_bundle=best_losing_bundle,
            best_losing_bundle_reported_value=best_losing_value,
        ))
    return candidates


def run_late_reflection_trigger(
    *,
    instance: AuctionInstance,
    proxies_by_bidder: dict[str, Any],
    bids_by_bidder: dict[str, XorBid],
    config: LateReflectionConfig,
    mechanism: str,
    round_idx: int | None,
    trigger_reason: str,
    allocation_relevant_bidders: dict[str, str],
    marginality_scores: dict[str, MarginalityScore] | None = None,
    scenario_name: str = "",
    arm: str = "",
    allocated_bundle_by_bidder: dict[str, Bundle] | None = None,
    demanded_bundle_by_bidder: dict[str, Bundle | None] | None = None,
    prices: dict[Item, float] | None = None,
    recent_events_by_bidder: dict[str, list[Any]] | None = None,
    recent_refinements_by_bidder: dict[str, list[Any]] | None = None,
    contested_goods: set[Item] | None = None,
    client_override: LlmClient | None = None,
) -> LateReflectionTriggerResult:
    """Run late reflection for every selected bidder once.

    Resolves ``config.scope``:

    - ``"all_bidders"``: every bidder in ``instance.bidder_ids``.
    - ``"allocation_relevant"``: exactly ``allocation_relevant_bidders``
      (unchanged, binary, pre-existing behaviour).
    - ``"allocation_marginal"``: ranks every bidder in ``marginality_scores``
      (see :func:`sealed_marginality_scores`/:func:`clock_marginality_scores`)
      and selects only the top ``config.late_reflection_max_bidders``
      positive-scoring bidders (see
      :func:`rank_and_select_marginal_bidders`) -- every scored bidder,
      selected or not, is reported in the returned ``candidates`` list for
      inspection.

    For each selected bidder, builds their context, runs their reflection +
    follow-up flow, then re-solves a before/after WDP+VCG snapshot
    (:func:`snapshot_from_bids`) and stamps every returned record with the
    shared allocation/welfare/revenue/surplus delta columns. A proxy that
    isn't an :class:`LlmInferredXorProxy` (e.g. the DNF/hybrid baselines) is
    skipped with an explanatory ``error_message`` rather than raising.
    """
    if not config.enabled:
        return LateReflectionTriggerResult(records=[], candidates=[])

    candidates: list[LateReflectionCandidateRecord] = []
    marginal_info_by_bidder: dict[str, tuple[MarginalityScore, int]] = {}
    if config.scope == "all_bidders":
        relevant = {bidder_id: "all_bidders_scope" for bidder_id in instance.bidder_ids}
    elif config.scope == "allocation_marginal":
        ranked, relevant = rank_and_select_marginal_bidders(
            marginality_scores or {}, config.late_reflection_max_bidders
        )
        candidates = _build_candidate_records(
            ranked=ranked,
            selected=relevant,
            scenario_name=scenario_name,
            mechanism=mechanism,
            arm=arm,
            round_idx=round_idx,
            scope=config.scope,
            allocated_bundle_by_bidder=allocated_bundle_by_bidder,
            demanded_bundle_by_bidder=demanded_bundle_by_bidder,
            recent_events_by_bidder=recent_events_by_bidder,
            bids_by_bidder=bids_by_bidder,
        )
        marginal_info_by_bidder = {
            bidder_id: (score, rank) for bidder_id, score, rank in ranked
        }
    else:
        relevant = dict(allocation_relevant_bidders)
        marginal_info_by_bidder = {}

    if not relevant:
        return LateReflectionTriggerResult(records=[], candidates=candidates)

    before = snapshot_from_bids(instance, bids_by_bidder)
    resolved_followup = config.resolved_followup(mechanism)

    all_records: list[LateReflectionRecord] = []
    for bidder_id, reason in sorted(relevant.items()):
        proxy_obj = proxies_by_bidder[bidder_id]
        inner = getattr(proxy_obj, "proxy", proxy_obj)
        if not isinstance(inner, LlmInferredXorProxy):
            all_records.append(LateReflectionRecord(
                scenario=scenario_name,
                mechanism=mechanism,
                arm=arm,
                round_idx=round_idx,
                bidder_id=bidder_id,
                trigger_reason=trigger_reason,
                scope_rule=config.scope,
                allocation_relevant_reason=reason,
                error_message=(
                    "proxy is not an LlmInferredXorProxy; late reflection skipped"
                ),
            ))
            continue

        context = build_late_reflection_context(
            bidder_id=bidder_id,
            mechanism=mechanism,
            round_idx=round_idx or 0,
            proxy=inner,
            allocated_bundle=(allocated_bundle_by_bidder or {}).get(bidder_id),
            demanded_bundle=(demanded_bundle_by_bidder or {}).get(bidder_id),
            current_prices=prices,
            recent_events=(recent_events_by_bidder or {}).get(bidder_id),
            recent_refinements=(recent_refinements_by_bidder or {}).get(bidder_id),
            contested_goods=contested_goods,
        )

        try:
            bidder_records = run_late_reflection_for_bidder(
                proxy=inner,
                context=context,
                config=config,
                resolved_followup=resolved_followup,
                prices=prices,
                instance=instance,
                bidder_id=bidder_id,
                scenario=scenario_name,
                arm=arm,
                mechanism=mechanism,
                round_idx=round_idx,
                trigger_reason=trigger_reason,
                scope_rule=config.scope,
                allocation_relevant_reason=reason,
                client_override=client_override,
            )
        except Exception as exc:  # last-resort guard: never abort the auction
            bidder_records = [LateReflectionRecord(
                scenario=scenario_name,
                mechanism=mechanism,
                arm=arm,
                round_idx=round_idx,
                bidder_id=bidder_id,
                trigger_reason=trigger_reason,
                scope_rule=config.scope,
                allocation_relevant_reason=reason,
                error_message=f"late reflection failed unexpectedly: {exc}",
            )]
        all_records.extend(bidder_records)

    after_bids = dict(bids_by_bidder)
    for bidder_id in relevant:
        proxy_obj = proxies_by_bidder[bidder_id]
        inner = getattr(proxy_obj, "proxy", proxy_obj)
        if isinstance(inner, LlmInferredXorProxy) and inner._cached_bid is not None:
            after_bids[bidder_id] = inner._cached_bid

    after = snapshot_from_bids(instance, after_bids)
    allocation_changed = before.allocation != after.allocation

    for rec in all_records:
        rec.allocation_before = before.allocation
        rec.allocation_after = after.allocation
        rec.allocation_changed_after_reflection = allocation_changed
        rec.true_welfare_before = before.true_welfare
        rec.true_welfare_after = after.true_welfare
        rec.welfare_delta_after_reflection = after.true_welfare - before.true_welfare
        rec.reported_welfare_before = before.reported_welfare
        rec.reported_welfare_after = after.reported_welfare
        rec.reported_welfare_delta_after_reflection = (
            after.reported_welfare - before.reported_welfare
        )
        rec.revenue_before = before.revenue
        rec.revenue_after = after.revenue
        rec.surplus_before = before.surplus
        rec.surplus_after = after.surplus

        info = marginal_info_by_bidder.get(rec.bidder_id)
        if info is not None:
            score, rank = info
            rec.marginality_score = score.score
            rec.marginality_rank = rank
            rec.marginality_selected = rec.bidder_id in relevant
            rec.marginality_reasons = "; ".join(score.reasons)

    return LateReflectionTriggerResult(records=all_records, candidates=candidates)
