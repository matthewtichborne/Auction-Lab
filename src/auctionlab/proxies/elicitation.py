"""Shared candidate-selection logic for proxy-mediated clock elicitation.

Both the ascending-clock experiment runner (``proxy_clock_runner``) and the
LLM proxy's ``clock_demand_with_refinement`` method use the same rules to
decide which bundles are worth refining in a given round.  This module
provides the shared implementation so the two paths stay in sync.
"""

from __future__ import annotations

from auctionlab.auction_types import Bundle
from auctionlab.bids.xor import XorAtomicBid


def pivotality_score(
    event_type: str,
    *,
    surplus: float = 0.0,
    gap: float = 0.0,
    margin_threshold: float = 0.0,
    tie_threshold: float = 0.0,
) -> float:
    """Pivotality score in [0, 1] for one elicitation candidate.

    Higher score means the event is more likely to affect the final
    allocation.

    ``demand_changed`` is always 1.0 (the bidder already shifted away).
    ``near_zero_surplus`` decays linearly from 1.0 (surplus = 0) to 0.0
    (surplus = margin_threshold).  ``near_tie`` decays similarly over the
    gap between the best and runner-up surplus.

    Uncertainty is implicitly 1.0: both callers skip already-refined
    bundles, so every candidate is unrefined.  When per-bundle confidence
    tracking is added to XorAtomicBid the score should become
    pivotality × (1 − confidence).
    """
    if event_type == "demand_changed":
        return 1.0
    if event_type == "near_zero_surplus":
        if margin_threshold <= 0.0:
            return 1.0
        return max(0.0, 1.0 - surplus / margin_threshold)
    if event_type == "near_tie":
        if tie_threshold <= 0.0:
            return 1.0
        return max(0.0, 1.0 - gap / tie_threshold)
    return 0.0


def candidate_refinements(
    *,
    ranked: list[tuple[XorAtomicBid, float]],
    current_primary_bundle: Bundle | None,
    previous_primary_bundle: Bundle | None,
    margin_threshold: float,
    tie_threshold: float,
    round_idx: int,
) -> list[tuple[Bundle, str, str, float]]:
    """Return ``(bundle, event_type, reason, score)`` candidates for refinement.

    ``ranked`` is a list of ``(atom, surplus)`` pairs sorted by descending
    surplus (best first), as produced by ``_ranked_surplus_atoms`` or
    ``LlmInferredXorProxy.ranked_surplus_atoms``.

    Three event types are detected:

    * ``near_zero_surplus`` — the best atom's surplus is in [0, margin_threshold].
    * ``demand_changed`` — the previous primary bundle differs from the current one.
    * ``near_tie`` — the gap between the best and runner-up surplus is ≤ tie_threshold.

    The ``score`` field holds the pivotality score so callers can sort by
    priority when the refinement budget is limited.
    """
    candidates: list[tuple[Bundle, str, str, float]] = []

    if not ranked:
        return candidates

    best_atom, best_surplus = ranked[0]

    if 0.0 <= best_surplus <= margin_threshold:
        score = pivotality_score(
            "near_zero_surplus",
            surplus=best_surplus,
            margin_threshold=margin_threshold,
        )
        candidates.append((
            best_atom.bundle,
            "near_zero_surplus",
            f"clock round {round_idx}: best surplus near zero",
            score,
        ))

    if (
        previous_primary_bundle is not None
        and previous_primary_bundle != current_primary_bundle
    ):
        candidates.append((
            previous_primary_bundle,
            "demand_changed",
            f"clock round {round_idx}: primary demand changed",
            1.0,
        ))

    if len(ranked) > 1:
        runner_up_atom, second_surplus = ranked[1]
        gap = best_surplus - second_surplus
        if gap <= tie_threshold:
            score = pivotality_score(
                "near_tie",
                gap=gap,
                tie_threshold=tie_threshold,
            )
            candidates.append((
                runner_up_atom.bundle,
                "near_tie",
                f"clock round {round_idx}: top bundles nearly tied",
                score,
            ))

    return candidates
