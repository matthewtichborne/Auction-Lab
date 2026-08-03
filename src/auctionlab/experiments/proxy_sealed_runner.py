"""Proxy-mediated sealed-bid experiments.

This module does not change ``run_sealed_xor_vcg``: that function remains a
pure mechanism over a list of XOR bids. Instead, it adds an elicitation
*phase* in front of it. Each proxy first reports an initial bid via
``submit_bid()``. If ``elicitation_rounds > 0``, the runner repeatedly:

1. solves a provisional winner-determination problem over the proxies'
   current bids,
2. derives :class:`ElicitationEvent`\\ s from that provisional allocation
   according to ``feedback_rule``,
3. sends those events to the relevant proxies via
   ``receive_provisional_feedback``, and
4. re-reads each proxy's (possibly refined) bid.

After all rounds, a final sealed XOR VCG auction is run over the proxies'
final bids. This is a preference-elicitation experiment built around the
sealed mechanism, not a replacement for it.

Feedback rules
--------------
``none``
    No feedback; proxies' initial bids are used directly.

``allocated_bundle``
    Provisionally-winning bidders are asked to refine the bundle they won.

``lost_interested_bundle``
    Losing bidders are asked to refine their highest-bid bundle.

``all_provisional``
    Both of the above.

``competitive``
    Winning bidders refine their allocated bundle. The runner then removes
    each winning atom in turn and re-solves the reported WDP, exposing a
    bounded frontier of direct counterfactual allocations. Losing bidders
    additionally contribute their single highest-scoring challenger. The
    rule stops once these allocation-relevant bundles have already been
    queried; it does not descend through the remaining candidate universe.

``all_valued_bundles``
    Every bidder (winner or loser) gets refinement events for **all** their
    positive-value bid atoms, ordered by descending bid value. The
    ``max_refinements_per_bidder`` cap then limits how many actually fire.
    This surfaces large complement-group bundles that ``competitive`` may
    miss when PV underestimates the synergy and the displacement cost
    makes the competitive score negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from auctionlab.auctions.sealed_vcg import run_sealed_xor_vcg
from auctionlab.auction_types import Bundle, Item, validate_bidder_keys
from auctionlab.bids.xor import XorBid
from auctionlab.experiments._trajectory_util import (
    aggregate_query_counts,
    logger_total_tokens,
    refinement_cap_fields,
    true_welfare_for_allocation,
)
from auctionlab.experiments.event_policy import (
    best_neighbour_bundle,
    best_scarcity_avoiding_bundle,
    contested_goods_from_bundles,
    correction_fraction,
)
from auctionlab.experiments.runner import MechanismResult
from auctionlab.instances.base import AuctionInstance
from auctionlab.llm.late_reflection import (
    LateReflectionCandidateRecord,
    LateReflectionConfig,
    LateReflectionRecord,
    run_late_reflection_trigger,
    sealed_allocation_relevant_bidders,
    sealed_marginality_scores,
)
from auctionlab.proxies.base import (
    ElicitationEvent,
    RefinementRecord,
    SealedAuctionProxy,
    clone_xor_bid,
)
from auctionlab.solvers.wdp_ilp import WdpResult, solve_wdp_xor_ilp


_VALID_FEEDBACK_RULES = {
    "none",
    "allocated_bundle",
    "lost_interested_bundle",
    "all_provisional",
    "competitive",
    "all_valued_bundles",
}

_VALID_STOPPING_RULES = {
    "fixed_rounds",
    "no_new_refinements",
}
_VALID_LOSER_CHALLENGER_POLICIES = {
    "off",
    "shadow_price",
}

MECHANISM_NAME = "proxy_sealed_vcg"


@dataclass(frozen=True)
class ProxySealedConfig:
    """Configuration for the proxy-mediated sealed elicitation phase.

    ``max_refinements_per_bidder`` and ``max_total_refinements`` are safety
    caps, not tuning targets: refinement count should be an outcome of the
    elicitation events and mechanism (feedback rule, elicitation rounds), not
    a lever used to shape results. Both default to 0 (unlimited) and should
    normally be left high enough that they never bind in main experiments.
    Bidder/bundle value queries
    are already deduplicated (a bundle is refined at most once per proxy;
    see ``LlmInferredXorProxy``/``LlmAuctionProxyAdapter``), so these caps
    only guard against runaway query volume, not repeat queries.
    """

    elicitation_rounds: int = 0
    feedback_rule: str = "none"
    # ``fixed_rounds`` always executes ``elicitation_rounds`` cycles.
    # ``no_new_refinements`` treats that value as a maximum and stops after
    # the first completed cycle that produces no new refinement queries.
    stopping_rule: str = "fixed_rounds"
    # Independent loser challengers are optional. Winner-removal
    # counterfactuals still expose losing bidders when they are relevant to a
    # reported alternative allocation.
    loser_challenger_policy: str = "off"
    # Cap on refinement queries per bidder. 0 means unlimited.
    max_refinements_per_bidder: int = 0
    # Cap on refinement queries summed across all bidders. 0 means unlimited.
    max_total_refinements: int = 0
    # Independently switchable event-policy factors used by the 8x8 ablation.
    incumbent_verification: bool = True
    pivotal_challengers: bool = False
    pivotal_gap_threshold: float = 100.0
    scarcity_fallbacks: bool = False
    large_correction_followup: bool = False
    correction_followup_threshold: float = 0.25
    terminal_regret_audit: bool = False

    def __post_init__(self) -> None:
        if self.elicitation_rounds < 0:
            raise ValueError("elicitation_rounds must be non-negative")
        if self.feedback_rule not in _VALID_FEEDBACK_RULES:
            raise ValueError(
                f"feedback_rule must be one of {sorted(_VALID_FEEDBACK_RULES)}, "
                f"got {self.feedback_rule!r}"
            )
        if self.stopping_rule not in _VALID_STOPPING_RULES:
            raise ValueError(
                f"stopping_rule must be one of "
                f"{sorted(_VALID_STOPPING_RULES)}, "
                f"got {self.stopping_rule!r}"
            )
        if self.loser_challenger_policy not in _VALID_LOSER_CHALLENGER_POLICIES:
            raise ValueError(
                "loser_challenger_policy must be one of "
                f"{sorted(_VALID_LOSER_CHALLENGER_POLICIES)}, got "
                f"{self.loser_challenger_policy!r}"
            )
        if self.max_refinements_per_bidder < 0:
            raise ValueError("max_refinements_per_bidder must be non-negative")
        if self.max_total_refinements < 0:
            raise ValueError("max_total_refinements must be non-negative")
        if self.pivotal_gap_threshold < 0:
            raise ValueError("pivotal_gap_threshold must be non-negative")
        if not 0.0 <= self.correction_followup_threshold <= 1.0:
            raise ValueError(
                "correction_followup_threshold must lie in [0, 1]"
            )


def _best_positive_value_bundle(bid: XorBid) -> Bundle | None:
    """Return the highest-value non-empty atom of ``bid``, if any is positive."""
    best_atom = None
    for atom in bid.atoms:
        if not atom.bundle or atom.value <= 0.0:
            continue
        if best_atom is None or atom.value > best_atom.value:
            best_atom = atom
    return best_atom.bundle if best_atom is not None else None


def _compute_item_shadow_prices(
    provisional_allocation: dict[str, Bundle],
    bids_by_bidder: dict[str, XorBid],
) -> dict[Item, float]:
    """Estimate the per-item cost of displacing provisionally-allocated items.

    For each item in a winner's allocated bundle, the shadow price is
    approximated as the winner's bid value for that bundle divided by the
    bundle size. This is a uniform distribution heuristic; the true marginal
    contribution would require additional WDP solves.
    """
    shadow: dict[Item, float] = {}
    for winner_id, winner_bundle in provisional_allocation.items():
        if not winner_bundle:
            continue
        winner_bid = bids_by_bidder.get(winner_id)
        if winner_bid is None:
            continue
        atom = next(
            (a for a in winner_bid.atoms if a.bundle == winner_bundle),
            None,
        )
        if atom is None:
            continue
        per_item = atom.value / len(winner_bundle)
        for item in winner_bundle:
            shadow[item] = per_item
    return shadow


def _competitive_bundle(
    bid: XorBid,
    provisional_allocation: dict[str, Bundle],
    bids_by_bidder: dict[str, XorBid],
    *,
    excluded_bundles: set[Bundle] | None = None,
) -> Bundle | None:
    """Find the bundle in ``bid`` most likely to change the WDP outcome.

    Competitiveness score = bid_value - displacement_cost, where
    displacement_cost is the sum of shadow prices for items that would need
    to be taken from their current provisional winners. A higher score means
    the bidder is closer to being able to profitably displace existing winners.
    """
    shadow = _compute_item_shadow_prices(provisional_allocation, bids_by_bidder)
    best_bundle: Bundle | None = None
    best_score = float("-inf")
    excluded = excluded_bundles or set()
    for atom in bid.atoms:
        if (
            not atom.bundle
            or atom.value <= 0.0
            or atom.bundle in excluded
        ):
            continue
        score = atom.value - sum(shadow.get(item, 0.0) for item in atom.bundle)
        if score > best_score:
            best_score = score
            best_bundle = atom.bundle
    return best_bundle


def _forced_allocation_gap(
    *,
    instance: AuctionInstance,
    bids_by_bidder: dict[str, XorBid],
    bidder_id: str,
    bundle: Bundle,
    incumbent_welfare: float,
) -> float:
    """Reported welfare loss from forcing one bidder/bundle pair."""
    bidder_atom = next(
        (
            atom
            for atom in bids_by_bidder[bidder_id].atoms
            if atom.bundle == bundle
        ),
        None,
    )
    if bidder_atom is None:
        return float("inf")
    residual_bids: list[XorBid] = []
    for other_id in instance.bidder_ids:
        if other_id == bidder_id:
            continue
        residual_bids.append(XorBid(
            bidder_id=other_id,
            atoms=[
                atom
                for atom in bids_by_bidder[other_id].atoms
                if atom.bundle.isdisjoint(bundle)
            ],
        ))
    residual = solve_wdp_xor_ilp(instance.items, residual_bids)
    return incumbent_welfare - (bidder_atom.value + residual.welfare)


def _pivotal_challenger_event(
    *,
    instance: AuctionInstance,
    bids_by_bidder: dict[str, XorBid],
    provisional: WdpResult,
    attempted_bundles_by_bidder: dict[str, set[Bundle]],
    gap_threshold: float,
    round_idx: int,
    event_type: str = "pivotal_challenger",
) -> ElicitationEvent | None:
    """Select one globally closest unqueried forced-allocation challenger."""
    candidates: list[tuple[float, str, Bundle]] = []
    for bidder_id in instance.bidder_ids:
        excluded = set(attempted_bundles_by_bidder[bidder_id])
        incumbent = provisional.allocation.get(bidder_id, frozenset())
        if incumbent:
            excluded.add(incumbent)
        # Exact pivotality is evaluated only for a bounded shadow-price
        # shortlist (one bundle per bidder), not the full candidate support.
        bundle = _competitive_bundle(
            bids_by_bidder[bidder_id],
            provisional.allocation,
            bids_by_bidder,
            excluded_bundles=excluded,
        )
        if bundle is None:
            continue
        gap = _forced_allocation_gap(
            instance=instance,
            bids_by_bidder=bids_by_bidder,
            bidder_id=bidder_id,
            bundle=bundle,
            incumbent_welfare=provisional.welfare,
        )
        if abs(gap) <= gap_threshold:
            candidates.append((abs(gap), bidder_id, bundle))
    if not candidates:
        return None
    abs_gap, bidder_id, bundle = min(
        candidates,
        key=lambda row: (row[0], row[1], tuple(sorted(row[2]))),
    )
    return ElicitationEvent(
        mechanism=MECHANISM_NAME,
        event_type=event_type,
        bidder_id=bidder_id,
        bundle=bundle,
        allocated_bundle=provisional.allocation.get(
            bidder_id, frozenset()
        ),
        allocation_gap=abs_gap,
        reason=(
            f"sealed round {round_idx + 1}: closest unqueried forced-"
            f"allocation challenger (reported welfare gap={abs_gap:.2f})"
        ),
        round_idx=round_idx,
    )


def _scarcity_fallback_events(
    *,
    instance: AuctionInstance,
    bids_by_bidder: dict[str, XorBid],
    provisional: WdpResult,
    attempted_bundles_by_bidder: dict[str, set[Bundle]],
    round_idx: int,
) -> list[ElicitationEvent]:
    """Select at most two alternatives that avoid currently contested goods."""
    primary = {
        bidder_id: (
            provisional.allocation.get(bidder_id, frozenset())
            or _best_positive_value_bundle(bids_by_bidder[bidder_id])
        )
        for bidder_id in instance.bidder_ids
    }
    contested = contested_goods_from_bundles(primary.values())
    ranked: list[tuple[float, str, Bundle]] = []
    for bidder_id in instance.bidder_ids:
        bundle = best_scarcity_avoiding_bundle(
            bids_by_bidder[bidder_id],
            primary[bidder_id],
            contested,
            excluded=attempted_bundles_by_bidder[bidder_id],
        )
        if bundle is None:
            continue
        value = next(
            atom.value
            for atom in bids_by_bidder[bidder_id].atoms
            if atom.bundle == bundle
        )
        ranked.append((-value, bidder_id, bundle))
    events: list[ElicitationEvent] = []
    for _, bidder_id, bundle in sorted(ranked)[:2]:
        avoided = sorted(
            set(primary[bidder_id] or frozenset()) & contested - set(bundle)
        )
        events.append(ElicitationEvent(
            mechanism=MECHANISM_NAME,
            event_type="scarcity_avoiding_fallback",
            bidder_id=bidder_id,
            bundle=bundle,
            allocated_bundle=provisional.allocation.get(
                bidder_id, frozenset()
            ),
            reason=(
                f"sealed round {round_idx + 1}: fallback avoids contested "
                f"goods {avoided}"
            ),
            round_idx=round_idx,
        ))
    return events


def _competitive_frontier(
    *,
    instance: AuctionInstance,
    bids_by_bidder: dict[str, XorBid],
    provisional: WdpResult,
    loser_challenger_policy: str = "off",
) -> dict[str, list[tuple[Bundle, str]]]:
    """Return a small allocation-relevant candidate frontier per bidder.

    The frontier deliberately avoids walking every atom in descending score.
    It contains:

    * bundles exposed by removing each current winning atom and re-solving;
    * for losers, their single most competitive reported bundle.

    This keeps competitive elicitation local to allocations that can
    plausibly replace or modify the incumbent outcome.
    """
    frontier: dict[str, list[tuple[Bundle, str]]] = {
        bidder_id: [] for bidder_id in instance.bidder_ids
    }

    def _append(bidder_id: str, bundle: Bundle, source: str) -> None:
        if not bundle:
            return
        if all(existing != bundle for existing, _ in frontier[bidder_id]):
            frontier[bidder_id].append((bundle, source))

    # Closest allocation-level alternatives: remove one incumbent atom and
    # solve the reported WDP again.
    for removed_bidder in instance.bidder_ids:
        removed_bundle = provisional.allocation.get(
            removed_bidder, frozenset()
        )
        if not removed_bundle:
            continue
        counterfactual_bids: list[XorBid] = []
        for bidder_id in instance.bidder_ids:
            bid = bids_by_bidder[bidder_id]
            atoms = list(bid.atoms)
            if bidder_id == removed_bidder:
                atoms = [
                    atom
                    for atom in atoms
                    if atom.bundle != removed_bundle
                ]
            counterfactual_bids.append(
                XorBid(bidder_id=bidder_id, atoms=atoms)
            )
        alternative = solve_wdp_xor_ilp(
            instance.items, counterfactual_bids
        )
        for bidder_id in instance.bidder_ids:
            _append(
                bidder_id,
                alternative.allocation.get(bidder_id, frozenset()),
                "counterfactual",
            )

    if loser_challenger_policy == "shadow_price":
        for bidder_id in instance.bidder_ids:
            allocated = provisional.allocation.get(
                bidder_id, frozenset()
            )
            if allocated:
                continue
            target = _competitive_bundle(
                bids_by_bidder[bidder_id],
                provisional.allocation,
                bids_by_bidder,
            )
            if target is not None:
                _append(
                    bidder_id,
                    target,
                    "loser_challenger",
                )

    return frontier


def _all_valued_bundle_events(
    bidder_id: str,
    bid: XorBid,
    allocated_bundle: Bundle,
    round_idx: int,
) -> list[ElicitationEvent]:
    """Events for every positive-value atom in ``bid``, descending by value.

    Used by the ``all_valued_bundles`` feedback rule so the proxy is asked
    to refine its highest-value bundles first (including complement groups
    that may only appear as large, non-winning atoms). The runner's
    max_refinements_per_bidder cap truncates how many actually fire.
    """
    sorted_atoms = sorted(
        (a for a in bid.atoms if a.bundle and a.value > 0.0),
        key=lambda a: a.value,
        reverse=True,
    )
    return [
        ElicitationEvent(
            mechanism=MECHANISM_NAME,
            event_type="valued_bundle",
            bidder_id=bidder_id,
            bundle=atom.bundle,
            allocated_bundle=allocated_bundle,
            reason=(
                f"sealed round {round_idx + 1}: refine bundle "
                f"(bid_value={atom.value:.0f})"
            ),
            round_idx=round_idx,
        )
        for atom in sorted_atoms
    ]


def _provisional_events(
    *,
    instance: AuctionInstance,
    bids_by_bidder: dict[str, XorBid],
    provisional: WdpResult,
    feedback_rule: str,
    round_idx: int,
    loser_challenger_policy: str = "off",
    attempted_bundles_by_bidder: dict[str, set[Bundle]] | None = None,
) -> list[ElicitationEvent]:
    """Derive elicitation events from a pre-computed provisional allocation."""
    if feedback_rule == "none":
        return []

    events: list[ElicitationEvent] = []
    competitive_frontier = (
        _competitive_frontier(
            instance=instance,
            bids_by_bidder=bids_by_bidder,
            provisional=provisional,
            loser_challenger_policy=loser_challenger_policy,
        )
        if feedback_rule == "competitive"
        else {}
    )

    for bidder_id in instance.bidder_ids:
        allocated_bundle = provisional.allocation.get(bidder_id, frozenset())
        attempted = (
            attempted_bundles_by_bidder.get(bidder_id, set())
            if attempted_bundles_by_bidder is not None
            else set()
        )

        if feedback_rule == "all_valued_bundles":
            # Ask every bidder to refine ALL their positive-value bundles in
            # descending value order.  The runner cap limits actual queries.
            events.extend(
                _all_valued_bundle_events(
                    bidder_id,
                    bids_by_bidder[bidder_id],
                    allocated_bundle,
                    round_idx,
                )
            )
            continue

        if allocated_bundle:
            if feedback_rule in (
                "allocated_bundle",
                "all_provisional",
                "competitive",
            ):
                target = allocated_bundle
                event_type = "allocated_bundle"
                reason = (
                    f"sealed round {round_idx + 1}: provisionally allocated"
                )
                if feedback_rule == "competitive" and target in attempted:
                    next_frontier = next(
                        (
                            (bundle, source)
                            for bundle, source in competitive_frontier[
                                bidder_id
                            ]
                            if bundle not in attempted
                        ),
                        None,
                    )
                    target = (
                        next_frontier[0]
                        if next_frontier is not None
                        else None
                    )
                    if next_frontier is not None:
                        source = next_frontier[1]
                        event_type = f"competitive_{source}"
                        reason = (
                            f"sealed round {round_idx + 1}: unqueried "
                            f"{source.replace('_', ' ')} on allocation frontier"
                        )
                if target is not None:
                    events.append(
                        ElicitationEvent(
                            mechanism=MECHANISM_NAME,
                            event_type=event_type,
                            bidder_id=bidder_id,
                            bundle=target,
                            allocated_bundle=allocated_bundle,
                            reason=reason,
                            round_idx=round_idx,
                        )
                    )
        else:
            if feedback_rule == "competitive":
                next_frontier = next(
                    (
                        (bundle, source)
                        for bundle, source in competitive_frontier[bidder_id]
                        if bundle not in attempted
                    ),
                    None,
                )
                if next_frontier is not None:
                    target, source = next_frontier
                    events.append(
                        ElicitationEvent(
                            mechanism=MECHANISM_NAME,
                            event_type=f"competitive_{source}",
                            bidder_id=bidder_id,
                            bundle=target,
                            allocated_bundle=frozenset(),
                            reason=(
                                f"sealed round {round_idx + 1}: most "
                                f"relevant unqueried {source.replace('_', ' ')}"
                            ),
                            round_idx=round_idx,
                        )
                    )
            elif feedback_rule in ("lost_interested_bundle", "all_provisional"):
                interested_bundle = _best_positive_value_bundle(
                    bids_by_bidder[bidder_id]
                )
                if interested_bundle is not None:
                    events.append(
                        ElicitationEvent(
                            mechanism=MECHANISM_NAME,
                            event_type="lost_interested_bundle",
                            bidder_id=bidder_id,
                            bundle=interested_bundle,
                            allocated_bundle=frozenset(),
                            reason=(
                                f"sealed round {round_idx + 1}: lost, refine "
                                "best reported bundle"
                            ),
                            round_idx=round_idx,
                        )
                    )

    return events


def _proxy_stats_snapshot(
    proxies_by_bidder: dict[str, SealedAuctionProxy],
) -> dict[str, dict[str, int]]:
    """Per-bidder cumulative query counters, copied out of the live ``ProxyStats``."""
    snapshot: dict[str, dict[str, int]] = {}
    for bidder_id, proxy in proxies_by_bidder.items():
        stats = proxy.stats()
        snapshot[bidder_id] = {
            "value_queries": stats.value_queries,
            "demand_queries": stats.demand_queries,
            "nl_queries": stats.nl_queries,
            "refinement_queries": stats.refinement_queries,
        }
    return snapshot


def run_proxy_sealed_vcg_trajectory(
    instance: AuctionInstance,
    proxies: list[SealedAuctionProxy],
    config: ProxySealedConfig,
    *,
    logger: Any | None = None,
    late_reflection_config: LateReflectionConfig | None = None,
    late_reflection_client: Any | None = None,
    scenario_name: str = "",
) -> list[MechanismResult]:
    """Run a proxy-mediated sealed XOR VCG experiment, recording every round.

    Returns one :class:`MechanismResult` per executed round. Under
    ``stopping_rule="fixed_rounds"`` this is index
    0..``config.elicitation_rounds``. Under
    ``stopping_rule="no_new_refinements"``, ``elicitation_rounds`` is a
    maximum and the trajectory can end earlier.

    - round 0 is each proxy's initial bid (after any shared NL/interest-map/
      provisional-valuation initialisation the caller already performed),
      before any sealed feedback/refinement.
    - round r (1..R) is the allocation/result after r sealed
      feedback/refinement cycles.

    Proxy state (and therefore ``proxies``) is never reset between rounds:
    the same proxy objects accumulate refinements across the whole
    trajectory. ``run_proxy_sealed_vcg_experiment`` is the special case that
    returns only the final round, kept for backward compatibility.

    ``logger``, if given (an :class:`~auctionlab.llm.logging.LlmCallLogger`),
    is used to attribute LLM token usage and query counts to each round.
    Token usage is read via ``total_tokens()`` (unaffected by ``mark()``).
    Value/demand/nl query *counts* are read via ``stats_since_mark()``,
    bucketed by prompt type exactly like
    :func:`auctionlab.experiments.run_config.collect_arm_stats` -- that is
    the canonical source for the query counts already shown in the CLI's
    final per-arm summary (including refinement- and ground-truth-triggered
    queries, which ``ProxyStats.value_queries`` alone does not capture; see
    :mod:`auctionlab.experiments._trajectory_util`). Without a logger (e.g.
    toy/scripted proxies in tests), falls back to summing ``ProxyStats``
    fields across bidders.

    ``late_reflection_config``, if given and ``.enabled``, fires the
    ``late_reflection`` elicitation event exactly once: after the final
    round's (``round_number == config.elicitation_rounds``) ordinary
    feedback/refinement events are applied, but before that round's final
    bids/allocation are recorded -- i.e. using the pre-final round's
    provisional allocation to decide which bidders are allocation-relevant
    (see :func:`~auctionlab.llm.late_reflection.sealed_allocation_relevant_bidders`).
    Requires ``config.elicitation_rounds >= 1``; with 0 rounds there is no
    pre-final provisional state to trigger from, so late reflection is a
    no-op. Records are attached to the final round's
    ``MechanismResult.metadata["late_reflection_records"]``.
    """
    proxies_by_bidder = {proxy.bidder_id: proxy for proxy in proxies}
    validate_bidder_keys(
        bidder_ids=instance.bidder_ids,
        values=proxies_by_bidder,
        label="proxies",
    )

    initial_bids: dict[str, XorBid] = {
        bidder_id: clone_xor_bid(proxies_by_bidder[bidder_id].submit_bid())
        for bidder_id in instance.bidder_ids
    }

    prev_stats = {
        bidder_id: {
            "value_queries": 0,
            "demand_queries": 0,
            "nl_queries": 0,
            "refinement_queries": 0,
        }
        for bidder_id in instance.bidder_ids
    }
    prev_tok_in, prev_tok_out = logger_total_tokens(logger)
    cumulative_tokens_in = 0
    cumulative_tokens_out = 0
    prev_vq, prev_dq, prev_nl = 0, 0, 0

    def _record_round(round_idx: int) -> MechanismResult:
        nonlocal prev_stats, prev_tok_in, prev_tok_out
        nonlocal cumulative_tokens_in, cumulative_tokens_out
        nonlocal prev_vq, prev_dq, prev_nl

        bids_now: dict[str, XorBid] = {
            bidder_id: clone_xor_bid(proxies_by_bidder[bidder_id].submit_bid())
            for bidder_id in instance.bidder_ids
        }
        outcome = run_sealed_xor_vcg(
            items=instance.items,
            bids=[bids_now[bidder_id] for bidder_id in instance.bidder_ids],
        )

        stats_now = _proxy_stats_snapshot(proxies_by_bidder)
        new_by_bidder = {
            bidder_id: {
                key: stats_now[bidder_id][key] - prev_stats[bidder_id][key]
                for key in stats_now[bidder_id]
            }
            for bidder_id in instance.bidder_ids
        }

        tok_in_total, tok_out_total = logger_total_tokens(logger)
        new_tok_in = tok_in_total - prev_tok_in
        new_tok_out = tok_out_total - prev_tok_out
        cumulative_tokens_in += new_tok_in
        cumulative_tokens_out += new_tok_out
        prev_tok_in, prev_tok_out = tok_in_total, tok_out_total

        cum_vq, cum_dq, cum_nl = aggregate_query_counts(logger, stats_now)
        new_vq = cum_vq - prev_vq
        new_dq = cum_dq - prev_dq
        new_nl = cum_nl - prev_nl
        prev_vq, prev_dq, prev_nl = cum_vq, cum_dq, cum_nl

        refinement_query_count_by_bidder = {
            bidder_id: stats_now[bidder_id]["refinement_queries"]
            for bidder_id in instance.bidder_ids
        }
        refinement_records_by_bidder: dict[str, list[RefinementRecord]] = {
            bidder_id: list(
                getattr(proxy, "refinement_records", lambda: [])()
            )
            for bidder_id, proxy in proxies_by_bidder.items()
        }
        cap_fields = refinement_cap_fields(
            refinement_query_count_by_bidder,
            max_refinements_per_bidder=config.max_refinements_per_bidder,
            max_total_refinements=config.max_total_refinements,
        )

        if round_idx == 0:
            mechanism = f"{MECHANISM_NAME}_static"
        else:
            mechanism = (
                f"{MECHANISM_NAME}_elicited_{config.feedback_rule}_{round_idx}"
            )

        result = MechanismResult(
            mechanism=mechanism,
            allocation=outcome.allocation,
            welfare=outcome.welfare,
            payments=outcome.payments,
            revenue=sum(outcome.payments.values()),
            rounds=round_idx or None,
            query_count=sum(refinement_query_count_by_bidder.values()),
            metadata={
                "elicitation_rounds": round_idx,
                "requested_elicitation_rounds": config.elicitation_rounds,
                "feedback_rule": config.feedback_rule,
                "stopping_rule": config.stopping_rule,
                "loser_challenger_policy": config.loser_challenger_policy,
                "incumbent_verification": config.incumbent_verification,
                "pivotal_challengers": config.pivotal_challengers,
                "pivotal_gap_threshold": config.pivotal_gap_threshold,
                "scarcity_fallbacks": config.scarcity_fallbacks,
                "large_correction_followup": (
                    config.large_correction_followup
                ),
                "correction_followup_threshold": (
                    config.correction_followup_threshold
                ),
                "terminal_regret_audit": config.terminal_regret_audit,
                "termination_reason": "",
                "events_blocked_by_refinement_cap": 0,
                "max_refinements_per_bidder": config.max_refinements_per_bidder,
                "max_total_refinements": config.max_total_refinements,
                "refinement_query_count_by_bidder": (
                    refinement_query_count_by_bidder
                ),
                **cap_fields,
                "new_refinement_query_count_by_bidder": {
                    bidder_id: new_by_bidder[bidder_id]["refinement_queries"]
                    for bidder_id in instance.bidder_ids
                },
                "cumulative_value_queries": cum_vq,
                "new_value_queries": new_vq,
                "cumulative_demand_queries": cum_dq,
                "new_demand_queries": new_dq,
                "cumulative_nl_queries": cum_nl,
                "new_nl_queries": new_nl,
                "tokens_in": cumulative_tokens_in,
                "tokens_out": cumulative_tokens_out,
                "new_tokens_in": new_tok_in,
                "new_tokens_out": new_tok_out,
                "initial_bids": initial_bids,
                "final_bids": bids_now,
                "refinement_records_by_bidder": refinement_records_by_bidder,
                "vcg_counterfactuals": outcome.vcg_counterfactuals,
            },
        )

        prev_stats = stats_now
        return result

    trajectory = [_record_round(0)]
    attempted_bundles_by_bidder: dict[str, set[Bundle]] = {
        bidder_id: {
            record.bundle
            for record in getattr(
                proxies_by_bidder[bidder_id],
                "refinement_records",
                lambda: [],
            )()
            if record.bundle
        }
        for bidder_id in instance.bidder_ids
    }
    late_reflection_records: list[LateReflectionRecord] = []
    late_reflection_candidates: list[LateReflectionCandidateRecord] = []
    late_reflection_fired = False
    scarcity_queried_bidders: set[str] = set()
    correction_followup_bidders: set[str] = set()

    for round_number in range(1, config.elicitation_rounds + 1):
        bids_by_bidder: dict[str, XorBid] = {
            bidder_id: proxies_by_bidder[bidder_id].current_bid()
            for bidder_id in instance.bidder_ids
        }

        provisional = solve_wdp_xor_ilp(
            instance.items,
            [bids_by_bidder[bidder_id] for bidder_id in instance.bidder_ids],
        )
        # provisional.welfare is the WDP objective over bidders' current
        # REPORTED bids -- it is not ground-truth welfare, and can overstate
        # it when a proxy's reported bid is miscalibrated. Always print both
        # so the two are never conflated (matches the reported/true
        # convention used everywhere else in the CLI's summaries).
        provisional_true_welfare = true_welfare_for_allocation(
            instance, provisional.allocation
        )

        print(
            f"\n  ── sealed round {round_number}/{config.elicitation_rounds}"
            f"  reported welfare {provisional.welfare:.0f}"
            f"  true welfare {provisional_true_welfare:.0f}",
            flush=True,
        )
        for bidder_id in instance.bidder_ids:
            alloc = provisional.allocation.get(bidder_id, frozenset())
            if alloc:
                bundle_str = "{" + ", ".join(sorted(alloc)) + "}"
                print(f"    {bidder_id:<14}  →  {bundle_str}", flush=True)

        events = _provisional_events(
            instance=instance,
            bids_by_bidder=bids_by_bidder,
            provisional=provisional,
            feedback_rule=config.feedback_rule,
            round_idx=round_number - 1,
            loser_challenger_policy=config.loser_challenger_policy,
            attempted_bundles_by_bidder=(
                attempted_bundles_by_bidder
                if config.feedback_rule == "competitive"
                else None
            ),
        )
        if not config.incumbent_verification:
            events = [
                event
                for event in events
                if event.event_type != "allocated_bundle"
            ]
        if config.pivotal_challengers:
            pivotal = _pivotal_challenger_event(
                instance=instance,
                bids_by_bidder=bids_by_bidder,
                provisional=provisional,
                attempted_bundles_by_bidder=attempted_bundles_by_bidder,
                gap_threshold=config.pivotal_gap_threshold,
                round_idx=round_number - 1,
            )
            if pivotal is not None:
                events.append(pivotal)
        if config.scarcity_fallbacks:
            events.extend(
                event
                for event in _scarcity_fallback_events(
                    instance=instance,
                    bids_by_bidder=bids_by_bidder,
                    provisional=provisional,
                    attempted_bundles_by_bidder=attempted_bundles_by_bidder,
                    round_idx=round_number - 1,
                )
                if event.bidder_id not in scarcity_queried_bidders
            )
        # A bidder/bundle pair should be queried at most once even when
        # several independent policy factors identify it in the same round.
        deduplicated: list[ElicitationEvent] = []
        seen_event_keys: set[tuple[str, Bundle | None]] = set()
        for event in events:
            key = (event.bidder_id, event.bundle)
            if key not in seen_event_keys:
                seen_event_keys.add(key)
                deduplicated.append(event)
        events = deduplicated
        no_eligible_competitive_events = (
            config.feedback_rule == "competitive" and not events
        )

        refinements_before = sum(
            proxy.stats().refinement_queries
            for proxy in proxies_by_bidder.values()
        )
        events_blocked_by_refinement_cap = 0
        event_idx = 0
        while event_idx < len(events):
            event = events[event_idx]
            event_idx += 1
            bidder_id = event.bidder_id
            if (
                config.max_refinements_per_bidder > 0
                and proxies_by_bidder[bidder_id].stats().refinement_queries
                >= config.max_refinements_per_bidder
            ):
                events_blocked_by_refinement_cap += 1
                continue
            if config.max_total_refinements > 0 and (
                sum(
                    p.stats().refinement_queries
                    for p in proxies_by_bidder.values()
                )
                >= config.max_total_refinements
            ):
                events_blocked_by_refinement_cap += 1
                continue

            bundle_str = (
                "{" + ",".join(sorted(event.bundle)) + "}"
                if event.bundle
                else "∅"
            )
            print(
                f"  {bidder_id:<12}  {event.event_type}  {bundle_str}",
                flush=True,
            )

            records_getter = getattr(
                proxies_by_bidder[bidder_id],
                "refinement_records",
                None,
            )
            records_before = (
                len(records_getter()) if callable(records_getter) else 0
            )
            proxies_by_bidder[bidder_id].receive_provisional_feedback(event)
            if event.bundle:
                attempted_bundles_by_bidder[bidder_id].add(event.bundle)
            if event.event_type == "scarcity_avoiding_fallback":
                scarcity_queried_bidders.add(bidder_id)
            new_records = (
                records_getter()[records_before:]
                if callable(records_getter)
                else []
            )
            if (
                config.large_correction_followup
                and bidder_id not in correction_followup_bidders
                and event.event_type != "large_correction_followup"
                and any(
                    correction_fraction(record)
                    >= config.correction_followup_threshold
                    for record in new_records
                )
            ):
                source = new_records[-1].bundle
                neighbour = best_neighbour_bundle(
                    proxies_by_bidder[bidder_id].current_bid(),
                    source,
                    excluded=attempted_bundles_by_bidder[bidder_id],
                )
                if neighbour is not None:
                    correction_followup_bidders.add(bidder_id)
                    events.append(ElicitationEvent(
                        mechanism=MECHANISM_NAME,
                        event_type="large_correction_followup",
                        bidder_id=bidder_id,
                        bundle=neighbour,
                        allocated_bundle=event.allocated_bundle,
                        reason=(
                            f"sealed round {round_number}: one-step neighbour "
                            f"after correction fraction "
                            f"{correction_fraction(new_records[-1]):.3f}"
                        ),
                        round_idx=round_number - 1,
                    ))

        refinements_after_ordinary = sum(
            proxy.stats().refinement_queries
            for proxy in proxies_by_bidder.values()
        )
        no_new_ordinary_refinements = (
            refinements_after_ordinary == refinements_before
        )
        is_max_round = round_number == config.elicitation_rounds
        is_convergence_candidate = (
            config.stopping_rule == "no_new_refinements"
            and no_new_ordinary_refinements
        )
        if (
            config.terminal_regret_audit
            and (is_max_round or is_convergence_candidate)
        ):
            terminal_bids = {
                bidder_id: proxies_by_bidder[bidder_id].current_bid()
                for bidder_id in instance.bidder_ids
            }
            terminal_outcome = solve_wdp_xor_ilp(
                instance.items,
                [
                    terminal_bids[bidder_id]
                    for bidder_id in instance.bidder_ids
                ],
            )
            terminal_event = _pivotal_challenger_event(
                instance=instance,
                bids_by_bidder=terminal_bids,
                provisional=terminal_outcome,
                attempted_bundles_by_bidder=attempted_bundles_by_bidder,
                gap_threshold=config.pivotal_gap_threshold,
                round_idx=round_number - 1,
                event_type="terminal_regret_challenger",
            )
            if terminal_event is not None:
                bidder_id = terminal_event.bidder_id
                if (
                    (
                        config.max_refinements_per_bidder == 0
                        or proxies_by_bidder[
                            bidder_id
                        ].stats().refinement_queries
                        < config.max_refinements_per_bidder
                    )
                    and (
                        config.max_total_refinements == 0
                        or sum(
                            proxy.stats().refinement_queries
                            for proxy in proxies_by_bidder.values()
                        ) < config.max_total_refinements
                    )
                ):
                    proxies_by_bidder[
                        bidder_id
                    ].receive_provisional_feedback(terminal_event)
                    attempted_bundles_by_bidder[bidder_id].add(
                        terminal_event.bundle
                    )
                    events.append(terminal_event)
                    no_new_ordinary_refinements = False
                    is_convergence_candidate = False
        if (
            (is_max_round or is_convergence_candidate)
            and late_reflection_config is not None
            and late_reflection_config.enabled
            and not late_reflection_fired
        ):
            relevant = sealed_allocation_relevant_bidders(events)
            post_refinement_bids = {
                bidder_id: proxies_by_bidder[bidder_id].current_bid()
                for bidder_id in instance.bidder_ids
            }
            marginality_scores = None
            if late_reflection_config.scope == "allocation_marginal":
                marginality_scores = sealed_marginality_scores(
                    bidder_ids=instance.bidder_ids,
                    provisional_allocation=dict(provisional.allocation),
                    # trajectory currently holds rounds 0..round_number-1 --
                    # trajectory[-1] is exactly the previous round's final
                    # (post-refinement) allocation.
                    previous_allocation=(
                        trajectory[-1].allocation if trajectory else None
                    ),
                    bids_by_bidder=post_refinement_bids,
                    events=events,
                )
            lr_result = run_late_reflection_trigger(
                instance=instance,
                proxies_by_bidder=proxies_by_bidder,
                bids_by_bidder=post_refinement_bids,
                config=late_reflection_config,
                mechanism="sealed",
                round_idx=round_number,
                trigger_reason="sealed_pre_final_round",
                allocation_relevant_bidders=relevant,
                marginality_scores=marginality_scores,
                scenario_name=scenario_name,
                arm=f"proxy_sealed_{config.feedback_rule}",
                allocated_bundle_by_bidder=dict(provisional.allocation),
                client_override=late_reflection_client,
            )
            late_reflection_records.extend(lr_result.records)
            late_reflection_candidates.extend(lr_result.candidates)
            late_reflection_fired = True

        trajectory.append(_record_round(round_number))
        trajectory[-1].metadata["events_blocked_by_refinement_cap"] = (
            events_blocked_by_refinement_cap
        )
        new_refinements = sum(
            trajectory[-1].metadata[
                "new_refinement_query_count_by_bidder"
            ].values()
        )
        if config.feedback_rule == "competitive":
            converged = (
                config.stopping_rule == "no_new_refinements"
                and no_eligible_competitive_events
            )
        else:
            converged = (
                config.stopping_rule == "no_new_refinements"
                and new_refinements == 0
            )
        if converged or is_max_round:
            trajectory[-1].metadata["termination_reason"] = (
                (
                    "refinement_cap_reached"
                    if events_blocked_by_refinement_cap
                    else (
                        "no_eligible_refinements"
                        if config.feedback_rule == "competitive"
                        else "no_new_refinements"
                    )
                )
                if converged
                else "max_rounds_reached"
            )
            trajectory[-1].metadata["late_reflection_records"] = (
                late_reflection_records
            )
            trajectory[-1].metadata["late_reflection_candidates"] = (
                late_reflection_candidates
            )
        if converged:
            reason = trajectory[-1].metadata["termination_reason"]
            print(
                f"  sealed elicitation stopped after round {round_number}: "
                + (
                    "refinement safety cap prevented further queries"
                    if reason == "refinement_cap_reached"
                    else (
                        "no eligible unqueried competitive bundles"
                        if reason == "no_eligible_refinements"
                        else "no new refinement queries"
                    )
                ),
                flush=True,
            )
            break

    if config.elicitation_rounds == 0:
        trajectory[-1].metadata["termination_reason"] = "no_elicitation_rounds"

    return trajectory


def run_proxy_sealed_vcg_experiment(
    instance: AuctionInstance,
    proxies: list[SealedAuctionProxy],
    config: ProxySealedConfig,
    *,
    late_reflection_config: LateReflectionConfig | None = None,
    late_reflection_client: Any | None = None,
    scenario_name: str = "",
) -> MechanismResult:
    """Run a proxy-mediated sealed XOR VCG experiment, returning the final round.

    With ``config.elicitation_rounds == 0`` this reproduces the static
    sealed-proxy baseline: each proxy's initial ``submit_bid()`` is used
    directly. With ``elicitation_rounds > 0``, proxies are given a chance to
    refine their bids in response to provisional-allocation feedback before
    the final auction is run. See :func:`run_proxy_sealed_vcg_trajectory` for
    the full per-round trajectory.
    """
    return run_proxy_sealed_vcg_trajectory(
        instance,
        proxies,
        config,
        late_reflection_config=late_reflection_config,
        late_reflection_client=late_reflection_client,
        scenario_name=scenario_name,
    )[-1]
