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
    Winning bidders refine their allocated bundle (same as
    ``allocated_bundle``). Losing bidders refine the bundle in their bid
    with the highest *competitiveness score*: bid_value minus the estimated
    cost of displacing the items it needs from their current provisional
    winners. This is more targeted than ``lost_interested_bundle`` because
    it focuses the query budget on bundles that are close to changing the
    WDP outcome.

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
from auctionlab.experiments.runner import MechanismResult
from auctionlab.instances.base import AuctionInstance
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

MECHANISM_NAME = "proxy_sealed_vcg"


@dataclass(frozen=True)
class ProxySealedConfig:
    """Configuration for the proxy-mediated sealed elicitation phase.

    ``max_refinements_per_bidder`` and ``max_total_refinements`` are safety
    caps, not tuning targets: refinement count should be an outcome of the
    elicitation events and mechanism (feedback rule, elicitation rounds), not
    a lever used to shape results. Both default to 0 (unlimited) and should
    normally be left high enough that they never bind in main experiments --
    see ``docs/parameter_tuning_methodology.md``. Bidder/bundle value queries
    are already deduplicated (a bundle is refined at most once per proxy;
    see ``LlmInferredXorProxy``/``LlmAuctionProxyAdapter``), so these caps
    only guard against runaway query volume, not repeat queries.
    """

    elicitation_rounds: int = 0
    feedback_rule: str = "none"
    # Cap on refinement queries per bidder. 0 means unlimited.
    max_refinements_per_bidder: int = 0
    # Cap on refinement queries summed across all bidders. 0 means unlimited.
    max_total_refinements: int = 0

    def __post_init__(self) -> None:
        if self.elicitation_rounds < 0:
            raise ValueError("elicitation_rounds must be non-negative")
        if self.feedback_rule not in _VALID_FEEDBACK_RULES:
            raise ValueError(
                f"feedback_rule must be one of {sorted(_VALID_FEEDBACK_RULES)}, "
                f"got {self.feedback_rule!r}"
            )
        if self.max_refinements_per_bidder < 0:
            raise ValueError("max_refinements_per_bidder must be non-negative")
        if self.max_total_refinements < 0:
            raise ValueError("max_total_refinements must be non-negative")


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
    for atom in bid.atoms:
        if not atom.bundle or atom.value <= 0.0:
            continue
        score = atom.value - sum(shadow.get(item, 0.0) for item in atom.bundle)
        if score > best_score:
            best_score = score
            best_bundle = atom.bundle
    return best_bundle


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
                f"sealed round {round_idx}: refine bundle "
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
) -> list[ElicitationEvent]:
    """Derive elicitation events from a pre-computed provisional allocation."""
    if feedback_rule == "none":
        return []

    events: list[ElicitationEvent] = []

    for bidder_id in instance.bidder_ids:
        allocated_bundle = provisional.allocation.get(bidder_id, frozenset())

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
                events.append(
                    ElicitationEvent(
                        mechanism=MECHANISM_NAME,
                        event_type="allocated_bundle",
                        bidder_id=bidder_id,
                        bundle=allocated_bundle,
                        allocated_bundle=allocated_bundle,
                        reason=(
                            f"sealed round {round_idx}: provisionally allocated"
                        ),
                        round_idx=round_idx,
                    )
                )
        else:
            if feedback_rule == "competitive":
                target = _competitive_bundle(
                    bids_by_bidder[bidder_id],
                    provisional.allocation,
                    bids_by_bidder,
                )
                if target is not None:
                    events.append(
                        ElicitationEvent(
                            mechanism=MECHANISM_NAME,
                            event_type="competitive_bundle",
                            bidder_id=bidder_id,
                            bundle=target,
                            allocated_bundle=frozenset(),
                            reason=(
                                f"sealed round {round_idx}: most competitive "
                                "unallocated bundle"
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
                                f"sealed round {round_idx}: lost, refine "
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
) -> list[MechanismResult]:
    """Run a proxy-mediated sealed XOR VCG experiment, recording every round.

    Returns one :class:`MechanismResult` per round, index 0..``config.elicitation_rounds``:

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
                "feedback_rule": config.feedback_rule,
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
            },
        )

        prev_stats = stats_now
        return result

    trajectory = [_record_round(0)]

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
        )

        for event in events:
            bidder_id = event.bidder_id
            if (
                config.max_refinements_per_bidder > 0
                and proxies_by_bidder[bidder_id].stats().refinement_queries
                >= config.max_refinements_per_bidder
            ):
                continue
            if config.max_total_refinements > 0 and (
                sum(
                    p.stats().refinement_queries
                    for p in proxies_by_bidder.values()
                )
                >= config.max_total_refinements
            ):
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

            proxies_by_bidder[bidder_id].receive_provisional_feedback(event)

        trajectory.append(_record_round(round_number))

    return trajectory


def run_proxy_sealed_vcg_experiment(
    instance: AuctionInstance,
    proxies: list[SealedAuctionProxy],
    config: ProxySealedConfig,
) -> MechanismResult:
    """Run a proxy-mediated sealed XOR VCG experiment, returning the final round.

    With ``config.elicitation_rounds == 0`` this reproduces the static
    sealed-proxy baseline: each proxy's initial ``submit_bid()`` is used
    directly. With ``elicitation_rounds > 0``, proxies are given a chance to
    refine their bids in response to provisional-allocation feedback before
    the final auction is run. See :func:`run_proxy_sealed_vcg_trajectory` for
    the full per-round trajectory.
    """
    return run_proxy_sealed_vcg_trajectory(instance, proxies, config)[-1]
