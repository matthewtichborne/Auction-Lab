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

from auctionlab.auctions.sealed_vcg import run_sealed_xor_vcg
from auctionlab.auction_types import Bundle, Item, validate_bidder_keys
from auctionlab.bids.xor import XorBid
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
    """Configuration for the proxy-mediated sealed elicitation phase."""

    elicitation_rounds: int = 0
    feedback_rule: str = "none"
    # Cap on refinement queries per bidder. 0 means unlimited.
    max_refinements_per_bidder: int = 0

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


def run_proxy_sealed_vcg_experiment(
    instance: AuctionInstance,
    proxies: list[SealedAuctionProxy],
    config: ProxySealedConfig,
) -> MechanismResult:
    """Run a proxy-mediated sealed XOR VCG experiment.

    With ``config.elicitation_rounds == 0`` this reproduces the static
    sealed-proxy baseline: each proxy's initial ``submit_bid()`` is used
    directly. With ``elicitation_rounds > 0``, proxies are given a chance to
    refine their bids in response to provisional-allocation feedback before
    the final auction is run.
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

    refinement_query_count_by_bidder = {
        bidder_id: 0 for bidder_id in instance.bidder_ids
    }

    if config.elicitation_rounds > 0:
        for round_idx in range(config.elicitation_rounds):
            bids_by_bidder: dict[str, XorBid] = {
                bidder_id: proxies_by_bidder[bidder_id].current_bid()
                for bidder_id in instance.bidder_ids
            }

            provisional = solve_wdp_xor_ilp(
                instance.items,
                [bids_by_bidder[bidder_id] for bidder_id in instance.bidder_ids],
            )

            print(
                f"\n  ── sealed round {round_idx + 1}/{config.elicitation_rounds}"
                f"  welfare {provisional.welfare:.0f}",
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
                round_idx=round_idx,
            )

            for event in events:
                bidder_id = event.bidder_id
                if (
                    config.max_refinements_per_bidder > 0
                    and refinement_query_count_by_bidder[bidder_id]
                    >= config.max_refinements_per_bidder
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
                refinement_query_count_by_bidder[bidder_id] = (
                    proxies_by_bidder[bidder_id].stats().refinement_queries
                )

    final_bids: dict[str, XorBid] = {
        bidder_id: proxies_by_bidder[bidder_id].submit_bid()
        for bidder_id in instance.bidder_ids
    }

    refinement_records_by_bidder: dict[str, list[RefinementRecord]] = {
        bidder_id: list(
            getattr(proxy, "refinement_records", lambda: [])()
        )
        for bidder_id, proxy in proxies_by_bidder.items()
    }

    outcome = run_sealed_xor_vcg(
        items=instance.items,
        bids=[final_bids[bidder_id] for bidder_id in instance.bidder_ids],
    )

    if config.elicitation_rounds == 0:
        mechanism = f"{MECHANISM_NAME}_static"
    else:
        mechanism = (
            f"{MECHANISM_NAME}_elicited_{config.feedback_rule}_"
            f"{config.elicitation_rounds}"
        )

    return MechanismResult(
        mechanism=mechanism,
        allocation=outcome.allocation,
        welfare=outcome.welfare,
        payments=outcome.payments,
        revenue=sum(outcome.payments.values()),
        rounds=config.elicitation_rounds or None,
        query_count=sum(refinement_query_count_by_bidder.values()),
        metadata={
            "elicitation_rounds": config.elicitation_rounds,
            "feedback_rule": config.feedback_rule,
            "max_refinements_per_bidder": config.max_refinements_per_bidder,
            "refinement_query_count_by_bidder": (
                refinement_query_count_by_bidder
            ),
            "initial_bids": initial_bids,
            "final_bids": final_bids,
            "refinement_records_by_bidder": refinement_records_by_bidder,
        },
    )
