"""Proxy-mediated ascending-clock experiments.

``run_ascending_clock_with_supplementary`` remains the pure clock engine: it
only knows about a ``demand_oracle`` callable, item prices, and excess
demand. This module builds a ``demand_oracle`` on top of
:class:`~auctionlab.proxies.base.ClockAuctionProxy` proxies and, when
``ProxyClockConfig.elicited`` is set, raises generic
:class:`~auctionlab.proxies.base.ElicitationEvent`\\ s (near-zero surplus,
changed primary demand, near-tie runner-up) and forwards them to
``proxy.refine`` before answering demand for that round. With
``elicited=False`` no events are raised and behavior matches the static
clock baseline.

Final allocation is still produced by
``finalize_from_supplementary_vcg`` (ascending clock + supplementary VCG),
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from auctionlab.auction_types import Bundle, Item, validate_bidder_keys
from auctionlab.auctions.clock import (
    ClockConfig,
    ClockState,
    finalize_from_supplementary_vcg,
    run_ascending_clock_with_supplementary,
)
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.experiments._trajectory_util import refinement_cap_fields
from auctionlab.experiments.runner import MechanismResult
from auctionlab.instances.base import AuctionInstance, DemandResponse, demand_rank_key
from auctionlab.proxies.base import (
    ClockAuctionProxy,
    ElicitationEvent,
    RefinementRecord,
    clone_xor_bid,
)
from auctionlab.proxies.elicitation import (
    candidate_refinements,
    pivotality_score,
)


MECHANISM_NAME = "proxy_clock_vcg"


@dataclass(frozen=True)
class ProxyClockConfig:
    """Configuration for the proxy-mediated clock elicitation phase.

    ``max_refinements_per_bidder`` and ``max_total_refinements`` are safety
    caps, not tuning targets: refinement count should fall out of the
    elicitation events (near-tie/near-zero-surplus/demand-changed) and
    ``top_k``, not be dialled down to shape results. Both default to 0
    (unlimited) -- see ``docs/parameter_tuning_methodology.md``. Refined
    bundles are already deduplicated per bidder (``state.refined_bundles``),
    so these caps only guard against runaway query volume.
    """

    top_k: int = 1
    elicited: bool = False
    margin_threshold: float = 100.0
    tie_threshold: float = 100.0
    # Cap on refinement queries per bidder. 0 means unlimited.
    max_refinements_per_bidder: int = 0
    # Cap on refinement queries summed across all bidders. 0 means unlimited.
    max_total_refinements: int = 0
    priority_scoring: bool = False

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.margin_threshold < 0.0:
            raise ValueError("margin_threshold must be non-negative")
        if self.tie_threshold < 0.0:
            raise ValueError("tie_threshold must be non-negative")
        if self.max_refinements_per_bidder < 0:
            raise ValueError("max_refinements_per_bidder must be non-negative")
        if self.max_total_refinements < 0:
            raise ValueError("max_total_refinements must be non-negative")


def _ranked_surplus_atoms(
    bid: XorBid,
    prices: dict[Item, float],
) -> list[tuple[XorAtomicBid, float]]:
    """Rank every atom of ``bid`` by surplus under ``prices``."""
    ranked = [
        (atom, atom.value - sum(prices[item] for item in atom.bundle))
        for atom in bid.atoms
    ]
    ranked.sort(
        key=lambda ranked_atom: demand_rank_key(
            ranked_atom[0].bundle,
            ranked_atom[0].value,
            prices,
        )
    )
    return ranked


@dataclass
class _BidderElicitationState:
    previous_primary_bundle: Bundle | None = None
    refined_bundles: set[Bundle] = field(default_factory=set)
    round_idx: int = 0


# Module-level aliases kept for test backward compatibility.
_pivotality_score = pivotality_score
_candidate_refinements = candidate_refinements


@dataclass(frozen=True)
class BidderRoundObservation:
    """Everything a diagnostic recorder needs from one bidder's clock call.

    Purely additive: exposes data ``_make_demand_oracle`` already computes
    internally (or would compute for free) so
    :mod:`auctionlab.experiments.proxy_clock_trajectory` can build detailed
    per-round diagnostics without re-implementing (and risking drift from)
    the event-detection / refinement logic below.
    """

    bidder_id: str
    round_idx: int
    prices: dict[Item, float]
    ranked: list[tuple[XorAtomicBid, float]]
    current_primary_bundle: Bundle | None
    previous_primary_bundle: Bundle | None
    candidates: list[tuple[Bundle, str, str, float]]
    fired_events: list[ElicitationEvent]
    response: DemandResponse


OnBidderRound = Callable[[BidderRoundObservation], None]


def _make_demand_oracle(
    *,
    proxies_by_bidder: dict[str, ClockAuctionProxy],
    states: dict[str, _BidderElicitationState],
    proxy_config: ProxyClockConfig,
    on_bidder_round: OnBidderRound | None = None,
):
    _printed_rounds: set[int] = set()

    def demand_oracle(bidder_id: str, prices: dict[Item, float]):
        proxy = proxies_by_bidder[bidder_id]
        state = states[bidder_id]
        round_idx = state.round_idx

        if round_idx not in _printed_rounds:
            _printed_rounds.add(round_idx)
            prices_str = "  ".join(
                f"{k}={v:.0f}" for k, v in sorted(prices.items())
            )
            print(f"\n  ── round {round_idx + 1}  {prices_str}", flush=True)

        ranked: list[tuple[XorAtomicBid, float]] = []
        candidates: list[tuple[Bundle, str, str, float]] = []
        fired_events: list[ElicitationEvent] = []
        current_primary_bundle: Bundle | None = None

        if proxy_config.elicited or on_bidder_round is not None:
            bid = proxy.current_bid()
            ranked = _ranked_surplus_atoms(bid, prices)
            current_primary_bundle = (
                ranked[0][0].bundle if ranked and ranked[0][1] > 0.0 else None
            )

        previous_primary_bundle = state.previous_primary_bundle

        if proxy_config.elicited:
            candidates = candidate_refinements(
                ranked=ranked,
                current_primary_bundle=current_primary_bundle,
                previous_primary_bundle=previous_primary_bundle,
                margin_threshold=proxy_config.margin_threshold,
                tie_threshold=proxy_config.tie_threshold,
                round_idx=round_idx,
            )

            if proxy_config.priority_scoring:
                candidates = sorted(candidates, key=lambda c: c[3], reverse=True)

            seen: set[Bundle] = set()
            for bundle, event_type, reason, _score in candidates:
                if (
                    not bundle
                    or bundle in seen
                    or bundle in state.refined_bundles
                    or (
                        proxy_config.max_refinements_per_bidder > 0
                        and proxy.stats().refinement_queries
                        >= proxy_config.max_refinements_per_bidder
                    )
                    or (
                        proxy_config.max_total_refinements > 0
                        and sum(
                            p.stats().refinement_queries
                            for p in proxies_by_bidder.values()
                        )
                        >= proxy_config.max_total_refinements
                    )
                ):
                    continue

                seen.add(bundle)
                state.refined_bundles.add(bundle)
                # Near-tie events span two bundles: the best and runner-up.
                multi_bundles: tuple[Bundle, ...] | None = None
                if event_type == "near_tie" and current_primary_bundle is not None:
                    multi_bundles = (current_primary_bundle, bundle)
                bundle_str = "{" + ",".join(sorted(bundle)) + "}"
                print(
                    f"  {bidder_id:<12}  {event_type}  {bundle_str}",
                    flush=True,
                )
                event = ElicitationEvent(
                    mechanism=MECHANISM_NAME,
                    event_type=event_type,
                    bidder_id=bidder_id,
                    bundle=bundle,
                    bundles=multi_bundles,
                    prices=dict(prices),
                    reason=reason,
                    round_idx=round_idx,
                )
                proxy.refine(event)
                fired_events.append(event)

        response = proxy.demand_at_prices(
            prices,
            round_idx=round_idx,
            top_k=proxy_config.top_k,
        )
        state.previous_primary_bundle = response.primary_bundle
        state.round_idx += 1

        if on_bidder_round is not None:
            on_bidder_round(
                BidderRoundObservation(
                    bidder_id=bidder_id,
                    round_idx=round_idx,
                    prices=dict(prices),
                    ranked=ranked,
                    current_primary_bundle=current_primary_bundle,
                    previous_primary_bundle=previous_primary_bundle,
                    candidates=candidates,
                    fired_events=fired_events,
                    response=response,
                )
            )

        return response

    return demand_oracle


def _build_clock_mechanism_result(
    *,
    instance: AuctionInstance,
    proxies_by_bidder: dict[str, ClockAuctionProxy],
    proxy_config: ProxyClockConfig,
    state: ClockState,
    initial_bids: dict[str, XorBid],
) -> MechanismResult:
    """Finalize a completed clock run into the standard :class:`MechanismResult`.

    Shared by :func:`run_proxy_clock_experiment` and
    :func:`~auctionlab.experiments.proxy_clock_trajectory.run_proxy_clock_trajectory`
    so both build the exact same final-round result from one clock run,
    instead of the trajectory recorder re-running (and double-billing) the
    clock.
    """
    outcome = finalize_from_supplementary_vcg(
        items=instance.items,
        supplementary_atoms=state.supplementary,
    )

    refinement_query_count_by_bidder = {
        bidder_id: proxies_by_bidder[bidder_id].stats().refinement_queries
        for bidder_id in instance.bidder_ids
    }
    demand_query_count_by_bidder = {
        bidder_id: proxies_by_bidder[bidder_id].stats().demand_queries
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
        max_refinements_per_bidder=proxy_config.max_refinements_per_bidder,
        max_total_refinements=proxy_config.max_total_refinements,
    )

    mode = "elicited" if proxy_config.elicited else "static"
    mechanism = f"{MECHANISM_NAME}_{mode}_top_{proxy_config.top_k}"

    return MechanismResult(
        mechanism=mechanism,
        allocation=outcome.allocation,
        welfare=outcome.welfare,
        payments=outcome.payments,
        revenue=sum(outcome.payments.values()),
        rounds=len(state.history),
        query_count=(
            sum(demand_query_count_by_bidder.values())
            + sum(refinement_query_count_by_bidder.values())
        ),
        metadata={
            "top_k": proxy_config.top_k,
            "elicited": proxy_config.elicited,
            "margin_threshold": proxy_config.margin_threshold,
            "tie_threshold": proxy_config.tie_threshold,
            "max_refinements_per_bidder": (
                proxy_config.max_refinements_per_bidder
            ),
            "max_total_refinements": proxy_config.max_total_refinements,
            "priority_scoring": proxy_config.priority_scoring,
            "refinement_query_count_by_bidder": (
                refinement_query_count_by_bidder
            ),
            **cap_fields,
            "demand_query_count_by_bidder": demand_query_count_by_bidder,
            "refinement_records_by_bidder": refinement_records_by_bidder,
            "initial_bids": initial_bids,
            "final_bids": {
                bidder_id: proxies_by_bidder[bidder_id].current_bid()
                for bidder_id in instance.bidder_ids
            },
            "supplementary_atoms": sum(
                len(atoms) for atoms in state.supplementary.values()
            ),
            "final_prices": state.prices,
        },
    )


def run_proxy_clock_experiment(
    instance: AuctionInstance,
    proxies: list[ClockAuctionProxy],
    clock_config: ClockConfig,
    proxy_config: ProxyClockConfig,
) -> MechanismResult:
    """Run the ascending clock + supplementary VCG over proxy demand.

    With ``proxy_config.elicited=False``, this asks each proxy for demand
    every round with no refinement events, reproducing the static clock
    baseline. With ``elicited=True``, near-zero-surplus, demand-changed, and
    near-tie rounds trigger ``proxy.refine`` calls (bounded by
    ``max_refinements_per_bidder``) before demand is read for that round.
    See :func:`~auctionlab.experiments.proxy_clock_trajectory.run_proxy_clock_trajectory`
    for the full per-round diagnostic trajectory.
    """
    proxies_by_bidder = {proxy.bidder_id: proxy for proxy in proxies}
    validate_bidder_keys(
        bidder_ids=instance.bidder_ids,
        values=proxies_by_bidder,
        label="proxies",
    )

    states = {
        bidder_id: _BidderElicitationState()
        for bidder_id in instance.bidder_ids
    }

    initial_bids: dict[str, XorBid] = {
        bidder_id: clone_xor_bid(proxies_by_bidder[bidder_id].current_bid())
        for bidder_id in instance.bidder_ids
    }

    demand_oracle = _make_demand_oracle(
        proxies_by_bidder=proxies_by_bidder,
        states=states,
        proxy_config=proxy_config,
    )

    state, _provisional = run_ascending_clock_with_supplementary(
        items=instance.items,
        bidder_ids=instance.bidder_ids,
        demand_oracle=demand_oracle,
        cfg=clock_config,
    )

    return _build_clock_mechanism_result(
        instance=instance,
        proxies_by_bidder=proxies_by_bidder,
        proxy_config=proxy_config,
        state=state,
        initial_bids=initial_bids,
    )
