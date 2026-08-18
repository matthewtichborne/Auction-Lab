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
from typing import Any, Callable

from auctionlab.auction_types import Bundle, Item, validate_bidder_keys
from auctionlab.auctions.clock import (
    ClockConfig,
    ClockState,
    compute_excess_demand,
    finalize_from_supplementary_vcg,
    record_supplementary_bids,
    run_ascending_clock_with_supplementary,
)
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.experiments._trajectory_util import refinement_cap_fields
from auctionlab.experiments.event_policy import (
    best_neighbour_bundle,
    best_scarcity_avoiding_bundle,
    correction_fraction,
)
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
from auctionlab.solvers.wdp_ilp import solve_wdp_xor_ilp


MECHANISM_NAME = "proxy_clock_vcg"
_VALID_TOP_K_FRONTIER_POLICIES = {
    "off",
    "all",
    "allocation_pivotal",
}
_VALID_EVENT_FRAMEWORKS = {
    "legacy",
    "targeted_v1",
    "native_v1",
    "frontier_v1",
}
_VALID_SUPPLEMENTARY_SUPPORT_POLICIES = {"all_atoms", "demand_revealed"}


@dataclass(frozen=True)
class ProxyClockConfig:
    """Configuration for the proxy-mediated clock elicitation phase.

    ``max_refinements_per_bidder`` and ``max_total_refinements`` are safety
    caps, not tuning targets: refinement count should fall out of the
    elicitation events (near-tie/near-zero-surplus/demand-changed) and
    ``top_k``, not be dialled down to shape results. Both default to 0
    (unlimited). Refined
    bundles are already deduplicated per bidder (``state.refined_bundles``),
    so these caps only guard against runaway query volume.
    """

    top_k: int = 1
    elicited: bool = False
    # Both thresholds are in currency units and so are scale-dependent: they
    # decide when two bundles are close enough to count as a near-tie, or a
    # surplus close enough to zero to be treated as marginal. A population
    # with different bundle values would need them revisited.
    margin_threshold: float = 100.0
    tie_threshold: float = 100.0
    # Exact-query each previously unseen bundle when it enters the bidder's
    # positive-surplus top-k demand frontier.
    refine_top_k_frontier: bool = False
    top_k_frontier_policy: str = "off"
    # Audit a bounded winner-removal frontier at the initial reported
    # allocation and whenever exact corrections change that allocation.
    allocation_counterfactual_frontier: bool = False
    # Cap on refinement queries per bidder. 0 means unlimited.
    max_refinements_per_bidder: int = 0
    # Cap on refinement queries summed across all bidders. 0 means unlimited.
    max_total_refinements: int = 0
    priority_scoring: bool = False
    # Exact-query newly selected supplementary-WDP bundles whenever reported
    # refinements change that allocation.
    allocation_change_audit: bool = True
    # Before VCG pricing, exact-query the reported winning allocation and its
    # closest single-winner-removal counterfactuals until no new audit bundle
    # remains. This prevents the last correction from exposing an unaudited
    # allocation and then terminating immediately.
    terminal_stability_audit: bool = True
    # Independently switchable event-policy factors used by the 8x8 ablation.
    incumbent_verification: bool = True
    pivotal_challengers: bool = False
    scarcity_fallbacks: bool = False
    large_correction_followup: bool = False
    correction_followup_threshold: float = 0.25
    gate_near_zero_surplus: bool = False
    terminal_regret_audit: bool = False
    event_framework: str = "legacy"
    demand_switch_verification: bool = False
    contested_bundle_refinement: bool = False
    terminal_winner_verification: bool = True
    terminal_vcg_witness_verification: bool = True
    terminal_best_losing_challenger: bool = False
    # Original clock-native price-path events, independently switchable for
    # the native-v1 factorial ablation.
    native_near_zero_surplus: bool = True
    native_demand_changed: bool = True
    native_near_tie: bool = True
    # Sparse post-clock verification.  The ascending clock runs without
    # exact value queries; its revealed demand path then defines a terminal
    # allocation/challenger frontier.
    frontier_winner_verification: bool = False
    frontier_pivotal_challengers: bool = False
    frontier_winner_closure: bool = False
    frontier_vcg_witness_verification: bool = False
    # Freeze the initial bidder-removal witness set, query it once, and
    # recompute exactly once. This is mutually exclusive with iterative VCG
    # witness closure above.
    frontier_vcg_single_pass: bool = False
    # Restrict the frozen witness set to bundles observed in top-k clock
    # demand at some point on the price path.
    frontier_vcg_revealed_only: bool = False
    # After winner allocation closure, iteratively verify only bidder-removal
    # witnesses observed on the clock demand path.  Any allocation change is
    # closed again before recomputing witnesses.  Termination is structural:
    # every bidder/bundle pair is queried at most once and no numerical query
    # budget is imposed.
    frontier_staged_revealed_vcg_closure: bool = False
    # ``all_atoms`` preserves the historical implementation. Under
    # ``demand_revealed``, only top-k positive-surplus demands observed along
    # the price path enter the mechanism's supplementary bid language.
    #
    # Worth knowing when comparing arms: under ``all_atoms`` the terminal
    # supplementary WDP ranges over the same atom set the sealed auction
    # starts from, so a clock policy limited to winner closure reduces to
    # sealed incumbent verification and the two produce identical results.
    # The mechanisms only diverge once eligibility is restricted to what the
    # price path revealed.
    supplementary_support_policy: str = "all_atoms"

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.margin_threshold < 0.0:
            raise ValueError("margin_threshold must be non-negative")
        if self.tie_threshold < 0.0:
            raise ValueError("tie_threshold must be non-negative")
        if self.top_k_frontier_policy not in _VALID_TOP_K_FRONTIER_POLICIES:
            raise ValueError(
                "top_k_frontier_policy must be one of "
                f"{sorted(_VALID_TOP_K_FRONTIER_POLICIES)}, got "
                f"{self.top_k_frontier_policy!r}"
            )
        if self.event_framework not in _VALID_EVENT_FRAMEWORKS:
            raise ValueError(
                "event_framework must be one of "
                f"{sorted(_VALID_EVENT_FRAMEWORKS)}, got "
                f"{self.event_framework!r}"
            )
        if (
            self.supplementary_support_policy
            not in _VALID_SUPPLEMENTARY_SUPPORT_POLICIES
        ):
            raise ValueError(
                "supplementary_support_policy must be one of "
                f"{sorted(_VALID_SUPPLEMENTARY_SUPPORT_POLICIES)}, got "
                f"{self.supplementary_support_policy!r}"
            )
        if self.max_refinements_per_bidder < 0:
            raise ValueError("max_refinements_per_bidder must be non-negative")
        if self.max_total_refinements < 0:
            raise ValueError("max_total_refinements must be non-negative")
        if not 0.0 <= self.correction_followup_threshold <= 1.0:
            raise ValueError(
                "correction_followup_threshold must lie in [0, 1]"
            )
        if (
            self.frontier_vcg_witness_verification
            and self.frontier_vcg_single_pass
        ):
            raise ValueError(
                "frontier VCG closure and single-pass verification are "
                "mutually exclusive"
            )


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
    seen_top_k_frontier_bundles: set[Bundle] = field(default_factory=set)
    scarcity_fallback_issued: bool = False
    correction_followup_issued: bool = False
    contested_bundle_issued: bool = False
    round_idx: int = 0


@dataclass
class _ClockAuditState:
    previous_allocation: dict[str, Bundle] | None = None
    checked_rounds: set[int] = field(default_factory=set)
    pending_events: dict[tuple[int, str], list[ElicitationEvent]] = field(
        default_factory=dict
    )
    allocation_change_queries: int = 0
    terminal_queries: int = 0
    terminal_iterations: int = 0
    terminal_events: list[ElicitationEvent] = field(default_factory=list)
    counterfactual_frontier_queries: int = 0
    pre_terminal_allocation: dict[str, Bundle] | None = None
    pre_terminal_reported_welfare: float | None = None
    pre_terminal_true_welfare: float | None = None
    pre_terminal_revenue: float | None = None
    post_terminal_allocation: dict[str, Bundle] | None = None
    post_terminal_reported_welfare: float | None = None
    post_terminal_true_welfare: float | None = None
    post_terminal_revenue: float | None = None
    terminal_challenger_issued: bool = False
    revealed_atoms: dict[str, dict[Bundle, XorAtomicBid]] = field(
        default_factory=dict
    )


def _refinement_cap_allows(
    *,
    bidder_id: str,
    proxies_by_bidder: dict[str, ClockAuctionProxy],
    proxy_config: ProxyClockConfig,
) -> bool:
    if (
        proxy_config.max_refinements_per_bidder > 0
        and proxies_by_bidder[bidder_id].stats().refinement_queries
        >= proxy_config.max_refinements_per_bidder
    ):
        return False
    return not (
        proxy_config.max_total_refinements > 0
        and sum(
            proxy.stats().refinement_queries
            for proxy in proxies_by_bidder.values()
        )
        >= proxy_config.max_total_refinements
    )


def _reported_wdp(
    instance: AuctionInstance,
    proxies_by_bidder: dict[str, ClockAuctionProxy],
    support_atoms: dict[str, list[XorAtomicBid]] | None = None,
):
    return solve_wdp_xor_ilp(
        instance.items,
        [
            XorBid(
                bidder_id=bidder_id,
                atoms=list(support_atoms.get(bidder_id, [])),
            )
            if support_atoms is not None
            else proxies_by_bidder[bidder_id].current_bid()
            for bidder_id in instance.bidder_ids
        ],
    )


def _revealed_support_lists(
    audit_state: _ClockAuditState,
) -> dict[str, list[XorAtomicBid]]:
    """Atoms each bidder actually demanded somewhere on the price path.

    This is the restriction that makes the clock cheap. A bundle never
    demanded at any posted price cannot become a terminal witness, however
    high its provisional value, so the price path acts as a free relevance
    filter over the candidate support.
    """
    return {
        bidder_id: list(atoms.values())
        for bidder_id, atoms in audit_state.revealed_atoms.items()
    }


def _current_atom(
    proxy: ClockAuctionProxy, bundle: Bundle
) -> XorAtomicBid | None:
    return next(
        (atom for atom in proxy.current_bid().atoms if atom.bundle == bundle),
        None,
    )


def _true_welfare_for_allocation(
    instance: AuctionInstance,
    allocation: dict[str, Bundle],
) -> float:
    return sum(
        instance.value_of(
            bidder_id,
            allocation.get(bidder_id, frozenset()),
        )
        for bidder_id in instance.bidder_ids
    )


def _winner_removal_frontier(
    *,
    instance: AuctionInstance,
    proxies_by_bidder: dict[str, ClockAuctionProxy],
    outcome,
    remove_entire_bidder: bool = False,
    support_atoms: dict[str, list[XorAtomicBid]] | None = None,
) -> list[tuple[str, Bundle]]:
    """Bundles exposed by removing each winner's atom or entire bid.

    The two modes answer different questions. Removing a single atom asks
    what would win if this bidder could not have *this* bundle, which is the
    allocation-side counterfactual. Removing the bidder entirely gives the
    bidder-removal economy that VCG prices against. Only the second yields
    genuine payment witnesses, which is why the payment-oriented events pass
    ``remove_entire_bidder=True``.
    """
    candidates: list[tuple[str, Bundle]] = []
    seen: set[tuple[str, Bundle]] = set()
    for removed_bidder in instance.bidder_ids:
        removed_bundle = outcome.allocation.get(
            removed_bidder, frozenset()
        )
        if not removed_bundle:
            continue
        counterfactual_bids: list[XorBid] = []
        for bidder_id in instance.bidder_ids:
            if remove_entire_bidder and bidder_id == removed_bidder:
                continue
            atoms = list(
                support_atoms.get(bidder_id, [])
                if support_atoms is not None
                else proxies_by_bidder[bidder_id].current_bid().atoms
            )
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
            bundle = alternative.allocation.get(
                bidder_id, frozenset()
            )
            key = (bidder_id, bundle)
            if bundle and key not in seen:
                seen.add(key)
                candidates.append(key)
    return candidates


def _forced_allocation_gap(
    *,
    instance: AuctionInstance,
    proxies_by_bidder: dict[str, ClockAuctionProxy],
    bidder_id: str,
    bundle: Bundle,
    incumbent_welfare: float | None = None,
    support_atoms: dict[str, list[XorAtomicBid]] | None = None,
) -> float:
    """Reported welfare loss from forcing ``bidder_id`` to receive ``bundle``."""
    bidder_atom = next(
        (
            atom
            for atom in (
                support_atoms.get(bidder_id, [])
                if support_atoms is not None
                else proxies_by_bidder[bidder_id].current_bid().atoms
            )
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
        atoms = [
            atom
            for atom in (
                support_atoms.get(other_id, [])
                if support_atoms is not None
                else proxies_by_bidder[other_id].current_bid().atoms
            )
            if atom.bundle.isdisjoint(bundle)
        ]
        residual_bids.append(XorBid(bidder_id=other_id, atoms=atoms))

    residual = solve_wdp_xor_ilp(instance.items, residual_bids)
    incumbent = (
        incumbent_welfare
        if incumbent_welfare is not None
        else _reported_wdp(
            instance, proxies_by_bidder, support_atoms=support_atoms
        ).welfare
    )
    return incumbent - (bidder_atom.value + residual.welfare)


def _audit_changed_allocation(
    *,
    round_idx: int,
    prices: dict[Item, float],
    instance: AuctionInstance,
    proxies_by_bidder: dict[str, ClockAuctionProxy],
    states: dict[str, _BidderElicitationState],
    proxy_config: ProxyClockConfig,
    audit_state: _ClockAuditState,
) -> None:
    """Refine bundles newly selected after the reported WDP allocation moves."""
    if round_idx in audit_state.checked_rounds:
        return
    audit_state.checked_rounds.add(round_idx)

    revealed_support = (
        _revealed_support_lists(audit_state)
        if proxy_config.supplementary_support_policy == "demand_revealed"
        else None
    )
    outcome = _reported_wdp(
        instance, proxies_by_bidder, support_atoms=revealed_support
    )
    previous = audit_state.previous_allocation
    if previous is None and not proxy_config.allocation_counterfactual_frontier:
        audit_state.previous_allocation = dict(outcome.allocation)
        return
    allocation_changed = (
        previous is None or outcome.allocation != previous
    )
    allocation_before_queries = dict(outcome.allocation)
    if allocation_changed:
        for bidder_id in instance.bidder_ids:
            bundle = outcome.allocation.get(bidder_id, frozenset())
            if (
                not bundle
                or (
                    previous is not None
                    and bundle == previous.get(bidder_id, frozenset())
                )
                or bundle in states[bidder_id].refined_bundles
                or not _refinement_cap_allows(
                    bidder_id=bidder_id,
                    proxies_by_bidder=proxies_by_bidder,
                    proxy_config=proxy_config,
                )
            ):
                continue
            event = ElicitationEvent(
                mechanism=MECHANISM_NAME,
                event_type="allocation_changed_bundle",
                bidder_id=bidder_id,
                bundle=bundle,
                allocated_bundle=bundle,
                prices=dict(prices),
                reason=(
                    f"clock round {round_idx + 1}: supplementary allocation "
                    "changed to this unqueried bundle"
                ),
                round_idx=round_idx,
            )
            states[bidder_id].refined_bundles.add(bundle)
            proxies_by_bidder[bidder_id].refine(event)
            if revealed_support is not None:
                atom = _current_atom(proxies_by_bidder[bidder_id], bundle)
                if atom is not None:
                    audit_state.revealed_atoms.setdefault(
                        bidder_id, {}
                    )[bundle] = atom
            audit_state.pending_events.setdefault(
                (round_idx, bidder_id), []
            ).append(event)
            audit_state.allocation_change_queries += 1
            print(
                f"  {bidder_id:<12}  allocation_changed_bundle  "
                f"{{{','.join(sorted(bundle))}}}",
                flush=True,
            )

        revealed_support = (
            _revealed_support_lists(audit_state)
            if proxy_config.supplementary_support_policy == "demand_revealed"
            else None
        )
        outcome = _reported_wdp(
            instance, proxies_by_bidder, support_atoms=revealed_support
        )
        if proxy_config.allocation_counterfactual_frontier:
            selected_bidders: set[str] = set()
            for bidder_id, bundle in _winner_removal_frontier(
                instance=instance,
                proxies_by_bidder=proxies_by_bidder,
                outcome=outcome,
                support_atoms=revealed_support,
            ):
                if (
                    bidder_id in selected_bidders
                    or bundle in states[bidder_id].refined_bundles
                    or not _refinement_cap_allows(
                        bidder_id=bidder_id,
                        proxies_by_bidder=proxies_by_bidder,
                        proxy_config=proxy_config,
                    )
                ):
                    continue
                selected_bidders.add(bidder_id)
                event = ElicitationEvent(
                    mechanism=MECHANISM_NAME,
                    event_type="allocation_counterfactual",
                    bidder_id=bidder_id,
                    bundle=bundle,
                    allocated_bundle=outcome.allocation.get(
                        bidder_id, frozenset()
                    ),
                    prices=dict(prices),
                    reason=(
                        f"clock round {round_idx + 1}: unqueried bundle "
                        "entered a winner-removal allocation frontier"
                    ),
                    round_idx=round_idx,
                )
                states[bidder_id].refined_bundles.add(bundle)
                proxies_by_bidder[bidder_id].refine(event)
                audit_state.pending_events.setdefault(
                    (round_idx, bidder_id), []
                ).append(event)
                audit_state.counterfactual_frontier_queries += 1
                print(
                    f"  {bidder_id:<12}  allocation_counterfactual  "
                    f"{{{','.join(sorted(bundle))}}}",
                    flush=True,
                )

    # Keep the pre-query allocation as the comparison point. If this round's
    # exact corrections move the WDP, the next round observes that change and
    # audits the newly selected bundles in a bounded cascade.
    audit_state.previous_allocation = allocation_before_queries


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
    instance: AuctionInstance | None = None,
    audit_state: _ClockAuditState | None = None,
):
    _printed_rounds: set[int] = set()
    _round_response_buffer: dict[str, DemandResponse] = {}
    _contested_goods_by_round: dict[int, set[Item]] = {}

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

        if (
            proxy_config.elicited
            and proxy_config.incumbent_verification
            and proxy_config.allocation_change_audit
            and instance is not None
            and audit_state is not None
        ):
            _audit_changed_allocation(
                round_idx=round_idx,
                prices=prices,
                instance=instance,
                proxies_by_bidder=proxies_by_bidder,
                states=states,
                proxy_config=proxy_config,
                audit_state=audit_state,
            )

        ranked: list[tuple[XorAtomicBid, float]] = []
        candidates: list[tuple[Bundle, str, str, float]] = []
        fired_events: list[ElicitationEvent] = (
            audit_state.pending_events.pop((round_idx, bidder_id), [])
            if audit_state is not None
            else []
        )
        current_primary_bundle: Bundle | None = None

        if proxy_config.elicited or on_bidder_round is not None:
            bid = proxy.current_bid()
            ranked = _ranked_surplus_atoms(bid, prices)
            current_primary_bundle = (
                ranked[0][0].bundle if ranked and ranked[0][1] > 0.0 else None
            )

        previous_primary_bundle = state.previous_primary_bundle

        if proxy_config.elicited:
            previous_contested = _contested_goods_by_round.get(
                round_idx - 1, set()
            )
            if proxy_config.event_framework == "targeted_v1":
                if (
                    proxy_config.demand_switch_verification
                    and previous_primary_bundle is not None
                    and previous_primary_bundle != current_primary_bundle
                ):
                    if current_primary_bundle is not None:
                        candidates.append((
                            current_primary_bundle,
                            "demand_switch_entered",
                            (
                                f"clock round {round_idx}: newly entered "
                                "primary demand after a demand switch"
                            ),
                            1.0,
                        ))
                    candidates.append((
                        previous_primary_bundle,
                        "demand_switch_abandoned",
                        (
                            f"clock round {round_idx}: abandoned primary "
                            "demand at its switching boundary"
                        ),
                        1.0,
                    ))

                if (
                    proxy_config.contested_bundle_refinement
                    and not state.contested_bundle_issued
                    and previous_contested
                    and ranked
                ):
                    best_surplus = ranked[0][1]
                    contested_options = [
                        (best_surplus - surplus, atom)
                        for atom, surplus in ranked
                        if surplus > 0.0
                        and atom.bundle != current_primary_bundle
                        and atom.bundle & previous_contested
                        and atom.bundle not in state.refined_bundles
                    ]
                    if contested_options:
                        gap, atom = min(
                            contested_options,
                            key=lambda row: (
                                row[0],
                                -len(row[1].bundle & previous_contested),
                                tuple(sorted(row[1].bundle)),
                            ),
                        )
                        state.contested_bundle_issued = True
                        candidates.append((
                            atom.bundle,
                            "contested_bundle_alternative",
                            (
                                f"clock round {round_idx}: closest unresolved "
                                "positive-surplus alternative involving "
                                f"contested goods {sorted(previous_contested)}; "
                                f"surplus gap={gap:.2f}"
                            ),
                            1.0,
                        ))
            elif proxy_config.event_framework == "frontier_v1":
                # Price discovery is deliberately uninterrupted.  Exact
                # queries are selected from the completed clock path below.
                candidates = []
            else:
                candidates = candidate_refinements(
                    ranked=ranked,
                    current_primary_bundle=current_primary_bundle,
                    previous_primary_bundle=previous_primary_bundle,
                    margin_threshold=proxy_config.margin_threshold,
                    tie_threshold=proxy_config.tie_threshold,
                    round_idx=round_idx,
                )
                if proxy_config.event_framework == "native_v1":
                    enabled_native_events = {
                        "near_zero_surplus": (
                            proxy_config.native_near_zero_surplus
                        ),
                        "demand_changed": proxy_config.native_demand_changed,
                        "near_tie": proxy_config.native_near_tie,
                    }
                    candidates = [
                        candidate
                        for candidate in candidates
                        if enabled_native_events[candidate[1]]
                    ]
                if proxy_config.gate_near_zero_surplus:
                    candidates = [
                        candidate
                        for candidate in candidates
                        if candidate[1] != "near_zero_surplus"
                        or bool(candidate[0] & previous_contested)
                    ]

            if (
                proxy_config.event_framework == "legacy"
                and
                proxy_config.pivotal_challengers
                and instance is not None
                and current_primary_bundle is not None
            ):
                incumbent = _reported_wdp(instance, proxies_by_bidder)
                if incumbent.allocation.get(
                    bidder_id, frozenset()
                ) != current_primary_bundle:
                    pivotal_options: list[tuple[float, Bundle]] = []
                    for atom, surplus in ranked[:3]:
                        if (
                            surplus <= 0.0
                            or atom.bundle in state.refined_bundles
                        ):
                            continue
                        gap = _forced_allocation_gap(
                            instance=instance,
                            proxies_by_bidder=proxies_by_bidder,
                            bidder_id=bidder_id,
                            bundle=atom.bundle,
                            incumbent_welfare=incumbent.welfare,
                        )
                        if abs(gap) <= proxy_config.tie_threshold:
                            pivotal_options.append((abs(gap), atom.bundle))
                    if pivotal_options:
                        gap, bundle = min(
                            pivotal_options,
                            key=lambda row: (
                                row[0], tuple(sorted(row[1]))
                            ),
                        )
                        candidates.append((
                            bundle,
                            "pivotal_challenger",
                            (
                                f"clock round {round_idx}: closest forced-"
                                f"allocation challenger; reported gap={gap:.2f}"
                            ),
                            1.0,
                        ))

            if (
                proxy_config.event_framework == "legacy"
                and
                proxy_config.scarcity_fallbacks
                and not state.scarcity_fallback_issued
                and previous_contested
                and current_primary_bundle is not None
            ):
                fallback = best_scarcity_avoiding_bundle(
                    proxy.current_bid(),
                    current_primary_bundle,
                    previous_contested,
                    excluded=state.refined_bundles,
                )
                if fallback is not None:
                    state.scarcity_fallback_issued = True
                    candidates.append((
                        fallback,
                        "scarcity_avoiding_fallback",
                        (
                            f"clock round {round_idx}: alternative avoids "
                            f"previously contested goods "
                            f"{sorted(previous_contested)}"
                        ),
                        1.0,
                    ))

            frontier_policy = (
                "off"
                if proxy_config.event_framework in {
                    "targeted_v1", "frontier_v1"
                }
                else proxy_config.top_k_frontier_policy
            )
            if (
                frontier_policy == "off"
                and proxy_config.refine_top_k_frontier
            ):
                frontier_policy = "all"

            allocation_gaps: dict[tuple[Bundle, str], float] = {}
            incumbent_welfare: float | None = None
            if frontier_policy == "allocation_pivotal":
                incumbent_welfare = _reported_wdp(
                    instance, proxies_by_bidder
                ).welfare
                filtered_candidates: list[
                    tuple[Bundle, str, str, float]
                ] = []
                for bundle, event_type, reason, score in candidates:
                    if event_type not in {"near_tie", "demand_changed"}:
                        filtered_candidates.append(
                            (bundle, event_type, reason, score)
                        )
                        continue
                    gap = _forced_allocation_gap(
                        instance=instance,
                        proxies_by_bidder=proxies_by_bidder,
                        bidder_id=bidder_id,
                        bundle=bundle,
                        incumbent_welfare=incumbent_welfare,
                    )
                    if gap <= proxy_config.tie_threshold:
                        filtered_candidates.append((
                            bundle,
                            f"allocation_pivotal_{event_type}",
                            (
                                f"{reason}; reported forced-allocation "
                                f"gap={gap:.2f}"
                            ),
                            score,
                        ))
                        allocation_gaps[(
                            bundle,
                            f"allocation_pivotal_{event_type}",
                        )] = gap
                candidates = filtered_candidates

            if frontier_policy in {"all", "allocation_pivotal"}:
                top_k_frontier = [
                    atom.bundle
                    for atom, surplus in ranked[:proxy_config.top_k]
                    if surplus > 0.0
                ]
                for bundle in top_k_frontier:
                    if bundle not in state.seen_top_k_frontier_bundles:
                        gap: float | None = None
                        if frontier_policy == "allocation_pivotal":
                            gap = _forced_allocation_gap(
                                instance=instance,
                                proxies_by_bidder=proxies_by_bidder,
                                bidder_id=bidder_id,
                                bundle=bundle,
                                incumbent_welfare=incumbent_welfare,
                            )
                            if gap > proxy_config.tie_threshold:
                                continue
                            allocation_gaps[(
                                bundle,
                                "allocation_pivotal_top_k_frontier",
                            )] = gap
                        event_type = (
                            "allocation_pivotal_top_k_frontier"
                            if frontier_policy == "allocation_pivotal"
                            else "top_k_frontier"
                        )
                        candidates.append((
                            bundle,
                            event_type,
                            (
                                f"clock round {round_idx}: bundle newly entered "
                                f"positive-surplus top-{proxy_config.top_k} demand"
                                + (
                                    f"; reported forced-allocation gap={gap:.2f}"
                                    if gap is not None
                                    else ""
                                )
                            ),
                            1.0,
                        ))
                state.seen_top_k_frontier_bundles.update(top_k_frontier)

            if proxy_config.priority_scoring:
                candidates = sorted(candidates, key=lambda c: c[3], reverse=True)

            seen: set[Bundle] = set()
            candidate_idx = 0
            while candidate_idx < len(candidates):
                bundle, event_type, reason, _score = candidates[candidate_idx]
                candidate_idx += 1
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
                if (
                    event_type == "near_tie"
                    and current_primary_bundle is not None
                ):
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
                    allocation_gap=allocation_gaps.get(
                        (bundle, event_type)
                    ),
                    reason=reason,
                    round_idx=round_idx,
                )
                records_getter = getattr(
                    proxy, "refinement_records", None
                )
                records_before = (
                    len(records_getter())
                    if callable(records_getter)
                    else 0
                )
                proxy.refine(event)
                fired_events.append(event)
                new_records = (
                    records_getter()[records_before:]
                    if callable(records_getter)
                    else []
                )
                if (
                    proxy_config.large_correction_followup
                    and not state.correction_followup_issued
                    and event_type != "large_correction_followup"
                    and any(
                        correction_fraction(record)
                        >= proxy_config.correction_followup_threshold
                        for record in new_records
                    )
                ):
                    neighbour = best_neighbour_bundle(
                        proxy.current_bid(),
                        new_records[-1].bundle,
                        excluded=state.refined_bundles,
                    )
                    if neighbour is not None:
                        state.correction_followup_issued = True
                        candidates.append((
                            neighbour,
                            "large_correction_followup",
                            (
                                f"clock round {round_idx}: one-step neighbour "
                                "after a large exact-value correction"
                            ),
                            1.0,
                        ))

        response = proxy.demand_at_prices(
            prices,
            round_idx=round_idx,
            top_k=proxy_config.top_k,
        )
        demanded = set(
            response.primary_bundles
            or (
                [response.primary_bundle]
                if response.primary_bundle is not None
                else []
            )
        )
        # Retain the revealed path independently of the supplementary-support
        # policy.  frontier_v1 uses it for query selection even when the final
        # supplementary WDP retains the full provisional candidate language.
        if audit_state is not None:
            revealed = audit_state.revealed_atoms.setdefault(bidder_id, {})
            for atom in response.supplementary_atoms:
                if atom.bundle in demanded:
                    revealed[atom.bundle] = atom
        if proxy_config.supplementary_support_policy == "demand_revealed":
            response = DemandResponse(
                primary_bundle=response.primary_bundle,
                primary_bundles=list(response.primary_bundles),
                supplementary_atoms=[
                    atom
                    for atom in response.supplementary_atoms
                    if atom.bundle in demanded
                ],
            )
        state.previous_primary_bundle = response.primary_bundle
        state.round_idx += 1

        _round_response_buffer[bidder_id] = response
        if (
            instance is not None
            and len(_round_response_buffer) == len(proxies_by_bidder)
        ):
            primary_demands = {
                current_bidder: (
                    current_response.primary_bundles
                    or (
                        [current_response.primary_bundle]
                        if current_response.primary_bundle
                        else []
                    )
                )
                for current_bidder, current_response
                in _round_response_buffer.items()
            }
            excess = compute_excess_demand(instance.items, primary_demands)
            _contested_goods_by_round[round_idx] = {
                item for item, amount in excess.items() if amount > 0
            }
            _round_response_buffer.clear()

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


def _terminal_clock_frontier_audit(
    *,
    instance: AuctionInstance,
    proxies_by_bidder: dict[str, ClockAuctionProxy],
    states: dict[str, _BidderElicitationState],
    proxy_config: ProxyClockConfig,
    state: ClockState,
    audit_state: _ClockAuditState,
) -> None:
    """Verify a sparse frontier discovered by the completed clock path.

    The clock itself makes no exact value queries under ``frontier_v1``.
    Terminal candidates are restricted to bundles actually admitted to the
    supplementary clock support.  We first freeze the reported allocation
    and, optionally, one strongest overlapping losing demand per winner.
    Winner closure then verifies only newly selected allocation bundles.
    VCG witness closure is an explicitly separate, potentially less sparse
    treatment used to measure the additional cost of payment verification.
    """
    enabled = any((
        proxy_config.frontier_winner_verification,
        proxy_config.frontier_pivotal_challengers,
        proxy_config.frontier_winner_closure,
        proxy_config.frontier_vcg_witness_verification,
        proxy_config.frontier_vcg_single_pass,
        proxy_config.frontier_staged_revealed_vcg_closure,
    ))
    if not proxy_config.elicited or not enabled:
        return

    def support() -> dict[str, list[XorAtomicBid]] | None:
        return (
            {
                bidder_id: list(state.supplementary.get(bidder_id, []))
                for bidder_id in instance.bidder_ids
            }
            if proxy_config.supplementary_support_policy == "demand_revealed"
            else None
        )

    pre_terminal = finalize_from_supplementary_vcg(
        items=instance.items,
        supplementary_atoms=state.supplementary,
    )
    audit_state.pre_terminal_allocation = dict(pre_terminal.allocation)
    audit_state.pre_terminal_reported_welfare = pre_terminal.welfare
    audit_state.pre_terminal_true_welfare = _true_welfare_for_allocation(
        instance, pre_terminal.allocation
    )
    audit_state.pre_terminal_revenue = sum(pre_terminal.payments.values())
    round_idx = len(state.history)

    def query(
        bidder_id: str,
        bundle: Bundle,
        event_type: str,
        outcome,
        reason: str,
    ) -> bool:
        if (
            not bundle
            or bundle in states[bidder_id].refined_bundles
            or not _refinement_cap_allows(
                bidder_id=bidder_id,
                proxies_by_bidder=proxies_by_bidder,
                proxy_config=proxy_config,
            )
        ):
            return False
        event = ElicitationEvent(
            mechanism=MECHANISM_NAME,
            event_type=event_type,
            bidder_id=bidder_id,
            bundle=bundle,
            allocated_bundle=outcome.allocation.get(
                bidder_id, frozenset()
            ),
            prices=dict(state.prices),
            reason=reason,
            round_idx=round_idx,
        )
        states[bidder_id].refined_bundles.add(bundle)
        proxies_by_bidder[bidder_id].refine(event)
        atom = _current_atom(proxies_by_bidder[bidder_id], bundle)
        if atom is not None:
            record_supplementary_bids(state, {bidder_id: [atom]})
        audit_state.terminal_queries += 1
        audit_state.terminal_events.append(event)
        print(
            f"  {bidder_id:<12}  {event_type}  "
            f"{{{','.join(sorted(bundle))}}}",
            flush=True,
        )
        return True

    initial_support = support()
    initial_outcome = _reported_wdp(
        instance, proxies_by_bidder, support_atoms=initial_support
    )

    # Freeze challenger selection before exact winner corrections change the
    # ranking.  A candidate must have been revealed on the clock path, overlap
    # the relevant winner, and lie within the configured forced-allocation
    # gap. Recently priced-out demands remain informative challengers.
    challengers: list[tuple[str, Bundle, float]] = []
    if proxy_config.frontier_pivotal_challengers:
        seen: set[tuple[str, Bundle]] = set()
        for winner_id in instance.bidder_ids:
            winning_bundle = initial_outcome.allocation.get(
                winner_id, frozenset()
            )
            if not winning_bundle:
                continue
            options: list[tuple[float, float, str, Bundle]] = []
            for bidder_id in instance.bidder_ids:
                if bidder_id == winner_id:
                    continue
                allocated = initial_outcome.allocation.get(
                    bidder_id, frozenset()
                )
                # Challenger eligibility comes only from demands revealed on
                # the price path, even when the final WDP keeps all candidate
                # atoms as provisional support.
                atoms = audit_state.revealed_atoms.get(
                    bidder_id, {}
                ).values()
                for atom in atoms:
                    terminal_surplus = atom.value - sum(
                        state.prices[item] for item in atom.bundle
                    )
                    if (
                        not atom.bundle
                        or atom.bundle == allocated
                        or not atom.bundle & winning_bundle
                    ):
                        continue
                    gap = _forced_allocation_gap(
                        instance=instance,
                        proxies_by_bidder=proxies_by_bidder,
                        bidder_id=bidder_id,
                        bundle=atom.bundle,
                        incumbent_welfare=initial_outcome.welfare,
                        support_atoms=initial_support,
                    )
                    if gap <= proxy_config.tie_threshold:
                        options.append((
                            gap, -terminal_surplus, bidder_id, atom.bundle
                        ))
            if options:
                gap, _, bidder_id, bundle = min(
                    options,
                    key=lambda row: (
                        row[0], row[1], row[2], tuple(sorted(row[3]))
                    ),
                )
                if (bidder_id, bundle) not in seen:
                    seen.add((bidder_id, bundle))
                    challengers.append((bidder_id, bundle, gap))

    # Freeze VCG candidates from the same pre-query snapshot as winners and
    # pivotal challengers. No exact correction can expose another witness in
    # single-pass mode.
    single_pass_witnesses: list[tuple[str, Bundle]] = []
    if proxy_config.frontier_vcg_single_pass:
        for bidder_id, bundle in _winner_removal_frontier(
            instance=instance,
            proxies_by_bidder=proxies_by_bidder,
            outcome=initial_outcome,
            remove_entire_bidder=True,
            support_atoms=initial_support,
        ):
            if (
                proxy_config.frontier_vcg_revealed_only
                and bundle not in audit_state.revealed_atoms.get(
                    bidder_id, {}
                )
            ):
                continue
            single_pass_witnesses.append((bidder_id, bundle))

    audit_state.terminal_iterations += 1
    if proxy_config.frontier_winner_verification:
        for bidder_id in instance.bidder_ids:
            bundle = initial_outcome.allocation.get(
                bidder_id, frozenset()
            )
            query(
                bidder_id,
                bundle,
                "frontier_winner",
                initial_outcome,
                "winner selected by the terminal clock-revealed support",
            )
    for bidder_id, bundle, gap in challengers:
        query(
            bidder_id,
            bundle,
            "frontier_pivotal_challenger",
            initial_outcome,
            (
                "strongest overlapping losing demand revealed by the clock; "
                f"reported forced-allocation gap={gap:.2f}"
            ),
        )
    for bidder_id, bundle in single_pass_witnesses:
        query(
            bidder_id,
            bundle,
            (
                "frontier_vcg_single_pass_revealed"
                if proxy_config.frontier_vcg_revealed_only
                else "frontier_vcg_single_pass_all"
            ),
            initial_outcome,
            (
                "frozen pre-query bidder-removal VCG witness"
                + (
                    " also observed on the clock demand path"
                    if proxy_config.frontier_vcg_revealed_only
                    else ""
                )
            ),
        )

    max_iterations = 1 + sum(
        len(proxy.current_bid().atoms)
        for proxy in proxies_by_bidder.values()
    )
    if proxy_config.frontier_winner_closure:
        for _ in range(max_iterations):
            audit_state.terminal_iterations += 1
            outcome = _reported_wdp(
                instance, proxies_by_bidder, support_atoms=support()
            )
            queried = sum(
                query(
                    bidder_id,
                    outcome.allocation.get(bidder_id, frozenset()),
                    "frontier_closure_winner",
                    outcome,
                    "newly allocated bundle exposed by exact terminal corrections",
                )
                for bidder_id in instance.bidder_ids
            )
            if queried == 0:
                break

    if proxy_config.frontier_vcg_witness_verification:
        for _ in range(max_iterations):
            audit_state.terminal_iterations += 1
            current_support = support()
            outcome = _reported_wdp(
                instance, proxies_by_bidder, support_atoms=current_support
            )
            queried = 0
            for bidder_id, bundle in _winner_removal_frontier(
                instance=instance,
                proxies_by_bidder=proxies_by_bidder,
                outcome=outcome,
                remove_entire_bidder=True,
                support_atoms=current_support,
            ):
                queried += query(
                    bidder_id,
                    bundle,
                    "frontier_vcg_witness",
                    outcome,
                    "bundle appearing in a clock-allocation VCG counterfactual",
                )
            if proxy_config.frontier_winner_closure:
                refreshed = _reported_wdp(
                    instance, proxies_by_bidder, support_atoms=support()
                )
                for bidder_id in instance.bidder_ids:
                    queried += query(
                        bidder_id,
                        refreshed.allocation.get(
                            bidder_id, frozenset()
                        ),
                        "frontier_closure_winner",
                        refreshed,
                        "new winner exposed during VCG-witness verification",
                    )
            if queried == 0:
                break

    if proxy_config.frontier_staged_revealed_vcg_closure:
        for _ in range(max_iterations):
            audit_state.terminal_iterations += 1
            current_support = support()
            outcome = _reported_wdp(
                instance, proxies_by_bidder, support_atoms=current_support
            )
            queried = 0
            for bidder_id, bundle in _winner_removal_frontier(
                instance=instance,
                proxies_by_bidder=proxies_by_bidder,
                outcome=outcome,
                remove_entire_bidder=True,
                support_atoms=current_support,
            ):
                if bundle not in audit_state.revealed_atoms.get(
                    bidder_id, {}
                ):
                    continue
                queried += query(
                    bidder_id,
                    bundle,
                    "frontier_staged_revealed_vcg_witness",
                    outcome,
                    (
                        "bidder-removal VCG witness observed on the clock "
                        "demand path after winner closure"
                    ),
                )

            # Witness corrections can change the allocation.  Close every
            # newly selected bundle before recomputing the revealed witness
            # frontier; query() provides global bidder/bundle deduplication.
            refreshed = _reported_wdp(
                instance, proxies_by_bidder, support_atoms=support()
            )
            for bidder_id in instance.bidder_ids:
                queried += query(
                    bidder_id,
                    refreshed.allocation.get(bidder_id, frozenset()),
                    "frontier_closure_winner",
                    refreshed,
                    "new winner exposed during revealed VCG-witness closure",
                )
            if queried == 0:
                break

    post_terminal = finalize_from_supplementary_vcg(
        items=instance.items,
        supplementary_atoms=state.supplementary,
    )
    audit_state.post_terminal_allocation = dict(post_terminal.allocation)
    audit_state.post_terminal_reported_welfare = post_terminal.welfare
    audit_state.post_terminal_true_welfare = _true_welfare_for_allocation(
        instance, post_terminal.allocation
    )
    audit_state.post_terminal_revenue = sum(post_terminal.payments.values())
    audit_state.previous_allocation = dict(post_terminal.allocation)


def _terminal_stability_audit(
    *,
    instance: AuctionInstance,
    proxies_by_bidder: dict[str, ClockAuctionProxy],
    states: dict[str, _BidderElicitationState],
    proxy_config: ProxyClockConfig,
    state: ClockState,
    audit_state: _ClockAuditState,
) -> None:
    """Audit final winners and their nearest WDP counterfactuals to stability.

    The clock can terminate in the same round that an exact correction changes
    the supplementary allocation. Finalizing immediately would price a newly
    exposed, still-provisional alternative. We therefore query the current
    winners plus the allocation obtained when each winning atom is removed,
    then re-solve. The loop ends only when this finite candidate frontier
    contains no unqueried bundle (or a configured safety cap binds).
    """
    if proxy_config.event_framework == "frontier_v1":
        _terminal_clock_frontier_audit(
            instance=instance,
            proxies_by_bidder=proxies_by_bidder,
            states=states,
            proxy_config=proxy_config,
            state=state,
            audit_state=audit_state,
        )
        return

    targeted = proxy_config.event_framework == "targeted_v1"
    winner_verification = (
        proxy_config.terminal_winner_verification
        if targeted
        else proxy_config.terminal_stability_audit
    )
    witness_verification = (
        proxy_config.terminal_vcg_witness_verification
        if targeted
        else proxy_config.terminal_stability_audit
    )
    challenger_verification = (
        proxy_config.terminal_best_losing_challenger
        if targeted
        else proxy_config.terminal_regret_audit
    )
    if not proxy_config.elicited or not (
        winner_verification
        or witness_verification
        or challenger_verification
    ):
        return

    pre_terminal = finalize_from_supplementary_vcg(
        items=instance.items,
        supplementary_atoms=state.supplementary,
    )
    audit_state.pre_terminal_allocation = dict(pre_terminal.allocation)
    audit_state.pre_terminal_reported_welfare = pre_terminal.welfare
    audit_state.pre_terminal_true_welfare = _true_welfare_for_allocation(
        instance, pre_terminal.allocation
    )
    audit_state.pre_terminal_revenue = sum(pre_terminal.payments.values())

    max_iterations = 1 + sum(
        len(proxy.current_bid().atoms)
        for proxy in proxies_by_bidder.values()
    )
    round_idx = len(state.history)

    for _ in range(max_iterations):
        audit_state.terminal_iterations += 1
        revealed_support = (
            {
                bidder_id: list(state.supplementary.get(bidder_id, []))
                for bidder_id in instance.bidder_ids
            }
            if proxy_config.supplementary_support_policy == "demand_revealed"
            else None
        )
        outcome = _reported_wdp(
            instance, proxies_by_bidder, support_atoms=revealed_support
        )
        candidates: list[tuple[str, Bundle, str]] = []

        if winner_verification:
            for bidder_id in instance.bidder_ids:
                bundle = outcome.allocation.get(bidder_id, frozenset())
                if bundle:
                    candidates.append(
                        (bidder_id, bundle, "terminal_winner")
                    )

        if witness_verification:
            for bidder_id, bundle in _winner_removal_frontier(
                instance=instance,
                proxies_by_bidder=proxies_by_bidder,
                outcome=outcome,
                remove_entire_bidder=targeted,
                support_atoms=revealed_support,
            ):
                candidates.append(
                    (
                        bidder_id,
                        bundle,
                        (
                            "terminal_vcg_witness"
                            if targeted
                            else "terminal_counterfactual"
                        ),
                    )
                )

        if (
            challenger_verification
            and (
                not targeted
                or not audit_state.terminal_challenger_issued
            )
        ):
            regret_candidates: list[tuple[float, str, Bundle]] = []
            for bidder_id in instance.bidder_ids:
                incumbent_bundle = outcome.allocation.get(
                    bidder_id, frozenset()
                )
                bid = proxies_by_bidder[bidder_id].current_bid()
                atoms = sorted(
                    (
                        atom for atom in bid.atoms
                        if atom.bundle
                        and atom.value > 0.0
                        and atom.bundle != incumbent_bundle
                        and atom.bundle
                        not in states[bidder_id].refined_bundles
                    ),
                    key=lambda atom: (
                        -atom.value,
                        len(atom.bundle),
                        tuple(sorted(atom.bundle)),
                    ),
                )[:3]
                for atom in atoms:
                    gap = _forced_allocation_gap(
                        instance=instance,
                        proxies_by_bidder=proxies_by_bidder,
                        bidder_id=bidder_id,
                        bundle=atom.bundle,
                        incumbent_welfare=outcome.welfare,
                    )
                    if abs(gap) <= proxy_config.tie_threshold:
                        regret_candidates.append(
                            (abs(gap), bidder_id, atom.bundle)
                        )
            if regret_candidates:
                _, bidder_id, bundle = min(
                    regret_candidates,
                    key=lambda row: (
                        row[0], row[1], tuple(sorted(row[2]))
                    ),
                )
                candidates.append(
                    (
                        bidder_id,
                        bundle,
                        (
                            "terminal_best_losing_challenger"
                            if targeted
                            else "terminal_regret_challenger"
                        ),
                    )
                )

        queried = 0
        seen: set[tuple[str, Bundle]] = set()
        for bidder_id, bundle, event_type in candidates:
            key = (bidder_id, bundle)
            if (
                key in seen
                or bundle in states[bidder_id].refined_bundles
                or not _refinement_cap_allows(
                    bidder_id=bidder_id,
                    proxies_by_bidder=proxies_by_bidder,
                    proxy_config=proxy_config,
                )
            ):
                continue
            seen.add(key)
            event = ElicitationEvent(
                mechanism=MECHANISM_NAME,
                event_type=event_type,
                bidder_id=bidder_id,
                bundle=bundle,
                allocated_bundle=outcome.allocation.get(
                    bidder_id, frozenset()
                ),
                prices=dict(state.prices),
                reason=(
                    "terminal supplementary-WDP stability audit before "
                    "allocation and VCG pricing"
                ),
                round_idx=round_idx,
            )
            states[bidder_id].refined_bundles.add(bundle)
            proxies_by_bidder[bidder_id].refine(event)
            if proxy_config.supplementary_support_policy == "demand_revealed":
                atom = _current_atom(proxies_by_bidder[bidder_id], bundle)
                if atom is not None:
                    record_supplementary_bids(
                        state, {bidder_id: [atom]}
                    )
            queried += 1
            audit_state.terminal_queries += 1
            audit_state.terminal_events.append(event)
            if targeted and event_type == "terminal_best_losing_challenger":
                audit_state.terminal_challenger_issued = True
            print(
                f"  {bidder_id:<12}  {event_type}  "
                f"{{{','.join(sorted(bundle))}}}",
                flush=True,
            )

        if queried == 0:
            break

        if proxy_config.supplementary_support_policy == "all_atoms":
            record_supplementary_bids(
                state,
                {
                    bidder_id: list(
                        proxies_by_bidder[bidder_id].current_bid().atoms
                    )
                    for bidder_id in instance.bidder_ids
                },
            )

    post_terminal = finalize_from_supplementary_vcg(
        items=instance.items,
        supplementary_atoms=state.supplementary,
    )
    audit_state.post_terminal_allocation = dict(post_terminal.allocation)
    audit_state.post_terminal_reported_welfare = post_terminal.welfare
    audit_state.post_terminal_true_welfare = _true_welfare_for_allocation(
        instance, post_terminal.allocation
    )
    audit_state.post_terminal_revenue = sum(post_terminal.payments.values())
    audit_state.previous_allocation = dict(post_terminal.allocation)


def _build_clock_mechanism_result(
    *,
    instance: AuctionInstance,
    proxies_by_bidder: dict[str, ClockAuctionProxy],
    proxy_config: ProxyClockConfig,
    state: ClockState,
    initial_bids: dict[str, XorBid],
    audit_state: _ClockAuditState | None = None,
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
            "refine_top_k_frontier": proxy_config.refine_top_k_frontier,
            "top_k_frontier_policy": (
                "all"
                if (
                    proxy_config.refine_top_k_frontier
                    and proxy_config.top_k_frontier_policy == "off"
                )
                else proxy_config.top_k_frontier_policy
            ),
            "allocation_counterfactual_frontier": (
                proxy_config.allocation_counterfactual_frontier
            ),
            "max_refinements_per_bidder": (
                proxy_config.max_refinements_per_bidder
            ),
            "max_total_refinements": proxy_config.max_total_refinements,
            "priority_scoring": proxy_config.priority_scoring,
            "allocation_change_audit": proxy_config.allocation_change_audit,
            "terminal_stability_audit": proxy_config.terminal_stability_audit,
            "incumbent_verification": (
                proxy_config.incumbent_verification
            ),
            "pivotal_challengers": proxy_config.pivotal_challengers,
            "scarcity_fallbacks": proxy_config.scarcity_fallbacks,
            "large_correction_followup": (
                proxy_config.large_correction_followup
            ),
            "correction_followup_threshold": (
                proxy_config.correction_followup_threshold
            ),
            "gate_near_zero_surplus": (
                proxy_config.gate_near_zero_surplus
            ),
            "terminal_regret_audit": (
                proxy_config.terminal_regret_audit
            ),
            "event_framework": proxy_config.event_framework,
            "demand_switch_verification": (
                proxy_config.demand_switch_verification
            ),
            "contested_bundle_refinement": (
                proxy_config.contested_bundle_refinement
            ),
            "terminal_winner_verification": (
                proxy_config.terminal_winner_verification
            ),
            "terminal_vcg_witness_verification": (
                proxy_config.terminal_vcg_witness_verification
            ),
            "terminal_best_losing_challenger": (
                proxy_config.terminal_best_losing_challenger
            ),
            "allocation_change_audit_queries": (
                audit_state.allocation_change_queries if audit_state else 0
            ),
            "allocation_counterfactual_frontier_queries": (
                audit_state.counterfactual_frontier_queries
                if audit_state
                else 0
            ),
            "terminal_stability_audit_queries": (
                audit_state.terminal_queries if audit_state else 0
            ),
            "terminal_stability_audit_iterations": (
                audit_state.terminal_iterations if audit_state else 0
            ),
            "pre_terminal_allocation": (
                audit_state.pre_terminal_allocation if audit_state else None
            ),
            "pre_terminal_reported_welfare": (
                audit_state.pre_terminal_reported_welfare
                if audit_state
                else None
            ),
            "pre_terminal_true_welfare": (
                audit_state.pre_terminal_true_welfare
                if audit_state
                else None
            ),
            "pre_terminal_revenue": (
                audit_state.pre_terminal_revenue if audit_state else None
            ),
            "post_terminal_allocation": (
                audit_state.post_terminal_allocation if audit_state else None
            ),
            "post_terminal_reported_welfare": (
                audit_state.post_terminal_reported_welfare
                if audit_state
                else None
            ),
            "post_terminal_true_welfare": (
                audit_state.post_terminal_true_welfare
                if audit_state
                else None
            ),
            "post_terminal_revenue": (
                audit_state.post_terminal_revenue if audit_state else None
            ),
            "refinement_query_count_by_bidder": (
                refinement_query_count_by_bidder
            ),
            **cap_fields,
            "demand_query_count_by_bidder": demand_query_count_by_bidder,
            "refinement_records_by_bidder": refinement_records_by_bidder,
            "vcg_counterfactuals": outcome.vcg_counterfactuals,
            "initial_bids": initial_bids,
            "final_bids": {
                bidder_id: proxies_by_bidder[bidder_id].current_bid()
                for bidder_id in instance.bidder_ids
            },
            "supplementary_atoms": sum(
                len(atoms) for atoms in state.supplementary.values()
            ),
            "supplementary_bids": {
                bidder_id: list(atoms)
                for bidder_id, atoms in state.supplementary.items()
            },
            "supplementary_support_policy": (
                proxy_config.supplementary_support_policy
            ),
            "native_near_zero_surplus": (
                proxy_config.native_near_zero_surplus
            ),
            "native_demand_changed": proxy_config.native_demand_changed,
            "native_near_tie": proxy_config.native_near_tie,
            "frontier_winner_verification": (
                proxy_config.frontier_winner_verification
            ),
            "frontier_pivotal_challengers": (
                proxy_config.frontier_pivotal_challengers
            ),
            "frontier_winner_closure": (
                proxy_config.frontier_winner_closure
            ),
            "frontier_vcg_witness_verification": (
                proxy_config.frontier_vcg_witness_verification
            ),
            "frontier_vcg_single_pass": (
                proxy_config.frontier_vcg_single_pass
            ),
            "frontier_vcg_revealed_only": (
                proxy_config.frontier_vcg_revealed_only
            ),
            "frontier_staged_revealed_vcg_closure": (
                proxy_config.frontier_staged_revealed_vcg_closure
            ),
            "final_prices": state.prices,
            "clock_history": list(state.history),
        },
    )


def run_proxy_clock_experiment(
    instance: AuctionInstance,
    proxies: list[ClockAuctionProxy],
    clock_config: ClockConfig,
    proxy_config: ProxyClockConfig,
    *,
    scenario_name: str = "",
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

    audit_state = _ClockAuditState()
    demand_oracle = _make_demand_oracle(
        proxies_by_bidder=proxies_by_bidder,
        states=states,
        proxy_config=proxy_config,
        instance=instance,
        audit_state=audit_state,
    )

    state, _provisional = run_ascending_clock_with_supplementary(
        items=instance.items,
        bidder_ids=instance.bidder_ids,
        demand_oracle=demand_oracle,
        cfg=clock_config,
    )
    if proxy_config.supplementary_support_policy == "demand_revealed":
        # Synchronise any exact mid-clock corrections without leaking
        # unrevealed candidate atoms into the mechanism's bid support.
        record_supplementary_bids(
            state, _revealed_support_lists(audit_state)
        )
    _terminal_stability_audit(
        instance=instance,
        proxies_by_bidder=proxies_by_bidder,
        states=states,
        proxy_config=proxy_config,
        state=state,
        audit_state=audit_state,
    )

    return _build_clock_mechanism_result(
        instance=instance,
        proxies_by_bidder=proxies_by_bidder,
        proxy_config=proxy_config,
        state=state,
        initial_bids=initial_bids,
        audit_state=audit_state,
    )
