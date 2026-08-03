"""Ascending item-price clock with supplementary XOR-bid finalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from auctionlab.auction_types import Bundle, ClockRoundRecord, Item
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.instances.base import DemandResponse
from auctionlab.payments.vcg import compute_vcg_payments_with_witnesses
from auctionlab.solvers.wdp_ilp import WdpResult, solve_wdp_xor_ilp


@dataclass(frozen=True)
class ClockConfig:
    """Price-update and stopping parameters for the clock phase."""

    max_rounds: int = 50
    price_step: float = 5.0
    reserve: float = 0.0


@dataclass
class ClockState:
    """Mutable clock state and the supplementary atoms observed so far."""

    round_idx: int
    prices: Dict[Item, float]
    history: List[ClockRoundRecord]
    supplementary: Dict[str, List[XorAtomicBid]]


@dataclass(frozen=True)
class ClockSupplementaryOutcome:
    """Final WDP allocation and VCG payments over supplementary bids."""

    allocation: Dict[str, Bundle]
    welfare: float
    payments: Dict[str, float]
    vcg_counterfactuals: Dict[str, WdpResult]


DemandOracle = Callable[[str, Dict[Item, float]], DemandResponse]
# primary_bundles (or [primary_bundle] as a fallback) drives excess demand;
# supplementary atoms only enrich the bid set used for final winner
# determination.

def bundle_price(bundle: Bundle, prices: Dict[Item, float]) -> float:
    """Return a bundle's price under linear per-item prices."""
    return sum(prices[item] for item in bundle)


def compute_excess_demand(
    items: List[Item],
    demands: Dict[str, List[Bundle]],
) -> Dict[Item, int]:
    """Return demand minus unit supply for every item."""
    excess_demand = {item: -1 for item in items}

    for bidder_id, demanded_bundles in demands.items():
        demanded_items: set[Item] = set()

        for bundle in demanded_bundles:
            demanded_items.update(bundle)

        for item in demanded_items:
            if item not in excess_demand:
                raise ValueError(f"Demand contains unknown item {item}")
            excess_demand[item] += 1

    return excess_demand


def has_no_excess_demand(excess: Dict[Item, int]) -> bool:
    """Return whether every item's demand is at most its unit supply."""
    return all(v <= 0 for v in excess.values())


def update_prices(
    prices: Dict[Item, float],
    excess: Dict[Item, int],
    price_step: float,
) -> Dict[Item, float]:
    new_prices: Dict[Item, float] = {}

    for item, price in prices.items():
        if excess[item] > 0:
            new_prices[item] = price + price_step
        else:
            new_prices[item] = price

    return new_prices


def record_supplementary_bids(
    state: ClockState,
    supplementary_atoms: Dict[str, List[XorAtomicBid]],
) -> None:
    """Keep each bidder's most recently observed value for each bundle.

    Always overwrites with the latest observation rather than keeping
    whichever value has been highest so far. Proxies report their entire
    current bid as supplementary atoms each round, so a later round's value
    for a bundle is always at least as informed as an earlier one --
    including when refinement legitimately corrects an earlier
    overestimate downward. Keeping the historical maximum would silently
    re-inflate a bundle's value right after it had been corrected.
    """
    for bidder_id, atoms in supplementary_atoms.items():
        if bidder_id not in state.supplementary:
            state.supplementary[bidder_id] = []

        for atom in atoms:
            if not atom.bundle:
                continue

            existing_idx = None
            for idx, old_atom in enumerate(state.supplementary[bidder_id]):
                if old_atom.bundle == atom.bundle:
                    existing_idx = idx
                    break

            if existing_idx is None:
                state.supplementary[bidder_id].append(atom)
            else:
                state.supplementary[bidder_id][existing_idx] = atom


def greedy_provisional_allocation(
    bidder_ids: List[str],
    demands: Dict[str, List[Bundle]],
) -> Dict[str, Bundle]:
    """
    Simple provisional allocation rule.

    For each bidder, try to allocate their first demanded bundle that does not
    conflict with already allocated items.

    This is only a debugging/provisional allocation. The final allocation should
    still come from finalize_from_supplementary_vcg(...).
    """
    allocation: Dict[str, Bundle] = {
        bidder_id: frozenset() for bidder_id in bidder_ids
    }
    allocated_items: set[Item] = set()

    for bidder_id in bidder_ids:
        for bundle in demands.get(bidder_id, []):
            if not bundle:
                continue

            if allocated_items.isdisjoint(bundle):
                allocation[bidder_id] = bundle
                allocated_items.update(bundle)
                break

    return allocation


def run_ascending_clock_with_supplementary(
    items: List[Item],
    bidder_ids: List[str],
    demand_oracle: DemandOracle,
    cfg: ClockConfig,
) -> Tuple[ClockState, Dict[str, Bundle]]:
    """Run the clock until primary demands clear or the round cap is reached.

    Supplementary atoms are accumulated throughout, but only primary bundles
    determine excess demand and price increases.
    """
    prices = {it: cfg.reserve for it in items}

    state = ClockState(
        round_idx=0,
        prices=prices,
        history=[],
        supplementary={bidder_id: [] for bidder_id in bidder_ids},
    )

    provisional_allocation: Dict[str, Bundle] = {
        bidder_id: frozenset() for bidder_id in bidder_ids
    }

    for round_idx in range(cfg.max_rounds):
        state.round_idx = round_idx

        demand_responses: Dict[str, DemandResponse] = {
            bidder_id: demand_oracle(bidder_id, state.prices)
            for bidder_id in bidder_ids
        }
        primary_demands: Dict[str, List[Bundle]] = {}
        supplementary_atoms: Dict[str, List[XorAtomicBid]] = {}

        for bidder_id, response in demand_responses.items():
            normalised_atoms: List[XorAtomicBid] = []
            for atom in response.supplementary_atoms:
                normalised_atoms.append(
                    XorAtomicBid(
                        bundle=frozenset(atom.bundle),
                        value=atom.value,
                    )
                )

            supplementary_atoms[bidder_id] = normalised_atoms

            if response.primary_bundles:
                primary_demands[bidder_id] = [
                    frozenset(bundle) for bundle in response.primary_bundles
                ]
            elif response.primary_bundle is None:
                primary_demands[bidder_id] = []
            else:
                primary_demands[bidder_id] = [
                    frozenset(response.primary_bundle)
                ]

        state.history.append(
            ClockRoundRecord(
                round_idx=round_idx,
                prices=dict(state.prices),
                demands=primary_demands,
            )
        )

        record_supplementary_bids(state, supplementary_atoms)

        excess = compute_excess_demand(items, primary_demands)

        if has_no_excess_demand(excess):
            provisional_allocation = greedy_provisional_allocation(
                bidder_ids=bidder_ids,
                demands=primary_demands,
            )
            break

        state.prices = update_prices(
            prices=state.prices,
            excess=excess,
            price_step=cfg.price_step,
        )
    return state, provisional_allocation


def finalize_from_supplementary_vcg(
    items: List[Item],
    supplementary_atoms: Dict[str, List[XorAtomicBid]],
) -> ClockSupplementaryOutcome:
    """Finalize allocation and VCG payments in the supplementary bid space."""
    bids = [
        XorBid(bidder_id=bidder_id, atoms=atoms)
        for bidder_id, atoms in supplementary_atoms.items()
    ]

    full = solve_wdp_xor_ilp(items, bids)
    payment_result = compute_vcg_payments_with_witnesses(items, bids, full)

    return ClockSupplementaryOutcome(
        allocation=full.allocation,
        welfare=full.welfare,
        payments=payment_result.payments,
        vcg_counterfactuals=payment_result.counterfactuals,
    )
