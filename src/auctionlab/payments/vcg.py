"""VCG payment computation for allocations selected by the XOR WDP solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from auctionlab.auction_types import Item, Bundle
from auctionlab.bids.xor import XorBid
from auctionlab.solvers.wdp_ilp import WdpResult, solve_wdp_xor_ilp


@dataclass(frozen=True)
class VcgPaymentResult:
    """VCG payments and the WDP witnesses used to compute them.

    ``counterfactuals[bidder_id]`` is the welfare-maximising reported
    allocation after removing ``bidder_id``.  These WDPs are already required
    by VCG pricing; retaining them adds no winner-determination solves.
    """

    payments: Dict[str, float]
    counterfactuals: Dict[str, WdpResult]


def vcg_witness_count(
    counterfactuals: Dict[str, WdpResult],
    bidder_id: str,
    bundle: Bundle,
) -> int:
    """Count bidder-removal VCG witnesses selecting ``bundle`` for a bidder."""
    if not bundle:
        return 0
    return sum(
        witness.allocation.get(bidder_id, frozenset()) == bundle
        for witness in counterfactuals.values()
    )


def compute_vcg_payments_with_witnesses(
    items: List[Item],
    bids: List[XorBid],
    full: WdpResult | None = None,
    *,
    tolerance: float = 1e-6,
) -> VcgPaymentResult:
    """Compute VCG payments and retain every bidder-removal WDP result."""
    if full is None:
        full = solve_wdp_xor_ilp(items, bids)

    payments: Dict[str, float] = {}
    counterfactuals: Dict[str, WdpResult] = {}

    for bid in bids:
        bidder_id = bid.bidder_id

        bids_excluded = [
            other_bid
            for other_bid in bids
            if other_bid.bidder_id != bidder_id
        ]

        without_i = solve_wdp_xor_ilp(items, bids_excluded)
        counterfactuals[bidder_id] = without_i

        allocated_bundle: Bundle = full.allocation.get(bidder_id, frozenset())
        value_i = bid.value_of(allocated_bundle)

        welfare_of_others_in_full = full.welfare - value_i
        raw_payment = without_i.welfare - welfare_of_others_in_full

        # Tiny negative values can arise from solver tolerances, but a
        # materially negative externality indicates inconsistent WDP results.
        if raw_payment < -tolerance:
            raise ValueError(
                f"Negative VCG payment for {bidder_id}: {raw_payment}"
            )

        payments[bidder_id] = max(raw_payment, 0.0)

    return VcgPaymentResult(
        payments=payments,
        counterfactuals=counterfactuals,
    )


def compute_vcg_payments(
    items: List[Item],
    bids: List[XorBid],
    full: WdpResult | None = None,
    *,
    tolerance: float = 1e-6,
) -> Dict[str, float]:
    """Compute each bidder's externality on the other bidders.

    The payment is ``W_without_i - (W_full - value_i(allocation_i))``.
    A supplied full WDP result avoids solving the same allocation twice.
    Use :func:`compute_vcg_payments_with_witnesses` when the bidder-removal
    allocations are also required for diagnostics.
    """
    return compute_vcg_payments_with_witnesses(
        items,
        bids,
        full,
        tolerance=tolerance,
    ).payments
