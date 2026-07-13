"""VCG payment computation for allocations selected by the XOR WDP solver."""

from __future__ import annotations

from typing import Dict, List

from auctionlab.auction_types import Item, Bundle
from auctionlab.bids.xor import XorBid
from auctionlab.solvers.wdp_ilp import WdpResult, solve_wdp_xor_ilp


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
    """
    if full is None:
        full = solve_wdp_xor_ilp(items, bids)

    payments: Dict[str, float] = {}

    for bid in bids:
        bidder_id = bid.bidder_id

        bids_excluded = [
            other_bid
            for other_bid in bids
            if other_bid.bidder_id != bidder_id
        ]

        welfare_without_i = solve_wdp_xor_ilp(items, bids_excluded).welfare

        allocated_bundle: Bundle = full.allocation.get(bidder_id, frozenset())
        value_i = bid.value_of(allocated_bundle)

        welfare_of_others_in_full = full.welfare - value_i

        raw_payment = welfare_without_i - welfare_of_others_in_full

        # Tiny negative values can arise from solver tolerances, but a
        # materially negative externality indicates inconsistent WDP results.
        if raw_payment < -tolerance:
            raise ValueError(
                f"Negative VCG payment for {bidder_id}: {raw_payment}"
            )

        payments[bidder_id] = max(raw_payment, 0.0)

    return payments
