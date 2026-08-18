"""One-shot sealed XOR auction with VCG payments.

Solves winner determination over the reported bids and prices each winner by
the welfare the remaining bidders forgo. Payments are therefore computed in
reported space: they equal oracle VCG payments only when the bidder-removal
optima are themselves supported by accurate reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from auctionlab.auction_types import Item, Bundle
from auctionlab.bids.xor import XorBid
from auctionlab.payments.vcg import compute_vcg_payments_with_witnesses
from auctionlab.solvers.wdp_ilp import WdpResult, solve_wdp_xor_ilp


@dataclass(frozen=True)
class SealedVcgOutcome:
    allocation: Dict[str, Bundle]
    welfare: float
    payments: Dict[str, float]
    vcg_counterfactuals: Dict[str, WdpResult]


def run_sealed_xor_vcg(
    items: List[Item],
    bids: List[XorBid],
) -> SealedVcgOutcome:
    """
    Sealed-bid combinatorial auction:
      - welfare-max allocation under reported XOR bids
      - VCG payments in reported bid space
    """
    full = solve_wdp_xor_ilp(items, bids)
    payment_result = compute_vcg_payments_with_witnesses(items, bids, full)

    return SealedVcgOutcome(
        allocation=full.allocation,
        welfare=full.welfare,
        payments=payment_result.payments,
        vcg_counterfactuals=payment_result.counterfactuals,
    )
