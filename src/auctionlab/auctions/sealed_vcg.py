from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from auctionlab.auction_types import Item, Bundle
from auctionlab.bids.xor import XorBid
from auctionlab.payments.vcg import compute_vcg_payments
from auctionlab.solvers.wdp_ilp import solve_wdp_xor_ilp


@dataclass(frozen=True)
class SealedVcgOutcome:
    allocation: Dict[str, Bundle]
    welfare: float
    payments: Dict[str, float]


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
    payments = compute_vcg_payments(items, bids, full)

    return SealedVcgOutcome(
        allocation=full.allocation,
        welfare=full.welfare,
        payments=payments,
    )