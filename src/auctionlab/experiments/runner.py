from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from auctionlab.auction_types import Bundle
from auctionlab.auctions.clock import (
    ClockConfig,
    finalize_from_supplementary_vcg,
    run_ascending_clock_with_supplementary,
)
from auctionlab.auctions.sealed_vcg import run_sealed_xor_vcg
from auctionlab.instances.base import AuctionInstance, make_demand_oracle


@dataclass(frozen=True)
class MechanismResult:
    mechanism: str
    allocation: Dict[str, Bundle]
    welfare: float
    payments: Dict[str, float]
    revenue: float
    rounds: int | None
    query_count: int
    metadata: Dict[str, Any]


def run_sealed_vcg_experiment(instance: AuctionInstance) -> MechanismResult:
    """
    Run sealed-bid XOR VCG on the full valuation table.
    """
    bids = instance.to_xor_bids()

    outcome = run_sealed_xor_vcg(
        items=instance.items,
        bids=bids,
    )

    return MechanismResult(
        mechanism="sealed_xor_vcg",
        allocation=outcome.allocation,
        welfare=outcome.welfare,
        payments=outcome.payments,
        revenue=sum(outcome.payments.values()),
        rounds=None,
        query_count=len(instance.bidder_ids),
        metadata={
            "vcg_counterfactuals": outcome.vcg_counterfactuals,
        },
    )


def run_clock_supplementary_vcg_experiment(
    instance: AuctionInstance,
    cfg: ClockConfig,
    *,
    top_k: int = 1,
) -> MechanismResult:
    """
    Run ascending clock demand elicitation followed by supplementary XOR VCG.
    """
    demand_oracle = make_demand_oracle(instance, top_k=top_k)

    state, _provisional = run_ascending_clock_with_supplementary(
        items=instance.items,
        bidder_ids=instance.bidder_ids,
        demand_oracle=demand_oracle,
        cfg=cfg,
    )

    outcome = finalize_from_supplementary_vcg(
        items=instance.items,
        supplementary_atoms=state.supplementary,
    )

    return MechanismResult(
        mechanism=f"clock_supplementary_vcg_top_{top_k}",
        allocation=outcome.allocation,
        welfare=outcome.welfare,
        payments=outcome.payments,
        revenue=sum(outcome.payments.values()),
        rounds=len(state.history),
        query_count=len(state.history) * len(instance.bidder_ids),
        metadata={
            "supplementary_atoms": state.supplementary,
            "final_prices": state.prices,
            "top_k": top_k,
            "vcg_counterfactuals": outcome.vcg_counterfactuals,
        },
    )
