from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

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


@dataclass(frozen=True)
class ExperimentResult:
    instance_name: str
    results: List[MechanismResult]


@dataclass(frozen=True)
class BatchSummary:
    n_instances: int

    sealed_avg_welfare: float
    clock_avg_welfare: float
    clock_avg_efficiency: float

    sealed_avg_revenue: float
    clock_avg_revenue: float

    clock_avg_rounds: float
    sealed_avg_query_count: float
    clock_avg_query_count: float

    allocation_match_rate: float
    welfare_match_rate: float


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


def run_all_mechanisms_on_instance(
    instance: AuctionInstance,
    *,
    instance_name: str = "unnamed_instance",
    clock_cfg: ClockConfig | None = None,
    clock_top_k_values: List[int] | None = None,
) -> ExperimentResult:
    """
    Run all currently implemented mechanisms on one auction instance.
    """
    if clock_cfg is None:
        clock_cfg = ClockConfig()

    if clock_top_k_values is None:
        clock_top_k_values = [1]

    results = [run_sealed_vcg_experiment(instance)]

    for top_k in clock_top_k_values:
        results.append(
            run_clock_supplementary_vcg_experiment(
                instance,
                clock_cfg,
                top_k=top_k,
            )
        )

    return ExperimentResult(
        instance_name=instance_name,
        results=results,
    )


def run_batch_experiments(
    instances: Iterable[tuple[str, AuctionInstance]],
    *,
    clock_cfg: ClockConfig | None = None,
    clock_top_k_values: List[int] | None = None,
) -> List[ExperimentResult]:
    """
    Run all mechanisms on a collection of named instances.
    """
    results: List[ExperimentResult] = []

    for instance_name, instance in instances:
        result = run_all_mechanisms_on_instance(
            instance=instance,
            instance_name=instance_name,
            clock_cfg=clock_cfg,
            clock_top_k_values=clock_top_k_values,
        )
        results.append(result)

    return results


def get_result_by_mechanism(
    result: ExperimentResult,
    mechanism: str,
) -> MechanismResult:
    """
    Fetch one mechanism result from an ExperimentResult.
    """
    for mechanism_result in result.results:
        if mechanism_result.mechanism == mechanism:
            return mechanism_result

    raise KeyError(f"No mechanism result named {mechanism}")


def summarize_batch_results(
    batch_results: List[ExperimentResult],
    *,
    clock_mechanism: str = "clock_supplementary_vcg_top_1",
    welfare_tolerance: float = 1e-6,
) -> BatchSummary:
    """
    Compute aggregate statistics over a batch of experiment results.
    """
    if not batch_results:
        raise ValueError("Cannot summarize an empty batch")

    sealed_results: List[MechanismResult] = []
    clock_results: List[MechanismResult] = []

    for result in batch_results:
        sealed_results.append(get_result_by_mechanism(result, "sealed_xor_vcg"))
        clock_results.append(get_result_by_mechanism(result, clock_mechanism))

    n = len(batch_results)

    sealed_total_welfare = sum(r.welfare for r in sealed_results)
    clock_total_welfare = sum(r.welfare for r in clock_results)

    sealed_total_revenue = sum(r.revenue for r in sealed_results)
    clock_total_revenue = sum(r.revenue for r in clock_results)

    sealed_total_queries = sum(r.query_count for r in sealed_results)
    clock_total_queries = sum(r.query_count for r in clock_results)

    clock_round_values = [
        r.rounds
        for r in clock_results
        if r.rounds is not None
    ]

    allocation_matches = 0
    welfare_matches = 0
    efficiency_values: List[float] = []

    for sealed, clock in zip(sealed_results, clock_results):
        if sealed.allocation == clock.allocation:
            allocation_matches += 1

        if abs(sealed.welfare - clock.welfare) <= welfare_tolerance:
            welfare_matches += 1

        if sealed.welfare > welfare_tolerance:
            efficiency_values.append(clock.welfare / sealed.welfare)
        else:
            efficiency_values.append(1.0)

    return BatchSummary(
        n_instances=n,
        sealed_avg_welfare=sealed_total_welfare / n,
        clock_avg_welfare=clock_total_welfare / n,
        clock_avg_efficiency=sum(efficiency_values) / n,
        sealed_avg_revenue=sealed_total_revenue / n,
        clock_avg_revenue=clock_total_revenue / n,
        clock_avg_rounds=sum(clock_round_values) / len(clock_round_values),
        sealed_avg_query_count=sealed_total_queries / n,
        clock_avg_query_count=clock_total_queries / n,
        allocation_match_rate=allocation_matches / n,
        welfare_match_rate=welfare_matches / n,
    )
