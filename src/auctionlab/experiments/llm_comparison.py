"""Compare LLM-proxy mechanisms with full-information auction outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from auctionlab.auction_types import Bundle
from auctionlab.auctions.clock import ClockConfig
from auctionlab.bids.xor import XorBid
from auctionlab.experiments._trajectory_util import (
    refinement_cap_fields,
    true_welfare_for_allocation,
)
from auctionlab.experiments.export import allocation_to_str
from auctionlab.experiments.llm_runner import (
    run_clock_llm_proxy_experiment,
    run_sealed_llm_proxy_experiment,
)
from auctionlab.experiments.runner import (
    MechanismResult,
    run_sealed_vcg_experiment,
)
from auctionlab.instances.base import AuctionInstance
from auctionlab.llm.proxies import LlmInferredXorProxy
from auctionlab.proxies.base import RefinementRecord


@dataclass(frozen=True)
class LlmSealedComparisonResult:
    """Full-information and sealed LLM-proxy outcomes for one instance."""

    instance_name: str
    instance: AuctionInstance
    full_info: MechanismResult
    llm_proxy: MechanismResult
    llm_true_welfare: float


@dataclass(frozen=True)
class LlmClockComparisonResult:
    """Full-information and clock LLM-proxy outcomes for one instance."""

    instance_name: str
    instance: AuctionInstance
    full_info: MechanismResult
    clock_llm_proxy: MechanismResult
    clock_llm_true_welfare: float


def allocation_matches(
    a: dict[str, Bundle],
    b: dict[str, Bundle],
) -> bool:
    return a == b


def payment_diagnostic_fields(
    full_info: MechanismResult,
    reported: MechanismResult,
) -> dict[str, Any]:
    """Comparable VCG payment diagnostics for detailed and trajectory CSVs."""
    bidder_ids = set(full_info.payments) | set(reported.payments)
    payment_absolute_error = sum(
        abs(
            reported.payments.get(bidder_id, 0.0)
            - full_info.payments.get(bidder_id, 0.0)
        )
        for bidder_id in bidder_ids
    )
    revenue_absolute_error = abs(reported.revenue - full_info.revenue)
    welfare_denominator = max(full_info.welfare, 1.0)
    return {
        "full_info_payments": ";".join(
            f"{bidder_id}:{full_info.payments.get(bidder_id, 0.0)}"
            for bidder_id in sorted(bidder_ids)
        ),
        "reported_payments": ";".join(
            f"{bidder_id}:{reported.payments.get(bidder_id, 0.0)}"
            for bidder_id in sorted(bidder_ids)
        ),
        "payment_absolute_error": payment_absolute_error,
        "payment_error_over_optimum_welfare": (
            payment_absolute_error / welfare_denominator
        ),
        "revenue_absolute_error": revenue_absolute_error,
        "revenue_absolute_error_over_optimum_welfare": (
            revenue_absolute_error / welfare_denominator
        ),
        "revenue_loss": (
            (full_info.revenue - reported.revenue) / full_info.revenue
            if full_info.revenue > 0
            else ""
        ),
        "revenue_absolute_percentage_error": (
            revenue_absolute_error / full_info.revenue
            if full_info.revenue > 0
            else ""
        ),
    }


def xor_bid_to_str(bid: XorBid) -> str:
    atoms = sorted(
        bid.atoms,
        key=lambda atom: (
            len(atom.bundle),
            tuple(sorted(atom.bundle)),
        ),
    )
    return ";".join(
        f"[{','.join(sorted(atom.bundle))}]:{atom.value}"
        for atom in atoms
    )


def reported_bids_to_str(
    reported_bids: list[XorBid] | dict[str, XorBid],
) -> str:
    if isinstance(reported_bids, dict):
        bids_by_bidder = reported_bids
    else:
        bids_by_bidder = {
            bid.bidder_id: bid
            for bid in reported_bids
        }

    return "|".join(
        f"{bidder_id}={{{xor_bid_to_str(bids_by_bidder[bidder_id])}}}"
        for bidder_id in sorted(bids_by_bidder)
    )


def vcg_counterfactuals_to_str(counterfactuals: dict[str, Any]) -> str:
    """Serialize bidder-removal WDP witnesses for CSV provenance."""
    parts: list[str] = []
    for removed_bidder in sorted(counterfactuals):
        witness = counterfactuals[removed_bidder]
        parts.append(
            f"remove={removed_bidder};welfare={witness.welfare};"
            f"allocation={allocation_to_str(witness.allocation)}"
        )
    return "|".join(parts)


def refinement_record_to_str(record: RefinementRecord) -> str:
    old = "none" if record.old_value is None else record.old_value
    bundle = ",".join(sorted(record.bundle))
    return (
        f"{record.bidder_id}:[{bundle}] {old}->{record.new_value} "
        f"{record.reason or record.event_type}"
    )


def refinement_records_to_str(
    records_by_bidder: dict[str, list[RefinementRecord]],
) -> str:
    return "; ".join(
        refinement_record_to_str(record)
        for bidder_id in sorted(records_by_bidder)
        for record in records_by_bidder[bidder_id]
    )


def run_sealed_llm_comparison(
    *,
    instance: AuctionInstance,
    instance_name: str,
    proxies: dict[str, LlmInferredXorProxy],
    candidate_bundles: list[Bundle] | None = None,
    candidate_bundles_by_bidder: dict[str, list[Bundle]] | None = None,
    discount_inferred: bool = True,
    use_anchor_values: bool = True,
) -> LlmSealedComparisonResult:
    """Run full-information and sealed proxy mechanisms on the same instance."""
    full_info = run_sealed_vcg_experiment(instance)
    llm_proxy = run_sealed_llm_proxy_experiment(
        instance,
        proxies,
        candidate_bundles,
        candidate_bundles_by_bidder=candidate_bundles_by_bidder,
        discount_inferred=discount_inferred,
        use_anchor_values=use_anchor_values,
    )

    return LlmSealedComparisonResult(
        instance_name=instance_name,
        instance=instance,
        full_info=full_info,
        llm_proxy=llm_proxy,
        llm_true_welfare=true_welfare_for_allocation(
            instance,
            llm_proxy.allocation,
        ),
    )


def bidder_counts_to_str(counts_by_bidder: dict[str, int]) -> str:
    return ";".join(
        f"{bidder_id}:{count}"
        for bidder_id, count in sorted(counts_by_bidder.items())
    )


def refinement_cap_row_fields(
    refinement_query_count_by_bidder: dict[str, int],
    *,
    max_refinements_per_bidder: int = 0,
    max_total_refinements: int = 0,
) -> dict[str, Any]:
    """CSV-row-friendly wrapper around :func:`refinement_cap_fields`.

    Identical scalar fields, but ``per_bidder_refinement_queries`` is a
    ``bidder:count;...`` string (matching this module's other bidder-keyed
    columns) instead of a raw dict, so it round-trips cleanly through
    ``csv.DictWriter``.
    """
    fields = refinement_cap_fields(
        refinement_query_count_by_bidder,
        max_refinements_per_bidder=max_refinements_per_bidder,
        max_total_refinements=max_total_refinements,
    )
    fields["per_bidder_refinement_queries"] = bidder_counts_to_str(
        fields["per_bidder_refinement_queries"]
    )
    return fields


def epsilon_by_bidder_to_str(epsilon_by_bidder: dict[str, float]) -> str:
    return ";".join(
        f"{bidder_id}:{epsilon_by_bidder[bidder_id]}"
        for bidder_id in sorted(epsilon_by_bidder)
    )


def sealed_llm_comparison_to_row(
    result: LlmSealedComparisonResult,
    *,
    welfare_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Serialize a sealed comparison, keeping reported and true welfare distinct."""
    full_info = result.full_info
    llm_proxy = result.llm_proxy

    if full_info.welfare > welfare_tolerance:
        efficiency = result.llm_true_welfare / full_info.welfare
    else:
        efficiency = 1.0

    return {
        "instance_name": result.instance_name,
        "full_info_welfare": full_info.welfare,
        "llm_proxy_reported_welfare": llm_proxy.welfare,
        "llm_proxy_true_welfare": result.llm_true_welfare,
        "efficiency": efficiency,
        "full_info_revenue": full_info.revenue,
        "llm_proxy_revenue": llm_proxy.revenue,
        **payment_diagnostic_fields(full_info, llm_proxy),
        "full_info_query_count": full_info.query_count,
        "llm_proxy_query_count": llm_proxy.query_count,
        "allocation_match": allocation_matches(
            full_info.allocation,
            llm_proxy.allocation,
        ),
        "welfare_match": (
            abs(full_info.welfare - result.llm_true_welfare)
            <= welfare_tolerance
        ),
        "full_info_allocation": allocation_to_str(full_info.allocation),
        "llm_proxy_allocation": allocation_to_str(llm_proxy.allocation),
        "full_info_vcg_counterfactuals": vcg_counterfactuals_to_str(
            full_info.metadata["vcg_counterfactuals"]
        ),
        "llm_proxy_vcg_counterfactuals": vcg_counterfactuals_to_str(
            llm_proxy.metadata["vcg_counterfactuals"]
        ),
        "candidate_bundle_count": llm_proxy.metadata[
            "candidate_bundle_count"
        ],
        "discount_inferred": llm_proxy.metadata["discount_inferred"],
        "epsilon_by_bidder": epsilon_by_bidder_to_str(
            llm_proxy.metadata["epsilon_by_bidder"]
        ),
        "llm_proxy_reported_bids": reported_bids_to_str(
            llm_proxy.metadata["reported_bids"]
        ),
    }


def run_clock_llm_comparison(
    *,
    instance: AuctionInstance,
    instance_name: str,
    proxies: dict[str, LlmInferredXorProxy],
    candidate_bundles: list[Bundle] | None = None,
    candidate_bundles_by_bidder: dict[str, list[Bundle]] | None = None,
    cfg: ClockConfig | None = None,
    top_k: int = 1,
    discount_inferred: bool = True,
    use_anchor_values: bool = True,
    elicited: bool = False,
    margin_threshold: float = 100.0,
    tie_threshold: float = 100.0,
    max_refinement_queries_per_bidder: int = 0,
    refinement_strategy: str = "value_query",
) -> LlmClockComparisonResult:
    """Run full-information and clock proxy mechanisms on the same instance."""
    full_info = run_sealed_vcg_experiment(instance)
    clock_llm_proxy = run_clock_llm_proxy_experiment(
        instance,
        proxies,
        candidate_bundles,
        cfg,
        candidate_bundles_by_bidder=candidate_bundles_by_bidder,
        top_k=top_k,
        discount_inferred=discount_inferred,
        use_anchor_values=use_anchor_values,
        elicited=elicited,
        margin_threshold=margin_threshold,
        tie_threshold=tie_threshold,
        max_refinement_queries_per_bidder=(
            max_refinement_queries_per_bidder
        ),
        refinement_strategy=refinement_strategy,
    )

    return LlmClockComparisonResult(
        instance_name=instance_name,
        instance=instance,
        full_info=full_info,
        clock_llm_proxy=clock_llm_proxy,
        clock_llm_true_welfare=true_welfare_for_allocation(
            instance,
            clock_llm_proxy.allocation,
        ),
    )


def clock_llm_comparison_to_row(
    result: LlmClockComparisonResult,
    *,
    welfare_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Serialize a clock comparison with benchmark-valued efficiency."""
    full_info = result.full_info
    clock_llm = result.clock_llm_proxy

    if full_info.welfare > welfare_tolerance:
        efficiency = result.clock_llm_true_welfare / full_info.welfare
    else:
        efficiency = 1.0

    return {
        "instance_name": result.instance_name,
        "full_info_welfare": full_info.welfare,
        "clock_llm_reported_welfare": clock_llm.welfare,
        "clock_llm_true_welfare": result.clock_llm_true_welfare,
        "efficiency": efficiency,
        "full_info_revenue": full_info.revenue,
        "clock_llm_revenue": clock_llm.revenue,
        **payment_diagnostic_fields(full_info, clock_llm),
        "full_info_query_count": full_info.query_count,
        "clock_llm_query_count": clock_llm.query_count,
        "clock_demand_query_count": clock_llm.metadata.get(
            "clock_demand_query_count", ""
        ),
        "clock_rounds": (
            clock_llm.rounds
            if clock_llm.rounds is not None
            else ""
        ),
        "allocation_match": allocation_matches(
            full_info.allocation,
            clock_llm.allocation,
        ),
        "welfare_match": (
            abs(full_info.welfare - result.clock_llm_true_welfare)
            <= welfare_tolerance
        ),
        "full_info_allocation": allocation_to_str(full_info.allocation),
        "clock_llm_allocation": allocation_to_str(clock_llm.allocation),
        "full_info_vcg_counterfactuals": vcg_counterfactuals_to_str(
            full_info.metadata["vcg_counterfactuals"]
        ),
        "clock_llm_vcg_counterfactuals": vcg_counterfactuals_to_str(
            clock_llm.metadata["vcg_counterfactuals"]
        ),
        "candidate_bundle_count": clock_llm.metadata[
            "candidate_bundle_count"
        ],
        "top_k": clock_llm.metadata["top_k"],
        "elicited": clock_llm.metadata["elicited"],
        "margin_threshold": clock_llm.metadata["margin_threshold"],
        "tie_threshold": clock_llm.metadata["tie_threshold"],
        "max_refinement_queries_per_bidder": clock_llm.metadata[
            "max_refinement_queries_per_bidder"
        ],
        "refinement_strategy": clock_llm.metadata["refinement_strategy"],
        "refinement_query_count_by_bidder": ";".join(
            f"{bidder_id}:{count}"
            for bidder_id, count in sorted(
                clock_llm.metadata[
                    "refinement_query_count_by_bidder"
                ].items()
            )
        ),
        **refinement_cap_row_fields(
            clock_llm.metadata["refinement_query_count_by_bidder"],
            max_refinements_per_bidder=clock_llm.metadata[
                "max_refinement_queries_per_bidder"
            ],
        ),
        "discount_inferred": clock_llm.metadata["discount_inferred"],
        "epsilon_by_bidder": epsilon_by_bidder_to_str(
            clock_llm.metadata["epsilon_by_bidder"]
        ),
        "clock_llm_reported_bids": reported_bids_to_str(
            clock_llm.metadata["reported_bids"]
        ),
    }


def run_batch_sealed_llm_comparisons(
    comparison_inputs: list[
        tuple[
            str,
            AuctionInstance,
            dict[str, LlmInferredXorProxy],
            list[Bundle],
        ]
    ],
    *,
    discount_inferred: bool = True,
) -> list[LlmSealedComparisonResult]:
    return [
        run_sealed_llm_comparison(
            instance=instance,
            instance_name=instance_name,
            proxies=proxies,
            candidate_bundles=candidate_bundles,
            discount_inferred=discount_inferred,
        )
        for instance_name, instance, proxies, candidate_bundles
        in comparison_inputs
    ]


def sealed_llm_comparisons_to_rows(
    results: list[LlmSealedComparisonResult],
) -> list[dict[str, Any]]:
    return [
        sealed_llm_comparison_to_row(result)
        for result in results
    ]


def proxy_sealed_result_to_row(
    *,
    instance_name: str,
    instance: AuctionInstance,
    result: MechanismResult,
    welfare_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Serialize a :class:`MechanismResult` from ``run_proxy_sealed_vcg_experiment``."""
    full_info = run_sealed_vcg_experiment(instance)
    true_welfare = true_welfare_for_allocation(instance, result.allocation)

    if full_info.welfare > welfare_tolerance:
        efficiency = true_welfare / full_info.welfare
    else:
        efficiency = 1.0

    return {
        "instance_name": instance_name,
        "mechanism": result.mechanism,
        "full_info_welfare": full_info.welfare,
        "proxy_reported_welfare": result.welfare,
        "proxy_true_welfare": true_welfare,
        "efficiency": efficiency,
        "full_info_revenue": full_info.revenue,
        "proxy_revenue": result.revenue,
        **payment_diagnostic_fields(full_info, result),
        "allocation_match": allocation_matches(
            full_info.allocation,
            result.allocation,
        ),
        "welfare_match": (
            abs(full_info.welfare - true_welfare) <= welfare_tolerance
        ),
        "full_info_allocation": allocation_to_str(full_info.allocation),
        "proxy_allocation": allocation_to_str(result.allocation),
        "full_info_vcg_counterfactuals": vcg_counterfactuals_to_str(
            full_info.metadata["vcg_counterfactuals"]
        ),
        "proxy_vcg_counterfactuals": vcg_counterfactuals_to_str(
            result.metadata["vcg_counterfactuals"]
        ),
        "elicitation_rounds": result.metadata["elicitation_rounds"],
        "requested_elicitation_rounds": result.metadata.get(
            "requested_elicitation_rounds",
            result.metadata["elicitation_rounds"],
        ),
        "feedback_rule": result.metadata["feedback_rule"],
        "loser_challenger_policy": result.metadata.get(
            "loser_challenger_policy", "off"
        ),
        "stopping_rule": result.metadata.get("stopping_rule", "fixed_rounds"),
        "incumbent_verification": result.metadata.get(
            "incumbent_verification", True
        ),
        "pivotal_challengers": result.metadata.get(
            "pivotal_challengers", False
        ),
        "scarcity_fallbacks": result.metadata.get(
            "scarcity_fallbacks", False
        ),
        "large_correction_followup": result.metadata.get(
            "large_correction_followup", False
        ),
        "correction_followup_threshold": result.metadata.get(
            "correction_followup_threshold", 0.25
        ),
        "terminal_regret_audit": result.metadata.get(
            "terminal_regret_audit", False
        ),
        "termination_reason": result.metadata.get("termination_reason", ""),
        "events_blocked_by_refinement_cap": result.metadata.get(
            "events_blocked_by_refinement_cap", 0
        ),
        "max_refinements_per_bidder": result.metadata[
            "max_refinements_per_bidder"
        ],
        "max_total_refinements": result.metadata.get("max_total_refinements", 0),
        "refinement_query_count_by_bidder": ";".join(
            f"{bidder_id}:{count}"
            for bidder_id, count in sorted(
                result.metadata["refinement_query_count_by_bidder"].items()
            )
        ),
        **refinement_cap_row_fields(
            result.metadata["refinement_query_count_by_bidder"],
            max_refinements_per_bidder=result.metadata[
                "max_refinements_per_bidder"
            ],
            max_total_refinements=result.metadata.get(
                "max_total_refinements", 0
            ),
        ),
        "initial_bids": reported_bids_to_str(result.metadata["initial_bids"]),
        "final_bids": reported_bids_to_str(result.metadata["final_bids"]),
        "initial_reported_bids": reported_bids_to_str(
            result.metadata["initial_bids"]
        ),
        "final_reported_bids": reported_bids_to_str(
            result.metadata["final_bids"]
        ),
        "refinement_records": refinement_records_to_str(
            result.metadata["refinement_records_by_bidder"]
        ),
    }


def proxy_sealed_trajectory_to_rows(
    *,
    scenario_name: str,
    scenario_seed: Any,
    num_goods: int,
    num_bidders: int,
    instance: AuctionInstance,
    trajectory: list[MechanismResult],
    comparison_welfare: float | None = None,
    welfare_tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    """Serialize a :func:`run_proxy_sealed_vcg_trajectory` result to CSV rows.

    One row per round (0..R). ``comparison_welfare``, if given, is another
    mechanism's true welfare on the same instance (e.g. the plain sealed LLM
    comparison arm); it drives ``mechanism_relative_efficiency``. Without it,
    that column is left blank.
    """
    full_info = run_sealed_vcg_experiment(instance)

    rows: list[dict[str, Any]] = []
    prev_true_welfare: float | None = None
    prev_allocation: dict[str, Bundle] | None = None

    for result in trajectory:
        true_welfare = true_welfare_for_allocation(instance, result.allocation)

        if full_info.welfare > welfare_tolerance:
            global_efficiency = true_welfare / full_info.welfare
        else:
            global_efficiency = 1.0

        mechanism_relative_efficiency: float | str = ""
        if comparison_welfare is not None and comparison_welfare > welfare_tolerance:
            mechanism_relative_efficiency = true_welfare / comparison_welfare

        allocation_changed = (
            False
            if prev_allocation is None
            else not allocation_matches(result.allocation, prev_allocation)
        )
        welfare_delta = (
            0.0 if prev_true_welfare is None else true_welfare - prev_true_welfare
        )

        rows.append({
            "scenario": scenario_name,
            "scenario_seed": scenario_seed,
            "num_goods": num_goods,
            "num_bidders": num_bidders,
            "feedback_rule": result.metadata["feedback_rule"],
            "loser_challenger_policy": result.metadata.get(
                "loser_challenger_policy", "off"
            ),
            "requested_elicitation_rounds": result.metadata.get(
                "requested_elicitation_rounds",
                result.metadata["elicitation_rounds"],
            ),
            "stopping_rule": result.metadata.get(
                "stopping_rule", "fixed_rounds"
            ),
            "incumbent_verification": result.metadata.get(
                "incumbent_verification", True
            ),
            "pivotal_challengers": result.metadata.get(
                "pivotal_challengers", False
            ),
            "scarcity_fallbacks": result.metadata.get(
                "scarcity_fallbacks", False
            ),
            "large_correction_followup": result.metadata.get(
                "large_correction_followup", False
            ),
            "terminal_regret_audit": result.metadata.get(
                "terminal_regret_audit", False
            ),
            "termination_reason": result.metadata.get(
                "termination_reason", ""
            ),
            "events_blocked_by_refinement_cap": result.metadata.get(
                "events_blocked_by_refinement_cap", 0
            ),
            "max_refinement_queries_per_bidder": result.metadata[
                "max_refinements_per_bidder"
            ],
            "round": result.metadata["elicitation_rounds"],
            "true_welfare": true_welfare,
            "reported_welfare": result.welfare,
            "full_info_welfare": full_info.welfare,
            "global_efficiency": global_efficiency,
            "mechanism_relative_efficiency": mechanism_relative_efficiency,
            "revenue": result.revenue,
            **payment_diagnostic_fields(full_info, result),
            "true_surplus": true_welfare - result.revenue,
            "allocation_repr": allocation_to_str(result.allocation),
            "allocation_changed_from_previous_round": allocation_changed,
            "welfare_delta_from_previous_round": welfare_delta,
            "new_value_queries": result.metadata["new_value_queries"],
            "cumulative_value_queries": result.metadata["cumulative_value_queries"],
            "new_demand_queries": result.metadata["new_demand_queries"],
            "cumulative_demand_queries": result.metadata["cumulative_demand_queries"],
            "new_nl_queries": result.metadata["new_nl_queries"],
            "cumulative_nl_queries": result.metadata["cumulative_nl_queries"],
            "new_tokens_in": result.metadata["new_tokens_in"],
            "cumulative_tokens_in": result.metadata["tokens_in"],
            "new_tokens_out": result.metadata["new_tokens_out"],
            "cumulative_tokens_out": result.metadata["tokens_out"],
            "num_refinements_this_round": sum(
                result.metadata["new_refinement_query_count_by_bidder"].values()
            ),
            "cumulative_refinements": sum(
                result.metadata["refinement_query_count_by_bidder"].values()
            ),
            "max_total_refinements": result.metadata.get(
                "max_total_refinements", 0
            ),
            **refinement_cap_row_fields(
                result.metadata["refinement_query_count_by_bidder"],
                max_refinements_per_bidder=result.metadata[
                    "max_refinements_per_bidder"
                ],
                max_total_refinements=result.metadata.get(
                    "max_total_refinements", 0
                ),
            ),
        })

        prev_true_welfare = true_welfare
        prev_allocation = result.allocation

    return rows


def proxy_clock_result_to_row(
    *,
    instance_name: str,
    instance: AuctionInstance,
    result: MechanismResult,
    welfare_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Serialize a :class:`MechanismResult` from ``run_proxy_clock_experiment``."""
    full_info = run_sealed_vcg_experiment(instance)
    true_welfare = true_welfare_for_allocation(instance, result.allocation)

    if full_info.welfare > welfare_tolerance:
        efficiency = true_welfare / full_info.welfare
    else:
        efficiency = 1.0
    pre_terminal_revenue = result.metadata.get("pre_terminal_revenue")
    terminal_revenue_abs_error_improvement: float | str = ""
    if pre_terminal_revenue is not None:
        terminal_revenue_abs_error_improvement = (
            abs(pre_terminal_revenue - full_info.revenue)
            - abs(result.revenue - full_info.revenue)
        )

    return {
        "instance_name": instance_name,
        "mechanism": result.mechanism,
        "full_info_welfare": full_info.welfare,
        "proxy_reported_welfare": result.welfare,
        "proxy_true_welfare": true_welfare,
        "efficiency": efficiency,
        "full_info_revenue": full_info.revenue,
        "proxy_revenue": result.revenue,
        **payment_diagnostic_fields(full_info, result),
        "rounds": result.rounds if result.rounds is not None else "",
        "allocation_match": allocation_matches(
            full_info.allocation,
            result.allocation,
        ),
        "welfare_match": (
            abs(full_info.welfare - true_welfare) <= welfare_tolerance
        ),
        "full_info_allocation": allocation_to_str(full_info.allocation),
        "proxy_allocation": allocation_to_str(result.allocation),
        "full_info_vcg_counterfactuals": vcg_counterfactuals_to_str(
            full_info.metadata["vcg_counterfactuals"]
        ),
        "proxy_vcg_counterfactuals": vcg_counterfactuals_to_str(
            result.metadata["vcg_counterfactuals"]
        ),
        "top_k": result.metadata["top_k"],
        "elicited": result.metadata["elicited"],
        "supplementary_support_policy": result.metadata.get(
            "supplementary_support_policy", "all_atoms"
        ),
        "margin_threshold": result.metadata["margin_threshold"],
        "tie_threshold": result.metadata["tie_threshold"],
        "refine_top_k_frontier": result.metadata.get(
            "refine_top_k_frontier", False
        ),
        "top_k_frontier_policy": result.metadata.get(
            "top_k_frontier_policy", "off"
        ),
        "allocation_counterfactual_frontier": result.metadata.get(
            "allocation_counterfactual_frontier", False
        ),
        "allocation_change_audit": result.metadata.get(
            "allocation_change_audit", False
        ),
        "terminal_stability_audit": result.metadata.get(
            "terminal_stability_audit", False
        ),
        "incumbent_verification": result.metadata.get(
            "incumbent_verification", True
        ),
        "pivotal_challengers": result.metadata.get(
            "pivotal_challengers", False
        ),
        "scarcity_fallbacks": result.metadata.get(
            "scarcity_fallbacks", False
        ),
        "large_correction_followup": result.metadata.get(
            "large_correction_followup", False
        ),
        "correction_followup_threshold": result.metadata.get(
            "correction_followup_threshold", 0.25
        ),
        "gate_near_zero_surplus": result.metadata.get(
            "gate_near_zero_surplus", False
        ),
        "terminal_regret_audit": result.metadata.get(
            "terminal_regret_audit", False
        ),
        "allocation_change_audit_queries": result.metadata.get(
            "allocation_change_audit_queries", 0
        ),
        "allocation_counterfactual_frontier_queries": result.metadata.get(
            "allocation_counterfactual_frontier_queries", 0
        ),
        "terminal_stability_audit_queries": result.metadata.get(
            "terminal_stability_audit_queries", 0
        ),
        "terminal_stability_audit_iterations": result.metadata.get(
            "terminal_stability_audit_iterations", 0
        ),
        "pre_terminal_allocation": (
            allocation_to_str(result.metadata["pre_terminal_allocation"])
            if result.metadata.get("pre_terminal_allocation") is not None
            else ""
        ),
        "pre_terminal_reported_welfare": result.metadata.get(
            "pre_terminal_reported_welfare", ""
        ),
        "pre_terminal_true_welfare": result.metadata.get(
            "pre_terminal_true_welfare", ""
        ),
        "pre_terminal_revenue": (
            "" if pre_terminal_revenue is None else pre_terminal_revenue
        ),
        "post_terminal_allocation": (
            allocation_to_str(result.metadata["post_terminal_allocation"])
            if result.metadata.get("post_terminal_allocation") is not None
            else ""
        ),
        "post_terminal_reported_welfare": result.metadata.get(
            "post_terminal_reported_welfare", ""
        ),
        "post_terminal_true_welfare": result.metadata.get(
            "post_terminal_true_welfare", ""
        ),
        "post_terminal_revenue": result.metadata.get(
            "post_terminal_revenue", ""
        ),
        "terminal_revenue_abs_error_improvement": (
            terminal_revenue_abs_error_improvement
        ),
        "max_refinements_per_bidder": result.metadata[
            "max_refinements_per_bidder"
        ],
        "max_total_refinements": result.metadata.get("max_total_refinements", 0),
        "refinement_query_count_by_bidder": ";".join(
            f"{bidder_id}:{count}"
            for bidder_id, count in sorted(
                result.metadata["refinement_query_count_by_bidder"].items()
            )
        ),
        **refinement_cap_row_fields(
            result.metadata["refinement_query_count_by_bidder"],
            max_refinements_per_bidder=result.metadata[
                "max_refinements_per_bidder"
            ],
            max_total_refinements=result.metadata.get(
                "max_total_refinements", 0
            ),
        ),
        "initial_bids": reported_bids_to_str(result.metadata["initial_bids"]),
        "final_bids": reported_bids_to_str(result.metadata["final_bids"]),
        "initial_reported_bids": reported_bids_to_str(
            result.metadata["initial_bids"]
        ),
        "final_reported_bids": reported_bids_to_str(
            result.metadata["final_bids"]
        ),
        "refinement_records": refinement_records_to_str(
            result.metadata["refinement_records_by_bidder"]
        ),
        "final_prices": result.metadata["final_prices"],
        "supplementary_atoms_total": result.metadata.get("supplementary_atoms", ""),
        "failure_classification": result.metadata.get("failure_classification", ""),
        # Best-vs-final round diagnostics (only populated for trajectory
        # runs -- see run_proxy_clock_trajectory._compute_best_final_round_diagnostics).
        "final_round": result.metadata.get("final_round", ""),
        "termination_reason": result.metadata.get("termination_reason", ""),
        "final_true_welfare": result.metadata.get("final_true_welfare", ""),
        "final_reported_welfare": result.metadata.get("final_reported_welfare", ""),
        "final_true_efficiency": result.metadata.get("final_true_efficiency", ""),
        "final_true_surplus": result.metadata.get("final_true_surplus", ""),
        "final_revenue": result.metadata.get("final_revenue", ""),
        "best_round": result.metadata.get("best_round", ""),
        "best_true_welfare": result.metadata.get("best_true_welfare", ""),
        "best_true_efficiency": result.metadata.get("best_true_efficiency", ""),
        "best_reported_welfare_at_best_round": result.metadata.get(
            "best_reported_welfare_at_best_round", ""
        ),
        "best_true_surplus_at_best_round": result.metadata.get(
            "best_true_surplus_at_best_round", ""
        ),
        "cumulative_value_queries_at_best_round": result.metadata.get(
            "cumulative_value_queries_at_best_round", ""
        ),
        "final_round_events": result.metadata.get("final_round_events", ""),
        "final_round_demand_changed_events": result.metadata.get(
            "final_round_demand_changed_events", ""
        ),
        "final_round_refinements": result.metadata.get("final_round_refinements", ""),
        "final_welfare_minus_best_welfare": result.metadata.get(
            "final_welfare_minus_best_welfare", ""
        ),
    }
