"""Compare LLM-proxy mechanisms with full-information auction outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from auctionlab.auction_types import Bundle
from auctionlab.auctions.clock import ClockConfig
from auctionlab.bids.xor import XorBid
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


def true_welfare_for_allocation(
    instance: AuctionInstance,
    allocation: dict[str, Bundle],
) -> float:
    """Evaluate any reported allocation against benchmark instance values."""
    return float(
        sum(
            instance.value_of(
                bidder_id,
                allocation.get(bidder_id, frozenset()),
            )
            for bidder_id in instance.bidder_ids
        )
    )


def allocation_matches(
    a: dict[str, Bundle],
    b: dict[str, Bundle],
) -> bool:
    return a == b


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
        "allocation_match": allocation_matches(
            full_info.allocation,
            result.allocation,
        ),
        "welfare_match": (
            abs(full_info.welfare - true_welfare) <= welfare_tolerance
        ),
        "full_info_allocation": allocation_to_str(full_info.allocation),
        "proxy_allocation": allocation_to_str(result.allocation),
        "elicitation_rounds": result.metadata["elicitation_rounds"],
        "feedback_rule": result.metadata["feedback_rule"],
        "max_refinements_per_bidder": result.metadata[
            "max_refinements_per_bidder"
        ],
        "refinement_query_count_by_bidder": ";".join(
            f"{bidder_id}:{count}"
            for bidder_id, count in sorted(
                result.metadata["refinement_query_count_by_bidder"].items()
            )
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

    return {
        "instance_name": instance_name,
        "mechanism": result.mechanism,
        "full_info_welfare": full_info.welfare,
        "proxy_reported_welfare": result.welfare,
        "proxy_true_welfare": true_welfare,
        "efficiency": efficiency,
        "full_info_revenue": full_info.revenue,
        "proxy_revenue": result.revenue,
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
        "top_k": result.metadata["top_k"],
        "elicited": result.metadata["elicited"],
        "margin_threshold": result.metadata["margin_threshold"],
        "tie_threshold": result.metadata["tie_threshold"],
        "max_refinements_per_bidder": result.metadata[
            "max_refinements_per_bidder"
        ],
        "refinement_query_count_by_bidder": ";".join(
            f"{bidder_id}:{count}"
            for bidder_id, count in sorted(
                result.metadata["refinement_query_count_by_bidder"].items()
            )
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
    }


def ceca_result_to_row(
    *,
    instance_name: str,
    instance: AuctionInstance,
    result: MechanismResult,
    welfare_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Serialize a :class:`MechanismResult` from ``run_proxy_ceca_experiment``.

    Not generic over ``proxy_sealed_result_to_row``/``proxy_clock_result_to_row``
    despite the shared shape -- CECA's metadata keys (``payment_rule``,
    ``converged``, ``stage1_welfare``/``stage2_welfare``,
    ``pruning_query_count_by_bidder``) have no sealed/clock analogue, and
    CECA tracks no ``refinement_records_by_bidder`` (every round *is* an
    elicitation step; there's no separate refinement-event layer).
    """
    full_info = run_sealed_vcg_experiment(instance)
    true_welfare = true_welfare_for_allocation(instance, result.allocation)

    if full_info.welfare > welfare_tolerance:
        efficiency = true_welfare / full_info.welfare
    else:
        efficiency = 1.0

    reported_welfare = result.welfare  # WDP objective = sum of reported values for winners
    welfare_understatement = reported_welfare - true_welfare
    reported_true_ratio = (
        true_welfare / reported_welfare
        if abs(reported_welfare) > welfare_tolerance
        else (1.0 if abs(true_welfare) <= welfare_tolerance else float("nan"))
    )
    revenue = result.revenue
    true_surplus = true_welfare - revenue

    return {
        "instance_name": instance_name,
        "mechanism": result.mechanism,
        "ceca_initial_bid_mode": result.metadata.get("ceca_initial_bid_mode", "full_proxy"),
        "ceca_variant": result.metadata.get("ceca_variant", "prior"),
        "full_info_welfare": full_info.welfare,
        "proxy_reported_welfare": reported_welfare,
        "reported_allocated_welfare": reported_welfare,
        "proxy_true_welfare": true_welfare,
        "true_allocated_welfare": true_welfare,
        "reported_true_welfare_ratio": reported_true_ratio,
        "welfare_understatement_or_overstatement": welfare_understatement,
        "efficiency": efficiency,
        "full_info_revenue": full_info.revenue,
        "proxy_revenue": revenue,
        "true_surplus": true_surplus,
        "negative_true_surplus": true_surplus < -welfare_tolerance,
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
        "payment_rule": result.metadata["payment_rule"],
        "ceca_internal_price_rule": result.metadata.get(
            "ceca_internal_price_rule", "lindahl_manifest"
        ),
        "converged": result.metadata["converged"],
        "ceca_rounds": result.metadata.get("ceca_rounds", result.rounds or ""),
        "stage1_welfare": result.metadata["stage1_welfare"],
        "stage2_welfare": result.metadata["stage2_welfare"],
        "initial_manifest_total_atoms": result.metadata.get(
            "initial_manifest_total_atoms", ""
        ),
        "final_manifest_total_atoms": result.metadata.get(
            "final_manifest_total_atoms", ""
        ),
        "manifest_growth_total": result.metadata.get("manifest_growth_total", ""),
        "manifest_growth_by_bidder": ";".join(
            f"{b}:{g}"
            for b, g in sorted(
                result.metadata.get("manifest_growth_by_bidder", {}).items()
            )
        ),
        "demanded_bundle_count_by_bidder": ";".join(
            f"{b}:{c}"
            for b, c in sorted(
                result.metadata.get("demanded_bundle_count_by_bidder", {}).items()
            )
        ),
        "unique_demanded_bundle_count_by_bidder": ";".join(
            f"{b}:{c}"
            for b, c in sorted(
                result.metadata.get("unique_demanded_bundle_count_by_bidder", {}).items()
            )
        ),
        "duplicate_demand_count_by_bidder": ";".join(
            f"{b}:{c}"
            for b, c in sorted(
                result.metadata.get("duplicate_demand_count_by_bidder", {}).items()
            )
        ),
        "unchanged_demand_count_by_bidder": ";".join(
            f"{b}:{c}"
            for b, c in sorted(
                result.metadata.get("unchanged_demand_count_by_bidder", {}).items()
            )
        ),
        "demand_query_count_by_bidder": ";".join(
            f"{bidder_id}:{count}"
            for bidder_id, count in sorted(
                result.metadata["demand_query_count_by_bidder"].items()
            )
        ),
        "pruning_query_count_by_bidder": ";".join(
            f"{bidder_id}:{count}"
            for bidder_id, count in sorted(
                result.metadata["pruning_query_count_by_bidder"].items()
            )
        ),
        "final_manifest_sizes": ";".join(
            f"{bidder_id}:{count}"
            for bidder_id, count in sorted(
                result.metadata["final_manifest_sizes"].items()
            )
        ),
        "initial_manifest_sizes": ";".join(
            f"{bidder_id}:{count}"
            for bidder_id, count in sorted(
                result.metadata.get("initial_manifest_sizes", {}).items()
            )
        ),
        "initial_bids": reported_bids_to_str(result.metadata["initial_bids"]),
        "final_bids": reported_bids_to_str(result.metadata["final_bids"]),
        "proxy_architecture": result.metadata.get("proxy_architecture", "unknown"),
        "ceca_atomic_trimming": result.metadata.get("ceca_atomic_trimming", True),
        "ceca_trim_value_tolerance": result.metadata.get("ceca_trim_value_tolerance", 0.0),
        "ceca_trimmed_demand_count": result.metadata.get("ceca_trimmed_demand_count", 0),
        "ceca_total_trim_items_removed": result.metadata.get("ceca_total_trim_items_removed", 0),
        "ceca_total_trim_value_queries": result.metadata.get("ceca_total_trim_value_queries", 0),
        "ceca_avg_raw_demand_size": result.metadata.get("ceca_avg_raw_demand_size", 0.0),
        "ceca_avg_trimmed_atom_size": result.metadata.get("ceca_avg_trimmed_atom_size", 0.0),
        "ceca_repeated_raw_demand_count": result.metadata.get("ceca_repeated_raw_demand_count", 0),
        "ceca_repeated_trimmed_atom_count": result.metadata.get("ceca_repeated_trimmed_atom_count", 0),
        # Feature 3: stopped reason (explicit field, not buried in converged bool).
        "stopped_reason": result.metadata.get("stopped_reason", "max_rounds"),
        # Feature 2: no-new-information diagnostics.
        "total_no_new_information": result.metadata.get("total_no_new_information", 0),
        "no_new_information_count_by_bidder": ";".join(
            f"{b}:{c}"
            for b, c in sorted(
                result.metadata.get("no_new_information_count_by_bidder", {}).items()
            )
        ),
        "total_useful_counterexamples": result.metadata.get("total_useful_counterexamples", ""),
        "useful_counterexample_count_by_bidder": ";".join(
            f"{b}:{c}"
            for b, c in sorted(
                result.metadata.get("useful_counterexample_count_by_bidder", {}).items()
            )
        ),
        # Feature 1: corrected trim diagnostics.
        "num_demands_trimmed_to_smaller_atom": result.metadata.get(
            "num_demands_trimmed_to_smaller_atom", 0
        ),
        "total_raw_demand_items": result.metadata.get("total_raw_demand_items", 0),
        "total_inserted_atom_items": result.metadata.get("total_inserted_atom_items", 0),
        "total_net_items_removed": result.metadata.get("total_net_items_removed", 0),
        "avg_raw_demand_size": result.metadata.get("avg_raw_demand_size", 0.0),
        "avg_inserted_atom_size": result.metadata.get("avg_inserted_atom_size", 0.0),
        # Tasks 1–2: atom insertion diagnostics.
        "atom_insertion_count": len(result.metadata.get("atom_insertion_log", [])),
        "outside_interest_insertion_total": result.metadata.get(
            "outside_interest_insertion_total", 0
        ),
        "outside_interest_insertion_by_bidder": ";".join(
            f"{b}:{c}"
            for b, c in sorted(
                result.metadata.get(
                    "outside_interest_insertion_count_by_bidder", {}
                ).items()
            )
            if c > 0
        ),
        # Task 4: per-round no-info stats.
        "no_info_round_count": result.metadata.get("no_info_round_count", 0),
        "bidders_exhausted_by_repetition": ";".join(
            sorted(result.metadata.get("bidders_exhausted_by_repetition", []))
        ),
        # Task 5: bidder exhaustion.
        "exhaustion_event_count": len(result.metadata.get("exhaustion_events", [])),
        # Demand universe constraint.
        "ceca_demand_universe": result.metadata.get("ceca_demand_universe", "all_items"),
        "out_of_universe_demand_count": result.metadata.get("out_of_universe_demand_count", 0),
        "rejected_out_of_universe_count": result.metadata.get("rejected_out_of_universe_count", 0),
        "projected_demand_count": result.metadata.get("projected_demand_count", 0),
    }


def ceca_winner_diagnostics_rows(
    *,
    instance_name: str,
    instance: AuctionInstance,
    result: MechanismResult,
) -> list[dict[str, Any]]:
    """Generate per-winner value and payment diagnostics for a CECA result.

    For each winning bidder, compares the proxy's reported value for the
    allocated bundle against the bidder's true value, and breaks down
    payment into true and reported surplus.
    """
    rows: list[dict[str, Any]] = []
    ceca_variant = result.metadata.get("ceca_variant", "prior")
    ceca_initial_bid_mode = result.metadata.get("ceca_initial_bid_mode", "full_proxy")
    payment_rule = result.metadata["payment_rule"]
    final_bids: dict[str, Any] = result.metadata.get("final_bids", {})

    for bidder_id, bundle in sorted(result.allocation.items()):
        if not bundle:
            continue
        reported_value = (
            final_bids[bidder_id].value_of(bundle)
            if bidder_id in final_bids
            else float("nan")
        )
        true_value = instance.value_of(bidder_id, bundle)
        payment = result.payments.get(bidder_id, 0.0)
        rows.append({
            "scenario": instance_name,
            "ceca_variant": ceca_variant,
            "ceca_initial_bid_mode": ceca_initial_bid_mode,
            "payment_rule": payment_rule,
            "bidder_id": bidder_id,
            "allocated_bundle": "{" + ",".join(sorted(bundle)) + "}",
            "reported_value": reported_value,
            "true_value": true_value,
            "value_error": reported_value - true_value,
            "payment": payment,
            "true_surplus": true_value - payment,
            "reported_surplus": reported_value - payment,
        })
    return rows
