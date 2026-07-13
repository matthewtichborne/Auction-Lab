from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from auctionlab.experiments.runner import (
    BatchSummary,
    ExperimentResult,
    MechanismResult,
    get_result_by_mechanism,
)

from auctionlab.experiments.diagnostics import (
    count_full_atoms,
    count_supplementary_atoms,
    missing_sealed_winning_bundles,
    sealed_winning_bundles_observed,
    supplementary_coverage_ratio,
)
from auctionlab.instances.base import AuctionInstance


def allocation_to_str(allocation: Dict[str, frozenset[str]]) -> str:
    """
    Stable string representation of an allocation.

    Example:
      i1:[];i2:[B];i3:[A,C]
    """
    parts: List[str] = []

    for bidder_id in sorted(allocation.keys()):
        bundle = allocation[bidder_id]
        bundle_str = ",".join(sorted(bundle))
        parts.append(f"{bidder_id}:[{bundle_str}]")

    return ";".join(parts)


def mechanism_result_to_row(
    instance_name: str,
    result: MechanismResult,
) -> Dict[str, Any]:
    """
    Flatten one mechanism result into a CSV row.
    """
    return {
        "instance_name": instance_name,
        "mechanism": result.mechanism,
        "welfare": result.welfare,
        "revenue": result.revenue,
        "rounds": result.rounds if result.rounds is not None else "",
        "query_count": result.query_count,
        "allocation": allocation_to_str(result.allocation),
    }


def batch_results_to_mechanism_rows(
    batch_results: List[ExperimentResult],
) -> List[Dict[str, Any]]:
    """
    Convert batch results into one row per instance-mechanism pair.
    """
    rows: List[Dict[str, Any]] = []

    for experiment_result in batch_results:
        for mechanism_result in experiment_result.results:
            rows.append(
                mechanism_result_to_row(
                    instance_name=experiment_result.instance_name,
                    result=mechanism_result,
                )
            )

    return rows


def batch_summaries_to_rows(
    summaries_by_top_k: dict[int, BatchSummary],
) -> list[dict[str, Any]]:
    """
    Convert top-k batch summaries into rows ordered by top_k.
    """
    rows: list[dict[str, Any]] = []

    for top_k in sorted(summaries_by_top_k):
        summary = summaries_by_top_k[top_k]
        rows.append(
            {
                "top_k": top_k,
                "n_instances": summary.n_instances,
                "sealed_avg_welfare": summary.sealed_avg_welfare,
                "clock_avg_welfare": summary.clock_avg_welfare,
                "clock_avg_efficiency": summary.clock_avg_efficiency,
                "sealed_avg_revenue": summary.sealed_avg_revenue,
                "clock_avg_revenue": summary.clock_avg_revenue,
                "clock_avg_rounds": summary.clock_avg_rounds,
                "sealed_avg_query_count": summary.sealed_avg_query_count,
                "clock_avg_query_count": summary.clock_avg_query_count,
                "allocation_match_rate": summary.allocation_match_rate,
                "welfare_match_rate": summary.welfare_match_rate,
            }
        )

    return rows


def batch_results_to_comparison_rows(
    batch_results: List[ExperimentResult],
    *,
    clock_mechanism: str = "clock_supplementary_vcg_top_1",
    instances_by_name: Dict[str, AuctionInstance] | None = None,
    welfare_tolerance: float = 1e-6,
) -> List[Dict[str, Any]]:
    """
    Convert batch results into one comparison row per instance.

    If instances_by_name is provided, adds supplementary coverage diagnostics.
    """
    rows: List[Dict[str, Any]] = []

    for experiment_result in batch_results:
        sealed = get_result_by_mechanism(experiment_result, "sealed_xor_vcg")
        clock = get_result_by_mechanism(
            experiment_result,
            clock_mechanism,
        )

        if sealed.welfare > welfare_tolerance:
            efficiency = clock.welfare / sealed.welfare
        else:
            efficiency = 1.0

        allocation_match = sealed.allocation == clock.allocation
        welfare_match = abs(sealed.welfare - clock.welfare) <= welfare_tolerance

        row: Dict[str, Any] = {
            "instance_name": experiment_result.instance_name,
            "clock_mechanism": clock_mechanism,
            "sealed_welfare": sealed.welfare,
            "clock_welfare": clock.welfare,
            "efficiency": efficiency,
            "sealed_revenue": sealed.revenue,
            "clock_revenue": clock.revenue,
            "sealed_query_count": sealed.query_count,
            "clock_query_count": clock.query_count,
            "clock_rounds": clock.rounds if clock.rounds is not None else "",
            "allocation_match": allocation_match,
            "welfare_match": welfare_match,
            "sealed_allocation": allocation_to_str(sealed.allocation),
            "clock_allocation": allocation_to_str(clock.allocation),
        }

        if instances_by_name is not None:
            instance = instances_by_name[experiment_result.instance_name]
            supplementary_atoms = clock.metadata["supplementary_atoms"]

            missing = missing_sealed_winning_bundles(
                sealed_allocation=sealed.allocation,
                supplementary_atoms=supplementary_atoms,
            )

            row.update(
                {
                    "full_atoms_count": count_full_atoms(instance),
                    "supplementary_atoms_count": count_supplementary_atoms(
                        supplementary_atoms
                    ),
                    "supplementary_coverage_ratio": supplementary_coverage_ratio(
                        instance,
                        supplementary_atoms,
                    ),
                    "sealed_winning_bundles_observed": sealed_winning_bundles_observed(
                        sealed.allocation,
                        supplementary_atoms,
                    ),
                    "missing_sealed_winning_bundles": allocation_to_str(missing),
                }
            )

        rows.append(row)

    return rows


def write_csv(
    rows: List[Dict[str, Any]],
    path: str | Path,
) -> None:
    """
    Write rows to a CSV file.

    If rows is empty, writes an empty file with no header.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("")
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
