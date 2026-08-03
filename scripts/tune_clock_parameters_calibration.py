#!/usr/bin/env python3
"""Tune dimensionless clock parameters on frozen calibration environments."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.auctions.clock import ClockConfig  # noqa: E402
from auctionlab.experiments.llm_comparison import payment_diagnostic_fields  # noqa: E402
from auctionlab.experiments.proxy_clock_runner import (  # noqa: E402
    ProxyClockConfig,
    run_proxy_clock_experiment,
)
from auctionlab.experiments.runner import run_sealed_vcg_experiment  # noqa: E402
from auctionlab.llm.value_calibration import load_calibration_config  # noqa: E402
from scripts.validate_pv_calibration import (  # noqa: E402
    _benchmark_paths,
    _load_artefacts,
    _proxy_adapters,
    _true_welfare,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--pv-calibration-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--price-step-fractions", type=float, nargs="+", default=[0.1, 0.2, 0.4]
    )
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--tie-threshold-fractions",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.8],
    )
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--efficiency-band", type=float, default=0.005)
    args = parser.parse_args(argv)
    if any(value <= 0 for value in args.price_step_fractions):
        parser.error("--price-step-fractions must be positive")
    if any(value <= 0 for value in args.top_k_values):
        parser.error("--top-k-values must be positive")
    if any(value < 0 for value in args.tie_threshold_fractions):
        parser.error("--tie-threshold-fractions must be non-negative")
    return args


def disclosed_budget_per_good(artefact: dict[str, Any]) -> float:
    """Median disclosed budget cap divided by the number of available goods."""
    environment = artefact["environment"]
    budgets = [float(bidder["budget_cap"]) for bidder in environment["bidders"]]
    return statistics.median(budgets) / len(environment["goods"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["price_step_fraction"]),
            int(row["top_k"]),
            float(row["tie_threshold_fraction"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        result: dict[str, Any] = {
            "price_step_fraction": key[0],
            "top_k": key[1],
            "tie_threshold_fraction": key[2],
            "environments": len(members),
        }
        for metric in (
            "efficiency",
            "payment_error_over_optimum_welfare",
            "value_queries",
            "rounds",
        ):
            values = [float(row[metric]) for row in members]
            result[f"mean_{metric}"] = statistics.fmean(values)
            result[f"min_{metric}"] = min(values)
            result[f"max_{metric}"] = max(values)
        output.append(result)
    return output


def _select(summary: list[dict[str, Any]], band: float) -> dict[str, Any]:
    best_efficiency = max(float(row["mean_efficiency"]) for row in summary)
    eligible = [
        row
        for row in summary
        if float(row["mean_efficiency"]) >= best_efficiency - band
    ]
    return min(
        eligible,
        key=lambda row: (
            float(row["mean_payment_error_over_optimum_welfare"]),
            float(row["mean_value_queries"]),
            float(row["mean_rounds"]),
            float(row["price_step_fraction"]),
            int(row["top_k"]),
            float(row["tie_threshold_fraction"]),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    calibration = load_calibration_config(args.pv_calibration_config)
    artefacts = _load_artefacts(_benchmark_paths(args.benchmark_dir))
    grid = list(itertools.product(
        args.price_step_fractions,
        args.top_k_values,
        args.tie_threshold_fractions,
    ))
    rows: list[dict[str, Any]] = []
    for (domain, instance_index), artefact in sorted(artefacts.items()):
        reference = disclosed_budget_per_good(artefact)
        for price_fraction, top_k, tie_fraction in grid:
            scenario, proxies = _proxy_adapters(artefact, calibration)
            truth = run_sealed_vcg_experiment(scenario.instance)
            result = run_proxy_clock_experiment(
                scenario.instance,
                proxies,
                ClockConfig(
                    max_rounds=args.max_rounds,
                    price_step=price_fraction * reference,
                ),
                ProxyClockConfig(
                    top_k=top_k,
                    elicited=True,
                    tie_threshold=tie_fraction * reference,
                    event_framework="targeted_v1",
                    incumbent_verification=True,
                    allocation_change_audit=True,
                    demand_switch_verification=True,
                    contested_bundle_refinement=True,
                    terminal_stability_audit=False,
                    terminal_winner_verification=True,
                    terminal_vcg_witness_verification=True,
                    terminal_best_losing_challenger=True,
                ),
            )
            welfare = _true_welfare(scenario.instance, result.allocation)
            pricing = payment_diagnostic_fields(truth, result)
            rows.append({
                "domain": domain,
                "environment_instance": instance_index,
                "budget_per_good_reference": reference,
                "price_step_fraction": price_fraction,
                "price_step_absolute": price_fraction * reference,
                "top_k": top_k,
                "tie_threshold_fraction": tie_fraction,
                "tie_threshold_absolute": tie_fraction * reference,
                "efficiency": welfare / truth.welfare,
                "payment_error_over_optimum_welfare": pricing[
                    "payment_error_over_optimum_welfare"
                ],
                "revenue_loss": pricing["revenue_loss"],
                "value_queries": sum(
                    result.metadata["refinement_query_count_by_bidder"].values()
                ),
                "rounds": result.rounds or 0,
            })

    folds: list[dict[str, Any]] = []
    indices = sorted({int(row["environment_instance"]) for row in rows})
    for held_out in indices:
        train = [row for row in rows if int(row["environment_instance"]) != held_out]
        test = [row for row in rows if int(row["environment_instance"]) == held_out]
        selected = _select(_summarise(train), args.efficiency_band)
        matching = [
            row for row in test
            if float(row["price_step_fraction"]) == float(selected["price_step_fraction"])
            and int(row["top_k"]) == int(selected["top_k"])
            and float(row["tie_threshold_fraction"]) == float(selected["tie_threshold_fraction"])
        ]
        folds.append({
            "held_out_instance": held_out,
            "price_step_fraction": selected["price_step_fraction"],
            "top_k": selected["top_k"],
            "tie_threshold_fraction": selected["tie_threshold_fraction"],
            "heldout_mean_efficiency": statistics.fmean(
                float(row["efficiency"]) for row in matching
            ),
            "heldout_mean_payment_error_over_optimum_welfare": statistics.fmean(
                float(row["payment_error_over_optimum_welfare"])
                for row in matching
            ),
            "heldout_mean_value_queries": statistics.fmean(
                float(row["value_queries"]) for row in matching
            ),
        })

    summary = _summarise(rows)
    selected = _select(summary, args.efficiency_band)
    recommendation = {
        "scale_definition": "median disclosed budget cap / number of goods",
        "selection_rule": (
            "within efficiency_band of best mean efficiency; then minimise "
            "payment error, exact value queries, and rounds"
        ),
        "efficiency_band": args.efficiency_band,
        "selected": selected,
        "leave_one_instance_out": folds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "clock_parameter_runs.csv", rows)
    _write_csv(args.output_dir / "clock_parameter_summary.csv", summary)
    _write_csv(args.output_dir / "clock_parameter_folds.csv", folds)
    (args.output_dir / "clock_parameter_recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(recommendation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
