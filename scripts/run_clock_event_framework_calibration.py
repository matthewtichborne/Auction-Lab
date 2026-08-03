#!/usr/bin/env python3
"""Evaluate four targeted-clock event policies on nine calibration auctions.

This runner is fully offline: it replays the frozen out-of-domain benchmark
artefacts with deterministic exact value queries.  It is the design-selection
study for ``clock-targeted-v1`` and deliberately does not use the PC-build
scalability outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.auctions.clock import ClockConfig  # noqa: E402
from auctionlab.experiments.llm_comparison import (  # noqa: E402
    payment_diagnostic_fields,
)
from auctionlab.experiments.proxy_clock_runner import (  # noqa: E402
    ProxyClockConfig,
    run_proxy_clock_experiment,
)
from auctionlab.experiments.run_config import (  # noqa: E402
    refinement_records_to_rows,
)
from auctionlab.experiments.runner import run_sealed_vcg_experiment  # noqa: E402
from auctionlab.llm.value_calibration import load_calibration_config  # noqa: E402
from scripts.validate_pv_calibration import (  # noqa: E402
    _benchmark_paths,
    _load_artefacts,
    _proxy_adapters,
    _true_welfare,
)
from scripts.tune_clock_parameters_calibration import (  # noqa: E402
    disclosed_budget_per_good,
)


TREATMENTS: dict[str, dict[str, bool]] = {
    "core": {
        "contested_bundle_refinement": False,
        "terminal_vcg_witness_verification": False,
        "terminal_best_losing_challenger": False,
    },
    "core_contested": {
        "contested_bundle_refinement": True,
        "terminal_vcg_witness_verification": False,
        "terminal_best_losing_challenger": False,
    },
    "core_terminal_vcg": {
        "contested_bundle_refinement": False,
        "terminal_vcg_witness_verification": True,
        "terminal_best_losing_challenger": True,
    },
    "clock_targeted_v1": {
        "contested_bundle_refinement": True,
        "terminal_vcg_witness_verification": True,
        "terminal_best_losing_challenger": True,
    },
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--pv-calibration-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--price-step", type=float, default=50.0)
    parser.add_argument("--price-step-fraction", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--tie-threshold", type=float, default=100.0)
    parser.add_argument("--tie-threshold-fraction", type=float, default=None)
    args = parser.parse_args(argv)
    if args.price_step_fraction is not None and args.price_step_fraction <= 0:
        parser.error("--price-step-fraction must be positive")
    if (
        args.tie_threshold_fraction is not None
        and args.tie_threshold_fraction < 0
    ):
        parser.error("--tie-threshold-fraction must be non-negative")
    return args


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for treatment in TREATMENTS:
        members = [row for row in rows if row["treatment"] == treatment]
        aggregate: dict[str, Any] = {
            "treatment": treatment,
            "environments": len(members),
        }
        for metric in (
            "efficiency",
            "payment_error_over_optimum_welfare",
            "revenue_loss",
            "value_queries",
            "rounds",
        ):
            values = [
                float(row[metric])
                for row in members
                if math.isfinite(float(row[metric]))
            ]
            aggregate[f"mean_{metric}"] = statistics.fmean(values)
            aggregate[f"min_{metric}"] = min(values)
            aggregate[f"max_{metric}"] = max(values)
        aggregate["allocation_match_rate"] = statistics.fmean(
            1.0 if row["allocation_match"] else 0.0 for row in members
        )
        output.append(aggregate)
    return output


def _plot(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for metric, label, multiplier in (
        ("mean_efficiency", "Mean efficiency (%)", 100.0),
        (
            "mean_payment_error_over_optimum_welfare",
            "Mean payment error / optimum welfare (%)",
            100.0,
        ),
        ("mean_value_queries", "Mean exact value queries", 1.0),
    ):
        figure, axis = plt.subplots(figsize=(7.5, 4.2))
        labels = [row["treatment"] for row in summary]
        values = [float(row[metric]) * multiplier for row in summary]
        axis.bar(range(len(labels)), values)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(plots / f"{metric}.png", dpi=180)
        figure.savefig(plots / f"{metric}.pdf")
        plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    calibration = load_calibration_config(args.pv_calibration_config)
    artefacts = _load_artefacts(_benchmark_paths(args.benchmark_dir))
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for (domain, instance_index), artefact in sorted(artefacts.items()):
        reference = disclosed_budget_per_good(artefact)
        price_step = (
            args.price_step_fraction * reference
            if args.price_step_fraction is not None
            else args.price_step
        )
        tie_threshold = (
            args.tie_threshold_fraction * reference
            if args.tie_threshold_fraction is not None
            else args.tie_threshold
        )
        for treatment, switches in TREATMENTS.items():
            scenario, proxies = _proxy_adapters(artefact, calibration)
            truth = run_sealed_vcg_experiment(scenario.instance)
            result = run_proxy_clock_experiment(
                scenario.instance,
                proxies,
                ClockConfig(
                    max_rounds=args.max_rounds,
                    price_step=price_step,
                ),
                ProxyClockConfig(
                    top_k=args.top_k,
                    elicited=True,
                    tie_threshold=tie_threshold,
                    event_framework="targeted_v1",
                    incumbent_verification=True,
                    allocation_change_audit=True,
                    demand_switch_verification=True,
                    terminal_winner_verification=True,
                    terminal_stability_audit=False,
                    **switches,
                ),
            )
            true_welfare = _true_welfare(
                scenario.instance, result.allocation
            )
            pricing = payment_diagnostic_fields(truth, result)
            refinement_counts = result.metadata[
                "refinement_query_count_by_bidder"
            ]
            rows.append({
                "domain": domain,
                "environment_instance": instance_index,
                "treatment": treatment,
                "budget_per_good_reference": reference,
                "price_step": price_step,
                "tie_threshold": tie_threshold,
                "efficiency": true_welfare / truth.welfare,
                "true_welfare": true_welfare,
                "full_info_welfare": truth.welfare,
                "revenue": result.revenue,
                "full_info_revenue": truth.revenue,
                **pricing,
                "value_queries": sum(refinement_counts.values()),
                "rounds": result.rounds or 0,
                "allocation_match": result.allocation == truth.allocation,
            })
            annotated = refinement_records_to_rows(
                f"{domain}_instance{instance_index}",
                f"clock_{treatment}",
                result.metadata["refinement_records_by_bidder"],
                final_allocation=result.allocation,
                reported_vcg_counterfactuals=result.metadata[
                    "vcg_counterfactuals"
                ],
                full_info_allocation=truth.allocation,
                full_info_vcg_counterfactuals=truth.metadata.get(
                    "vcg_counterfactuals"
                ),
            )
            event_rows.extend(
                {
                    "domain": domain,
                    "environment_instance": instance_index,
                    "treatment": treatment,
                    **row,
                }
                for row in annotated
            )

    summary = _aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "clock_event_framework_runs.csv", rows)
    _write_csv(args.output_dir / "clock_event_framework_summary.csv", summary)
    _write_csv(args.output_dir / "clock_event_framework_events.csv", event_rows)
    _plot(args.output_dir, summary)
    (args.output_dir / "clock_event_framework_config.json").write_text(
        json.dumps(
            {
                "treatments": TREATMENTS,
                "calibration": calibration.to_dict(),
                "max_rounds": args.max_rounds,
                "price_step": args.price_step,
                "price_step_fraction": args.price_step_fraction,
                "top_k": args.top_k,
                "tie_threshold": args.tie_threshold,
                "tie_threshold_fraction": args.tie_threshold_fraction,
                "environment_count": len(artefacts),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Evaluated {len(rows)} clock cells across {len(artefacts)} environments.")
    for row in summary:
        print(
            f"{row['treatment']}: efficiency={row['mean_efficiency']:.3f}, "
            "payment_error/welfare="
            f"{row['mean_payment_error_over_optimum_welfare']:.3f}, "
            f"VQs={row['mean_value_queries']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
