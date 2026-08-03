#!/usr/bin/env python3
"""Offline five-seed robustness grid for the recommended clock policy.

Every cell replays a frozen 8x8 elicitation pack with deterministic exact
queries.  No LLM endpoint is contacted.  The runner varies price increment,
top-k demand width, and the allocation-pivotal tie threshold, then writes
seed-level and aggregate pricing/efficiency diagnostics plus a deterministic
candidate recommendation.  It is resumable at the grid-cell level.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Sequence


@dataclass(frozen=True)
class ClockGridCell:
    seed: int
    price_step: float
    top_k: int
    tie_threshold: float
    pack: Path
    run_dir: Path


def _slug_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_cells(args: argparse.Namespace) -> list[ClockGridCell]:
    return [
        ClockGridCell(
            seed=seed,
            price_step=price_step,
            top_k=top_k,
            tie_threshold=tie_threshold,
            pack=(
                args.elicitation_pack_dir
                / f"seed_{seed}"
                / "anchor_8x8"
                / "frozen_elicitation.json"
            ),
            run_dir=(
                args.output_dir
                / (
                    f"price_{_slug_number(price_step)}_topk_{top_k}_"
                    f"tie_{_slug_number(tie_threshold)}"
                )
                / f"seed_{seed}"
            ),
        )
        for price_step in args.price_steps
        for top_k in args.top_k_values
        for tie_threshold in args.tie_thresholds
        for seed in args.seeds
    ]


def build_command(
    cell: ClockGridCell, args: argparse.Namespace
) -> list[str]:
    command = [
        sys.executable,
        "examples/run_live_llm_curated_batch.py",
        "--scenario", "pc_build",
        "--scenario-spec", str(args.scenario_spec),
        "--num-goods", "8",
        "--num-bidders", "8",
        "--scenario-seed", str(cell.seed),
        "--selection-policy", "coverage_stratified",
        "--seed-type", "structured",
        "--elicitation-pack", str(cell.pack),
        "--ask-initial-question",
        "--use-interest-map",
        "--use-provisional-valuations",
        "--person-query-mode", "deterministic",
        "--skip-baselines",
        "--elicited-clock",
        "--top-k", str(cell.top_k),
        "--max-rounds", str(args.max_rounds),
        "--price-step", str(cell.price_step),
        "--clock-tie-threshold", str(cell.tie_threshold),
        "--event-policy", args.event_policy,
        "--llm-cache-mode", "off",
        "--log-dir", str(cell.run_dir),
    ]
    if args.pv_calibration_config is not None:
        command.extend([
            "--pv-calibration-config",
            str(args.pv_calibration_config),
        ])
    return command


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-spec",
        type=Path,
        default=Path("scenarios/pc_build_v3/pc_build_population_16x16.json"),
    )
    parser.add_argument(
        "--elicitation-pack-dir",
        type=Path,
        default=Path("outputs/elicitation_packs/scalability"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pv-calibration-config", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--price-steps", type=float, nargs="+", default=[25.0, 50.0, 100.0]
    )
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--tie-thresholds", type=float, nargs="+", default=[50.0, 100.0, 200.0]
    )
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument(
        "--event-policy",
        choices=["recommended", "final-v1"],
        default="final-v1",
        help="Frozen clock architecture to test (default: final-v1).",
    )
    parser.add_argument(
        "--efficiency-band",
        type=float,
        default=0.005,
        help=(
            "Configurations within this absolute mean-efficiency distance "
            "of the best enter the pricing/query tie-break (default: 0.005)."
        ),
    )
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if any(value <= 0 for value in args.price_steps):
        parser.error("--price-steps must be positive")
    if any(value <= 0 for value in args.top_k_values):
        parser.error("--top-k-values must be positive")
    if any(value < 0 for value in args.tie_thresholds):
        parser.error("--tie-thresholds must be non-negative")
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    if args.efficiency_band < 0:
        parser.error("--efficiency-band must be non-negative")
    return args


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_cell(cell: ClockGridCell) -> dict[str, Any]:
    summary_rows = _read_csv(cell.run_dir / "curated_run_summary.csv")
    arm = next(
        row for row in summary_rows if row["arm"] == f"proxy clock k={cell.top_k}"
    )
    detail = _read_csv(
        cell.run_dir / f"curated_clock_proxy_elicited_top_{cell.top_k}.csv"
    )[0]
    return {
        "seed": cell.seed,
        "price_step": cell.price_step,
        "top_k": cell.top_k,
        "tie_threshold": cell.tie_threshold,
        "efficiency": _number(arm, "efficiency"),
        "revenue": _number(arm, "revenue"),
        "full_info_revenue": _number(detail, "full_info_revenue"),
        "revenue_loss": _number(detail, "revenue_loss"),
        "revenue_absolute_percentage_error": _number(
            detail, "revenue_absolute_percentage_error"
        ),
        "payment_error_over_optimum_welfare": _number(
            detail, "payment_error_over_optimum_welfare"
        ),
        "value_queries": _number(arm, "vq"),
        "demand_queries": _number(arm, "dq"),
        "rounds": _number(detail, "rounds"),
        "allocation_match": detail.get("allocation_match", ""),
        "run_dir": str(cell.run_dir),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (float(row["price_step"]), int(row["top_k"]), float(row["tie_threshold"])),
            [],
        ).append(row)
    output: list[dict[str, Any]] = []
    metrics = (
        "efficiency",
        "revenue_loss",
        "revenue_absolute_percentage_error",
        "payment_error_over_optimum_welfare",
        "value_queries",
        "demand_queries",
        "rounds",
    )
    for (price_step, top_k, tie_threshold), members in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "price_step": price_step,
            "top_k": top_k,
            "tie_threshold": tie_threshold,
            "seeds": len(members),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in members if math.isfinite(float(row[metric]))]
            aggregate[f"mean_{metric}"] = fmean(values) if values else math.nan
            aggregate[f"min_{metric}"] = min(values) if values else math.nan
            aggregate[f"max_{metric}"] = max(values) if values else math.nan
        aggregate["allocation_match_rate"] = fmean(
            1.0 if str(row["allocation_match"]).lower() == "true" else 0.0
            for row in members
        )
        output.append(aggregate)
    return output


def _recommend(summary: list[dict[str, Any]], efficiency_band: float) -> dict[str, Any]:
    best_efficiency = max(float(row["mean_efficiency"]) for row in summary)
    eligible = [
        row
        for row in summary
        if float(row["mean_efficiency"]) >= best_efficiency - efficiency_band
    ]
    selected = min(
        eligible,
        key=lambda row: (
            float(row["mean_payment_error_over_optimum_welfare"]),
            float(row["mean_value_queries"]),
            float(row["mean_rounds"]),
            float(row["price_step"]),
            int(row["top_k"]),
            float(row["tie_threshold"]),
        ),
    )
    return {
        "selection_rule": (
            "within efficiency_band of best mean efficiency; then minimise "
            "mean welfare-normalised payment error, value queries, and rounds"
        ),
        "efficiency_band": efficiency_band,
        "best_mean_efficiency": best_efficiency,
        "eligible_configuration_count": len(eligible),
        "selected": selected,
    }


def _plot_summary(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    """Plot parameter sensitivity without collapsing the three-way grid."""
    if not summary:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV outputs were still written.")
        return

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    tie_thresholds = sorted({float(row["tie_threshold"]) for row in summary})
    top_k_values = sorted({int(row["top_k"]) for row in summary})
    metrics = (
        ("mean_efficiency", "Mean efficiency", 100.0),
        (
            "mean_payment_error_over_optimum_welfare",
            "Mean payment error / optimum welfare",
            100.0,
        ),
        ("mean_value_queries", "Mean exact value queries", 1.0),
    )
    for metric, ylabel, multiplier in metrics:
        figure, axes = plt.subplots(
            1,
            len(tie_thresholds),
            figsize=(4.4 * len(tie_thresholds), 3.8),
            sharey=True,
            squeeze=False,
        )
        for axis, tie_threshold in zip(axes[0], tie_thresholds):
            for top_k in top_k_values:
                members = sorted(
                    (
                        row
                        for row in summary
                        if float(row["tie_threshold"]) == tie_threshold
                        and int(row["top_k"]) == top_k
                    ),
                    key=lambda row: float(row["price_step"]),
                )
                axis.plot(
                    [float(row["price_step"]) for row in members],
                    [float(row[metric]) * multiplier for row in members],
                    marker="o",
                    label=f"top-k={top_k}",
                )
            axis.set_title(f"tie threshold={tie_threshold:g}")
            axis.set_xlabel("Clock price increment")
            axis.grid(alpha=0.25)
        axes[0][0].set_ylabel(
            f"{ylabel} (%)" if multiplier == 100.0 else ylabel
        )
        axes[0][-1].legend(frameon=False)
        figure.tight_layout()
        stem = metric.removeprefix("mean_")
        figure.savefig(plots / f"clock_{stem}.png", dpi=180)
        figure.savefig(plots / f"clock_{stem}.pdf")
        plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cells = build_cells(args)
    for cell in cells:
        if not cell.pack.exists():
            raise FileNotFoundError(f"Missing frozen pack: {cell.pack}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    for index, cell in enumerate(cells, start=1):
        command = build_command(cell, args)
        if args.dry_run:
            print(" ".join(command))
            continue
        summary_path = cell.run_dir / "curated_run_summary.csv"
        if args.aggregate_only or (summary_path.exists() and not args.rerun_complete):
            continue
        cell.run_dir.mkdir(parents=True, exist_ok=True)
        log_path = cell.run_dir / "clock_parameter_runner.log"
        print(
            f"[{index}/{len(cells)}] seed={cell.seed} price={cell.price_step:g} "
            f"top_k={cell.top_k} tie={cell.tie_threshold:g}",
            flush=True,
        )
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            failures.append(
                {
                    "seed": cell.seed,
                    "price_step": cell.price_step,
                    "top_k": cell.top_k,
                    "tie_threshold": cell.tie_threshold,
                    "returncode": result.returncode,
                    "log": str(log_path),
                }
            )
            if args.fail_fast:
                break
        else:
            print(f"  complete in {time.monotonic() - started:.1f}s", flush=True)

    if args.dry_run:
        return 0
    rows = [
        _collect_cell(cell)
        for cell in cells
        if (cell.run_dir / "curated_run_summary.csv").exists()
    ]
    summary = _aggregate(rows)
    _write_csv(args.output_dir / "clock_parameter_runs.csv", rows)
    _write_csv(args.output_dir / "clock_parameter_summary.csv", summary)
    _plot_summary(args.output_dir, summary)
    (args.output_dir / "clock_parameter_failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    if summary:
        recommendation = _recommend(summary, args.efficiency_band)
        (args.output_dir / "clock_parameter_recommendation.json").write_text(
            json.dumps(recommendation, indent=2) + "\n", encoding="utf-8"
        )
        selected = recommendation["selected"]
        print(
            "Recommended candidate: "
            f"price_step={selected['price_step']:g}, top_k={selected['top_k']}, "
            f"tie_threshold={selected['tie_threshold']:g}"
        )
    print(f"Collected {len(rows)}/{len(cells)} cells; failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
