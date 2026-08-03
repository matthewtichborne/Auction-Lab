#!/usr/bin/env python3
"""Run the elicitation-event ablation over the complete scalability grid.

Every treatment replays an existing frozen elicitation pack and uses
deterministic person value queries.  No model endpoint is contacted.  The
runner is resumable at the (seed, case, treatment) level and can execute a
small number of independent subprocesses concurrently.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

try:
    from scripts.run_event_ablation_8x8 import (
        TREATMENTS,
        Treatment,
        _aggregate_events,
        _bool,
        _float,
        _mechanism_details,
        _read_arm_rows,
        _read_event_rows,
        _write_csv,
    )
    from scripts.run_scalability_experiment import (
        ScalabilityRun,
        build_scalability_runs,
    )
except ModuleNotFoundError:  # direct ``python scripts/...py`` execution
    from run_event_ablation_8x8 import (
        TREATMENTS,
        Treatment,
        _aggregate_events,
        _bool,
        _float,
        _mechanism_details,
        _read_arm_rows,
        _read_event_rows,
        _write_csv,
    )
    from run_scalability_experiment import (
        ScalabilityRun,
        build_scalability_runs,
    )


@dataclass(frozen=True)
class AblationCell:
    run: ScalabilityRun
    treatment: Treatment
    pack: Path
    run_dir: Path


def build_ablation_cells(
    *,
    sizes: Sequence[int],
    fixed_size: int,
    seeds: Sequence[int],
    treatments: Sequence[Treatment],
    elicitation_pack_dir: Path,
    output_dir: Path,
) -> list[AblationCell]:
    """Return the Cartesian product of scalability cases and treatments."""
    runs = build_scalability_runs(
        sizes=sizes,
        fixed_size=fixed_size,
        seeds=seeds,
    )
    return [
        AblationCell(
            run=run,
            treatment=treatment,
            pack=(
                elicitation_pack_dir
                / f"seed_{run.seed}"
                / run.case_name
                / "frozen_elicitation.json"
            ),
            run_dir=(
                output_dir
                / f"seed_{run.seed}"
                / run.case_name
                / treatment.name
            ),
        )
        for run in runs
        for treatment in treatments
    ]


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-spec",
        type=Path,
        default=Path(
            "scenarios/pc_build_v3/pc_build_population_16x16.json"
        ),
    )
    parser.add_argument(
        "--elicitation-pack-dir",
        type=Path,
        default=Path("outputs/elicitation_packs/scalability"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/event_ablation_scalability"),
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9, 10]
    )
    parser.add_argument("--fixed-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--treatments",
        nargs="+",
        choices=[treatment.name for treatment in TREATMENTS],
        default=[treatment.name for treatment in TREATMENTS],
    )
    parser.add_argument("--sealed-rounds", type=int, default=10)
    parser.add_argument("--clock-rounds", type=int, default=20)
    parser.add_argument("--clock-top-k", type=int, default=3)
    parser.add_argument("--price-step", type=float, default=50.0)
    parser.add_argument("--pivotal-gap-threshold", type=float, default=100.0)
    parser.add_argument("--correction-threshold", type=float, default=0.25)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Concurrent deterministic auction subprocesses (default: 1). "
            "Try 2 first; OR-Tools work is CPU-intensive."
        ),
    )
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


def build_command(cell: AblationCell, args: argparse.Namespace) -> list[str]:
    run = cell.run
    return [
        sys.executable,
        "examples/run_live_llm_curated_batch.py",
        "--scenario", "pc_build",
        "--scenario-spec", str(args.scenario_spec),
        "--num-goods", str(run.num_goods),
        "--num-bidders", str(run.num_bidders),
        "--scenario-seed", str(run.seed),
        "--selection-policy", "coverage_stratified",
        "--seed-type", "structured",
        "--elicitation-pack", str(cell.pack),
        "--ask-initial-question",
        "--use-interest-map",
        "--use-provisional-valuations",
        "--person-query-mode", "deterministic",
        "--skip-baselines",
        "--sealed-elicitation-rounds", str(args.sealed_rounds),
        "--sealed-stopping-rule", "no_new_refinements",
        "--sealed-feedback-rule", "competitive",
        "--sealed-loser-challenger-policy", "shadow_price",
        "--elicited-clock",
        "--top-k", str(args.clock_top_k),
        "--clock-top-k-frontier-policy", "allocation_pivotal",
        "--clock-allocation-counterfactual-frontier",
        "--max-rounds", str(args.clock_rounds),
        "--price-step", str(args.price_step),
        "--event-pivotal-gap-threshold", str(args.pivotal_gap_threshold),
        "--event-correction-threshold", str(args.correction_threshold),
        "--llm-cache-mode", "off",
        "--log-dir", str(cell.run_dir),
        *cell.treatment.flags,
    ]


def _cell_label(cell: AblationCell) -> str:
    return (
        f"seed={cell.run.seed} case={cell.run.case_name} "
        f"treatment={cell.treatment.name}"
    )


def _execute_cell(
    cell: AblationCell,
    args: argparse.Namespace,
) -> dict[str, object]:
    summary = cell.run_dir / "curated_run_summary.csv"
    if summary.exists() and not args.rerun_complete:
        return {"cell": cell, "status": "complete", "elapsed": 0.0}

    cell.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = cell.run_dir / "ablation_runner.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            build_command(cell, args),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        while True:
            try:
                returncode = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = int(time.monotonic() - started)
                print(
                    f"  … {_cell_label(cell)} still running ({elapsed}s); "
                    f"details: {log_path}",
                    flush=True,
                )
    elapsed = time.monotonic() - started
    return {
        "cell": cell,
        "status": "complete" if returncode == 0 else "failed",
        "returncode": returncode,
        "elapsed": elapsed,
    }


def _x_value(run: ScalabilityRun) -> int:
    return run.num_bidders if run.series == "bidders" else run.num_goods


def _read_completed_cells(
    cells: Iterable[AblationCell],
    *,
    clock_top_k: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    run_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    incomplete: list[dict[str, object]] = []
    for cell in cells:
        summary_path = cell.run_dir / "curated_run_summary.csv"
        if not summary_path.exists():
            incomplete.append({
                "seed": cell.run.seed,
                "series": cell.run.series,
                "case": cell.run.case_name,
                "num_goods": cell.run.num_goods,
                "num_bidders": cell.run.num_bidders,
                "treatment": cell.treatment.name,
                "reason": "missing curated_run_summary.csv",
            })
            continue
        try:
            arm_rows = _read_arm_rows(summary_path)
            if len(arm_rows) != 2:
                raise ValueError(f"expected 2 proxy arms, found {len(arm_rows)}")
            for row in arm_rows:
                mechanism = (
                    "sealed" if row["arm"] == "proxy sealed" else "clock"
                )
                details = _mechanism_details(
                    cell.run_dir, mechanism, clock_top_k
                )
                full_info_revenue = _float(details, "full_info_revenue")
                revenue = _float(row, "revenue")
                run_rows.append({
                    "seed": cell.run.seed,
                    "series": cell.run.series,
                    "case": cell.run.case_name,
                    "x_value": _x_value(cell.run),
                    "num_goods": cell.run.num_goods,
                    "num_bidders": cell.run.num_bidders,
                    "treatment": cell.treatment.name,
                    "mechanism": mechanism,
                    "efficiency": _float(row, "efficiency"),
                    "true_welfare": _float(row, "true_welfare"),
                    "full_info_welfare": _float(row, "full_info_welfare"),
                    "revenue": revenue,
                    "full_info_revenue": full_info_revenue,
                    "revenue_abs_error": abs(revenue - full_info_revenue),
                    "revenue_abs_error_pct": (
                        100.0 * abs(revenue - full_info_revenue)
                        / full_info_revenue
                        if full_info_revenue else math.nan
                    ),
                    "allocation_match": _bool(details["allocation_match"]),
                    "surplus": _float(row, "surplus"),
                    "value_queries": _float(row, "vq"),
                    "demand_queries": _float(row, "dq"),
                    "nl_queries": _float(row, "nl"),
                    "run_dir": str(cell.run_dir),
                })
            case_events = _read_event_rows(
                cell.run_dir / "curated_refinement_records.csv",
                seed=cell.run.seed,
                treatment=cell.treatment.name,
            )
            for event in case_events:
                event.update({
                    "series": cell.run.series,
                    "case": cell.run.case_name,
                    "x_value": _x_value(cell.run),
                    "num_goods": cell.run.num_goods,
                    "num_bidders": cell.run.num_bidders,
                })
            event_rows.extend(case_events)
        except (KeyError, StopIteration, ValueError) as exc:
            incomplete.append({
                "seed": cell.run.seed,
                "series": cell.run.series,
                "case": cell.run.case_name,
                "num_goods": cell.run.num_goods,
                "num_bidders": cell.run.num_bidders,
                "treatment": cell.treatment.name,
                "reason": str(exc),
            })
    return run_rows, event_rows, incomplete


def _mean(rows: Sequence[dict[str, object]], field: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if not math.isnan(float(row[field]))
    ]
    return mean(values) if values else math.nan


def aggregate_outcomes(
    rows: Sequence[dict[str, object]],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key, members in sorted(grouped.items()):
        output.append({
            **dict(zip(group_fields, key)),
            "datasets": len(members),
            "seeds": len({str(row["seed"]) for row in members}),
            "mean_efficiency": _mean(members, "efficiency"),
            "mean_revenue": _mean(members, "revenue"),
            "mean_revenue_abs_error": _mean(members, "revenue_abs_error"),
            "mean_revenue_abs_error_pct": _mean(
                members, "revenue_abs_error_pct"
            ),
            "mean_surplus": _mean(members, "surplus"),
            "mean_value_queries": _mean(members, "value_queries"),
            "mean_demand_queries": _mean(members, "demand_queries"),
            "allocation_match_rate": mean(
                bool(row["allocation_match"]) for row in members
            ),
        })
    return output


def paired_deltas(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    keys = ("seed", "case", "mechanism")
    controls = {
        tuple(str(row[field]) for field in keys): row
        for row in rows
        if row["treatment"] == "control"
    }
    output: list[dict[str, object]] = []
    for row in rows:
        if row["treatment"] == "control":
            continue
        key = tuple(str(row[field]) for field in keys)
        control = controls.get(key)
        if control is None:
            continue
        output.append({
            "seed": row["seed"],
            "series": row["series"],
            "case": row["case"],
            "x_value": row["x_value"],
            "num_goods": row["num_goods"],
            "num_bidders": row["num_bidders"],
            "treatment": row["treatment"],
            "mechanism": row["mechanism"],
            "efficiency_delta_pp": 100.0 * (
                float(row["efficiency"]) - float(control["efficiency"])
            ),
            "revenue_abs_error_delta": (
                float(row["revenue_abs_error"])
                - float(control["revenue_abs_error"])
            ),
            "revenue_abs_error_pct_delta_pp": (
                float(row["revenue_abs_error_pct"])
                - float(control["revenue_abs_error_pct"])
            ),
            "value_query_delta": (
                float(row["value_queries"])
                - float(control["value_queries"])
            ),
            "allocation_match_delta": (
                int(bool(row["allocation_match"]))
                - int(bool(control["allocation_match"]))
            ),
        })
    return output


def aggregate_paired_deltas(
    rows: Sequence[dict[str, object]],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key, members in sorted(grouped.items()):
        efficiency = [float(row["efficiency_delta_pp"]) for row in members]
        output.append({
            **dict(zip(group_fields, key)),
            "paired_datasets": len(members),
            "seeds": len({str(row["seed"]) for row in members}),
            "mean_efficiency_delta_pp": mean(efficiency),
            "efficiency_wins": sum(value > 1e-9 for value in efficiency),
            "efficiency_ties": sum(abs(value) <= 1e-9 for value in efficiency),
            "efficiency_losses": sum(value < -1e-9 for value in efficiency),
            "mean_revenue_abs_error_delta": _mean(
                members, "revenue_abs_error_delta"
            ),
            "mean_revenue_abs_error_pct_delta_pp": _mean(
                members, "revenue_abs_error_pct_delta_pp"
            ),
            "mean_value_query_delta": _mean(members, "value_query_delta"),
            "mean_allocation_match_delta": _mean(
                members, "allocation_match_delta"
            ),
        })
    return output


def _plot_scaling(
    summary: Sequence[dict[str, object]],
    output_dir: Path,
) -> None:
    if not summary:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV outputs were still written.")
        return
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    treatments = [
        treatment.name for treatment in TREATMENTS
        if any(row["treatment"] == treatment.name for row in summary)
    ]
    for mechanism in ("sealed", "clock"):
        for metric, label, scale in (
            ("mean_efficiency", "Mean allocative efficiency (%)", 100.0),
            ("mean_value_queries", "Mean value queries", 1.0),
            (
                "mean_revenue_abs_error_pct",
                "Mean absolute VCG revenue error (%)",
                1.0,
            ),
        ):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
            for axis, series in zip(axes, ("goods", "bidders", "joint")):
                relevant = [
                    row for row in summary
                    if row["mechanism"] == mechanism
                    and row["series"] in {series, "anchor"}
                ]
                for treatment in treatments:
                    points = sorted(
                        (
                            int(row["x_value"]), scale * float(row[metric])
                        )
                        for row in relevant
                        if row["treatment"] == treatment
                    )
                    if points:
                        axis.plot(
                            [point[0] for point in points],
                            [point[1] for point in points],
                            marker="o",
                            linewidth=1.2,
                            label=treatment,
                        )
                axis.set_title(series)
                axis.set_xlabel("Number of goods/bidders")
                axis.grid(alpha=0.25)
            axes[0].set_ylabel(label)
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
            fig.tight_layout(rect=(0, 0, 0.84, 1))
            fig.savefig(
                plots / f"{mechanism}_{metric}_by_series.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)


def _plot_paired_scaling(
    summary: Sequence[dict[str, object]],
    output_dir: Path,
) -> None:
    """Plot treatment-minus-control effects along each scalability path."""
    if not summary:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    treatments = [
        treatment.name for treatment in TREATMENTS
        if treatment.name != "control"
        and any(row["treatment"] == treatment.name for row in summary)
    ]
    for mechanism in ("sealed", "clock"):
        for metric, label in (
            ("mean_efficiency_delta_pp", "Efficiency change vs control (pp)"),
            ("mean_value_query_delta", "Value-query change vs control"),
            (
                "mean_revenue_abs_error_pct_delta_pp",
                "VCG revenue-error change vs control (pp)",
            ),
        ):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
            for axis, series in zip(axes, ("goods", "bidders", "joint")):
                relevant = [
                    row for row in summary
                    if row["mechanism"] == mechanism
                    and row["series"] in {series, "anchor"}
                ]
                for treatment in treatments:
                    points = sorted(
                        (
                            int(row["x_value"]), float(row[metric])
                        )
                        for row in relevant
                        if row["treatment"] == treatment
                    )
                    if points:
                        axis.plot(
                            [point[0] for point in points],
                            [point[1] for point in points],
                            marker="o",
                            linewidth=1.2,
                            label=treatment,
                        )
                axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
                axis.set_title(series)
                axis.set_xlabel("Number of goods/bidders")
                axis.grid(alpha=0.25)
            axes[0].set_ylabel(label)
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
            fig.tight_layout(rect=(0, 0, 0.84, 1))
            fig.savefig(
                plots / f"{mechanism}_{metric}_by_series.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)


def _write_outputs(
    args: argparse.Namespace,
    cells: Sequence[AblationCell],
) -> tuple[int, int]:
    run_rows, event_rows, incomplete = _read_completed_cells(
        cells, clock_top_k=args.clock_top_k
    )
    _write_csv(args.output_dir / "event_ablation_runs.csv", run_rows)
    pooled = aggregate_outcomes(
        run_rows, group_fields=("treatment", "mechanism")
    )
    _write_csv(args.output_dir / "event_ablation_summary.csv", pooled)
    series = aggregate_outcomes(
        run_rows,
        group_fields=("treatment", "mechanism", "series", "x_value"),
    )
    _write_csv(args.output_dir / "event_ablation_series_summary.csv", series)
    deltas = paired_deltas(run_rows)
    _write_csv(args.output_dir / "event_ablation_paired_deltas.csv", deltas)
    _write_csv(
        args.output_dir / "event_ablation_paired_summary.csv",
        aggregate_paired_deltas(
            deltas, group_fields=("treatment", "mechanism")
        ),
    )
    _write_csv(
        args.output_dir / "event_ablation_paired_series_summary.csv",
        paired_series := aggregate_paired_deltas(
            deltas,
            group_fields=(
                "treatment", "mechanism", "series", "x_value"
            ),
        ),
    )
    _write_csv(
        args.output_dir / "event_ablation_paired_seed_summary.csv",
        aggregate_paired_deltas(
            deltas, group_fields=("treatment", "mechanism", "seed")
        ),
    )
    _write_csv(args.output_dir / "event_ablation_event_rows.csv", event_rows)
    _write_csv(
        args.output_dir / "event_ablation_event_summary.csv",
        _aggregate_events(event_rows),
    )
    _write_csv(args.output_dir / "event_ablation_incomplete.csv", incomplete)
    _plot_scaling(series, args.output_dir)
    _plot_paired_scaling(paired_series, args.output_dir)
    return len(run_rows), len(incomplete)


def main(argv: Sequence[str] | None = None) -> None:
    args = _args(argv)
    selected = [
        treatment for treatment in TREATMENTS
        if treatment.name in set(args.treatments)
    ]
    cells = build_ablation_cells(
        sizes=args.sizes,
        fixed_size=args.fixed_size,
        seeds=args.seeds,
        treatments=selected,
        elicitation_pack_dir=args.elicitation_pack_dir,
        output_dir=args.output_dir,
    )
    missing_packs = sorted({cell.pack for cell in cells if not cell.pack.exists()})
    if missing_packs:
        preview = "\n".join(f"  - {path}" for path in missing_packs[:10])
        suffix = "" if len(missing_packs) <= 10 else f"\n  ... and {len(missing_packs)-10} more"
        raise FileNotFoundError(
            f"Missing {len(missing_packs)} frozen packs:\n{preview}{suffix}"
        )

    print(
        f"Ablation plan: {len(cells)} treatment cells, "
        f"{len(cells) * 2} mechanism observations, jobs={args.jobs}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "event_ablation_config.json").write_text(
        json.dumps({
            "scenario_spec": str(args.scenario_spec),
            "elicitation_pack_dir": str(args.elicitation_pack_dir),
            "sizes": args.sizes,
            "fixed_size": args.fixed_size,
            "seeds": args.seeds,
            "treatments": args.treatments,
            "sealed_rounds": args.sealed_rounds,
            "clock_rounds": args.clock_rounds,
            "clock_top_k": args.clock_top_k,
            "price_step": args.price_step,
            "pivotal_gap_threshold": args.pivotal_gap_threshold,
            "correction_threshold": args.correction_threshold,
            "jobs": args.jobs,
            "cells": len(cells),
            "mechanism_observations": len(cells) * 2,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        for cell in cells:
            print(" ".join(build_command(cell, args)))
        return

    failures: list[dict[str, object]] = []
    if not args.aggregate_only:
        executor = ThreadPoolExecutor(max_workers=args.jobs)
        pending_cells = iter(cells)
        future_cells = {}
        for _ in range(min(args.jobs, len(cells))):
            cell = next(pending_cells)
            future_cells[executor.submit(_execute_cell, cell, args)] = cell
        try:
            completed = 0
            while future_cells:
                future = next(as_completed(future_cells))
                cell = future_cells.pop(future)
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:  # preserve remaining independent cells
                    result = {"status": "failed", "error": repr(exc)}
                status = str(result["status"])
                elapsed = float(result.get("elapsed", 0.0))
                marker = "✓" if status == "complete" else "✗"
                print(
                    f"{marker} [{completed}/{len(cells)}] {_cell_label(cell)} "
                    f"{status} ({elapsed:.1f}s)",
                    flush=True,
                )
                if status == "failed":
                    failures.append({
                        "seed": cell.run.seed,
                        "case": cell.run.case_name,
                        "treatment": cell.treatment.name,
                        "returncode": result.get("returncode"),
                        "error": result.get("error"),
                    })
                next_cell = next(pending_cells, None)
                if next_cell is not None:
                    future_cells[
                        executor.submit(_execute_cell, next_cell, args)
                    ] = next_cell
        except KeyboardInterrupt:
            for future in future_cells:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            print(
                "Interrupted after active cells finished; completed outputs "
                "remain resumable.",
                flush=True,
            )
            raise
        else:
            executor.shutdown(wait=True)

    if not args.aggregate_only:
        (args.output_dir / "event_ablation_failures.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
    observations, incomplete = _write_outputs(args, cells)
    print(
        f"Wrote {observations} mechanism observations; "
        f"{len(failures)} failed cells; {incomplete} incomplete cells."
    )


if __name__ == "__main__":
    main()
