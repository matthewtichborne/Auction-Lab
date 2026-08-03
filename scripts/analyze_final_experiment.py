#!/usr/bin/env python3
"""Build final paired metrics and pre-specified illustrative trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.experiments.final_pipeline import (  # noqa: E402
    load_final_experiment_spec,
)
from auctionlab.experiments.scalability_analysis import (  # noqa: E402
    ScalabilityCase,
    load_scalability_results,
    write_scalability_tables,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


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


def _key(case: ScalabilityCase) -> tuple[str, str]:
    return str(case.values["seed"]), str(case.values["case"])


def paired_rows(
    sealed: Sequence[ScalabilityCase],
    clock: Sequence[ScalabilityCase],
) -> list[dict[str, Any]]:
    sealed_by_key = {_key(case): case.values for case in sealed}
    clock_by_key = {_key(case): case.values for case in clock}
    if set(sealed_by_key) != set(clock_by_key):
        raise ValueError("sealed and clock completed-case sets differ")
    rows: list[dict[str, Any]] = []
    for key in sorted(sealed_by_key, key=lambda value: (int(value[0]), value[1])):
        sealed_row = sealed_by_key[key]
        clock_row = clock_by_key[key]
        rows.append(
            {
                "seed": key[0],
                "case": key[1],
                "series": sealed_row["series"],
                "num_goods": sealed_row["num_goods"],
                "num_bidders": sealed_row["num_bidders"],
                "initial_efficiency": sealed_row["initial_efficiency"],
                "sealed_efficiency": sealed_row["efficiency"],
                "clock_efficiency": clock_row["efficiency"],
                "clock_minus_sealed_efficiency": (
                    float(clock_row["efficiency"])
                    - float(sealed_row["efficiency"])
                ),
                "sealed_efficiency_gain_pct": sealed_row[
                    "efficiency_gain_from_initial_pct"
                ],
                "clock_efficiency_gain_pct": clock_row[
                    "efficiency_gain_from_initial_pct"
                ],
                "initial_revenue_loss": sealed_row["initial_revenue_loss"],
                "sealed_revenue_loss": sealed_row["revenue_loss"],
                "clock_revenue_loss": clock_row["revenue_loss"],
                "sealed_revenue_loss_improvement": sealed_row[
                    "revenue_loss_improvement_from_initial"
                ],
                "clock_revenue_loss_improvement": clock_row[
                    "revenue_loss_improvement_from_initial"
                ],
                "sealed_payment_error_over_optimum_welfare": sealed_row[
                    "payment_error_over_optimum_welfare"
                ],
                "clock_payment_error_over_optimum_welfare": clock_row[
                    "payment_error_over_optimum_welfare"
                ],
                "sealed_value_queries": sealed_row["value_queries"],
                "clock_value_queries": clock_row["value_queries"],
            }
        )
    return rows


def _finite(row: dict[str, Any], key: str) -> bool:
    try:
        return math.isfinite(float(row[key]))
    except (KeyError, TypeError, ValueError):
        return False


def select_examples(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the four selection rules frozen in the final specification."""
    selected: list[dict[str, Any]] = []

    efficiency_candidates = [
        row for row in rows if _finite(row, "sealed_efficiency_gain_pct")
    ]
    if efficiency_candidates:
        selected.append({
            "selection_rule": "largest_sealed_efficiency_improvement",
            **max(
                efficiency_candidates,
                key=lambda row: (
                    float(row["sealed_efficiency_gain_pct"]),
                    -int(row["seed"]),
                    str(row["case"]),
                ),
            ),
        })

    revenue_candidates = [
        row
        for row in rows
        if _finite(row, "sealed_revenue_loss_improvement")
    ]
    if revenue_candidates:
        selected.append({
            "selection_rule": "largest_sealed_revenue_loss_reduction",
            **max(
                revenue_candidates,
                key=lambda row: (
                    float(row["sealed_revenue_loss_improvement"]),
                    -int(row["seed"]),
                    str(row["case"]),
                ),
            ),
        })

    full_efficiency = [
        row
        for row in rows
        if _finite(row, "initial_efficiency")
        and float(row["initial_efficiency"]) < 1.0 - 1e-9
        and float(row["sealed_efficiency"]) >= 1.0 - 1e-9
    ]
    if full_efficiency:
        median_queries = statistics.median(
            float(row["sealed_value_queries"]) for row in full_efficiency
        )
        selected.append({
            "selection_rule": (
                "median_query_case_reaching_full_efficiency_from_below"
            ),
            **min(
                full_efficiency,
                key=lambda row: (
                    abs(float(row["sealed_value_queries"]) - median_queries),
                    int(row["seed"]),
                    str(row["case"]),
                ),
            ),
        })

    non_improving = [
        row
        for row in rows
        if _finite(row, "sealed_efficiency_gain_pct")
        and float(row["sealed_efficiency_gain_pct"]) <= 1e-12
    ]
    if non_improving:
        median_queries = statistics.median(
            float(row["sealed_value_queries"]) for row in non_improving
        )
        selected.append({
            "selection_rule": "representative_non_improving_case",
            **min(
                non_improving,
                key=lambda row: (
                    abs(float(row["sealed_value_queries"]) - median_queries),
                    abs(float(row["sealed_efficiency_gain_pct"])),
                    int(row["seed"]),
                    str(row["case"]),
                ),
            ),
        })
    return selected


def _trajectory_rows(
    input_dir: Path, examples: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        path = (
            input_dir
            / f"seed_{example['seed']}"
            / str(example["case"])
            / "curated_proxy_sealed_trajectory.csv"
        )
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "selection_rule": example["selection_rule"],
                        "seed": example["seed"],
                        "case": example["case"],
                        **row,
                    }
                )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = load_final_experiment_spec(args.spec, verify_files=True)
    sealed, sealed_incomplete = load_scalability_results(
        args.input_dir, arm_filter="proxy sealed"
    )
    clock, clock_incomplete = load_scalability_results(
        args.input_dir, arm_filter="proxy clock"
    )
    expected = int(spec["dataset"]["case_count"])
    if sealed_incomplete or clock_incomplete:
        raise SystemExit("Final suite contains incomplete cases")
    if len(sealed) != expected or len(clock) != expected:
        raise SystemExit(
            f"Expected {expected} paired cases; found sealed={len(sealed)}, "
            f"clock={len(clock)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_scalability_tables(args.output_dir / "sealed", sealed, [])
    write_scalability_tables(args.output_dir / "clock", clock, [])
    paired = paired_rows(sealed, clock)
    examples = select_examples(paired)
    trajectories = _trajectory_rows(args.input_dir, examples)
    _write_csv(args.output_dir / "paired_mechanism_metrics.csv", paired)
    _write_csv(args.output_dir / "selected_examples.csv", examples)
    _write_csv(args.output_dir / "selected_sealed_trajectories.csv", trajectories)
    (args.output_dir / "selected_examples.json").write_text(
        json.dumps(examples, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Analysed {len(paired)} paired cases; selected "
        f"{len(examples)} rule-based examples."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
