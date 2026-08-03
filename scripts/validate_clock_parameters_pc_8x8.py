#!/usr/bin/env python3
"""Verify calibrated dimensionless clock parameters on frozen PC 8x8 anchors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-spec", type=Path, required=True)
    parser.add_argument("--elicitation-pack-dir", type=Path, required=True)
    parser.add_argument("--pv-calibration-config", type=Path, required=True)
    parser.add_argument("--parameter-recommendation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--nominal-price-step", type=float, default=50.0)
    parser.add_argument("--nominal-top-k", type=int, default=3)
    parser.add_argument("--nominal-tie-threshold", type=float, default=100.0)
    parser.add_argument("--comparison-price-step", type=float, default=25.0)
    parser.add_argument(
        "--treatments",
        nargs="+",
        choices=("targeted_nominal", "targeted_step_comparison", "targeted_tuned"),
        default=("targeted_nominal", "targeted_step_comparison", "targeted_tuned"),
    )
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _budget_reference(
    scenario_spec: dict[str, Any], pack: dict[str, Any]
) -> float:
    budgets = {
        profile["bidder_id"]: float(profile["budget_cap"])
        for profile in scenario_spec["bidder_profiles"]
    }
    bidder_ids = pack["scenario"]["bidder_ids"]
    return statistics.median(budgets[bidder] for bidder in bidder_ids) / len(
        pack["scenario"]["items"]
    )


def _command(
    args: argparse.Namespace,
    *,
    seed: int,
    pack_path: Path,
    run_dir: Path,
    price_step: float,
    top_k: int,
    tie_threshold: float,
) -> list[str]:
    return [
        sys.executable,
        "examples/run_live_llm_curated_batch.py",
        "--scenario", "pc_build",
        "--scenario-spec", str(args.scenario_spec),
        "--num-goods", "8",
        "--num-bidders", "8",
        "--scenario-seed", str(seed),
        "--selection-policy", "coverage_stratified",
        "--seed-type", "structured",
        "--elicitation-pack", str(pack_path),
        "--person-query-mode", "deterministic",
        "--skip-baselines",
        "--elicited-clock",
        "--top-k", str(top_k),
        "--max-rounds", str(args.max_rounds),
        "--price-step", str(price_step),
        "--clock-tie-threshold", str(tie_threshold),
        "--event-policy", "final-v1",
        "--pv-calibration-config", str(args.pv_calibration_config),
        "--llm-cache-mode", "off",
        "--log-dir", str(run_dir),
    ]


def _collect(
    *,
    treatment: str,
    seed: int,
    reference: float,
    price_step: float,
    top_k: int,
    tie_threshold: float,
    run_dir: Path,
) -> dict[str, Any]:
    summary = _read_csv(run_dir / "curated_run_summary.csv")
    arm = next(row for row in summary if row["arm"].startswith("proxy clock"))
    detail = _read_csv(next(run_dir.glob("curated_clock_proxy_elicited_top_*.csv")))[0]
    return {
        "treatment": treatment,
        "seed": seed,
        "budget_per_good_reference": reference,
        "price_step": price_step,
        "top_k": top_k,
        "tie_threshold": tie_threshold,
        "efficiency": float(arm["efficiency"]),
        "payment_error_over_optimum_welfare": float(
            detail["payment_error_over_optimum_welfare"]
        ),
        "revenue_loss": float(detail["revenue_loss"]),
        "value_queries": float(arm["vq"]),
        "rounds": float(detail["rounds"]),
        "allocation_match": detail["allocation_match"],
        "run_dir": str(run_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = json.loads(args.scenario_spec.read_text(encoding="utf-8"))
    recommendation = json.loads(
        args.parameter_recommendation.read_text(encoding="utf-8")
    )["selected"]
    failures: list[dict[str, Any]] = []
    planned: list[tuple[str, int, float, int, float, float, Path, Path]] = []
    for seed in args.seeds:
        pack_path = (
            args.elicitation_pack_dir
            / f"seed_{seed}"
            / "anchor_8x8"
            / "frozen_elicitation.json"
        )
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        reference = _budget_reference(spec, pack)
        settings = (
            (
                "targeted_nominal",
                args.nominal_price_step,
                args.nominal_top_k,
                args.nominal_tie_threshold,
            ),
            (
                "targeted_step_comparison",
                args.comparison_price_step,
                args.nominal_top_k,
                args.nominal_tie_threshold,
            ),
            (
                "targeted_tuned",
                float(recommendation["price_step_fraction"]) * reference,
                int(recommendation["top_k"]),
                float(recommendation["tie_threshold_fraction"]) * reference,
            ),
        )
        for treatment, price_step, top_k, tie_threshold in settings:
            if treatment not in args.treatments:
                continue
            run_dir = args.output_dir / treatment / f"seed_{seed}"
            planned.append((
                treatment, seed, price_step, top_k, tie_threshold,
                reference, pack_path, run_dir,
            ))

    for index, row in enumerate(planned, start=1):
        treatment, seed, price_step, top_k, tie_threshold, _, pack, run_dir = row
        summary_path = run_dir / "curated_run_summary.csv"
        if summary_path.exists() and not args.rerun_complete:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(planned)}] {treatment} seed={seed}: "
            f"step={price_step:.3f}, top_k={top_k}, tie={tie_threshold:.3f}",
            flush=True,
        )
        with (run_dir / "verification_runner.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                _command(
                    args,
                    seed=seed,
                    pack_path=pack,
                    run_dir=run_dir,
                    price_step=price_step,
                    top_k=top_k,
                    tie_threshold=tie_threshold,
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            failures.append({
                "treatment": treatment,
                "seed": seed,
                "returncode": result.returncode,
            })
            if args.fail_fast:
                break

    rows = [
        _collect(
            treatment=treatment,
            seed=seed,
            reference=reference,
            price_step=price_step,
            top_k=top_k,
            tie_threshold=tie_threshold,
            run_dir=run_dir,
        )
        for (
            treatment,
            seed,
            price_step,
            top_k,
            tie_threshold,
            reference,
            _,
            run_dir,
        ) in planned
        if (run_dir / "curated_run_summary.csv").exists()
    ]
    summaries: list[dict[str, Any]] = []
    for treatment in args.treatments:
        members = [row for row in rows if row["treatment"] == treatment]
        summary: dict[str, Any] = {"treatment": treatment, "seeds": len(members)}
        for metric in (
            "efficiency",
            "payment_error_over_optimum_welfare",
            "revenue_loss",
            "value_queries",
            "rounds",
        ):
            values = [float(row[metric]) for row in members]
            summary[f"mean_{metric}"] = (
                statistics.fmean(values) if values else math.nan
            )
        summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "pc_clock_parameter_verification.csv", rows)
    _write_csv(args.output_dir / "pc_clock_parameter_summary.csv", summaries)
    (args.output_dir / "pc_clock_parameter_failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
