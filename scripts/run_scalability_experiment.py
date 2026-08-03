#!/usr/bin/env python3
"""Run goods, bidders, and joint scalability paths over a nested size grid.

Arguments after ``--`` are passed to ``examples/run_live_llm_curated_batch.py``.
The runner itself supplies the scenario, size, seed, selection policy, spec,
and per-case log directory.

Example::

    ./venv/bin/python scripts/run_scalability_experiment.py \
      --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
      --output-dir outputs/scalability/sealed_r3 \
      --seeds 0 1 2 \
      --sizes 4 5 6 7 8 9 10 \
      -- \
      --provider gemini --model gemini-3.5-flash-lite \
      --skip-baselines --sealed-elicitation-rounds 3
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ScalabilityRun:
    seed: int
    series: str
    num_goods: int
    num_bidders: int
    case_name: str


def build_scalability_runs(
    *,
    sizes: Sequence[int],
    fixed_size: int,
    seeds: Sequence[int],
) -> list[ScalabilityRun]:
    """Build three scaling paths, sharing one anchor per seed."""
    normalized = sorted(set(sizes))
    if not normalized:
        raise ValueError("sizes must not be empty")
    if any(size < 1 for size in normalized):
        raise ValueError("sizes must all be positive")
    if fixed_size not in normalized:
        raise ValueError("fixed_size must also appear in sizes")
    if not seeds:
        raise ValueError("seeds must not be empty")

    runs: list[ScalabilityRun] = []
    for seed in seeds:
        for size in normalized:
            if size == fixed_size:
                runs.append(
                    ScalabilityRun(
                        seed=seed,
                        series="anchor",
                        num_goods=fixed_size,
                        num_bidders=fixed_size,
                        case_name=f"anchor_{fixed_size}x{fixed_size}",
                    )
                )
                continue
            runs.extend(
                [
                    ScalabilityRun(
                        seed=seed,
                        series="goods",
                        num_goods=size,
                        num_bidders=fixed_size,
                        case_name=f"goods_{size}x{fixed_size}",
                    ),
                    ScalabilityRun(
                        seed=seed,
                        series="bidders",
                        num_goods=fixed_size,
                        num_bidders=size,
                        case_name=f"bidders_{fixed_size}x{size}",
                    ),
                    ScalabilityRun(
                        seed=seed,
                        series="joint",
                        num_goods=size,
                        num_bidders=size,
                        case_name=f"joint_{size}x{size}",
                    ),
                ]
            )
    return runs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7, 8, 9, 10],
        help="Selected goods/bidder counts (default: every integer 4 through 10).",
    )
    parser.add_argument("--fixed-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--selection-policy",
        choices=["seeded_sample", "stratified", "coverage_stratified"],
        default="coverage_stratified",
        help=(
            "Seed-sensitive nested selection policy "
            "(default: coverage_stratified)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Rerun cases that already contain curated_run_summary.csv.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed case instead of continuing the grid.",
    )
    parser.add_argument(
        "--skip-economic-preflight",
        action="store_true",
        help=(
            "Skip full-information allocation checks during preflight. "
            "Structural coverage checks still run for coverage_stratified."
        ),
    )
    parser.add_argument(
        "--elicitation-pack-dir",
        type=Path,
        default=None,
        help=(
            "Replay one frozen pack per case from "
            "DIR/seed_<seed>/<case>/frozen_elicitation.json. This keeps "
            "the initial person/proxy state deterministic across repeated "
            "scalability runs."
        ),
    )
    parser.add_argument(
        "live_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- forwarded to the live auction runner.",
    )
    args = parser.parse_args(argv)
    if args.live_args and args.live_args[0] == "--":
        args.live_args = args.live_args[1:]
    return args


_MANAGED_FLAGS = {
    "--scenario",
    "--scenario-spec",
    "--num-goods",
    "--num-bidders",
    "--scenario-seed",
    "--selection-policy",
    "--log-dir",
}


def _validate_live_args(live_args: Sequence[str]) -> None:
    conflicting = sorted(
        {
            token.split("=", 1)[0]
            for token in live_args
            if token.split("=", 1)[0] in _MANAGED_FLAGS
        }
    )
    if conflicting:
        raise ValueError(
            "these arguments are managed by the scalability runner and "
            f"must not be passed after --: {', '.join(conflicting)}"
        )


def build_command(
    run: ScalabilityRun,
    *,
    scenario_spec: Path,
    selection_policy: str,
    case_dir: Path,
    live_args: Sequence[str],
    elicitation_pack: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "examples/run_live_llm_curated_batch.py",
        "--scenario",
        "pc_build",
        "--scenario-spec",
        str(scenario_spec),
        "--num-goods",
        str(run.num_goods),
        "--num-bidders",
        str(run.num_bidders),
        "--scenario-seed",
        str(run.seed),
        "--selection-policy",
        selection_policy,
        "--log-dir",
        str(case_dir),
    ]
    if elicitation_pack is not None:
        command += ["--elicitation-pack", str(elicitation_pack)]
    return [*command, *live_args]


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "series",
        "case",
        "num_goods",
        "num_bidders",
        "status",
        "return_code",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_coverage_preflight(
    args: argparse.Namespace,
    runs: Sequence[ScalabilityRun],
) -> dict[str, object]:
    from auctionlab.instances.population_design import (
        population_coverage_report,
        validate_nested_scalability_samples,
    )
    from auctionlab.instances.scenario_spec import load_scenario_profile_spec

    spec = load_scenario_profile_spec(args.scenario_spec)
    generation = spec.generation if isinstance(spec.generation, dict) else {}
    sample_constraints = generation.get("sample_constraints")
    population_constraints = generation.get("population_constraints")
    population = population_coverage_report(
        spec,
        constraints=population_constraints,
    )
    samples = validate_nested_scalability_samples(
        spec,
        seeds=args.seeds,
        sizes=args.sizes,
        fixed_size=args.fixed_size,
        constraints=sample_constraints,
        include_economic=not args.skip_economic_preflight,
    )
    return {
        "scenario_spec": str(args.scenario_spec),
        "selection_policy": args.selection_policy,
        "population": population,
        "sample_validation": samples,
        "cases": samples["cases"],
        "passed": population["passed"] and samples["passed"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_live_args(args.live_args)
        runs = build_scalability_runs(
            sizes=args.sizes,
            fixed_size=args.fixed_size,
            seeds=args.seeds,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if args.selection_policy == "coverage_stratified":
        try:
            preflight = _run_coverage_preflight(args, runs)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Coverage preflight failed: {exc}") from exc
        preflight_path = args.output_dir / "scalability_preflight.json"
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(
            json.dumps(preflight, indent=2),
            encoding="utf-8",
        )
        if not preflight["passed"]:
            print(f"Coverage preflight FAILED: {preflight_path}")
            for violation in preflight["population"]["violations"]:
                print(f"  - population: {violation}")
            for violation in preflight["sample_validation"]["violations"]:
                print(f"  - sample: {violation}")
            return 2
        print(f"Coverage preflight passed: {preflight_path}")

    manifest_rows: list[dict[str, object]] = []
    failures = 0
    for run in runs:
        case_dir = args.output_dir / f"seed_{run.seed}" / run.case_name
        summary_path = case_dir / "curated_run_summary.csv"
        if summary_path.exists() and not args.rerun_complete:
            status = "skipped_complete"
            return_code = 0
            print(f"SKIP {run.case_name} seed={run.seed}: complete")
        else:
            command = build_command(
                run,
                scenario_spec=args.scenario_spec,
                selection_policy=args.selection_policy,
                case_dir=case_dir,
                live_args=args.live_args,
                elicitation_pack=(
                    None
                    if args.elicitation_pack_dir is None
                    else (
                        args.elicitation_pack_dir
                        / f"seed_{run.seed}"
                        / run.case_name
                        / "frozen_elicitation.json"
                    )
                ),
            )
            print(shlex.join(command), flush=True)
            if args.dry_run:
                status = "dry_run"
                return_code = 0
            else:
                case_dir.mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(command, check=False)
                return_code = completed.returncode
                status = "completed" if return_code == 0 else "failed"
                failures += int(return_code != 0)

        manifest_rows.append(
            {
                "seed": run.seed,
                "series": run.series,
                "case": run.case_name,
                "num_goods": run.num_goods,
                "num_bidders": run.num_bidders,
                "status": status,
                "return_code": return_code,
                "output_dir": case_dir,
            }
        )
        _write_manifest(args.output_dir / "scalability_runs.csv", manifest_rows)
        if return_code != 0 and args.fail_fast:
            break

    print(
        f"Scalability cases: {len(manifest_rows)} processed, "
        f"{failures} failed. Manifest: "
        f"{args.output_dir / 'scalability_runs.csv'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
