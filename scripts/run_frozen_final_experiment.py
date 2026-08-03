#!/usr/bin/env python3
"""Verify a frozen final specification and execute its paired scalability suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.experiments.final_pipeline import (  # noqa: E402
    load_final_experiment_spec,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--mechanisms",
        choices=["both", "sealed", "clock"],
        default="both",
        help="Run both elicited mechanisms or only one arm.",
    )
    return parser.parse_args(argv)


def build_command(
    spec: dict,
    *,
    output_dir: Path,
    rerun_complete: bool = False,
    fail_fast: bool = False,
    mechanisms: str = "both",
) -> list[str]:
    dataset = spec["dataset"]
    clock = spec["clock"]
    sealed = spec["sealed"]
    command = [
        sys.executable,
        "scripts/run_scalability_experiment.py",
        "--scenario-spec", spec["scenario"]["path"],
        "--output-dir", str(output_dir),
        "--sizes", *[str(value) for value in dataset["sizes"]],
        "--fixed-size", str(dataset["fixed_size"]),
        "--seeds", *[str(value) for value in dataset["seeds"]],
        "--selection-policy", spec["scenario"]["selection_policy"],
        "--elicitation-pack-dir", dataset["elicitation_pack_dir"],
    ]
    if rerun_complete:
        command.append("--rerun-complete")
    if fail_fast:
        command.append("--fail-fast")
    command.extend([
        "--",
        "--ask-initial-question",
        "--use-interest-map",
        "--use-provisional-valuations",
        "--person-query-mode", "deterministic",
        "--skip-baselines",
        "--pv-calibration-config", spec["calibration"]["path"],
        "--event-policy", spec["event_policy"]["name"],
        "--event-correction-threshold",
        str(spec["event_policy"]["correction_threshold"]),
        "--sealed-elicitation-rounds",
        str(sealed["max_rounds"] if mechanisms != "clock" else 0),
        "--sealed-stopping-rule", sealed["stopping_rule"],
    ])
    if mechanisms != "sealed":
        command.extend([
            "--elicited-clock",
            "--top-k", str(clock["top_k"]),
            "--max-rounds", str(clock["max_rounds"]),
            "--price-step", str(clock["price_step"]),
            "--clock-tie-threshold", str(clock["tie_threshold"]),
        ])
    command.extend(["--llm-cache-mode", "off"])
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = load_final_experiment_spec(args.spec, verify_files=True)
    command = build_command(
        spec,
        output_dir=args.output_dir,
        rerun_complete=args.rerun_complete,
        fail_fast=args.fail_fast,
        mechanisms=args.mechanisms,
    )
    print("Verified every frozen input hash.")
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
