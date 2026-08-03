#!/usr/bin/env python3
"""Create the immutable, content-addressed final experiment specification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.experiments.final_pipeline import (  # noqa: E402
    build_final_experiment_spec,
    write_final_experiment_spec,
)


def _parse_model(value: str) -> dict[str, str]:
    try:
        provider, model = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "model must be PROVIDER:MODEL"
        ) from exc
    if not provider or not model:
        raise argparse.ArgumentTypeError("model must be PROVIDER:MODEL")
    return {"provider": provider, "model": model}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-spec", type=Path, required=True)
    parser.add_argument("--elicitation-pack-dir", type=Path, required=True)
    parser.add_argument("--pv-calibration-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--fixed-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sealed-max-rounds", type=int, default=40)
    parser.add_argument("--correction-threshold", type=float, default=0.25)
    parser.add_argument("--clock-max-rounds", type=int, default=50)
    parser.add_argument("--clock-price-step", type=float, required=True)
    parser.add_argument("--clock-top-k", type=int, required=True)
    parser.add_argument("--clock-tie-threshold", type=float, required=True)
    parser.add_argument(
        "--robustness-model",
        action="append",
        type=_parse_model,
        default=[],
        help="Alternative proxy model as PROVIDER:MODEL; repeat as needed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = build_final_experiment_spec(
        scenario_spec=args.scenario_spec,
        elicitation_pack_dir=args.elicitation_pack_dir,
        calibration_config=args.pv_calibration_config,
        seeds=args.seeds,
        sizes=args.sizes,
        fixed_size=args.fixed_size,
        sealed_max_rounds=args.sealed_max_rounds,
        correction_threshold=args.correction_threshold,
        clock_max_rounds=args.clock_max_rounds,
        clock_price_step=args.clock_price_step,
        clock_top_k=args.clock_top_k,
        clock_tie_threshold=args.clock_tie_threshold,
        robustness_models=args.robustness_model,
    )
    path = write_final_experiment_spec(args.output, spec)
    print(
        f"Frozen final experiment: {path}\n"
        f"  cases: {spec['dataset']['case_count']}\n"
        f"  calibration: {spec['calibration']['effective_config_hash'][:16]}\n"
        f"  clock: {spec['clock']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
