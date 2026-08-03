#!/usr/bin/env python3
"""Build aggregate tables and plots from scalability experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from auctionlab.experiments.scalability_analysis import (
    load_scalability_results,
    plot_scalability_results,
    write_scalability_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot scalability experiment results."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root containing seed_*/goods_*, bidders_*, joint_*, and anchor_* cases.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination (default: INPUT_DIR/analysis).",
    )
    parser.add_argument(
        "--arm",
        default=None,
        help="Case-insensitive arm substring when summaries contain multiple auction arms.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "svg", "pdf"),
        default="png",
        help="Plot image format (default: png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir or input_dir / "analysis")

    cases, incomplete = load_scalability_results(
        input_dir,
        arm_filter=args.arm,
    )
    if not cases:
        raise SystemExit(
            f"No completed scalability cases found below {input_dir}"
        )

    results_path, incomplete_path = write_scalability_tables(
        output_dir,
        cases,
        incomplete,
    )
    try:
        plots = plot_scalability_results(
            output_dir,
            cases,
            image_format=args.format,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Completed cases: {len(cases)}")
    print(f"Incomplete cases: {len(incomplete)}")
    for row in incomplete:
        print(f"  - {row['case']}: {row['reason']}")
    print(f"Metrics CSV: {results_path}")
    print(f"Incomplete report: {incomplete_path}")
    print(f"Plots: {output_dir / 'plots'} ({len(plots)} files)")


if __name__ == "__main__":
    main()
