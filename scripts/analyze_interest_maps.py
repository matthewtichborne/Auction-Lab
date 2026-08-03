#!/usr/bin/env python3
"""Aggregate and plot interest-map diagnostics from frozen elicitation packs."""

from __future__ import annotations

import argparse
from pathlib import Path

from auctionlab.experiments.interest_map_analysis import (
    aggregate_cases,
    aggregate_summary,
    load_interest_map_results,
    plot_interest_map_results,
    write_interest_map_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline candidate-support and reconstruction analysis for frozen "
            "interest maps. This command makes no LLM calls."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Frozen-pack file or root containing scalability case packs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination for CSV tables and plots.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "svg", "pdf"),
        default="png",
        help="Plot image format (default: png).",
    )
    parser.add_argument(
        "--include-masters",
        action="store_true",
        help=(
            "Include catalogue master packs as additional observations. "
            "Disabled by default to avoid double-counting projected bidders."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bidder_rows, invalid = load_interest_map_results(
        args.input_dir,
        include_masters=args.include_masters,
    )
    if not bidder_rows:
        raise SystemExit(
            f"No valid frozen elicitation packs found below {args.input_dir}"
        )
    case_rows = aggregate_cases(bidder_rows)
    summary_rows = aggregate_summary(case_rows)
    paths = write_interest_map_tables(
        args.output_dir,
        bidder_rows,
        case_rows,
        summary_rows,
        invalid,
    )
    try:
        plots = plot_interest_map_results(
            args.output_dir,
            case_rows,
            image_format=args.format,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print("Interest-map analysis completed without LLM calls.")
    print(f"Bidder observations: {len(bidder_rows)}")
    print(f"Auction cases: {len(case_rows)}")
    print(f"Invalid packs: {len(invalid)}")
    print(f"Bidder metrics: {paths[0]}")
    print(f"Case metrics: {paths[1]}")
    print(f"Summary: {paths[2]}")
    print(f"Invalid-pack report: {paths[3]}")
    print(f"Plots: {Path(args.output_dir) / 'plots'} ({len(plots)} files)")


if __name__ == "__main__":
    main()
