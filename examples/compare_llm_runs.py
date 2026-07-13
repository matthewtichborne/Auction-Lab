"""Combine analyzed LLM run directories into one comparison CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from auctionlab.experiments.llm_analysis import (
    COMPARISON_FIELDS,
    compare_llm_runs,
)
from auctionlab.experiments.llm_analysis_io import (
    parse_labeled_path,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare analyzed LLM experiment run directories."
    )
    parser.add_argument("--run", nargs="+", required=True)
    parser.add_argument(
        "--output",
        default="outputs/llm_runs/llm_run_comparison.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        runs = [parse_labeled_path(raw) for raw in args.run]
        rows = compare_llm_runs(runs)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    output_path = Path(args.output)
    write_csv_rows(output_path, COMPARISON_FIELDS, rows)
    for label, path in runs:
        print(f"Loaded run {label}: {path}")
    print(f"Comparison rows: {len(rows)}")
    print(f"Wrote comparison: {output_path}")


if __name__ == "__main__":
    main()
