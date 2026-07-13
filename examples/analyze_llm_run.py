"""Generate all canonical analysis CSVs for one LLM experiment directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from auctionlab.experiments.llm_analysis import analyze_llm_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze curated outputs from one LLM experiment run."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-value-errors", action="store_true")
    parser.add_argument("--skip-allocation-losses", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or args.input_dir)
    counts = analyze_llm_run(
        args.input_dir,
        output_dir,
        skip_value_errors=args.skip_value_errors,
        skip_allocation_losses=args.skip_allocation_losses,
    )
    print(
        f"Loaded {counts['static_sealed_rows']} static sealed rows, "
        f"{counts['static_clock_rows']} static clock rows, "
        f"{counts['proxy_sealed_rows']} proxy sealed rows, "
        f"{counts['proxy_clock_rows']} proxy clock rows"
    )
    print(f"Wrote analysis outputs to: {output_dir}")


if __name__ == "__main__":
    main()
