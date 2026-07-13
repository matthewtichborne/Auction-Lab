"""Run and export a configurable batch of non-LLM random auction instances."""

from __future__ import annotations

import argparse
from pathlib import Path

from auctionlab.auctions.clock import ClockConfig
from auctionlab.experiments.export import (
    batch_results_to_comparison_rows,
    batch_results_to_mechanism_rows,
    batch_summaries_to_rows,
    write_csv,
)
from auctionlab.experiments.runner import (
    BatchSummary,
    run_batch_experiments,
    summarize_batch_results,
)
from auctionlab.instances.random import make_random_xor_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run random sealed and clock auction experiments."
    )
    parser.add_argument("--n-instances", type=int, default=20)
    parser.add_argument("--n-items", type=int, default=5)
    parser.add_argument("--n-bidders", type=int, default=4)
    parser.add_argument("--atoms-per-bidder", type=int, default=6)
    parser.add_argument("--max-bundle-size", type=int, default=3)
    parser.add_argument("--min-value", type=float, default=1.0)
    parser.add_argument("--max-value", type=float, default=20.0)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--price-step", type=float, default=1.0)
    parser.add_argument("--reserve", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed-offset", type=int, default=0)
    return parser.parse_args()


def print_summary(top_k: int, summary: BatchSummary) -> None:
    print(f"Batch summary for top_k={top_k}")
    print("-------------------------")
    print("Instances:", summary.n_instances)
    print("Sealed avg welfare:", summary.sealed_avg_welfare)
    print("Clock avg welfare:", summary.clock_avg_welfare)
    print("Clock avg efficiency:", summary.clock_avg_efficiency)
    print("Sealed avg revenue:", summary.sealed_avg_revenue)
    print("Clock avg revenue:", summary.clock_avg_revenue)
    print("Clock avg rounds:", summary.clock_avg_rounds)
    print("Sealed avg query count:", summary.sealed_avg_query_count)
    print("Clock avg query count:", summary.clock_avg_query_count)
    print("Allocation match rate:", summary.allocation_match_rate)
    print("Welfare match rate:", summary.welfare_match_rate)
    print()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    instances = []
    instances_by_name = {}

    seeds = range(args.seed_offset, args.seed_offset + args.n_instances)

    for seed in seeds:
        instance_name = f"random_seed_{seed}"
        instance = make_random_xor_instance(
            n_items=args.n_items,
            n_bidders=args.n_bidders,
            atoms_per_bidder=args.atoms_per_bidder,
            max_bundle_size=args.max_bundle_size,
            min_value=args.min_value,
            max_value=args.max_value,
            seed=seed,
        )

        instances.append((instance_name, instance))
        instances_by_name[instance_name] = instance

    batch_results = run_batch_experiments(
        instances,
        clock_cfg=ClockConfig(
            max_rounds=args.max_rounds,
            price_step=args.price_step,
            reserve=args.reserve,
        ),
        clock_top_k_values=args.top_k,
    )

    summaries_by_top_k: dict[int, BatchSummary] = {}

    for top_k in args.top_k:
        clock_mechanism = f"clock_supplementary_vcg_top_{top_k}"
        summary = summarize_batch_results(
            batch_results,
            clock_mechanism=clock_mechanism,
        )
        summaries_by_top_k[top_k] = summary
        print_summary(top_k, summary)

    mechanism_path = output_dir / "random_mechanism_results.csv"
    mechanism_rows = batch_results_to_mechanism_rows(batch_results)
    write_csv(mechanism_rows, mechanism_path)
    print(f"Wrote {mechanism_path}")

    for top_k in args.top_k:
        clock_mechanism = f"clock_supplementary_vcg_top_{top_k}"
        comparison_rows = batch_results_to_comparison_rows(
            batch_results,
            clock_mechanism=clock_mechanism,
            instances_by_name=instances_by_name,
        )
        output_path = output_dir / f"random_comparison_results_top_{top_k}.csv"
        write_csv(comparison_rows, output_path)
        print(f"Wrote {output_path}")

    summary_path = output_dir / "random_summary_results.csv"
    write_csv(batch_summaries_to_rows(summaries_by_top_k), summary_path)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
