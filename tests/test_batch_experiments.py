from __future__ import annotations

from auctionlab.auctions.clock import ClockConfig
from auctionlab.experiments.runner import (
    run_batch_experiments,
    summarize_batch_results,
)
from auctionlab.instances.random import make_random_xor_instance


def test_batch_experiment_runner_summarizes_random_instances():
    instances = []

    for seed in range(3):
        instance = make_random_xor_instance(
            n_items=4,
            n_bidders=3,
            atoms_per_bidder=4,
            max_bundle_size=2,
            min_value=1.0,
            max_value=10.0,
            seed=seed,
        )

        instances.append((f"seed_{seed}", instance))

    batch_results = run_batch_experiments(
        instances,
        clock_cfg=ClockConfig(max_rounds=30, price_step=1.0, reserve=0.0),
    )

    summary = summarize_batch_results(batch_results)

    assert summary.n_instances == 3

    assert summary.sealed_avg_welfare >= 0.0
    assert summary.clock_avg_welfare >= 0.0

    assert 0.0 <= summary.clock_avg_efficiency <= 1.0
    assert summary.sealed_avg_revenue >= 0.0
    assert summary.clock_avg_revenue >= 0.0

    assert summary.clock_avg_rounds > 0.0
    assert summary.sealed_avg_query_count == 3.0
    assert summary.clock_avg_query_count > 0.0

    assert 0.0 <= summary.allocation_match_rate <= 1.0
    assert 0.0 <= summary.welfare_match_rate <= 1.0