from __future__ import annotations

from auctionlab.auctions.clock import ClockConfig
from auctionlab.experiments.runner import run_all_mechanisms_on_instance


def test_experiment_runner_compares_sealed_and_clock(
    toy_instance,
    expected_toy_allocation,
):
    result = run_all_mechanisms_on_instance(
        toy_instance,
        instance_name="toy",
        clock_cfg=ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
    )

    assert result.instance_name == "toy"
    assert len(result.results) == 2

    by_name = {r.mechanism: r for r in result.results}

    sealed = by_name["sealed_xor_vcg"]
    clock = by_name["clock_supplementary_vcg_top_1"]

    assert sealed.allocation == expected_toy_allocation
    assert sealed.welfare == 23.0
    assert sealed.payments == {"i1": 0.0, "i2": 1.0, "i3": 13.0}
    assert sealed.revenue == 14.0
    assert sealed.rounds is None
    assert sealed.query_count == 3

    assert clock.allocation == expected_toy_allocation
    assert clock.welfare == 23.0
    assert clock.payments == {"i1": 0.0, "i2": 1.0, "i3": 13.0}
    assert clock.revenue == 14.0
    assert clock.rounds == 11
    assert clock.query_count == 33


def test_experiment_runner_runs_each_clock_top_k(toy_instance):
    result = run_all_mechanisms_on_instance(
        toy_instance,
        clock_cfg=ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0),
        clock_top_k_values=[1, 2],
    )

    assert [mechanism_result.mechanism for mechanism_result in result.results] == [
        "sealed_xor_vcg",
        "clock_supplementary_vcg_top_1",
        "clock_supplementary_vcg_top_2",
    ]
    assert result.results[1].metadata["top_k"] == 1
    assert result.results[2].metadata["top_k"] == 2
