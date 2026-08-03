from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_clock_parameter_robustness_8x8 import (
    _aggregate,
    _recommend,
    build_cells,
    build_command,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        scenario_spec=Path("scenario.json"),
        elicitation_pack_dir=tmp_path / "packs",
        output_dir=tmp_path / "runs",
        pv_calibration_config=Path("calibration.json"),
        seeds=[0, 1],
        price_steps=[25.0, 50.0],
        top_k_values=[1, 3],
        tie_thresholds=[100.0],
        max_rounds=50,
        event_policy="final-v1",
    )


def test_grid_is_full_cartesian_product(tmp_path):
    cells = build_cells(_args(tmp_path))
    assert len(cells) == 8
    assert {cell.seed for cell in cells} == {0, 1}
    assert {cell.price_step for cell in cells} == {25.0, 50.0}
    assert {cell.top_k for cell in cells} == {1, 3}


def test_command_uses_recommended_policy_and_frozen_pack(tmp_path):
    args = _args(tmp_path)
    cell = build_cells(args)[0]
    command = build_command(cell, args)
    joined = " ".join(command)
    assert "--event-policy final-v1" in joined
    assert "--person-query-mode deterministic" in joined
    assert "--llm-cache-mode off" in joined
    assert "--pv-calibration-config calibration.json" in joined
    assert str(cell.pack) in command


def test_recommendation_uses_payment_error_inside_efficiency_band():
    rows = [
        {
            "price_step": 25.0,
            "top_k": 3,
            "tie_threshold": 100.0,
            "mean_efficiency": 0.970,
            "mean_payment_error_over_optimum_welfare": 0.10,
            "mean_value_queries": 20.0,
            "mean_rounds": 30.0,
        },
        {
            "price_step": 50.0,
            "top_k": 3,
            "tie_threshold": 100.0,
            "mean_efficiency": 0.968,
            "mean_payment_error_over_optimum_welfare": 0.05,
            "mean_value_queries": 12.0,
            "mean_rounds": 20.0,
        },
        {
            "price_step": 100.0,
            "top_k": 1,
            "tie_threshold": 100.0,
            "mean_efficiency": 0.94,
            "mean_payment_error_over_optimum_welfare": 0.01,
            "mean_value_queries": 5.0,
            "mean_rounds": 10.0,
        },
    ]
    recommendation = _recommend(rows, efficiency_band=0.005)
    assert recommendation["selected"]["price_step"] == 50.0


def test_aggregate_keeps_seed_dispersion():
    base = {
        "price_step": 50.0,
        "top_k": 3,
        "tie_threshold": 100.0,
        "revenue_loss": 0.2,
        "revenue_absolute_percentage_error": 0.2,
        "payment_error_over_optimum_welfare": 0.1,
        "value_queries": 10.0,
        "demand_queries": 20.0,
        "rounds": 15.0,
        "allocation_match": "False",
    }
    rows = [
        {**base, "seed": 0, "efficiency": 0.9},
        {**base, "seed": 1, "efficiency": 1.0, "allocation_match": "True"},
    ]
    summary = _aggregate(rows)
    assert summary[0]["mean_efficiency"] == 0.95
    assert summary[0]["min_efficiency"] == 0.9
    assert summary[0]["max_efficiency"] == 1.0
    assert summary[0]["allocation_match_rate"] == 0.5
