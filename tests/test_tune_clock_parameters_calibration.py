from __future__ import annotations

from scripts.tune_clock_parameters_calibration import (
    _select,
    disclosed_budget_per_good,
)


def test_budget_per_good_reference_is_scale_free():
    artefact = {
        "environment": {
            "goods": [{"good_id": "A"}, {"good_id": "B"}],
            "bidders": [
                {"budget_cap": 100.0},
                {"budget_cap": 200.0},
                {"budget_cap": 300.0},
            ],
        }
    }
    assert disclosed_budget_per_good(artefact) == 100.0


def test_selection_uses_payment_error_inside_efficiency_band():
    rows = [
        {
            "mean_efficiency": 0.970,
            "mean_payment_error_over_optimum_welfare": 0.2,
            "mean_value_queries": 10.0,
            "mean_rounds": 10.0,
            "price_step_fraction": 0.1,
            "top_k": 3,
            "tie_threshold_fraction": 0.4,
        },
        {
            "mean_efficiency": 0.968,
            "mean_payment_error_over_optimum_welfare": 0.1,
            "mean_value_queries": 12.0,
            "mean_rounds": 12.0,
            "price_step_fraction": 0.2,
            "top_k": 3,
            "tie_threshold_fraction": 0.4,
        },
    ]
    assert _select(rows, 0.005)["price_step_fraction"] == 0.2
