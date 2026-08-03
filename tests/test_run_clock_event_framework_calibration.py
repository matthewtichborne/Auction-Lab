from __future__ import annotations

from scripts.run_clock_event_framework_calibration import TREATMENTS, _aggregate


def test_clock_framework_has_compact_prespecified_four_treatments():
    assert list(TREATMENTS) == [
        "core",
        "core_contested",
        "core_terminal_vcg",
        "clock_targeted_v1",
    ]
    assert TREATMENTS["clock_targeted_v1"] == {
        "contested_bundle_refinement": True,
        "terminal_vcg_witness_verification": True,
        "terminal_best_losing_challenger": True,
    }


def test_clock_framework_aggregate_preserves_ranges():
    rows = []
    for treatment in TREATMENTS:
        for efficiency in (0.9, 1.0):
            rows.append({
                "treatment": treatment,
                "efficiency": efficiency,
                "payment_error_over_optimum_welfare": 0.2,
                "revenue_loss": 0.1,
                "value_queries": 10.0,
                "rounds": 5.0,
                "allocation_match": efficiency == 1.0,
            })
    summary = _aggregate(rows)
    assert len(summary) == 4
    assert all(row["mean_efficiency"] == 0.95 for row in summary)
    assert all(row["min_efficiency"] == 0.9 for row in summary)
    assert all(row["max_efficiency"] == 1.0 for row in summary)
    assert all(row["allocation_match_rate"] == 0.5 for row in summary)
