from scripts.analyze_final_experiment import select_examples


def _row(seed, case, initial, sealed, gain, revenue_gain, queries):
    return {
        "seed": str(seed),
        "case": case,
        "initial_efficiency": initial,
        "sealed_efficiency": sealed,
        "sealed_efficiency_gain_pct": gain,
        "sealed_revenue_loss_improvement": revenue_gain,
        "sealed_value_queries": queries,
    }


def test_example_selection_is_rule_based_and_deterministic():
    rows = [
        _row(0, "anchor_8x8", 0.8, 1.0, 20.0, 0.1, 12),
        _row(1, "anchor_8x8", 0.9, 1.0, 10.0, 0.4, 20),
        _row(2, "anchor_8x8", 0.9, 0.9, 0.0, 0.0, 5),
        _row(3, "anchor_8x8", 0.8, 1.0, 20.0, 0.2, 16),
    ]
    selected = {
        row["selection_rule"]: row for row in select_examples(rows)
    }
    assert selected["largest_sealed_efficiency_improvement"]["seed"] == "0"
    assert selected["largest_sealed_revenue_loss_reduction"]["seed"] == "1"
    assert selected[
        "median_query_case_reaching_full_efficiency_from_below"
    ]["seed"] == "3"
    assert selected["representative_non_improving_case"]["seed"] == "2"
