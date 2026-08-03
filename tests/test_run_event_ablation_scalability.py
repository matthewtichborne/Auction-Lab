from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_event_ablation_8x8 import (
    CLOCK_FRONTIER_TREATMENTS,
    CLOCK_FRONTIER_SINGLE_PASS_TREATMENTS,
    CLOCK_REVEALED_TOP_K_TREATMENTS,
    CLOCK_FINAL_TREATMENTS,
    CLOCK_FOCUSED_CLOSURE_TREATMENTS,
    CLOCK_LEAN_TREATMENTS,
    CLOCK_NATIVE_TREATMENTS,
    TREATMENTS,
    _effective_top_k,
)
from scripts.run_event_ablation_scalability import (
    aggregate_outcomes,
    aggregate_paired_deltas,
    build_ablation_cells,
    paired_deltas,
)


def test_complete_grid_has_95_datasets_and_760_treatment_cells(
    tmp_path: Path,
):
    cells = build_ablation_cells(
        sizes=[4, 5, 6, 7, 8, 9, 10],
        fixed_size=8,
        seeds=[0, 1, 2, 3, 4],
        treatments=TREATMENTS,
        elicitation_pack_dir=tmp_path / "packs",
        output_dir=tmp_path / "runs",
    )

    assert len(cells) == 95 * 8
    assert len({(cell.run.seed, cell.run.case_name) for cell in cells}) == 95
    assert any(cell.run_dir.parts[-3:] == (
        "seed_4", "joint_10x10", "all_targeted_events"
    ) for cell in cells)


def test_clock_lean_treatments_are_mechanism_specific_and_uncapped():
    treatments = {row.name: row.flags for row in CLOCK_LEAN_TREATMENTS}

    assert set(treatments) == {
        "pv_only",
        "allocation_only",
        "terminal_winner_only",
        "terminal_settlement",
        "lean_combined",
    }
    for flags in treatments.values():
        assert "--no-clock-event-demand-switch-verification" in flags
        assert "--no-clock-event-contested-bundle-refinement" in flags
        assert "--no-clock-event-terminal-best-losing-challenger" in flags
        assert not any("max-total-refinement" in flag for flag in flags)
        assert not any("max-refinements-per-bidder" in flag for flag in flags)

    assert "--event-incumbent-verification" in treatments["allocation_only"]
    assert "--event-incumbent-verification" in treatments["lean_combined"]
    assert "--clock-event-terminal-vcg-witness-verification" in treatments[
        "terminal_settlement"
    ]


def test_clock_native_treatments_form_complete_three_event_factorial():
    treatments = {row.name: row.flags for row in CLOCK_NATIVE_TREATMENTS}

    assert len(treatments) == 8
    combinations = {
        (
            "--clock-native-near-zero-surplus" in flags,
            "--clock-native-demand-changed" in flags,
            "--clock-native-near-tie" in flags,
        )
        for flags in treatments.values()
    }
    assert len(combinations) == 8
    for flags in treatments.values():
        assert (
            flags[flags.index("--clock-event-framework") + 1]
            == "native_v1"
        )
        assert (
            flags[flags.index("--clock-supplementary-support-policy") + 1]
            == "all_atoms"
        )
        assert "--no-event-incumbent-verification" in flags
        assert "--no-clock-event-terminal-stability-audit" in flags
        assert not any("max-total-refinement" in flag for flag in flags)


def test_clock_frontier_treatments_are_staged_and_uncapped():
    treatments = {
        row.name: row.flags for row in CLOCK_FRONTIER_TREATMENTS
    }
    assert set(treatments) == {
        "frontier_pv_only",
        "frontier_winners",
        "frontier_winners_pivotal",
        "frontier_winners_pivotal_closure",
        "frontier_winners_pivotal_closure_vcg",
    }
    for flags in treatments.values():
        assert (
            flags[flags.index("--clock-event-framework") + 1]
            == "frontier_v1"
        )
        assert (
            flags[
                flags.index("--clock-supplementary-support-policy") + 1
            ]
            == "all_atoms"
        )
        assert not any("max-total-refinement" in flag for flag in flags)
        assert not any("max-refinements-per-bidder" in flag for flag in flags)
    assert "--clock-frontier-winner-verification" in treatments[
        "frontier_winners"
    ]
    assert "--clock-frontier-pivotal-challengers" in treatments[
        "frontier_winners_pivotal"
    ]
    assert "--clock-frontier-winner-closure" in treatments[
        "frontier_winners_pivotal_closure"
    ]
    assert "--clock-frontier-vcg-witness-verification" in treatments[
        "frontier_winners_pivotal_closure_vcg"
    ]


def test_clock_frontier_single_pass_crosses_source_and_winner_queries():
    treatments = {
        row.name: row.flags
        for row in CLOCK_FRONTIER_SINGLE_PASS_TREATMENTS
    }
    assert set(treatments) == {
        "single_pass_pv_only",
        "single_pass_all",
        "single_pass_all_winners",
        "single_pass_revealed",
        "single_pass_revealed_winners",
    }
    for name, flags in treatments.items():
        assert "--no-clock-frontier-winner-closure" in flags
        assert "--no-clock-frontier-vcg-witness-verification" in flags
        assert not any("max-total-refinement" in flag for flag in flags)
        if name != "single_pass_pv_only":
            assert "--clock-frontier-pivotal-challengers" in flags
            assert "--clock-frontier-vcg-single-pass" in flags
    assert "--clock-frontier-vcg-revealed-only" in treatments[
        "single_pass_revealed"
    ]
    assert "--clock-frontier-vcg-revealed-only" in treatments[
        "single_pass_revealed_winners"
    ]
    assert "--clock-frontier-winner-verification" in treatments[
        "single_pass_all_winners"
    ]


def test_clock_revealed_top_k_treatments_only_vary_demand_width():
    treatments = {
        row.name: row.flags for row in CLOCK_REVEALED_TOP_K_TREATMENTS
    }
    assert set(treatments) == {
        "revealed_topk_pv_only",
        "revealed_topk_3",
        "revealed_topk_5",
        "revealed_topk_8",
    }
    for name, flags in treatments.items():
        assert "--no-clock-frontier-winner-verification" in flags
        assert "--no-clock-frontier-pivotal-challengers" in flags
        assert "--no-clock-frontier-winner-closure" in flags
        assert "--no-clock-frontier-vcg-witness-verification" in flags
        if name == "revealed_topk_pv_only":
            assert "--no-clock-frontier-vcg-single-pass" in flags
            assert "--top-k" not in flags
        else:
            expected = name.rsplit("_", 1)[-1]
            assert "--clock-frontier-vcg-single-pass" in flags
            assert "--clock-frontier-vcg-revealed-only" in flags
            assert flags[flags.index("--top-k") + 1] == expected


def test_effective_top_k_uses_final_treatment_override():
    assert _effective_top_k(("--foo", "bar"), 3) == 3
    assert _effective_top_k(("--top-k", "5"), 3) == 5
    assert _effective_top_k(
        ("--top-k", "3", "--top-k", "8"), 1
    ) == 8


def test_final_clock_ablation_is_frozen_three_arm_comparison():
    treatments = {row.name: row.flags for row in CLOCK_FINAL_TREATMENTS}
    assert set(treatments) == {
        "final_pv_only",
        "final_revealed_witness_top3",
        "final_unrestricted_witness",
    }
    for flags in treatments.values():
        assert "--no-clock-frontier-winner-verification" in flags
        assert "--no-clock-frontier-pivotal-challengers" in flags
        assert "--no-clock-frontier-winner-closure" in flags
        assert "--no-clock-frontier-vcg-witness-verification" in flags
    revealed = treatments["final_revealed_witness_top3"]
    assert "--clock-frontier-vcg-single-pass" in revealed
    assert "--clock-frontier-vcg-revealed-only" in revealed
    assert revealed[revealed.index("--top-k") + 1] == "3"
    unrestricted = treatments["final_unrestricted_witness"]
    assert "--clock-frontier-vcg-single-pass" in unrestricted
    assert "--no-clock-frontier-vcg-revealed-only" in unrestricted


def test_focused_clock_closure_ablation_is_uncapped_and_staged():
    treatments = {
        row.name: row.flags for row in CLOCK_FOCUSED_CLOSURE_TREATMENTS
    }
    assert set(treatments) == {
        "focused_pv_only",
        "focused_revealed_witness_top3",
        "focused_winner_closure",
        "focused_winner_closure_revealed_vcg",
        "focused_revealed_winner_sandwich",
        "focused_full_frontier_closure",
    }
    for flags in treatments.values():
        assert not any("max-total-refinement" in flag for flag in flags)
        assert not any("max-refinements-per-bidder" in flag for flag in flags)
        assert flags[flags.index("--top-k") + 1] == "3"
    staged = treatments["focused_winner_closure_revealed_vcg"]
    assert "--clock-frontier-winner-closure" in staged
    assert "--clock-frontier-staged-revealed-vcg-closure" in staged
    assert "--no-clock-frontier-vcg-witness-verification" in staged
    sandwich = treatments["focused_revealed_winner_sandwich"]
    assert "--clock-frontier-vcg-single-pass" in sandwich
    assert "--clock-frontier-vcg-revealed-only" in sandwich
    assert "--clock-frontier-winner-closure" in sandwich
    assert "--clock-frontier-staged-revealed-vcg-closure" in sandwich
    full = treatments["focused_full_frontier_closure"]
    assert "--clock-frontier-winner-verification" in full
    assert "--clock-frontier-pivotal-challengers" in full
    assert "--clock-frontier-vcg-witness-verification" in full


def _row(
    *,
    seed: int,
    treatment: str,
    efficiency: float,
    queries: float,
) -> dict[str, object]:
    return {
        "seed": seed,
        "series": "goods",
        "case": "goods_4x8",
        "x_value": 4,
        "num_goods": 4,
        "num_bidders": 8,
        "treatment": treatment,
        "mechanism": "sealed",
        "efficiency": efficiency,
        "revenue": 50.0,
        "revenue_abs_error": 10.0,
        "revenue_abs_error_pct": 20.0,
        "surplus": 40.0,
        "value_queries": queries,
        "demand_queries": 0.0,
        "allocation_match": False,
    }


def test_paired_deltas_compare_each_dataset_with_its_control():
    rows = [
        _row(seed=0, treatment="control", efficiency=0.8, queries=10),
        _row(seed=0, treatment="scarcity_fallbacks", efficiency=0.9, queries=12),
        _row(seed=1, treatment="scarcity_fallbacks", efficiency=0.7, queries=9),
    ]

    deltas = paired_deltas(rows)

    assert len(deltas) == 1
    assert deltas[0]["efficiency_delta_pp"] == pytest.approx(10.0)
    assert deltas[0]["value_query_delta"] == pytest.approx(2.0)


def test_pooled_and_paired_summaries_report_dataset_counts_and_wins():
    rows = [
        _row(seed=0, treatment="control", efficiency=0.8, queries=10),
        _row(seed=0, treatment="scarcity_fallbacks", efficiency=0.9, queries=12),
        _row(seed=1, treatment="control", efficiency=0.85, queries=11),
        _row(seed=1, treatment="scarcity_fallbacks", efficiency=0.85, queries=11),
    ]

    outcomes = aggregate_outcomes(
        rows, group_fields=("treatment", "mechanism")
    )
    scarcity = next(
        row for row in outcomes if row["treatment"] == "scarcity_fallbacks"
    )
    assert scarcity["datasets"] == 2
    assert scarcity["seeds"] == 2
    assert scarcity["mean_efficiency"] == pytest.approx(0.875)

    summary = aggregate_paired_deltas(
        paired_deltas(rows), group_fields=("treatment", "mechanism")
    )[0]
    assert summary["paired_datasets"] == 2
    assert summary["efficiency_wins"] == 1
    assert summary["efficiency_ties"] == 1
    assert summary["efficiency_losses"] == 0
