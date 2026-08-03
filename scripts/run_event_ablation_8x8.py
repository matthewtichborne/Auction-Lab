#!/usr/bin/env python3
"""Run a deterministic 8x8 elicitation-event ablation over frozen packs.

No model endpoint is contacted: every treatment replays the same frozen
opening answer, interest map and provisional valuations for a seed, while
deterministic value queries use the environment lookup table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Treatment:
    name: str
    flags: tuple[str, ...]


TREATMENTS = (
    Treatment("control", ()),
    Treatment(
        "without_incumbent_verification",
        ("--no-event-incumbent-verification",),
    ),
    Treatment("pivotal_challengers", ("--event-pivotal-challengers",)),
    Treatment("scarcity_fallbacks", ("--event-scarcity-fallbacks",)),
    Treatment(
        "large_correction_followup",
        ("--event-large-correction-followup",),
    ),
    Treatment(
        "gated_near_zero_surplus",
        ("--event-gate-near-zero-surplus",),
    ),
    Treatment(
        "terminal_regret_audit",
        ("--event-terminal-regret-audit",),
    ),
    Treatment(
        "all_targeted_events",
        (
            "--event-pivotal-challengers",
            "--event-scarcity-fallbacks",
            "--event-large-correction-followup",
            "--event-gate-near-zero-surplus",
            "--event-terminal-regret-audit",
        ),
    ),
)


_COUNTERFACTUAL_FLAGS = (
    "--sealed-feedback-rule", "competitive",
    "--sealed-loser-challenger-policy", "off",
    "--clock-top-k-frontier-policy", "allocation_pivotal",
    "--clock-allocation-counterfactual-frontier",
)

PRIMARY_TREATMENTS = (
    Treatment(
        "incumbent_only",
        (
            "--event-policy", "custom",
            "--sealed-feedback-rule", "allocated_bundle",
            "--sealed-loser-challenger-policy", "off",
            "--clock-top-k-frontier-policy", "off",
            "--no-clock-allocation-counterfactual-frontier",
            "--event-incumbent-verification",
            "--no-event-scarcity-fallbacks",
            "--no-sealed-event-large-correction-followup",
            "--no-clock-event-large-correction-followup",
        ),
    ),
    Treatment(
        "plus_counterfactuals",
        (
            "--event-policy", "custom",
            *_COUNTERFACTUAL_FLAGS,
            "--event-incumbent-verification",
            "--no-event-scarcity-fallbacks",
            "--no-sealed-event-large-correction-followup",
            "--no-clock-event-large-correction-followup",
        ),
    ),
    Treatment(
        "plus_scarcity",
        (
            "--event-policy", "custom",
            *_COUNTERFACTUAL_FLAGS,
            "--event-incumbent-verification",
            "--event-scarcity-fallbacks",
            "--no-sealed-event-large-correction-followup",
            "--no-clock-event-large-correction-followup",
        ),
    ),
    Treatment("recommended", ("--event-policy", "recommended")),
    Treatment(
        "recommended_without_incumbent",
        (
            "--event-policy", "custom",
            *_COUNTERFACTUAL_FLAGS,
            "--no-event-incumbent-verification",
            "--event-scarcity-fallbacks",
            "--sealed-event-large-correction-followup",
            "--no-clock-event-large-correction-followup",
        ),
    ),
    Treatment(
        "recommended_without_counterfactuals",
        (
            "--event-policy", "custom",
            "--sealed-feedback-rule", "allocated_bundle",
            "--sealed-loser-challenger-policy", "off",
            "--clock-top-k-frontier-policy", "off",
            "--no-clock-allocation-counterfactual-frontier",
            "--event-incumbent-verification",
            "--event-scarcity-fallbacks",
            "--sealed-event-large-correction-followup",
            "--no-clock-event-large-correction-followup",
        ),
    ),
    Treatment(
        "recommended_without_scarcity",
        (
            "--event-policy", "custom",
            *_COUNTERFACTUAL_FLAGS,
            "--event-incumbent-verification",
            "--no-event-scarcity-fallbacks",
            "--sealed-event-large-correction-followup",
            "--no-clock-event-large-correction-followup",
        ),
    ),
    Treatment(
        "recommended_without_large_correction",
        (
            "--event-policy", "custom",
            *_COUNTERFACTUAL_FLAGS,
            "--event-incumbent-verification",
            "--event-scarcity-fallbacks",
            "--no-sealed-event-large-correction-followup",
            "--no-clock-event-large-correction-followup",
        ),
    ),
)


_CLOCK_TARGETED_CORE_FLAGS = (
    "--event-policy", "custom",
    "--clock-event-framework", "targeted_v1",
    "--clock-event-demand-switch-verification",
    "--clock-event-terminal-winner-verification",
    "--event-incumbent-verification",
    "--no-clock-event-terminal-stability-audit",
    "--no-clock-event-terminal-best-losing-challenger",
    "--no-clock-event-large-correction-followup",
    "--no-event-scarcity-fallbacks",
    "--no-event-pivotal-challengers",
    "--no-event-gate-near-zero-surplus",
    "--no-event-terminal-regret-audit",
)

CLOCK_TARGETED_TREATMENTS = (
    Treatment(
        "clock_core",
        (*_CLOCK_TARGETED_CORE_FLAGS,
         "--no-clock-event-contested-bundle-refinement",
         "--no-clock-event-terminal-vcg-witness-verification"),
    ),
    Treatment(
        "clock_core_contested",
        (*_CLOCK_TARGETED_CORE_FLAGS,
         "--clock-event-contested-bundle-refinement",
         "--no-clock-event-terminal-vcg-witness-verification"),
    ),
    Treatment(
        "clock_core_terminal_vcg",
        (*_CLOCK_TARGETED_CORE_FLAGS,
         "--no-clock-event-contested-bundle-refinement",
         "--clock-event-terminal-vcg-witness-verification",
         "--clock-event-terminal-best-losing-challenger"),
    ),
    Treatment(
        "clock_targeted_v1",
        (*_CLOCK_TARGETED_CORE_FLAGS,
         "--clock-event-contested-bundle-refinement",
         "--clock-event-terminal-vcg-witness-verification",
         "--clock-event-terminal-best-losing-challenger"),
    ),
)


# Query-efficient, mechanism-specific clock ablation.  Every treatment uses
# the same provisional-value clock and differs only in exact value-query
# events.  There is deliberately no query budget: query volume is induced by
# the enabled events, while the clock runner deduplicates bidder/bundle pairs.
_CLOCK_LEAN_COMMON_FLAGS = (
    "--event-policy", "custom",
    "--clock-event-framework", "targeted_v1",
    "--clock-supplementary-support-policy", "demand_revealed",
    "--no-clock-event-demand-switch-verification",
    "--no-clock-event-contested-bundle-refinement",
    "--no-clock-event-terminal-best-losing-challenger",
    "--no-clock-event-terminal-stability-audit",
    "--no-clock-event-large-correction-followup",
    "--no-event-scarcity-fallbacks",
    "--no-event-pivotal-challengers",
    "--no-event-gate-near-zero-surplus",
    "--no-event-terminal-regret-audit",
)

CLOCK_LEAN_TREATMENTS = (
    Treatment(
        "pv_only",
        (*_CLOCK_LEAN_COMMON_FLAGS,
         "--no-event-incumbent-verification",
         "--no-clock-event-terminal-winner-verification",
         "--no-clock-event-terminal-vcg-witness-verification"),
    ),
    Treatment(
        "allocation_only",
        (*_CLOCK_LEAN_COMMON_FLAGS,
         "--event-incumbent-verification",
         "--no-clock-event-terminal-winner-verification",
         "--no-clock-event-terminal-vcg-witness-verification"),
    ),
    Treatment(
        "terminal_winner_only",
        (*_CLOCK_LEAN_COMMON_FLAGS,
         "--no-event-incumbent-verification",
         "--clock-event-terminal-winner-verification",
         "--no-clock-event-terminal-vcg-witness-verification"),
    ),
    Treatment(
        "terminal_settlement",
        (*_CLOCK_LEAN_COMMON_FLAGS,
         "--no-event-incumbent-verification",
         "--clock-event-terminal-winner-verification",
         "--clock-event-terminal-vcg-witness-verification"),
    ),
    Treatment(
        "lean_combined",
        (*_CLOCK_LEAN_COMMON_FLAGS,
         "--event-incumbent-verification",
         "--clock-event-terminal-winner-verification",
         "--clock-event-terminal-vcg-witness-verification"),
    ),
)


# Restored pre-shared-policy clock framework.  These three events depend only
# on the ascending price path: approaching dropout, abandoning primary demand,
# and a close runner-up.  The 2^3 grid estimates their individual and
# interaction effects without any sealed/WDP-triggered elicitation events.
_CLOCK_NATIVE_COMMON_FLAGS = (
    "--event-policy", "custom",
    "--clock-event-framework", "native_v1",
    "--clock-supplementary-support-policy", "all_atoms",
    "--no-event-incumbent-verification",
    "--clock-top-k-frontier-policy", "off",
    "--no-clock-allocation-counterfactual-frontier",
    "--no-clock-event-demand-switch-verification",
    "--no-clock-event-contested-bundle-refinement",
    "--no-clock-event-terminal-winner-verification",
    "--no-clock-event-terminal-vcg-witness-verification",
    "--no-clock-event-terminal-best-losing-challenger",
    "--no-clock-event-terminal-stability-audit",
    "--no-clock-event-large-correction-followup",
    "--no-event-scarcity-fallbacks",
    "--no-event-pivotal-challengers",
    "--no-event-gate-near-zero-surplus",
    "--no-event-terminal-regret-audit",
)


def _native_treatment(
    name: str,
    *,
    near_zero: bool,
    demand_changed: bool,
    near_tie: bool,
) -> Treatment:
    def flag(enabled: bool, name: str) -> str:
        return f"--{'no-' if not enabled else ''}{name}"

    return Treatment(
        name,
        (
            *_CLOCK_NATIVE_COMMON_FLAGS,
            flag(near_zero, "clock-native-near-zero-surplus"),
            flag(demand_changed, "clock-native-demand-changed"),
            flag(near_tie, "clock-native-near-tie"),
        ),
    )


CLOCK_NATIVE_TREATMENTS = (
    _native_treatment(
        "native_pv_only",
        near_zero=False,
        demand_changed=False,
        near_tie=False,
    ),
    _native_treatment(
        "native_near_zero",
        near_zero=True,
        demand_changed=False,
        near_tie=False,
    ),
    _native_treatment(
        "native_demand_changed",
        near_zero=False,
        demand_changed=True,
        near_tie=False,
    ),
    _native_treatment(
        "native_near_tie",
        near_zero=False,
        demand_changed=False,
        near_tie=True,
    ),
    _native_treatment(
        "native_near_zero_changed",
        near_zero=True,
        demand_changed=True,
        near_tie=False,
    ),
    _native_treatment(
        "native_near_zero_tie",
        near_zero=True,
        demand_changed=False,
        near_tie=True,
    ),
    _native_treatment(
        "native_changed_tie",
        near_zero=False,
        demand_changed=True,
        near_tie=True,
    ),
    _native_treatment(
        "native_full",
        near_zero=True,
        demand_changed=True,
        near_tie=True,
    ),
)


# Two-stage clock-discovered verification.  No exact value query interrupts
# price discovery.  The complete provisional candidate language remains
# available to the supplementary WDP, while only top-k bundles revealed along
# the clock path can become pivotal-challenger queries. Query counts are
# induced by the terminal frontier, never by a numerical query budget.
_CLOCK_FRONTIER_COMMON_FLAGS = (
    "--event-policy", "custom",
    "--clock-event-framework", "frontier_v1",
    "--clock-supplementary-support-policy", "all_atoms",
    "--clock-top-k-frontier-policy", "off",
    "--no-clock-allocation-counterfactual-frontier",
    "--no-event-incumbent-verification",
    "--no-clock-event-terminal-stability-audit",
    "--no-clock-event-terminal-winner-verification",
    "--no-clock-event-terminal-vcg-witness-verification",
    "--no-clock-event-terminal-best-losing-challenger",
    "--no-event-pivotal-challengers",
    "--no-event-scarcity-fallbacks",
    "--no-event-terminal-regret-audit",
)


def _frontier_treatment(
    name: str,
    *,
    winners: bool,
    challengers: bool,
    closure: bool,
    vcg: bool,
) -> Treatment:
    def flag(enabled: bool, name: str) -> str:
        return f"--{'no-' if not enabled else ''}{name}"

    return Treatment(name, (
        *_CLOCK_FRONTIER_COMMON_FLAGS,
        flag(winners, "clock-frontier-winner-verification"),
        flag(challengers, "clock-frontier-pivotal-challengers"),
        flag(closure, "clock-frontier-winner-closure"),
        flag(vcg, "clock-frontier-vcg-witness-verification"),
    ))


CLOCK_FRONTIER_TREATMENTS = (
    _frontier_treatment(
        "frontier_pv_only",
        winners=False, challengers=False, closure=False, vcg=False,
    ),
    _frontier_treatment(
        "frontier_winners",
        winners=True, challengers=False, closure=False, vcg=False,
    ),
    _frontier_treatment(
        "frontier_winners_pivotal",
        winners=True, challengers=True, closure=False, vcg=False,
    ),
    _frontier_treatment(
        "frontier_winners_pivotal_closure",
        winners=True, challengers=True, closure=True, vcg=False,
    ),
    _frontier_treatment(
        "frontier_winners_pivotal_closure_vcg",
        winners=True, challengers=True, closure=True, vcg=True,
    ),
)


def _single_pass_treatment(
    name: str,
    *,
    enabled: bool,
    winners: bool,
    revealed_only: bool,
) -> Treatment:
    def flag(value: bool, name: str) -> str:
        return f"--{'no-' if not value else ''}{name}"

    return Treatment(name, (
        *_CLOCK_FRONTIER_COMMON_FLAGS,
        flag(winners, "clock-frontier-winner-verification"),
        flag(enabled, "clock-frontier-pivotal-challengers"),
        "--no-clock-frontier-winner-closure",
        "--no-clock-frontier-vcg-witness-verification",
        flag(enabled, "clock-frontier-vcg-single-pass"),
        flag(revealed_only, "clock-frontier-vcg-revealed-only"),
    ))


CLOCK_FRONTIER_SINGLE_PASS_TREATMENTS = (
    _single_pass_treatment(
        "single_pass_pv_only",
        enabled=False, winners=False, revealed_only=False,
    ),
    _single_pass_treatment(
        "single_pass_all",
        enabled=True, winners=False, revealed_only=False,
    ),
    _single_pass_treatment(
        "single_pass_all_winners",
        enabled=True, winners=True, revealed_only=False,
    ),
    _single_pass_treatment(
        "single_pass_revealed",
        enabled=True, winners=False, revealed_only=True,
    ),
    _single_pass_treatment(
        "single_pass_revealed_winners",
        enabled=True, winners=True, revealed_only=True,
    ),
)


def _revealed_top_k_treatment(name: str, *, top_k: int | None) -> Treatment:
    enabled = top_k is not None

    def flag(value: bool, option: str) -> str:
        return f"--{'no-' if not value else ''}{option}"

    return Treatment(name, (
        *_CLOCK_FRONTIER_COMMON_FLAGS,
        "--no-clock-frontier-winner-verification",
        "--no-clock-frontier-pivotal-challengers",
        "--no-clock-frontier-winner-closure",
        "--no-clock-frontier-vcg-witness-verification",
        flag(enabled, "clock-frontier-vcg-single-pass"),
        flag(enabled, "clock-frontier-vcg-revealed-only"),
        *(("--top-k", str(top_k)) if top_k is not None else ()),
    ))


CLOCK_REVEALED_TOP_K_TREATMENTS = (
    _revealed_top_k_treatment("revealed_topk_pv_only", top_k=None),
    _revealed_top_k_treatment("revealed_topk_3", top_k=3),
    _revealed_top_k_treatment("revealed_topk_5", top_k=5),
    _revealed_top_k_treatment("revealed_topk_8", top_k=8),
)


# Confirmatory clock ablation frozen after exploratory policy development.
# The unrestricted arm is a diagnostic upper bound, not a candidate primary
# mechanism: it removes only the requirement that a witness was revealed on
# the top-3 clock demand path.
CLOCK_FINAL_TREATMENTS = (
    _single_pass_treatment(
        "final_pv_only",
        enabled=False, winners=False, revealed_only=False,
    ),
    Treatment(
        "final_revealed_witness_top3",
        (
            *_CLOCK_FRONTIER_COMMON_FLAGS,
            "--no-clock-frontier-winner-verification",
            "--no-clock-frontier-pivotal-challengers",
            "--no-clock-frontier-winner-closure",
            "--no-clock-frontier-vcg-witness-verification",
            "--clock-frontier-vcg-single-pass",
            "--clock-frontier-vcg-revealed-only",
            "--top-k", "3",
        ),
    ),
    Treatment(
        "final_unrestricted_witness",
        (
            *_CLOCK_FRONTIER_COMMON_FLAGS,
            "--no-clock-frontier-winner-verification",
            "--no-clock-frontier-pivotal-challengers",
            "--no-clock-frontier-winner-closure",
            "--no-clock-frontier-vcg-witness-verification",
            "--clock-frontier-vcg-single-pass",
            "--no-clock-frontier-vcg-revealed-only",
            "--top-k", "3",
        ),
    ),
)


# Focused follow-up after the sparse final clock policy proved primarily
# useful for pricing rather than allocation repair.  These treatments isolate
# allocation closure, revealed VCG closure, and the unrestricted full-frontier
# upper benchmark.  All use the same completed top-3 clock path and impose no
# numerical refinement-query budget.
def _focused_clock_treatment(
    name: str,
    *,
    winner_verification: bool = False,
    winner_closure: bool = False,
    single_pass_revealed: bool = False,
    staged_revealed_closure: bool = False,
    pivotal: bool = False,
    full_vcg_closure: bool = False,
) -> Treatment:
    def flag(value: bool, option: str) -> str:
        return f"--{'no-' if not value else ''}{option}"

    return Treatment(name, (
        *_CLOCK_FRONTIER_COMMON_FLAGS,
        flag(winner_verification, "clock-frontier-winner-verification"),
        flag(pivotal, "clock-frontier-pivotal-challengers"),
        flag(winner_closure, "clock-frontier-winner-closure"),
        flag(full_vcg_closure, "clock-frontier-vcg-witness-verification"),
        flag(single_pass_revealed, "clock-frontier-vcg-single-pass"),
        flag(single_pass_revealed, "clock-frontier-vcg-revealed-only"),
        flag(
            staged_revealed_closure,
            "clock-frontier-staged-revealed-vcg-closure",
        ),
        "--top-k", "3",
    ))


CLOCK_FOCUSED_CLOSURE_TREATMENTS = (
    _focused_clock_treatment("focused_pv_only"),
    _focused_clock_treatment(
        "focused_revealed_witness_top3",
        single_pass_revealed=True,
    ),
    _focused_clock_treatment(
        "focused_winner_closure",
        winner_closure=True,
    ),
    _focused_clock_treatment(
        "focused_winner_closure_revealed_vcg",
        winner_closure=True,
        staged_revealed_closure=True,
    ),
    _focused_clock_treatment(
        "focused_revealed_winner_sandwich",
        winner_closure=True,
        single_pass_revealed=True,
        staged_revealed_closure=True,
    ),
    _focused_clock_treatment(
        "focused_full_frontier_closure",
        winner_verification=True,
        winner_closure=True,
        pivotal=True,
        full_vcg_closure=True,
    ),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-spec",
        type=Path,
        default=Path(
            "scenarios/pc_build_v3/pc_build_population_16x16.json"
        ),
    )
    parser.add_argument(
        "--elicitation-pack-dir",
        type=Path,
        default=Path("outputs/elicitation_packs/scalability"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/event_ablation_8x8"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--treatment-set",
        choices=[
            "diagnostic",
            "primary",
            "clock-targeted",
            "clock-lean",
            "clock-native",
            "clock-frontier",
            "clock-frontier-single-pass",
            "clock-revealed-topk",
            "clock-final",
            "clock-focused-closure",
        ],
        default="diagnostic",
        help=(
            "'primary' runs the staged recommended-policy construction and "
            "leave-one-component-out checks used by the final pipeline."
        ),
    )
    parser.add_argument(
        "--treatments",
        nargs="+",
        choices=[
            treatment.name
            for treatment in (
                *TREATMENTS,
                *PRIMARY_TREATMENTS,
                *CLOCK_TARGETED_TREATMENTS,
                *CLOCK_LEAN_TREATMENTS,
                *CLOCK_NATIVE_TREATMENTS,
                *CLOCK_FRONTIER_TREATMENTS,
                *CLOCK_FRONTIER_SINGLE_PASS_TREATMENTS,
                *CLOCK_REVEALED_TOP_K_TREATMENTS,
                *CLOCK_FINAL_TREATMENTS,
                *CLOCK_FOCUSED_CLOSURE_TREATMENTS,
            )
        ],
        default=None,
    )
    parser.add_argument("--pv-calibration-config", type=Path, default=None)
    parser.add_argument("--sealed-rounds", type=int, default=20)
    parser.add_argument("--clock-rounds", type=int, default=50)
    parser.add_argument("--clock-top-k", type=int, default=3)
    parser.add_argument("--price-step", type=float, default=50.0)
    parser.add_argument("--pivotal-gap-threshold", type=float, default=100.0)
    parser.add_argument("--correction-threshold", type=float, default=0.25)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--verbose-runs",
        action="store_true",
        help=(
            "Stream each auction to the terminal. By default each treatment "
            "is captured in RUN_DIR/ablation_runner.log."
        ),
    )
    return parser.parse_args()


def _read_arm_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["arm"].startswith("proxy ")
        ]


def _mechanism_details(
    run_dir: Path,
    mechanism: str,
    clock_top_k: int,
) -> dict[str, str]:
    path = (
        run_dir / "curated_sealed_proxy_elicited.csv"
        if mechanism == "sealed"
        else run_dir
        / f"curated_clock_proxy_elicited_top_{clock_top_k}.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def _effective_top_k(flags: tuple[str, ...], default: int) -> int:
    """Return the final treatment-level --top-k override, if present."""
    value = default
    for index, flag in enumerate(flags[:-1]):
        if flag == "--top-k":
            value = int(flags[index + 1])
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["treatment"]), str(row["mechanism"])), []
        ).append(row)
    summary: list[dict[str, object]] = []
    for (treatment, mechanism), members in sorted(grouped.items()):
        def mean(key: str) -> float:
            values = [
                float(member[key])
                for member in members
                if not math.isnan(float(member[key]))
            ]
            return sum(values) / len(values) if values else math.nan

        summary.append({
            "treatment": treatment,
            "mechanism": mechanism,
            "seeds": len(members),
            "mean_efficiency": mean("efficiency"),
            "mean_revenue": mean("revenue"),
            "mean_revenue_abs_error": mean("revenue_abs_error"),
            "mean_revenue_loss": mean("revenue_loss"),
            "mean_payment_error_over_optimum_welfare": mean(
                "payment_error_over_optimum_welfare"
            ),
            "mean_surplus": mean("surplus"),
            "mean_value_queries": mean("value_queries"),
            "mean_demand_queries": mean("demand_queries"),
            "mean_supplementary_atoms": mean("supplementary_atoms"),
        })
    return summary


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_event_rows(
    path: Path,
    *,
    seed: int,
    treatment: str,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{
        "seed": seed,
        "treatment": treatment,
        "mechanism": (
            "sealed" if str(row["arm"]).startswith("proxy_sealed") else "clock"
        ),
        "event_type": row["event_type"],
        "bidder_id": row["bidder_id"],
        "bundle": row["bundle"],
        "abs_correction": abs(
            float(row["new_value"]) - float(row["old_value"])
        ),
        "allocation_hit": _bool(row["appears_in_final_allocation"]),
        "pricing_witness_hit": _bool(
            row["appears_in_any_reported_vcg_witness"]
        ),
        "oracle_witness_hit": (
            None
            if row["appears_in_any_full_info_vcg_witness"] == ""
            else _bool(row["appears_in_any_full_info_vcg_witness"])
        ),
    } for row in rows]


def _aggregate_events(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((
            str(row["treatment"]),
            str(row["mechanism"]),
            str(row["event_type"]),
        ), []).append(row)
    output: list[dict[str, object]] = []
    for (treatment, mechanism, event_type), members in sorted(
        grouped.items()
    ):
        count = len(members)
        oracle_members = [
            member for member in members
            if member["oracle_witness_hit"] is not None
        ]
        output.append({
            "treatment": treatment,
            "mechanism": mechanism,
            "event_type": event_type,
            "queries": count,
            "mean_abs_correction": sum(
                float(member["abs_correction"]) for member in members
            ) / count,
            "allocation_hit_rate": sum(
                bool(member["allocation_hit"]) for member in members
            ) / count,
            "pricing_witness_hit_rate": sum(
                bool(member["pricing_witness_hit"]) for member in members
            ) / count,
            "oracle_witness_hit_rate": (
                sum(
                    bool(member["oracle_witness_hit"])
                    for member in oracle_members
                ) / len(oracle_members)
                if oracle_members
                else math.nan
            ),
        })
    return output


def _plot(summary: list[dict[str, object]], output_dir: Path) -> None:
    if not summary:
        return
    os.environ.setdefault(
        "MPLCONFIGDIR", str(output_dir / ".matplotlib")
    )
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV outputs were still written.")
        return
    for metric, label in (
        ("mean_efficiency", "Mean allocative efficiency"),
        ("mean_value_queries", "Mean value queries"),
        ("mean_revenue", "Mean reported-bid VCG revenue"),
        ("mean_revenue_abs_error", "Mean absolute VCG revenue error"),
    ):
        fig, axis = plt.subplots(figsize=(11, 5.5))
        mechanisms = sorted({str(row["mechanism"]) for row in summary})
        treatment_catalog = (
            *TREATMENTS,
            *PRIMARY_TREATMENTS,
            *CLOCK_TARGETED_TREATMENTS,
            *CLOCK_LEAN_TREATMENTS,
            *CLOCK_NATIVE_TREATMENTS,
            *CLOCK_FRONTIER_TREATMENTS,
            *CLOCK_FRONTIER_SINGLE_PASS_TREATMENTS,
            *CLOCK_REVEALED_TOP_K_TREATMENTS,
            *CLOCK_FINAL_TREATMENTS,
            *CLOCK_FOCUSED_CLOSURE_TREATMENTS,
        )
        treatments = [
            treatment.name
            for treatment in treatment_catalog
            if any(
                row["treatment"] == treatment.name for row in summary
            )
        ]
        width = 0.36
        for index, mechanism in enumerate(mechanisms):
            values = [
                next(
                    (
                        float(row[metric])
                        for row in summary
                        if row["treatment"] == treatment
                        and row["mechanism"] == mechanism
                    ),
                    math.nan,
                )
                for treatment in treatments
            ]
            offsets = [
                position + (index - (len(mechanisms) - 1) / 2) * width
                for position in range(len(treatments))
            ]
            axis.bar(offsets, values, width=width, label=mechanism)
        axis.set_xticks(range(len(treatments)))
        axis.set_xticklabels(treatments, rotation=28, ha="right")
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        plots = output_dir / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots / f"{metric}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = _args()
    treatment_pool = {
        "diagnostic": TREATMENTS,
        "primary": PRIMARY_TREATMENTS,
        "clock-targeted": CLOCK_TARGETED_TREATMENTS,
        "clock-lean": CLOCK_LEAN_TREATMENTS,
        "clock-native": CLOCK_NATIVE_TREATMENTS,
        "clock-frontier": CLOCK_FRONTIER_TREATMENTS,
        "clock-frontier-single-pass": CLOCK_FRONTIER_SINGLE_PASS_TREATMENTS,
        "clock-revealed-topk": CLOCK_REVEALED_TOP_K_TREATMENTS,
        "clock-final": CLOCK_FINAL_TREATMENTS,
        "clock-focused-closure": CLOCK_FOCUSED_CLOSURE_TREATMENTS,
    }[args.treatment_set]
    requested_treatments = (
        [treatment.name for treatment in treatment_pool]
        if args.treatments is None
        else args.treatments
    )
    selected = {
        treatment.name: treatment
        for treatment in treatment_pool
        if treatment.name in requested_treatments
    }
    if args.treatment_set in {
        "clock-targeted", "clock-lean", "clock-native", "clock-frontier",
        "clock-frontier-single-pass",
        "clock-revealed-topk",
        "clock-final",
        "clock-focused-closure",
    }:
        args.sealed_rounds = 0
    unknown_for_set = sorted(set(requested_treatments) - set(selected))
    if unknown_for_set:
        raise SystemExit(
            f"Treatments {unknown_for_set} do not belong to "
            f"--treatment-set {args.treatment_set}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    total_cells = len(args.seeds) * len(selected)
    cell_index = 0

    for seed in args.seeds:
        pack = (
            args.elicitation_pack_dir
            / f"seed_{seed}"
            / "anchor_8x8"
            / "frozen_elicitation.json"
        )
        if not pack.exists():
            raise FileNotFoundError(f"Missing frozen 8x8 pack: {pack}")
        for treatment in selected.values():
            cell_index += 1
            run_dir = (
                args.output_dir / f"seed_{seed}" / treatment.name
            )
            summary_path = run_dir / "curated_run_summary.csv"
            command = [
                sys.executable,
                "examples/run_live_llm_curated_batch.py",
                "--scenario", "pc_build",
                "--scenario-spec", str(args.scenario_spec),
                "--num-goods", "8",
                "--num-bidders", "8",
                "--scenario-seed", str(seed),
                "--selection-policy", "coverage_stratified",
                "--seed-type", "structured",
                "--elicitation-pack", str(pack),
                "--ask-initial-question",
                "--use-interest-map",
                "--use-provisional-valuations",
                "--skip-baselines",
                "--sealed-elicitation-rounds", str(args.sealed_rounds),
                "--sealed-stopping-rule", "no_new_refinements",
                "--elicited-clock",
                "--top-k", str(args.clock_top_k),
                "--max-rounds", str(args.clock_rounds),
                "--price-step", str(args.price_step),
                "--event-pivotal-gap-threshold",
                str(args.pivotal_gap_threshold),
                "--event-correction-threshold",
                str(args.correction_threshold),
                "--llm-cache-mode", "off",
                "--log-dir", str(run_dir),
                *(
                    [
                        "--pv-calibration-config",
                        str(args.pv_calibration_config),
                    ]
                    if args.pv_calibration_config is not None
                    else []
                ),
                *(
                    [
                        "--sealed-feedback-rule", "competitive",
                        "--sealed-loser-challenger-policy", "shadow_price",
                        "--clock-top-k-frontier-policy", "allocation_pivotal",
                        "--clock-allocation-counterfactual-frontier",
                    ]
                    if args.treatment_set == "diagnostic"
                    else []
                ),
                *treatment.flags,
            ]
            status = (
                f"[{cell_index}/{total_cells}] seed={seed} "
                f"treatment={treatment.name}"
            )
            if args.dry_run:
                print(" ".join(command), flush=True)
                continue
            if not summary_path.exists() or args.rerun_complete:
                print(f"▶ {status}", flush=True)
                run_dir.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                if args.verbose_runs:
                    result = subprocess.run(command, check=False)
                    returncode = result.returncode
                else:
                    log_path = run_dir / "ablation_runner.log"
                    with log_path.open(
                        "w", encoding="utf-8"
                    ) as log_handle:
                        process = subprocess.Popen(
                            command,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                        )
                        while True:
                            try:
                                returncode = process.wait(timeout=15)
                                break
                            except subprocess.TimeoutExpired:
                                elapsed = int(time.monotonic() - started)
                                print(
                                    f"  … still running ({elapsed}s); "
                                    f"details: {log_path}",
                                    flush=True,
                                )
                elapsed = time.monotonic() - started
                if returncode:
                    print(
                        f"✗ {status} failed after {elapsed:.1f}s "
                        f"(exit {returncode})",
                        flush=True,
                    )
                    failure = {
                        "seed": seed,
                        "treatment": treatment.name,
                        "returncode": returncode,
                    }
                    failures.append(failure)
                    if args.fail_fast:
                        raise SystemExit(returncode)
                    continue
                print(f"✓ {status} complete in {elapsed:.1f}s", flush=True)
            else:
                print(f"↷ {status} already complete; skipping", flush=True)
            for row in _read_arm_rows(summary_path):
                mechanism = (
                    "sealed" if row["arm"] == "proxy sealed" else "clock"
                )
                treatment_top_k = _effective_top_k(
                    treatment.flags, args.clock_top_k
                )
                details = _mechanism_details(
                    run_dir, mechanism, treatment_top_k
                )
                full_info_revenue = _float(
                    details, "full_info_revenue"
                )
                revenue = _float(row, "revenue")
                run_rows.append({
                    "seed": seed,
                    "treatment": treatment.name,
                    "mechanism": mechanism,
                    "efficiency": _float(row, "efficiency"),
                    "true_welfare": _float(row, "true_welfare"),
                    "full_info_welfare": _float(row, "full_info_welfare"),
                    "revenue": revenue,
                    "full_info_revenue": full_info_revenue,
                    "revenue_abs_error": abs(
                        revenue - full_info_revenue
                    ),
                    "revenue_loss": _float(details, "revenue_loss"),
                    "payment_error_over_optimum_welfare": _float(
                        details, "payment_error_over_optimum_welfare"
                    ),
                    "allocation_match": _bool(
                        details["allocation_match"]
                    ),
                    "surplus": _float(row, "surplus"),
                    "value_queries": _float(row, "vq"),
                    "demand_queries": _float(row, "dq"),
                    "nl_queries": _float(row, "nl"),
                    "supplementary_support_policy": details.get(
                        "supplementary_support_policy", "all_atoms"
                    ),
                    "supplementary_atoms": _float(
                        details, "supplementary_atoms_total"
                    ),
                    "run_dir": str(run_dir),
                })
            event_rows.extend(_read_event_rows(
                run_dir / "curated_refinement_records.csv",
                seed=seed,
                treatment=treatment.name,
            ))

    if args.dry_run:
        return
    _write_csv(args.output_dir / "event_ablation_runs.csv", run_rows)
    summary = _aggregate(run_rows)
    _write_csv(args.output_dir / "event_ablation_summary.csv", summary)
    _write_csv(
        args.output_dir / "event_ablation_event_rows.csv", event_rows
    )
    _write_csv(
        args.output_dir / "event_ablation_event_summary.csv",
        _aggregate_events(event_rows),
    )
    _plot(summary, args.output_dir)
    (args.output_dir / "event_ablation_failures.json").write_text(
        json.dumps(failures, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(run_rows)} arm rows; {len(failures)} failed treatments."
    )


if __name__ == "__main__":
    main()
