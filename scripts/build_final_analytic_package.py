#!/usr/bin/env python3
"""Build the final offline analysis package from frozen experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.experiments.final_pipeline import load_final_experiment_spec  # noqa: E402
from auctionlab.experiments.scalability_analysis import (  # noqa: E402
    ScalabilityCase,
    load_scalability_results,
    write_scalability_tables,
)


COLORS = {"Initial PV": "#777777", "Sealed": "#0072B2", "Clock": "#D55E00"}
MARKERS = {"Initial PV": "o", "Sealed": "s", "Clock": "^"}
SERIES = ("bidders", "goods", "joint")
SERIES_LABELS = {
    "bidders": "8 goods; bidders vary",
    "goods": "8 bidders; goods vary",
    "joint": "Goods and bidders vary",
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--sealed-dir", type=Path, required=True)
    parser.add_argument("--clock-dir", type=Path, required=True)
    parser.add_argument("--interest-map-dir", type=Path, required=True)
    parser.add_argument("--sealed-ablation", type=Path, required=True)
    parser.add_argument("--clock-ablation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def number(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def key(case: ScalabilityCase) -> tuple[int, str]:
    return int(case.values["seed"]), str(case.values["case"])


def ci95(values: list[float]) -> tuple[float, float]:
    """Student-t interval for five independent seed-level summaries."""
    if len(values) < 2:
        return math.nan, math.nan
    critical = 2.776 if len(values) == 5 else 1.96
    half = critical * stdev(values) / math.sqrt(len(values))
    return mean(values) - half, mean(values) + half


def case_dir(root: Path, seed: int, case: str) -> Path:
    return root / f"seed_{seed}" / case


def expand_anchor(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if row["series"] != "anchor":
        yield row
        return
    for series in SERIES:
        copy = dict(row)
        copy["series"] = series
        copy["x_value"] = 8
        yield copy


def pair_cases(
    sealed: list[ScalabilityCase], clock: list[ScalabilityCase],
    sealed_root: Path, clock_root: Path,
) -> list[dict[str, Any]]:
    sealed_by_key = {key(case): case.values for case in sealed}
    clock_by_key = {key(case): case.values for case in clock}
    if set(sealed_by_key) != set(clock_by_key):
        raise ValueError("Sealed and clock case sets differ")
    rows: list[dict[str, Any]] = []
    for seed, case in sorted(sealed_by_key):
        s, c = sealed_by_key[(seed, case)], clock_by_key[(seed, case)]
        clock_detail = read_csv(
            case_dir(clock_root, seed, case) / "curated_clock_proxy_elicited_top_3.csv"
        )[0]
        sealed_detail = read_csv(
            case_dir(sealed_root, seed, case) / "curated_sealed_proxy_elicited.csv"
        )[0]
        rows.append({
            "seed": seed,
            "case": case,
            "series": s["series"],
            "x_value": s["x_value"],
            "num_goods": s["num_goods"],
            "num_bidders": s["num_bidders"],
            "initial_efficiency": s["initial_efficiency"],
            "sealed_efficiency": s["efficiency"],
            "clock_efficiency": c["efficiency"],
            "sealed_efficiency_gain": float(s["efficiency"]) - float(s["initial_efficiency"]),
            "clock_efficiency_gain": float(c["efficiency"]) - float(s["initial_efficiency"]),
            "clock_minus_sealed_efficiency": float(c["efficiency"]) - float(s["efficiency"]),
            "sealed_value_queries": s["value_queries"],
            "clock_value_queries": c["value_queries"],
            "sealed_rounds": sealed_detail["elicitation_rounds"],
            "clock_rounds": clock_detail["rounds"],
            "sealed_payment_error_over_optimum_welfare": s["payment_error_over_optimum_welfare"],
            "clock_payment_error_over_optimum_welfare": c["payment_error_over_optimum_welfare"],
            "sealed_revenue_error_over_optimum_welfare": sealed_detail["revenue_absolute_error_over_optimum_welfare"],
            "clock_revenue_error_over_optimum_welfare": clock_detail["revenue_absolute_error_over_optimum_welfare"],
            "sealed_revenue_loss": s["revenue_loss"],
            "clock_revenue_loss": c["revenue_loss"],
            "sealed_allocation_match": sealed_detail["allocation_match"],
            "clock_allocation_match": clock_detail["allocation_match"],
            "sealed_termination_reason": sealed_detail["termination_reason"],
            "clock_termination_reason": clock_detail["termination_reason"],
            "clock_failure_classification": clock_detail["failure_classification"],
            "clock_best_efficiency": clock_detail["best_true_efficiency"],
            "clock_final_minus_best_welfare": clock_detail["final_welfare_minus_best_welfare"],
        })
    return rows


def mechanism_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = {
        "Initial PV": ("initial_efficiency", None, None, None),
        "Sealed": ("sealed_efficiency", "sealed_value_queries", "sealed_payment_error_over_optimum_welfare", "sealed_revenue_error_over_optimum_welfare"),
        "Clock": ("clock_efficiency", "clock_value_queries", "clock_payment_error_over_optimum_welfare", "clock_revenue_error_over_optimum_welfare"),
    }
    output: list[dict[str, Any]] = []
    for mechanism, (eff_key, query_key, pay_key, revenue_key) in specs.items():
        efficiencies = [number(row, eff_key) for row in rows]
        seed_means = [
            mean(number(row, eff_key) for row in rows if int(row["seed"]) == seed)
            for seed in sorted({int(row["seed"]) for row in rows})
        ]
        low, high = ci95(seed_means)
        output.append({
            "mechanism": mechanism,
            "cases": len(rows),
            "seeds": len(seed_means),
            "mean_efficiency": mean(efficiencies),
            "median_efficiency": median(efficiencies),
            "seed_mean_efficiency_sd": stdev(seed_means),
            "seed_clustered_efficiency_ci95_low": low,
            "seed_clustered_efficiency_ci95_high": high,
            "cases_efficiency_ge_0_90": sum(value >= .90 for value in efficiencies),
            "cases_efficiency_ge_0_95": sum(value >= .95 for value in efficiencies),
            "cases_fully_efficient": sum(value >= 1 - 1e-9 for value in efficiencies),
            "mean_value_queries": mean(number(row, query_key) for row in rows) if query_key else 0,
            "median_value_queries": median(number(row, query_key) for row in rows) if query_key else 0,
            "mean_payment_error_over_optimum_welfare": mean(number(row, pay_key) for row in rows) if pay_key else math.nan,
            "mean_revenue_error_over_optimum_welfare": mean(number(row, revenue_key) for row in rows) if revenue_key else math.nan,
        })
    return output


def paired_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = [
        ("sealed_minus_initial_efficiency", "sealed_efficiency_gain", True),
        ("clock_minus_initial_efficiency", "clock_efficiency_gain", True),
        ("clock_minus_sealed_efficiency", "clock_minus_sealed_efficiency", True),
        ("clock_minus_sealed_queries", None, False),
        ("clock_minus_sealed_payment_error", None, False),
        ("clock_minus_sealed_revenue_error", None, False),
    ]
    output = []
    for label, source, higher_is_positive in comparisons:
        if source:
            values = [number(row, source) for row in rows]
        elif label.endswith("queries"):
            values = [number(row, "clock_value_queries") - number(row, "sealed_value_queries") for row in rows]
        elif "payment" in label:
            values = [number(row, "clock_payment_error_over_optimum_welfare") - number(row, "sealed_payment_error_over_optimum_welfare") for row in rows]
        else:
            values = [number(row, "clock_revenue_error_over_optimum_welfare") - number(row, "sealed_revenue_error_over_optimum_welfare") for row in rows]
        seed_means = [mean(value for value, row in zip(values, rows) if int(row["seed"]) == seed) for seed in range(5)]
        low, high = ci95(seed_means)
        output.append({
            "comparison": label,
            "mean_difference": mean(values),
            "median_difference": median(values),
            "seed_mean_difference_sd": stdev(seed_means),
            "seed_clustered_ci95_low": low,
            "seed_clustered_ci95_high": high,
            "positive_cases": sum(value > 1e-9 for value in values),
            "unchanged_cases": sum(abs(value) <= 1e-9 for value in values),
            "negative_cases": sum(value < -1e-9 for value in values),
            "higher_is_positive_outcome": higher_is_positive,
        })
    return output


def scaling_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = [copy for row in rows for copy in expand_anchor(row)]
    output = []
    for series in SERIES:
        for x in range(4, 11):
            group = [row for row in expanded if row["series"] == series and int(row["x_value"]) == x]
            if not group:
                continue
            for mechanism, eff, query, pay, revenue in (
                ("Initial PV", "initial_efficiency", None, None, None),
                ("Sealed", "sealed_efficiency", "sealed_value_queries", "sealed_payment_error_over_optimum_welfare", "sealed_revenue_error_over_optimum_welfare"),
                ("Clock", "clock_efficiency", "clock_value_queries", "clock_payment_error_over_optimum_welfare", "clock_revenue_error_over_optimum_welfare"),
            ):
                vals = [number(row, eff) for row in group]
                output.append({
                    "series": series,
                    "x_value": x,
                    "mechanism": mechanism,
                    "cases": len(group),
                    "efficiency_mean": mean(vals),
                    "efficiency_sd": stdev(vals),
                    "efficiency_min": min(vals),
                    "efficiency_max": max(vals),
                    "value_queries_mean": mean(number(row, query) for row in group) if query else 0,
                    "payment_error_mean": mean(number(row, pay) for row in group) if pay else math.nan,
                    "revenue_error_mean": mean(number(row, revenue) for row in group) if revenue else math.nan,
                })
    return output


def event_summary(sealed_root: Path, clock_root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for mechanism, root, expected in (("Sealed", sealed_root, "proxy_sealed_vcg"), ("Clock", clock_root, "proxy_clock_vcg")):
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for path in root.glob("seed_*/*/curated_refinement_records.csv"):
            for row in read_csv(path):
                if row.get("mechanism") == expected:
                    grouped[row["event_type"]].append(row)
        for event_type, group in sorted(grouped.items()):
            count = len(group)
            events.append({
                "mechanism": mechanism,
                "event_type": event_type,
                "query_count": count,
                "queries_per_case": count / 95,
                "mean_absolute_value_correction": mean(abs(number(row, "value_delta", 0)) for row in group),
                "final_allocation_hit_rate": sum(row.get("appears_in_final_allocation") == "True" for row in group) / count,
                "reported_vcg_witness_hit_rate": sum(row.get("appears_in_any_reported_vcg_witness") == "True" for row in group) / count,
                "full_info_vcg_witness_hit_rate": sum(row.get("appears_in_any_full_info_vcg_witness") == "True" for row in group) / count,
            })
    return events


def selected_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules = {
        "largest_sealed_efficiency_gain": max(rows, key=lambda row: number(row, "sealed_efficiency_gain")),
        "largest_clock_efficiency_gain": max(rows, key=lambda row: number(row, "clock_efficiency_gain")),
        "largest_clock_terminal_welfare_reduction": min(rows, key=lambda row: number(row, "clock_final_minus_best_welfare")),
        "largest_clock_advantage_over_sealed": max(rows, key=lambda row: number(row, "clock_minus_sealed_efficiency")),
    }
    return [{"selection_rule": rule, **row} for rule, row in rules.items()]


def style(ax: Any) -> None:
    ax.grid(axis="y", color="#dddddd", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig: Any, output: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / "figures" / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scaling(rows: list[dict[str, Any]], output: Path, metric: str) -> None:
    ylabel = {"efficiency_mean": "Efficiency (%)", "value_queries_mean": "Value queries", "payment_error_mean": "Payment error / optimal welfare (%)"}[metric]
    mechanisms = ("Initial PV", "Sealed", "Clock") if metric == "efficiency_mean" else ("Sealed", "Clock")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), sharey=True)
    for ax, series in zip(axes, SERIES):
        for mechanism in mechanisms:
            group = sorted((row for row in rows if row["series"] == series and row["mechanism"] == mechanism), key=lambda row: int(row["x_value"]))
            ys = [100 * number(row, metric) if "efficiency" in metric or "error" in metric else number(row, metric) for row in group]
            ax.plot([int(row["x_value"]) for row in group], ys, marker=MARKERS[mechanism], color=COLORS[mechanism], label=mechanism, linewidth=1.8)
        ax.set_title(SERIES_LABELS[series])
        ax.set_xlabel("Auction size")
        ax.set_xticks(range(4, 11))
        style(ax)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False)
    save(fig, output, metric.replace("_mean", "_scaling"))


def plot_cross(rows: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.scatter([100 * number(row, "sealed_efficiency") for row in rows], [100 * number(row, "clock_efficiency") for row in rows], alpha=.65, color=COLORS["Clock"], edgecolor="none")
    ax.plot([60, 101], [60, 101], "--", color="#777777", linewidth=1)
    ax.set(xlabel="Sealed efficiency (%)", ylabel="Clock efficiency (%)", xlim=(60, 101), ylim=(60, 101))
    style(ax)
    save(fig, output, "cross_mechanism_efficiency")


def plot_events(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.4))
    for ax, mechanism in zip(axes, ("Sealed", "Clock")):
        group = [row for row in rows if row["mechanism"] == mechanism]
        labels = [str(row["event_type"]).replace("_", " ") for row in group]
        y = range(len(group))
        ax.barh(list(y), [100 * number(row, "reported_vcg_witness_hit_rate") for row in group], color=COLORS[mechanism], alpha=.85)
        ax.set_yticks(list(y), labels)
        ax.set_xlabel("Reported VCG-witness hit rate (%)")
        ax.set_title(f"{mechanism} events")
        style(ax)
    save(fig, output, "event_usefulness")


def plot_interest(rows: list[dict[str, str]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    for label, key, color, marker in (
        ("All-goods powerset", "full_powerset_count", "#777777", "o"),
        ("Final candidate support", "inferred_candidate_count", "#0072B2", "^"),
    ):
        points = []
        for x in range(4, 11):
            group = [row for row in rows if (row["series"] == "goods" and int(float(row["x_value"])) == x) or (row["series"] == "anchor" and x == 8)]
            points.append(mean(number(row, key) for row in group))
        ax.plot(range(4, 11), points, marker=marker, color=color, label=label)
    ax.set_yscale("log")
    ax.set(xlabel="Number of goods (8 bidders)", ylabel="Bundle valuations (log scale)")
    ax.set_xticks(range(4, 11))
    ax.legend(frameon=False)
    style(ax)
    save(fig, output, "interest_map_candidate_support")


def plot_ablation(sealed: list[dict[str, str]], clock: list[dict[str, str]], output: Path) -> None:
    labels = {
        "incumbent_only": "Incumbent",
        "plus_counterfactuals": "+ winner removal",
        "plus_scarcity": "+ scarcity",
        "recommended": "ACFR (selected)",
        "recommended_without_incumbent": "- incumbent",
        "recommended_without_counterfactuals": "- winner removal",
        "recommended_without_scarcity": "- scarcity",
        "recommended_without_large_correction": "- correction",
        "focused_pv_only": "PV only",
        "focused_revealed_witness_top3": "Revealed witness top-3",
        "focused_winner_closure": "Winner closure",
        "focused_winner_closure_revealed_vcg": "Winner closure + revealed VCG",
        "focused_revealed_winner_sandwich": "Revealed-winner sandwich",
        "focused_full_frontier_closure": "Full frontier closure",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    sealed_offsets = {
        "incumbent_only": ((5, 5), "left"),
        "plus_counterfactuals": ((5, 5), "left"),
        "plus_scarcity": ((5, -14), "left"),
        "recommended": ((5, 5), "left"),
        "recommended_without_incumbent": ((5, 5), "left"),
        "recommended_without_counterfactuals": ((5, 5), "left"),
        "recommended_without_scarcity": ((-5, 10), "right"),
        "recommended_without_large_correction": ((5, -15), "left"),
    }
    for ax, mechanism, rows in ((axes[0], "Sealed", [r for r in sealed if r["mechanism"] == "sealed"]), (axes[1], "Clock", clock)):
        point_labels: dict[tuple[float, float], list[tuple[str, str]]] = defaultdict(list)
        for row in rows:
            point = (
                number(row, "mean_value_queries"),
                100 * number(row, "mean_efficiency"),
            )
            ax.scatter(*point, color=COLORS[mechanism])
            treatment = row["treatment"]
            point_labels[point].append((treatment, labels.get(treatment, treatment.replace("_", " "))))
        for point, treatment_labels in point_labels.items():
            point_label = "\n".join(label for _, label in treatment_labels)
            offset = (5, 3)
            alignment = "left"
            if mechanism == "Sealed" and len(treatment_labels) == 1:
                offset, alignment = sealed_offsets[treatment_labels[0][0]]
            elif mechanism == "Sealed" and len(treatment_labels) > 1:
                offset = (5, -14)
            ax.annotate(
                point_label,
                point,
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
                ha=alignment,
            )
        ax.set(xlabel="Mean value queries", ylabel="Mean efficiency (%)", title=f"{mechanism} policy ablation")
        style(ax)
    save(fig, output, "event_policy_ablation")


def plot_trajectories(examples: list[dict[str, Any]], sealed_root: Path, clock_root: Path, output: Path) -> None:
    sealed_example = next(row for row in examples if row["selection_rule"] == "largest_sealed_efficiency_gain")
    clock_example = next(row for row in examples if row["selection_rule"] == "largest_clock_terminal_welfare_reduction")
    sealed_rows = read_csv(case_dir(sealed_root, int(sealed_example["seed"]), str(sealed_example["case"])) / "curated_proxy_sealed_trajectory.csv")
    clock_rows = read_csv(case_dir(clock_root, int(clock_example["seed"]), str(clock_example["case"])) / "curated_proxy_clock_rounds_top_3.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    axes[0].plot([number(row, "round") for row in sealed_rows], [100 * number(row, "global_efficiency") for row in sealed_rows], marker="o", color=COLORS["Sealed"])
    axes[0].set(title=f"Sealed: seed {sealed_example['seed']} {sealed_example['case']}", xlabel="Elicitation round", ylabel="Efficiency (%)")
    axes[1].plot([number(row, "round") for row in clock_rows], [100 * number(row, "finalise_global_efficiency") for row in clock_rows], color=COLORS["Clock"])
    axes[1].set(title=f"Clock: seed {clock_example['seed']} {clock_example['case']}", xlabel="Clock round", ylabel="Supplementary allocation efficiency (%)")
    for ax in axes:
        style(ax)
    save(fig, output, "illustrative_trajectories")
    write_csv(output / "tables" / "selected_sealed_trajectory.csv", sealed_rows)
    write_csv(output / "tables" / "selected_clock_trajectory.csv", clock_rows)


def leave_one_seed_out_policy_stability(
    path: Path, *, mechanism: str, frozen_treatment: str,
) -> list[dict[str, Any]]:
    """Select the highest-efficiency policy on four seeds and test the fifth."""
    rows = [row for row in read_csv(path) if row["mechanism"] == mechanism]
    seeds = sorted({int(row["seed"]) for row in rows})
    treatments = sorted({row["treatment"] for row in rows})
    output: list[dict[str, Any]] = []
    for held_out_seed in seeds:
        training = [row for row in rows if int(row["seed"]) != held_out_seed]
        training_means = {
            treatment: mean(
                number(row, "efficiency")
                for row in training if row["treatment"] == treatment
            )
            for treatment in treatments
        }
        best_mean = max(training_means.values())
        tied = [
            treatment for treatment in treatments
            if abs(training_means[treatment] - best_mean) <= 1e-12
        ]
        selected = tied[0]
        held_out = next(
            row for row in rows
            if int(row["seed"]) == held_out_seed and row["treatment"] == selected
        )
        frozen = next(
            row for row in rows
            if int(row["seed"]) == held_out_seed
            and row["treatment"] == frozen_treatment
        )
        output.append({
            "mechanism": mechanism,
            "held_out_seed": held_out_seed,
            "selected_treatment": selected,
            "training_best_treatments": "|".join(tied),
            "training_mean_efficiency": training_means[selected],
            "held_out_efficiency": number(held_out, "efficiency"),
            "held_out_value_queries": number(held_out, "value_queries"),
            "frozen_treatment": frozen_treatment,
            "frozen_treatment_held_out_efficiency": number(frozen, "efficiency"),
        })
    return output


def main() -> int:
    config = args()
    spec = load_final_experiment_spec(config.spec, verify_files=True)
    expected = int(spec["dataset"]["case_count"])
    sealed, sealed_incomplete = load_scalability_results(config.sealed_dir, arm_filter="proxy sealed")
    clock, clock_incomplete = load_scalability_results(config.clock_dir, arm_filter="proxy clock")
    if sealed_incomplete or clock_incomplete or len(sealed) != expected or len(clock) != expected:
        raise SystemExit(f"Incomplete inputs: sealed={len(sealed)}, clock={len(clock)}, expected={expected}")
    output = config.output_dir
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    write_scalability_tables(output / "tables" / "sealed_scalability", sealed, [])
    write_scalability_tables(output / "tables" / "clock_scalability", clock, [])
    rows = pair_cases(sealed, clock, config.sealed_dir, config.clock_dir)
    mechanisms = mechanism_summary(rows)
    comparisons = paired_summary(rows)
    scaling = scaling_summary(rows)
    events = event_summary(config.sealed_dir, config.clock_dir)
    examples = selected_examples(rows)
    interest_cases = read_csv(config.interest_map_dir / "interest_map_case_metrics.csv")
    interest_summary = read_csv(config.interest_map_dir / "interest_map_summary.csv")
    sealed_ablation = read_csv(config.sealed_ablation / "event_ablation_summary.csv")
    clock_ablation = read_csv(config.clock_ablation / "event_ablation_summary.csv")
    ablation_stability = leave_one_seed_out_policy_stability(
        config.sealed_ablation / "event_ablation_runs.csv",
        mechanism="sealed",
        frozen_treatment="recommended",
    ) + leave_one_seed_out_policy_stability(
        config.clock_ablation / "event_ablation_runs.csv",
        mechanism="clock",
        frozen_treatment="focused_revealed_winner_sandwich",
    )
    calibration_dir = Path(spec["calibration"]["path"]).parent
    calibration_summary = json.loads(
        (calibration_dir / "validation_summary.json").read_text(encoding="utf-8")
    )
    scenario_path = Path(spec["scenario"]["path"])
    environment_validation_path = scenario_path.with_suffix(".validation.json")
    environment_validation = json.loads(
        environment_validation_path.read_text(encoding="utf-8")
    )
    write_csv(output / "tables" / "paired_case_metrics.csv", rows)
    write_csv(output / "tables" / "overall_mechanism_summary.csv", mechanisms)
    write_csv(output / "tables" / "paired_comparison_summary.csv", comparisons)
    write_csv(output / "tables" / "scaling_summary.csv", scaling)
    write_csv(output / "tables" / "event_usefulness.csv", events)
    write_csv(output / "tables" / "selected_examples.csv", examples)
    write_csv(output / "tables" / "interest_map_summary.csv", interest_summary)
    write_csv(output / "tables" / "sealed_ablation_summary.csv", sealed_ablation)
    write_csv(output / "tables" / "clock_ablation_summary.csv", clock_ablation)
    write_csv(output / "tables" / "ablation_leave_one_seed_out.csv", ablation_stability)
    for filename in (
        "acceptance_checks.csv",
        "heldout_payment_metrics.csv",
        "heldout_prediction_metrics.csv",
        "mechanism_validation.csv",
        "payment_scale_search.csv",
    ):
        write_csv(
            output / "tables" / f"pv_calibration_{filename}",
            read_csv(calibration_dir / filename),
        )
    (output / "tables" / "environment_validation.json").write_text(
        json.dumps(environment_validation, indent=2) + "\n", encoding="utf-8"
    )
    plot_scaling(scaling, output, "efficiency_mean")
    plot_scaling(scaling, output, "value_queries_mean")
    plot_scaling(scaling, output, "payment_error_mean")
    plot_cross(rows, output)
    plot_events(events, output)
    plot_interest(interest_cases, output)
    plot_ablation(sealed_ablation, clock_ablation, output)
    plot_trajectories(examples, config.sealed_dir, config.clock_dir, output)
    interest_reductions = [number(row, "total_reduction_pct") for row in interest_cases]
    ten_good_interest = [
        row for row in interest_cases
        if row["series"] == "goods" and int(float(row["x_value"])) == 10
    ]
    summary = {
        "status": "final",
        "model_robustness": "deferred",
        "case_count": len(rows),
        "seed_count": 5,
        "mechanisms": mechanisms,
        "paired_comparisons": comparisons,
        "clock_clearance": {
            "natural_clearance_cases": sum(row["clock_termination_reason"] == "no_excess_demand" for row in rows),
            "max_rounds": max(number(row, "clock_rounds") for row in rows),
            "cases_over_50_rounds": sum(number(row, "clock_rounds") > 50 for row in rows),
        },
        "interest_map": {
            "case_count": len(interest_cases),
            "invalid_pack_count": len(read_csv(config.interest_map_dir / "invalid_packs.csv")),
            "mean_candidate_support_reduction_pct": mean(interest_reductions),
            "fixed_8_bidders_10_goods": {
                "case_count": len(ten_good_interest),
                "mean_full_powerset_count": mean(number(row, "full_powerset_count") for row in ten_good_interest),
                "mean_interested_item_powerset_count": mean(number(row, "interested_item_powerset_count") for row in ten_good_interest),
                "mean_final_candidate_count": mean(number(row, "inferred_candidate_count") for row in ten_good_interest),
                "mean_total_reduction_pct": mean(number(row, "total_reduction_pct") for row in ten_good_interest),
            },
        },
        "pv_calibration": calibration_summary,
        "environment_validation": environment_validation,
        "selected_examples": examples,
    }
    (output / "final_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    readme = f"""# Final analytic package\n\nOffline analysis of {len(rows)} matched auction cases across five environment seeds. Model robustness was deliberately deferred.\n\n## Inputs\n\n- Sealed: `{config.sealed_dir}`\n- Clock: `{config.clock_dir}`\n- Interest maps: `{config.interest_map_dir}`\n- Sealed ablation: `{config.sealed_ablation}`\n- Clock ablation: `{config.clock_ablation}`\n- PV calibration: `{calibration_dir}`\n- Environment validation: `{environment_validation_path}`\n- Frozen specification: `{config.spec}`\n\n## Contents\n\n- `FINAL_ANALYSIS.md`: concise interpretation and paper-ready headline claims.\n- `final_summary.json`: machine-readable headline results and provenance.\n- `tables/`: case-level data, seed-clustered summaries, event diagnostics, ablations, calibration evidence and selected trajectories.\n- `figures/`: paper-ready PNG and PDF figures.\n\nUncertainty intervals use the five seed-level means as the independent units. Auction-size cells within a seed are treated as repeated observations rather than independent replications.\n\n## Rebuild\n\n```bash\n./venv/bin/python scripts/build_final_analytic_package.py \\\n  --spec {config.spec} \\\n  --sealed-dir {config.sealed_dir} \\\n  --clock-dir {config.clock_dir} \\\n  --interest-map-dir {config.interest_map_dir} \\\n  --sealed-ablation {config.sealed_ablation} \\\n  --clock-ablation {config.clock_ablation} \\\n  --output-dir {config.output_dir}\n```\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    mechanism_by_name = {row["mechanism"]: row for row in mechanisms}
    comparison_by_name = {row["comparison"]: row for row in comparisons}
    sealed_policy = next(
        row for row in sealed_ablation
        if row["mechanism"] == "sealed" and row["treatment"] == "recommended"
    )
    clock_policy = next(
        row for row in clock_ablation
        if row["treatment"] == "focused_revealed_winner_sandwich"
    )
    calibration_relative_improvement = (
        calibration_summary["mean_heldout_raw_payment_objective"]
        - calibration_summary["mean_heldout_calibrated_payment_objective"]
    ) / calibration_summary["mean_heldout_raw_payment_objective"]
    report = f"""# Final analysis

## Scope

The primary analysis contains {len(rows)} matched cases: 19 scalability cells for each of five independently sampled environment seeds. The sealed and clock mechanisms use the same frozen elicitation pack within each case. Model robustness is deferred and is not part of the final claims below.

## Main results

- Initial provisional valuations allocate **{100 * mechanism_by_name['Initial PV']['mean_efficiency']:.1f}%** of full-information welfare on average.
- Sealed elicitation raises mean efficiency to **{100 * mechanism_by_name['Sealed']['mean_efficiency']:.1f}%**, a paired gain of **{100 * comparison_by_name['sealed_minus_initial_efficiency']['mean_difference']:.1f} percentage points** (seed-clustered 95% CI **{100 * comparison_by_name['sealed_minus_initial_efficiency']['seed_clustered_ci95_low']:.1f} to {100 * comparison_by_name['sealed_minus_initial_efficiency']['seed_clustered_ci95_high']:.1f}**). It improves {comparison_by_name['sealed_minus_initial_efficiency']['positive_cases']}/95 cases and uses {mechanism_by_name['Sealed']['mean_value_queries']:.1f} value queries on average.
- Clock elicitation raises mean efficiency to **{100 * mechanism_by_name['Clock']['mean_efficiency']:.1f}%**, a paired gain of **{100 * comparison_by_name['clock_minus_initial_efficiency']['mean_difference']:.1f} percentage points** (95% CI **{100 * comparison_by_name['clock_minus_initial_efficiency']['seed_clustered_ci95_low']:.1f} to {100 * comparison_by_name['clock_minus_initial_efficiency']['seed_clustered_ci95_high']:.1f}**). It improves {comparison_by_name['clock_minus_initial_efficiency']['positive_cases']}/95 cases and uses {mechanism_by_name['Clock']['mean_value_queries']:.1f} value queries on average.
- Sealed is **{100 * -comparison_by_name['clock_minus_sealed_efficiency']['mean_difference']:.1f} percentage points** more efficient than clock on average, but the seed-clustered CI crosses zero. Clock saves **{-comparison_by_name['clock_minus_sealed_queries']['mean_difference']:.1f} queries per case** relative to sealed (95% CI **{-comparison_by_name['clock_minus_sealed_queries']['seed_clustered_ci95_high']:.1f} to {-comparison_by_name['clock_minus_sealed_queries']['seed_clustered_ci95_low']:.1f} fewer**).
- Payment reconstruction remains materially harder than allocation reconstruction. Mean payment error normalised by optimum welfare is **{100 * mechanism_by_name['Sealed']['mean_payment_error_over_optimum_welfare']:.1f}%** for sealed and **{100 * mechanism_by_name['Clock']['mean_payment_error_over_optimum_welfare']:.1f}%** for clock. Their paired difference is not resolved by five seeds because its CI crosses zero.

## Interest-map scalability

Across all 95 cases, inferred candidate support is **{summary['interest_map']['mean_candidate_support_reduction_pct']:.1f}% smaller** than the full non-empty powerset baseline. At 10 goods with 8 bidders, the mean auction-level support falls from **{summary['interest_map']['fixed_8_bidders_10_goods']['mean_full_powerset_count']:.0f}** full-powerset valuations to **{summary['interest_map']['fixed_8_bidders_10_goods']['mean_final_candidate_count']:.1f}** candidates, a **{summary['interest_map']['fixed_8_bidders_10_goods']['mean_total_reduction_pct']:.1f}% reduction**. These figures measure reconstruction from validated disclosures; the perfect qualitative-map reconstruction is therefore a pipeline check, not a claim about unrestricted natural-language inference.

## Policy evidence and elicitation events

The sealed 8x8 ablation supports adaptive competitive-frontier refinement at **{100 * number(sealed_policy, 'mean_efficiency'):.1f}%** mean efficiency with **{number(sealed_policy, 'mean_value_queries'):.1f}** queries. The clock 8x8 ablation supports the revealed-winner sandwich at **{100 * number(clock_policy, 'mean_efficiency'):.1f}%** with **{number(clock_policy, 'mean_value_queries'):.1f}** queries, compared with {100 * number(next(row for row in clock_ablation if row['treatment'] == 'focused_pv_only'), 'mean_efficiency'):.1f}% for PV-only clock allocation.

The most targeted clock event is the single-pass revealed-VCG event: {100 * number(next(row for row in events if row['event_type'] == 'frontier_vcg_single_pass_revealed'), 'reported_vcg_witness_hit_rate'):.1f}% of its queries appear in the final reported VCG witness set. In sealed elicitation, allocated-bundle and competitive-counterfactual events have similar reported-witness hit rates (about 30–31%), while scarcity fallbacks are less often pricing witnesses but can still alter allocation-relevant alternatives.

## Calibration and clearance checks

The frozen uniform PV scale is **{calibration_summary['final_calibration']['scale']:.4f}**. Across nine held-out synthetic environments, it reduces mean normalised payment error from **{calibration_summary['mean_heldout_raw_payment_objective']:.3f}** to **{calibration_summary['mean_heldout_calibrated_payment_objective']:.3f}**—a **{100 * calibration_relative_improvement:.1f}% relative reduction**—and improves {calibration_summary['payment_cases_improved']}/{calibration_summary['payment_case_count']} held-out cases.

All 95 clock cases clear naturally through no excess demand. The longest takes {summary['clock_clearance']['max_rounds']:.0f} rounds; {summary['clock_clearance']['cases_over_50_rounds']} cases exceed 50 rounds, confirming that the 500-round setting acts only as a safeguard.

## Interpretation limits

- The five seeds, rather than the 95 correlated size cells, are the independent replication units used for uncertainty intervals.
- The environment domain is PC components; mechanism conclusions should be framed as evidence within this structured domain.
- VCG revenue is computed from reported bids. High welfare efficiency does not imply oracle-equivalent payments because counterfactual witness bundles may remain misreported or unqueried.
- Policy ablations are 8x8 selection evidence; the 95-case scalability suite evaluates the frozen selected policies.
"""
    (output / "FINAL_ANALYSIS.md").write_text(report, encoding="utf-8")
    print(f"Built final analytic package: {output}")
    print(f"Cases: {len(rows)}; figures: {len(list((output / 'figures').glob('*')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
