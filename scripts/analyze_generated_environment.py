#!/usr/bin/env python3
"""Analyse a frozen generated population and its accepted scalability cells.

This runner is deliberately offline: it reads the accepted scenario and the
validation report written during population generation.  In particular, it
uses the exact selected goods and bidders recorded for each accepted cell, so
it cannot silently resample a different experimental environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import matplotlib.pyplot as plt


STRATUM_LABELS = {
    "gaming_performance": "Gaming / performance",
    "budget_office": "Budget / office",
    "professional_ai": "Professional / AI",
    "reseller_procurement": "Reseller / procurement",
}
SERIES_LABELS = {
    "goods": "Goods vary",
    "bidders": "Bidders vary",
    "joint": "Both vary",
    "anchor": "8 x 8 anchor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-spec", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def pairwise_mean_jaccard(sets: list[list[str]]) -> float:
    values = [jaccard(left, right) for left, right in combinations(sets, 2)]
    return mean(values) if values else 1.0


def population_tables(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    goods = spec["goods"]
    profiles = spec["bidder_profiles"]
    good_ids = [good["id"] for good in goods]
    bidder_rows: list[dict[str, Any]] = []
    for bidder in profiles:
        positive = {good for good, value in bidder["base_values"].items() if float(value) > 0}
        core = positive & set(bidder.get("core_items", []))
        secondary = positive & set(bidder.get("secondary_items", []))
        low = positive & set(bidder.get("low_interest_items", []))
        bidder_rows.append({
            "bidder_id": bidder["bidder_id"],
            "stratum": bidder.get("archetype_category") or "unclassified",
            "positive_goods": len(positive),
            "interest_density": len(positive) / len(good_ids),
            "core_goods": len(core),
            "secondary_goods": len(secondary),
            "low_positive_goods": len(low),
            "zero_value_goods": len(good_ids) - len(positive),
            "substitute_groups": len(bidder.get("substitute_groups", [])),
            "choose_one_groups": sum(
                group.get("acquisition_mode") == "choose_one"
                for group in bidder.get("substitute_groups", [])
            ),
            "can_use_multiple_groups": sum(
                group.get("acquisition_mode") == "can_use_multiple"
                for group in bidder.get("substitute_groups", [])
            ),
            "mean_substitute_group_size": (
                mean(len(group["items"]) for group in bidder.get("substitute_groups", []))
                if bidder.get("substitute_groups") else 0
            ),
            "complement_groups": len(bidder.get("complement_groups", [])),
            "mean_complement_group_size": (
                mean(len(group["items"]) for group in bidder.get("complement_groups", []))
                if bidder.get("complement_groups") else 0
            ),
            "mean_complement_bonus": (
                mean(float(group["bonus"]) for group in bidder.get("complement_groups", []))
                if bidder.get("complement_groups") else 0
            ),
            "budget_cap": bidder.get("budget_cap"),
            "saturation_start": bidder.get("saturation_start"),
        })

    good_rows: list[dict[str, Any]] = []
    for good in goods:
        good_id = good["id"]
        positive_bidders = [
            bidder for bidder in profiles
            if float(bidder["base_values"].get(good_id, 0)) > 0
        ]
        core_bidders = [
            bidder for bidder in positive_bidders
            if good_id in bidder.get("core_items", [])
        ]
        good_rows.append({
            "good_id": good_id,
            "category": good.get("category") or "uncategorized",
            "positive_bidders": len(positive_bidders),
            "positive_bidder_share": len(positive_bidders) / len(profiles),
            "core_bidders": len(core_bidders),
            "positive_strata": len({
                bidder.get("archetype_category") or "unclassified"
                for bidder in positive_bidders
            }),
        })
    return bidder_rows, good_rows


def sample_table(validation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in validation["sample_validation"]["cases"]:
        structural = case["structural"]
        economic = case["economic"]
        positive_by_good = structural["positive_bidders_by_good"]
        allocation = economic["allocation"]
        allocated_goods = {good for bundle in allocation.values() for good in bundle}
        rows.append({
            "seed": int(case["seed"]),
            "case": case["case"],
            "series": case["series"],
            "num_goods": int(structural["num_goods"]),
            "num_bidders": int(structural["num_bidders"]),
            "selected_goods": "|".join(case["selected_goods"]),
            "selected_bidders": "|".join(case["selected_bidders"]),
            "good_categories": len(structural["selected_good_categories"]),
            "bidder_strata": len(structural["selected_bidder_strata"]),
            "mean_interest_density": float(structural["mean_interest_density"]),
            "bidders_with_exclusions": len(structural["bidders_with_exclusions"]),
            "surviving_substitute_groups": len(structural["surviving_substitute_groups"]),
            "distinct_substitute_groups": len(structural["distinct_substitute_groups"]),
            "bidders_with_substitutes": len(structural["bidders_with_substitute_groups"]),
            "surviving_complement_groups": len(structural["surviving_complement_groups"]),
            "distinct_complement_groups": len(structural["distinct_complement_groups"]),
            "bidders_with_complements": len(structural["bidders_with_complement_groups"]),
            "contested_goods": sum(len(bidders) >= 2 for bidders in positive_by_good.values()),
            "min_positive_bidders_per_good": min(map(len, positive_by_good.values())),
            "mean_positive_bidders_per_good": mean(map(len, positive_by_good.values())),
            "full_information_welfare": float(economic["full_information_welfare"]),
            "full_information_winners": int(economic["num_winners"]),
            "largest_winner_welfare_share": float(economic["largest_winner_welfare_share"]),
            "allocated_good_share": len(allocated_goods) / len(case["selected_goods"]),
            "validation_passed": bool(case["passed"]),
        })
    return rows


def diversity_table(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        grouped[(row["series"], row["num_goods"], row["num_bidders"])].append(row)
    output: list[dict[str, Any]] = []
    for (series, num_goods, num_bidders), rows in sorted(grouped.items()):
        good_sets = [row["selected_goods"].split("|") for row in rows]
        bidder_sets = [row["selected_bidders"].split("|") for row in rows]
        output.append({
            "series": series,
            "num_goods": num_goods,
            "num_bidders": num_bidders,
            "seeds": len(rows),
            "distinct_good_sets": len({tuple(sorted(items)) for items in good_sets}),
            "distinct_bidder_sets": len({tuple(sorted(items)) for items in bidder_sets}),
            "mean_pairwise_good_jaccard": pairwise_mean_jaccard(good_sets),
            "mean_pairwise_bidder_jaccard": pairwise_mean_jaccard(bidder_sets),
        })
    return output


def overview_rows(
    spec: dict[str, Any], bidder_rows: list[dict[str, Any]],
    good_rows: list[dict[str, Any]], samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"metric": "master_goods", "value": len(spec["goods"])},
        {"metric": "master_bidders", "value": len(spec["bidder_profiles"])},
        {"metric": "bidder_strata", "value": len({row["stratum"] for row in bidder_rows})},
        {"metric": "good_categories", "value": len({row["category"] for row in good_rows})},
        {"metric": "mean_population_interest_density", "value": mean(row["interest_density"] for row in bidder_rows)},
        {"metric": "min_population_interest_density", "value": min(row["interest_density"] for row in bidder_rows)},
        {"metric": "max_population_interest_density", "value": max(row["interest_density"] for row in bidder_rows)},
        {"metric": "mean_positive_bidders_per_good", "value": mean(row["positive_bidders"] for row in good_rows)},
        {"metric": "min_positive_bidders_per_good", "value": min(row["positive_bidders"] for row in good_rows)},
        {"metric": "max_positive_bidders_per_good", "value": max(row["positive_bidders"] for row in good_rows)},
        {"metric": "choose_one_groups", "value": sum(row["choose_one_groups"] for row in bidder_rows)},
        {"metric": "can_use_multiple_groups", "value": sum(row["can_use_multiple_groups"] for row in bidder_rows)},
        {"metric": "complement_groups", "value": sum(row["complement_groups"] for row in bidder_rows)},
        {"metric": "accepted_scalability_cells", "value": len(samples)},
        {"metric": "mean_sample_interest_density", "value": mean(row["mean_interest_density"] for row in samples)},
        {"metric": "mean_full_information_winners", "value": mean(row["full_information_winners"] for row in samples)},
        {"metric": "median_full_information_winners", "value": median(row["full_information_winners"] for row in samples)},
        {"metric": "mean_largest_winner_welfare_share", "value": mean(row["largest_winner_welfare_share"] for row in samples)},
        {"metric": "max_largest_winner_welfare_share", "value": max(row["largest_winner_welfare_share"] for row in samples)},
        {"metric": "mean_allocated_good_share", "value": mean(row["allocated_good_share"] for row in samples)},
        {"metric": "all_cells_contested_on_every_good", "value": all(row["contested_goods"] == row["num_goods"] for row in samples)},
        {"metric": "all_cells_passed_validation", "value": all(row["validation_passed"] for row in samples)},
    ]


def plot_environment(
    bidder_rows: list[dict[str, Any]], good_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]], output_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)

    ordered_goods = sorted(good_rows, key=lambda row: (row["category"], row["good_id"]))
    axes[0].bar(range(len(ordered_goods)), [row["positive_bidders"] for row in ordered_goods], color="#0072B2")
    axes[0].set_xticks(range(len(ordered_goods)), [row["good_id"] for row in ordered_goods], rotation=65, ha="right", fontsize=7)
    axes[0].set_ylabel("Positive bidders (of 16)")
    axes[0].set_title("(a) Competition by good")
    axes[0].set_ylim(0, 16.8)

    strata = sorted({row["stratum"] for row in bidder_rows})
    values = [[100 * row["interest_density"] for row in bidder_rows if row["stratum"] == stratum] for stratum in strata]
    axes[1].boxplot(values, tick_labels=[STRATUM_LABELS.get(value, value) for value in strata], patch_artist=True,
                    boxprops={"facecolor": "#56B4E9", "alpha": .65}, medianprops={"color": "black"})
    for index, group in enumerate(values, start=1):
        axes[1].scatter([index] * len(group), group, color="#0072B2", s=18, zorder=3)
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    axes[1].set_ylabel("Positive-good share (%)")
    axes[1].set_title("(b) Interest density by stratum")

    scatter = axes[2].scatter(
        [row["num_goods"] for row in samples],
        [row["full_information_winners"] for row in samples],
        s=[20 + 8 * row["num_bidders"] for row in samples],
        c=[100 * row["largest_winner_welfare_share"] for row in samples],
        cmap="viridis_r", vmin=20, vmax=80, alpha=.72, edgecolor="white", linewidth=.4,
    )
    axes[2].set_xlabel("Number of goods")
    axes[2].set_ylabel("Efficient-allocation winners")
    axes[2].set_title("(c) Allocation non-triviality")
    colorbar = fig.colorbar(scatter, ax=axes[2], pad=.02)
    colorbar.set_label("Largest winner welfare share (%)", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"environment_structure.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    options = parse_args()
    spec = json.loads(options.scenario_spec.read_text(encoding="utf-8"))
    validation = json.loads(options.validation_report.read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise ValueError("validation report is not marked as passed")
    bidder_rows, good_rows = population_tables(spec)
    samples = sample_table(validation)
    diversity = diversity_table(samples)
    overview = overview_rows(spec, bidder_rows, good_rows, samples)

    tables = options.output_dir / "tables"
    write_csv(tables / "environment_overview.csv", overview)
    write_csv(tables / "environment_bidders.csv", bidder_rows)
    write_csv(tables / "environment_goods.csv", good_rows)
    write_csv(tables / "environment_samples.csv", samples)
    write_csv(tables / "environment_sample_diversity.csv", diversity)
    plot_environment(bidder_rows, good_rows, samples, options.output_dir)

    audit = {
        "scenario_spec": str(options.scenario_spec),
        "validation_report": str(options.validation_report),
        "validation_passed": validation["passed"],
        "population_rows": len(bidder_rows),
        "good_rows": len(good_rows),
        "sample_rows": len(samples),
        "expected_sample_rows": 95,
        "all_sample_rows_passed": all(row["validation_passed"] for row in samples),
        "all_goods_contested_in_every_sample": all(row["contested_goods"] == row["num_goods"] for row in samples),
        "finite_numeric_outputs": all(
            not isinstance(value, float) or math.isfinite(value)
            for rows in (overview, bidder_rows, good_rows, samples, diversity)
            for row in rows for value in row.values()
        ),
    }
    (options.output_dir / "environment_analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if len(samples) != audit["expected_sample_rows"] or not audit["all_sample_rows_passed"]:
        raise ValueError(f"environment analysis audit failed: {audit}")
    print(f"Analysed {len(bidder_rows)} bidders, {len(good_rows)} goods, and {len(samples)} accepted cells")
    print(f"Tables: {tables}")
    print(f"Figure: {options.output_dir / 'figures/environment_structure.pdf'}")


if __name__ == "__main__":
    main()
