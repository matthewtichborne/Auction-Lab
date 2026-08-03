#!/usr/bin/env python3
"""Build provisional paper figures from completed frozen and mechanism runs.

The script is intentionally offline: it reads existing CSV artifacts and
makes no LLM calls.  Interest-map tables should first be generated with
``scripts/analyze_interest_maps.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import matplotlib.pyplot as plt


SERIES_LABELS = {
    "bidders": "8 goods, bidders vary",
    "goods": "8 bidders, goods vary",
    "joint": "Goods and bidders vary",
}
SERIES_ORDER = ("bidders", "goods", "joint")
ARM_STYLE = {
    "Initial PV": {"color": "#777777", "marker": "o"},
    "Sealed": {"color": "#0072B2", "marker": "s"},
    "Clock": {"color": "#D55E00", "marker": "^"},
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _seed_from_root(root: Path) -> int:
    manifest = _read_csv(root / "scalability_runs.csv")
    seeds = {int(float(row["seed"])) for row in manifest if row.get("seed")}
    if len(seeds) != 1:
        raise ValueError(f"Expected one seed below {root}, found {sorted(seeds)}")
    return next(iter(seeds))


def _case_dirs(root: Path, seed: int) -> Iterable[Path]:
    seed_dir = root / f"seed_{seed}"
    for path in sorted(seed_dir.iterdir()):
        if path.is_dir() and (path / "curated_run_summary.csv").exists():
            yield path


def _load_mechanism_rows(
    roots: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    outcome_rows: list[dict[str, Any]] = []
    sealed_events: list[dict[str, Any]] = []
    clock_events: list[dict[str, Any]] = []

    for root in roots:
        seed = _seed_from_root(root)
        sealed_metrics = {
            row["case"]: row
            for row in _read_csv(root / "analysis" / "sealed" / "scalability_metrics.csv")
        }
        clock_metrics = {
            row["case"]: row
            for row in _read_csv(root / "analysis" / "clock" / "scalability_metrics.csv")
        }
        for case_dir in _case_dirs(root, seed):
            case = case_dir.name
            if case not in sealed_metrics or case not in clock_metrics:
                raise ValueError(f"Missing aggregate metrics for {root}/{case}")
            sealed_trajectory = _read_csv(
                case_dir / "curated_proxy_sealed_trajectory.csv"
            )
            initial = min(
                sealed_trajectory,
                key=lambda row: int(float(row["round"])),
            )
            base = sealed_metrics[case]
            series = base["series"]
            x_value = int(float(base["x_value"]))
            outcome_rows.append(
                {
                    "seed": seed,
                    "case": case,
                    "series": series,
                    "x": x_value,
                    "num_goods": int(float(base["num_goods"])),
                    "num_bidders": int(float(base["num_bidders"])),
                    "initial_efficiency_pct": 100.0
                    * _number(initial, "global_efficiency"),
                    "sealed_efficiency_pct": _number(
                        sealed_metrics[case], "efficiency_pct"
                    ),
                    "clock_efficiency_pct": _number(
                        clock_metrics[case], "efficiency_pct"
                    ),
                    "sealed_queries": _number(
                        sealed_metrics[case], "person_queries"
                    ),
                    "clock_queries": _number(
                        clock_metrics[case], "person_queries"
                    ),
                    "candidates": _number(
                        sealed_metrics[case], "candidate_bundles_sent_to_pv"
                    ),
                    "sealed_revenue_difference_pct": _number(
                        sealed_metrics[case], "revenue_difference_pct"
                    ),
                    "clock_revenue_difference_pct": _number(
                        clock_metrics[case], "revenue_difference_pct"
                    ),
                }
            )
            sealed_path = case_dir / "curated_refinement_records.csv"
            if sealed_path.exists():
                for row in _read_csv(sealed_path):
                    if row.get("mechanism") != "proxy_sealed_vcg":
                        continue
                    row["_seed"] = str(seed)
                    row["_case"] = case
                    sealed_events.append(row)
            clock_path = (
                case_dir / "curated_proxy_clock_event_usefulness_top_3.csv"
            )
            if clock_path.exists():
                for row in _read_csv(clock_path):
                    row["_seed"] = str(seed)
                    row["_case"] = case
                    clock_events.append(row)
    return outcome_rows, sealed_events, clock_events


def _expand_anchor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        if row["series"] != "anchor":
            expanded.append(row)
            continue
        for series in SERIES_ORDER:
            copy = dict(row)
            copy["series"] = series
            expanded.append(copy)
    return expanded


def _group_values(
    rows: list[dict[str, Any]],
    series: str,
    key: str,
) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["series"] == series and not math.isnan(float(row[key])):
            grouped[int(row["x"])].append(float(row[key]))
    return dict(grouped)


def _style_axes(ax: Any) -> None:
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_scaling(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    metric: str,
) -> Path:
    expanded = _expand_anchor(rows)
    if metric == "efficiency":
        keys = {
            "Initial PV": "initial_efficiency_pct",
            "Sealed": "sealed_efficiency_pct",
            "Clock": "clock_efficiency_pct",
        }
        ylabel = "True allocation efficiency (%)"
        filename = "mechanism_efficiency_scaling.pdf"
    else:
        keys = {
            "Sealed": "sealed_queries",
            "Clock": "clock_queries",
        }
        ylabel = "Exact person value queries"
        filename = "mechanism_query_scaling.pdf"

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.55), sharey=True)
    for ax, series in zip(axes, SERIES_ORDER):
        for label, key in keys.items():
            grouped = _group_values(expanded, series, key)
            xs = sorted(grouped)
            means = [mean(grouped[x]) for x in xs]
            lows = [min(grouped[x]) for x in xs]
            highs = [max(grouped[x]) for x in xs]
            style = ARM_STYLE[label]
            ax.plot(
                xs,
                means,
                label=label,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4.5,
            )
            ax.fill_between(
                xs,
                lows,
                highs,
                color=style["color"],
                alpha=0.12,
                linewidth=0,
            )
            for x in xs:
                ax.scatter(
                    [x] * len(grouped[x]),
                    grouped[x],
                    color=style["color"],
                    s=11,
                    alpha=0.38,
                    linewidths=0,
                )
        ax.set_title(SERIES_LABELS[series], fontsize=9.5)
        ax.set_xlabel("Number varied")
        ax.set_xticks(range(4, 11))
        _style_axes(ax)
    axes[0].set_ylabel(ylabel)
    if metric == "efficiency":
        axes[0].set_ylim(68, 102)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_cross_mechanism(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    colors = {
        "bidders": "#009E73",
        "goods": "#CC79A7",
        "joint": "#E69F00",
        "anchor": "#333333",
    }
    for series in ("bidders", "goods", "joint", "anchor"):
        subset = [row for row in rows if row["series"] == series]
        if not subset:
            continue
        ax.scatter(
            [row["sealed_efficiency_pct"] for row in subset],
            [row["clock_efficiency_pct"] for row in subset],
            color=colors[series],
            label=SERIES_LABELS.get(series, "8x8 anchor"),
            s=34,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.4,
        )
    low = min(
        min(row["sealed_efficiency_pct"], row["clock_efficiency_pct"])
        for row in rows
    ) - 1
    ax.plot([low, 101], [low, 101], color="#666666", linestyle="--", linewidth=1)
    ax.set_xlim(low, 101)
    ax.set_ylim(low, 101)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Sealed efficiency (%)")
    ax.set_ylabel("Clock efficiency (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style_axes(ax)
    path = output_dir / "cross_mechanism_efficiency.pdf"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _event_summary(
    rows: list[dict[str, Any]],
    *,
    clock: bool,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["event_type"]].append(row)
    result = []
    for event_type, event_rows in grouped.items():
        item = {
            "event_type": event_type,
            "count": len(event_rows),
            "mean_abs_correction": mean(
                _number(row, "abs_correction" if clock else "value_delta")
                for row in event_rows
            ),
        }
        if clock:
            item["final_hit_pct"] = 100.0 * mean(
                1.0
                if row.get("appears_in_final_allocation", "").lower() == "true"
                else 0.0
                for row in event_rows
            )
            item["oracle_hit_pct"] = 100.0 * mean(
                1.0
                if row.get("appears_in_full_info_allocation", "").lower()
                == "true"
                else 0.0
                for row in event_rows
            )
        result.append(item)
    return sorted(result, key=lambda item: item["count"], reverse=True)


def _pretty_event(value: str) -> str:
    return value.replace("_", " ").replace("allocation ", "alloc. ")


def _plot_events(
    sealed_summary: list[dict[str, Any]],
    clock_summary: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    panels = [
        ("Sealed events", sealed_summary, "#0072B2", False),
        ("Clock events", clock_summary, "#D55E00", True),
    ]
    for ax, (title, summary, color, is_clock) in zip(axes, panels):
        ordered = list(reversed(summary))
        ys = list(range(len(ordered)))
        counts = [item["count"] for item in ordered]
        ax.barh(ys, counts, color=color, alpha=0.8)
        ax.set_yticks(ys, [_pretty_event(item["event_type"]) for item in ordered])
        ax.set_xlabel("Exact value queries")
        ax.set_title(title)
        for y, item in zip(ys, ordered):
            annotation = f"  mean |Δ|={item['mean_abs_correction']:.0f}"
            if is_clock:
                annotation += f"; final hit={item['final_hit_pct']:.0f}%"
            ax.text(item["count"], y, annotation, va="center", fontsize=7.5)
        ax.set_xlim(0, max(counts) * 1.85)
        _style_axes(ax)
    fig.tight_layout()
    path = output_dir / "event_usefulness.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_illustrative_trajectory(root: Path, output_dir: Path) -> Path:
    seed = _seed_from_root(root)
    case_dir = root / f"seed_{seed}" / "anchor_8x8"
    sealed_trajectory = _read_csv(
        case_dir / "curated_proxy_sealed_trajectory.csv"
    )
    summary = _read_csv(case_dir / "curated_run_summary.csv")
    final_efficiency = {
        row["arm"]: 100.0 * _number(row, "efficiency")
        for row in summary
        if row["arm"] in ("proxy sealed", "proxy clock k=3")
    }
    initial_efficiency = 100.0 * _number(
        min(
            sealed_trajectory,
            key=lambda row: int(float(row["round"])),
        ),
        "global_efficiency",
    )

    sealed_events = [
        row
        for row in _read_csv(case_dir / "curated_refinement_records.csv")
        if row.get("mechanism") == "proxy_sealed_vcg"
    ]
    clock_events = _read_csv(
        case_dir / "curated_proxy_clock_event_usefulness_top_3.csv"
    )
    trajectories = [
        (
            "Sealed",
            sealed_events,
            "round_idx",
            final_efficiency["proxy sealed"],
        ),
        (
            "Clock",
            clock_events,
            "round",
            final_efficiency["proxy clock k=3"],
        ),
    ]
    event_colors = {
        "allocated_bundle": "#0072B2",
        "competitive_counterfactual": "#56B4E9",
        "allocation_changed_bundle": "#D55E00",
        "allocation_counterfactual": "#E69F00",
        "near_zero_surplus": "#009E73",
        "terminal_counterfactual": "#CC79A7",
        "allocation_pivotal_near_tie": "#666666",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for ax, (title, event_rows, round_key, final_eff) in zip(axes, trajectories):
        event_order = list(dict.fromkeys(row["event_type"] for row in event_rows))
        y_for_event = {
            event_type: idx for idx, event_type in enumerate(event_order)
        }
        rounds = [int(float(row[round_key])) for row in event_rows]
        seen_at_position: dict[tuple[int, str], int] = defaultdict(int)
        for row in event_rows:
            round_idx = int(float(row[round_key]))
            event_type = row["event_type"]
            duplicate = seen_at_position[(round_idx, event_type)]
            seen_at_position[(round_idx, event_type)] += 1
            offset = 0.0 if duplicate == 0 else 0.07 * (
                1 if duplicate % 2 else -1
            ) * ((duplicate + 1) // 2)
            ax.scatter(
                round_idx + offset,
                y_for_event[event_type],
                color=event_colors.get(event_type, "#444444"),
                s=28,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.35,
            )
        query_ax = ax.twinx()
        unique_rounds = sorted(set(rounds))
        cumulative = [
            sum(round_idx <= current for round_idx in rounds)
            for current in unique_rounds
        ]
        query_ax.step(
            [0, *unique_rounds],
            [0, *cumulative],
            where="post",
            color="#333333",
            linewidth=1.6,
            label="Cumulative VQs",
        )
        ax.set_title(
            f"{title}: {initial_efficiency:.1f}% → {final_eff:.1f}%",
            fontsize=10,
        )
        ax.set_xlabel("Round")
        ax.set_yticks(
            list(range(len(event_order))),
            [_pretty_event(value) for value in event_order],
            fontsize=8,
        )
        query_ax.set_ylabel("Cumulative exact VQs", color="#333333")
        query_ax.set_ylim(0, len(event_rows) * 1.12)
        query_ax.tick_params(axis="y", colors="#333333")
        ax.set_xlim(-0.5, max(rounds) + 0.7)
        ax.set_ylim(-0.6, len(event_order) - 0.4)
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        query_ax.spines["top"].set_visible(False)
    fig.tight_layout()
    path = output_dir / "illustrative_8x8_trajectory.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _summary(
    outcome_rows: list[dict[str, Any]],
    sealed_events: list[dict[str, Any]],
    clock_events: list[dict[str, Any]],
    interest_case_path: Path,
) -> dict[str, Any]:
    initial = [row["initial_efficiency_pct"] for row in outcome_rows]
    sealed = [row["sealed_efficiency_pct"] for row in outcome_rows]
    clock = [row["clock_efficiency_pct"] for row in outcome_rows]
    sealed_q = [row["sealed_queries"] for row in outcome_rows]
    clock_q = [row["clock_queries"] for row in outcome_rows]
    differences = [
        row["clock_efficiency_pct"] - row["sealed_efficiency_pct"]
        for row in outcome_rows
    ]
    tolerance = 1e-8

    interest_rows = _read_csv(interest_case_path)
    interest_goods = [
        row
        for row in interest_rows
        if row["series"] == "goods"
        or (row["series"] == "anchor" and int(float(row["x_value"])) == 8)
    ]
    goods_10 = [
        row
        for row in interest_rows
        if row["series"] == "goods" and int(float(row["x_value"])) == 10
    ]
    anchor = [row for row in interest_rows if row["series"] == "anchor"]

    sealed_event_summary = _event_summary(sealed_events, clock=False)
    clock_event_summary = _event_summary(clock_events, clock=True)
    return {
        "status": "provisional",
        "mechanism_seed_count": len({row["seed"] for row in outcome_rows}),
        "mechanism_case_count": len(outcome_rows),
        "interest_map_seed_count": len(
            {int(float(row["seed"])) for row in interest_rows}
        ),
        "interest_map_case_count": len(interest_rows),
        "initial_efficiency_pct": {
            "mean": mean(initial),
            "median": median(initial),
            "min": min(initial),
            "max": max(initial),
        },
        "sealed": {
            "efficiency_pct_mean": mean(sealed),
            "efficiency_pct_median": median(sealed),
            "efficiency_pct_min": min(sealed),
            "efficiency_pct_max": max(sealed),
            "queries_mean": mean(sealed_q),
            "queries_median": median(sealed_q),
            "improved_cases": sum(
                final > start + tolerance
                for final, start in zip(sealed, initial)
            ),
            "worsened_cases": sum(
                final < start - tolerance
                for final, start in zip(sealed, initial)
            ),
            "unchanged_cases": sum(
                abs(final - start) <= tolerance
                for final, start in zip(sealed, initial)
            ),
            "mean_gain_pp": mean(
                final - start for final, start in zip(sealed, initial)
            ),
        },
        "clock": {
            "efficiency_pct_mean": mean(clock),
            "efficiency_pct_median": median(clock),
            "efficiency_pct_min": min(clock),
            "efficiency_pct_max": max(clock),
            "queries_mean": mean(clock_q),
            "queries_median": median(clock_q),
            "improved_cases": sum(
                final > start + tolerance
                for final, start in zip(clock, initial)
            ),
            "worsened_cases": sum(
                final < start - tolerance
                for final, start in zip(clock, initial)
            ),
            "unchanged_cases": sum(
                abs(final - start) <= tolerance
                for final, start in zip(clock, initial)
            ),
            "mean_gain_pp": mean(
                final - start for final, start in zip(clock, initial)
            ),
        },
        "cross_mechanism": {
            "mean_clock_minus_sealed_pp": mean(differences),
            "ties": sum(abs(value) <= tolerance for value in differences),
            "clock_wins": sum(value > tolerance for value in differences),
            "sealed_wins": sum(value < -tolerance for value in differences),
            "clock_to_sealed_query_ratio": mean(clock_q) / mean(sealed_q),
        },
        "interest_map": {
            "mean_total_reduction_pct_all_cases": mean(
                _number(row, "total_reduction_pct") for row in interest_rows
            ),
            "fixed_8_bidders_10_goods": {
                "full_powerset_mean": mean(
                    _number(row, "full_powerset_count") for row in goods_10
                ),
                "interested_powerset_mean": mean(
                    _number(row, "interested_item_powerset_count")
                    for row in goods_10
                ),
                "candidate_support_mean": mean(
                    _number(row, "inferred_candidate_count") for row in goods_10
                ),
                "total_reduction_pct_mean": mean(
                    _number(row, "total_reduction_pct") for row in goods_10
                ),
            },
            "anchor_8x8": {
                "full_powerset_mean": mean(
                    _number(row, "full_powerset_count") for row in anchor
                ),
                "candidate_support_mean": mean(
                    _number(row, "inferred_candidate_count") for row in anchor
                ),
                "total_reduction_pct_mean": mean(
                    _number(row, "total_reduction_pct") for row in anchor
                ),
            },
            "fixed_bidder_case_count": len(interest_goods),
        },
        "sealed_events": sealed_event_summary,
        "clock_events": clock_event_summary,
    }


def _write_outcome_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mechanism-input",
        type=Path,
        nargs="+",
        required=True,
        help="Completed scalability roots, one per seed.",
    )
    parser.add_argument(
        "--interest-map-cases",
        type=Path,
        required=True,
        help="Combined interest_map_case_metrics.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcome_rows, sealed_events, clock_events = _load_mechanism_rows(
        args.mechanism_input
    )
    if not outcome_rows:
        raise SystemExit("No mechanism result rows found")

    summary = _summary(
        outcome_rows,
        sealed_events,
        clock_events,
        args.interest_map_cases,
    )
    summary_path = args.output_dir / "provisional_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_outcome_csv(args.output_dir / "provisional_outcomes.csv", outcome_rows)

    sealed_summary = _event_summary(sealed_events, clock=False)
    clock_summary = _event_summary(clock_events, clock=True)
    paths = [
        _plot_scaling(outcome_rows, args.output_dir, metric="efficiency"),
        _plot_scaling(outcome_rows, args.output_dir, metric="queries"),
        _plot_cross_mechanism(outcome_rows, args.output_dir),
        _plot_events(sealed_summary, clock_summary, args.output_dir),
        _plot_illustrative_trajectory(args.mechanism_input[0], args.output_dir),
    ]
    print(
        f"Wrote {len(outcome_rows)} provisional mechanism observations "
        f"from {summary['mechanism_seed_count']} seeds."
    )
    print(f"Summary: {summary_path}")
    for path in paths:
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
