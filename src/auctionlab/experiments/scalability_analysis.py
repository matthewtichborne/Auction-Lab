"""Aggregate and plot outputs from scalability experiment directories."""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


CASE_RE = re.compile(
    r"^(?P<series>goods|bidders|joint|anchor)_(?P<goods>\d+)x(?P<bidders>\d+)$"
)

SCALABILITY_FIELDS = [
    "seed",
    "case",
    "series",
    "x_value",
    "num_goods",
    "num_bidders",
    "scenario",
    "arm",
    "efficiency",
    "efficiency_pct",
    "initial_efficiency",
    "efficiency_gain_from_initial_pct",
    "true_welfare",
    "full_info_welfare",
    "welfare_loss",
    "revenue",
    "full_info_revenue",
    "revenue_difference",
    "revenue_difference_pct",
    "revenue_loss",
    "revenue_absolute_percentage_error",
    "payment_error_over_optimum_welfare",
    "initial_revenue_loss",
    "revenue_loss_improvement_from_initial",
    "surplus",
    "value_queries",
    "demand_queries",
    "nl_queries",
    "person_queries",
    "person_tokens_in",
    "person_tokens_out",
    "person_tokens_total",
    "shared_proxy_tokens_in",
    "shared_proxy_tokens_out",
    "candidate_bundles_generated",
    "candidate_bundles_sent_to_pv",
    "candidate_bundles_truncated",
    "pv_calibration_family",
    "pv_calibration_scale",
    "pv_calibration_size_gamma",
    "pv_calibration_size_threshold",
    "pv_calibration_budget_cap",
    "pv_calibration_config_hash",
]

#: Calibration columns echoed from each case's ``curated_run_summary.csv``.
#: Aggregating cases run under different calibrations would silently mix
#: treatments, so the calibration travels with every row.
_CALIBRATION_PASSTHROUGH = (
    "pv_calibration_family",
    "pv_calibration_scale",
    "pv_calibration_size_gamma",
    "pv_calibration_size_threshold",
    "pv_calibration_budget_cap",
    "pv_calibration_config_hash",
)

INCOMPLETE_FIELDS = ["seed", "case", "path", "reason"]


@dataclass(frozen=True)
class ScalabilityCase:
    """One successfully completed scalability experiment case."""

    values: dict[str, str | int | float]


def _read_csv(path: Path) -> list[dict[str, str]]:
    # Large 9x9/10x10 diagnostic rows can embed complete bundle/allocation
    # structures in one CSV field and exceed the stdlib's conservative
    # 128 KiB default. These are trusted local experiment artefacts.
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(
    row: dict[str, str],
    key: str,
    *,
    default: float = math.nan,
) -> float:
    raw = row.get(key, "")
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _integer(row: dict[str, str], key: str) -> int:
    value = _number(row, key, default=0.0)
    return int(value) if math.isfinite(value) else 0


def _seed_label(case_dir: Path, root: Path) -> str:
    for parent in (case_dir, *case_dir.parents):
        if parent == root.parent:
            break
        if parent.name.startswith("seed_"):
            return parent.name.removeprefix("seed_")
    return ""


def _choose_arm(
    rows: Sequence[dict[str, str]],
    arm_filter: str | None,
) -> dict[str, str]:
    auction_rows = [
        row
        for row in rows
        if not row.get("arm", "").lower().startswith("shared initial")
    ]
    if arm_filter:
        matching = [
            row
            for row in auction_rows
            if arm_filter.lower() in row.get("arm", "").lower()
        ]
        if len(matching) != 1:
            raise ValueError(
                f"arm filter {arm_filter!r} matched {len(matching)} rows"
            )
        return matching[0]
    if len(auction_rows) != 1:
        arms = ", ".join(row.get("arm", "") for row in auction_rows)
        raise ValueError(
            "expected exactly one auction arm; use --arm to select one "
            f"(found: {arms or 'none'})"
        )
    return auction_rows[0]


def _find_result_row(
    case_dir: Path,
    arm: str,
) -> dict[str, str] | None:
    mechanism_hint = "clock" if "clock" in arm.lower() else "sealed"
    paths = sorted(
        case_dir.glob(f"curated_{mechanism_hint}_proxy_elicited*.csv")
    )
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_csv(path))
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    exact = [row for row in rows if row.get("mechanism") == arm]
    return exact[0] if len(exact) == 1 else rows[0]


def _candidate_totals(case_dir: Path) -> tuple[int, int, int]:
    path = case_dir / "curated_pv_candidate_bundle_stats.csv"
    if not path.exists():
        return 0, 0, 0
    rows = _read_csv(path)
    return (
        sum(_integer(row, "candidate_bundles_generated") for row in rows),
        sum(_integer(row, "candidate_bundles_sent_to_pv") for row in rows),
        sum(_integer(row, "candidate_bundles_truncated") for row in rows),
    )


def _initial_trajectory_row(case_dir: Path) -> dict[str, str] | None:
    path = case_dir / "curated_proxy_sealed_trajectory.csv"
    if not path.exists():
        return None
    rows = _read_csv(path)
    if not rows:
        return None
    return min(rows, key=lambda row: _integer(row, "round"))


def _x_value(series: str, goods: int, bidders: int) -> int:
    if series == "bidders":
        return bidders
    return goods


def load_scalability_results(
    input_dir: str | Path,
    *,
    arm_filter: str | None = None,
) -> tuple[list[ScalabilityCase], list[dict[str, str]]]:
    """Load completed cases and describe incomplete or invalid case folders."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    case_dirs = sorted(
        path
        for path in root.rglob("*")
        if path.is_dir()
        and CASE_RE.fullmatch(path.name)
        and (
            (path / "curated_run_summary.csv").exists()
            or (path / "calls.jsonl").exists()
        )
    )
    completed: list[ScalabilityCase] = []
    incomplete: list[dict[str, str]] = []

    for case_dir in case_dirs:
        match = CASE_RE.fullmatch(case_dir.name)
        assert match is not None
        series = match.group("series")
        goods = int(match.group("goods"))
        bidders = int(match.group("bidders"))
        seed = _seed_label(case_dir, root)
        summary_path = case_dir / "curated_run_summary.csv"

        if not summary_path.exists():
            incomplete.append(
                {
                    "seed": seed,
                    "case": case_dir.name,
                    "path": str(case_dir),
                    "reason": "missing curated_run_summary.csv",
                }
            )
            continue

        try:
            summary_rows = _read_csv(summary_path)
            arm_row = _choose_arm(summary_rows, arm_filter)
        except (OSError, ValueError) as exc:
            incomplete.append(
                {
                    "seed": seed,
                    "case": case_dir.name,
                    "path": str(case_dir),
                    "reason": str(exc),
                }
            )
            continue

        shared_row = next(
            (
                row
                for row in summary_rows
                if row.get("arm", "").lower().startswith("shared initial")
            ),
            {},
        )
        result_row = _find_result_row(case_dir, arm_row.get("arm", ""))
        full_info_revenue = (
            _number(result_row, "full_info_revenue")
            if result_row is not None
            else math.nan
        )
        revenue = _number(arm_row, "revenue")
        if not math.isfinite(revenue) and result_row is not None:
            revenue = _number(result_row, "proxy_revenue")
        revenue_difference = (
            revenue - full_info_revenue
            if math.isfinite(revenue) and math.isfinite(full_info_revenue)
            else math.nan
        )
        revenue_difference_pct = (
            100.0 * revenue_difference / full_info_revenue
            if math.isfinite(revenue_difference) and full_info_revenue != 0
            else math.nan
        )

        efficiency = _number(arm_row, "efficiency")
        true_welfare = _number(arm_row, "true_welfare")
        full_info_welfare = _number(arm_row, "full_info_welfare")
        vq = _integer(arm_row, "vq")
        dq = _integer(arm_row, "dq")
        nl = _integer(arm_row, "nl")
        tokens_in = (
            _integer(arm_row, "tok_in")
            + _integer(shared_row, "person_tok_in")
        )
        tokens_out = (
            _integer(arm_row, "tok_out")
            + _integer(shared_row, "person_tok_out")
        )
        shared_proxy_tokens_in = _integer(shared_row, "proxy_tok_in")
        shared_proxy_tokens_out = _integer(shared_row, "proxy_tok_out")
        # Backward compatibility for summaries written before role-specific
        # frozen-replay accounting was added.
        if "proxy_tok_in" not in shared_row:
            shared_proxy_tokens_in = _integer(shared_row, "tok_in")
        if "proxy_tok_out" not in shared_row:
            shared_proxy_tokens_out = _integer(shared_row, "tok_out")
        generated, sent, truncated = _candidate_totals(case_dir)
        initial_row = _initial_trajectory_row(case_dir)
        initial_efficiency = (
            _number(initial_row, "global_efficiency")
            if initial_row is not None
            else math.nan
        )
        revenue_loss = (
            (full_info_revenue - revenue) / full_info_revenue
            if math.isfinite(revenue)
            and math.isfinite(full_info_revenue)
            and full_info_revenue != 0
            else math.nan
        )
        revenue_absolute_percentage_error = (
            _number(result_row, "revenue_absolute_percentage_error")
            if result_row is not None
            else math.nan
        )
        payment_error_over_optimum_welfare = (
            _number(result_row, "payment_error_over_optimum_welfare")
            if result_row is not None
            else math.nan
        )
        initial_revenue_loss = (
            _number(initial_row, "revenue_loss")
            if initial_row is not None
            else math.nan
        )

        values: dict[str, str | int | float] = {
            "seed": seed,
            "case": case_dir.name,
            "series": series,
            "x_value": _x_value(series, goods, bidders),
            "num_goods": goods,
            "num_bidders": bidders,
            "scenario": arm_row.get("scenario", ""),
            "arm": arm_row.get("arm", ""),
            "efficiency": efficiency,
            "efficiency_pct": efficiency * 100.0,
            "initial_efficiency": initial_efficiency,
            "efficiency_gain_from_initial_pct": (
                100.0 * (efficiency - initial_efficiency)
                if math.isfinite(efficiency)
                and math.isfinite(initial_efficiency)
                else math.nan
            ),
            "true_welfare": true_welfare,
            "full_info_welfare": full_info_welfare,
            "welfare_loss": full_info_welfare - true_welfare,
            "revenue": revenue,
            "full_info_revenue": full_info_revenue,
            "revenue_difference": revenue_difference,
            "revenue_difference_pct": revenue_difference_pct,
            "revenue_loss": revenue_loss,
            "revenue_absolute_percentage_error": (
                revenue_absolute_percentage_error
            ),
            "payment_error_over_optimum_welfare": (
                payment_error_over_optimum_welfare
            ),
            "initial_revenue_loss": initial_revenue_loss,
            "revenue_loss_improvement_from_initial": (
                initial_revenue_loss - revenue_loss
                if math.isfinite(initial_revenue_loss)
                and math.isfinite(revenue_loss)
                else math.nan
            ),
            "surplus": _number(arm_row, "surplus"),
            "value_queries": vq,
            "demand_queries": dq,
            "nl_queries": nl,
            "person_queries": vq + dq + nl,
            "person_tokens_in": tokens_in,
            "person_tokens_out": tokens_out,
            "person_tokens_total": tokens_in + tokens_out,
            "shared_proxy_tokens_in": shared_proxy_tokens_in,
            "shared_proxy_tokens_out": shared_proxy_tokens_out,
            "candidate_bundles_generated": generated,
            "candidate_bundles_sent_to_pv": sent,
            "candidate_bundles_truncated": truncated,
        }
        for field_name in _CALIBRATION_PASSTHROUGH:
            # Older summaries predate these columns; blank rather than fail.
            values[field_name] = arm_row.get(
                field_name, shared_row.get(field_name, "")
            )
        completed.append(ScalabilityCase(values))

    completed.sort(
        key=lambda case: (
            str(case.values["series"]),
            int(case.values["x_value"]),
            str(case.values["seed"]),
        )
    )
    return completed, incomplete


def write_rows(
    path: str | Path,
    fields: Sequence[str],
    rows: Iterable[dict[str, object]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_scalability_tables(
    output_dir: str | Path,
    cases: Sequence[ScalabilityCase],
    incomplete: Sequence[dict[str, str]],
) -> tuple[Path, Path]:
    """Write the tidy metrics table and incomplete-case report."""
    output = Path(output_dir)
    results_path = output / "scalability_metrics.csv"
    incomplete_path = output / "incomplete_cases.csv"
    write_rows(results_path, SCALABILITY_FIELDS, (case.values for case in cases))
    write_rows(incomplete_path, INCOMPLETE_FIELDS, incomplete)
    return results_path, incomplete_path


def _aggregate_metric(
    cases: Sequence[ScalabilityCase],
    metric: str,
) -> dict[str, list[tuple[int, float, float, float, int]]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for case in cases:
        series = str(case.values["series"])
        if series == "anchor":
            continue
        value = float(case.values[metric])
        if math.isfinite(value):
            key = (series, int(case.values["x_value"]))
            grouped.setdefault(key, []).append(value)

    # An anchor is the common point for all three scaling series.
    for case in cases:
        if case.values["series"] != "anchor":
            continue
        value = float(case.values[metric])
        if not math.isfinite(value):
            continue
        goods = int(case.values["num_goods"])
        bidders = int(case.values["num_bidders"])
        grouped.setdefault(("goods", goods), []).append(value)
        grouped.setdefault(("bidders", bidders), []).append(value)
        if goods == bidders:
            grouped.setdefault(("joint", goods), []).append(value)

    result: dict[str, list[tuple[int, float, float, float, int]]] = {}
    for (series, x_value), values in grouped.items():
        result.setdefault(series, []).append(
            (x_value, mean(values), min(values), max(values), len(values))
        )
    for points in result.values():
        points.sort(key=lambda point: point[0])
    return result


PLOT_SPECS = [
    ("person_queries", "Person-side queries", "Queries"),
    ("efficiency_pct", "Allocative efficiency", "Efficiency (%)"),
    (
        "efficiency_gain_from_initial_pct",
        "Efficiency gain from provisional allocation",
        "Percentage points",
    ),
    (
        "revenue_difference_pct",
        "Proxy versus full-information total revenue",
        "Revenue difference (%)",
    ),
    ("revenue_loss", "VCG revenue loss", "Fraction of oracle revenue"),
    (
        "payment_error_over_optimum_welfare",
        "Bidder-level VCG payment error",
        "Absolute payment error / optimal welfare",
    ),
    ("true_welfare", "True welfare", "Value"),
    ("surplus", "True bidder surplus", "Value"),
    ("person_tokens_total", "Person-side LLM token usage", "Tokens"),
    (
        "candidate_bundles_sent_to_pv",
        "Candidate bundles sent for provisional valuation",
        "Bundles",
    ),
]


def plot_scalability_results(
    output_dir: str | Path,
    cases: Sequence[ScalabilityCase],
    *,
    image_format: str = "png",
) -> list[Path]:
    """Generate one plot per canonical scalability metric."""
    output = Path(output_dir)
    # Keep Matplotlib and fontconfig caches with the analysis artifacts.  This
    # works in restricted/headless environments and avoids writing to $HOME.
    cache_dir = output / ".plot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib; install the project dependencies "
            "and rerun this command."
        ) from exc

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    anchor = next(
        (case for case in cases if case.values["series"] == "anchor"),
        None,
    )
    fixed_size = (
        int(anchor.values["num_goods"])
        if anchor is not None
        else 8
    )
    styles = {
        "goods": (f"Goods ({fixed_size} bidders)", "o"),
        "bidders": (f"Bidders ({fixed_size} goods)", "s"),
        "joint": ("Goods and bidders", "^"),
    }
    written: list[Path] = []

    for metric, title, ylabel in PLOT_SPECS:
        aggregated = _aggregate_metric(cases, metric)
        if not aggregated:
            continue
        figure, axis = plt.subplots(figsize=(8.2, 5.2))
        for series in ("goods", "bidders", "joint"):
            points = aggregated.get(series, [])
            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            lows = [point[2] for point in points]
            highs = [point[3] for point in points]
            label, marker = styles[series]
            axis.plot(xs, ys, marker=marker, linewidth=2, label=label)
            if any(point[4] > 1 for point in points):
                axis.fill_between(xs, lows, highs, alpha=0.14)

        axis.set_title(title)
        axis.set_xlabel("Number of goods or bidders")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend()
        if metric == "efficiency_pct":
            axis.set_ylim(0, 105)
        if metric == "revenue_difference_pct":
            axis.axhline(0, color="black", linewidth=1, alpha=0.55)
        figure.tight_layout()
        path = plot_dir / f"{metric}.{image_format}"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        written.append(path)
    return written
