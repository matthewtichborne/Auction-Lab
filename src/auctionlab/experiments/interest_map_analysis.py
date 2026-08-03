"""Offline analysis of interest maps stored in frozen elicitation packs."""

from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from auctionlab.llm.frozen_elicitation import (
    FrozenElicitationPack,
    load_frozen_elicitation_pack,
)


CASE_RE = re.compile(
    r"^(?P<series>goods|bidders|joint|anchor)_"
    r"(?P<goods>\d+)x(?P<bidders>\d+)$"
)
MASTER_RE = re.compile(
    r"^goods_(?P<goods>\d+)_master_"
    r"(?P<goods_again>\d+)x(?P<bidders>\d+)\.json$"
)

ACCURACY_FIELDS = (
    "item_precision",
    "item_recall",
    "item_f1",
    "group_item_set_precision",
    "group_item_set_recall",
    "mode_accuracy_on_matched_groups",
    "exact_group_and_mode_recall",
    "choose_one_precision",
    "choose_one_recall",
    "complement_group_precision",
    "complement_group_recall",
    "candidate_set_precision",
    "candidate_set_recall",
)

BIDDER_FIELDS = (
    "pack_path",
    "seed",
    "case",
    "series",
    "x_value",
    "num_goods",
    "num_bidders",
    "bidder_id",
    "interested_item_count",
    "excluded_item_count",
    "full_powerset_count",
    "interested_item_powerset_count",
    "inferred_candidate_count",
    "oracle_candidate_count",
    "pv_candidate_count_sent",
    "pv_api_call_count",
    "exclusion_reduction_count",
    "substitute_reduction_count",
    "total_reduction_count",
    "exclusion_reduction_pct",
    "substitute_reduction_pct_of_full",
    "substitute_reduction_pct_of_interested_support",
    "total_reduction_pct",
    *ACCURACY_FIELDS,
    "missed_positive_item_count",
    "false_positive_item_count",
    "missed_oracle_candidate_count",
    "extra_candidate_count",
    "dangerous_false_exclusivity_count",
    "person_answer_final_passed",
    "person_answer_first_attempt_success",
    "person_answer_attempt_count",
    "person_answer_repair_count",
    "interest_map_first_attempt_success",
    "interest_map_final_success",
    "interest_map_attempt_count",
    "interest_map_retry_count",
    "interest_map_fallback_used",
)

CASE_FIELDS = (
    "pack_path",
    "seed",
    "case",
    "series",
    "x_value",
    "num_goods",
    "num_bidders",
    "full_powerset_count",
    "interested_item_powerset_count",
    "inferred_candidate_count",
    "oracle_candidate_count",
    "pv_candidate_count_sent",
    "pv_api_call_count",
    "exclusion_reduction_count",
    "substitute_reduction_count",
    "total_reduction_count",
    "exclusion_reduction_pct",
    "substitute_reduction_pct_of_full",
    "substitute_reduction_pct_of_interested_support",
    "total_reduction_pct",
    *tuple(f"mean_{field}" for field in ACCURACY_FIELDS),
    "total_missed_positive_items",
    "total_false_positive_items",
    "total_missed_oracle_candidates",
    "total_extra_candidates",
    "total_dangerous_false_exclusivity",
    "person_answer_final_pass_rate",
    "person_answer_first_attempt_success_rate",
    "mean_person_answer_repairs",
    "interest_map_first_attempt_success_rate",
    "interest_map_final_success_rate",
    "mean_interest_map_retries",
    "interest_map_fallback_rate",
)

SUMMARY_METRICS = (
    "full_powerset_count",
    "interested_item_powerset_count",
    "inferred_candidate_count",
    "oracle_candidate_count",
    "pv_candidate_count_sent",
    "pv_api_call_count",
    "exclusion_reduction_pct",
    "substitute_reduction_pct_of_full",
    "substitute_reduction_pct_of_interested_support",
    "total_reduction_pct",
    *tuple(f"mean_{field}" for field in ACCURACY_FIELDS),
    "total_missed_positive_items",
    "total_missed_oracle_candidates",
    "total_dangerous_false_exclusivity",
    "person_answer_first_attempt_success_rate",
    "mean_person_answer_repairs",
    "interest_map_first_attempt_success_rate",
    "mean_interest_map_retries",
)

SUMMARY_FIELDS = (
    "series",
    "x_value",
    "num_cases",
    "num_seeds",
    *tuple(
        f"{metric}_{suffix}"
        for metric in SUMMARY_METRICS
        for suffix in ("mean", "min", "max")
    ),
)


def _percentage(reduction: float, baseline: float) -> float:
    return 100.0 * reduction / baseline if baseline else 0.0


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean_present(values: Iterable[Any]) -> float | None:
    present = [
        value
        for raw in values
        if (value := _optional_float(raw)) is not None
    ]
    return mean(present) if present else None


def _seed_from_path(path: Path, pack: FrozenElicitationPack) -> str:
    for parent in path.parents:
        if parent.name.startswith("seed_"):
            return parent.name.removeprefix("seed_")
    return "" if pack.scenario_seed is None else str(pack.scenario_seed)


def _case_identity(
    path: Path,
    pack: FrozenElicitationPack,
) -> tuple[str, str, int]:
    match = CASE_RE.fullmatch(path.parent.name)
    if match:
        series = match.group("series")
        goods = int(match.group("goods"))
        bidders = int(match.group("bidders"))
        x_value = bidders if series == "bidders" else goods
        return path.parent.name, series, x_value
    master = MASTER_RE.fullmatch(path.name)
    if master:
        goods = int(master.group("goods"))
        return path.stem, "master", goods
    return (
        path.stem,
        "pack",
        len(pack.items),
    )


def discover_frozen_packs(
    input_dir: str | Path,
    *,
    include_masters: bool = False,
) -> list[Path]:
    """Find projected case packs, optionally including catalogue masters.

    Projected case packs are preferred because masters and projections contain
    overlapping bidder elicitation. Excluding masters by default prevents
    accidental double counting.
    """
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        return [root]

    projected = sorted(root.rglob("frozen_elicitation.json"))
    masters = sorted(root.glob("seed_*/masters/*.json"))
    if include_masters:
        return projected + masters
    return projected or masters


def _interest_map_call_stats(
    pack: FrozenElicitationPack,
    bidder_id: str,
) -> tuple[bool | None, bool | None, int, int]:
    calls = [
        call
        for call in pack.generation_calls
        if call.get("bidder_id") == bidder_id
        and call.get("prompt_type") == "proxy_interest_map"
    ]
    if not calls:
        return None, None, 0, 0
    calls.sort(key=lambda call: int(call.get("attempt") or 0))
    first_success = bool(calls[0].get("success"))
    final_success = any(bool(call.get("success")) for call in calls)
    attempts = len(calls)
    return first_success, final_success, attempts, max(0, attempts - 1)


def bidder_metrics(
    path: Path,
    pack: FrozenElicitationPack,
) -> list[dict[str, Any]]:
    """Return one support-reduction and reconstruction row per bidder."""
    case, series, x_value = _case_identity(path, pack)
    seed = _seed_from_path(path, pack)
    num_goods = len(pack.items)
    num_bidders = len(pack.bidder_ids)
    full_count = 2 ** num_goods - 1
    rows: list[dict[str, Any]] = []

    for bidder_id in pack.bidder_ids:
        entry = pack.bidders[bidder_id]
        if entry.interest_map is None:
            raise ValueError(
                f"{path}: bidder {bidder_id} has no inferred interest map"
            )
        interested_count = len(entry.interest_map.interested_items)
        excluded_count = len(entry.interest_map.excluded_items)
        interested_power = 2 ** interested_count - 1
        inferred_count = len(entry.candidate_bundles)
        accuracy = dict(entry.interest_map_accuracy or {})
        oracle_count = int(
            accuracy.get("oracle_candidate_count", inferred_count)
        )
        pv_sent = (
            entry.pv_candidate_stats.candidate_bundles_sent_to_pv
            if entry.pv_candidate_stats is not None
            else (
                len(entry.raw_pv_values)
                if entry.raw_pv_values is not None
                else 0
            )
        )
        pv_calls = (
            entry.pv_chunk_stats.pv_chunks
            if entry.pv_chunk_stats is not None
            else int(entry.raw_pv_values is not None)
        )
        exclusion_reduction = full_count - interested_power
        substitute_reduction = interested_power - inferred_count
        total_reduction = full_count - inferred_count

        history = entry.person_answer_verification_history or []
        final_verification = entry.person_answer_verification or {}
        final_passed = bool(final_verification.get("passed", False))
        first_passed = (
            bool(history[0].get("passed"))
            if history
            else (final_passed and entry.person_answer_attempt_count == 1)
        )
        im_first, im_final, im_attempts, im_retries = (
            _interest_map_call_stats(pack, bidder_id)
        )

        row: dict[str, Any] = {
            "pack_path": str(path),
            "seed": seed,
            "case": case,
            "series": series,
            "x_value": x_value,
            "num_goods": num_goods,
            "num_bidders": num_bidders,
            "bidder_id": bidder_id,
            "interested_item_count": interested_count,
            "excluded_item_count": excluded_count,
            "full_powerset_count": full_count,
            "interested_item_powerset_count": interested_power,
            "inferred_candidate_count": inferred_count,
            "oracle_candidate_count": oracle_count,
            "pv_candidate_count_sent": pv_sent,
            "pv_api_call_count": pv_calls,
            "exclusion_reduction_count": exclusion_reduction,
            "substitute_reduction_count": substitute_reduction,
            "total_reduction_count": total_reduction,
            "exclusion_reduction_pct": _percentage(
                exclusion_reduction, full_count
            ),
            "substitute_reduction_pct_of_full": _percentage(
                substitute_reduction, full_count
            ),
            "substitute_reduction_pct_of_interested_support": _percentage(
                substitute_reduction, interested_power
            ),
            "total_reduction_pct": _percentage(
                total_reduction, full_count
            ),
            "missed_positive_item_count": len(
                accuracy.get("missed_positive_items", [])
            ),
            "false_positive_item_count": len(
                accuracy.get("false_positive_items", [])
            ),
            "missed_oracle_candidate_count": int(
                accuracy.get("missed_oracle_candidate_count", 0)
            ),
            "extra_candidate_count": int(
                accuracy.get("extra_candidate_count", 0)
            ),
            "dangerous_false_exclusivity_count": int(
                accuracy.get("dangerous_false_exclusivity_count", 0)
            ),
            "person_answer_final_passed": final_passed,
            "person_answer_first_attempt_success": first_passed,
            "person_answer_attempt_count": entry.person_answer_attempt_count,
            "person_answer_repair_count": max(
                0, entry.person_answer_attempt_count - 1
            ),
            "interest_map_first_attempt_success": im_first,
            "interest_map_final_success": im_final,
            "interest_map_attempt_count": im_attempts,
            "interest_map_retry_count": im_retries,
            "interest_map_fallback_used": (
                entry.interest_map_fallback_used
            ),
        }
        row.update({
            field: accuracy.get(field)
            for field in ACCURACY_FIELDS
        })
        rows.append(row)
    return rows


def load_interest_map_results(
    input_dir: str | Path,
    *,
    include_masters: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Load all valid frozen packs and report unreadable packs separately."""
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for path in discover_frozen_packs(
        input_dir, include_masters=include_masters
    ):
        try:
            pack = load_frozen_elicitation_pack(path)
            rows.extend(bidder_metrics(path, pack))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            invalid.append({"path": str(path), "reason": str(exc)})
    rows.sort(
        key=lambda row: (
            str(row["series"]),
            int(row["x_value"]),
            str(row["seed"]),
            str(row["bidder_id"]),
        )
    )
    return rows, invalid


def aggregate_cases(
    bidder_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in bidder_rows:
        grouped[str(row["pack_path"])].append(row)

    result: list[dict[str, Any]] = []
    for pack_path, rows in grouped.items():
        first = rows[0]

        def total(field: str) -> float:
            return sum(float(row[field]) for row in rows)

        full = total("full_powerset_count")
        interested = total("interested_item_powerset_count")
        inferred = total("inferred_candidate_count")
        exclusion = full - interested
        substitute = interested - inferred
        row: dict[str, Any] = {
            "pack_path": pack_path,
            "seed": first["seed"],
            "case": first["case"],
            "series": first["series"],
            "x_value": first["x_value"],
            "num_goods": first["num_goods"],
            "num_bidders": len(rows),
            "full_powerset_count": int(full),
            "interested_item_powerset_count": int(interested),
            "inferred_candidate_count": int(inferred),
            "oracle_candidate_count": int(total("oracle_candidate_count")),
            "pv_candidate_count_sent": int(total("pv_candidate_count_sent")),
            "pv_api_call_count": int(total("pv_api_call_count")),
            "exclusion_reduction_count": int(exclusion),
            "substitute_reduction_count": int(substitute),
            "total_reduction_count": int(full - inferred),
            "exclusion_reduction_pct": _percentage(exclusion, full),
            "substitute_reduction_pct_of_full": _percentage(
                substitute, full
            ),
            "substitute_reduction_pct_of_interested_support": _percentage(
                substitute, interested
            ),
            "total_reduction_pct": _percentage(full - inferred, full),
            "total_missed_positive_items": int(
                total("missed_positive_item_count")
            ),
            "total_false_positive_items": int(
                total("false_positive_item_count")
            ),
            "total_missed_oracle_candidates": int(
                total("missed_oracle_candidate_count")
            ),
            "total_extra_candidates": int(
                total("extra_candidate_count")
            ),
            "total_dangerous_false_exclusivity": int(
                total("dangerous_false_exclusivity_count")
            ),
            "person_answer_final_pass_rate": _mean_present(
                entry["person_answer_final_passed"] for entry in rows
            ),
            "person_answer_first_attempt_success_rate": _mean_present(
                entry["person_answer_first_attempt_success"]
                for entry in rows
            ),
            "mean_person_answer_repairs": _mean_present(
                entry["person_answer_repair_count"] for entry in rows
            ),
            "interest_map_first_attempt_success_rate": _mean_present(
                entry["interest_map_first_attempt_success"]
                for entry in rows
            ),
            "interest_map_final_success_rate": _mean_present(
                entry["interest_map_final_success"] for entry in rows
            ),
            "mean_interest_map_retries": _mean_present(
                entry["interest_map_retry_count"] for entry in rows
            ),
            "interest_map_fallback_rate": _mean_present(
                entry["interest_map_fallback_used"] for entry in rows
            ),
        }
        row.update({
            f"mean_{field}": _mean_present(
                entry[field] for entry in rows
            )
            for field in ACCURACY_FIELDS
        })
        result.append(row)

    result.sort(
        key=lambda row: (
            str(row["series"]),
            int(row["x_value"]),
            str(row["seed"]),
        )
    )
    return result


def _expanded_series_rows(
    case_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for raw in case_rows:
        row = dict(raw)
        if row["series"] != "anchor":
            expanded.append(row)
            continue
        goods = int(row["num_goods"])
        bidders = int(row["num_bidders"])
        for series, x_value in (
            ("goods", goods),
            ("bidders", bidders),
            *((("joint", goods),) if goods == bidders else ()),
        ):
            clone = dict(row)
            clone["series"] = series
            clone["x_value"] = x_value
            expanded.append(clone)
    return expanded


def aggregate_summary(
    case_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in _expanded_series_rows(case_rows):
        grouped[(str(row["series"]), int(row["x_value"]))].append(row)

    result: list[dict[str, Any]] = []
    for (series, x_value), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "series": series,
            "x_value": x_value,
            "num_cases": len(rows),
            "num_seeds": len({str(row["seed"]) for row in rows}),
        }
        for metric in SUMMARY_METRICS:
            values = [
                value
                for row in rows
                if (value := _optional_float(row.get(metric))) is not None
            ]
            summary[f"{metric}_mean"] = mean(values) if values else None
            summary[f"{metric}_min"] = min(values) if values else None
            summary[f"{metric}_max"] = max(values) if values else None
        result.append(summary)
    return result


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_interest_map_tables(
    output_dir: str | Path,
    bidder_rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    invalid: Sequence[Mapping[str, str]],
) -> tuple[Path, Path, Path, Path]:
    output = Path(output_dir)
    bidder_path = output / "interest_map_bidder_metrics.csv"
    case_path = output / "interest_map_case_metrics.csv"
    summary_path = output / "interest_map_summary.csv"
    invalid_path = output / "invalid_packs.csv"
    _write_csv(bidder_path, BIDDER_FIELDS, bidder_rows)
    _write_csv(case_path, CASE_FIELDS, case_rows)
    _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    _write_csv(invalid_path, ("path", "reason"), invalid)
    return bidder_path, case_path, summary_path, invalid_path


def _plot_panels(
    plt: Any,
    case_rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[tuple[str, str]],
    title: str,
    ylabel: str,
    path: Path,
    percent: bool = False,
) -> None:
    expanded = _expanded_series_rows(case_rows)
    series_order = ("goods", "bidders", "joint")
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    colors = plt.get_cmap("tab10").colors
    for axis, series in zip(axes, series_order):
        series_rows = [row for row in expanded if row["series"] == series]
        for metric_idx, (metric, label) in enumerate(metrics):
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in series_rows:
                value = _optional_float(row.get(metric))
                if value is not None:
                    grouped[int(row["x_value"])].append(value)
            xs = sorted(grouped)
            if not xs:
                continue
            means = [mean(grouped[x]) for x in xs]
            lows = [min(grouped[x]) for x in xs]
            highs = [max(grouped[x]) for x in xs]
            color = colors[metric_idx % len(colors)]
            for x in xs:
                axis.scatter(
                    [x] * len(grouped[x]),
                    grouped[x],
                    color=color,
                    alpha=0.22,
                    s=20,
                )
            axis.plot(
                xs, means, marker="o", linewidth=2, color=color, label=label
            )
            if any(len(grouped[x]) > 1 for x in xs):
                axis.fill_between(xs, lows, highs, color=color, alpha=0.10)
        axis.set_title({
            "goods": "Vary goods",
            "bidders": "Vary bidders",
            "joint": "Vary both",
        }[series])
        axis.set_xlabel("Number of goods or bidders")
        axis.grid(True, alpha=0.25)
        if percent:
            axis.set_ylim(-2, 102)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for axis in axes[1:]:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                break
    if handles:
        figure.legend(
            handles, labels, loc="lower center", ncol=min(len(labels), 4)
        )
        figure.subplots_adjust(bottom=0.22)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0.08 if handles else 0, 1, 0.94))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_interest_map_results(
    output_dir: str | Path,
    case_rows: Sequence[Mapping[str, Any]],
    *,
    image_format: str = "png",
) -> list[Path]:
    """Write the canonical interest-map support and accuracy plots."""
    output = Path(output_dir)
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
    specs = (
        (
            "candidate_support_size",
            (
                ("full_powerset_count", "Full powerset"),
                (
                    "interested_item_powerset_count",
                    "Interested-item powerset",
                ),
                ("inferred_candidate_count", "Final inferred support"),
            ),
            "Candidate-support size",
            "Potential bundle valuations",
            False,
        ),
        (
            "total_reduction_pct",
            (("total_reduction_pct", "Total reduction"),),
            "Reduction relative to full powerset",
            "Reduction (%)",
            True,
        ),
        (
            "reduction_source",
            (
                ("exclusion_reduction_pct", "Excluded goods"),
                (
                    "substitute_reduction_pct_of_full",
                    "Choose-one substitutes",
                ),
            ),
            "Source of candidate-support reduction",
            "Percentage points of full powerset",
            True,
        ),
        (
            "item_reconstruction_accuracy",
            (
                ("mean_item_precision", "Item precision"),
                ("mean_item_recall", "Item recall"),
                ("mean_item_f1", "Item F1"),
            ),
            "Interest-item reconstruction accuracy",
            "Score (%)",
            True,
        ),
        (
            "structural_reconstruction_accuracy",
            (
                (
                    "mean_exact_group_and_mode_recall",
                    "Substitute group + mode recall",
                ),
                (
                    "mean_mode_accuracy_on_matched_groups",
                    "Mode accuracy on matched groups",
                ),
                (
                    "mean_complement_group_recall",
                    "Complement-group recall",
                ),
                ("mean_candidate_set_recall", "Candidate-set recall"),
            ),
            "Structural reconstruction accuracy",
            "Score (%)",
            True,
        ),
        (
            "error_risk_diagnostics",
            (
                (
                    "total_missed_positive_items",
                    "Missed positive items",
                ),
                (
                    "total_missed_oracle_candidates",
                    "Missed oracle candidates",
                ),
                (
                    "total_dangerous_false_exclusivity",
                    "Dangerous false exclusivity",
                ),
            ),
            "Interest-map error-risk diagnostics",
            "Errors per auction",
            False,
        ),
        (
            "elicitation_reliability",
            (
                (
                    "person_answer_first_attempt_success_rate",
                    "Person answer first-attempt pass",
                ),
                (
                    "interest_map_first_attempt_success_rate",
                    "Interest map first-attempt parse",
                ),
            ),
            "Initial elicitation reliability",
            "Success rate (%)",
            True,
        ),
    )

    # Accuracy and rates are stored as fractions; plot them as percentages.
    plot_rows = [dict(row) for row in case_rows]
    for row in plot_rows:
        for key in (
            *(f"mean_{field}" for field in ACCURACY_FIELDS),
            "person_answer_final_pass_rate",
            "person_answer_first_attempt_success_rate",
            "interest_map_first_attempt_success_rate",
            "interest_map_final_success_rate",
            "interest_map_fallback_rate",
        ):
            value = _optional_float(row.get(key))
            if value is not None:
                row[key] = 100.0 * value

    written: list[Path] = []
    for stem, metrics, title, ylabel, percent in specs:
        path = plot_dir / f"{stem}.{image_format}"
        _plot_panels(
            plt,
            plot_rows,
            metrics=metrics,
            title=title,
            ylabel=ylabel,
            path=path,
            percent=percent,
        )
        written.append(path)
    return written

