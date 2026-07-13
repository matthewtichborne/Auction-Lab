"""Consolidated per-run analysis and cross-run comparison for LLM experiments.

Output filenames and column shapes remain stable so existing experiment
artifacts can be analyzed alongside newer hosted-model runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from auctionlab.experiments.llm_allocation_diagnostics import (
    compute_allocation_loss_records,
    group_allocation_loss_records,
    parse_allocation,
    write_allocation_loss_aggregate_csv,
    write_allocation_loss_records_csv,
)
from auctionlab.experiments.llm_analysis_io import (
    load_csv_by_key,
    read_curated_result_rows,
    read_proxy_result_rows,
    sort_result_rows,
    write_csv_rows,
)
from auctionlab.experiments.llm_value_diagnostics import (
    compute_value_error_records,
    group_value_error_records,
    parse_reported_bids,
    write_value_error_aggregate_csv,
    write_value_error_records_csv,
)
from auctionlab.instances.nl_scenarios import (
    NaturalLanguageAuctionScenario,
    curated_natural_language_scenarios,
)


SUMMARY_FIELDS = [
    "scenario",
    "seed_type",
    "mechanism",
    "top_k",
    "efficiency",
    "true_welfare",
    "reported_welfare",
    "full_info_welfare",
    "query_count",
    "rounds",
    "allocation_match",
    "welfare_match",
    "inferred_bids",
]

AGGREGATE_FIELDS = [
    "seed_type",
    "mechanism",
    "top_k",
    "n",
    "avg_efficiency",
    "avg_true_welfare",
    "avg_reported_welfare",
    "avg_full_info_welfare",
    "avg_query_count",
    "avg_rounds",
    "allocation_match_rate",
    "welfare_match_rate",
]

COMPARISON_FIELDS = [
    "run_label",
    "scenario",
    "seed_type",
    "mechanism",
    "top_k",
    "efficiency",
    "true_welfare",
    "reported_welfare",
    "full_info_welfare",
    "query_count",
    "rounds",
    "allocation_match",
    "welfare_match",
    "reported_welfare_gap",
    "reported_welfare_ratio",
    "value_mae",
    "value_rmse",
    "value_mean_signed_error",
    "value_overreport_rate",
    "value_underreport_rate",
    "value_max_absolute_error",
    "value_mean_absolute_relative_error",
    "changed_bidder_count",
    "changed_bidder_rate",
    "welfare_loss",
    "positive_delta_count",
    "negative_delta_count",
    "zero_delta_count",
]

VALUE_DIAGNOSTIC_FIELDS = {
    "value_mae": "mae",
    "value_rmse": "rmse",
    "value_mean_signed_error": "mean_signed_error",
    "value_overreport_rate": "overreport_rate",
    "value_underreport_rate": "underreport_rate",
    "value_max_absolute_error": "max_absolute_error",
    "value_mean_absolute_relative_error": "mean_absolute_relative_error",
}

ALLOCATION_DIAGNOSTIC_FIELDS = {
    "changed_bidder_count": "changed_bidder_count",
    "changed_bidder_rate": "changed_bidder_rate",
    "welfare_loss": "welfare_loss",
    "positive_delta_count": "positive_delta_count",
    "negative_delta_count": "negative_delta_count",
    "zero_delta_count": "zero_delta_count",
}


def infer_seed_type(instance_name: str) -> str:
    return "implicit" if instance_name.endswith("_implicit") else "explicit"


def _result_row_to_summary(
    row: dict[str, str],
    *,
    mechanism: str,
    top_k: str,
    true_welfare_field: str,
    reported_welfare_field: str,
    reported_bids_field: str,
    query_count_field: str | None = None,
    rounds_field: str | None = None,
) -> dict[str, str]:
    """Normalize one result-CSV row into the common summary schema."""
    scenario = row["instance_name"]
    return {
        "scenario": scenario,
        "seed_type": row.get("seed_type") or infer_seed_type(scenario),
        "mechanism": mechanism,
        "top_k": top_k,
        "efficiency": row["efficiency"],
        "true_welfare": row[true_welfare_field],
        "reported_welfare": row[reported_welfare_field],
        "full_info_welfare": row["full_info_welfare"],
        "query_count": row[query_count_field] if query_count_field else "",
        "rounds": row[rounds_field] if rounds_field else "",
        "allocation_match": row["allocation_match"],
        "welfare_match": row["welfare_match"],
        "inferred_bids": row[reported_bids_field],
    }


def sealed_row_to_summary(row: dict[str, str]) -> dict[str, str]:
    return _result_row_to_summary(
        row,
        mechanism="sealed_llm_proxy_vcg",
        top_k="",
        true_welfare_field="llm_proxy_true_welfare",
        reported_welfare_field="llm_proxy_reported_welfare",
        reported_bids_field="llm_proxy_reported_bids",
        query_count_field="llm_proxy_query_count",
    )


def clock_row_to_summary(
    row: dict[str, str],
    top_k: int,
) -> dict[str, str]:
    return _result_row_to_summary(
        row,
        mechanism="clock_llm_proxy_vcg",
        top_k=str(top_k),
        true_welfare_field="clock_llm_true_welfare",
        reported_welfare_field="clock_llm_reported_welfare",
        reported_bids_field="clock_llm_reported_bids",
        query_count_field="clock_llm_query_count",
        rounds_field="clock_rounds",
    )


def proxy_sealed_row_to_summary(row: dict[str, str]) -> dict[str, str]:
    """Normalize a ``curated_sealed_proxy_elicited.csv`` row.

    Uses ``final_bids`` (the post-refinement bid) as the reported bids, since
    that is what the proxy-mediated mechanism actually submitted.
    """
    return _result_row_to_summary(
        row,
        mechanism=row["mechanism"],
        top_k="",
        true_welfare_field="proxy_true_welfare",
        reported_welfare_field="proxy_reported_welfare",
        reported_bids_field="final_bids",
    )


def proxy_clock_row_to_summary(
    row: dict[str, str],
    top_k: int,
) -> dict[str, str]:
    """Normalize a ``curated_clock_proxy_elicited_top_*.csv`` row."""
    return _result_row_to_summary(
        row,
        mechanism=row["mechanism"],
        top_k=row.get("top_k") or str(top_k),
        true_welfare_field="proxy_true_welfare",
        reported_welfare_field="proxy_reported_welfare",
        reported_bids_field="final_bids",
        rounds_field="rounds",
    )


def build_summary_rows(
    sealed_rows: list[dict[str, str]],
    clock_rows_by_top_k: dict[int, list[dict[str, str]]],
    proxy_sealed_rows: list[dict[str, str]] = (),
    proxy_clock_rows_by_top_k: dict[int, list[dict[str, str]]] | None = None,
) -> list[dict[str, str]]:
    """Normalize static and proxy-mediated result CSV rows into one schema."""
    proxy_clock_rows_by_top_k = proxy_clock_rows_by_top_k or {}
    rows = [sealed_row_to_summary(row) for row in sealed_rows]
    for top_k, clock_rows in clock_rows_by_top_k.items():
        rows.extend(clock_row_to_summary(row, top_k) for row in clock_rows)
    rows.extend(
        proxy_sealed_row_to_summary(row) for row in proxy_sealed_rows
    )
    for top_k, proxy_clock_rows in proxy_clock_rows_by_top_k.items():
        rows.extend(
            proxy_clock_row_to_summary(row, top_k)
            for row in proxy_clock_rows
        )
    return sort_result_rows(rows)


def build_aggregate_rows(
    summary_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Aggregate normalized results by seed type, mechanism, and top-k."""
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in summary_rows:
        key = (row["seed_type"], row["mechanism"], row["top_k"])
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    for (seed_type, mechanism, top_k), rows in grouped.items():
        n = len(rows)
        rounds = [float(row["rounds"]) for row in rows if row["rounds"]]
        query_counts = [
            float(row["query_count"]) for row in rows if row["query_count"]
        ]

        def average(field: str) -> str:
            return str(sum(float(row[field]) for row in rows) / n)

        aggregate_rows.append(
            {
                "seed_type": seed_type,
                "mechanism": mechanism,
                "top_k": top_k,
                "n": str(n),
                "avg_efficiency": average("efficiency"),
                "avg_true_welfare": average("true_welfare"),
                "avg_reported_welfare": average("reported_welfare"),
                "avg_full_info_welfare": average("full_info_welfare"),
                "avg_query_count": (
                    str(sum(query_counts) / len(query_counts))
                    if query_counts
                    else ""
                ),
                "avg_rounds": (
                    str(sum(rounds) / len(rounds)) if rounds else ""
                ),
                "allocation_match_rate": str(
                    sum(row["allocation_match"] == "True" for row in rows) / n
                ),
                "welfare_match_rate": str(
                    sum(row["welfare_match"] == "True" for row in rows) / n
                ),
            }
        )
    return sort_result_rows(aggregate_rows)


def write_summary_outputs(
    sealed_rows: list[dict[str, str]],
    clock_rows_by_top_k: dict[int, list[dict[str, str]]],
    summary_path: str | Path,
    aggregate_path: str | Path,
    proxy_sealed_rows: list[dict[str, str]] = (),
    proxy_clock_rows_by_top_k: dict[int, list[dict[str, str]]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summary_rows = build_summary_rows(
        sealed_rows,
        clock_rows_by_top_k,
        proxy_sealed_rows,
        proxy_clock_rows_by_top_k,
    )
    aggregate_rows = build_aggregate_rows(summary_rows)
    write_csv_rows(summary_path, SUMMARY_FIELDS, summary_rows)
    write_csv_rows(aggregate_path, AGGREGATE_FIELDS, aggregate_rows)
    return summary_rows, aggregate_rows


def _scenario_map(
    scenarios: Iterable[NaturalLanguageAuctionScenario] | None = None,
) -> dict[str, NaturalLanguageAuctionScenario]:
    return {
        scenario.name: scenario
        for scenario in (
            scenarios
            if scenarios is not None
            else curated_natural_language_scenarios()
        )
    }


def _scenario_for_row(
    row: dict[str, str],
    scenarios: dict[str, NaturalLanguageAuctionScenario],
) -> NaturalLanguageAuctionScenario:
    scenario_name = row["instance_name"]
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown curated scenario: {scenario_name}")
    return scenarios[scenario_name]


def _value_error_records_for_row(
    row: dict[str, str],
    scenario_by_name: dict[str, NaturalLanguageAuctionScenario],
    *,
    mechanism: str,
    top_k: str,
    reported_bids_field: str,
) -> list:
    scenario = _scenario_for_row(row, scenario_by_name)
    return compute_value_error_records(
        scenario=scenario.name,
        seed_type=row.get("seed_type") or scenario.seed_type,
        mechanism=mechanism,
        top_k=top_k,
        reported_bids=parse_reported_bids(row[reported_bids_field]),
        instance=scenario.instance,
    )


def analyze_value_errors(
    sealed_rows: list[dict[str, str]],
    clock_rows_by_top_k: dict[int, list[dict[str, str]]],
    output_dir: str | Path,
    scenarios: Iterable[NaturalLanguageAuctionScenario] | None = None,
    proxy_sealed_rows: list[dict[str, str]] = (),
    proxy_clock_rows_by_top_k: dict[int, list[dict[str, str]]] | None = None,
) -> int:
    """Write bundle-level LLM value-estimation diagnostics for one run.

    Proxy-mediated rows are diagnosed against ``final_bids`` (the
    post-refinement bid actually submitted), not the original static
    inferred bid.
    """
    proxy_clock_rows_by_top_k = proxy_clock_rows_by_top_k or {}
    scenario_by_name = _scenario_map(scenarios)
    records = []
    for row in sealed_rows:
        records.extend(
            _value_error_records_for_row(
                row,
                scenario_by_name,
                mechanism="sealed_llm_proxy_vcg",
                top_k="",
                reported_bids_field="llm_proxy_reported_bids",
            )
        )
    for top_k, rows in clock_rows_by_top_k.items():
        for row in rows:
            records.extend(
                _value_error_records_for_row(
                    row,
                    scenario_by_name,
                    mechanism="clock_llm_proxy_vcg",
                    top_k=str(top_k),
                    reported_bids_field="clock_llm_reported_bids",
                )
            )
    for row in proxy_sealed_rows:
        records.extend(
            _value_error_records_for_row(
                row,
                scenario_by_name,
                mechanism=row["mechanism"],
                top_k="",
                reported_bids_field="final_bids",
            )
        )
    for top_k, rows in proxy_clock_rows_by_top_k.items():
        for row in rows:
            records.extend(
                _value_error_records_for_row(
                    row,
                    scenario_by_name,
                    mechanism=row["mechanism"],
                    top_k=row.get("top_k") or str(top_k),
                    reported_bids_field="final_bids",
                )
            )

    output_path = Path(output_dir)
    write_value_error_records_csv(
        records, output_path / "llm_value_errors.csv"
    )
    write_value_error_aggregate_csv(
        group_value_error_records(records),
        output_path / "llm_value_error_aggregate.csv",
    )
    return len(records)


def _allocation_loss_records_for_row(
    row: dict[str, str],
    scenario_by_name: dict[str, NaturalLanguageAuctionScenario],
    *,
    mechanism: str,
    top_k: str,
    allocation_field: str,
) -> list:
    scenario = _scenario_for_row(row, scenario_by_name)
    return compute_allocation_loss_records(
        scenario=scenario.name,
        seed_type=row.get("seed_type") or scenario.seed_type,
        mechanism=mechanism,
        top_k=top_k,
        instance=scenario.instance,
        full_info_allocation=parse_allocation(row["full_info_allocation"]),
        llm_allocation=parse_allocation(row[allocation_field]),
    )


def analyze_allocation_losses(
    sealed_rows: list[dict[str, str]],
    clock_rows_by_top_k: dict[int, list[dict[str, str]]],
    output_dir: str | Path,
    scenarios: Iterable[NaturalLanguageAuctionScenario] | None = None,
    proxy_sealed_rows: list[dict[str, str]] = (),
    proxy_clock_rows_by_top_k: dict[int, list[dict[str, str]]] | None = None,
) -> int:
    """Write bidder-level allocation changes and true-welfare losses.

    Proxy-mediated rows are diagnosed against the allocation produced by the
    proxy-mediated mechanism itself (``proxy_allocation``).
    """
    proxy_clock_rows_by_top_k = proxy_clock_rows_by_top_k or {}
    scenario_by_name = _scenario_map(scenarios)
    records = []
    for row in sealed_rows:
        records.extend(
            _allocation_loss_records_for_row(
                row,
                scenario_by_name,
                mechanism="sealed_llm_proxy_vcg",
                top_k="",
                allocation_field="llm_proxy_allocation",
            )
        )
    for top_k, rows in clock_rows_by_top_k.items():
        for row in rows:
            records.extend(
                _allocation_loss_records_for_row(
                    row,
                    scenario_by_name,
                    mechanism="clock_llm_proxy_vcg",
                    top_k=str(top_k),
                    allocation_field="clock_llm_allocation",
                )
            )
    for row in proxy_sealed_rows:
        records.extend(
            _allocation_loss_records_for_row(
                row,
                scenario_by_name,
                mechanism=row["mechanism"],
                top_k="",
                allocation_field="proxy_allocation",
            )
        )
    for top_k, rows in proxy_clock_rows_by_top_k.items():
        for row in rows:
            records.extend(
                _allocation_loss_records_for_row(
                    row,
                    scenario_by_name,
                    mechanism=row["mechanism"],
                    top_k=row.get("top_k") or str(top_k),
                    allocation_field="proxy_allocation",
                )
            )

    output_path = Path(output_dir)
    write_allocation_loss_records_csv(
        records, output_path / "llm_allocation_losses.csv"
    )
    write_allocation_loss_aggregate_csv(
        group_allocation_loss_records(records),
        output_path / "llm_allocation_loss_aggregate.csv",
    )
    return len(records)


def analyze_llm_run(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    skip_value_errors: bool = False,
    skip_allocation_losses: bool = False,
    scenarios: Iterable[NaturalLanguageAuctionScenario] | None = None,
) -> dict[str, int]:
    """Generate all canonical per-run CSV outputs.

    Historical filenames are intentionally preserved for downstream notebooks
    and previously generated run directories.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir is not None else input_path
    sealed_rows, clock_rows = read_curated_result_rows(input_path)
    proxy_sealed_rows, proxy_clock_rows = read_proxy_result_rows(input_path)
    summary_rows, aggregate_rows = write_summary_outputs(
        sealed_rows,
        clock_rows,
        output_path / "curated_summary.csv",
        output_path / "curated_summary_aggregate.csv",
        proxy_sealed_rows,
        proxy_clock_rows,
    )
    counts = {
        "static_sealed_rows": len(sealed_rows),
        "static_clock_rows": sum(len(rows) for rows in clock_rows.values()),
        "proxy_sealed_rows": len(proxy_sealed_rows),
        "proxy_clock_rows": sum(
            len(rows) for rows in proxy_clock_rows.values()
        ),
        "summary_rows": len(summary_rows),
        "aggregate_rows": len(aggregate_rows),
    }
    if not skip_value_errors:
        counts["value_error_records"] = analyze_value_errors(
            sealed_rows,
            clock_rows,
            output_path,
            scenarios,
            proxy_sealed_rows,
            proxy_clock_rows,
        )
    if not skip_allocation_losses:
        counts["allocation_loss_records"] = analyze_allocation_losses(
            sealed_rows,
            clock_rows,
            output_path,
            scenarios,
            proxy_sealed_rows,
            proxy_clock_rows,
        )
    return counts


def _lookup_diagnostic(
    rows_by_key: dict[tuple[str, str, str, str], dict[str, str]],
    key: tuple[str, str, str, str],
) -> dict[str, str] | None:
    if key in rows_by_key:
        return rows_by_key[key]
    scenario, _seed_type, mechanism, top_k = key
    matches = [
        row
        for candidate_key, row in rows_by_key.items()
        if (
            candidate_key[0] == scenario
            and candidate_key[2] == mechanism
            and candidate_key[3] == top_k
        )
    ]
    return matches[0] if len(matches) == 1 else None


def join_run_rows(
    run_label: str,
    summary_rows_by_key: dict[
        tuple[str, str, str, str], dict[str, str]
    ],
    value_rows_by_key: dict[
        tuple[str, str, str, str], dict[str, str]
    ] | None = None,
    allocation_rows_by_key: dict[
        tuple[str, str, str, str], dict[str, str]
    ] | None = None,
) -> list[dict[str, str]]:
    value_rows_by_key = value_rows_by_key or {}
    allocation_rows_by_key = allocation_rows_by_key or {}
    joined_rows = []
    for key, summary in summary_rows_by_key.items():
        true_welfare = float(summary["true_welfare"])
        reported_welfare = float(summary["reported_welfare"])
        value_diagnostics = _lookup_diagnostic(value_rows_by_key, key)
        allocation_diagnostics = _lookup_diagnostic(
            allocation_rows_by_key, key
        )
        row = {
            "run_label": run_label,
            **{field: summary[field] for field in SUMMARY_FIELDS[:-1]},
            "reported_welfare_gap": str(true_welfare - reported_welfare),
            "reported_welfare_ratio": (
                str(reported_welfare / true_welfare)
                if true_welfare > 0.0
                else ""
            ),
        }
        row.update(
            {
                output_field: (
                    value_diagnostics.get(source_field, "")
                    if value_diagnostics is not None
                    else ""
                )
                for output_field, source_field
                in VALUE_DIAGNOSTIC_FIELDS.items()
            }
        )
        row.update(
            {
                output_field: (
                    allocation_diagnostics.get(source_field, "")
                    if allocation_diagnostics is not None
                    else ""
                )
                for output_field, source_field
                in ALLOCATION_DIAGNOSTIC_FIELDS.items()
            }
        )
        joined_rows.append(row)
    return joined_rows


def compare_llm_runs(
    runs: list[tuple[str, Path]],
) -> list[dict[str, str]]:
    """Join analyzed runs, leaving unavailable optional diagnostics blank."""
    rows = []
    for run_label, run_dir in runs:
        summary_path = run_dir / "curated_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing required curated summary: {summary_path}"
            )
        value_path = run_dir / "llm_value_error_aggregate.csv"
        allocation_path = run_dir / "llm_allocation_loss_aggregate.csv"
        rows.extend(
            join_run_rows(
                run_label,
                load_csv_by_key(summary_path),
                load_csv_by_key(value_path) if value_path.exists() else None,
                (
                    load_csv_by_key(allocation_path)
                    if allocation_path.exists()
                    else None
                ),
            )
        )
    return sort_result_rows(rows, suffix_fields=("run_label",))
