#!/usr/bin/env python3
"""Offline experiment-grid runner/aggregator for auction proxy experiments.

Reuses :mod:`auctionlab.experiments.grid_config` (:class:`ExperimentGrid`,
:class:`ExperimentArm`) to turn a JSON grid config into one
``examples/run_live_llm_curated_batch.py`` invocation per arm, and
aggregates the resulting per-run CSVs into a single results table plus
summary reports. See ``docs/parameter_tuning_methodology.md`` for the
methodological reasoning behind the grid's environment / candidate-policy /
valuation-calibration / sealed-treatment / clock-treatment / safety-limit
category split, and why development, final, and robustness grids are kept
separate.

Three modes:

- ``--dry-run``: prints the exact command and run directory for every arm.
  Validates the grid config and reports estimated run labels/output
  directories. Creates no directories, makes no subprocess or LLM calls.
- ``--aggregate-only``: reads existing completed run directories under
  ``--output-root`` and writes the aggregate CSVs/report. Makes no
  subprocess or LLM calls.
- normal run mode (default, no flag): dispatches each arm as a subprocess
  invocation of ``examples/run_live_llm_curated_batch.py``, then
  aggregates. This is the only mode that can make live LLM calls (via the
  invoked script) -- it is never exercised by this repository's test suite.

Usage::

    # Validate a grid and see exactly what would run.
    ./venv/bin/python scripts/run_proxy_parameter_grid.py \\
        --grid-file configs/auction_development_grid_example.json \\
        --output-root outputs/parameter_grid --dry-run

    # Aggregate whatever has already completed under --output-root.
    ./venv/bin/python scripts/run_proxy_parameter_grid.py \\
        --grid-file configs/auction_development_grid_example.json \\
        --output-root outputs/parameter_grid --aggregate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from auctionlab.experiments.export import write_csv_variable_rows
from auctionlab.experiments.grid_config import (
    ExperimentArm,
    ExperimentGrid,
    grid_methodology_warnings,
    load_experiment_grid,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CURATED_BATCH_SCRIPT = REPO_ROOT / "examples" / "run_live_llm_curated_batch.py"

RESULT_COLUMNS: list[str] = [
    "run_label", "config_name", "scenario_spec", "num_goods", "num_bidders",
    "scenario_seed", "arm", "top_k",
    "final_true_welfare", "full_info_welfare", "final_efficiency",
    "best_true_welfare", "best_efficiency",
    "reported_welfare", "revenue", "true_surplus",
    "value_queries", "demand_queries", "nl_queries",
    "estimated_tokens_in", "tokens_out",
    "clock_rounds", "termination_reason", "failure_classification",
    "fi_coverage", "supplementary_atoms_total",
    "candidate_bundles_generated", "candidate_bundles_sent_to_pv",
    "candidate_bundles_truncated", "candidate_truncation_reason",
    "max_candidate_bundles", "pv_max_tokens", "max_bundle_size",
    "sealed_feedback_rule", "sealed_elicitation_rounds",
    "clock_tie_threshold", "clock_margin_threshold", "price_step", "max_rounds",
    "per_bidder_refinement_query_limit", "global_refinement_query_safety_limit",
    "total_refinement_queries", "per_bidder_refinement_queries",
    "cap_binding_indicator", "safety_cap_hit",
]

_ARM_TOPK_RE = re.compile(r"k=(\d+)")


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def build_command(
    arm: ExperimentArm,
    run_dir: Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    """Build the argv for one ``examples/run_live_llm_curated_batch.py`` call.

    CLI-level ``provider``/``model``/``api_key``/``base_url`` overrides win
    over whatever the arm itself specifies -- they're grid-runner-wide
    connection settings, not per-arm experimental variables.
    """
    args = arm.to_cli_args()

    def _override(flag: str, value: str | None) -> None:
        if value is None:
            return
        if flag in args:
            args[args.index(flag) + 1] = str(value)
        else:
            args.extend([flag, str(value)])

    _override("--provider", provider)
    _override("--model", model)
    _override("--api-key", api_key)
    _override("--base-url", base_url)

    return [sys.executable, str(CURATED_BATCH_SCRIPT), *args, "--log-dir", str(run_dir)]


def _redact_command(command: list[str]) -> list[str]:
    """Replace the value following ``--api-key``, if present, for display.

    Only used for dry-run printing/reporting -- :func:`dispatch_run` always
    builds and passes the real, unredacted command to the subprocess so a
    live run still authenticates correctly.
    """
    redacted = list(command)
    if "--api-key" in redacted:
        idx = redacted.index("--api-key")
        if idx + 1 < len(redacted):
            redacted[idx + 1] = "<redacted>"
    return redacted


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def planned_run_dir_label(output_root: Path, grid_name: str, arm_name: str) -> Path:
    """A representative (not guaranteed-final) run directory path for dry-run
    reporting -- the real timestamp/uniqueness suffix is only assigned at
    dispatch time by :func:`allocate_run_dir`."""
    return output_root / grid_name / f"{arm_name}__<timestamp>"


def allocate_run_dir(output_root: Path, grid_name: str, arm_name: str) -> Path:
    """Create and return a fresh, never-before-existing run directory.

    Never silently overwrites: if the timestamped directory already exists
    (e.g. two arms dispatched within the same second, or a re-run), a
    numeric suffix is appended until an unused path is found.
    """
    base = output_root / grid_name
    ts = _timestamp()
    candidate = base / f"{arm_name}__{ts}"
    suffix = 0
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = base / f"{arm_name}__{ts}_{suffix}"


def write_run_manifest(
    run_dir: Path,
    *,
    grid: ExperimentGrid,
    arm: ExperimentArm,
    command: list[str],
) -> None:
    manifest = {
        "run_label": run_dir.name,
        "config_name": arm.name,
        "grid_name": grid.grid_name,
        "grid_type": grid.grid_type,
        "label": arm.label,
        "environment": arm.environment.model_dump(),
        "arm_config": arm.model_dump(),
        "command": command,
        "timestamp": _timestamp(),
        "git_commit": _git_commit(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

def dry_run(
    grid: ExperimentGrid,
    output_root: Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    warnings = grid_methodology_warnings(grid)
    for warning in warnings:
        print(f"METHODOLOGY WARNING: {warning}")

    for arm in grid.arms:
        run_dir = planned_run_dir_label(output_root, grid.grid_name, arm.name)
        command = build_command(
            arm, run_dir, provider=provider, model=model, api_key=api_key, base_url=base_url
        )
        display_command = _redact_command(command)
        plans.append({
            "arm_name": arm.name,
            "label": arm.label,
            "run_dir": str(run_dir),
            # Redacted even in the returned plan -- dry-run output/plans may
            # be printed, logged, or diffed by a caller, and should never
            # carry a live secret just because a real command was built to
            # validate its construction.
            "command": display_command,
        })
        print(f"[dry-run] arm={arm.name!r} label={arm.label!r}")
        print(f"          run_dir (estimated): {run_dir}")
        print("          command: " + " ".join(display_command))
    return plans


# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------

SubprocessRunner = Callable[[list[str]], "subprocess.CompletedProcess"]


def dispatch_run(
    grid: ExperimentGrid,
    output_root: Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> list[dict[str, Any]]:
    """Dispatch every arm as a subprocess call. Only mode that can make live
    LLM calls (via the invoked script) -- never used by this repo's tests."""
    results: list[dict[str, Any]] = []
    for arm in grid.arms:
        run_dir = allocate_run_dir(output_root, grid.grid_name, arm.name)
        command = build_command(
            arm, run_dir, provider=provider, model=model, api_key=api_key, base_url=base_url
        )
        write_run_manifest(run_dir, grid=grid, arm=arm, command=command)
        print(f"Running arm={arm.name!r} -> {run_dir}")
        completed = subprocess_runner(command)
        returncode = getattr(completed, "returncode", 0)
        if returncode != 0:
            print(f"WARNING: arm {arm.name!r} exited with code {returncode}", file=sys.stderr)
        results.append({"arm_name": arm.name, "run_dir": run_dir, "returncode": returncode})
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def discover_run_dirs(output_root: Path, grid_name: str | None = None) -> list[Path]:
    root = (output_root / grid_name) if grid_name else output_root
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("run_manifest.json")})


def load_run_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _to_number(value: Any) -> Any:
    """Best-effort numeric coercion of a CSV string cell; passes through
    non-numeric/blank values unchanged."""
    if value is None or value == "":
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _blank_result_row(**overrides: Any) -> dict[str, Any]:
    row = {col: "" for col in RESULT_COLUMNS}
    row.update(overrides)
    return row


def _candidate_bundle_stats_for_scenario(
    pv_rows: list[dict[str, str]], scenario: str
) -> dict[str, Any]:
    rows = [r for r in pv_rows if r.get("scenario") == scenario]
    if not rows:
        return {}
    generated = sum(int(r["candidate_bundles_generated"]) for r in rows if r.get("candidate_bundles_generated"))
    sent = sum(int(r["candidate_bundles_sent_to_pv"]) for r in rows if r.get("candidate_bundles_sent_to_pv"))
    truncated = any(r.get("candidate_bundles_truncated") in ("True", "true", "1") for r in rows)
    reasons = sorted({r["candidate_truncation_reason"] for r in rows if r.get("candidate_truncation_reason")})
    max_candidate_bundles = rows[0].get("max_candidate_bundles", "")
    return {
        "candidate_bundles_generated": generated,
        "candidate_bundles_sent_to_pv": sent,
        "candidate_bundles_truncated": truncated,
        "candidate_truncation_reason": "; ".join(reasons),
        "max_candidate_bundles": max_candidate_bundles,
    }


def _copy_refinement_cap_fields(row: dict[str, Any], source: dict[str, str]) -> None:
    """Copy the per-arm refinement-cap bookkeeping columns, if present.

    ``source`` is one row from a mechanism-specific CSV
    (``curated_sealed_proxy_elicited.csv`` /
    ``curated_clock_proxy_elicited_top_{k}.csv``), already carrying these
    via :func:`auctionlab.experiments._trajectory_util.refinement_cap_fields`.
    """
    for field in (
        "total_refinement_queries",
        "per_bidder_refinement_queries",
        "cap_binding_indicator",
        "safety_cap_hit",
    ):
        if field in source:
            row[field] = source[field]


def _fi_coverage_for_scenario(coverage_rows: list[dict[str, str]], scenario: str) -> Any:
    rows = [r for r in coverage_rows if r.get("scenario") == scenario]
    if not rows:
        return ""
    included = [
        1.0 if r.get("included_in_supplementary_by_final") in ("True", "true", "1") else 0.0
        for r in rows
    ]
    return sum(included) / len(included)


def arm_rows_from_run_dir(run_dir: Path) -> list[dict[str, Any]]:
    """Build full-width result rows for every arm found in one run directory."""
    manifest = load_run_manifest(run_dir) or {}
    env = manifest.get("environment", {})
    arm_config = manifest.get("arm_config", {})
    pcp = arm_config.get("proxy_candidate_policy", {})
    ct = arm_config.get("clock_treatment", {})
    st = arm_config.get("sealed_treatment", {})
    sl = arm_config.get("safety_limits", {})

    common = {
        "run_label": manifest.get("run_label", run_dir.name),
        "config_name": manifest.get("config_name", run_dir.name),
        "scenario_spec": env.get("scenario_spec", "") or "",
        "num_goods": env.get("num_goods", ""),
        "num_bidders": env.get("num_bidders", ""),
        "scenario_seed": env.get("scenario_seed", ""),
        "pv_max_tokens": pcp.get("pv_max_tokens", ""),
        "max_bundle_size": pcp.get("max_bundle_size", ""),
        "sealed_feedback_rule": st.get("sealed_feedback_rule", ""),
        "sealed_elicitation_rounds": st.get("sealed_elicitation_rounds", ""),
        "clock_tie_threshold": ct.get("clock_tie_threshold", ""),
        "clock_margin_threshold": ct.get("clock_margin_threshold", ""),
        "price_step": ct.get("price_step", ""),
        "max_rounds": ct.get("max_rounds", ""),
        # null in the manifest (no limit configured) surfaces as "" here,
        # same as every other unset column -- never a literal 0.
        "per_bidder_refinement_query_limit": sl.get("per_bidder_refinement_query_limit") or "",
        "global_refinement_query_safety_limit": sl.get("global_refinement_query_safety_limit") or "",
    }

    summary_rows = _read_csv_rows(run_dir / "curated_run_summary.csv")
    pv_rows = _read_csv_rows(run_dir / "curated_pv_candidate_bundle_stats.csv")

    # Per-mechanism enrichment sources, keyed by instance_name/scenario.
    sealed_rows = {r["instance_name"]: r for r in _read_csv_rows(run_dir / "curated_sealed_llm_comparison.csv")}
    proxy_sealed_rows = {r["instance_name"]: r for r in _read_csv_rows(run_dir / "curated_sealed_proxy_elicited.csv")}
    clock_rows_by_topk: dict[int, dict[str, dict]] = {}
    proxy_clock_rows_by_topk: dict[int, dict[str, dict]] = {}
    coverage_rows_by_topk: dict[int, list[dict]] = {}
    for path in run_dir.glob("curated_clock_llm_comparison_top_*.csv"):
        k = int(re.search(r"top_(\d+)", path.name).group(1))
        clock_rows_by_topk[k] = {r["instance_name"]: r for r in _read_csv_rows(path)}
    for path in run_dir.glob("curated_clock_proxy_elicited_top_*.csv"):
        k = int(re.search(r"top_(\d+)", path.name).group(1))
        proxy_clock_rows_by_topk[k] = {r["instance_name"]: r for r in _read_csv_rows(path)}
    for path in run_dir.glob("curated_proxy_clock_coverage_top_*.csv"):
        k = int(re.search(r"top_(\d+)", path.name).group(1))
        coverage_rows_by_topk[k] = _read_csv_rows(path)

    rows: list[dict[str, Any]] = []
    for srow in summary_rows:
        arm_label = srow.get("arm", "")
        if arm_label.startswith("shared initial"):
            continue
        scenario = srow.get("scenario", "")
        row = _blank_result_row(
            **common,
            arm=arm_label,
            final_true_welfare=_to_number(srow.get("true_welfare")),
            full_info_welfare=_to_number(srow.get("full_info_welfare")),
            final_efficiency=_to_number(srow.get("efficiency")),
            revenue=_to_number(srow.get("revenue")),
            true_surplus=_to_number(srow.get("surplus")),
            value_queries=_to_number(srow.get("vq")),
            demand_queries=_to_number(srow.get("dq")),
            nl_queries=_to_number(srow.get("nl")),
            estimated_tokens_in=_to_number(srow.get("tok_in")),
            tokens_out=_to_number(srow.get("tok_out")),
        )

        candidate_stats = _candidate_bundle_stats_for_scenario(pv_rows, scenario)
        row.update({k: v for k, v in candidate_stats.items() if v != ""})

        if arm_label == "sealed" and scenario in sealed_rows:
            r = sealed_rows[scenario]
            row["reported_welfare"] = _to_number(r.get("llm_proxy_reported_welfare"))
        elif arm_label == "proxy sealed" and scenario in proxy_sealed_rows:
            r = proxy_sealed_rows[scenario]
            row["reported_welfare"] = _to_number(r.get("proxy_reported_welfare"))
            _copy_refinement_cap_fields(row, r)
        else:
            m = _ARM_TOPK_RE.search(arm_label)
            top_k = int(m.group(1)) if m else None
            if top_k is not None:
                row["top_k"] = top_k
                if arm_label.startswith("clock") and top_k in clock_rows_by_topk and scenario in clock_rows_by_topk[top_k]:
                    r = clock_rows_by_topk[top_k][scenario]
                    row["reported_welfare"] = _to_number(r.get("clock_llm_reported_welfare"))
                    row["clock_rounds"] = _to_number(r.get("clock_rounds"))
                elif arm_label.startswith("proxy clock") and top_k in proxy_clock_rows_by_topk and scenario in proxy_clock_rows_by_topk[top_k]:
                    r = proxy_clock_rows_by_topk[top_k][scenario]
                    row["reported_welfare"] = _to_number(r.get("proxy_reported_welfare"))
                    row["clock_rounds"] = _to_number(r.get("rounds"))
                    row["termination_reason"] = r.get("termination_reason", "")
                    row["failure_classification"] = r.get("failure_classification", "")
                    row["supplementary_atoms_total"] = _to_number(r.get("supplementary_atoms_total"))
                    row["best_true_welfare"] = _to_number(r.get("best_true_welfare"))
                    row["best_efficiency"] = _to_number(r.get("best_true_efficiency"))
                    _copy_refinement_cap_fields(row, r)
                    if r.get("final_true_welfare"):
                        row["final_true_welfare"] = _to_number(r.get("final_true_welfare"))
                    if r.get("final_true_efficiency"):
                        row["final_efficiency"] = _to_number(r.get("final_true_efficiency"))
                    if top_k in coverage_rows_by_topk:
                        row["fi_coverage"] = _to_number(
                            _fi_coverage_for_scenario(coverage_rows_by_topk[top_k], scenario)
                        )

        rows.append(row)

    return rows


def aggregate(
    output_root: Path,
    *,
    grid_name: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    run_dirs = discover_run_dirs(output_root, grid_name)
    results: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        results.extend(arm_rows_from_run_dir(run_dir))

    summary_by_config = _summarize_by(results, key="config_name")
    summary_by_environment = _summarize_by(
        results,
        key=lambda r: (r["scenario_spec"], r["num_goods"], r["num_bidders"], r["scenario_seed"]),
        key_columns=("scenario_spec", "num_goods", "num_bidders", "scenario_seed"),
    )
    report = _build_report_markdown(results, summary_by_config, summary_by_environment)
    return results, summary_by_config, summary_by_environment, report


def _mean(values: list[float]) -> float | str:
    values = [v for v in values if isinstance(v, (int, float))]
    return (sum(values) / len(values)) if values else ""


def _summarize_by(
    results: list[dict[str, Any]],
    *,
    key: Any,
    key_columns: tuple[str, ...] = ("config_name",),
) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in results:
        k = key(row) if callable(key) else row.get(key)
        groups.setdefault(k, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for k, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        key_values = k if isinstance(k, tuple) else (k,)
        arms = {r["arm"] for r in rows}
        sealed_effs = [r["final_efficiency"] for r in rows if r["arm"] in ("sealed", "proxy sealed")]
        clock_effs = [r["final_efficiency"] for r in rows if "clock" in r["arm"]]
        proxy_sealed_effs = [r["final_efficiency"] for r in rows if r["arm"] == "proxy sealed"]
        proxy_clock_effs = [r["final_efficiency"] for r in rows if r["arm"].startswith("proxy clock")]
        sealed_vq = [r["value_queries"] for r in rows if r["arm"] in ("sealed", "proxy sealed")]
        clock_vq = [r["value_queries"] for r in rows if "clock" in r["arm"]]

        row_out: dict[str, Any] = {}
        for col, val in zip(key_columns, key_values):
            row_out[col] = val
        row_out.update({
            "n_arms": len(rows),
            "arms_present": ";".join(sorted(arms)),
            "mean_final_efficiency": _mean([v for v in (r["final_efficiency"] for r in rows) if v != ""]),
            "best_sealed_efficiency": max((v for v in sealed_effs if v != ""), default=""),
            "best_clock_efficiency": max((v for v in clock_effs if v != ""), default=""),
            "best_proxy_sealed_efficiency": max((v for v in proxy_sealed_effs if v != ""), default=""),
            "best_proxy_clock_efficiency": max((v for v in proxy_clock_effs if v != ""), default=""),
            "mean_sealed_value_queries": _mean([v for v in sealed_vq if v != ""]),
            "mean_clock_value_queries": _mean([v for v in clock_vq if v != ""]),
        })
        if key_columns == ("config_name",):
            row_out["config_name"] = k
        summary_rows.append(row_out)
    return summary_rows


def _build_report_markdown(
    results: list[dict[str, Any]],
    summary_by_config: list[dict[str, Any]],
    summary_by_environment: list[dict[str, Any]],
) -> str:
    lines = [
        "# Parameter grid report",
        "",
        f"{len(results)} arm-result rows aggregated. See "
        "`docs/parameter_tuning_methodology.md` for how development/final/"
        "robustness grids should be interpreted differently.",
        "",
        "## Summary by config (arm)",
        "",
        "| config_name | n_arms | mean_final_efficiency | best_sealed_efficiency | best_clock_efficiency |",
        "|---|---|---|---|---|",
    ]
    for row in summary_by_config:
        lines.append(
            f"| {row['config_name']} | {row['n_arms']} | "
            f"{row['mean_final_efficiency']} | {row['best_sealed_efficiency']} | "
            f"{row['best_clock_efficiency']} |"
        )

    lines += [
        "",
        "## Summary by environment",
        "",
        "| scenario_spec | num_goods | num_bidders | scenario_seed | best_proxy_sealed_efficiency | best_proxy_clock_efficiency | mean_sealed_vq | mean_clock_vq |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary_by_environment:
        lines.append(
            f"| {row['scenario_spec']} | {row['num_goods']} | {row['num_bidders']} | "
            f"{row['scenario_seed']} | {row['best_proxy_sealed_efficiency']} | "
            f"{row['best_proxy_clock_efficiency']} | {row['mean_sealed_value_queries']} | "
            f"{row['mean_clock_value_queries']} |"
        )

    # Effect of top_k, where present.
    topk_rows = [r for r in results if r["top_k"] != ""]
    if topk_rows:
        lines += ["", "## Effect of clock top_k", "", "| top_k | n | mean_final_efficiency | mean_clock_rounds | mean_value_queries |", "|---|---|---|---|---|"]
        by_topk: dict[int, list[dict]] = {}
        for row in topk_rows:
            by_topk.setdefault(row["top_k"], []).append(row)
        for k in sorted(by_topk):
            rows = by_topk[k]
            effs = [r["final_efficiency"] for r in rows if r["final_efficiency"] != ""]
            rounds = [r["clock_rounds"] for r in rows if r["clock_rounds"] != ""]
            vqs = [r["value_queries"] for r in rows if r["value_queries"] != ""]
            lines.append(
                f"| {k} | {len(rows)} | {_mean(effs)} | {_mean(rounds)} | {_mean(vqs)} |"
            )

    # Effect of sealed feedback rule / elicitation depth, where present.
    sealed_treatment_rows = [
        r for r in results if r["arm"] == "proxy sealed" and r["sealed_feedback_rule"] != ""
    ]
    if sealed_treatment_rows:
        lines += ["", "## Effect of sealed feedback rule and elicitation depth", "", "| feedback_rule | rounds | n | mean_final_efficiency | mean_value_queries |", "|---|---|---|---|---|"]
        by_treatment: dict[tuple, list[dict]] = {}
        for row in sealed_treatment_rows:
            key = (row["sealed_feedback_rule"], row["sealed_elicitation_rounds"])
            by_treatment.setdefault(key, []).append(row)
        for (rule, rounds), rows in sorted(by_treatment.items(), key=lambda kv: str(kv[0])):
            effs = [r["final_efficiency"] for r in rows if r["final_efficiency"] != ""]
            vqs = [r["value_queries"] for r in rows if r["value_queries"] != ""]
            lines.append(f"| {rule} | {rounds} | {len(rows)} | {_mean(effs)} | {_mean(vqs)} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-file", required=True, type=str)
    parser.add_argument("--output-root", required=True, type=str)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_root = Path(args.output_root)

    if args.dry_run and args.aggregate_only:
        raise SystemExit("--dry-run and --aggregate-only are mutually exclusive")

    grid = load_experiment_grid(args.grid_file)
    print(f"Loaded grid {grid.grid_name!r} ({grid.grid_type}) with {len(grid.arms)} arm(s)")

    if args.dry_run:
        dry_run(
            grid, output_root,
            provider=args.provider, model=args.model,
            api_key=args.api_key, base_url=args.base_url,
        )
        return

    if not args.aggregate_only:
        dispatch_run(
            grid, output_root,
            provider=args.provider, model=args.model,
            api_key=args.api_key, base_url=args.base_url,
        )

    results, summary_by_config, summary_by_environment, report = aggregate(
        output_root, grid_name=grid.grid_name
    )
    write_csv_variable_rows(results, output_root / "parameter_grid_results.csv")
    write_csv_variable_rows(summary_by_config, output_root / "parameter_grid_summary_by_config.csv")
    write_csv_variable_rows(summary_by_environment, output_root / "parameter_grid_summary_by_environment.csv")
    (output_root / "parameter_grid_report.md").write_text(report)
    print(f"Wrote aggregate outputs to {output_root}")


if __name__ == "__main__":
    main()
