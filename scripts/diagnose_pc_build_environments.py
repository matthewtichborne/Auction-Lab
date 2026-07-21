#!/usr/bin/env python3
"""Deterministic diagnostic comparing the manual vs. generated PC-build specs.

Compares the "v0 manual" ``ScenarioProfileSpec`` (exported from the
hard-coded archetype builders in :mod:`auctionlab.instances.structured`)
against the "gemini_trial_final" LLM-generated spec, at three sizes
(6x6, 8x8, 10x10), on a battery of full-information economic diagnostics.

Nothing in this script makes an LLM/API call. All valuations come from the
frozen spec's base values via the deterministic
``generate_full_valuations``/``value_bundle`` machinery; the only place an
LLM ever appears is as *historical, cached* log data (``calls.jsonl``)
consulted read-only for the provisional-value-vs-ground-truth check.

Usage::

    ./venv/bin/python scripts/diagnose_pc_build_environments.py

    # Compare against a different generated spec, and/or add a third
    # "calibrated" environment (e.g. a candidate frozen from
    # calibrate_generated_pc_build_spec.py), writing to a separate directory:
    ./venv/bin/python scripts/diagnose_pc_build_environments.py \\
        --generated-spec scenarios/pc_build_v1/pc_build_profiles_v4_gemini_trial.json \\
        --calibrated-spec scenarios/pc_build_v1/pc_build_profiles_generated_calibrated.json \\
        --output-dir scenarios/pc_build_v1/diagnostics_calibrated

Writes to ``--output-dir`` (default ``scenarios/pc_build_v1/diagnostics/``):
    - environment_size_summary.csv
    - per_good_positive_bidders.csv
    - full_info_allocation.csv
    - pv_vs_gt_comparison.csv
    - pv_gt_error_breakdown.csv
    - comparison_report.md
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import json
import random
import re
import statistics
from pathlib import Path

from auctionlab.instances.scenario_spec import (
    BidderProfileSpec,
    ComplementGroupSpec,
    GoodSpec,
    ScenarioProfileSpec,
    SubstituteGroupSpec,
    load_scenario_profile_spec,
)
from auctionlab.instances.structured import (
    PC_GOOD_CATALOG,
    PC_ITEM_DESCRIPTIONS,
    _ARCHETYPE_BUILDERS,
    _ARCHETYPE_ORDER,
    BidderPreferenceProfile,
    value_bundle,
)
from auctionlab.instances.structured_spec import make_pc_build_scenario_from_spec, profile_from_spec
from auctionlab.solvers.wdp_ilp import solve_wdp_xor_ilp

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_SPEC_PATH = REPO_ROOT / "scenarios/pc_build_v1/pc_build_profiles_gemini_trial_final.json"
OUTPUT_DIR = REPO_ROOT / "scenarios/pc_build_v1/diagnostics"

SIZES: list[tuple[int, int]] = [(6, 6), (8, 8), (10, 10)]
EXPORT_SEED = 0

RESELLER_ID = {"manual": "pc_reseller", "generated": "reseller_pro"}

# Fixed-N (not percentile) thresholds so results are comparable across sizes
# with very different total bundle-table sizes.
TOP_N_GLOBAL = 10       # "top-10 true bundles" (item 8)
TOP_N_HIGH_VALUE = 20   # reseller high-value-bundle share (item 13)
TOP_K_GOODS_PER_BIDDER = 3  # "top-valued goods" per bidder (item 7)
_TOL = 1e-6

LOG_DIRS: dict[tuple[int, int], Path | None] = {
    (6, 6): REPO_ROOT / "logs/spec_gemini_trial_6x6_seed0",
    (8, 8): REPO_ROOT / "logs/spec_gemini_trial_8x8_seed0",
    (10, 10): None,  # no cached run at this size
}


# ---------------------------------------------------------------------------
# Manual spec (inlined from scripts/export_current_pc_build_profiles.py to
# avoid a fragile cross-script import when this file is run directly rather
# than via `-m`)
# ---------------------------------------------------------------------------

def build_manual_spec(seed: int = EXPORT_SEED) -> ScenarioProfileSpec:
    items = list(PC_GOOD_CATALOG)
    items_set = set(items)
    rng = random.Random(seed)

    goods = [GoodSpec(id=item, description=PC_ITEM_DESCRIPTIONS[item]) for item in items]

    bidder_profiles: list[BidderProfileSpec] = []
    for bidder_id in _ARCHETYPE_ORDER:
        profile = _ARCHETYPE_BUILDERS[bidder_id](items_set, rng)  # type: ignore[operator]
        bidder_profiles.append(
            BidderProfileSpec(
                bidder_id=profile.bidder_id,
                role=profile.role,
                budget_range=profile.budget_range,
                base_values=dict(profile.base_values),
                substitute_groups=[
                    SubstituteGroupSpec(
                        items=sorted(sg.items), backup_factor=sg.backup_factor, description=sg.description
                    )
                    for sg in profile.substitute_groups
                ],
                complement_groups=[
                    ComplementGroupSpec(items=sorted(cg.items), bonus=cg.bonus, description=cg.description)
                    for cg in profile.complement_groups
                ],
                budget_cap=profile.budget_cap,
                saturation_start=profile.saturation_start,
                saturation_penalty=profile.saturation_penalty,
                notes=profile.notes,
                core_items=sorted(profile.core_items),
                secondary_items=sorted(profile.secondary_items),
                low_interest_items=sorted(profile.low_interest_items),
            )
        )

    return ScenarioProfileSpec(
        schema_version="pc_build_profile_spec_v1",
        domain="pc_build",
        description="Manual baseline PC-build profile universe (in-memory, not written to disk).",
        goods=goods,
        bidder_profiles=bidder_profiles,
        generation={
            "source": "scripts/export_current_pc_build_profiles.py:build_manual_spec",
            "export_seed": seed,
            "method": "hard_coded_archetype_builders",
        },
        notes="v0 manual spec, built in-memory by diagnose_pc_build_environments.py.",
    )


# ---------------------------------------------------------------------------
# Per-bidder helpers (complement / saturation contribution shares)
# ---------------------------------------------------------------------------

def _complement_contribution_share(profile: BidderPreferenceProfile, items: list[str]) -> float:
    grand = frozenset(items)
    grand_value = value_bundle(profile, grand)
    if grand_value <= 0:
        return 0.0
    total_bonus = sum(cg.bonus for cg in profile.complement_groups if cg.items.issubset(grand))
    return total_bonus / grand_value


def _saturation_contribution_share(profile: BidderPreferenceProfile, items: list[str]) -> float:
    """Share of grand-bundle value *lost* to the saturation penalty (0 if no saturation)."""
    if profile.saturation_start is None:
        return 0.0
    grand = frozenset(items)
    actual = value_bundle(profile, grand)
    unsaturated = dataclasses.replace(profile, saturation_start=None, saturation_penalty=0.0)
    unsat_value = value_bundle(unsaturated, grand)
    if unsat_value <= 0:
        return 0.0
    return max(0.0, (unsat_value - actual) / unsat_value)


def _budget_cap_binding(profile: BidderPreferenceProfile, items: list[str]) -> bool:
    if profile.budget_cap is None:
        return False
    grand = frozenset(items)
    actual = value_bundle(profile, grand)
    uncapped = dataclasses.replace(profile, budget_cap=None)
    return actual < value_bundle(uncapped, grand) - _TOL


# ---------------------------------------------------------------------------
# Cached-log PV vs. ground-truth comparison (item 15; generated spec only)
# ---------------------------------------------------------------------------

def load_pv_vs_gt(
    log_dir: Path | None,
    bidder_ids: set[str],
    *,
    winners: dict[str, frozenset] | None = None,
) -> list[dict]:
    """Compare LLM provisional valuations against cached ground-truth values.

    Reads *only* already-written ``calls.jsonl`` log files from a prior run
    -- no live LLM/API call is made here. ``proxy_provisional_valuations``
    entries hold the LLM's estimate for each bidder's initial candidate
    bundles (their interest-derived bundle shortlist); ``value_query_gt``
    entries hold the true valuation-table value the harness logged for
    comparison. Bundles are matched by (bidder_id, frozenset(items)).

    If ``winners`` (bidder_id -> winning bundle, from the full-info WDP
    solve for this size) is supplied, each row is tagged ``is_allocated``:
    whether that exact candidate bundle is the bidder's full-info winning
    bundle. Most PV candidate bundles are small interest-derived shortlists
    rather than the eventual winning bundle, so a low allocated-bundle count
    here is expected, not a bug.
    """
    if log_dir is None or not log_dir.exists():
        return []

    pv_by_key: dict[tuple[str, frozenset], float] = {}
    gt_by_key: dict[tuple[str, frozenset], float] = {}

    calls_path = log_dir / "calls.jsonl"
    if not calls_path.exists():
        return []

    with open(calls_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            bidder_id = entry.get("bidder_id")
            if bidder_id not in bidder_ids:
                continue
            prompt_type = entry.get("prompt_type")

            if prompt_type == "proxy_provisional_valuations":
                parsed = entry.get("parsed_response") or {}
                valuations = parsed.get("valuations") or {}
                for bundle_str, value in valuations.items():
                    try:
                        bundle = frozenset(ast.literal_eval(bundle_str))
                    except (ValueError, SyntaxError):
                        continue
                    pv_by_key[(bidder_id, bundle)] = float(value)

            elif prompt_type == "value_query_gt":
                m_bundle = re.search(r"bundle=(\[.*\])", entry.get("prompt", ""))
                m_value = re.search(r"ground_truth=([\-0-9.eE]+)", entry.get("raw_response", ""))
                if not (m_bundle and m_value):
                    continue
                try:
                    bundle = frozenset(ast.literal_eval(m_bundle.group(1)))
                except (ValueError, SyntaxError):
                    continue
                gt_by_key[(bidder_id, bundle)] = float(m_value.group(1))

    rows: list[dict] = []
    for (bidder_id, bundle), pv in sorted(pv_by_key.items(), key=lambda kv: (kv[0][0], -kv[1])):
        gt = gt_by_key.get((bidder_id, bundle))
        if gt is None:
            continue
        signed_error = pv - gt
        abs_error = abs(signed_error)
        pct_error = (signed_error / gt) if gt != 0 else None
        abs_pct_error = abs(pct_error) if pct_error is not None else None
        is_allocated = winners.get(bidder_id) == bundle if winners is not None else None
        rows.append(
            {
                "bidder": bidder_id,
                "bundle": "+".join(sorted(bundle)),
                "bundle_size": len(bundle),
                "pv_value": pv,
                "gt_value": gt,
                "signed_error": signed_error,
                "abs_error": abs_error,
                "pct_error": pct_error,
                "abs_pct_error": abs_pct_error,
                "is_allocated": is_allocated,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# PV vs. GT error summaries / breakdowns (extends item 15)
# ---------------------------------------------------------------------------

def _summarize_error_rows(rows: list[dict]) -> dict:
    """Aggregate stats for one group of PV-vs-GT rows.

    ``mean_ratio``/``median_ratio`` are the reported/true *scale* diagnostic:
    mean(PV)/mean(GT) and median(PV)/median(GT) over the group -- a single
    systematic multiplier close to 1.0 means PV and GT are on the same
    scale on average even if individual bundle estimates are noisy; a
    multiplier far from 1.0 (e.g. consistently < 1) indicates the LLM proxy
    is under- or over-valuing bundles as a class, not just imprecisely.
    """
    if not rows:
        return {
            "count": 0,
            "mean_signed_error": None,
            "mean_abs_error": None,
            "median_abs_error": None,
            "mean_abs_pct_error": None,
            "mean_pv": None,
            "mean_gt": None,
            "mean_ratio": None,
            "median_pv": None,
            "median_gt": None,
            "median_ratio": None,
        }

    signed_errors = [r["signed_error"] for r in rows]
    abs_errors = [r["abs_error"] for r in rows]
    abs_pct_errors = [r["abs_pct_error"] for r in rows if r["abs_pct_error"] is not None]
    pv_values = [r["pv_value"] for r in rows]
    gt_values = [r["gt_value"] for r in rows]

    mean_pv = statistics.fmean(pv_values)
    mean_gt = statistics.fmean(gt_values)
    median_pv = statistics.median(pv_values)
    median_gt = statistics.median(gt_values)

    return {
        "count": len(rows),
        "mean_signed_error": statistics.fmean(signed_errors),
        "mean_abs_error": statistics.fmean(abs_errors),
        "median_abs_error": statistics.median(abs_errors),
        "mean_abs_pct_error": statistics.fmean(abs_pct_errors) if abs_pct_errors else None,
        "mean_pv": mean_pv,
        "mean_gt": mean_gt,
        "mean_ratio": (mean_pv / mean_gt) if mean_gt != 0 else None,
        "median_pv": median_pv,
        "median_gt": median_gt,
        "median_ratio": (median_pv / median_gt) if median_gt != 0 else None,
    }


def build_pv_gt_error_breakdown(rows: list[dict], environment: str, size: str) -> list[dict]:
    """Group PV-vs-GT rows by bidder / bundle size / allocated-vs-not / overall.

    One row per group with ``group_by`` in {"overall", "bidder",
    "bundle_size", "allocated"} and ``group_value`` naming the specific
    group, plus the signed/absolute/percentage error stats and the
    reported/true scale ratios from :func:`_summarize_error_rows`.
    """
    if not rows:
        return []

    breakdown_rows: list[dict] = []

    def _emit(group_by: str, group_value: str, subset: list[dict]) -> None:
        breakdown_rows.append(
            {
                "environment": environment,
                "size": size,
                "group_by": group_by,
                "group_value": group_value,
                **_summarize_error_rows(subset),
            }
        )

    _emit("overall", "all", rows)

    by_bidder: dict[str, list[dict]] = {}
    for r in rows:
        by_bidder.setdefault(r["bidder"], []).append(r)
    for bidder_id, subset in sorted(by_bidder.items()):
        _emit("bidder", bidder_id, subset)

    by_bundle_size: dict[int, list[dict]] = {}
    for r in rows:
        by_bundle_size.setdefault(r["bundle_size"], []).append(r)
    for bundle_size, subset in sorted(by_bundle_size.items()):
        _emit("bundle_size", str(bundle_size), subset)

    if any(r["is_allocated"] is not None for r in rows):
        allocated_rows = [r for r in rows if r["is_allocated"]]
        non_allocated_rows = [r for r in rows if r["is_allocated"] is False]
        if allocated_rows:
            _emit("allocated", "allocated", allocated_rows)
        if non_allocated_rows:
            _emit("allocated", "non_allocated", non_allocated_rows)

    return breakdown_rows


# ---------------------------------------------------------------------------
# Per (environment, size) diagnostics
# ---------------------------------------------------------------------------

def compute_size_metrics(
    env_name: str,
    spec: ScenarioProfileSpec,
    num_goods: int,
    num_bidders: int,
    *,
    reseller_id: str | None = None,
    compute_pv_gt: bool | None = None,
) -> dict:
    """Compute the full diagnostic battery for one (environment, size) pair.

    ``env_name`` is a free-form label used for output tagging; it only has
    special meaning for the two built-in environments ("manual" / "generated")
    via ``RESELLER_ID`` and the PV-vs-GT default below. Calibration candidates
    (see ``calibrate_generated_pc_build_spec.py``) pass an arbitrary label
    plus explicit ``reseller_id``/``compute_pv_gt`` instead.

    - ``reseller_id``: overrides the ``RESELLER_ID`` lookup (falls back to
      ``"reseller_pro"`` if ``env_name`` isn't a recognized key).
    - ``compute_pv_gt``: whether to run the cached-log PV-vs-GT comparison
      (item 15). Defaults to ``env_name == "generated"``.
    """
    reseller_id = reseller_id if reseller_id is not None else RESELLER_ID.get(env_name, "reseller_pro")
    if compute_pv_gt is None:
        compute_pv_gt = env_name == "generated"

    scenario = make_pc_build_scenario_from_spec(
        spec, num_goods, num_bidders, seed=0, selection_policy="prefix"
    )
    instance = scenario.instance
    items = list(instance.items)
    bidder_ids = list(instance.bidder_ids)
    items_set = set(items)
    bidders_by_id = {b.bidder_id: b for b in spec.bidder_profiles}

    # --- 1-4: full-info optimum, winners, largest share, avg winning size ---
    wdp = solve_wdp_xor_ilp(items, instance.to_xor_bids())
    winners = {b: bundle for b, bundle in wdp.allocation.items() if bundle}
    per_winner_value = {b: instance.valuations[b].get(bundle, 0.0) for b, bundle in winners.items()}
    optimum = wdp.welfare
    num_winners = len(winners)
    largest_share = (
        max(per_winner_value.values()) / optimum if optimum > 0 and per_winner_value else 0.0
    )
    avg_winning_bundle_size = (
        statistics.fmean(len(b) for b in winners.values()) if winners else 0.0
    )

    # --- 5-6: contested goods / positive-value-per-good ---
    positive_per_good: dict[str, int] = {}
    for good in items:
        singleton = frozenset([good])
        positive_per_good[good] = sum(
            1 for bid in bidder_ids if instance.valuations[bid].get(singleton, 0.0) > 0
        )
    num_contested_goods = sum(1 for c in positive_per_good.values() if c >= 2)

    # --- 7: avg pairwise bidder overlap over top-valued goods ---
    top_goods_per_bidder: dict[str, set[str]] = {}
    for bid in bidder_ids:
        ranked = sorted(
            items, key=lambda g: instance.valuations[bid].get(frozenset([g]), 0.0), reverse=True
        )
        k = min(TOP_K_GOODS_PER_BIDDER, len(items))
        top_goods_per_bidder[bid] = {
            g for g in ranked[:k] if instance.valuations[bid].get(frozenset([g]), 0.0) > 0
        }
    pairwise_jaccards = []
    for i in range(len(bidder_ids)):
        for j in range(i + 1, len(bidder_ids)):
            a = top_goods_per_bidder[bidder_ids[i]]
            b = top_goods_per_bidder[bidder_ids[j]]
            union = a | b
            pairwise_jaccards.append(len(a & b) / len(union) if union else 0.0)
    avg_pairwise_overlap = statistics.fmean(pairwise_jaccards) if pairwise_jaccards else 0.0

    # --- 8, 12, 13: global bundle-value ranking (shared across three metrics) ---
    all_triples = [
        (bid, bundle, val) for bid in bidder_ids for bundle, val in instance.valuations[bid].items()
    ]
    all_triples.sort(key=lambda t: t[2], reverse=True)

    top10 = all_triples[:TOP_N_GLOBAL]
    share_top10_size_ge4 = (
        sum(1 for _, bundle, _ in top10 if len(bundle) >= 4) / len(top10) if top10 else 0.0
    )

    positive_values = [v for _, _, v in all_triples if v > 0]
    value_min = min(positive_values) if positive_values else 0.0
    value_median = statistics.median(positive_values) if positive_values else 0.0
    value_mean = statistics.fmean(positive_values) if positive_values else 0.0
    value_max = max(positive_values) if positive_values else 0.0

    top_high_value = all_triples[:TOP_N_HIGH_VALUE]
    top_high_value_total = sum(v for _, _, v in top_high_value)
    reseller_count_in_top = sum(1 for bid, _, _ in top_high_value if bid == reseller_id)
    reseller_value_share_in_top = (
        sum(v for bid, _, v in top_high_value if bid == reseller_id) / top_high_value_total
        if top_high_value_total > 0
        else 0.0
    )

    # --- 9-11: complement / saturation / budget-cap ---
    profiles = {bid: profile_from_spec(bidders_by_id[bid], items_set) for bid in bidder_ids}
    complement_shares = [_complement_contribution_share(profiles[bid], items) for bid in bidder_ids]
    avg_complement_share = statistics.fmean(complement_shares) if complement_shares else 0.0
    max_complement_share = max(complement_shares, default=0.0)

    saturation_contribs = [_saturation_contribution_share(profiles[bid], items) for bid in bidder_ids]
    avg_saturation_contrib = statistics.fmean(saturation_contribs) if saturation_contribs else 0.0

    num_budget_cap_binding = sum(1 for bid in bidder_ids if _budget_cap_binding(profiles[bid], items))

    # --- 14: full-info allocation table rows ---
    allocation_rows = [
        {
            "environment": env_name,
            "size": f"{num_goods}x{num_bidders}",
            "bidder": bid,
            "bundle": "+".join(sorted(bundle)),
            "bundle_size": len(bundle),
            "true_value": per_winner_value[bid],
        }
        for bid, bundle in sorted(winners.items())
    ]

    per_good_rows = [
        {
            "environment": env_name,
            "size": f"{num_goods}x{num_bidders}",
            "good": good,
            "num_bidders_with_positive_value": count,
            "num_bidders_total": len(bidder_ids),
        }
        for good, count in positive_per_good.items()
    ]

    # --- 15: PV vs GT from cached logs (generated spec only, by default) ---
    log_dir = LOG_DIRS.get((num_goods, num_bidders))
    pv_gt_rows: list[dict] = []
    pv_gt_breakdown_rows: list[dict] = []
    if compute_pv_gt:
        pv_gt_rows = load_pv_vs_gt(log_dir, set(bidder_ids), winners=winners)
        for row in pv_gt_rows:
            row["environment"] = env_name
            row["size"] = f"{num_goods}x{num_bidders}"
        pv_gt_breakdown_rows = build_pv_gt_error_breakdown(
            pv_gt_rows, env_name, f"{num_goods}x{num_bidders}"
        )

    pv_gt_overall = pv_gt_breakdown_rows[0] if pv_gt_breakdown_rows else {}

    summary = {
        "environment": env_name,
        "size": f"{num_goods}x{num_bidders}",
        "num_goods": num_goods,
        "num_bidders": num_bidders,
        "full_info_optimum_welfare": optimum,
        "num_winners": num_winners,
        "largest_winner_welfare_share": largest_share,
        "avg_winning_bundle_size": avg_winning_bundle_size,
        "num_contested_goods": num_contested_goods,
        "mean_bidders_positive_per_good": statistics.fmean(positive_per_good.values()),
        "min_bidders_positive_per_good": min(positive_per_good.values()),
        "max_bidders_positive_per_good": max(positive_per_good.values()),
        "avg_pairwise_top_goods_overlap": avg_pairwise_overlap,
        "share_top10_bundles_size_ge4": share_top10_size_ge4,
        "avg_complement_contribution_share": avg_complement_share,
        "max_complement_contribution_share": max_complement_share,
        "avg_saturation_penalty_contribution": avg_saturation_contrib,
        "num_budget_cap_binding_bidders": num_budget_cap_binding,
        "value_min_positive": value_min,
        "value_median_positive": value_median,
        "value_mean_positive": value_mean,
        "value_max_positive": value_max,
        "reseller_id": reseller_id,
        "reseller_count_in_top20_global": reseller_count_in_top,
        "reseller_value_share_in_top20_global": reseller_value_share_in_top,
        "pv_gt_log_available": bool(pv_gt_rows) or (compute_pv_gt and log_dir is not None and log_dir.exists()),
        "pv_gt_num_matched_bundles": len(pv_gt_rows),
        "pv_gt_mean_signed_error": pv_gt_overall.get("mean_signed_error"),
        "pv_gt_mean_abs_error": pv_gt_overall.get("mean_abs_error"),
        "pv_gt_median_abs_error": pv_gt_overall.get("median_abs_error"),
        "pv_gt_mean_abs_pct_diff": pv_gt_overall.get("mean_abs_pct_error"),
        "pv_gt_scale_ratio_mean": pv_gt_overall.get("mean_ratio"),
        "pv_gt_scale_ratio_median": pv_gt_overall.get("median_ratio"),
    }

    return {
        "summary": summary,
        "per_good_rows": per_good_rows,
        "allocation_rows": allocation_rows,
        "pv_gt_rows": pv_gt_rows,
        "pv_gt_breakdown_rows": pv_gt_breakdown_rows,
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(x, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    if pct:
        return f"{x:.1%}"
    if isinstance(x, float):
        return f"{x:,.2f}"
    return str(x)


def _cell(field: str, pct: bool = False):
    def _render(row: dict) -> str:
        return _fmt(row.get(field), pct=pct)
    return _render


def _cell_contested(row: dict) -> str:
    return f"{row['num_contested_goods']}/{row['num_goods']}"


def _cell_positive_per_good(row: dict) -> str:
    return (
        f"{_fmt(row['mean_bidders_positive_per_good'])} "
        f"({row['min_bidders_positive_per_good']}-{row['max_bidders_positive_per_good']})"
    )


def _cell_complement(row: dict) -> str:
    return (
        f"{_fmt(row['avg_complement_contribution_share'], pct=True)} / "
        f"{_fmt(row['max_complement_contribution_share'], pct=True)}"
    )


def _cell_value_scale(row: dict) -> str:
    return (
        f"{_fmt(row['value_min_positive'])} / {_fmt(row['value_median_positive'])} / "
        f"{_fmt(row['value_mean_positive'])} / {_fmt(row['value_max_positive'])}"
    )


def _cell_reseller_id(row: dict) -> str:
    return f"`{row['reseller_id']}`"


def _cell_pv_gt(row: dict) -> str:
    if row.get("pv_gt_log_available"):
        return f"{row['pv_gt_num_matched_bundles']} / {_fmt(row['pv_gt_mean_abs_pct_diff'], pct=True)}"
    return "n/a"


def _cell_pv_gt_scale(row: dict) -> str:
    if row.get("pv_gt_log_available"):
        return f"{_fmt(row['pv_gt_scale_ratio_mean'])} / {_fmt(row['pv_gt_scale_ratio_median'])}"
    return "n/a"


# (label, per-environment cell renderer) -- data-driven so the per-size table
# scales to any number of environments (manual/generated/calibrated/...).
_METRIC_ROWS: list[tuple[str, "callable"]] = [
    ("Full-info optimum welfare", _cell("full_info_optimum_welfare")),
    ("Winner count", _cell("num_winners")),
    ("Largest winner welfare share", _cell("largest_winner_welfare_share", pct=True)),
    ("Avg winning bundle size", _cell("avg_winning_bundle_size")),
    ("Contested goods (>=2 bidders w/ positive value)", _cell_contested),
    ("Bidders w/ positive value per good (mean / min-max)", _cell_positive_per_good),
    (f"Avg pairwise overlap, top-{TOP_K_GOODS_PER_BIDDER} valued goods", _cell("avg_pairwise_top_goods_overlap", pct=True)),
    (f"Share of top-{TOP_N_GLOBAL} true bundles with size >= 4", _cell("share_top10_bundles_size_ge4", pct=True)),
    ("Complement contribution share (avg / max)", _cell_complement),
    ("Avg saturation-penalty contribution", _cell("avg_saturation_penalty_contribution", pct=True)),
    ("Budget-cap-binding bidders", _cell("num_budget_cap_binding_bidders")),
    ("Positive bundle value: min / median / mean / max", _cell_value_scale),
    ("Reseller bidder id", _cell_reseller_id),
    (f"Reseller count in top-{TOP_N_HIGH_VALUE} global bundles", _cell("reseller_count_in_top20_global")),
    (f"Reseller value share within top-{TOP_N_HIGH_VALUE} global bundles", _cell("reseller_value_share_in_top20_global", pct=True)),
    ("PV vs. GT (cached log): matched bundles / mean abs % diff", _cell_pv_gt),
    ("PV vs. GT scale ratio: mean(PV)/mean(GT) / median(PV)/median(GT)", _cell_pv_gt_scale),
]

_ENV_LABELS = {"manual": "Manual", "generated": "Generated", "calibrated": "Calibrated"}


def _env_label(env_name: str) -> str:
    return _ENV_LABELS.get(env_name, env_name.replace("_", " ").title())


def _delta(by_key: dict, env_a: str, env_b: str, field: str, pct: bool = False) -> str:
    parts = []
    for num_goods, num_bidders in SIZES:
        size_key = f"{num_goods}x{num_bidders}"
        av = by_key[(env_a, size_key)][field]
        bv = by_key[(env_b, size_key)][field]
        parts.append(f"{size_key}: {_fmt(av, pct)} -> {_fmt(bv, pct)}")
    return "; ".join(parts)


def build_markdown_report(all_summaries: list[dict], environment_order: list[str]) -> str:
    by_key = {(s["environment"], s["size"]): s for s in all_summaries}

    lines: list[str] = []
    lines.append("# PC-build environment comparison")
    lines.append("")
    lines.append(
        "Deterministic diagnostic only -- no live LLM/API calls were made. "
        "All figures come from the frozen `ScenarioProfileSpec` files evaluated "
        "against the true valuation tables and the exact WDP ILP solver."
    )
    lines.append("")
    if "manual" in environment_order:
        lines.append(
            "- **manual**: `pc_build_profiles_v0_manual.json`-equivalent spec, built "
            "in-memory from the hard-coded archetype builders in "
            "`auctionlab.instances.structured` (export seed 0). Reseller archetype: `pc_reseller`."
        )
    if "generated" in environment_order:
        lines.append(
            "- **generated**: the LLM-generated spec passed via `--generated-spec` "
            "(default `pc_build_profiles_gemini_trial_final.json`). Reseller archetype: `reseller_pro`."
        )
    if "calibrated" in environment_order:
        lines.append(
            "- **calibrated**: the candidate spec passed via `--calibrated-spec` -- typically a "
            "multiplier-adjusted version of the generated spec produced by "
            "`calibrate_generated_pc_build_spec.py`."
        )
    lines.append("")

    for num_goods, num_bidders in SIZES:
        size_key = f"{num_goods}x{num_bidders}"
        rows_for_size = {env: by_key[(env, size_key)] for env in environment_order}

        lines.append(f"## {size_key}")
        lines.append("")
        lines.append("| Metric | " + " | ".join(_env_label(e) for e in environment_order) + " |")
        lines.append("|---|" + "---:|" * len(environment_order))
        for label, render in _METRIC_ROWS:
            cells = " | ".join(render(rows_for_size[e]) for e in environment_order)
            lines.append(f"| {label} | {cells} |")
        lines.append("")

    if "manual" in environment_order and "generated" in environment_order:
        lines.append("## Why the generated environment is harder or easier than the manual one")
        lines.append("")

        lines.append(
            "- **Concentration.** Largest-winner welfare share (" +
            _delta(by_key, "manual", "generated", "largest_winner_welfare_share", pct=True) +
            ") and winner count (" + _delta(by_key, "manual", "generated", "num_winners") + ") indicate whether "
            "welfare concentrates in one dominant bidder or spreads across several -- a lower largest-winner share "
            "and higher winner count generally make an environment *harder* for proxy mechanisms, since more "
            "bidders are actually competing for the outcome rather than a single archetype sweeping the auction."
        )
        lines.append(
            "- **Contestedness / overlap.** Contested goods (" +
            _delta(by_key, "manual", "generated", "num_contested_goods") +
            ") and pairwise top-valued-good overlap (" +
            _delta(by_key, "manual", "generated", "avg_pairwise_top_goods_overlap", pct=True) +
            ") measure how much bidders' interests collide. Higher overlap means more real substitution/competition "
            "pressure the mechanism must resolve correctly, rather than bidders simply partitioning the goods by "
            "disjoint interest."
        )
        lines.append(
            "- **Combinatorial structure.** Share of top-" + str(TOP_N_GLOBAL) + " bundles with size >= 4 (" +
            _delta(by_key, "manual", "generated", "share_top10_bundles_size_ge4", pct=True) +
            ") and complement contribution share (avg: " +
            _delta(by_key, "manual", "generated", "avg_complement_contribution_share", pct=True) +
            ") capture how much value depends on winning larger, synergistic bundles rather than one or two items "
            "-- a domain that rewards big bundles is harder for an LLM proxy (or a mechanism) that under-elicits "
            "complementarities."
        )
        lines.append(
            "- **Budget/saturation friction.** Saturation-penalty contribution (avg: " +
            _delta(by_key, "manual", "generated", "avg_saturation_penalty_contribution", pct=True) +
            ") and budget-cap-binding bidder counts (" +
            _delta(by_key, "manual", "generated", "num_budget_cap_binding_bidders") + ") reflect how often a "
            "bidder's *reported* preference is actually constrained by a cap rather than by raw item value -- more "
            "binding constraints add another dimension a proxy has to get right beyond simple item valuation."
        )
        lines.append(
            "- **Value-scale spread.** Positive-bundle-value spread (min/median/mean/max, see per-size tables "
            "above) shows whether values cluster tightly or vary by orders of magnitude; a wider spread makes "
            "rank-preserving approximation easier even when absolute-dollar accuracy is poor, while a tight spread "
            "punishes small errors more in relative terms."
        )
        lines.append(
            "- **Reseller behavior.** The reseller archetype's presence in the top-" + str(TOP_N_HIGH_VALUE) +
            " global bundles (count: " + _delta(by_key, "manual", "generated", "reseller_count_in_top20_global") +
            "; value share: " +
            _delta(by_key, "manual", "generated", "reseller_value_share_in_top20_global", pct=True) +
            ") is a useful tracer for whether \"near-additive, buy-everything-cheaply\" bidding remains a dominant "
            "strategy in the generated spec the way it was hand-tuned to be in the manual archetypes."
        )
        for num_goods, num_bidders in SIZES:
            size_key = f"{num_goods}x{num_bidders}"
            g = by_key[("generated", size_key)]
            if g["pv_gt_log_available"] and g["pv_gt_num_matched_bundles"]:
                lines.append(
                    f"- **Elicitation fidelity at {size_key} (generated only).** Against cached "
                    f"`calls.jsonl` logs, {g['pv_gt_num_matched_bundles']} initial-candidate-bundle "
                    f"valuations had a matching ground-truth entry, with a mean absolute percentage "
                    f"difference of {_fmt(g['pv_gt_mean_abs_pct_diff'], pct=True)} between the LLM's "
                    "provisional value and the true bundle value, and a mean signed error of "
                    f"{_fmt(g['pv_gt_mean_signed_error'])} (positive = LLM overestimates on average). "
                    f"The reported/true scale ratio is {_fmt(g['pv_gt_scale_ratio_mean'])} "
                    f"(mean PV / mean GT) and {_fmt(g['pv_gt_scale_ratio_median'])} (median PV / median GT) "
                    "-- a ratio far from 1.0 indicates a systematic scale error (the proxy consistently "
                    "over- or under-values bundles as a class), not just noisy per-bundle estimates. "
                    "See `pv_gt_error_breakdown.csv` for the same stats split by bidder, bundle size, "
                    "and allocated-vs-non-allocated bundle."
                )
            else:
                lines.append(
                    f"- **Elicitation fidelity at {size_key} (generated only).** No cached `calls.jsonl` "
                    "log exists at this size, so item 15 could not be computed here (see `logs/` -- only "
                    "6x6 and 8x8 spec_gemini_trial runs are cached)."
                )
        lines.append("")

    if "calibrated" in environment_order:
        lines.append("## Calibrated candidate vs. generated and manual")
        lines.append("")
        lines.append(
            "- **Optimum welfare scale.** manual -> generated (" +
            _delta(by_key, "manual", "generated", "full_info_optimum_welfare") +
            "); generated -> calibrated (" +
            _delta(by_key, "generated", "calibrated", "full_info_optimum_welfare") +
            "). Calibration is working on the welfare-scale axis if the second gap shrinks the first."
        )
        lines.append(
            "- **Reseller dominance.** Reseller top-" + str(TOP_N_HIGH_VALUE) +
            " count, generated -> calibrated (" +
            _delta(by_key, "generated", "calibrated", "reseller_count_in_top20_global") +
            "); value share, generated -> calibrated (" +
            _delta(by_key, "generated", "calibrated", "reseller_value_share_in_top20_global", pct=True) + ")."
        )
        lines.append(
            "- **Positive-bundle-value scale.** Median positive bundle value: manual -> generated (" +
            _delta(by_key, "manual", "generated", "value_median_positive") +
            "); generated -> calibrated (" +
            _delta(by_key, "generated", "calibrated", "value_median_positive") + ")."
        )
        lines.append(
            "- **Concentration.** Largest-winner welfare share, generated -> calibrated (" +
            _delta(by_key, "generated", "calibrated", "largest_winner_welfare_share", pct=True) + ")."
        )
        lines.append("")
    lines.append(
        "## Definitions / methodology notes (read before citing numbers above)"
    )
    lines.append("")
    lines.append(
        f"- \"Top-valued goods\" per bidder (item 7) = the top {TOP_K_GOODS_PER_BIDDER} goods by singleton "
        "value (ties broken by catalog order), restricted to goods with strictly positive singleton value. "
        "Overlap between two bidders is the Jaccard index of their top-good sets, averaged over all bidder pairs."
    )
    lines.append(
        f"- \"Top-{TOP_N_GLOBAL} true bundles\" (item 8) and \"top-{TOP_N_HIGH_VALUE} high-valued bundles\" "
        "(item 13) are both taken from a single global ranking of every (bidder, bundle, true value) triple "
        "across all bidders, sorted by value descending -- a fixed count, not a percentile, so it is comparable "
        "across sizes with very different bundle-table sizes (2^n - 1 per bidder)."
    )
    lines.append(
        "- \"Contested good\" (item 5) = a good for which at least 2 bidders have strictly positive singleton value."
    )
    lines.append(
        "- Complement/saturation/budget-cap bindingness (items 9-11) are computed by comparing each bidder's actual "
        "grand-bundle value against a counterfactual profile with that one constraint removed, exactly as in "
        "`scripts/validate_scenario_spec.py`."
    )
    lines.append(
        "- Item 15 (PV vs. GT) reads *only* pre-existing `logs/spec_gemini_trial_{size}_seed0/calls.jsonl` files "
        "from a prior live run; it makes no new LLM/API call. `proxy_provisional_valuations` entries hold the "
        "LLM's initial-candidate-bundle value estimates; `value_query_gt` entries hold the harness's logged true "
        "value for the same (bidder, bundle) pair. Only bundles with both an entry are compared."
    )
    lines.append(
        "- `pv_gt_error_breakdown.csv` groups the same matched-bundle rows by bidder, by bundle size, and by "
        "whether the candidate bundle is exactly the bidder's full-info winning bundle (`allocated` vs "
        "`non_allocated`) -- most PV candidate bundles are small interest-derived shortlists rather than the "
        "eventual winner, so an empty/small `allocated` group is expected. Each group reports mean signed error, "
        "mean/median absolute error, mean absolute percentage error, and the mean(PV)/mean(GT) and "
        "median(PV)/median(GT) scale ratios."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--generated-spec",
        type=Path,
        default=GENERATED_SPEC_PATH,
        help=f"Path to the LLM-generated ScenarioProfileSpec JSON (default: {GENERATED_SPEC_PATH}).",
    )
    parser.add_argument(
        "--calibrated-spec",
        type=Path,
        default=None,
        help=(
            "Optional path to a third ScenarioProfileSpec JSON (e.g. a calibrated candidate "
            "frozen from calibrate_generated_pc_build_spec.py) to include as a 'calibrated' "
            "environment alongside manual/generated. Omit to compare just manual vs. generated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to write CSV/markdown outputs to (default: {OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    manual_spec = build_manual_spec()
    generated_spec = load_scenario_profile_spec(args.generated_spec)

    specs: dict[str, ScenarioProfileSpec] = {"manual": manual_spec, "generated": generated_spec}
    if args.calibrated_spec is not None:
        specs["calibrated"] = load_scenario_profile_spec(args.calibrated_spec)

    all_summaries: list[dict] = []
    all_per_good_rows: list[dict] = []
    all_allocation_rows: list[dict] = []
    all_pv_gt_rows: list[dict] = []
    all_pv_gt_breakdown_rows: list[dict] = []

    for env_name, spec in specs.items():
        for num_goods, num_bidders in SIZES:
            result = compute_size_metrics(env_name, spec, num_goods, num_bidders)
            all_summaries.append(result["summary"])
            all_per_good_rows.extend(result["per_good_rows"])
            all_allocation_rows.extend(result["allocation_rows"])
            all_pv_gt_rows.extend(result["pv_gt_rows"])
            all_pv_gt_breakdown_rows.extend(result["pv_gt_breakdown_rows"])
            print(f"  computed {env_name} {num_goods}x{num_bidders}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "environment_size_summary.csv", all_summaries)
    _write_csv(output_dir / "per_good_positive_bidders.csv", all_per_good_rows)
    _write_csv(output_dir / "full_info_allocation.csv", all_allocation_rows)
    _write_csv(output_dir / "pv_vs_gt_comparison.csv", all_pv_gt_rows)
    _write_csv(output_dir / "pv_gt_error_breakdown.csv", all_pv_gt_breakdown_rows)

    report = build_markdown_report(all_summaries, list(specs.keys()))
    (output_dir / "comparison_report.md").write_text(report, encoding="utf-8")

    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
