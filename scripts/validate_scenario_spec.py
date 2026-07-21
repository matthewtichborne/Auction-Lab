#!/usr/bin/env python3
"""Validate a frozen PC-build ``ScenarioProfileSpec`` and report diagnostics.

Performs schema validation, a profile-level summary, and economic
diagnostics (welfare, allocation shape, free-disposal repair, saturation
and budget-cap bindingness, complement/substitute structure, allocation
concentration, and — optionally — a fully deterministic full-information
mechanism benchmark) for several generated environment sizes. Writes a
single JSON report and evaluates it against configurable acceptance
criteria.

Nothing in this script makes an LLM/API call. The mechanism benchmark uses
the true full valuation tables directly (via
:func:`auctionlab.instances.base.make_demand_oracle`), not LLM proxies.

Usage::

    ./venv/bin/python scripts/validate_scenario_spec.py \\
        scenarios/pc_build_v1/pc_build_profiles_v0_manual.json \\
        --output scenarios/pc_build_v1/validation_report_v0_manual.json

    # Include the (slower) deterministic sealed + clock k=1,2,3 benchmark:
    ./venv/bin/python scripts/validate_scenario_spec.py \\
        scenarios/pc_build_v1/pc_build_profiles_v0_manual.json \\
        --output scenarios/pc_build_v1/validation_report_v0_manual.json \\
        --mechanism-benchmark

    # Override acceptance thresholds:
    ./venv/bin/python scripts/validate_scenario_spec.py \\
        scenarios/pc_build_v1/pc_build_profiles_v0_manual.json \\
        --output report.json --thresholds my_thresholds.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from pathlib import Path

from auctionlab.auctions.clock import (
    ClockConfig,
    compute_excess_demand,
    finalize_from_supplementary_vcg,
    has_no_excess_demand,
    run_ascending_clock_with_supplementary,
)
from auctionlab.auctions.sealed_vcg import run_sealed_xor_vcg
from auctionlab.instances.base import AuctionInstance, make_demand_oracle
from auctionlab.instances.scenario_spec import ScenarioProfileSpec, validate_spec_dict
from auctionlab.instances.structured import (
    BidderPreferenceProfile,
    all_nonempty_bundles,
    enforce_monotonicity,
    value_bundle,
)
from auctionlab.instances.structured_spec import (
    SELECTION_POLICIES,
    make_pc_build_scenario_from_spec,
    profile_from_spec,
)
from auctionlab.solvers.wdp_ilp import solve_wdp_xor_ilp

DEFAULT_SIZES: list[tuple[int, int]] = [(4, 4), (6, 6), (8, 8), (10, 10)]
_TOL = 1e-6


# ---------------------------------------------------------------------------
# Profile summary
# ---------------------------------------------------------------------------

def build_profile_summary(spec: ScenarioProfileSpec) -> dict:
    all_base_values = [
        v for bp in spec.bidder_profiles for v in bp.base_values.values()
    ]
    sub_counts = [len(bp.substitute_groups) for bp in spec.bidder_profiles]
    comp_counts = [len(bp.complement_groups) for bp in spec.bidder_profiles]
    budget_los = [bp.budget_range[0] for bp in spec.bidder_profiles]
    budget_his = [bp.budget_range[1] for bp in spec.bidder_profiles]

    return {
        "num_goods": len(spec.goods),
        "num_bidders": len(spec.bidder_profiles),
        "average_base_value": statistics.fmean(all_base_values) if all_base_values else 0.0,
        "min_base_value": min(all_base_values) if all_base_values else None,
        "max_base_value": max(all_base_values) if all_base_values else None,
        "average_complement_groups_per_bidder": statistics.fmean(comp_counts) if comp_counts else 0.0,
        "average_substitute_groups_per_bidder": statistics.fmean(sub_counts) if sub_counts else 0.0,
        "average_budget_low": statistics.fmean(budget_los) if budget_los else 0.0,
        "average_budget_high": statistics.fmean(budget_his) if budget_his else 0.0,
        "num_bidders_with_saturation": sum(
            1 for bp in spec.bidder_profiles if bp.saturation_start is not None
        ),
        "num_bidders_with_budget_cap": sum(
            1 for bp in spec.bidder_profiles if bp.budget_cap is not None
        ),
    }


# ---------------------------------------------------------------------------
# Free-disposal repair: share + magnitude
# ---------------------------------------------------------------------------

def _repair_stats(items: list[str], profile: BidderPreferenceProfile) -> dict:
    """Count and size the monotonicity repairs needed for one bidder's table."""
    bundles = all_nonempty_bundles(items)
    raw = {b: value_bundle(profile, b) for b in bundles}
    repaired = enforce_monotonicity(items, raw)

    magnitudes = [
        repaired[b] - raw[b] for b in bundles if repaired[b] > raw[b] + _TOL
    ]
    return {
        "count": len(magnitudes),
        "share": len(magnitudes) / len(bundles) if bundles else 0.0,
        "mean_magnitude": statistics.fmean(magnitudes) if magnitudes else 0.0,
        "max_magnitude": max(magnitudes) if magnitudes else 0.0,
    }


# ---------------------------------------------------------------------------
# Saturation / budget-cap bindingness
# ---------------------------------------------------------------------------

def _binding_stats(items: list[str], profile: BidderPreferenceProfile) -> dict:
    """Whether saturation / budget cap actually changes the grand-bundle value.

    Computed by comparing the actual grand-bundle value against a
    counterfactual profile with just that one constraint removed. If both
    constraints are active simultaneously on the same bidder, removing
    saturation alone may still be masked by an active budget cap (and vice
    versa) — a known simplification, harmless for specs (like v0) where no
    bidder combines both.
    """
    grand = frozenset(items)
    actual = value_bundle(profile, grand)

    saturation_binding = False
    if profile.saturation_start is not None:
        unsaturated = dataclasses.replace(profile, saturation_start=None, saturation_penalty=0.0)
        saturation_binding = actual < value_bundle(unsaturated, grand) - _TOL

    budget_cap_binding = False
    if profile.budget_cap is not None:
        uncapped = dataclasses.replace(profile, budget_cap=None)
        budget_cap_binding = actual < value_bundle(uncapped, grand) - _TOL

    return {
        "has_saturation": profile.saturation_start is not None,
        "saturation_binding": saturation_binding,
        "has_budget_cap": profile.budget_cap is not None,
        "budget_cap_binding": budget_cap_binding,
    }


# ---------------------------------------------------------------------------
# Complement contribution
# ---------------------------------------------------------------------------

def _complement_contribution_share(profile: BidderPreferenceProfile, items: list[str]) -> float:
    """Share of the grand-bundle value attributable to complement bonuses."""
    grand = frozenset(items)
    grand_value = value_bundle(profile, grand)
    if grand_value <= 0:
        return 0.0
    total_bonus = sum(
        cg.bonus for cg in profile.complement_groups if cg.items.issubset(grand)
    )
    return total_bonus / grand_value


# ---------------------------------------------------------------------------
# Substitute strength
# ---------------------------------------------------------------------------

def _substitute_backup_factors(profiles: list[BidderPreferenceProfile]) -> list[float]:
    return [sg.backup_factor for p in profiles for sg in p.substitute_groups]


# ---------------------------------------------------------------------------
# Deterministic full-info mechanism benchmark (no LLM calls)
# ---------------------------------------------------------------------------

def compute_mechanism_benchmark_diagnostics(
    instance: AuctionInstance,
    global_optimum_welfare: float,
    *,
    clock_price_step: float = 50.0,
    clock_max_rounds: int = 40,
    clock_reserve: float = 0.0,
) -> dict:
    """Sealed-bid VCG and ascending-clock (top_k=1,2,3) full-info benchmarks.

    Both mechanisms are run against the *true* full valuation table (via a
    deterministic demand oracle / direct XOR bids) — no LLM proxy is
    involved anywhere in this computation.
    """
    items = list(instance.items)
    bidder_ids = list(instance.bidder_ids)

    def _efficiency(welfare: float) -> float:
        return welfare / global_optimum_welfare if global_optimum_welfare > 0 else 0.0

    sealed = run_sealed_xor_vcg(items, instance.to_xor_bids())
    sealed_result = {
        "welfare": sealed.welfare,
        "efficiency": _efficiency(sealed.welfare),
        "total_payments": sum(sealed.payments.values()),
        "num_winners": sum(1 for bundle in sealed.allocation.values() if bundle),
    }

    clock_results: dict = {}
    for top_k in (1, 2, 3):
        oracle = make_demand_oracle(instance, top_k=top_k)
        cfg = ClockConfig(
            max_rounds=clock_max_rounds, price_step=clock_price_step, reserve=clock_reserve
        )
        state, _provisional = run_ascending_clock_with_supplementary(
            items, bidder_ids, oracle, cfg
        )
        last_demands = state.history[-1].demands if state.history else {}
        converged = has_no_excess_demand(compute_excess_demand(items, last_demands))

        outcome = finalize_from_supplementary_vcg(items, state.supplementary)
        clock_results[f"k{top_k}"] = {
            "welfare": outcome.welfare,
            "efficiency": _efficiency(outcome.welfare),
            "converged": converged,
            "num_rounds": state.round_idx + 1,
            "total_payments": sum(outcome.payments.values()),
        }

    return {"sealed_vcg": sealed_result, "clock": clock_results}


# ---------------------------------------------------------------------------
# Economic diagnostics (per size)
# ---------------------------------------------------------------------------

def compute_size_diagnostics(
    spec: ScenarioProfileSpec,
    num_goods: int,
    num_bidders: int,
    *,
    seed: int = 0,
    selection_policy: str = "prefix",
    mechanism_benchmark: bool = False,
    clock_price_step: float = 50.0,
    clock_max_rounds: int = 40,
    clock_reserve: float = 0.0,
) -> dict:
    scenario = make_pc_build_scenario_from_spec(
        spec, num_goods, num_bidders, seed=seed, selection_policy=selection_policy
    )
    instance = scenario.instance
    items = list(instance.items)
    bidder_ids = list(instance.bidder_ids)
    items_set = set(items)

    wdp = solve_wdp_xor_ilp(items, instance.to_xor_bids())
    winners = {b: bundle for b, bundle in wdp.allocation.items() if bundle}
    winner_sizes = {b: len(bundle) for b, bundle in winners.items()}

    per_winner_value = {
        b: instance.valuations[b].get(bundle, 0.0) for b, bundle in winners.items()
    }
    largest_share = (
        max(per_winner_value.values()) / wdp.welfare if wdp.welfare > 0 and per_winner_value else 0.0
    )
    welfare_hhi = (
        sum((v / wdp.welfare) ** 2 for v in per_winner_value.values())
        if wdp.welfare > 0
        else 0.0
    )
    num_items_allocated = sum(winner_sizes.values())

    bidders_by_id = {b.bidder_id: b for b in spec.bidder_profiles}
    top_sizes: list[int] = []
    grand_bundle_is_unique_top = 0
    repair_stats_per_bidder: dict[str, dict] = {}
    total_sub_groups = 0
    total_comp_groups = 0
    binding_per_bidder: dict[str, dict] = {}
    complement_shares: list[float] = []
    profiles: list[BidderPreferenceProfile] = []

    for bidder_id in bidder_ids:
        table = instance.valuations[bidder_id]
        max_val = max(table.values())
        top_size = min(len(b) for b, v in table.items() if v == max_val)
        top_sizes.append(top_size)
        if top_size == num_goods:
            grand_bundle_is_unique_top += 1

        profile = profile_from_spec(bidders_by_id[bidder_id], items_set)
        profiles.append(profile)
        repair_stats_per_bidder[bidder_id] = _repair_stats(items, profile)
        total_sub_groups += len(profile.substitute_groups)
        total_comp_groups += len(profile.complement_groups)
        binding_per_bidder[bidder_id] = _binding_stats(items, profile)
        complement_shares.append(_complement_contribution_share(profile, items))

    repair_counts = [s["count"] for s in repair_stats_per_bidder.values()]
    repair_magnitudes = [
        s["mean_magnitude"] for s in repair_stats_per_bidder.values() if s["count"] > 0
    ]
    table_size = 2 ** num_goods - 1
    total_repairable = table_size * len(bidder_ids)

    backup_factors = _substitute_backup_factors(profiles)

    result: dict = {
        "full_valuation_table_size_per_bidder": table_size,
        "global_optimum_welfare": wdp.welfare,
        "num_winners": len(winners),
        "winner_bundle_sizes": winner_sizes,
        "largest_winner_welfare_share": largest_share,
        "is_degenerate_single_winner": len(winners) == 1,
        "welfare_hhi": welfare_hhi,
        "num_items_allocated": num_items_allocated,
        "num_items_unallocated": num_goods - num_items_allocated,
        "share_items_allocated": num_items_allocated / num_goods if num_goods else 0.0,
        "mean_winner_bundle_size": statistics.fmean(winner_sizes.values()) if winner_sizes else 0.0,
        "max_winner_bundle_size": max(winner_sizes.values()) if winner_sizes else 0,
        "free_disposal_repair_count_total": sum(repair_counts),
        "free_disposal_repair_share_total": (
            sum(repair_counts) / total_repairable if total_repairable else 0.0
        ),
        "free_disposal_repair_mean_magnitude": (
            statistics.fmean(repair_magnitudes) if repair_magnitudes else 0.0
        ),
        "free_disposal_repair_max_magnitude": (
            max((s["max_magnitude"] for s in repair_stats_per_bidder.values()), default=0.0)
        ),
        "free_disposal_repair_count_per_bidder": dict(zip(bidder_ids, repair_counts)),
        "free_disposal_repair_share_per_bidder": {
            bidder_id: s["share"] for bidder_id, s in repair_stats_per_bidder.items()
        },
        "mean_top_valued_bundle_size": statistics.fmean(top_sizes) if top_sizes else 0.0,
        "num_bidders_with_grand_bundle_as_unique_top": grand_bundle_is_unique_top,
        "share_bidders_with_grand_bundle_as_unique_top": (
            grand_bundle_is_unique_top / len(bidder_ids) if bidder_ids else 0.0
        ),
        "complement_groups_available": total_comp_groups,
        "substitute_groups_available": total_sub_groups,
        "num_bidders_with_saturation_binding": sum(
            1 for s in binding_per_bidder.values() if s["saturation_binding"]
        ),
        "num_bidders_with_budget_cap_binding": sum(
            1 for s in binding_per_bidder.values() if s["budget_cap_binding"]
        ),
        "saturation_budget_binding_per_bidder": binding_per_bidder,
        "mean_complement_contribution_share": (
            statistics.fmean(complement_shares) if complement_shares else 0.0
        ),
        "max_complement_contribution_share": max(complement_shares, default=0.0),
        "num_bidders_with_complement_groups": sum(
            1 for p in profiles if p.complement_groups
        ),
        "mean_substitute_backup_factor": (
            statistics.fmean(backup_factors) if backup_factors else None
        ),
        "min_substitute_backup_factor": min(backup_factors, default=None),
        "max_substitute_backup_factor": max(backup_factors, default=None),
        "num_strong_substitute_groups": sum(1 for f in backup_factors if f < 0.2),
        "num_weak_substitute_groups": sum(1 for f in backup_factors if f >= 0.7),
    }

    if mechanism_benchmark:
        result["mechanism_benchmark"] = compute_mechanism_benchmark_diagnostics(
            instance,
            wdp.welfare,
            clock_price_step=clock_price_step,
            clock_max_rounds=clock_max_rounds,
            clock_reserve=clock_reserve,
        )

    return result


def build_economic_diagnostics(
    spec: ScenarioProfileSpec,
    sizes: list[tuple[int, int]],
    *,
    seed: int = 0,
    selection_policy: str = "prefix",
    mechanism_benchmark: bool = False,
    clock_price_step: float = 50.0,
    clock_max_rounds: int = 40,
    clock_reserve: float = 0.0,
) -> dict:
    diagnostics: dict = {}
    for num_goods, num_bidders in sizes:
        key = f"{num_goods}x{num_bidders}"
        if num_goods > len(spec.goods) or num_bidders > len(spec.bidder_profiles):
            diagnostics[key] = {
                "skipped": True,
                "reason": (
                    f"spec has only {len(spec.goods)} goods and "
                    f"{len(spec.bidder_profiles)} bidder profiles"
                ),
            }
            continue
        diagnostics[key] = compute_size_diagnostics(
            spec,
            num_goods,
            num_bidders,
            seed=seed,
            selection_policy=selection_policy,
            mechanism_benchmark=mechanism_benchmark,
            clock_price_step=clock_price_step,
            clock_max_rounds=clock_max_rounds,
            clock_reserve=clock_reserve,
        )
    return diagnostics


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------

def default_acceptance_thresholds() -> dict:
    """Default (lenient) acceptance thresholds.

    Concentration/structure checks (largest-winner share, degenerate
    single-winner, complement/substitute availability, grand-bundle
    dominance) only apply to sizes with at least
    ``min_bidders_for_concentration_checks`` bidders — small environments
    (e.g. 4x4) naturally tend toward a single dominant winner and sparse
    group structure, which is not itself a defect.
    """
    return {
        "min_bidders_for_concentration_checks": 6,
        "max_largest_winner_welfare_share": 0.85,
        "disallow_degenerate_single_winner": True,
        "max_free_disposal_repair_share": 0.6,
        "min_complement_groups_available": 1,
        "min_substitute_groups_available": 1,
        "max_share_bidders_with_grand_bundle_as_unique_top": 0.9,
        "max_welfare_hhi": 1.0,
    }


def _check(checks: list[dict], *, size: str | None, name: str, actual, threshold, passed: bool) -> None:
    checks.append(
        {
            "size": size,
            "name": name,
            "passed": bool(passed),
            "actual": actual,
            "threshold": threshold,
        }
    )


def evaluate_acceptance(report: dict, thresholds: dict) -> dict:
    """Evaluate a validation report against acceptance thresholds.

    Returns ``{"passed": bool, "thresholds": thresholds, "checks": [...]}``.
    Never raises; a schema-invalid report simply fails the one schema check.
    """
    checks: list[dict] = []

    schema_valid = report.get("schema_validation", {}).get("valid", False)
    _check(checks, size=None, name="schema_valid", actual=schema_valid, threshold=True, passed=schema_valid)

    if not schema_valid:
        return {"passed": False, "thresholds": thresholds, "checks": checks}

    min_bidders = thresholds["min_bidders_for_concentration_checks"]

    for size_key, diag in report.get("economic_diagnostics", {}).items():
        if diag.get("skipped"):
            continue

        _check(
            checks,
            size=size_key,
            name="free_disposal_repair_share",
            actual=diag["free_disposal_repair_share_total"],
            threshold=thresholds["max_free_disposal_repair_share"],
            passed=diag["free_disposal_repair_share_total"] <= thresholds["max_free_disposal_repair_share"],
        )
        _check(
            checks,
            size=size_key,
            name="welfare_hhi",
            actual=diag["welfare_hhi"],
            threshold=thresholds["max_welfare_hhi"],
            passed=diag["welfare_hhi"] <= thresholds["max_welfare_hhi"],
        )

        num_bidders = int(size_key.split("x")[1])
        if num_bidders < min_bidders:
            continue

        _check(
            checks,
            size=size_key,
            name="largest_winner_welfare_share",
            actual=diag["largest_winner_welfare_share"],
            threshold=thresholds["max_largest_winner_welfare_share"],
            passed=diag["largest_winner_welfare_share"] <= thresholds["max_largest_winner_welfare_share"],
        )
        if thresholds["disallow_degenerate_single_winner"]:
            _check(
                checks,
                size=size_key,
                name="not_degenerate_single_winner",
                actual=diag["is_degenerate_single_winner"],
                threshold=False,
                passed=not diag["is_degenerate_single_winner"],
            )
        _check(
            checks,
            size=size_key,
            name="complement_groups_available",
            actual=diag["complement_groups_available"],
            threshold=thresholds["min_complement_groups_available"],
            passed=diag["complement_groups_available"] >= thresholds["min_complement_groups_available"],
        )
        _check(
            checks,
            size=size_key,
            name="substitute_groups_available",
            actual=diag["substitute_groups_available"],
            threshold=thresholds["min_substitute_groups_available"],
            passed=diag["substitute_groups_available"] >= thresholds["min_substitute_groups_available"],
        )
        _check(
            checks,
            size=size_key,
            name="share_bidders_with_grand_bundle_as_unique_top",
            actual=diag["share_bidders_with_grand_bundle_as_unique_top"],
            threshold=thresholds["max_share_bidders_with_grand_bundle_as_unique_top"],
            passed=(
                diag["share_bidders_with_grand_bundle_as_unique_top"]
                <= thresholds["max_share_bidders_with_grand_bundle_as_unique_top"]
            ),
        )

    return {
        "passed": all(c["passed"] for c in checks),
        "thresholds": thresholds,
        "checks": checks,
    }


def load_thresholds(path: Path | None) -> dict:
    thresholds = default_acceptance_thresholds()
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        thresholds.update(overrides)
    return thresholds


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_validation_report(
    spec_path: Path,
    sizes: list[tuple[int, int]],
    *,
    seed: int = 0,
    selection_policy: str = "prefix",
    thresholds: dict | None = None,
    mechanism_benchmark: bool = False,
    clock_price_step: float = 50.0,
    clock_max_rounds: int = 40,
    clock_reserve: float = 0.0,
) -> dict:
    with open(spec_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    result = validate_spec_dict(raw)
    report: dict = {
        "spec_path": str(spec_path),
        "selection_policy": selection_policy,
        "scenario_seed": seed,
        "schema_validation": {
            "valid": result.valid,
            "errors": result.errors,
        },
    }

    if thresholds is None:
        thresholds = default_acceptance_thresholds()

    if not result.valid:
        report["acceptance"] = evaluate_acceptance(report, thresholds)
        return report

    spec = result.spec
    assert spec is not None
    report["profile_summary"] = build_profile_summary(spec)
    report["economic_diagnostics"] = build_economic_diagnostics(
        spec,
        sizes,
        seed=seed,
        selection_policy=selection_policy,
        mechanism_benchmark=mechanism_benchmark,
        clock_price_step=clock_price_step,
        clock_max_rounds=clock_max_rounds,
        clock_reserve=clock_reserve,
    )
    report["acceptance"] = evaluate_acceptance(report, thresholds)
    return report


# ---------------------------------------------------------------------------
# Multi-seed validation
# ---------------------------------------------------------------------------

_SCALAR_TYPES = (int, float, bool)


def _scalar_fields(d: dict) -> dict:
    """Restrict a diagnostics dict to plain scalar fields (skip nested dicts)."""
    return {k: v for k, v in d.items() if isinstance(v, _SCALAR_TYPES) and not isinstance(v, dict)}


def aggregate_metrics_across_seeds(per_seed_reports: dict[str, dict]) -> dict:
    """Mean/min/max of every scalar economic-diagnostics field, per size, across seeds.

    Sizes skipped in *every* seed are reported as ``{"skipped": True}``; a
    size skipped in only some seeds still aggregates over the seeds where it
    wasn't skipped.
    """
    size_keys: set[str] = set()
    for report in per_seed_reports.values():
        size_keys |= set(report.get("economic_diagnostics", {}))

    aggregate: dict = {}
    for size_key in sorted(size_keys):
        per_seed_scalars: dict[str, dict] = {}
        for seed_str, report in per_seed_reports.items():
            diag = report.get("economic_diagnostics", {}).get(size_key)
            if diag is None or diag.get("skipped"):
                continue
            per_seed_scalars[seed_str] = _scalar_fields(diag)

        if not per_seed_scalars:
            aggregate[size_key] = {"skipped": True}
            continue

        field_names: set[str] = set()
        for fields in per_seed_scalars.values():
            field_names |= set(fields)

        field_stats: dict = {}
        for field_name in sorted(field_names):
            values_by_seed = {
                seed_str: fields[field_name]
                for seed_str, fields in per_seed_scalars.items()
                if field_name in fields
            }
            values = list(values_by_seed.values())
            field_stats[field_name] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "values_by_seed": values_by_seed,
            }
        aggregate[size_key] = field_stats

    return aggregate


def build_multi_seed_report(
    spec_path: Path,
    sizes: list[tuple[int, int]],
    seeds: list[int],
    *,
    selection_policy: str = "prefix",
    thresholds: dict | None = None,
    mechanism_benchmark: bool = False,
    clock_price_step: float = 50.0,
    clock_max_rounds: int = 40,
    clock_reserve: float = 0.0,
) -> dict:
    """Run :func:`build_validation_report` once per seed and aggregate.

    Most useful with ``selection_policy in ("seeded_sample", "stratified")``,
    where different seeds actually select different goods/bidders subsets;
    under ``"prefix"`` every seed produces an identical report (prefix
    selection ignores the seed), so aggregation is a no-op but still valid.
    """
    if not seeds:
        raise ValueError("seeds must be a non-empty list")

    per_seed: dict[str, dict] = {}
    for seed in seeds:
        per_seed[str(seed)] = build_validation_report(
            spec_path,
            sizes,
            seed=seed,
            selection_policy=selection_policy,
            thresholds=thresholds,
            mechanism_benchmark=mechanism_benchmark,
            clock_price_step=clock_price_step,
            clock_max_rounds=clock_max_rounds,
            clock_reserve=clock_reserve,
        )

    return {
        "spec_path": str(spec_path),
        "selection_policy": selection_policy,
        "seeds": list(seeds),
        "all_seeds_schema_valid": all(
            r["schema_validation"]["valid"] for r in per_seed.values()
        ),
        "all_seeds_acceptance_passed": all(
            r["acceptance"]["passed"] for r in per_seed.values()
        ),
        "per_seed": per_seed,
        "aggregate": aggregate_metrics_across_seeds(per_seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="*",
        default=None,
        help="Override sizes as WxB pairs, e.g. --sizes 4x4 6x6 (default: 4x4 6x6 8x8 10x10)",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="Optional JSON file with acceptance-threshold overrides (merged over defaults).",
    )
    parser.add_argument(
        "--selection-policy",
        choices=list(SELECTION_POLICIES),
        default="prefix",
        help=(
            "How to pick num_goods/num_bidders from the spec. 'prefix' (default) "
            "takes spec declaration order. 'seeded_sample' takes a seeded random "
            "subset. 'stratified' picks one good per hardware category (CPU, GPU, "
            "motherboard, RAM, storage, power/cooling) and one bidder per broad "
            "archetype bucket where available, filling the rest via seeded shuffle "
            "-- use this when a plain prefix of a larger generated spec misses a "
            "complete platform of complementary goods."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Scenario selection seed (used by seeded_sample/stratified; ignored by prefix).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Run validation once per seed and report per-seed diagnostics plus "
            "mean/min/max aggregates, e.g. --seeds 0 1 2 3 4. Overrides --seed."
        ),
    )
    parser.add_argument(
        "--mechanism-benchmark",
        action="store_true",
        help=(
            "Also run a deterministic full-info sealed-VCG and ascending-clock "
            "(top_k=1,2,3) benchmark per size. No LLM calls; uses the true "
            "valuation table directly. Slower (multiple ILP solves per size)."
        ),
    )
    parser.add_argument("--clock-price-step", type=float, default=50.0)
    parser.add_argument("--clock-max-rounds", type=int, default=40)
    parser.add_argument("--clock-reserve", type=float, default=0.0)
    args = parser.parse_args()

    if args.sizes:
        sizes = []
        for spec_str in args.sizes:
            g, b = spec_str.lower().split("x")
            sizes.append((int(g), int(b)))
    else:
        sizes = DEFAULT_SIZES

    thresholds = load_thresholds(args.thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.seeds:
        report = build_multi_seed_report(
            args.spec_path,
            sizes,
            args.seeds,
            selection_policy=args.selection_policy,
            thresholds=thresholds,
            mechanism_benchmark=args.mechanism_benchmark,
            clock_price_step=args.clock_price_step,
            clock_max_rounds=args.clock_max_rounds,
            clock_reserve=args.clock_reserve,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")

        schema_status = "VALID" if report["all_seeds_schema_valid"] else "INVALID"
        acceptance_status = "PASSED" if report["all_seeds_acceptance_passed"] else "FAILED"
        print(
            f"{schema_status} (all seeds) / acceptance {acceptance_status} (all seeds): "
            f"{args.spec_path} [{args.selection_policy}, seeds={args.seeds}] -> {args.output}"
        )
        for seed_str, seed_report in report["per_seed"].items():
            if not seed_report["acceptance"]["passed"]:
                for check in seed_report["acceptance"]["checks"]:
                    if not check["passed"]:
                        print(
                            f"  - FAILED seed={seed_str} [{check['size']}] {check['name']}: "
                            f"actual={check['actual']} threshold={check['threshold']}"
                        )
        return

    report = build_validation_report(
        args.spec_path,
        sizes,
        seed=args.seed,
        selection_policy=args.selection_policy,
        thresholds=thresholds,
        mechanism_benchmark=args.mechanism_benchmark,
        clock_price_step=args.clock_price_step,
        clock_max_rounds=args.clock_max_rounds,
        clock_reserve=args.clock_reserve,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    status = "VALID" if report["schema_validation"]["valid"] else "INVALID"
    acceptance_status = "PASSED" if report["acceptance"]["passed"] else "FAILED"
    print(
        f"{status} / acceptance {acceptance_status}: "
        f"{args.spec_path} [{args.selection_policy}, seed={args.seed}] -> {args.output}"
    )
    if not report["schema_validation"]["valid"]:
        for err in report["schema_validation"]["errors"]:
            print(f"  - {err}")
    if not report["acceptance"]["passed"]:
        for check in report["acceptance"]["checks"]:
            if not check["passed"]:
                print(f"  - FAILED [{check['size']}] {check['name']}: actual={check['actual']} threshold={check['threshold']}")


if __name__ == "__main__":
    main()
