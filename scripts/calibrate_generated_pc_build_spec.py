#!/usr/bin/env python3
"""Deterministic calibration grid-search for the generated PC-build spec.

The diagnostics in ``diagnose_pc_build_environments.py`` show the LLM-generated
spec (``pc_build_profiles_gemini_trial_final.json``) has 25-32% lower
full-info welfare than the manual spec, much lower positive-bundle values
(e.g. 8x8 median 450 vs. manual's 1230), and lets the ``reseller_pro``
archetype dominate the top-20 global bundles (9/20 at 6x6, 4/20 at 8x8,
vs. 0/20 for the manual reseller). This script explores whether simple,
transparent multiplicative corrections to the *frozen spec's* base values,
complement bonuses, and saturation penalties can close that gap.

Nothing in this script makes an LLM/API call. Every candidate spec is
produced by pure arithmetic over the already-frozen generated spec, and
every candidate is scored with the same deterministic WDP ILP solver used
throughout this repo's validation tooling.

This script does **not** select or write a "winning" spec -- it only ranks
candidates and writes CSVs. A human reviews the ranking and decides whether
any candidate (and which multipliers) should be frozen as a new spec
version via ``write_scenario_profile_spec``.

Usage::

    ./venv/bin/python scripts/calibrate_generated_pc_build_spec.py

Writes to ``scenarios/pc_build_v1/diagnostics/``:
    - calibration_per_size.csv       (one row per candidate x size)
    - calibration_ranked_candidates.csv  (one row per candidate, ranked)
"""

from __future__ import annotations

import itertools
import statistics
import sys
from pathlib import Path

# Allow flat (same-directory) imports whether this file is run directly
# (`python scripts/calibrate_generated_pc_build_spec.py`, where sys.path[0]
# is already this directory) or collected by pytest (where it isn't).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_pc_build_environments import (  # noqa: E402
    GENERATED_SPEC_PATH,
    RESELLER_ID,
    SIZES,
    _write_csv,
    build_manual_spec,
    compute_size_metrics,
)

from auctionlab.instances.scenario_spec import ScenarioProfileSpec, load_scenario_profile_spec  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "scenarios/pc_build_v1/diagnostics"

RESELLER_BIDDER_ID = RESELLER_ID["generated"]  # "reseller_pro"

# Default calibration grid (task-specified). 3*3*2*2 = 36 candidates x 3
# sizes = 108 deterministic WDP solves; a few seconds on a laptop.
DEFAULT_GRID: dict[str, list[float]] = {
    "non_reseller_value_multiplier": [1.2, 1.35, 1.5],
    "reseller_value_multiplier": [0.7, 0.85, 1.0],
    "complement_multiplier": [1.0, 1.2],
    "saturation_penalty_multiplier": [1.0, 1.2],
}

# Acceptance thresholds used only to flag rows/candidates in the ranked
# output -- not to filter or auto-select anything.
ACCEPT_OPTIMUM_RATIO_LOW = 0.85
ACCEPT_OPTIMUM_RATIO_HIGH = 1.15
ACCEPT_MAX_RESELLER_TOP20_SHARE = 0.2

# Weight applied to reseller-dominance in the composite ranking score; see
# `_composite_rank_score` below.
RESELLER_DOMINANCE_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Calibration transform
# ---------------------------------------------------------------------------

def apply_calibration(
    spec: ScenarioProfileSpec,
    *,
    non_reseller_value_multiplier: float,
    reseller_value_multiplier: float,
    complement_multiplier: float,
    saturation_penalty_multiplier: float,
    reseller_id: str = RESELLER_BIDDER_ID,
) -> ScenarioProfileSpec:
    """Return a new spec with per-bidder multipliers applied.

    - every ``base_values`` entry is scaled by ``reseller_value_multiplier``
      for ``reseller_id``, ``non_reseller_value_multiplier`` for every other
      bidder;
    - every ``complement_groups[].bonus`` is scaled by
      ``complement_multiplier`` (all bidders, reseller included);
    - ``saturation_penalty`` is scaled by ``saturation_penalty_multiplier``
      for bidders that actually have ``saturation_start`` set (a no-op
      otherwise -- there's nothing to scale if saturation never kicks in).

    Budgets, budget caps, backup factors, and item classifications
    (core/secondary/low-interest) are left untouched: those aren't part of
    the value-scale mismatch this calibration targets.
    """
    new_bidder_profiles = []
    for bp in spec.bidder_profiles:
        value_multiplier = (
            reseller_value_multiplier if bp.bidder_id == reseller_id else non_reseller_value_multiplier
        )
        new_base_values = {item: value * value_multiplier for item, value in bp.base_values.items()}
        new_complement_groups = [
            cg.model_copy(update={"bonus": cg.bonus * complement_multiplier}) for cg in bp.complement_groups
        ]
        new_saturation_penalty = (
            bp.saturation_penalty * saturation_penalty_multiplier
            if bp.saturation_start is not None
            else bp.saturation_penalty
        )
        new_bidder_profiles.append(
            bp.model_copy(
                update={
                    "base_values": new_base_values,
                    "complement_groups": new_complement_groups,
                    "saturation_penalty": new_saturation_penalty,
                }
            )
        )
    return spec.model_copy(update={"bidder_profiles": new_bidder_profiles})


def iter_default_grid(grid: dict[str, list[float]] | None = None):
    """Yield one dict of calibration parameters per grid point (Cartesian product)."""
    grid = grid if grid is not None else DEFAULT_GRID
    keys = list(grid.keys())
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def _passes_acceptance(row: dict) -> bool:
    ratio = row["optimum_ratio_vs_manual"]
    if ratio is None:
        return False
    return (
        ACCEPT_OPTIMUM_RATIO_LOW <= ratio <= ACCEPT_OPTIMUM_RATIO_HIGH
        and row["reseller_top20_share"] <= ACCEPT_MAX_RESELLER_TOP20_SHARE
    )


def evaluate_candidate(
    candidate_spec: ScenarioProfileSpec,
    params: dict[str, float],
    manual_optimum_by_size: dict[str, float],
    *,
    candidate_label: str,
) -> list[dict]:
    """Evaluate one candidate spec at every size in ``SIZES``.

    Returns one row per size with: full-info optimum, optimum ratio vs. the
    manual spec at that size, winner count, largest-winner share, reseller
    top-20 count/share, positive-bundle-value median/mean/max, and an
    acceptance pass/fail flag (see module-level thresholds).
    """
    rows = []
    for num_goods, num_bidders in SIZES:
        size_key = f"{num_goods}x{num_bidders}"
        result = compute_size_metrics(
            candidate_label,
            candidate_spec,
            num_goods,
            num_bidders,
            reseller_id=RESELLER_BIDDER_ID,
            compute_pv_gt=False,
        )
        s = result["summary"]
        manual_optimum = manual_optimum_by_size[size_key]
        optimum_ratio = s["full_info_optimum_welfare"] / manual_optimum if manual_optimum else None

        row = {
            "candidate": candidate_label,
            **params,
            "size": size_key,
            "full_info_optimum": s["full_info_optimum_welfare"],
            "optimum_ratio_vs_manual": optimum_ratio,
            "num_winners": s["num_winners"],
            "largest_winner_welfare_share": s["largest_winner_welfare_share"],
            "reseller_top20_count": s["reseller_count_in_top20_global"],
            "reseller_top20_share": s["reseller_value_share_in_top20_global"],
            "positive_value_median": s["value_median_positive"],
            "positive_value_mean": s["value_mean_positive"],
            "positive_value_max": s["value_max_positive"],
        }
        row["acceptance_pass"] = _passes_acceptance(row)
        rows.append(row)
    return rows


def _composite_rank_score(mean_optimum_ratio: float | None, mean_reseller_share: float) -> float | None:
    """Lower is better: distance from optimum-ratio 1.0, plus reseller-dominance penalty.

    This is a ranking aid only -- it does not filter or select candidates.
    ``RESELLER_DOMINANCE_WEIGHT`` controls how much a candidate is penalized
    for reseller-dominant top-20 bundles relative to how far its welfare
    scale sits from the manual spec's.
    """
    if mean_optimum_ratio is None:
        return None
    return abs(mean_optimum_ratio - 1.0) + RESELLER_DOMINANCE_WEIGHT * mean_reseller_share


def summarize_candidate(rows_for_candidate: list[dict], params: dict[str, float], candidate_label: str) -> dict:
    optimum_ratios = [r["optimum_ratio_vs_manual"] for r in rows_for_candidate if r["optimum_ratio_vs_manual"] is not None]
    reseller_shares = [r["reseller_top20_share"] for r in rows_for_candidate]
    largest_shares = [r["largest_winner_welfare_share"] for r in rows_for_candidate]
    pos_medians = [r["positive_value_median"] for r in rows_for_candidate]
    pos_means = [r["positive_value_mean"] for r in rows_for_candidate]
    pos_maxes = [r["positive_value_max"] for r in rows_for_candidate]

    mean_optimum_ratio = statistics.fmean(optimum_ratios) if optimum_ratios else None
    mean_reseller_share = statistics.fmean(reseller_shares) if reseller_shares else 0.0

    return {
        "candidate": candidate_label,
        **params,
        "mean_optimum_ratio_vs_manual": mean_optimum_ratio,
        "mean_largest_winner_welfare_share": statistics.fmean(largest_shares) if largest_shares else 0.0,
        "mean_reseller_top20_share": mean_reseller_share,
        "max_reseller_top20_share": max(reseller_shares, default=0.0),
        "mean_positive_value_median": statistics.fmean(pos_medians) if pos_medians else 0.0,
        "mean_positive_value_mean": statistics.fmean(pos_means) if pos_means else 0.0,
        "mean_positive_value_max": statistics.fmean(pos_maxes) if pos_maxes else 0.0,
        "all_sizes_acceptance_pass": all(r["acceptance_pass"] for r in rows_for_candidate),
        "composite_rank_score": _composite_rank_score(mean_optimum_ratio, mean_reseller_share),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    manual_spec = build_manual_spec()
    generated_spec = load_scenario_profile_spec(GENERATED_SPEC_PATH)

    manual_optimum_by_size: dict[str, float] = {}
    for num_goods, num_bidders in SIZES:
        result = compute_size_metrics(
            "manual", manual_spec, num_goods, num_bidders,
            reseller_id=RESELLER_ID["manual"], compute_pv_gt=False,
        )
        manual_optimum_by_size[f"{num_goods}x{num_bidders}"] = result["summary"]["full_info_optimum_welfare"]

    per_size_rows: list[dict] = []
    candidate_summaries: list[dict] = []

    for i, params in enumerate(iter_default_grid()):
        label = f"cand_{i:03d}"
        candidate_spec = apply_calibration(generated_spec, reseller_id=RESELLER_BIDDER_ID, **params)
        rows = evaluate_candidate(candidate_spec, params, manual_optimum_by_size, candidate_label=label)
        per_size_rows.extend(rows)
        candidate_summaries.append(summarize_candidate(rows, params, label))
        print(f"  evaluated {label}: {params}")

    candidate_summaries.sort(
        key=lambda s: (s["composite_rank_score"] is None, s["composite_rank_score"])
    )
    for rank, s in enumerate(candidate_summaries, start=1):
        s["rank"] = rank
    # Put rank first for readability.
    candidate_summaries = [
        {"rank": s["rank"], **{k: v for k, v in s.items() if k != "rank"}} for s in candidate_summaries
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "calibration_per_size.csv", per_size_rows)
    _write_csv(OUTPUT_DIR / "calibration_ranked_candidates.csv", candidate_summaries)

    num_accepted = sum(1 for s in candidate_summaries if s["all_sizes_acceptance_pass"])
    print(
        f"\nEvaluated {len(candidate_summaries)} candidates x {len(SIZES)} sizes "
        f"({num_accepted} pass acceptance at all sizes) -> {OUTPUT_DIR}"
    )
    print("No spec was selected or written -- review calibration_ranked_candidates.csv and decide manually.")
    if candidate_summaries:
        top = candidate_summaries[0]
        print(
            f"Top-ranked by composite score: {top['candidate']} "
            f"(non_reseller={top['non_reseller_value_multiplier']}, "
            f"reseller={top['reseller_value_multiplier']}, "
            f"complement={top['complement_multiplier']}, "
            f"saturation={top['saturation_penalty_multiplier']}) "
            f"-- mean optimum ratio {top['mean_optimum_ratio_vs_manual']:.3f}, "
            f"mean reseller top-20 share {top['mean_reseller_top20_share']:.1%}"
        )


if __name__ == "__main__":
    main()
