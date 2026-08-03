#!/usr/bin/env python3
"""Fit a provisional-value calibration from frozen benchmark artefacts.

Offline by construction: this script reads only the JSON artefacts written by
``scripts/prepare_pv_calibration_benchmark.py`` and never constructs an LLM
client. Every number it reports comes from predictions that were frozen at
preparation time.

What it does:

1. loads every artefact and flattens it into (bidder, bundle) observations;
2. runs leave-one-domain-out cross-validation -- fit on four domains, score on
   the fifth -- which is the only evidence about whether a calibration
   transfers to a domain it has not seen;
3. refits on all domains to produce the shipped parameters;
4. writes per-observation, per-fold, aggregate and fitted-parameter CSVs,
   comparison plots, and a calibration JSON consumable directly by
   ``--pv-calibration-config``.

``size_threshold`` is held at 3 unless ``--size-threshold-grid`` is passed.
``size_gamma`` is fitted, never assumed: the report states plainly whether the
held-out folds support a size effect at all.

Example::

    ./venv/bin/python scripts/fit_pv_calibration.py \\
      --benchmark-dir outputs/pv_calibration/benchmark \\
      --output-dir outputs/pv_calibration/fit \\
      --objective budget_normalized_mae \\
      --family exponential
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.experiments.pv_calibration import (  # noqa: E402
    FITTING_OBJECTIVES,
    METRIC_FIELDS,
    FoldResult,
    PvObservation,
    errors_by_bundle_size,
    evaluate_predictions,
    fit_calibration,
    leave_one_domain_out,
    load_benchmark_artefact,
    load_observations,
    metrics_row,
    observation_rows,
)
from auctionlab.llm.value_calibration import (  # noqa: E402
    CALIBRATION_FAMILIES,
    ValueCalibration,
    write_calibration_config,
)


RAW = ValueCalibration(family="none")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--benchmark-dir",
        type=Path,
        help="Directory of pv_calibration_<domain>_seed<N>.json artefacts.",
    )
    source.add_argument(
        "--benchmark-file",
        type=Path,
        nargs="+",
        help="Explicit artefact files instead of a whole directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=list(FITTING_OBJECTIVES),
        default="budget_normalized_mae",
        help=(
            "Fitting objective. Both are reported either way; this chooses "
            "which one the parameters minimise. Budget-normalised MAE puts "
            "bidders with very different value scales on a common footing; "
            "the robust log error additionally treats a 2x under-estimate and "
            "a 2x over-estimate symmetrically."
        ),
    )
    parser.add_argument(
        "--family",
        choices=[f for f in CALIBRATION_FAMILIES if f != "none"],
        default="exponential",
    )
    parser.add_argument("--size-threshold", type=int, default=3)
    parser.add_argument(
        "--size-threshold-grid",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional small grid of size thresholds to select over, e.g. "
            "'2 3 4'. Selecting the threshold adds a parameter chosen on the "
            "benchmark, so the cross-validated fold table -- not the "
            "all-domain fit -- is the honest report."
        ),
    )
    parser.add_argument(
        "--no-budget-cap",
        action="store_true",
        help="Fit and report without the disclosed-budget cap.",
    )
    parser.add_argument("--scale-min", type=float, default=0.2)
    parser.add_argument("--scale-max", type=float, default=5.0)
    parser.add_argument("--gamma-min", type=float, default=0.6)
    parser.add_argument("--gamma-max", type=float, default=1.4)
    parser.add_argument("--grid-steps", type=int, default=25)
    parser.add_argument("--grid-passes", type=int, default=5)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib output (CSV and JSON are still written).",
    )
    return parser.parse_args(argv)


def _artefact_paths(args: argparse.Namespace) -> list[Path]:
    if args.benchmark_file:
        paths = [Path(p) for p in args.benchmark_file]
    else:
        paths = sorted(args.benchmark_dir.glob("pv_calibration_*.json"))
    if not paths:
        raise SystemExit(
            "Error: no benchmark artefacts found. Run "
            "scripts/prepare_pv_calibration_benchmark.py first."
        )
    return paths


def _write_csv(rows: Sequence[dict[str, Any]], path: Path, fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(fields) if fields else list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _fold_rows(folds: Sequence[FoldResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        calibration = fold.fit.calibration
        for variant, metrics in (
            ("raw", fold.test_metrics_raw),
            ("calibrated", fold.test_metrics_calibrated),
        ):
            row = metrics_row("fold_heldout", fold.held_out_domain, variant, metrics)
            row.update(
                {
                    "fit_scale": calibration.scale,
                    "fit_size_gamma": calibration.size_gamma,
                    "fit_size_threshold": calibration.size_threshold,
                    "fit_objective_value": fold.fit.objective_value,
                    "train_n": fold.fit.n_observations,
                }
            )
            rows.append(row)
    return rows


def _plot(
    observations: Sequence[PvObservation],
    calibration: ValueCalibration,
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    domains = sorted({observation.domain for observation in observations})

    # 1. Raw vs calibrated MAE per domain.
    raw_mae = []
    cal_mae = []
    for domain in domains:
        subset = [o for o in observations if o.domain == domain]
        raw_mae.append(evaluate_predictions(subset, RAW)["mae"])
        cal_mae.append(evaluate_predictions(subset, calibration)["mae"])

    figure, axis = plt.subplots(figsize=(9, 4.5))
    positions = range(len(domains))
    axis.bar([p - 0.2 for p in positions], raw_mae, width=0.4, label="raw")
    axis.bar([p + 0.2 for p in positions], cal_mae, width=0.4, label="calibrated")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(domains, rotation=20, ha="right")
    axis.set_ylabel("MAE (currency units)")
    axis.set_title("Provisional-value error by domain")
    axis.legend()
    figure.tight_layout()
    path = output_dir / "pv_calibration_error_by_domain.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    # 2. Error versus bundle size.
    raw_rows = errors_by_bundle_size(observations, RAW)
    cal_rows = errors_by_bundle_size(observations, calibration)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sizes = [row["bundle_size"] for row in raw_rows]
    axes[0].plot(sizes, [r["mae"] for r in raw_rows], marker="o", label="raw")
    axes[0].plot(sizes, [r["mae"] for r in cal_rows], marker="s", label="calibrated")
    axes[0].set_xlabel("bundle size")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Absolute error by bundle size")
    axes[0].legend()
    axes[1].axhline(0.0, color="grey", linewidth=0.8)
    axes[1].plot(
        sizes, [r["signed_bias"] for r in raw_rows], marker="o", label="raw"
    )
    axes[1].plot(
        sizes, [r["signed_bias"] for r in cal_rows], marker="s", label="calibrated"
    )
    axes[1].set_xlabel("bundle size")
    axes[1].set_ylabel("signed bias (predicted - true)")
    axes[1].set_title("Signed bias by bundle size")
    axes[1].legend()
    figure.tight_layout()
    path = output_dir / "pv_calibration_error_by_bundle_size.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    return written


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = _artefact_paths(args)
    observations, artefact_hashes = load_observations(paths)
    if not observations:
        raise SystemExit("Error: benchmark artefacts contain no observations")

    domains = sorted({observation.domain for observation in observations})
    seeds = sorted({observation.seed for observation in observations})
    proxy_models = sorted(
        {
            json.dumps(load_benchmark_artefact(path)["models"]["proxy"], sort_keys=True)
            for path in paths
        }
    )
    if len(proxy_models) > 1:
        print(
            "  WARNING: artefacts were produced by more than one proxy model; "
            "a calibration fitted across them describes no single estimator.",
            file=sys.stderr,
        )

    budget_cap = not args.no_budget_cap
    fit_kwargs = dict(
        objective=args.objective,
        family=args.family,
        size_threshold=args.size_threshold,
        size_threshold_grid=args.size_threshold_grid,
        budget_cap=budget_cap,
        scale_bounds=(args.scale_min, args.scale_max),
        gamma_bounds=(args.gamma_min, args.gamma_max),
        steps=args.grid_steps,
        passes=args.grid_passes,
    )

    print(
        f"  observations {len(observations)}  domains {len(domains)}  "
        f"seeds {seeds}  objective {args.objective}  family {args.family}"
    )

    folds = (
        leave_one_domain_out(observations, **fit_kwargs)
        if len(domains) >= 2
        else []
    )
    final = fit_calibration(observations, **fit_kwargs)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- per-observation ---------------------------------------------------
    _write_csv(
        observation_rows(observations, final.calibration),
        output_dir / "pv_calibration_observations.csv",
    )

    # -- per-fold and aggregate metrics ------------------------------------
    fold_fields = list(METRIC_FIELDS) + [
        "fit_scale",
        "fit_size_gamma",
        "fit_size_threshold",
        "fit_objective_value",
        "train_n",
    ]
    _write_csv(
        _fold_rows(folds),
        output_dir / "pv_calibration_folds.csv",
        fields=fold_fields,
    )

    aggregate_rows: list[dict[str, Any]] = []
    for variant, calibration in (("raw", RAW), ("calibrated", final.calibration)):
        aggregate_rows.append(
            metrics_row(
                "all_domains",
                "ALL",
                variant,
                evaluate_predictions(observations, calibration),
            )
        )
        for domain in domains:
            subset = [o for o in observations if o.domain == domain]
            aggregate_rows.append(
                metrics_row(
                    "domain",
                    domain,
                    variant,
                    evaluate_predictions(subset, calibration),
                )
            )
    _write_csv(
        aggregate_rows,
        output_dir / "pv_calibration_metrics.csv",
        fields=METRIC_FIELDS,
    )

    size_rows: list[dict[str, Any]] = []
    for variant, calibration in (("raw", RAW), ("calibrated", final.calibration)):
        for row in errors_by_bundle_size(observations, calibration):
            size_rows.append({"variant": variant, **row})
    _write_csv(size_rows, output_dir / "pv_calibration_by_bundle_size.csv")

    # -- fitted parameters -------------------------------------------------
    parameter_rows = [
        {
            "scope": "final_all_domains",
            "held_out_domain": "",
            "family": final.calibration.family,
            "scale": final.calibration.scale,
            "size_gamma": final.calibration.size_gamma,
            "size_threshold": final.calibration.size_threshold,
            "budget_cap": final.calibration.budget_cap,
            "objective": final.objective,
            "objective_value": final.objective_value,
            "n_observations": final.n_observations,
        }
    ]
    for fold in folds:
        parameter_rows.append(
            {
                "scope": "fold",
                "held_out_domain": fold.held_out_domain,
                "family": fold.fit.calibration.family,
                "scale": fold.fit.calibration.scale,
                "size_gamma": fold.fit.calibration.size_gamma,
                "size_threshold": fold.fit.calibration.size_threshold,
                "budget_cap": fold.fit.calibration.budget_cap,
                "objective": fold.fit.objective,
                "objective_value": fold.fit.objective_value,
                "n_observations": fold.fit.n_observations,
            }
        )
    _write_csv(parameter_rows, output_dir / "pv_calibration_parameters.csv")

    # -- calibration JSON --------------------------------------------------
    cv_summary = [
        {
            "held_out_domain": fold.held_out_domain,
            "scale": fold.fit.calibration.scale,
            "size_gamma": fold.fit.calibration.size_gamma,
            "size_threshold": fold.fit.calibration.size_threshold,
            "heldout_raw": {
                key: fold.test_metrics_raw.get(key)
                for key in (
                    "mae",
                    "budget_normalized_mae",
                    "robust_log_error",
                    "signed_bias",
                )
            },
            "heldout_calibrated": {
                key: fold.test_metrics_calibrated.get(key)
                for key in (
                    "mae",
                    "budget_normalized_mae",
                    "robust_log_error",
                    "signed_bias",
                )
            },
        }
        for fold in folds
    ]
    improved = sum(
        1
        for fold in folds
        if fold.test_metrics_calibrated["budget_normalized_mae"]
        < fold.test_metrics_raw["budget_normalized_mae"]
    )
    calibration = final.calibration.with_provenance(
        generator="scripts/fit_pv_calibration.py",
        created_at=datetime.now(timezone.utc).isoformat(),
        git_revision=_git_revision(),
        proxy_models=[json.loads(model) for model in proxy_models],
        benchmark_domains=domains,
        benchmark_seeds=seeds,
        benchmark_artefact_sha256=artefact_hashes,
        n_observations=len(observations),
        fitting_objective=args.objective,
        fitting_objective_value=final.objective_value,
        size_threshold_grid=list(final.size_threshold_grid),
        cross_validation={
            "scheme": "leave_one_domain_out",
            "folds": cv_summary,
            "folds_improved_on_heldout": improved,
            "folds_total": len(folds),
        },
        note=(
            "Out-of-domain calibration fitted on consumer-bundle benchmark "
            "domains, not on PC-build experimental instances. Treat as a "
            "robustness/sensitivity treatment until the held-out folds show "
            "consistent improvement."
        ),
    )
    config_path = write_calibration_config(
        calibration, output_dir / "pv_calibration.json"
    )

    plots: list[Path] = []
    if not args.no_plots:
        plots = _plot(observations, final.calibration, output_dir)

    # -- console report ----------------------------------------------------
    raw_all = evaluate_predictions(observations, RAW)
    cal_all = evaluate_predictions(observations, final.calibration)
    print("\n  fitted (all domains, in-sample):")
    print(
        f"    family={final.calibration.family}  "
        f"scale={final.calibration.scale:.4f}  "
        f"size_gamma={final.calibration.size_gamma:.4f}  "
        f"size_threshold={final.calibration.size_threshold}  "
        f"budget_cap={final.calibration.budget_cap}"
    )
    for label, metrics in (("raw", raw_all), ("calibrated", cal_all)):
        print(
            f"    {label:<11} mae={metrics['mae']:.1f}  "
            f"rmse={metrics['rmse']:.1f}  "
            f"bn_mae={metrics['budget_normalized_mae']:.4f}  "
            f"log_err={metrics['robust_log_error']:.4f}  "
            f"bias={metrics['signed_bias']:+.1f}"
        )

    if folds:
        print("\n  leave-one-domain-out (held-out budget-normalised MAE):")
        for fold in folds:
            raw_value = fold.test_metrics_raw["budget_normalized_mae"]
            cal_value = fold.test_metrics_calibrated["budget_normalized_mae"]
            verdict = "better" if cal_value < raw_value else "WORSE"
            print(
                f"    {fold.held_out_domain:<26} raw={raw_value:.4f}  "
                f"calibrated={cal_value:.4f}  [{verdict}]  "
                f"(scale={fold.fit.calibration.scale:.3f}, "
                f"gamma={fold.fit.calibration.size_gamma:.3f})"
            )
        gammas = [fold.fit.calibration.size_gamma for fold in folds]
        print(
            f"\n  size_gamma across folds: min={min(gammas):.3f} "
            f"max={max(gammas):.3f}  (fitted, never assumed)"
        )
        if not final.size_gamma_identifiable:
            print(
                "  size_gamma is UNIDENTIFIABLE here: no observed bundle "
                f"exceeds size_threshold={final.calibration.size_threshold}, "
                "so it was pinned to 1.0. Lower the threshold or include "
                "larger candidate bundles before reading a size effect."
            )
        elif max(abs(gamma - 1.0) for gamma in gammas) < 0.02:
            print(
                "  Fold gammas sit within 0.02 of 1.0: this benchmark does "
                "NOT support a bundle-size effect; prefer family=uniform."
            )
        print(
            f"  {improved}/{len(folds)} folds improved out of domain. "
            "Fewer than all is not evidence of generalisation."
        )

    print(f"\n  calibration JSON        →  {config_path}")
    print(f"  observations CSV        →  {output_dir / 'pv_calibration_observations.csv'}")
    print(f"  fold metrics CSV        →  {output_dir / 'pv_calibration_folds.csv'}")
    print(f"  aggregate metrics CSV   →  {output_dir / 'pv_calibration_metrics.csv'}")
    print(f"  parameters CSV          →  {output_dir / 'pv_calibration_parameters.csv'}")
    for path in plots:
        print(f"  plot                    →  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
