#!/usr/bin/env python3
"""Select and validate a uniform PV scale by held-out VCG payment error.

This command is fully offline.  The final design expects three independently
generated instances in each of three non-PC domains.  It leaves one instance
index out at a time, selects a uniform scale on the other six environments by
initial reported-bid VCG payment error, and validates raw/calibrated initial,
sealed and clock outcomes on the three held-out environments.  It then refits
one deployable scale on all nine environments after the fitting method has
passed held-out validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.auctions.clock import ClockConfig  # noqa: E402
from auctionlab.experiments.proxy_clock_runner import (  # noqa: E402
    ProxyClockConfig,
    run_proxy_clock_experiment,
)
from auctionlab.experiments.proxy_sealed_runner import (  # noqa: E402
    ProxySealedConfig,
    run_proxy_sealed_vcg_trajectory,
)
from auctionlab.experiments.pv_calibration import (  # noqa: E402
    evaluate_predictions,
    load_benchmark_artefact,
    observations_from_artefact,
)
from auctionlab.experiments.pv_calibration_environments import (  # noqa: E402
    GENERATED_CALIBRATION_DOMAINS,
    build_generated_environment_scenario,
)
from auctionlab.experiments.runner import run_sealed_vcg_experiment  # noqa: E402
from auctionlab.llm.clients import MockLlmClient  # noqa: E402
from auctionlab.llm.person_simulator import LlmPersonSimulator  # noqa: E402
from auctionlab.llm.proxies import (  # noqa: E402
    LlmAuctionProxyAdapter,
    LlmInferredXorProxy,
)
from auctionlab.llm.schemas import LlmInterestMap  # noqa: E402
from auctionlab.llm.value_calibration import (  # noqa: E402
    ValueCalibration,
    write_calibration_config,
)


RAW = ValueCalibration(family="none")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=[
            "payment_error_over_optimum_welfare",
            "revenue_absolute_error_over_optimum_welfare",
        ],
        default="payment_error_over_optimum_welfare",
        help=(
            "Held-out initial-VCG pricing loss minimised when selecting the "
            "uniform scale. The default sums absolute bidder-payment errors "
            "and normalises by full-information welfare."
        ),
    )
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.5)
    parser.add_argument("--grid-steps", type=int, default=21)
    parser.add_argument("--grid-passes", type=int, default=4)
    parser.add_argument("--sealed-max-rounds", type=int, default=25)
    parser.add_argument("--clock-max-rounds", type=int, default=50)
    parser.add_argument("--clock-price-step", type=float, default=50.0)
    parser.add_argument("--clock-top-k", type=int, default=3)
    parser.add_argument(
        "--efficiency-tolerance",
        type=float,
        default=0.02,
        help=(
            "Maximum permitted reduction in mean held-out efficiency for "
            "each downstream mechanism (default: 0.02). Worst-case changes "
            "are reported but do not independently veto calibration."
        ),
    )
    parser.add_argument(
        "--allow-rejected",
        action="store_true",
        help="Return success even when acceptance criteria fail.",
    )
    args = parser.parse_args(argv)
    if not 0 < args.scale_min < args.scale_max:
        parser.error("require 0 < --scale-min < --scale-max")
    if args.grid_steps < 2:
        parser.error("--grid-steps must be at least 2")
    if args.grid_passes < 1:
        parser.error("--grid-passes must be at least 1")
    return args


def _benchmark_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("pv_calibration_*.json"))
    if not paths:
        raise SystemExit(f"No benchmark artefacts found in {directory}")
    return paths


def _load_artefacts(
    paths: Sequence[Path],
) -> dict[tuple[str, int], dict[str, Any]]:
    artefacts: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        artefact = load_benchmark_artefact(path)
        domain = str(artefact["domain"])
        seed = int(artefact["seed"])
        key = (domain, seed)
        if key in artefacts:
            raise SystemExit(
                f"Duplicate frozen environment domain={domain} seed={seed}"
            )
        if artefact.get("environment") is None:
            raise SystemExit(
                f"{path} has no embedded generated environment. Re-run "
                "prepare_pv_calibration_benchmark.py with --environment-dir."
            )
        environment_index = int(
            artefact["environment"].get("instance_index", 0)
        )
        if environment_index != seed:
            raise SystemExit(
                f"{path}: artefact seed={seed} does not match embedded "
                f"environment instance_index={environment_index}"
            )
        artefacts[key] = artefact
    expected = set(GENERATED_CALIBRATION_DOMAINS)
    found_domains = {domain for domain, _ in artefacts}
    if found_domains != expected:
        raise SystemExit(
            "All three generated domains are required; "
            f"expected={sorted(expected)}, found={sorted(found_domains)}"
        )
    seeds_by_domain = {
        domain: {seed for candidate_domain, seed in artefacts if candidate_domain == domain}
        for domain in expected
    }
    unique_seed_sets = {tuple(sorted(seeds)) for seeds in seeds_by_domain.values()}
    if len(unique_seed_sets) != 1:
        raise SystemExit(
            f"Every domain must contain the same instance indices: {seeds_by_domain}"
        )
    seeds = next(iter(unique_seed_sets))
    if len(seeds) < 3:
        raise SystemExit(
            "At least three independent instances per domain are required; "
            f"found indices={list(seeds)}"
        )
    return artefacts


def _proxy_adapters(
    artefact: Mapping[str, Any],
    calibration: ValueCalibration,
):
    scenario = build_generated_environment_scenario(artefact["environment"])
    adapters: list[LlmAuctionProxyAdapter] = []
    for bidder_id in scenario.instance.bidder_ids:
        entry = artefact["bidders"][bidder_id]
        person = LlmPersonSimulator(
            bidder_id=bidder_id,
            scenario_description=scenario.scenario_description,
            person_seed=scenario.person_seeds[bidder_id],
            item_descriptions=scenario.item_descriptions,
            client=MockLlmClient([]),
            ground_truth_valuations=scenario.instance.valuations[bidder_id],
        )
        proxy = LlmInferredXorProxy(
            bidder_id=bidder_id,
            person=person,
            calibration=calibration,
        )
        interest_map = (
            None
            if entry.get("interest_map") is None
            else LlmInterestMap.model_validate(entry["interest_map"])
        )
        raw_values = {
            frozenset(row["bundle"]): float(row["value"])
            for row in entry["raw_provisional_values"]
        }
        candidates = [
            frozenset(bundle) for bundle in entry["candidate_bundles"]
        ]
        proxy.replay_elicitation(
            nl_question=entry["nl_question"],
            nl_answer=entry["nl_answer"],
            interest_map=interest_map,
            provisional_raw_values=raw_values,
        )
        adapters.append(
            LlmAuctionProxyAdapter(
                bidder_id=bidder_id,
                proxy=proxy,
                candidate_bundles=candidates,
            )
        )
    return scenario, adapters


def _true_welfare(instance, allocation: Mapping[str, frozenset[str]]) -> float:
    return sum(
        instance.value_of(bidder_id, bundle)
        for bidder_id, bundle in allocation.items()
    )


def _pricing_metrics(result, truth) -> dict[str, float]:
    """Return scale-comparable VCG payment and revenue diagnostics."""
    bidder_ids = set(truth.payments) | set(result.payments)
    payment_error = sum(
        abs(result.payments.get(bidder_id, 0.0) - truth.payments.get(bidder_id, 0.0))
        for bidder_id in bidder_ids
    )
    welfare_denominator = max(float(truth.welfare), 1.0)
    revenue_error = abs(float(result.revenue) - float(truth.revenue))
    if truth.revenue > 0:
        revenue_loss = (truth.revenue - result.revenue) / truth.revenue
        revenue_absolute_percentage_error = revenue_error / truth.revenue
    else:
        revenue_loss = math.nan
        revenue_absolute_percentage_error = math.nan
    return {
        "payment_absolute_error": payment_error,
        "payment_error_over_optimum_welfare": (
            payment_error / welfare_denominator
        ),
        "revenue_absolute_error": revenue_error,
        "revenue_absolute_error_over_optimum_welfare": (
            revenue_error / welfare_denominator
        ),
        "revenue_loss": revenue_loss,
        "revenue_absolute_percentage_error": (
            revenue_absolute_percentage_error
        ),
    }


def _initial_result(
    artefact: Mapping[str, Any], calibration: ValueCalibration
):
    scenario, proxies = _proxy_adapters(artefact, calibration)
    result = run_proxy_sealed_vcg_trajectory(
        scenario.instance,
        proxies,
        ProxySealedConfig(elicitation_rounds=0),
    )[-1]
    truth = run_sealed_vcg_experiment(scenario.instance)
    return scenario, result, truth


def _pricing_objective(
    artefacts: Sequence[Mapping[str, Any]],
    calibration: ValueCalibration,
    objective: str,
) -> float:
    if not artefacts:
        return math.inf
    values = []
    for artefact in artefacts:
        _scenario, result, truth = _initial_result(artefact, calibration)
        values.append(_pricing_metrics(result, truth)[objective])
    return statistics.fmean(values)


def _fit_uniform_payment_scale(
    artefacts: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> tuple[ValueCalibration, float, list[dict[str, float]]]:
    """Deterministic coarse-to-fine scale search over initial VCG pricing."""
    low = args.scale_min
    high = args.scale_max
    best_scale = 1.0
    best_score = math.inf
    trace: list[dict[str, float]] = []
    for pass_index in range(args.grid_passes):
        step = (high - low) / (args.grid_steps - 1)
        for grid_index in range(args.grid_steps):
            scale = low + grid_index * step
            candidate = ValueCalibration(
                family="uniform",
                scale=scale,
                budget_cap=True,
            )
            score = _pricing_objective(
                artefacts, candidate, args.objective
            )
            trace.append(
                {
                    "pass": pass_index + 1,
                    "scale": scale,
                    "score": score,
                }
            )
            improved = score < best_score - 1e-12
            tied_and_simpler = (
                abs(score - best_score) <= 1e-12
                and abs(math.log(scale)) < abs(math.log(best_scale))
            )
            if improved or tied_and_simpler:
                best_scale = scale
                best_score = score
        low = max(1e-6, best_scale - step)
        high = best_scale + step
    return (
        ValueCalibration(
            family="uniform",
            scale=best_scale,
            budget_cap=True,
        ),
        best_score,
        trace,
    )


def _mechanism_rows(
    artefact: Mapping[str, Any],
    *,
    variant: str,
    calibration: ValueCalibration,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    scenario = build_generated_environment_scenario(artefact["environment"])
    truth = run_sealed_vcg_experiment(scenario.instance)
    optimum = truth.welfare
    rows: list[dict[str, Any]] = []

    _, initial_proxies = _proxy_adapters(artefact, calibration)
    initial = run_proxy_sealed_vcg_trajectory(
        scenario.instance,
        initial_proxies,
        ProxySealedConfig(elicitation_rounds=0),
    )[-1]

    _, sealed_proxies = _proxy_adapters(artefact, calibration)
    sealed = run_proxy_sealed_vcg_trajectory(
        scenario.instance,
        sealed_proxies,
        ProxySealedConfig(
            elicitation_rounds=args.sealed_max_rounds,
            feedback_rule="competitive",
            stopping_rule="no_new_refinements",
            loser_challenger_policy="off",
        ),
    )[-1]

    _, clock_proxies = _proxy_adapters(artefact, calibration)
    clock = run_proxy_clock_experiment(
        scenario.instance,
        clock_proxies,
        ClockConfig(
            max_rounds=args.clock_max_rounds,
            price_step=args.clock_price_step,
        ),
        ProxyClockConfig(
            top_k=args.clock_top_k,
            elicited=True,
            margin_threshold=100.0,
            tie_threshold=100.0,
            top_k_frontier_policy="allocation_pivotal",
            allocation_counterfactual_frontier=True,
        ),
    )

    for mechanism, result in (
        ("initial", initial),
        ("sealed", sealed),
        ("clock", clock),
    ):
        welfare = _true_welfare(scenario.instance, result.allocation)
        pricing = _pricing_metrics(result, truth)
        rows.append(
            {
                "domain": artefact["domain"],
                "environment_instance": int(artefact.get("seed", 0)),
                "variant": variant,
                "family": calibration.family,
                "scale": calibration.scale,
                "size_gamma": calibration.size_gamma,
                "size_threshold": calibration.size_threshold,
                "mechanism": mechanism,
                "true_welfare": welfare,
                "full_information_welfare": optimum,
                "efficiency": welfare / optimum if optimum else 1.0,
                "reported_revenue": result.revenue,
                "full_information_revenue": truth.revenue,
                **pricing,
                "queries": result.query_count,
                "rounds": result.rounds or 0,
            }
        )
    return rows


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_scale_search(
    output_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Visualise each held-out fit and the final all-environment refit."""
    if not rows:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV outputs were still written.")
        return

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    folds = sorted(
        {str(row["held_out_instance"]) for row in rows},
        key=lambda value: (value == "all", value),
    )
    for fold in folds:
        members = [
            row for row in rows if str(row["held_out_instance"]) == fold
        ]
        # Coarse-to-fine passes revisit nearby scales.  Taking the best score
        # at each scale keeps the figure readable while retaining the search.
        by_scale: dict[float, float] = {}
        for row in members:
            scale = float(row["scale"])
            score = float(row["score"])
            by_scale[scale] = min(score, by_scale.get(scale, math.inf))
        points = sorted(by_scale.items())
        axis.plot(
            [point[0] for point in points],
            [100.0 * point[1] for point in points],
            linewidth=2.2 if fold == "all" else 1.1,
            alpha=1.0 if fold == "all" else 0.55,
            label="all-environment refit" if fold == "all" else f"hold out {fold}",
        )
    axis.set_xlabel("Uniform provisional-valuation scale")
    axis.set_ylabel("Mean normalised pricing objective (%)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "payment_error_by_scale.png", dpi=180)
    figure.savefig(output_dir / "payment_error_by_scale.pdf")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    artefacts = _load_artefacts(_benchmark_paths(args.benchmark_dir))
    all_observations = {
        key: observations_from_artefact(artefact)
        for key, artefact in artefacts.items()
    }

    prediction_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    instance_indices = sorted({seed for _, seed in artefacts})
    for held_out_instance in instance_indices:
        train_artefacts = [
            artefact
            for (_domain, seed), artefact in sorted(artefacts.items())
            if seed != held_out_instance
        ]
        test_items = [
            (domain, artefact)
            for (domain, seed), artefact in sorted(artefacts.items())
            if seed == held_out_instance
        ]
        test_artefacts = [artefact for _domain, artefact in test_items]
        calibration, training_score, trace = _fit_uniform_payment_scale(
            train_artefacts, args
        )
        for row in trace:
            search_rows.append(
                {"held_out_instance": held_out_instance, **row}
            )

        raw_pricing_score = _pricing_objective(
            test_artefacts, RAW, args.objective
        )
        calibrated_pricing_score = _pricing_objective(
            test_artefacts, calibration, args.objective
        )
        fold_rows.append(
            {
                "held_out_instance": held_out_instance,
                "train_environment_count": len(train_artefacts),
                "test_environment_count": len(test_artefacts),
                "fit_scale": calibration.scale,
                "training_payment_objective": training_score,
                "heldout_raw_payment_objective": raw_pricing_score,
                "heldout_calibrated_payment_objective": (
                    calibrated_pricing_score
                ),
                "heldout_improvement": (
                    raw_pricing_score - calibrated_pricing_score
                ),
                "heldout_relative_improvement": (
                    (raw_pricing_score - calibrated_pricing_score)
                    / raw_pricing_score
                    if raw_pricing_score
                    else 0.0
                ),
            }
        )

        test_observations = [
            observation
            for domain, artefact in test_items
            for observation in all_observations[
                (domain, int(artefact["seed"]))
            ]
        ]
        for variant, candidate in (
            ("raw", RAW),
            ("calibrated", calibration),
        ):
            metrics = evaluate_predictions(test_observations, candidate)
            prediction_rows.append(
                {
                    "held_out_instance": held_out_instance,
                    "variant": variant,
                    "family": candidate.family,
                    "scale": candidate.scale,
                    "size_gamma": candidate.size_gamma,
                    "size_threshold": candidate.size_threshold,
                    **metrics,
                }
            )

        for _domain, artefact in test_items:
            mechanism_rows.extend(
                _mechanism_rows(
                    artefact, variant="raw", calibration=RAW, args=args
                )
            )
            mechanism_rows.extend(
                _mechanism_rows(
                    artefact,
                    variant="calibrated",
                    calibration=calibration,
                    args=args,
                )
            )

    checks: list[dict[str, Any]] = []
    for mechanism in ("initial", "sealed", "clock"):
        raw_by_case = {
            (row["domain"], row["environment_instance"]): row["efficiency"]
            for row in mechanism_rows
            if row["mechanism"] == mechanism and row["variant"] == "raw"
        }
        calibrated_by_case = {
            (row["domain"], row["environment_instance"]): row["efficiency"]
            for row in mechanism_rows
            if row["mechanism"] == mechanism
            and row["variant"] == "calibrated"
        }
        mean_raw = statistics.fmean(raw_by_case.values())
        mean_calibrated = statistics.fmean(calibrated_by_case.values())
        worst_delta = min(
            calibrated_by_case[key] - raw_by_case[key]
            for key in raw_by_case
        )
        checks.append(
            {
                "mechanism": mechanism,
                "mean_raw_efficiency": mean_raw,
                "mean_calibrated_efficiency": mean_calibrated,
                "mean_efficiency_delta": mean_calibrated - mean_raw,
                "mean_within_tolerance": (
                    mean_calibrated - mean_raw
                    >= -args.efficiency_tolerance - 1e-12
                ),
                "worst_case_efficiency_delta": worst_delta,
            }
        )

    mean_raw_payment_error = statistics.fmean(
        float(row["heldout_raw_payment_objective"]) for row in fold_rows
    )
    mean_calibrated_payment_error = statistics.fmean(
        float(row["heldout_calibrated_payment_objective"])
        for row in fold_rows
    )
    payment_error_improved = (
        mean_calibrated_payment_error < mean_raw_payment_error
    )
    initial_payment_pairs = [
        (
            float(row["payment_error_over_optimum_welfare"]),
            next(
                float(candidate["payment_error_over_optimum_welfare"])
                for candidate in mechanism_rows
                if candidate["domain"] == row["domain"]
                and candidate["environment_instance"]
                == row["environment_instance"]
                and candidate["mechanism"] == "initial"
                and candidate["variant"] == "raw"
            ),
        )
        for row in mechanism_rows
        if row["mechanism"] == "initial" and row["variant"] == "calibrated"
    ]
    payment_cases_improved = sum(
        calibrated < raw - 1e-12
        for calibrated, raw in initial_payment_pairs
    )
    payment_case_majority = (
        payment_cases_improved > len(initial_payment_pairs) / 2
    )
    accepted = (
        payment_error_improved
        and payment_case_majority
        and all(check["mean_within_tolerance"] for check in checks)
    )
    final_calibration, final_training_score, final_trace = (
        _fit_uniform_payment_scale(list(artefacts.values()), args)
    )
    for row in final_trace:
        search_rows.append({"held_out_instance": "all", **row})
    final_calibration = ValueCalibration(
        family="uniform",
        scale=final_calibration.scale,
        size_gamma=1.0,
        size_threshold=3,
        budget_cap=True,
        provenance={
            "method": "three_domain_leave_one_instance_index_out",
            "objective": args.objective,
            "selected_family": "uniform",
            "accepted": accepted,
            "domains": sorted(GENERATED_CALIBRATION_DOMAINS),
            "environment_instances": instance_indices,
            "environment_count": len(artefacts),
            "heldout_mean_raw_payment_objective": (
                mean_raw_payment_error
            ),
            "heldout_mean_calibrated_payment_objective": (
                mean_calibrated_payment_error
            ),
            "heldout_payment_cases_improved": payment_cases_improved,
            "heldout_payment_case_count": len(initial_payment_pairs),
            "mean_efficiency_tolerance": args.efficiency_tolerance,
            "final_all_environment_objective": final_training_score,
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        prediction_rows, args.output_dir / "heldout_prediction_metrics.csv"
    )
    _write_csv(fold_rows, args.output_dir / "heldout_payment_metrics.csv")
    _write_csv(search_rows, args.output_dir / "payment_scale_search.csv")
    _plot_scale_search(args.output_dir, search_rows)
    _write_csv(mechanism_rows, args.output_dir / "mechanism_validation.csv")
    _write_csv(checks, args.output_dir / "acceptance_checks.csv")
    candidate_path = write_calibration_config(
        final_calibration, args.output_dir / "pv_calibration_candidate.json"
    )
    accepted_path = None
    if accepted:
        accepted_path = write_calibration_config(
            final_calibration, args.output_dir / "pv_calibration.json"
        )
    summary = {
        "accepted": accepted,
        "selected_family": "uniform",
        "selection_objective": args.objective,
        "environment_count": len(artefacts),
        "domains": sorted(GENERATED_CALIBRATION_DOMAINS),
        "environment_instances": instance_indices,
        "mean_heldout_raw_payment_objective": mean_raw_payment_error,
        "mean_heldout_calibrated_payment_objective": (
            mean_calibrated_payment_error
        ),
        "payment_error_improved_over_raw": payment_error_improved,
        "payment_cases_improved": payment_cases_improved,
        "payment_case_count": len(initial_payment_pairs),
        "payment_case_majority_improved": payment_case_majority,
        "mean_efficiency_tolerance": args.efficiency_tolerance,
        "folds": fold_rows,
        "mechanism_acceptance_checks": checks,
        "final_calibration": final_calibration.to_dict(),
        "candidate_config": str(candidate_path),
        "accepted_config": None if accepted_path is None else str(accepted_path),
    }
    summary_path = args.output_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("Selected family: uniform")
    print(
        f"Held-out {args.objective}: raw={mean_raw_payment_error:.4f}, "
        f"calibrated={mean_calibrated_payment_error:.4f}"
    )
    print(f"Final all-environment scale: {final_calibration.scale:.4f}")
    for check in checks:
        print(
            f"{check['mechanism']}: mean efficiency "
            f"{check['mean_raw_efficiency']:.3f} -> "
            f"{check['mean_calibrated_efficiency']:.3f}; "
            f"mean delta={check['mean_efficiency_delta']:+.3f}; "
            f"worst case={check['worst_case_efficiency_delta']:+.3f}"
        )
    print(f"Calibration accepted: {accepted}")
    print(f"Report: {summary_path}")
    if not accepted:
        print(
            "The candidate config was retained for diagnosis, but no accepted "
            "pv_calibration.json was written."
        )
    return 0 if accepted or args.allow_rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())
