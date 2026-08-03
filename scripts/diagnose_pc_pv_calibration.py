#!/usr/bin/env python3
"""Cross-fit PV calibration across two frozen 8x8 PC environments.

This is an in-domain diagnostic, not an out-of-domain calibration procedure:
each fold fits one PC environment and evaluates the other. It makes no LLM
calls and never writes an accepted runtime calibration.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
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
    FITTING_OBJECTIVES,
    PvObservation,
    evaluate_predictions,
    fit_calibration,
)
from auctionlab.experiments.runner import run_sealed_vcg_experiment  # noqa: E402
from auctionlab.instances.structured_spec import (  # noqa: E402
    make_pc_build_scenario_from_spec,
)
from auctionlab.llm.clients import MockLlmClient  # noqa: E402
from auctionlab.llm.frozen_elicitation import (  # noqa: E402
    FrozenElicitationPack,
    load_frozen_elicitation_pack,
    validate_pack_for_scenario,
)
from auctionlab.llm.person_simulator import LlmPersonSimulator  # noqa: E402
from auctionlab.llm.proxies import (  # noqa: E402
    LlmAuctionProxyAdapter,
    LlmInferredXorProxy,
)
from auctionlab.llm.value_calibration import ValueCalibration  # noqa: E402


RAW = ValueCalibration(family="none")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-spec", type=Path, required=True)
    parser.add_argument("--pack", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=list(FITTING_OBJECTIVES),
        default="budget_normalized_mae",
    )
    parser.add_argument("--size-threshold", type=int, default=3)
    parser.add_argument("--grid-steps", type=int, default=21)
    parser.add_argument("--grid-passes", type=int, default=4)
    parser.add_argument("--min-exponential-improvement", type=float, default=0.01)
    parser.add_argument("--sealed-max-rounds", type=int, default=25)
    parser.add_argument("--clock-max-rounds", type=int, default=50)
    parser.add_argument("--clock-price-step", type=float, default=50.0)
    parser.add_argument("--clock-top-k", type=int, default=3)
    parser.add_argument("--verbose-mechanisms", action="store_true")
    return parser.parse_args(argv)


def _load_case(
    pack_path: Path,
    scenario_spec: Path,
) -> tuple[str, Any, FrozenElicitationPack]:
    pack = load_frozen_elicitation_pack(pack_path)
    if len(pack.items) != 8 or len(pack.bidder_ids) != 8:
        raise ValueError(
            f"{pack_path}: expected an 8x8 pack, got "
            f"{len(pack.items)}x{len(pack.bidder_ids)}"
        )
    seed = int(pack.scenario_seed)
    scenario = make_pc_build_scenario_from_spec(
        scenario_spec,
        8,
        8,
        seed=seed,
        selection_policy=str(pack.selection_policy),
    )
    validate_pack_for_scenario(
        pack, scenario, scenario_spec_path=scenario_spec
    )
    return f"pc_seed_{seed}", scenario, pack


def _observations(label: str, scenario: Any, pack: FrozenElicitationPack):
    rows: list[PvObservation] = []
    for bidder_id in pack.bidder_ids:
        entry = pack.bidders[bidder_id]
        if entry.raw_pv_values is None:
            continue
        truth = scenario.instance.valuations[bidder_id]
        budget = (
            None
            if entry.interest_map is None
            else entry.interest_map.budget_hint
        )
        for bundle, raw_value in entry.raw_pv_values.items():
            if bundle not in truth:
                continue
            rows.append(
                PvObservation(
                    domain=label,
                    seed=int(pack.scenario_seed),
                    bidder_id=bidder_id,
                    bundle=bundle,
                    raw_value=float(raw_value),
                    true_value=float(truth[bundle]),
                    disclosed_budget=(
                        None if budget is None else float(budget)
                    ),
                )
            )
    if not rows:
        raise ValueError(f"{label}: no raw PV observations")
    return rows


def _fit(observations, family: str, args: argparse.Namespace):
    return fit_calibration(
        observations,
        objective=args.objective,
        family=family,
        size_threshold=args.size_threshold,
        budget_cap=True,
        steps=args.grid_steps,
        passes=args.grid_passes,
    ).calibration


def _adapters(scenario, pack, calibration):
    adapters = []
    for bidder_id in scenario.instance.bidder_ids:
        entry = pack.bidders[bidder_id]
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
        proxy.replay_elicitation(
            nl_question=entry.nl_question,
            nl_answer=entry.nl_answer,
            interest_map=entry.interest_map,
            provisional_raw_values=entry.raw_pv_values,
        )
        adapters.append(
            LlmAuctionProxyAdapter(
                bidder_id=bidder_id,
                proxy=proxy,
                candidate_bundles=list(entry.candidate_bundles),
            )
        )
    return adapters


def _true_welfare(instance, allocation: Mapping[str, frozenset[str]]) -> float:
    return sum(
        instance.value_of(bidder_id, bundle)
        for bidder_id, bundle in allocation.items()
    )


def _run_mechanisms(label, scenario, pack, variant, calibration, args):
    optimum = run_sealed_vcg_experiment(scenario.instance)
    stream = None if args.verbose_mechanisms else io.StringIO()
    output_context = (
        contextlib.nullcontext()
        if stream is None
        else contextlib.redirect_stdout(stream)
    )
    with output_context:
        initial = run_proxy_sealed_vcg_trajectory(
            scenario.instance,
            _adapters(scenario, pack, calibration),
            ProxySealedConfig(elicitation_rounds=0),
        )[-1]
        sealed = run_proxy_sealed_vcg_trajectory(
            scenario.instance,
            _adapters(scenario, pack, calibration),
            ProxySealedConfig(
                elicitation_rounds=args.sealed_max_rounds,
                feedback_rule="competitive",
                stopping_rule="no_new_refinements",
                loser_challenger_policy="off",
            ),
        )[-1]
        clock = run_proxy_clock_experiment(
            scenario.instance,
            _adapters(scenario, pack, calibration),
            ClockConfig(
                max_rounds=args.clock_max_rounds,
                price_step=args.clock_price_step,
            ),
            ProxyClockConfig(
                top_k=args.clock_top_k,
                elicited=True,
                margin_threshold=100,
                tie_threshold=100,
                top_k_frontier_policy="allocation_pivotal",
                allocation_counterfactual_frontier=True,
            ),
        )
    rows = []
    for mechanism, result in (
        ("initial", initial),
        ("sealed", sealed),
        ("clock", clock),
    ):
        welfare = _true_welfare(scenario.instance, result.allocation)
        rows.append(
            {
                "environment": label,
                "variant": variant,
                "family": calibration.family,
                "scale": calibration.scale,
                "size_gamma": calibration.size_gamma,
                "mechanism": mechanism,
                "true_welfare": welfare,
                "full_information_welfare": optimum.welfare,
                "efficiency": welfare / optimum.welfare,
                "revenue": result.revenue,
                "full_information_revenue": optimum.revenue,
                "revenue_absolute_error": abs(result.revenue - optimum.revenue),
                "queries": result.query_count,
                "rounds": result.rounds or 0,
            }
        )
    return rows


def _write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cases = {}
    for path in args.pack:
        label, scenario, pack = _load_case(path, args.scenario_spec)
        if label in cases:
            raise SystemExit(f"Duplicate environment {label}")
        cases[label] = (scenario, pack, _observations(label, scenario, pack))
    if len(cases) != 2:
        raise SystemExit("Exactly two distinct PC environments are required")

    prediction_rows: list[dict[str, Any]] = []
    fold_fits: dict[str, dict[str, ValueCalibration]] = {}
    labels = sorted(cases)
    for held_out in labels:
        train_label = next(label for label in labels if label != held_out)
        test = cases[held_out][2]
        train = cases[train_label][2]
        fold_fits[held_out] = {}
        for family in ("uniform", "exponential"):
            calibration = _fit(train, family, args)
            fold_fits[held_out][family] = calibration
            prediction_rows.append(
                {
                    "trained_on": train_label,
                    "held_out_environment": held_out,
                    "variant": family,
                    "scale": calibration.scale,
                    "size_gamma": calibration.size_gamma,
                    **evaluate_predictions(test, calibration),
                }
            )
        prediction_rows.append(
            {
                "trained_on": train_label,
                "held_out_environment": held_out,
                "variant": "raw",
                "scale": 1.0,
                "size_gamma": 1.0,
                **evaluate_predictions(test, RAW),
            }
        )

    mean_error = {
        variant: statistics.fmean(
            float(row[args.objective])
            for row in prediction_rows
            if row["variant"] == variant
        )
        for variant in ("raw", "uniform", "exponential")
    }
    relative_exp_gain = (
        (mean_error["uniform"] - mean_error["exponential"])
        / mean_error["uniform"]
        if mean_error["uniform"]
        else 0
    )
    selected = (
        "exponential"
        if relative_exp_gain >= args.min_exponential_improvement
        else "uniform"
    )

    mechanism_rows = []
    for label in labels:
        scenario, pack, _ = cases[label]
        mechanism_rows.extend(
            _run_mechanisms(label, scenario, pack, "raw", RAW, args)
        )
        mechanism_rows.extend(
            _run_mechanisms(
                label,
                scenario,
                pack,
                "cross_fitted",
                fold_fits[label][selected],
                args,
            )
        )

    comparisons = []
    for mechanism in ("initial", "sealed", "clock"):
        for label in labels:
            raw = next(
                row
                for row in mechanism_rows
                if row["environment"] == label
                and row["mechanism"] == mechanism
                and row["variant"] == "raw"
            )
            fitted = next(
                row
                for row in mechanism_rows
                if row["environment"] == label
                and row["mechanism"] == mechanism
                and row["variant"] == "cross_fitted"
            )
            comparisons.append(
                {
                    "environment": label,
                    "mechanism": mechanism,
                    "raw_efficiency": raw["efficiency"],
                    "cross_fitted_efficiency": fitted["efficiency"],
                    "efficiency_delta": (
                        fitted["efficiency"] - raw["efficiency"]
                    ),
                    "raw_queries": raw["queries"],
                    "cross_fitted_queries": fitted["queries"],
                    "query_delta": fitted["queries"] - raw["queries"],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        prediction_rows, args.output_dir / "crossfit_prediction_metrics.csv"
    )
    _write_csv(
        mechanism_rows, args.output_dir / "crossfit_mechanism_metrics.csv"
    )
    _write_csv(comparisons, args.output_dir / "crossfit_comparisons.csv")
    summary = {
        "diagnostic_only": True,
        "reason": (
            "PC environments belong to the experimental domain and must not "
            "select the final out-of-domain calibration."
        ),
        "selected_family_for_diagnostic": selected,
        "mean_crossfit_prediction_error": mean_error,
        "exponential_relative_improvement_over_uniform": relative_exp_gain,
        "fold_calibrations": {
            label: {
                family: calibration.to_dict()
                for family, calibration in fits.items()
            }
            for label, fits in fold_fits.items()
        },
        "mechanism_comparisons": comparisons,
    }
    summary_path = args.output_dir / "crossfit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Diagnostic family: {selected}")
    print(
        f"Cross-fit {args.objective}: raw={mean_error['raw']:.4f}, "
        f"uniform={mean_error['uniform']:.4f}, "
        f"exponential={mean_error['exponential']:.4f}"
    )
    for row in comparisons:
        print(
            f"{row['environment']} {row['mechanism']}: "
            f"eff {row['raw_efficiency']:.3f} -> "
            f"{row['cross_fitted_efficiency']:.3f}; "
            f"queries {row['raw_queries']} -> "
            f"{row['cross_fitted_queries']}"
        )
    print(f"Report: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
