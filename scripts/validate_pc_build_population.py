#!/usr/bin/env python3
"""Revalidate a frozen PC-build population without making any LLM calls."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from auctionlab.instances.population_design import (
    freeze_validated_nested_orders,
    population_coverage_report,
    validate_nested_scalability_samples,
)
from auctionlab.instances.scenario_spec import (
    load_scenario_profile_spec,
    write_scenario_profile_spec,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-spec", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=Path(
            "scenarios/pc_build_v2/population_design_16x16.json"
        ),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the spec here only when every validation check passes.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(range(4, 11)),
    )
    parser.add_argument("--fixed-size", type=int, default=8)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--skip-economic-sample-validation",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    design = json.loads(args.design.read_text(encoding="utf-8"))
    spec = load_scenario_profile_spec(args.scenario_spec)
    strata = {
        bidder["bidder_id"]: bidder["stratum"]
        for bidder in design["bidder_archetypes"]
    }
    population = population_coverage_report(
        spec,
        bidder_strata=strata,
        constraints=design["population_constraints"],
    )
    samples = validate_nested_scalability_samples(
        spec,
        seeds=args.validation_seeds,
        sizes=args.sizes,
        fixed_size=args.fixed_size,
        constraints=design["sample_constraints"],
        include_economic=not args.skip_economic_sample_validation,
    )
    passed = population["passed"] and samples["passed"]
    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_spec": str(args.scenario_spec),
        "design": str(args.design),
        "passed": passed,
        "population_coverage": population,
        "sample_validation": samples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Validation {'PASSED' if passed else 'FAILED'}: {args.report}")
    for violation in population["violations"]:
        print(f"  - population: {violation}")
    for violation in samples["violations"]:
        print(f"  - sample: {violation}")
    if not passed:
        return 1
    if args.output is not None:
        generation = dict(spec.generation or {})
        generation["population_constraints"] = design[
            "population_constraints"
        ]
        generation["sample_constraints"] = design["sample_constraints"]
        generation["coverage_orders"] = freeze_validated_nested_orders(
            spec,
            samples,
        )
        generation["coverage_order_validation"] = {
            "seeds": samples["seeds"],
            "sizes": samples["sizes"],
            "fixed_size": samples["fixed_size"],
            "include_economic": samples["include_economic"],
        }
        spec = spec.model_copy(update={"generation": generation})
        write_scenario_profile_spec(spec, args.output)
        print(f"Wrote validated population: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
