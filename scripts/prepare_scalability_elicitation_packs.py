#!/usr/bin/env python3
"""Generate catalogue-level master packs and project the scalability grid.

Initial elicitation depends on the selected goods, but not on the number of
rival bidders. For each seed this script therefore generates one master pack
per goods count, with enough bidders to cover both the fixed-bidder and joint
sweeps, then projects ordered bidder subsets into the 19 case directories
consumed by ``run_scalability_experiment.py``.

Arguments after ``--`` are forwarded to ``generate_frozen_elicitation.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.instances.population_design import (  # noqa: E402
    population_coverage_report,
    validate_nested_scalability_samples,
)
from auctionlab.instances.scenario_spec import (  # noqa: E402
    load_scenario_profile_spec,
)
from auctionlab.instances.structured_spec import (  # noqa: E402
    make_pc_build_scenario_from_spec,
)
from auctionlab.llm.frozen_elicitation import (  # noqa: E402
    file_sha256,
    load_frozen_elicitation_pack,
    project_frozen_pack_to_bidders,
    validate_pack_for_scenario,
    write_frozen_elicitation_pack,
)
from scripts.run_scalability_experiment import (  # noqa: E402
    ScalabilityRun,
    build_scalability_runs,
)


@dataclass(frozen=True)
class MasterPackPlan:
    seed: int
    num_goods: int
    num_bidders: int
    path: Path


def master_bidder_count(
    num_goods: int,
    *,
    fixed_size: int,
    max_size: int,
) -> int:
    """Smallest master bidder count covering all three scaling paths."""
    return max(
        fixed_size,
        num_goods,
        max_size if num_goods == fixed_size else 0,
    )


def build_master_pack_plans(
    *,
    sizes: Sequence[int],
    fixed_size: int,
    seeds: Sequence[int],
    output_dir: Path,
) -> list[MasterPackPlan]:
    normalized = sorted(set(sizes))
    if not normalized:
        raise ValueError("sizes must not be empty")
    if fixed_size not in normalized:
        raise ValueError("fixed_size must also appear in sizes")
    max_size = max(normalized)
    return [
        MasterPackPlan(
            seed=seed,
            num_goods=num_goods,
            num_bidders=master_bidder_count(
                num_goods,
                fixed_size=fixed_size,
                max_size=max_size,
            ),
            path=(
                output_dir
                / f"seed_{seed}"
                / "masters"
                / (
                    f"goods_{num_goods}_master_"
                    f"{num_goods}x"
                    f"{master_bidder_count(num_goods, fixed_size=fixed_size, max_size=max_size)}"
                    ".json"
                )
            ),
        )
        for seed in seeds
        for num_goods in normalized
    ]


def master_for_run(
    run: ScalabilityRun,
    plans: Sequence[MasterPackPlan],
) -> MasterPackPlan:
    matches = [
        plan
        for plan in plans
        if plan.seed == run.seed and plan.num_goods == run.num_goods
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one master for seed={run.seed}, goods={run.num_goods}; "
            f"found {len(matches)}"
        )
    master = matches[0]
    if run.num_bidders > master.num_bidders:
        raise ValueError(
            f"{run.case_name} needs {run.num_bidders} bidders but master "
            f"contains {master.num_bidders}"
        )
    return master


def _token_totals(calls: Sequence[dict[str, Any]]) -> dict[str, int]:
    def _logical(call: dict[str, Any], live: str, cached: str) -> int:
        cached_value = call.get(cached)
        return int(
            cached_value
            if cached_value is not None
            else (call.get(live) or 0)
        )

    return {
        "input_tokens": sum(
            _logical(call, "input_tokens", "cached_input_tokens")
            for call in calls
        ),
        "output_tokens": sum(
            _logical(call, "output_tokens", "cached_output_tokens")
            for call in calls
        ),
        "total_tokens": sum(
            _logical(call, "input_tokens", "cached_input_tokens")
            + _logical(call, "output_tokens", "cached_output_tokens")
            for call in calls
        ),
        "calls": len(calls),
    }


_MANAGED_LIVE_FLAGS = {
    "--scenario",
    "--scenario-spec",
    "--num-goods",
    "--num-bidders",
    "--scenario-seed",
    "--selection-policy",
    "--seed-type",
    "--output",
    "--write-elicitation-pack",
    "--prepare-elicitation-only",
    "--log-dir",
}

_SECRET_FLAGS = {
    "--api-key",
    "--person-api-key",
    "--proxy-api-key",
    "--verifier-api-key",
}


def redact_command(command: Sequence[str]) -> list[str]:
    """Return a manifest/log-safe command with credential values removed."""
    redacted: list[str] = []
    hide_next = False
    for token in command:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        flag, separator, _value = token.partition("=")
        if flag in _SECRET_FLAGS:
            if separator:
                redacted.append(f"{flag}=<redacted>")
            else:
                redacted.append(token)
                hide_next = True
            continue
        redacted.append(token)
    return redacted


def _validate_live_args(live_args: Sequence[str]) -> None:
    conflicts = sorted(
        {
            token.split("=", 1)[0]
            for token in live_args
            if token.split("=", 1)[0] in _MANAGED_LIVE_FLAGS
        }
    )
    if conflicts:
        raise ValueError(
            "these arguments are managed by the preparation runner and must "
            f"not be forwarded: {', '.join(conflicts)}"
        )


def build_master_command(
    plan: MasterPackPlan,
    *,
    scenario_spec: Path,
    selection_policy: str,
    live_args: Sequence[str],
) -> list[str]:
    command = [
        sys.executable,
        "scripts/generate_frozen_elicitation.py",
        "--scenario",
        "pc_build",
        "--scenario-spec",
        str(scenario_spec),
        "--num-goods",
        str(plan.num_goods),
        "--num-bidders",
        str(plan.num_bidders),
        "--scenario-seed",
        str(plan.seed),
        "--selection-policy",
        selection_policy,
        "--seed-type",
        "structured",
        "--output",
        str(plan.path),
    ]
    forwarded = list(live_args)

    def _has_flag(flag: str) -> bool:
        return any(
            token == flag or token.startswith(f"{flag}=")
            for token in forwarded
        )

    for required in (
        "--ask-initial-question",
        "--use-interest-map",
        "--use-provisional-valuations",
    ):
        if not _has_flag(required):
            forwarded.append(required)
    safe_defaults = {
        # OpenAI reasoning models occasionally consume the 1500-token person
        # budget internally before emitting the deliberately short JSON
        # answer.  A larger ceiling does not request a longer disclosure (the
        # prompt still caps it at 95 words), but prevents a rare hard 400 from
        # aborting an otherwise resumable overnight preparation run.
        "--person-nl-max-tokens": "4000",
        # Gemini reasoning models can consume much of the completion budget
        # internally before emitting the JSON interest map. 1500 occasionally
        # produced 60--100 visible tokens and a truncated object; 4000 keeps
        # this compact structured call comfortably parseable.
        "--interest-map-max-tokens": "4000",
        "--pv-max-tokens": "12000",
        "--max-parse-retries": "2",
        "--timeout": "240",
        "--llm-cache-mode": "read-write",
        "--llm-cache-path": str(
            plan.path.parents[2] / "preparation_cache.sqlite"
        ),
    }
    for flag, value in safe_defaults.items():
        if not _has_flag(flag):
            forwarded.extend([flag, value])
    command.extend(
        [
            "--log-dir",
            str(
                plan.path.parent
                / "logs"
                / plan.path.stem
            ),
        ]
    )
    return [*command, *forwarded]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7, 8, 9, 10],
    )
    parser.add_argument("--fixed-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--selection-policy",
        choices=["seeded_sample", "stratified", "coverage_stratified"],
        default="coverage_stratified",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--project-only",
        action="store_true",
        help="Do not generate; require all master packs to exist.",
    )
    parser.add_argument(
        "--regenerate-masters",
        action="store_true",
        help="Regenerate even when an existing master validates.",
    )
    parser.add_argument(
        "--skip-economic-preflight",
        action="store_true",
    )
    parser.add_argument(
        "live_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- forwarded to frozen-pack generation.",
    )
    args = parser.parse_args(argv)
    if args.live_args and args.live_args[0] == "--":
        args.live_args = args.live_args[1:]
    return args


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_scenario_profile_spec(args.scenario_spec)
    generation = spec.generation if isinstance(spec.generation, dict) else {}
    population = population_coverage_report(
        spec,
        constraints=generation.get("population_constraints"),
    )
    samples = validate_nested_scalability_samples(
        spec,
        seeds=args.seeds,
        sizes=args.sizes,
        fixed_size=args.fixed_size,
        constraints=generation.get("sample_constraints"),
        include_economic=not args.skip_economic_preflight,
    )
    return {
        "scenario_spec": str(args.scenario_spec),
        "scenario_spec_sha256": file_sha256(args.scenario_spec),
        "selection_policy": args.selection_policy,
        "population": population,
        "sample_validation": samples,
        "passed": population["passed"] and samples["passed"],
    }


def _scenario(
    *,
    scenario_spec: Path,
    plan_or_run: MasterPackPlan | ScalabilityRun,
    selection_policy: str,
):
    return make_pc_build_scenario_from_spec(
        scenario_spec,
        plan_or_run.num_goods,
        plan_or_run.num_bidders,
        seed=plan_or_run.seed,
        selection_policy=selection_policy,
    )


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_live_args(args.live_args)
        runs = build_scalability_runs(
            sizes=args.sizes,
            fixed_size=args.fixed_size,
            seeds=args.seeds,
        )
        plans = build_master_pack_plans(
            sizes=args.sizes,
            fixed_size=args.fixed_size,
            seeds=args.seeds,
            output_dir=args.output_dir,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    preflight = _preflight(args)
    preflight_path = args.output_dir / "preparation_preflight.json"
    _write_manifest(preflight_path, preflight)
    if not preflight["passed"]:
        print(f"Preparation preflight FAILED: {preflight_path}")
        return 2
    print(f"Preparation preflight passed: {preflight_path}")

    manifest: dict[str, Any] = {
        "format": "auctionlab.scalability_elicitation_preparation",
        "version": 1,
        "scenario_spec": str(args.scenario_spec),
        "scenario_spec_sha256": file_sha256(args.scenario_spec),
        "sizes": sorted(set(args.sizes)),
        "fixed_size": args.fixed_size,
        "seeds": args.seeds,
        "selection_policy": args.selection_policy,
        "masters": [],
        "cases": [],
    }
    manifest_path = args.output_dir / "preparation_manifest.json"

    for plan in plans:
        command = build_master_command(
            plan,
            scenario_spec=args.scenario_spec,
            selection_policy=args.selection_policy,
            live_args=args.live_args,
        )
        safe_command = redact_command(command)
        print(shlex.join(safe_command), flush=True)
        if args.dry_run:
            manifest["masters"].append({
                **asdict(plan),
                "path": str(plan.path),
                "status": "dry_run",
                "command": safe_command,
            })
            continue

        master_scenario = _scenario(
            scenario_spec=args.scenario_spec,
            plan_or_run=plan,
            selection_policy=args.selection_policy,
        )
        status = "generated"
        if plan.path.exists() and (
            not args.regenerate_masters or args.project_only
        ):
            try:
                existing = load_frozen_elicitation_pack(plan.path)
                validate_pack_for_scenario(
                    existing,
                    master_scenario,
                    scenario_spec_path=args.scenario_spec,
                )
                status = "reused_valid"
            except (OSError, ValueError) as exc:
                print(
                    f"Invalid existing master {plan.path}: {exc}",
                    file=sys.stderr,
                )
                return 2
        elif args.project_only:
            print(f"Missing required master: {plan.path}", file=sys.stderr)
            return 2
        else:
            plan.path.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            if completed.returncode != 0:
                manifest["masters"].append({
                    **asdict(plan),
                    "path": str(plan.path),
                    "status": "failed",
                    "return_code": completed.returncode,
                    "command": safe_command,
                })
                _write_manifest(manifest_path, manifest)
                return completed.returncode

        pack = load_frozen_elicitation_pack(plan.path)
        validate_pack_for_scenario(
            pack,
            master_scenario,
            scenario_spec_path=args.scenario_spec,
        )
        manifest["masters"].append({
            **asdict(plan),
            "path": str(plan.path),
            "status": status,
            "sha256": file_sha256(plan.path),
            "tokens": _token_totals(pack.generation_calls),
            "command": safe_command,
        })
        _write_manifest(manifest_path, manifest)

    if args.dry_run:
        _write_manifest(manifest_path, manifest)
        print(
            f"Dry run: {len(plans)} masters would produce "
            f"{len(runs)} projected cases. Manifest: {manifest_path}"
        )
        return 0

    for run in runs:
        master_plan = master_for_run(run, plans)
        parent = load_frozen_elicitation_pack(master_plan.path)
        target_scenario = _scenario(
            scenario_spec=args.scenario_spec,
            plan_or_run=run,
            selection_policy=args.selection_policy,
        )
        # Coverage-stratified samples are nested ordered subsets, but not
        # necessarily literal prefixes: e.g. the 8x4 bidder sample can select
        # positions [2, 5, 6, 9] from the 8x10 master. The projection helper
        # below already validates both containment and preserved parent order,
        # which is the actual requirement for safely reusing bidder packs.
        projected = project_frozen_pack_to_bidders(
            parent,
            target_scenario,
            scenario_spec_path=args.scenario_spec,
        )
        case_path = (
            args.output_dir
            / f"seed_{run.seed}"
            / run.case_name
            / "frozen_elicitation.json"
        )
        write_frozen_elicitation_pack(projected, case_path)
        manifest["cases"].append({
            "seed": run.seed,
            "series": run.series,
            "case": run.case_name,
            "num_goods": run.num_goods,
            "num_bidders": run.num_bidders,
            "path": str(case_path),
            "sha256": file_sha256(case_path),
            "parent_master": str(master_plan.path),
            "parent_master_sha256": file_sha256(master_plan.path),
            "logical_tokens": _token_totals(projected.generation_calls),
            "status": "projected_valid",
        })
        _write_manifest(manifest_path, manifest)

    physical = {
        key: sum(
            int(row["tokens"][key])
            for row in manifest["masters"]
            if "tokens" in row
        )
        for key in ("input_tokens", "output_tokens", "total_tokens", "calls")
    }
    manifest["summary"] = {
        "master_count": len(plans),
        "case_count": len(runs),
        "physical_generation_totals": physical,
    }
    _write_manifest(manifest_path, manifest)
    print(
        f"Prepared {len(plans)} master packs and {len(runs)} validated case "
        f"packs. Manifest: {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
