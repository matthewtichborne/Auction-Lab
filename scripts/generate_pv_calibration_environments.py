#!/usr/bin/env python3
"""Generate small out-of-domain PV calibration environments.

Each domain/instance uses one initial environment-model call. Invalid JSON or
invalid preference structure is repaired with a bounded retry. Completed
environment files are resumable and no person/proxy calls are made by this
command.  Instance zero retains the original filename, so extending from one
to three instances per domain generates exactly six new environments.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.experiments.pv_calibration_environments import (  # noqa: E402
    GENERATED_CALIBRATION_DOMAINS,
    build_environment_prompt,
    build_generated_environment_scenario,
    environment_file_name,
    load_generated_environment,
    validate_environment_payload,
)
from auctionlab.experiments.runner import run_sealed_vcg_experiment  # noqa: E402
from auctionlab.llm.parsing import extract_json_object  # noqa: E402
from scripts.generate_pc_build_population import build_client  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=[*GENERATED_CALIBRATION_DOMAINS, "all"],
        default=["all"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--instances-per-domain",
        type=int,
        default=3,
        help=(
            "Number of independent environments per domain (default: 3). "
            "Instance zero reuses the original filename."
        ),
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-repair-retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if "all" in args.domains:
        args.domains = list(GENERATED_CALIBRATION_DOMAINS)
    if args.instances_per_domain < 1:
        parser.error("--instances-per-domain must be at least 1")
    return args


def _economic_validation(payload: dict[str, Any]) -> dict[str, Any]:
    scenario = build_generated_environment_scenario(payload)
    optimum = run_sealed_vcg_experiment(scenario.instance)
    winner_values = [
        scenario.instance.value_of(bidder_id, bundle)
        for bidder_id, bundle in optimum.allocation.items()
        if bundle
    ]
    if len(winner_values) < 2:
        raise ValueError(
            "full-information allocation must have at least two winners"
        )
    largest_share = max(winner_values) / optimum.welfare if optimum.welfare else 1
    if largest_share > 0.9:
        raise ValueError(
            "largest winner welfare share must be <= 0.90, got "
            f"{largest_share:.3f}"
        )
    return {
        "full_information_welfare": optimum.welfare,
        "full_information_revenue": optimum.revenue,
        "winner_count": len(winner_values),
        "largest_winner_welfare_share": largest_share,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    planned = [
        (
            domain,
            instance_index,
            args.output_dir / environment_file_name(domain, instance_index),
        )
        for domain in args.domains
        for instance_index in range(args.instances_per_domain)
    ]
    if args.dry_run:
        print(f"Would generate {len(planned)} environment(s):")
        for domain, instance_index, path in planned:
            print(f"  {domain} instance={instance_index}: {path}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_generation"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = build_client(
        args.provider,
        args.model,
        args.api_key,
        args.temperature,
        args.max_tokens,
        base_url=args.base_url,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_effort,
    )
    manifest: list[dict[str, Any]] = []

    for domain, instance_index, output_path in planned:
        if output_path.exists() and not args.overwrite:
            try:
                existing = load_generated_environment(output_path)
                diagnostics = _economic_validation(existing)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"Invalid existing {output_path}: {exc}; regenerating")
            else:
                if existing.get("instance_index", 0) != instance_index:
                    raise ValueError(
                        "environment instance index does not match filename"
                    )
                print(
                    f"SKIP {domain} instance={instance_index}: complete "
                    f"({output_path})"
                )
                manifest.append(
                    {
                        "domain": domain,
                        "instance_index": instance_index,
                        "status": "skipped_complete",
                        "path": str(output_path),
                        "diagnostics": diagnostics,
                    }
                )
                continue

        last_error: str | None = None
        attempts: list[dict[str, Any]] = []
        for attempt in range(args.max_repair_retries + 1):
            prompt = build_environment_prompt(
                domain,
                instance_index=instance_index,
                repair_error=last_error,
            )
            stem = f"{domain}_instance{instance_index}"
            prompt_path = raw_dir / f"{stem}_attempt_{attempt + 1}_prompt.txt"
            response_path = (
                raw_dir / f"{stem}_attempt_{attempt + 1}_response.txt"
            )
            prompt_path.write_text(prompt, encoding="utf-8")
            print(
                f"{domain} instance={instance_index}: generation attempt "
                f"{attempt + 1}/"
                f"{args.max_repair_retries + 1}",
                flush=True,
            )
            raw = client.complete(prompt)
            response_path.write_text(raw, encoding="utf-8")
            try:
                parsed = extract_json_object(raw)
                validated = validate_environment_payload(
                    parsed,
                    expected_domain=domain,
                    expected_instance_index=instance_index,
                )
                diagnostics = _economic_validation(validated)
            except (ValueError, TypeError, KeyError) as exc:
                last_error = str(exc)
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": "invalid",
                        "error": last_error,
                        "prompt": str(prompt_path),
                        "response": str(response_path),
                    }
                )
                print(f"  invalid: {last_error}")
                continue

            validated["generation"] = {
                "provider": args.provider,
                "model": args.model,
                "temperature": args.temperature,
                "reasoning_effort": args.reasoning_effort,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "attempts": attempt + 1,
                "diagnostics": diagnostics,
            }
            output_path.write_text(
                json.dumps(validated, indent=2) + "\n", encoding="utf-8"
            )
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "status": "validated",
                    "prompt": str(prompt_path),
                    "response": str(response_path),
                }
            )
            print(
                f"  wrote {output_path} "
                f"(winners={diagnostics['winner_count']}, "
                f"welfare={diagnostics['full_information_welfare']:.1f})"
            )
            manifest.append(
                {
                    "domain": domain,
                    "instance_index": instance_index,
                    "status": "completed",
                    "path": str(output_path),
                    "attempts": attempts,
                    "diagnostics": diagnostics,
                }
            )
            break
        else:
            manifest.append(
                {
                    "domain": domain,
                    "instance_index": instance_index,
                    "status": "failed",
                    "path": str(output_path),
                    "attempts": attempts,
                    "error": last_error,
                }
            )

    manifest_path = args.output_dir / "environment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    failures = [row for row in manifest if row["status"] == "failed"]
    print(f"Manifest: {manifest_path} ({len(failures)} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
