#!/usr/bin/env python3
"""Prepare frozen out-of-domain benchmark artefacts for PV calibration.

This is the only step in the calibration pipeline that makes LLM calls. For
each (domain, seed) it runs the *current* elicitation design end to end --
canonical opening question, person answer with verification, proxy interest
map, interest-map candidate generation, raw provisional valuation -- and
freezes the raw predictions alongside the hidden ground truth.

The proxy provider/model is the estimator being calibrated. The person and
verifier models only stand in for the human; changing them changes the
benchmark, so all three are recorded in every artefact.

Nothing here writes into the PC-build experiment outputs.

Recommended generated-environment workflow (three domains, three independent
instances per domain)::

    ./venv/bin/python scripts/prepare_pv_calibration_benchmark.py \\
      --domains all \\
      --seeds 0 1 2 \\
      --environment-dir outputs/pv_calibration/environments \\
      --output-dir outputs/pv_calibration/benchmark \\
      --person-provider openai --person-model gpt-5.6-sol \\
      --proxy-provider gemini --proxy-model gemini-3.6-flash \\
      --verifier-provider openai --verifier-model gpt-5.6-sol \\
      --llm-cache-mode read-write \\
      --llm-cache-path cache/pv_calibration.sqlite

Re-running the same command resumes: completed artefacts are skipped unless
``--overwrite`` is given, and ``--llm-cache-mode read-write`` reuses cached
responses for any bidder that was already elicited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.experiments.pv_calibration import (  # noqa: E402
    BENCHMARK_DOMAINS,
    PV_CALIBRATION_BENCHMARK_FORMAT,
    PV_CALIBRATION_BENCHMARK_VERSION,
    artefact_file_name,
    build_benchmark_scenario,
    load_benchmark_artefact,
    sha256_text,
    write_benchmark_artefact,
)
from auctionlab.experiments.pv_calibration_environments import (  # noqa: E402
    GENERATED_CALIBRATION_DOMAINS,
    build_generated_environment_scenario,
    environment_file_name,
    load_generated_environment,
)
from auctionlab.llm.cache import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    CacheStats,
    LlmResponseCache,
)
from auctionlab.llm.frozen_elicitation import (  # noqa: E402
    ModelProvenance,
    scenario_fingerprint,
)
from auctionlab.llm.logging import LlmCallLogger  # noqa: E402
from auctionlab.llm.prompts import (  # noqa: E402
    CANONICAL_OPENING_QUESTION,
    build_provisional_valuation_prompt,
    canonical_opening_question,
)


class BenchmarkPreparationError(RuntimeError):
    """Raised when a bidder's elicitation fails and the policy is fail-closed."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["all"],
        choices=[*BENCHMARK_DOMAINS, "all"],
        help=(
            "Benchmark domains to prepare. With --environment-dir, 'all' "
            "means the three generated calibration domains."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--num-goods", type=int, default=None)
    parser.add_argument("--num-bidders", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--environment-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing generated calibration environments. When "
            "supplied, --seeds selects independent environment instance "
            "indices and these replace the legacy hand-authored fixtures; "
            "--num-goods/--num-bidders are invalid."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate artefacts that already exist (default: resume).",
    )

    for role, default_tokens in (
        ("person", 1500),
        ("proxy", 4000),
        ("verifier", 2000),
    ):
        parser.add_argument(f"--{role}-provider", type=str, default=None)
        parser.add_argument(f"--{role}-model", type=str, default=None)
        parser.add_argument(f"--{role}-base-url", type=str, default=None)
        parser.add_argument(f"--{role}-api-key", type=str, default=None)
        parser.add_argument(f"--{role}-temperature", type=float, default=0.0)
        parser.add_argument(
            f"--{role}-max-tokens", type=int, default=default_tokens
        )

    parser.add_argument("--person-nl-max-tokens", type=int, default=1500)
    parser.add_argument("--interest-map-max-tokens", type=int, default=2000)
    parser.add_argument(
        "--pv-max-tokens",
        type=int,
        default=4000,
        help="Token budget for each provisional-valuation call.",
    )
    parser.add_argument(
        "--pv-chunk-size",
        type=int,
        default=0,
        help=(
            "Split a bidder's candidate bundles into PV calls of at most this "
            "many bundles (0 = one call per bidder)."
        ),
    )
    parser.add_argument("--max-candidate-bundles", type=int, default=None)
    parser.add_argument("--max-parse-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--pv-failure-policy",
        choices=["raise", "skip_bidder"],
        default="raise",
        help=(
            "What to do when a bidder's PV call fails. 'raise' (default) "
            "aborts. 'skip_bidder' records the failure and omits that bidder "
            "from the artefact. Neither ever inserts zero values -- a zero "
            "prediction would be scored as a real under-estimate and bias the "
            "fitted scale."
        ),
    )
    parser.add_argument(
        "--interest-map-failure-policy",
        choices=["raise", "all_items"],
        default="raise",
    )
    parser.add_argument(
        "--llm-cache-mode",
        choices=["off", "read-write", "read-only", "refresh"],
        default="read-write",
    )
    parser.add_argument("--llm-cache-path", type=str, default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the planned artefacts and exit without constructing any "
            "client or making any call."
        ),
    )
    args = parser.parse_args(argv)

    if args.domains == ["all"] or "all" in args.domains:
        args.domains = list(
            GENERATED_CALIBRATION_DOMAINS
            if args.environment_dir is not None
            else BENCHMARK_DOMAINS
        )
    for role in ("person", "proxy", "verifier"):
        if not args.dry_run and (
            getattr(args, f"{role}_provider") is None
            or getattr(args, f"{role}_model") is None
        ):
            parser.error(f"--{role}-provider and --{role}-model are required")
    if args.environment_dir is not None:
        invalid = [
            domain
            for domain in args.domains
            if domain not in GENERATED_CALIBRATION_DOMAINS
        ]
        if invalid:
            parser.error(
                "--environment-dir supports exactly the generated domains "
                f"{list(GENERATED_CALIBRATION_DOMAINS)}; invalid={invalid}"
            )
        if args.num_goods is not None or args.num_bidders is not None:
            parser.error(
                "--num-goods/--num-bidders cannot be used with "
                "--environment-dir"
            )
    return args


def _planned_artefacts(args: argparse.Namespace) -> list[tuple[str, int, Path]]:
    return [
        (domain, seed, args.output_dir / artefact_file_name(domain, seed))
        for domain in args.domains
        for seed in args.seeds
    ]


def _prompt_version_hashes() -> dict[str, str]:
    """Hash the prompt text the benchmark is measuring.

    A calibration fitted against one PV prompt does not transfer to a
    different one, so the prompt's identity is frozen with the artefact. The
    probe uses a fixed synthetic input purely to obtain a stable rendering of
    the template.
    """
    probe = build_provisional_valuation_prompt(
        scenario_description="probe",
        item_descriptions={"A": "probe item a", "B": "probe item b"},
        nl_question="probe question",
        nl_answer="probe answer",
        candidate_bundles=[frozenset({"A"}), frozenset({"A", "B"})],
        interest_map=None,
    )
    return {
        "provisional_valuation_prompt_sha256": sha256_text(probe),
        "canonical_opening_question_sha256": sha256_text(
            CANONICAL_OPENING_QUESTION
        ),
    }


def prepare_domain_seed(
    *,
    domain: str,
    seed: int,
    args: argparse.Namespace,
    output_path: Path,
) -> dict[str, Any]:
    """Elicit one (domain, seed) benchmark and write its frozen artefact."""
    # Imported lazily: the live runner pulls in the whole experiment stack,
    # and --dry-run / --help must not pay for that or require provider SDKs.
    from examples.run_live_llm_curated_batch import (
        compute_elicitation_cache,
        make_live_client,
        make_live_persons_for_scenario,
    )

    environment_payload: dict[str, Any] | None = None
    environment_path: Path | None = None
    if args.environment_dir is None:
        scenario = build_benchmark_scenario(
            domain,
            seed=seed,
            num_goods=args.num_goods,
            num_bidders=args.num_bidders,
        )
    else:
        environment_path = (
            args.environment_dir / environment_file_name(domain, seed)
        )
        try:
            environment_payload = load_generated_environment(environment_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise BenchmarkPreparationError(
                f"{domain}: cannot load generated environment "
                f"{environment_path}: {exc}"
            ) from exc
        scenario = build_generated_environment_scenario(environment_payload)
    log_dir = output_path.parent / "generation" / f"{domain}_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = LlmCallLogger(log_dir / "calls.jsonl", append=False)
    cache = (
        LlmResponseCache(args.llm_cache_path)
        if args.llm_cache_mode != "off"
        else None
    )
    cache_stats = CacheStats()

    def client(role: str, max_tokens: int):
        return make_live_client(
            model=getattr(args, f"{role}_model"),
            provider=getattr(args, f"{role}_provider"),
            base_url=getattr(args, f"{role}_base_url"),
            api_key=getattr(args, f"{role}_api_key"),
            temperature=getattr(args, f"{role}_temperature"),
            max_tokens=max_tokens,
            timeout=args.timeout,
            cache=cache,
            cache_mode=args.llm_cache_mode,
            cache_stats=cache_stats,
            llm_role=role,
        )

    started = time.time()
    try:
        persons = make_live_persons_for_scenario(
            scenario,
            model=args.person_model,
            provider=args.person_provider,
            base_url=args.person_base_url,
            api_key=args.person_api_key,
            temperature=args.person_temperature,
            max_tokens=args.person_max_tokens,
            person_nl_max_tokens=args.person_nl_max_tokens,
            timeout=args.timeout,
            logger=logger,
            max_parse_retries=args.max_parse_retries,
            # The person answers from their qualitative seed. Deterministic
            # ground-truth lookups are deliberately NOT enabled: the estimator
            # under test must work from the disclosure alone.
            use_ground_truth=False,
            cache=cache,
            cache_mode=args.llm_cache_mode,
            cache_stats=cache_stats,
            verifier_provider=args.verifier_provider,
            verifier_model=args.verifier_model,
            verifier_base_url=args.verifier_base_url,
            verifier_api_key=args.verifier_api_key,
            verifier_temperature=args.verifier_temperature,
            verifier_max_tokens=args.verifier_max_tokens,
        )

        entries = compute_elicitation_cache(
            scenario=scenario,
            persons=persons,
            use_provisional_valuations=True,
            max_candidate_bundles=args.max_candidate_bundles,
            pv_client=client("proxy", args.pv_max_tokens),
            question_client=None,
            opening_question=canonical_opening_question(),
            interest_map_client=client("proxy", args.interest_map_max_tokens),
            pv_chunk_size=args.pv_chunk_size or None,
            # Fail closed. compute_elicitation_cache's 'zero' policy
            # zero-initialises every candidate bundle, which is a legitimate
            # degraded mode for an auction run but would silently poison a
            # calibration fit with fabricated zero predictions.
            pv_failure_policy="raise",
            interest_map_failure_policy=args.interest_map_failure_policy,
            pv_max_tokens=args.pv_max_tokens,
            max_parse_retries=args.max_parse_retries,
        )
    except Exception as exc:
        if args.pv_failure_policy == "raise":
            raise BenchmarkPreparationError(
                f"{domain} seed={seed}: elicitation failed "
                f"({type(exc).__name__}: {exc}). No artefact was written; "
                "no zero values were substituted."
            ) from exc
        print(
            f"  SKIP {domain} seed={seed}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return {}
    finally:
        if cache is not None:
            cache.close()

    bidders: dict[str, Any] = {}
    skipped: list[str] = []
    for bidder_id, entry in entries.items():
        if entry.raw_pv_values is None or entry.pv_degraded:
            # Reachable only if a future failure policy lets a degraded entry
            # through; never emit fabricated values into a calibration fit.
            skipped.append(bidder_id)
            continue
        truth = scenario.instance.valuations[bidder_id]
        budget_hint = (
            None
            if entry.interest_map is None
            else entry.interest_map.budget_hint
        )
        bidders[bidder_id] = {
            "nl_question": entry.nl_question,
            "nl_answer": entry.nl_answer,
            "person_seed_sha256": sha256_text(scenario.person_seeds[bidder_id]),
            "interest_map": (
                None
                if entry.interest_map is None
                else entry.interest_map.model_dump()
            ),
            "disclosed_budget": (
                None if budget_hint is None else float(budget_hint)
            ),
            "true_max_value": max(truth.values(), default=0.0),
            "candidate_bundles": [
                sorted(bundle) for bundle in entry.candidate_bundles
            ],
            "raw_provisional_values": [
                {"bundle": sorted(bundle), "value": float(value)}
                for bundle, value in sorted(
                    entry.raw_pv_values.items(),
                    key=lambda pair: (len(pair[0]), sorted(pair[0])),
                )
            ],
            "hidden_true_values": [
                {"bundle": sorted(bundle), "value": float(truth[bundle])}
                for bundle in sorted(
                    entry.raw_pv_values, key=lambda b: (len(b), sorted(b))
                )
                if bundle in truth
            ],
            "diagnostics": {
                "interest_map_fallback_used": entry.interest_map_fallback_used,
                "interest_map_quality_flags": list(
                    entry.interest_map_quality_flags
                ),
                "person_answer_attempt_count": entry.person_answer_attempt_count,
                "person_answer_verification": entry.person_answer_verification,
                "pv_scale_quality_flags": list(entry.pv_scale_quality_flags),
            },
        }

    if not bidders:
        raise BenchmarkPreparationError(
            f"{domain} seed={seed}: no bidder produced usable provisional "
            "values; refusing to write an empty artefact"
        )

    payload = {
        "format": PV_CALIBRATION_BENCHMARK_FORMAT,
        "version": PV_CALIBRATION_BENCHMARK_VERSION,
        "domain": domain,
        "seed": seed,
        "scenario_name": scenario.name,
        "scenario_fingerprint": scenario_fingerprint(scenario),
        "items": list(scenario.instance.items),
        "bidder_ids": list(bidders),
        "bidders": bidders,
        "skipped_bidder_ids": skipped,
        "prompt_versions": _prompt_version_hashes(),
        "models": {
            "person": vars(
                ModelProvenance(
                    args.person_provider,
                    args.person_model,
                    args.person_temperature,
                )
            ),
            "proxy": vars(
                ModelProvenance(
                    args.proxy_provider,
                    args.proxy_model,
                    args.proxy_temperature,
                )
            ),
            "verifier": vars(
                ModelProvenance(
                    args.verifier_provider,
                    args.verifier_model,
                    args.verifier_temperature,
                )
            ),
        },
        "generation_settings": {
            "num_goods": len(scenario.instance.items),
            "num_bidders": len(scenario.instance.bidder_ids),
            "max_candidate_bundles": args.max_candidate_bundles,
            "pv_chunk_size": args.pv_chunk_size or None,
            "pv_max_tokens": args.pv_max_tokens,
            "interest_map_max_tokens": args.interest_map_max_tokens,
            "max_parse_retries": args.max_parse_retries,
            "opening_question_policy": "canonical",
            "llm_cache_mode": args.llm_cache_mode,
        },
        "environment": environment_payload,
        "environment_source": (
            None
            if environment_path is None
            else {
                "path": str(environment_path),
                "sha256": sha256_text(
                    json.dumps(environment_payload, sort_keys=True)
                ),
            }
        ),
        "call_metadata": {
            "elapsed_seconds": round(time.time() - started, 3),
            "call_type_counts": {
                key: stats.calls for key, stats in logger.total_stats().items()
            },
            "tokens_in": logger.total_tokens()[0],
            "tokens_out": logger.total_tokens()[1],
            "cache": cache_stats.as_dict(),
            "calls_log": str(log_dir / "calls.jsonl"),
        },
    }
    write_benchmark_artefact(payload, output_path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    planned = _planned_artefacts(args)

    if args.dry_run:
        print(f"Would prepare {len(planned)} benchmark artefact(s):")
        for domain, seed, path in planned:
            state = "overwrite" if path.exists() else "new"
            if path.exists() and not args.overwrite:
                state = "skip (resume)"
            print(f"  {domain:<26} seed={seed}  {path}  [{state}]")
        print("Dry run: no clients constructed, no LLM calls made.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    failures = 0

    for domain, seed, path in planned:
        if path.exists() and not args.overwrite:
            try:
                existing = load_benchmark_artefact(path)
            except (OSError, ValueError) as exc:
                print(f"  invalid existing artefact {path}: {exc}; regenerating")
            else:
                environment_changed = False
                if args.environment_dir is not None:
                    current_environment_path = (
                        args.environment_dir / environment_file_name(
                            domain, seed
                        )
                    )
                    try:
                        current_environment = load_generated_environment(
                            current_environment_path
                        )
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        raise SystemExit(
                            f"Cannot validate current environment "
                            f"{current_environment_path}: {exc}"
                        ) from exc
                    environment_changed = (
                        existing.get("environment") != current_environment
                    )
                if environment_changed:
                    print(
                        f"  existing artefact {path} was prepared from a "
                        "different/legacy environment; regenerating"
                    )
                else:
                    print(f"  SKIP {domain} seed={seed}: complete ({path})")
                    manifest.append(
                        {
                            "domain": domain,
                            "seed": seed,
                            "path": str(path),
                            "status": "skipped_complete",
                            "bidders": len(existing["bidders"]),
                        }
                    )
                    continue

        print(f"\n=== {domain}  seed={seed} ===", flush=True)
        try:
            payload = prepare_domain_seed(
                domain=domain, seed=seed, args=args, output_path=path
            )
        except BenchmarkPreparationError as exc:
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr)
            manifest.append(
                {
                    "domain": domain,
                    "seed": seed,
                    "path": str(path),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue
        manifest.append(
            {
                "domain": domain,
                "seed": seed,
                "path": str(path),
                "status": "completed" if payload else "skipped_failed",
                "bidders": len(payload.get("bidders", {})),
            }
        )
        print(f"  wrote {path}")

    manifest_path = args.output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest: {manifest_path}  ({failures} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
