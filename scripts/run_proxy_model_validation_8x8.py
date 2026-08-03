#!/usr/bin/env python3
"""Validate alternative proxy models on fixed five-seed 8x8 disclosures.

For each model, the opening question and person answers are copied from the
primary frozen pack.  Only the interest map and raw provisional valuations are
regenerated.  The resulting pack is then replayed through the frozen sealed
and clock specifications with no calibration, isolating proxy-model
portability from person/environment changes and model-specific scaling.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auctionlab.experiments.final_pipeline import (  # noqa: E402
    load_final_experiment_spec,
)


def _parse_model(value: str) -> tuple[str, str]:
    try:
        provider, model = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "model must be PROVIDER:MODEL"
        ) from exc
    if not provider or not model:
        raise argparse.ArgumentTypeError("model must be PROVIDER:MODEL")
    return provider, model


def _slug(provider: str, model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", f"{provider}_{model}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=_parse_model,
        action="append",
        required=True,
        help="Alternative proxy model as PROVIDER:MODEL; repeat as needed.",
    )
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--interest-map-max-tokens", type=int, default=2000)
    parser.add_argument("--pv-chunk-size", type=int, default=64)
    parser.add_argument("--max-parse-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument(
        "--llm-cache-mode",
        choices=["off", "read-write", "read-only", "refresh"],
        default="read-write",
    )
    parser.add_argument(
        "--llm-cache-path", default="cache/final_model_validation.sqlite"
    )
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def _anchor_packs(spec: dict) -> dict[int, Path]:
    return {
        int(row["seed"]): Path(row["path"])
        for row in spec["dataset"]["packs"]
        if row["case"] == "anchor_8x8"
    }


def preparation_command(
    *,
    spec: dict,
    source_pack: Path,
    output_pack: Path,
    log_dir: Path,
    provider: str,
    model: str,
    args: argparse.Namespace,
) -> list[str]:
    seed = int(source_pack.parents[1].name.removeprefix("seed_"))
    return [
        sys.executable,
        "scripts/generate_frozen_elicitation.py",
        "--scenario", "pc_build",
        "--scenario-spec", spec["scenario"]["path"],
        "--num-goods", "8",
        "--num-bidders", "8",
        "--scenario-seed", str(seed),
        "--selection-policy", spec["scenario"]["selection_policy"],
        "--seed-type", "structured",
        "--disclosure-pack", str(source_pack),
        "--output", str(output_pack),
        "--proxy-provider", provider,
        "--proxy-model", model,
        "--person-query-mode", "deterministic",
        "--ask-initial-question",
        "--use-interest-map",
        "--use-provisional-valuations",
        "--interest-map-failure-policy", "raise",
        "--pv-failure-policy", "raise",
        "--interest-map-max-tokens", str(args.interest_map_max_tokens),
        "--pv-max-tokens", str(args.max_tokens),
        "--pv-chunk-size", str(args.pv_chunk_size),
        "--max-parse-retries", str(args.max_parse_retries),
        "--timeout", str(args.timeout),
        "--llm-cache-mode", args.llm_cache_mode,
        "--llm-cache-path", args.llm_cache_path,
        "--log-dir", str(log_dir),
    ]


def auction_command(
    *,
    spec: dict,
    pack: Path,
    seed: int,
    run_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "examples/run_live_llm_curated_batch.py",
        "--scenario", "pc_build",
        "--scenario-spec", spec["scenario"]["path"],
        "--num-goods", "8",
        "--num-bidders", "8",
        "--scenario-seed", str(seed),
        "--selection-policy", spec["scenario"]["selection_policy"],
        "--seed-type", "structured",
        "--elicitation-pack", str(pack),
        "--person-query-mode", "deterministic",
        "--skip-baselines",
        "--sealed-elicitation-rounds", str(spec["sealed"]["max_rounds"]),
        "--sealed-stopping-rule", spec["sealed"]["stopping_rule"],
        "--elicited-clock",
        "--top-k", str(spec["clock"]["top_k"]),
        "--max-rounds", str(spec["clock"]["max_rounds"]),
        "--price-step", str(spec["clock"]["price_step"]),
        "--clock-tie-threshold", str(spec["clock"]["tie_threshold"]),
        "--event-policy", spec.get("event_policy", {}).get(
            "name", "recommended"
        ),
        "--event-correction-threshold",
        str(spec["event_policy"]["correction_threshold"]),
        "--llm-cache-mode", "off",
        "--log-dir", str(run_dir),
    ]


def _run(command: list[str], log_path: Path, dry_run: bool) -> int:
    print(" ".join(command), flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _collect(provider: str, model: str, seed: int, run_dir: Path) -> list[dict[str, Any]]:
    summary = _read_csv(run_dir / "curated_run_summary.csv")
    rows: list[dict[str, Any]] = []
    for mechanism, arm_prefix, detail_glob in (
        ("sealed", "proxy sealed", "curated_sealed_proxy_elicited.csv"),
        ("clock", "proxy clock", "curated_clock_proxy_elicited_top_*.csv"),
    ):
        arm = next(row for row in summary if row["arm"].startswith(arm_prefix))
        detail_path = next(run_dir.glob(detail_glob))
        detail = _read_csv(detail_path)[0]
        rows.append({
            "provider": provider,
            "model": model,
            "seed": seed,
            "mechanism": mechanism,
            "efficiency": arm["efficiency"],
            "revenue_loss": detail.get("revenue_loss", ""),
            "payment_error_over_optimum_welfare": detail.get(
                "payment_error_over_optimum_welfare", ""
            ),
            "value_queries": arm["vq"],
            "demand_queries": arm["dq"],
            "allocation_match": detail["allocation_match"],
            "run_dir": str(run_dir),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = load_final_experiment_spec(args.spec, verify_files=True)
    source_packs = _anchor_packs(spec)
    expected_seeds = set(int(seed) for seed in spec["dataset"]["seeds"])
    if set(source_packs) != expected_seeds:
        raise SystemExit("Frozen specification does not contain every 8x8 anchor")
    failures: list[dict[str, Any]] = []
    for provider, model in args.model:
        model_root = args.output_dir / _slug(provider, model)
        for seed, source_pack in sorted(source_packs.items()):
            pack = model_root / f"seed_{seed}" / "frozen_elicitation.json"
            prep_log = model_root / f"seed_{seed}" / "preparation.log"
            run_dir = model_root / f"seed_{seed}" / "auction"
            if not pack.exists() or args.rerun_complete:
                code = _run(
                    preparation_command(
                        spec=spec,
                        source_pack=source_pack,
                        output_pack=pack,
                        log_dir=pack.parent / "preparation_calls",
                        provider=provider,
                        model=model,
                        args=args,
                    ),
                    prep_log,
                    args.dry_run,
                )
                if code:
                    failures.append({
                        "provider": provider, "model": model, "seed": seed,
                        "stage": "preparation", "returncode": code,
                    })
                    if args.fail_fast:
                        break
                    continue
            summary = run_dir / "curated_run_summary.csv"
            if not summary.exists() or args.rerun_complete:
                code = _run(
                    auction_command(
                        spec=spec, pack=pack, seed=seed, run_dir=run_dir
                    ),
                    run_dir / "model_validation_runner.log",
                    args.dry_run,
                )
                if code:
                    failures.append({
                        "provider": provider, "model": model, "seed": seed,
                        "stage": "auction", "returncode": code,
                    })
                    if args.fail_fast:
                        break
        if failures and args.fail_fast:
            break
    if args.dry_run:
        return 0
    rows: list[dict[str, Any]] = []
    for provider, model in args.model:
        model_root = args.output_dir / _slug(provider, model)
        for seed in sorted(expected_seeds):
            run_dir = model_root / f"seed_{seed}" / "auction"
            if (run_dir / "curated_run_summary.csv").exists():
                rows.extend(_collect(provider, model, seed, run_dir))
    _write_csv(args.output_dir / "proxy_model_validation.csv", rows)
    (args.output_dir / "proxy_model_validation_failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Collected {len(rows)} mechanism rows; failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
