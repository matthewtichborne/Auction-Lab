"""Versioned, content-addressed specification for the final result pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from auctionlab.llm.value_calibration import load_calibration_config
from auctionlab.experiments.run_config import (
    CLOCK_EVENT_POLICY_SANDWICH_V3,
    SEALED_EVENT_POLICY_V1,
)


FINAL_EXPERIMENT_FORMAT = "auctionlab.final_experiment"
FINAL_EXPERIMENT_VERSION = 3
SUPPORTED_FINAL_EXPERIMENT_VERSIONS = (2, 3)


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalability_case_names(
    sizes: Sequence[int], fixed_size: int
) -> list[str]:
    """Canonical 19-cell names, with the shared anchor emitted once."""
    ordered_sizes = sorted({int(size) for size in sizes})
    if fixed_size not in ordered_sizes:
        raise ValueError("fixed_size must be included in sizes")
    if any(size <= 0 for size in ordered_sizes):
        raise ValueError("sizes must be positive")
    cases: list[str] = []
    for size in ordered_sizes:
        if size != fixed_size:
            cases.append(f"goods_{size}x{fixed_size}")
    for size in ordered_sizes:
        if size != fixed_size:
            cases.append(f"bidders_{fixed_size}x{size}")
    for size in ordered_sizes:
        cases.append(
            f"anchor_{size}x{size}"
            if size == fixed_size
            else f"joint_{size}x{size}"
        )
    return cases


def build_pack_inventory(
    root: str | Path,
    *,
    seeds: Sequence[int],
    sizes: Sequence[int],
    fixed_size: int,
) -> list[dict[str, Any]]:
    pack_root = Path(root)
    inventory: list[dict[str, Any]] = []
    missing: list[Path] = []
    for seed in sorted({int(value) for value in seeds}):
        for case_name in scalability_case_names(sizes, fixed_size):
            path = (
                pack_root
                / f"seed_{seed}"
                / case_name
                / "frozen_elicitation.json"
            )
            if not path.exists():
                missing.append(path)
                continue
            inventory.append(
                {
                    "seed": seed,
                    "case": case_name,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} frozen elicitation packs:\n{preview}{suffix}"
        )
    return inventory


def build_final_experiment_spec(
    *,
    scenario_spec: str | Path,
    elicitation_pack_dir: str | Path,
    calibration_config: str | Path,
    seeds: Sequence[int],
    sizes: Sequence[int],
    fixed_size: int,
    sealed_max_rounds: int,
    correction_threshold: float,
    clock_max_rounds: int,
    clock_price_step: float,
    clock_top_k: int,
    clock_tie_threshold: float,
    robustness_models: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    scenario_path = Path(scenario_spec)
    calibration_path = Path(calibration_config)
    calibration = load_calibration_config(calibration_path)
    if calibration.provenance.get("accepted") is not True:
        raise ValueError(
            "final calibration provenance must record accepted=true"
        )
    if sealed_max_rounds <= 0 or clock_max_rounds <= 0:
        raise ValueError("mechanism max rounds must be positive")
    if clock_price_step <= 0 or clock_top_k <= 0:
        raise ValueError("clock price step and top_k must be positive")
    if clock_tie_threshold < 0:
        raise ValueError("clock tie threshold must be non-negative")
    if not 0 < correction_threshold <= 1:
        raise ValueError("correction threshold must be in (0, 1]")
    inventory = build_pack_inventory(
        elicitation_pack_dir,
        seeds=seeds,
        sizes=sizes,
        fixed_size=fixed_size,
    )
    return {
        "format": FINAL_EXPERIMENT_FORMAT,
        "version": FINAL_EXPERIMENT_VERSION,
        "status": "frozen",
        "scenario": {
            "path": str(scenario_path),
            "sha256": sha256_file(scenario_path),
            "selection_policy": "coverage_stratified",
        },
        "dataset": {
            "seeds": sorted({int(value) for value in seeds}),
            "sizes": sorted({int(value) for value in sizes}),
            "fixed_size": int(fixed_size),
            "case_count": len(inventory),
            "elicitation_pack_dir": str(elicitation_pack_dir),
            "packs": inventory,
        },
        "calibration": {
            "path": str(calibration_path),
            "file_sha256": sha256_file(calibration_path),
            "effective_config_hash": calibration.config_hash(),
            "config": calibration.to_dict(),
        },
        "event_policy": {
            "name": "final-v3",
            "correction_threshold": float(correction_threshold),
            "sealed": dict(SEALED_EVENT_POLICY_V1),
            "clock": dict(CLOCK_EVENT_POLICY_SANDWICH_V3),
        },
        "sealed": {
            "max_rounds": int(sealed_max_rounds),
            "stopping_rule": "no_new_refinements",
        },
        "clock": {
            "max_rounds": int(clock_max_rounds),
            "price_step": float(clock_price_step),
            "top_k": int(clock_top_k),
            "tie_threshold": float(clock_tie_threshold),
        },
        "queries": {
            "person_query_mode": "deterministic",
            "per_bidder_safety_cap": 0,
            "global_safety_cap": 0,
        },
        "primary_metrics": [
            "efficiency",
            "efficiency_gain_from_initial",
            "revenue_loss",
            "payment_error_over_optimum_welfare",
            "value_queries",
            "rounds",
            "runtime",
        ],
        "example_selection": [
            "largest_sealed_efficiency_improvement",
            "largest_sealed_revenue_loss_reduction",
            "median_query_case_reaching_full_efficiency_from_below",
            "representative_non_improving_case",
        ],
        "model_robustness": {
            "scope": "five_seed_8x8_fixed_disclosures",
            "models": [dict(model) for model in robustness_models],
            "calibration": "raw_per_model",
        },
    }


def write_final_experiment_spec(
    path: str | Path, spec: Mapping[str, Any]
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(spec), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_final_experiment_spec(
    path: str | Path,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    source = Path(path)
    spec = json.loads(source.read_text(encoding="utf-8"))
    if spec.get("format") != FINAL_EXPERIMENT_FORMAT:
        raise ValueError(f"{source}: unexpected final-experiment format")
    if spec.get("version") not in SUPPORTED_FINAL_EXPERIMENT_VERSIONS:
        raise ValueError(f"{source}: unsupported final-experiment version")
    if spec.get("status") != "frozen":
        raise ValueError(f"{source}: specification is not frozen")
    if verify_files:
        checks = [
            (spec["scenario"]["path"], spec["scenario"]["sha256"]),
            (
                spec["calibration"]["path"],
                spec["calibration"]["file_sha256"],
            ),
            *[
                (row["path"], row["sha256"])
                for row in spec["dataset"]["packs"]
            ],
        ]
        for file_path, expected_hash in checks:
            actual_hash = sha256_file(file_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"frozen input changed: {file_path}; "
                    f"expected {expected_hash}, got {actual_hash}"
                )
    return spec
