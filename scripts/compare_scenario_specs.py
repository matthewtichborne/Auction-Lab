#!/usr/bin/env python3
"""Compare the validation reports of two PC-build scenario specs.

Builds a validation report for each spec (same machinery as
``validate_scenario_spec.py``) and diffs their scalar fields: profile
summary stats, and per-size economic diagnostics. Useful for checking
what changed between, e.g., the v0 manual spec and a new LLM-generated
trial spec.

No LLM/API calls are made.

Usage::

    ./venv/bin/python scripts/compare_scenario_specs.py \\
        scenarios/pc_build_v1/pc_build_profiles_v0_manual.json \\
        scenarios/pc_build_v1/pc_build_profiles_v1_gemini_trial.json \\
        --output scenarios/pc_build_v1/comparison_v0_vs_v1.json

    # Flag changes over 20% as "significant" and exit(1) if any are found:
    ./venv/bin/python scripts/compare_scenario_specs.py spec_a.json spec_b.json \\
        --output diff.json --tolerance 0.2 --fail-on-significant-change
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.validate_scenario_spec import DEFAULT_SIZES, _scalar_fields, build_validation_report


def _diff_scalar_dicts(a: dict, b: dict, *, tolerance: float) -> dict:
    """Field-by-field diff of two same-shaped scalar dicts."""
    diff: dict = {}
    keys = sorted(set(a) | set(b))
    for key in keys:
        a_val = a.get(key)
        b_val = b.get(key)
        entry: dict[str, Any] = {"a": a_val, "b": b_val}

        if isinstance(a_val, bool) or isinstance(b_val, bool):
            entry["changed"] = a_val != b_val
        elif isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            delta = b_val - a_val
            entry["delta"] = delta
            entry["pct_change"] = (delta / a_val) if a_val not in (0, None) else None
            entry["changed"] = abs(delta) > 1e-9
            entry["significant"] = (
                entry["pct_change"] is not None and abs(entry["pct_change"]) > tolerance
            )
        else:
            entry["changed"] = a_val != b_val

        diff[key] = entry
    return diff


def compare_reports(report_a: dict, report_b: dict, *, tolerance: float = 0.1) -> dict:
    comparison: dict = {
        "spec_a": report_a.get("spec_path"),
        "spec_b": report_b.get("spec_path"),
        "schema_valid_a": report_a["schema_validation"]["valid"],
        "schema_valid_b": report_b["schema_validation"]["valid"],
    }

    if not (comparison["schema_valid_a"] and comparison["schema_valid_b"]):
        comparison["profile_summary_diff"] = {}
        comparison["economic_diagnostics_diff"] = {}
        comparison["significant_changes"] = []
        return comparison

    comparison["profile_summary_diff"] = _diff_scalar_dicts(
        _scalar_fields(report_a["profile_summary"]),
        _scalar_fields(report_b["profile_summary"]),
        tolerance=tolerance,
    )

    size_keys = sorted(
        set(report_a.get("economic_diagnostics", {})) | set(report_b.get("economic_diagnostics", {}))
    )
    econ_diff: dict = {}
    significant_changes: list[dict] = []

    for size_key in size_keys:
        diag_a = report_a.get("economic_diagnostics", {}).get(size_key, {})
        diag_b = report_b.get("economic_diagnostics", {}).get(size_key, {})
        skipped_a = diag_a.get("skipped", False)
        skipped_b = diag_b.get("skipped", False)

        if skipped_a or skipped_b:
            econ_diff[size_key] = {"skipped_a": skipped_a, "skipped_b": skipped_b}
            continue

        field_diff = _diff_scalar_dicts(
            _scalar_fields(diag_a), _scalar_fields(diag_b), tolerance=tolerance
        )
        econ_diff[size_key] = field_diff

        for field_name, entry in field_diff.items():
            if entry.get("significant") or (
                isinstance(entry.get("a"), bool) and entry.get("changed")
            ):
                significant_changes.append(
                    {"size": size_key, "field": field_name, **entry}
                )

    comparison["economic_diagnostics_diff"] = econ_diff
    comparison["significant_changes"] = significant_changes
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec_a", type=Path)
    parser.add_argument("spec_b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="*",
        default=None,
        help="Override sizes as WxB pairs (default: 4x4 6x6 8x8 10x10)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Fractional change beyond which a numeric field is flagged 'significant' (default 0.1 = 10%%).",
    )
    parser.add_argument(
        "--fail-on-significant-change",
        action="store_true",
        help="Exit with status 1 if any significant changes are found.",
    )
    args = parser.parse_args()

    if args.sizes:
        sizes = []
        for spec_str in args.sizes:
            g, b = spec_str.lower().split("x")
            sizes.append((int(g), int(b)))
    else:
        sizes = DEFAULT_SIZES

    report_a = build_validation_report(args.spec_a, sizes)
    report_b = build_validation_report(args.spec_b, sizes)
    comparison = compare_reports(report_a, report_b, tolerance=args.tolerance)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote comparison to {args.output}")
    num_significant = len(comparison["significant_changes"])
    print(f"{num_significant} significant change(s) (tolerance={args.tolerance:.0%})")
    for change in comparison["significant_changes"]:
        print(
            f"  - [{change['size']}] {change['field']}: "
            f"{change['a']} -> {change['b']}"
        )

    if args.fail_on_significant_change and num_significant > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
