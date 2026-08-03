from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List



def allocation_to_str(allocation: Dict[str, frozenset[str]]) -> str:
    """
    Stable string representation of an allocation.

    Example:
      i1:[];i2:[B];i3:[A,C]
    """
    parts: List[str] = []

    for bidder_id in sorted(allocation.keys()):
        bundle = allocation[bidder_id]
        bundle_str = ",".join(sorted(bundle))
        parts.append(f"{bidder_id}:[{bundle_str}]")

    return ";".join(parts)


def write_csv(
    rows: List[Dict[str, Any]],
    path: str | Path,
) -> None:
    """
    Write rows to a CSV file.

    If rows is empty, writes an empty file with no header. All rows must
    share the same keys (matches ``rows[0].keys()``); use
    :func:`write_csv_variable_rows` when rows come from heterogeneous
    sources and may not all share the same keys.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("")
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_variable_rows(
    rows: List[Dict[str, Any]],
    path: str | Path,
) -> None:
    """
    Write rows to a CSV file when rows may not all share the same keys.

    Fieldnames are the union of every row's keys, in first-seen order;
    missing keys in any given row are written as an empty cell. Useful for
    a summary table assembled from several differently-shaped per-arm
    dicts (e.g. a shared-initialization row vs. a per-mechanism row).
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
