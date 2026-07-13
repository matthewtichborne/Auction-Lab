from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Iterable, Sequence


RESULT_KEY_FIELDS = ("scenario", "seed_type", "mechanism", "top_k")

_CLOCK_RESULT_PATTERN = re.compile(
    r"curated_clock_llm_comparison_top_(\d+)\.csv"
)

_PROXY_CLOCK_RESULT_PATTERN = re.compile(
    r"curated_clock_proxy_elicited_top_(\d+)\.csv"
)


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, str]],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_labeled_path(
    raw: str,
    *,
    label_from_parent: bool = False,
) -> tuple[str, Path]:
    if "=" in raw:
        label, path_text = raw.split("=", maxsplit=1)
    else:
        path_text = raw
        path = Path(path_text)
        label = (
            path.parent.name
            if label_from_parent
            else path.name or path.parent.name
        )

    label = label.strip()
    path_text = path_text.strip()
    if not label:
        raise ValueError(f"Run label must not be empty: {raw!r}")
    if not path_text:
        raise ValueError(f"Run path must not be empty: {raw!r}")
    return label, Path(path_text)


def extract_top_k_from_filename(path: str | Path) -> int:
    path = Path(path)
    match = _CLOCK_RESULT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Cannot extract top_k from filename: {path.name}")
    return int(match.group(1))


def extract_proxy_clock_top_k_from_filename(path: str | Path) -> int:
    path = Path(path)
    match = _PROXY_CLOCK_RESULT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Cannot extract top_k from filename: {path.name}")
    return int(match.group(1))


def read_curated_result_rows(
    input_dir: str | Path,
    top_k: Iterable[int] | None = None,
) -> tuple[list[dict[str, str]], dict[int, list[dict[str, str]]]]:
    input_path = Path(input_dir)
    sealed_path = input_path / "curated_sealed_llm_comparison.csv"
    sealed_rows = load_csv_rows(sealed_path) if sealed_path.exists() else []

    if top_k is None:
        clock_paths = sorted(
            input_path.glob("curated_clock_llm_comparison_top_*.csv"),
            key=extract_top_k_from_filename,
        )
    else:
        clock_paths = [
            input_path / f"curated_clock_llm_comparison_top_{value}.csv"
            for value in sorted(set(top_k))
        ]

    clock_rows = {
        extract_top_k_from_filename(path): load_csv_rows(path)
        for path in clock_paths
        if path.exists()
    }
    return sealed_rows, clock_rows


def read_proxy_result_rows(
    input_dir: str | Path,
) -> tuple[list[dict[str, str]], dict[int, list[dict[str, str]]]]:
    """Load proxy-mediated elicitation CSVs, if present.

    Returns the same shape as :func:`read_curated_result_rows`: a list of
    sealed-proxy rows and a top_k-keyed mapping of clock-proxy rows.
    """
    input_path = Path(input_dir)
    sealed_path = input_path / "curated_sealed_proxy_elicited.csv"
    sealed_rows = load_csv_rows(sealed_path) if sealed_path.exists() else []

    clock_paths = sorted(
        input_path.glob("curated_clock_proxy_elicited_top_*.csv"),
        key=extract_proxy_clock_top_k_from_filename,
    )
    clock_rows = {
        extract_proxy_clock_top_k_from_filename(path): load_csv_rows(path)
        for path in clock_paths
    }
    return sealed_rows, clock_rows


def result_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(row.get(field, "") for field in RESULT_KEY_FIELDS)


def load_csv_by_key(
    path: str | Path,
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    keyed_rows: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in load_csv_rows(path):
        key = result_key(row)
        if key in keyed_rows:
            raise ValueError(f"Duplicate join key in {path}: {key}")
        keyed_rows[key] = row
    return keyed_rows


def _mechanism_rank(mechanism: str) -> int:
    """Group mechanisms: static sealed/clock first, then their proxy variants."""
    if mechanism == "sealed_llm_proxy_vcg":
        return 0
    if mechanism.startswith("proxy_sealed"):
        return 1
    if mechanism == "clock_llm_proxy_vcg":
        return 2
    if mechanism.startswith("proxy_clock"):
        return 3
    return 4


def sort_result_rows(
    rows: Iterable[dict[str, str]],
    *,
    prefix_fields: Sequence[str] = (),
    suffix_fields: Sequence[str] = (),
) -> list[dict[str, str]]:
    def sort_key(row: dict[str, str]) -> tuple[object, ...]:
        top_k = row.get("top_k", "")
        return (
            *(row.get(field, "") for field in prefix_fields),
            row.get("scenario", ""),
            row.get("seed_type", ""),
            _mechanism_rank(row.get("mechanism", "")),
            int(top_k) if top_k else -1,
            row.get("mechanism", ""),
            *(row.get(field, "") for field in suffix_fields),
        )

    return sorted(rows, key=sort_key)
