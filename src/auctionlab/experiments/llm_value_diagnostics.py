"""Diagnose LLM value estimation independently of allocation outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re

from auctionlab.auction_types import Bundle, bundle_label
from auctionlab.experiments.llm_analysis_io import write_csv_rows
from auctionlab.instances.base import AuctionInstance


VALUE_ERROR_FIELDS = [
    "scenario",
    "seed_type",
    "mechanism",
    "top_k",
    "bidder_id",
    "bundle",
    "bundle_label",
    "reported_value",
    "true_value",
    "signed_error",
    "absolute_error",
    "relative_error",
    "bundle_size",
]

VALUE_ERROR_AGGREGATE_FIELDS = [
    "scenario",
    "seed_type",
    "mechanism",
    "top_k",
    "n",
    "mae",
    "rmse",
    "mean_signed_error",
    "overreport_rate",
    "underreport_rate",
    "exact_match_rate",
    "max_absolute_error",
    "mean_relative_error",
    "mean_absolute_relative_error",
]

_ATOM_PATTERN = re.compile(
    r"^\[(?P<items>[^\[\]]*)\]:"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)


@dataclass(frozen=True)
class ValueErrorRecord:
    """Reported-versus-true value error for one bidder and queried bundle."""

    scenario: str
    seed_type: str
    mechanism: str
    top_k: str
    bidder_id: str
    bundle: Bundle
    bundle_label: str
    reported_value: float
    true_value: float
    signed_error: float
    absolute_error: float
    relative_error: float | None
    bundle_size: int


def parse_reported_bids(text: str) -> dict[str, dict[Bundle, float]]:
    if not text.strip():
        return {}

    reported_bids: dict[str, dict[Bundle, float]] = {}
    for bidder_part in text.split("|"):
        if not bidder_part or "=" not in bidder_part:
            raise ValueError(f"Malformed bidder entry: {bidder_part!r}")

        bidder_id, atoms_text = bidder_part.split("=", maxsplit=1)
        bidder_id = bidder_id.strip()
        if not bidder_id:
            raise ValueError("Bidder ID must not be empty")
        if bidder_id in reported_bids:
            raise ValueError(f"Duplicate bidder ID: {bidder_id}")
        if not atoms_text.startswith("{") or not atoms_text.endswith("}"):
            raise ValueError(f"Malformed bid for bidder {bidder_id}")

        inner = atoms_text[1:-1]
        atoms: dict[Bundle, float] = {}
        if inner:
            for atom_text in inner.split(";"):
                match = _ATOM_PATTERN.fullmatch(atom_text.strip())
                if match is None:
                    raise ValueError(
                        f"Malformed atom for bidder {bidder_id}: {atom_text!r}"
                    )

                items_text = match.group("items")
                items = (
                    [item.strip() for item in items_text.split(",")]
                    if items_text
                    else []
                )
                if any(not item for item in items):
                    raise ValueError(
                        f"Bundle contains an empty item for bidder {bidder_id}"
                    )
                if len(set(items)) != len(items):
                    raise ValueError(
                        f"Bundle contains duplicate items for bidder {bidder_id}"
                    )

                bundle = frozenset(items)
                if bundle in atoms:
                    raise ValueError(
                        f"Duplicate bundle for bidder {bidder_id}: "
                        f"{bundle_label(bundle)}"
                    )
                value = float(match.group("value"))
                if not math.isfinite(value):
                    raise ValueError("Reported values must be finite")
                atoms[bundle] = value

        reported_bids[bidder_id] = atoms

    return reported_bids


def compute_value_error_records(
    scenario: str,
    seed_type: str,
    mechanism: str,
    top_k: str,
    reported_bids: dict[str, dict[Bundle, float]],
    instance: AuctionInstance,
) -> list[ValueErrorRecord]:
    """Compare every reported atom with benchmark XOR-subset value semantics."""
    unknown_bidders = sorted(set(reported_bids) - set(instance.bidder_ids))
    if unknown_bidders:
        raise ValueError(f"Unknown bidder IDs: {unknown_bidders}")

    records: list[ValueErrorRecord] = []
    for bidder_id in sorted(reported_bids):
        for bundle, reported_value in sorted(
            reported_bids[bidder_id].items(),
            key=lambda item: (len(item[0]), tuple(sorted(item[0]))),
        ):
            true_value = float(instance.value_of(bidder_id, bundle))
            signed_error = reported_value - true_value
            records.append(
                ValueErrorRecord(
                    scenario=scenario,
                    seed_type=seed_type,
                    mechanism=mechanism,
                    top_k=top_k,
                    bidder_id=bidder_id,
                    bundle=bundle,
                    bundle_label=bundle_label(bundle),
                    reported_value=reported_value,
                    true_value=true_value,
                    signed_error=signed_error,
                    absolute_error=abs(signed_error),
                    relative_error=(
                        signed_error / true_value
                        if true_value > 0.0
                        else None
                    ),
                    bundle_size=len(bundle),
                )
            )
    return records


def aggregate_value_error_records(
    records: list[ValueErrorRecord],
) -> dict[str, str]:
    """Summarize estimation magnitude and over/under-reporting direction."""
    n = len(records)
    if n == 0:
        return {
            "n": "0",
            "mae": "0.0",
            "rmse": "0.0",
            "mean_signed_error": "0.0",
            "overreport_rate": "0.0",
            "underreport_rate": "0.0",
            "exact_match_rate": "0.0",
            "max_absolute_error": "0.0",
            "mean_relative_error": "",
            "mean_absolute_relative_error": "",
        }

    relative_errors = [
        record.relative_error
        for record in records
        if record.relative_error is not None
    ]
    return {
        "n": str(n),
        "mae": str(sum(record.absolute_error for record in records) / n),
        "rmse": str(
            math.sqrt(
                sum(record.signed_error**2 for record in records) / n
            )
        ),
        "mean_signed_error": str(
            sum(record.signed_error for record in records) / n
        ),
        "overreport_rate": str(
            sum(record.signed_error > 0.0 for record in records) / n
        ),
        "underreport_rate": str(
            sum(record.signed_error < 0.0 for record in records) / n
        ),
        "exact_match_rate": str(
            sum(record.signed_error == 0.0 for record in records) / n
        ),
        "max_absolute_error": str(
            max(record.absolute_error for record in records)
        ),
        "mean_relative_error": (
            str(sum(relative_errors) / len(relative_errors))
            if relative_errors
            else ""
        ),
        "mean_absolute_relative_error": (
            str(
                sum(abs(error) for error in relative_errors)
                / len(relative_errors)
            )
            if relative_errors
            else ""
        ),
    }


def value_error_records_to_rows(
    records: list[ValueErrorRecord],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        data = asdict(record)
        rows.append(
            {
                "scenario": record.scenario,
                "seed_type": record.seed_type,
                "mechanism": record.mechanism,
                "top_k": record.top_k,
                "bidder_id": record.bidder_id,
                "bundle": ",".join(sorted(data["bundle"])),
                "bundle_label": record.bundle_label,
                "reported_value": str(record.reported_value),
                "true_value": str(record.true_value),
                "signed_error": str(record.signed_error),
                "absolute_error": str(record.absolute_error),
                "relative_error": (
                    str(record.relative_error)
                    if record.relative_error is not None
                    else ""
                ),
                "bundle_size": str(record.bundle_size),
            }
        )
    return rows


def group_value_error_records(
    records: list[ValueErrorRecord],
) -> dict[tuple[str, str, str, str], list[ValueErrorRecord]]:
    grouped: dict[
        tuple[str, str, str, str],
        list[ValueErrorRecord],
    ] = {}
    for record in records:
        key = (
            record.scenario,
            record.seed_type,
            record.mechanism,
            record.top_k,
        )
        grouped.setdefault(key, []).append(record)
    return grouped


def write_value_error_records_csv(
    records: list[ValueErrorRecord],
    path: str | Path,
) -> None:
    write_csv_rows(
        path,
        VALUE_ERROR_FIELDS,
        value_error_records_to_rows(records),
    )


def write_value_error_aggregate_csv(
    grouped_records: dict[
        tuple[str, str, str, str],
        list[ValueErrorRecord],
    ],
    path: str | Path,
) -> None:
    rows = []
    for key in sorted(
        grouped_records,
        key=lambda value: (
            value[0],
            value[1],
            0 if value[2] == "sealed_llm_proxy_vcg" else 1,
            int(value[3]) if value[3] else -1,
        ),
    ):
        scenario, seed_type, mechanism, top_k = key
        rows.append(
            {
                "scenario": scenario,
                "seed_type": seed_type,
                "mechanism": mechanism,
                "top_k": top_k,
                **aggregate_value_error_records(grouped_records[key]),
            }
        )
    write_csv_rows(path, VALUE_ERROR_AGGREGATE_FIELDS, rows)
