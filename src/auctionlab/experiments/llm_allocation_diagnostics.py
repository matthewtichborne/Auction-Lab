"""Diagnose allocation changes and welfare loss under benchmark valuations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from auctionlab.auction_types import Bundle, bundle_label
from auctionlab.experiments.llm_analysis_io import write_csv_rows
from auctionlab.instances.base import AuctionInstance


ALLOCATION_LOSS_FIELDS = [
    "scenario",
    "seed_type",
    "mechanism",
    "top_k",
    "bidder_id",
    "full_info_bundle",
    "llm_bundle",
    "full_info_true_value",
    "llm_true_value",
    "true_value_delta",
    "full_info_bundle_label",
    "llm_bundle_label",
    "changed",
]

ALLOCATION_LOSS_AGGREGATE_FIELDS = [
    "scenario",
    "seed_type",
    "mechanism",
    "top_k",
    "n_bidders",
    "changed_bidder_count",
    "changed_bidder_rate",
    "full_info_true_welfare",
    "llm_true_welfare",
    "welfare_loss",
    "efficiency",
    "positive_delta_count",
    "negative_delta_count",
    "zero_delta_count",
]

_ALLOCATION_ENTRY_PATTERN = re.compile(
    r"^(?P<bidder_id>[^:;\|]+):\[(?P<items>[^\[\]]*)\]$"
)


@dataclass(frozen=True)
class AllocationLossRecord:
    """Per-bidder bundle and true-value change relative to full information."""

    scenario: str
    seed_type: str
    mechanism: str
    top_k: str
    bidder_id: str
    full_info_bundle: Bundle
    llm_bundle: Bundle
    full_info_true_value: float
    llm_true_value: float
    true_value_delta: float
    full_info_bundle_label: str
    llm_bundle_label: str
    changed: bool


def parse_allocation(text: str) -> dict[str, Bundle]:
    if not text.strip():
        return {}

    allocation: dict[str, Bundle] = {}
    for entry in re.split(r"[;|]", text):
        match = _ALLOCATION_ENTRY_PATTERN.fullmatch(entry.strip())
        if match is None:
            raise ValueError(f"Malformed allocation entry: {entry!r}")

        bidder_id = match.group("bidder_id").strip()
        if not bidder_id:
            raise ValueError("Bidder ID must not be empty")
        if bidder_id in allocation:
            raise ValueError(f"Duplicate bidder ID: {bidder_id}")

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
        allocation[bidder_id] = frozenset(items)

    return allocation


def compute_allocation_loss_records(
    scenario: str,
    seed_type: str,
    mechanism: str,
    top_k: str,
    instance: AuctionInstance,
    full_info_allocation: dict[str, Bundle],
    llm_allocation: dict[str, Bundle],
) -> list[AllocationLossRecord]:
    """Compare full-information and LLM allocations bidder by bidder.

    Unlike value-error diagnostics, these records measure downstream outcome
    effects after winner determination.
    """
    known_bidders = set(instance.bidder_ids)
    unknown_bidders = sorted(
        (set(full_info_allocation) | set(llm_allocation)) - known_bidders
    )
    if unknown_bidders:
        raise ValueError(f"Unknown bidder IDs: {unknown_bidders}")

    records = []
    for bidder_id in instance.bidder_ids:
        full_info_bundle = frozenset(
            full_info_allocation.get(bidder_id, frozenset())
        )
        llm_bundle = frozenset(
            llm_allocation.get(bidder_id, frozenset())
        )
        full_info_true_value = float(
            instance.value_of(bidder_id, full_info_bundle)
        )
        llm_true_value = float(instance.value_of(bidder_id, llm_bundle))
        records.append(
            AllocationLossRecord(
                scenario=scenario,
                seed_type=seed_type,
                mechanism=mechanism,
                top_k=top_k,
                bidder_id=bidder_id,
                full_info_bundle=full_info_bundle,
                llm_bundle=llm_bundle,
                full_info_true_value=full_info_true_value,
                llm_true_value=llm_true_value,
                true_value_delta=llm_true_value - full_info_true_value,
                full_info_bundle_label=bundle_label(full_info_bundle),
                llm_bundle_label=bundle_label(llm_bundle),
                changed=full_info_bundle != llm_bundle,
            )
        )
    return records


def aggregate_allocation_loss_records(
    records: list[AllocationLossRecord],
) -> dict[str, str]:
    """Summarize changed bidders and benchmark-valued welfare loss."""
    n_bidders = len(records)
    full_info_true_welfare = sum(
        record.full_info_true_value for record in records
    )
    llm_true_welfare = sum(record.llm_true_value for record in records)
    changed_bidder_count = sum(record.changed for record in records)

    return {
        "n_bidders": str(n_bidders),
        "changed_bidder_count": str(changed_bidder_count),
        "changed_bidder_rate": (
            str(changed_bidder_count / n_bidders)
            if n_bidders
            else "0.0"
        ),
        "full_info_true_welfare": str(full_info_true_welfare),
        "llm_true_welfare": str(llm_true_welfare),
        "welfare_loss": str(full_info_true_welfare - llm_true_welfare),
        "efficiency": (
            str(llm_true_welfare / full_info_true_welfare)
            if full_info_true_welfare > 0.0
            else ""
        ),
        "positive_delta_count": str(
            sum(record.true_value_delta > 0.0 for record in records)
        ),
        "negative_delta_count": str(
            sum(record.true_value_delta < 0.0 for record in records)
        ),
        "zero_delta_count": str(
            sum(record.true_value_delta == 0.0 for record in records)
        ),
    }


def allocation_loss_records_to_rows(
    records: list[AllocationLossRecord],
) -> list[dict[str, str]]:
    return [
        {
            "scenario": record.scenario,
            "seed_type": record.seed_type,
            "mechanism": record.mechanism,
            "top_k": record.top_k,
            "bidder_id": record.bidder_id,
            "full_info_bundle": ",".join(sorted(record.full_info_bundle)),
            "llm_bundle": ",".join(sorted(record.llm_bundle)),
            "full_info_true_value": str(record.full_info_true_value),
            "llm_true_value": str(record.llm_true_value),
            "true_value_delta": str(record.true_value_delta),
            "full_info_bundle_label": record.full_info_bundle_label,
            "llm_bundle_label": record.llm_bundle_label,
            "changed": str(record.changed),
        }
        for record in records
    ]


def group_allocation_loss_records(
    records: list[AllocationLossRecord],
) -> dict[tuple[str, str, str, str], list[AllocationLossRecord]]:
    grouped: dict[
        tuple[str, str, str, str],
        list[AllocationLossRecord],
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


def write_allocation_loss_records_csv(
    records: list[AllocationLossRecord],
    path: str | Path,
) -> None:
    write_csv_rows(
        path,
        ALLOCATION_LOSS_FIELDS,
        allocation_loss_records_to_rows(records),
    )


def write_allocation_loss_aggregate_csv(
    grouped_records: dict[
        tuple[str, str, str, str],
        list[AllocationLossRecord],
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
                **aggregate_allocation_loss_records(grouped_records[key]),
            }
        )
    write_csv_rows(path, ALLOCATION_LOSS_AGGREGATE_FIELDS, rows)
