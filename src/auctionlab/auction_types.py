"""Shared aliases and records used across the auction stack.

An item is a string identifier and a bundle is a frozenset of items, so
bundles are hashable and usable as dictionary keys. Bundles are rendered
through ``bundle_label`` rather than by formatting the set directly, because
set iteration order is not stable and result files must be reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, FrozenSet, List


Item = str
Bundle = FrozenSet[Item]


def bundle_label(bundle: Bundle) -> str:
    return f"[{','.join(sorted(bundle))}]"


def validate_bidder_keys(
    *,
    bidder_ids: list[str],
    values: Mapping[str, object],
    label: str,
) -> None:
    expected = set(bidder_ids)
    actual = set(values)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            f"{label} must contain exactly the instance bidder IDs "
            f"({', '.join(details)})"
        )


@dataclass(frozen=True)
class ClockRoundRecord:
    round_idx: int
    prices: Dict[Item, float]
    demands: Dict[str, List[Bundle]]  # bidder_id -> demand set at these prices