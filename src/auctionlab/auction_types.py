from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import FrozenSet, List, Dict, Optional


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


@dataclass
class CecaBidderDiagnostic:
    """Per-bidder satisfaction diagnostic captured during a CECA round step.

    Fields filled by the proxy's ``ceca_step`` (what the proxy knows):
      ``allocated_bundle``         tentative WDP allocation for this bidder
      ``allocated_lindahl_price``  Lindahl price of that bundle (Eq. 4)
      ``allocated_manifest_value`` manifest XOR bid value of that bundle
      ``allocated_utility``        manifest_value - lindahl_price
      ``best_bundle``              oracle's demanded bundle (best at prices)
      ``best_value``               oracle-reported value of best bundle
      ``best_price``               Lindahl price of best bundle
      ``best_utility``             best_value - best_price
      ``utility_gap``              best_utility - allocated_utility (>0 → unsatisfied)
      ``satisfied``                True iff bidder is happy with allocation

    Field annotated by ``run_ceca`` after the manifest upsert:
      ``demand_produced_new_info`` True iff the demanded bundle was new to the manifest
    """

    allocated_bundle: Bundle
    allocated_lindahl_price: float
    allocated_manifest_value: float
    allocated_utility: float
    best_bundle: Optional[Bundle]
    best_value: Optional[float]
    best_price: Optional[float]
    best_utility: Optional[float]
    utility_gap: Optional[float]
    satisfied: bool
    demand_produced_new_info: bool = False


@dataclass(frozen=True)
class CecaRoundRecord:
    round_idx: int
    allocation: Dict[str, Bundle]  # this round's tentative WDP allocation
    satisfied_by_bidder: Dict[str, bool]  # bidder_id -> happy at this round's prices
    diagnostics: Optional[Dict[str, CecaBidderDiagnostic]] = None