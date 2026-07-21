"""CECA-specific data types (archived legacy).

This module is archived legacy code and is not part of the main
sealed/clock event-driven proxy experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from auctionlab.auction_types import Bundle


@dataclass
class CecaBidderDiagnostic:
    """Per-bidder satisfaction diagnostic captured during a CECA round step."""

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
    allocation: Dict[str, Bundle]
    satisfied_by_bidder: Dict[str, bool]
    diagnostics: Optional[Dict[str, CecaBidderDiagnostic]] = None


@dataclass(frozen=True)
class CecaStepResponse:
    """One bidder's response to a CECA personalized-price demand step."""

    satisfied: bool
    demanded_bundle: Bundle | None
    value: float | None
    diagnostic: CecaBidderDiagnostic | None = None
