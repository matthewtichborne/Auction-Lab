"""Core auction-instance representation and deterministic demand oracles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from auctionlab.auction_types import CecaBidderDiagnostic, Item, Bundle
from auctionlab.bids.xor import XorBid, XorAtomicBid


ValuationTable = Dict[str, Dict[Bundle, float]]


@dataclass(frozen=True)
class DemandResponse:
    """One round's clock demand plus atoms retained for finalization.

    ``primary_bundle`` is the bidder's single best-surplus bundle: used for
    display, tie-break detection, and previous-bundle tracking in
    elicitation logic. ``primary_bundles`` is the (possibly larger) set of
    top-k best-surplus bundles that actually drives excess demand / price
    increases -- a bidder reporting multiple packages they'd accept
    contributes demand for the union of items across all of them, which is
    how ``top_k`` makes package-level competition visible to the clock
    before it overshoots on any single item. Oracles that don't populate
    ``primary_bundles`` fall back to ``[primary_bundle]`` in the clock loop.
    """

    primary_bundle: Bundle | None
    supplementary_atoms: List[XorAtomicBid]
    primary_bundles: List[Bundle] = field(default_factory=list)


@dataclass(frozen=True)
class CecaStepResponse:
    """One bidder's response to a CECA personalized-price demand step.

    ``satisfied=True`` means the bidder is happy with ``current_bundle`` at
    the offered Lindahl-style bundle prices, ending their participation in
    this round with no atom update. ``satisfied=False`` carries the
    bidder's true demanded bundle and its value, which the mechanism folds
    into that bidder's manifest XOR bid as a new atomic bid.
    """

    satisfied: bool
    demanded_bundle: Bundle | None
    value: float | None
    diagnostic: CecaBidderDiagnostic | None = None


@dataclass(frozen=True)
class AuctionInstance:
    """A small combinatorial auction with explicit XOR valuation tables."""
    items: List[Item]
    bidder_ids: List[str]
    valuations: ValuationTable

    def to_xor_bids(self) -> List[XorBid]:
        """
        Convert the valuation table into reported XOR bids.
        """
        bids: List[XorBid] = []

        for bidder_id in self.bidder_ids:
            atoms = [
                XorAtomicBid(bundle=bundle, value=value)
                for bundle, value in self.valuations[bidder_id].items()
            ]

            bids.append(XorBid(bidder_id=bidder_id, atoms=atoms))

        return bids

    def value_of(self, bidder_id: str, bundle: Bundle) -> float:
        """Evaluate a bundle using the best contained XOR atom."""
        best = 0.0

        for atomic_bundle, value in self.valuations[bidder_id].items():
            if atomic_bundle.issubset(bundle):
                best = max(best, value)

        return best


def bundle_price(bundle: Bundle, prices: Dict[Item, float]) -> float:
    return sum(prices[item] for item in bundle)


def demand_rank_key(
    bundle: Bundle,
    value: float,
    prices: Dict[Item, float],
) -> tuple[float, int, tuple[str, ...], float]:
    """
    Lower key is better.

    Tie-break:
      1. higher surplus
      2. smaller bundle
      3. lexicographic bundle order
      4. higher value
    """
    surplus = value - bundle_price(bundle, prices)

    return (
        -surplus,
        len(bundle),
        tuple(sorted(bundle)),
        -value,
    )


def make_demand_oracle(
    instance: AuctionInstance,
    *,
    top_k: int = 1,
):
    """Build a deterministic positive-surplus demand oracle.

    The bidder's ``top_k`` best-surplus bundles are reported as primary
    demand (``primary_bundles``) and drive excess demand / price increases
    across the union of items they contain; ``primary_bundle`` is just the
    single best of those, kept for display/tie-break purposes.
    ``supplementary_atoms`` always returns the bidder's entire known
    valuation table (not capped by ``top_k``), so final winner determination
    sees every bundle this bidder is known to value, not just whichever
    bundle happened to rank best at some round's prices.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    def demand_oracle(
        bidder_id: str,
        prices: Dict[Item, float],
    ) -> DemandResponse:
        candidates: list[
            tuple[tuple[float, int, tuple[str, ...], float], Bundle, float]
        ] = []

        for bundle, value in instance.valuations[bidder_id].items():
            surplus = value - bundle_price(bundle, prices)

            if surplus > 0.0:
                candidates.append(
                    (
                        demand_rank_key(bundle, value, prices),
                        bundle,
                        value,
                    )
                )

        supplementary_atoms = [
            XorAtomicBid(bundle=bundle, value=value)
            for bundle, value in instance.valuations[bidder_id].items()
            if bundle
        ]

        if not candidates:
            return DemandResponse(
                primary_bundle=None,
                supplementary_atoms=supplementary_atoms,
                primary_bundles=[],
            )

        candidates.sort(key=lambda x: x[0])

        return DemandResponse(
            primary_bundle=candidates[0][1],
            supplementary_atoms=supplementary_atoms,
            primary_bundles=[bundle for _rank_key, bundle, _value in candidates[:top_k]],
        )

    return demand_oracle
