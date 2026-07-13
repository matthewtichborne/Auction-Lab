"""Common proxy-mediated elicitation interfaces for sealed, clock, and CECA
auctions.

Mechanisms are sealed XOR VCG, ascending clock with supplementary VCG, and
CECA (Competitive Equilibrium Combinatorial Auction). Proxies maintain
candidate XOR bids and expose a generic ``refine`` hook that mechanisms can
call when they have useful feedback to share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from auctionlab.auction_types import Bundle, Item
from auctionlab.bids.xor import XorBid
from auctionlab.instances.base import CecaStepResponse, DemandResponse


def clone_xor_bid(bid: XorBid) -> XorBid:
    """Return a snapshot of ``bid`` unaffected by later in-place mutation."""
    return XorBid(bidder_id=bid.bidder_id, atoms=list(bid.atoms))


@dataclass
class ProxyStats:
    """Per-proxy accounting of how many queries a proxy has issued."""

    value_queries: int = 0
    demand_queries: int = 0
    nl_queries: int = 0
    refinement_queries: int = 0


@dataclass(frozen=True)
class RefinementRecord:
    """A single ``refine`` call: what bundle was revalued and by how much."""

    bidder_id: str
    mechanism: str
    event_type: str
    round_idx: int | None
    bundle: Bundle
    old_value: float | None
    new_value: float
    reason: str | None
    query_text: str | None = None
    response_summary: str | None = None


@dataclass(frozen=True)
class ElicitationEvent:
    """A generic mechanism-to-proxy elicitation signal.

    Examples include a sealed-bid provisional allocation, a sealed-bid
    losing-bidder feedback signal, a clock round with near-zero surplus, a
    clock round where primary demand changed, or a clock round where the
    top two candidates are nearly tied.

    ``bundle`` targets a single bundle (backward-compatible primary field).
    ``bundles`` carries multiple bundles when an event is naturally multi-bundle
    (e.g. a near-tie between two candidates). Proxies that support multi-bundle
    events should iterate over ``bundles``; single-bundle proxies fall back to
    ``bundle``.
    """

    mechanism: str
    event_type: str
    bidder_id: str
    bundle: Bundle | None = None
    bundles: tuple[Bundle, ...] | None = None
    prices: dict[Item, float] | None = None
    allocated_bundle: Bundle | None = None
    reason: str | None = None
    round_idx: int | None = None


@runtime_checkable
class AuctionProxy(Protocol):
    """Minimum interface shared by every mechanism-facing proxy."""

    bidder_id: str

    def current_bid(self) -> XorBid:
        """Return the proxy's current candidate XOR bid."""
        ...

    def refine(self, event: ElicitationEvent) -> None:
        """Update the candidate bid in response to an elicitation event."""
        ...

    def stats(self) -> ProxyStats:
        """Return the proxy's query-accounting state."""
        ...

    def refinement_records(self) -> list[RefinementRecord]:
        """Return the proxy's accumulated refinement history."""
        ...


@runtime_checkable
class ClockAuctionProxy(AuctionProxy, Protocol):
    """Proxy interface used by the ascending-clock mechanism."""

    def demand_at_prices(
        self,
        prices: dict[Item, float],
        round_idx: int,
        top_k: int,
    ) -> DemandResponse:
        """Return the proxy's demand at the given per-item prices."""
        ...


@runtime_checkable
class SealedAuctionProxy(AuctionProxy, Protocol):
    """Proxy interface used by the sealed-bid mechanism."""

    def submit_bid(self) -> XorBid:
        """Return the bid the proxy submits to the sealed mechanism."""
        ...

    def receive_provisional_feedback(self, event: ElicitationEvent) -> None:
        """Receive feedback derived from a provisional sealed allocation.

        The default behavior (provided by concrete implementations) is to
        forward the event to :meth:`refine`.
        """
        ...


@runtime_checkable
class CecaAuctionProxy(AuctionProxy, Protocol):
    """Proxy interface used by the CECA mechanism.

    A sibling protocol to :class:`ClockAuctionProxy`, not a subtype: CECA's
    prices are Lindahl-style and bundle-keyed (personalized per bidder from
    their own reported atomic bids), which doesn't fit
    ``demand_at_prices``'s shared item-keyed price vector.
    """

    def ceca_step(
        self,
        prices: Callable[[Bundle], float],
        current_bundle: Bundle,
        round_idx: int,
    ) -> CecaStepResponse:
        """Report satisfaction with ``current_bundle`` at bundle-keyed prices.

        If not satisfied, the response carries the bidder's true demanded
        bundle and value, which the mechanism folds into the bidder's
        manifest XOR bid.
        """
        ...
