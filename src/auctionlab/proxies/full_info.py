"""A deterministic, no-LLM proxy backed by an :class:`AuctionInstance`.

``FullInfoAuctionProxy`` plays the role of "person" and "proxy" at once: its
candidate XOR bid starts from one of a few simple initial beliefs and it
answers ``refine`` requests by looking up the bidder's true value for the
requested bundle in the underlying :class:`AuctionInstance`. This gives a
deterministic, model-noise-free baseline for testing proxy-mediated
elicitation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from auctionlab.auction_types import Bundle, CecaBidderDiagnostic, Item
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.instances.base import (
    AuctionInstance,
    CecaStepResponse,
    DemandResponse,
    demand_rank_key,
)
from auctionlab.proxies.base import ElicitationEvent, ProxyStats, RefinementRecord


_VALID_INITIAL_MODES = {"empty", "all_atoms", "selected"}


def _upsert_atom(bid: XorBid, bundle: Bundle, value: float) -> None:
    """Add or replace the atomic bid for ``bundle`` in-place."""
    replacement = XorAtomicBid(bundle=bundle, value=value)

    for idx, atom in enumerate(bid.atoms):
        if atom.bundle == bundle:
            bid.atoms[idx] = replacement
            return

    bid.atoms.append(replacement)


@dataclass
class FullInfoAuctionProxy:
    """Proxy whose "person" is an :class:`AuctionInstance`'s true valuation.

    ``initial`` controls the starting candidate bid:

    - ``"empty"``: start with no atoms.
    - ``"all_atoms"``: start with every atom from the instance's true
      valuation table for this bidder.
    - ``"selected"``: start with true-value atoms for ``initial_bundles``
      only.
    """

    bidder_id: str
    instance: AuctionInstance
    initial: str = "empty"
    initial_bundles: list[Bundle] | None = None
    _bid: XorBid = field(init=False, repr=False)
    _stats: ProxyStats = field(default_factory=ProxyStats, init=False)
    _refinement_records: list[RefinementRecord] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.initial not in _VALID_INITIAL_MODES:
            raise ValueError(
                f"initial must be one of {sorted(_VALID_INITIAL_MODES)}, "
                f"got {self.initial!r}"
            )

        atoms: list[XorAtomicBid] = []

        if self.initial == "all_atoms":
            atoms = [
                XorAtomicBid(bundle=bundle, value=value)
                for bundle, value in self.instance.valuations[
                    self.bidder_id
                ].items()
            ]
        elif self.initial == "selected":
            for bundle in self.initial_bundles or []:
                bundle = frozenset(bundle)
                if not bundle:
                    continue
                atoms.append(
                    XorAtomicBid(bundle=bundle, value=self._true_value(bundle))
                )

        self._bid = XorBid(bidder_id=self.bidder_id, atoms=atoms)

    def _true_value(self, bundle: Bundle) -> float:
        self._stats.value_queries += 1
        return self.instance.value_of(self.bidder_id, bundle)

    def current_bid(self) -> XorBid:
        return self._bid

    def submit_bid(self) -> XorBid:
        return self._bid

    def demand_at_prices(
        self,
        prices: dict[Item, float],
        round_idx: int = 0,
        top_k: int = 1,
    ) -> DemandResponse:
        """Return the utility-maximizing demand under true valuations.

        The bidder's ``top_k`` best-surplus bundles are reported as primary
        demand (``primary_bundles``) and drive excess demand / price
        increases across the union of items they contain; ``primary_bundle``
        is just the single best of those. ``supplementary_atoms`` always
        returns the bidder's entire true valuation table (not capped by
        ``top_k``), so final winner determination sees every bundle this
        bidder is known to value.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self._stats.demand_queries += 1

        candidates: list[tuple[tuple, Bundle, float]] = []
        for bundle, value in self.instance.valuations[self.bidder_id].items():
            surplus = value - sum(prices[item] for item in bundle)
            if surplus > 0.0:
                candidates.append(
                    (demand_rank_key(bundle, value, prices), bundle, value)
                )

        supplementary_atoms = [
            XorAtomicBid(bundle=bundle, value=value)
            for bundle, value in self.instance.valuations[self.bidder_id].items()
            if bundle
        ]

        if not candidates:
            return DemandResponse(
                primary_bundle=None,
                supplementary_atoms=supplementary_atoms,
                primary_bundles=[],
            )

        candidates.sort(key=lambda candidate: candidate[0])

        return DemandResponse(
            primary_bundle=candidates[0][1],
            supplementary_atoms=supplementary_atoms,
            primary_bundles=[bundle for _key, bundle, _value in candidates[:top_k]],
        )

    def ceca_step(
        self,
        prices: Callable[[Bundle], float],
        current_bundle: Bundle,
        round_idx: int = 0,
    ) -> CecaStepResponse:
        """Return the bidder's truthful response to a CECA price step.

        Ranks every bundle the bidder is known to value (plus
        ``current_bundle`` itself) by XOR-induced surplus and reports
        satisfaction only if ``current_bundle`` is already the best choice.
        Ties are broken in favour of the status-quo ``current_bundle``.
        """
        self._stats.demand_queries += 1

        candidate_bundles = set(self.instance.valuations[self.bidder_id]) | {
            current_bundle
        }

        best_bundle = current_bundle
        best_key: tuple | None = None
        for bundle in candidate_bundles:
            value = self.instance.value_of(self.bidder_id, bundle)
            surplus = value - prices(bundle)
            key = (-surplus, 0 if bundle == current_bundle else 1, tuple(sorted(bundle)), -value)
            if best_key is None or key < best_key:
                best_key = key
                best_bundle = bundle

        alloc_price = prices(current_bundle)
        alloc_value = self.instance.value_of(self.bidder_id, current_bundle)
        alloc_utility = alloc_value - alloc_price

        if best_bundle == current_bundle:
            diag = CecaBidderDiagnostic(
                allocated_bundle=current_bundle,
                allocated_lindahl_price=alloc_price,
                allocated_manifest_value=alloc_value,
                allocated_utility=alloc_utility,
                best_bundle=None,
                best_value=None,
                best_price=None,
                best_utility=None,
                utility_gap=None,
                satisfied=True,
            )
            return CecaStepResponse(satisfied=True, demanded_bundle=None, value=None, diagnostic=diag)

        best_value = self.instance.value_of(self.bidder_id, best_bundle)
        best_price = prices(best_bundle)
        best_utility = best_value - best_price
        diag = CecaBidderDiagnostic(
            allocated_bundle=current_bundle,
            allocated_lindahl_price=alloc_price,
            allocated_manifest_value=alloc_value,
            allocated_utility=alloc_utility,
            best_bundle=best_bundle,
            best_value=best_value,
            best_price=best_price,
            best_utility=best_utility,
            utility_gap=best_utility - alloc_utility,
            satisfied=False,
        )
        return CecaStepResponse(
            satisfied=False,
            demanded_bundle=best_bundle,
            value=best_value,
            diagnostic=diag,
        )

    def refine(self, event: ElicitationEvent) -> None:
        """Add or update the atom for ``event.bundle`` with its true value."""
        if not event.bundle:
            return

        bundle = frozenset(event.bundle)
        old_value = None
        for atom in self._bid.atoms:
            if atom.bundle == bundle:
                old_value = atom.value
                break

        value = self._true_value(bundle)
        self._stats.refinement_queries += 1
        _upsert_atom(self._bid, bundle, value)
        self._refinement_records.append(
            RefinementRecord(
                bidder_id=self.bidder_id,
                mechanism=event.mechanism,
                event_type=event.event_type,
                round_idx=event.round_idx,
                bundle=bundle,
                old_value=old_value,
                new_value=value,
                reason=event.reason,
            )
        )

    def receive_provisional_feedback(self, event: ElicitationEvent) -> None:
        self.refine(event)

    def stats(self) -> ProxyStats:
        return self._stats

    def refinement_records(self) -> list[RefinementRecord]:
        return list(self._refinement_records)
