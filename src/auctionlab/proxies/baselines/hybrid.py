"""ωh: hybrid proxy -- NL+inference early, exact proper-learning late.

For the first ``alpha`` refinement events, this proxy delegates to an
:class:`~auctionlab.llm.proxies.LlmAuctionProxyAdapter` (ωnvd/ωvd-style
NL-and-inference). From the ``alpha``-th event onward it switches to a
:class:`~auctionlab.proxies.dnf_learning.DnfLearningProxy` (ωxor-style exact
proper learning), and the LLM proxy's still-unconfirmed inferred values are
discounted by an extra factor of ``delta`` on each subsequent refinement
(``gamma' = delta * gamma``), reflecting growing distrust of the un-verified
inference as exact learning proceeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from auctionlab.auction_types import Item
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.instances.base import DemandResponse, demand_rank_key
from auctionlab.llm.proxies import LlmAuctionProxyAdapter
from auctionlab.proxies.base import ElicitationEvent, ProxyStats, RefinementRecord
from auctionlab.proxies.baselines.dnf_learning import DnfLearningProxy


@dataclass
class HybridProxy:
    """Combine an LLM proxy (early) and a DNF-learning proxy (late)."""

    bidder_id: str
    llm_proxy: LlmAuctionProxyAdapter
    dnf_proxy: DnfLearningProxy
    alpha: int
    delta: float
    _calls: int = field(default=0, init=False)
    _gamma_factor: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        if self.alpha < 1:
            raise ValueError("alpha must be a positive integer")
        if not (0.0 < self.delta < 1.0):
            raise ValueError("delta must be in (0, 1)")

    def current_bid(self) -> XorBid:
        dnf_bid = self.dnf_proxy.current_bid()
        dnf_bundles = {atom.bundle for atom in dnf_bid.atoms}

        atoms = list(dnf_bid.atoms)
        for atom in self.llm_proxy.current_bid().atoms:
            if atom.bundle in dnf_bundles:
                continue
            atoms.append(
                XorAtomicBid(
                    bundle=atom.bundle,
                    value=atom.value * self._gamma_factor,
                )
            )

        return XorBid(bidder_id=self.bidder_id, atoms=atoms)

    def submit_bid(self) -> XorBid:
        return self.current_bid()

    def demand_at_prices(
        self,
        prices: dict[Item, float],
        round_idx: int = 0,
        top_k: int = 1,
    ) -> DemandResponse:
        """Return demand at ``prices``.

        The bidder's ``top_k`` best-surplus bundles are reported as primary
        demand (``primary_bundles``) and drive excess demand / price
        increases across the union of items they contain; ``primary_bundle``
        is just the single best of those. ``supplementary_atoms`` always
        returns every atom currently in the combined bid (not capped by
        ``top_k``), so final winner determination sees everything this
        proxy currently knows.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        bid = self.current_bid()
        ranked = sorted(
            bid.atoms,
            key=lambda atom: demand_rank_key(atom.bundle, atom.value, prices),
        )
        candidates = [
            atom
            for atom in ranked
            if atom.value - sum(prices[item] for item in atom.bundle) > 0.0
        ]
        supplementary_atoms = [atom for atom in bid.atoms if atom.bundle]

        return DemandResponse(
            primary_bundle=candidates[0].bundle if candidates else None,
            supplementary_atoms=supplementary_atoms,
            primary_bundles=[atom.bundle for atom in candidates[:top_k]],
        )

    def refine(self, event: ElicitationEvent) -> None:
        if self._calls < self.alpha:
            self.llm_proxy.refine(event)
        else:
            self.dnf_proxy.refine(event)
            if self._calls == self.alpha:
                self._gamma_factor = 1.0
            else:
                self._gamma_factor *= self.delta

        self._calls += 1

    def receive_provisional_feedback(self, event: ElicitationEvent) -> None:
        self.refine(event)

    def stats(self) -> ProxyStats:
        llm_stats = self.llm_proxy.stats()
        dnf_stats = self.dnf_proxy.stats()
        return ProxyStats(
            value_queries=llm_stats.value_queries + dnf_stats.value_queries,
            demand_queries=llm_stats.demand_queries + dnf_stats.demand_queries,
            nl_queries=llm_stats.nl_queries + dnf_stats.nl_queries,
            refinement_queries=(
                llm_stats.refinement_queries + dnf_stats.refinement_queries
            ),
        )

    def refinement_records(self) -> list[RefinementRecord]:
        return [
            *self.llm_proxy.refinement_records(),
            *self.dnf_proxy.refinement_records(),
        ]
