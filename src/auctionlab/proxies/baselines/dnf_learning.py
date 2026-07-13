"""ωxor: a classical DNF/proper-learning baseline proxy.

This proxy implements the non-LLM-architecture baseline from the paper
(Algorithms 1-2): it starts with no information about the bidder's
preferences and learns an exact XOR bid purely from demand queries (Q_D) and
value queries (Q_V) issued to an :class:`~auctionlab.llm.person_simulator.LlmPersonSimulator`.

It conforms to the same :class:`~auctionlab.proxies.base.AuctionProxy`,
:class:`~auctionlab.proxies.base.ClockAuctionProxy`, and
:class:`~auctionlab.proxies.base.SealedAuctionProxy` protocols as
:class:`~auctionlab.proxies.full_info.FullInfoAuctionProxy` and
:class:`~auctionlab.llm.proxies.LlmAuctionProxyAdapter`, so it can be dropped
into the existing proxy-mediated clock and sealed runners as a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from auctionlab.auction_types import Bundle, Item
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.instances.base import CecaStepResponse, DemandResponse, demand_rank_key
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.proxies.base import ElicitationEvent, ProxyStats, RefinementRecord


def _upsert_atom(bid: XorBid, bundle: Bundle, value: float) -> None:
    """Add or replace the atomic bid for ``bundle`` in-place."""
    replacement = XorAtomicBid(bundle=bundle, value=value)

    for idx, atom in enumerate(bid.atoms):
        if atom.bundle == bundle:
            bid.atoms[idx] = replacement
            return

    bid.atoms.append(replacement)


@dataclass
class DnfLearningProxy:
    """Learn an exact XOR bid from demand and value queries (Algorithm 1-2).

    The candidate bid starts as a single atom ``{empty: 0}``. Each call to
    :meth:`refine`/:meth:`receive_provisional_feedback` first asks the person
    whether they are satisfied with ``event.bundle`` at ``event.prices``
    (Q_D), when prices are available (clock events). If not satisfied -- or
    if no prices are available (sealed-bid events) -- :meth:`_learn_atomic_bundle`
    (Algorithm 2) shrinks ``event.bundle`` to a minimal sub-bundle with the
    same value via value queries (Q_V), and that ``(bundle, value)`` atom is
    upserted into the candidate bid.

    When ``event.bundles`` is set (e.g. near-tie clock events), every bundle
    in that tuple is refined independently, matching the behaviour of
    :class:`~auctionlab.llm.proxies.LlmAuctionProxyAdapter`.

    Stats interpretation:
      - ``value_queries``: Q_V calls issued during :meth:`_learn_atomic_bundle`.
      - ``demand_queries``: Q_D calls from :meth:`demand_at_prices` *and* from
        the demand-satisfaction check inside :meth:`refine` (elicitation demand).
        Total LLM calls ≈ ``demand_queries + value_queries``.
      - ``refinement_queries``: number of ``(bundle, event)`` pairs processed by
        :meth:`refine`; used for per-bidder budget enforcement, not LLM accounting.
    """

    bidder_id: str
    person: LlmPersonSimulator
    items: list[Item]
    tolerance: float = 1e-9
    _bid: XorBid = field(init=False, repr=False)
    _stats: ProxyStats = field(default_factory=ProxyStats, init=False)
    _refinement_records: list[RefinementRecord] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Seed with the empty bundle plus the grand bundle at value 0, an
        # exploration probe matching Algorithm 1's price-zero initial demand
        # query. This gives the existing clock/sealed elicitation heuristics
        # (proxy_clock_runner's "near_tie" trigger, proxy_sealed_runner's
        # "allocated_bundle" feedback) a non-empty bundle to challenge on the
        # first round, kicking off learning.
        atoms = [XorAtomicBid(bundle=frozenset(), value=0.0)]
        grand_bundle = frozenset(self.items)
        if grand_bundle:
            atoms.append(XorAtomicBid(bundle=grand_bundle, value=0.0))
        self._bid = XorBid(bidder_id=self.bidder_id, atoms=atoms)

    def _value_query(self, bundle: Bundle) -> float:
        if not bundle:
            return 0.0
        self._stats.value_queries += 1
        return self.person.value_query(bundle)

    def _learn_atomic_bundle(self, bundle: Bundle) -> tuple[Bundle, float]:
        """Shrink ``bundle`` to a minimal sub-bundle with the same value.

        Two values are treated as equal when they differ by at most
        ``self.tolerance`` (absolute).  This guards against floating-point
        noise in LLM responses where stochastic rounding could prevent a
        genuinely redundant item from being removed.
        """
        shrunk = frozenset(bundle)
        value = self._value_query(shrunk)

        for item in sorted(bundle):
            if item not in shrunk:
                continue
            candidate = shrunk - {item}
            candidate_value = self._value_query(candidate)
            if abs(candidate_value - value) <= self.tolerance:
                shrunk = candidate

        return shrunk, value

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
        """Return demand at ``prices``.

        The bidder's ``top_k`` best-surplus bundles are reported as primary
        demand (``primary_bundles``) and drive excess demand / price
        increases across the union of items they contain; ``primary_bundle``
        is just the single best of those. ``supplementary_atoms`` always
        returns every atom currently in the learned bid (not capped by
        ``top_k``), so final winner determination sees everything this
        proxy has learned.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self._stats.demand_queries += 1

        ranked = sorted(
            self._bid.atoms,
            key=lambda atom: demand_rank_key(
                atom.bundle,
                atom.value,
                prices,
            ),
        )
        candidates = [
            atom
            for atom in ranked
            if atom.value - sum(prices[item] for item in atom.bundle) > 0.0
        ]
        supplementary_atoms = [
            atom for atom in self._bid.atoms if atom.bundle
        ]

        return DemandResponse(
            primary_bundle=candidates[0].bundle if candidates else None,
            supplementary_atoms=supplementary_atoms,
            primary_bundles=[atom.bundle for atom in candidates[:top_k]],
        )

    def ceca_step(
        self,
        prices: Callable[[Bundle], float],
        current_bundle: Bundle,
        round_idx: int = 0,
    ) -> CecaStepResponse:
        """CECA demand step per Algorithm 1: demand query, then Lxor_step if unsatisfied.

        Asks the person whether they prefer a different bundle given the
        current manifest atom prices.  If not satisfied, shrinks the
        preferred bundle to its minimal sub-bundle (Algorithm 2) and upserts
        that atom into the learned bid.
        """
        self._stats.demand_queries += 1
        current_bundle = frozenset(current_bundle)

        manifest_prices = {atom.bundle: atom.value for atom in self._bid.atoms}
        response = self.person.ceca_demand_query(current_bundle, manifest_prices)

        if response.satisfied:
            return CecaStepResponse(satisfied=True, demanded_bundle=None, value=None)

        preferred = (
            frozenset(response.preferred_bundle)
            if response.preferred_bundle is not None
            else current_bundle
        )
        if not preferred:
            preferred = current_bundle

        atomic_bundle, value = self._learn_atomic_bundle(preferred)
        _upsert_atom(self._bid, atomic_bundle, value)

        return CecaStepResponse(satisfied=False, demanded_bundle=atomic_bundle, value=value)

    def refine(self, event: ElicitationEvent) -> None:
        if event.bundles:
            for b in event.bundles:
                if b:
                    self._do_refine_bundle(frozenset(b), event)
        elif event.bundle:
            self._do_refine_bundle(frozenset(event.bundle), event)

    def _do_refine_bundle(self, bundle: Bundle, event: ElicitationEvent) -> None:
        """Refine a single bundle from an elicitation event."""
        self._stats.refinement_queries += 1

        if event.prices is not None:
            response = self.person.demand_query(bundle, event.prices)
            self._stats.demand_queries += 1
            if response.satisfied:
                # satisfied=True means value(bundle) >= cost at current prices:
                # use this as a lower-bound update so the bid tracks upward as
                # the clock raises prices, rather than staying at zero.
                cost = sum(event.prices.get(item, 0.0) for item in bundle)
                current = next(
                    (a.value for a in self._bid.atoms if a.bundle == bundle),
                    0.0,
                )
                if cost > current:
                    _upsert_atom(self._bid, bundle, cost)
                    self._refinement_records.append(
                        RefinementRecord(
                            bidder_id=self.bidder_id,
                            mechanism=event.mechanism,
                            event_type=event.event_type,
                            round_idx=event.round_idx,
                            bundle=bundle,
                            old_value=current or None,
                            new_value=cost,
                            reason=event.reason,
                        )
                    )
                return

        learned_bundle, value = self._learn_atomic_bundle(bundle)
        old_value = next(
            (a.value for a in self._bid.atoms if a.bundle == learned_bundle),
            None,
        )
        _upsert_atom(self._bid, learned_bundle, value)
        self._refinement_records.append(
            RefinementRecord(
                bidder_id=self.bidder_id,
                mechanism=event.mechanism,
                event_type=event.event_type,
                round_idx=event.round_idx,
                bundle=learned_bundle,
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
