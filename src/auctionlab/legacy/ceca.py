"""CECA: Competitive Equilibrium Combinatorial Auction (archived legacy).

This module is archived legacy code and is not part of the main
sealed/clock event-driven proxy experiments.


An iterative tâtonnement-style mechanism, structurally distinct from both
sealed XOR VCG and the ascending clock: each round, every bidder is offered
Lindahl-style personalized bundle prices (derived from their own previously
reported atomic bids) and asked whether they're satisfied with their
tentatively-allocated bundle at those prices. A bidder who isn't satisfied
reveals one new atomic bid (their true demanded bundle and its value),
folded into their manifest XOR bid. The round loop terminates at a true
competitive equilibrium -- the round where every bidder is simultaneously
satisfied -- not at a price-convergence threshold or fixed round count
(though an explicit ``max_rounds`` safety cap is added for the LLM-driven
port; the original has none, relying on the finite bundle lattice for
eventual termination).

Final allocation is a two-stage ILP: stage 1 maximizes total welfare
(:func:`~auctionlab.solvers.wdp_ilp.solve_wdp_xor_ilp`); stage 2 re-solves
holding that welfare fixed, maximizing the count of bidders who win a
non-empty bundle
(:func:`~auctionlab.solvers.wdp_ilp.solve_wdp_xor_ilp_max_winners`) -- a
tiebreak preferring broader allocations among welfare-maximizing ties.

Two payment finalizations are supported over the same final allocation:
faithful CECA pay-as-bid (:func:`finalize_ceca_pay_as_bid`, each bidder
pays their own reported value for the bundle they win -- no Clarke-pivot
computation, consistent with the source mechanism) and a VCG variant
(:func:`finalize_ceca_vcg`) for apples-to-apples comparison against
auctionlab's other VCG-based mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from auctionlab.auction_types import Bundle, Item
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.instances.base import AuctionInstance  # noqa: F401  # re-exported for legacy callers
from auctionlab.legacy.types import CecaBidderDiagnostic, CecaRoundRecord, CecaStepResponse
from auctionlab.payments.vcg import compute_vcg_payments
from auctionlab.solvers.wdp_ilp import (
    WdpResult,
    solve_wdp_xor_ilp,
    solve_wdp_xor_ilp_max_winners,
)


@dataclass(frozen=True)
class CecaConfig:
    """Stopping parameters for the CECA round loop.

    ``max_rounds`` is a safety cap absent from the original algorithm (a
    true competitive-equilibrium fixed point has no round cap); sound
    defensive engineering for an LLM-driven port, consistent with
    :class:`~auctionlab.auctions.clock.ClockConfig.max_rounds`.

    ``stop_on_no_new_information``: when True, halt after
    ``stall_patience`` consecutive rounds in which no new manifest atom
    was inserted (i.e. every unsatisfied demand already existed in the
    manifest at the same value).  Default False preserves original
    behaviour.

    ``stall_patience``: number of consecutive stall rounds required before
    early stopping.  Ignored when ``stop_on_no_new_information`` is False.

    ``stop_on_round_no_useful_counterexamples``: when True, halt after the
    **first** round where every unsatisfied bidder produced a demand that
    trimmed to an atom already in the manifest at the same value (i.e. the
    round was entirely no-new-information).  This is strictly stronger than
    ``stop_on_no_new_information`` with ``stall_patience=1``: it fires on the
    very first stall round rather than requiring *n* consecutive ones.
    Sets ``stopped_reason = "no_useful_counterexamples"``.
    """

    max_rounds: int = 50
    stop_on_no_new_information: bool = False
    stall_patience: int = 1
    stop_on_round_no_useful_counterexamples: bool = False


@dataclass
class CecaState:
    """Mutable CECA round state: each bidder's growing manifest XOR bid."""

    round_idx: int
    manifest_bids: Dict[str, XorBid]
    allocation: Dict[str, Bundle]
    history: List[CecaRoundRecord] = field(default_factory=list)
    converged: bool = False
    stopped_reason: str = "max_rounds"


@dataclass(frozen=True)
class CecaOutcome:
    """Final WDP allocation and payments for a finished CECA run."""

    allocation: Dict[str, Bundle]
    welfare: float
    payments: Dict[str, float]


CecaStepOracle = Callable[
    [str, Callable[[Bundle], float], Bundle, int], CecaStepResponse
]


def build_lindahl_prices(
    manifest_bid: XorBid,
) -> Callable[[Bundle], float]:
    """XOR-induced Lindahl prices per Equation 4 of the paper.

    ``ϕᵢ(b) = v*_{θ̃ᵢ}(b)``: the maximum value of any manifest atom that is
    a subset of ``b``.  Under pay-as-bid, every manifest atom ``a`` satisfies
    ``ϕ(a.bundle) = a.value``, giving surplus 0.  Any superset of a manifest
    atom is priced at *at least* that atom's value, which prevents bidders
    from cycling by exploiting implicitly-free supersets.
    """
    atoms = list(manifest_bid.atoms)

    def price_fn(bundle: Bundle) -> float:
        best = 0.0
        for atom in atoms:
            if atom.bundle and atom.bundle.issubset(bundle):
                best = max(best, atom.value)
        return best

    return price_fn


def _upsert_atom(bid: XorBid, bundle: Bundle, value: float) -> bool:
    """Replace ``bundle``'s atom in ``bid`` if present, else append it.

    Returns ``True`` if the manifest actually changed (new atom inserted
    or existing atom updated to a different value), ``False`` if the atom
    was already present at the same value (within 1e-9).
    """
    for idx, atom in enumerate(bid.atoms):
        if atom.bundle == bundle:
            if abs(atom.value - value) > 1e-9:
                bid.atoms[idx] = XorAtomicBid(bundle=bundle, value=value)
                return True
            return False  # existing atom at same value — no change
    bid.atoms.append(XorAtomicBid(bundle=bundle, value=value))
    return True


def run_ceca(
    items: List[Item],
    bidder_ids: List[str],
    ceca_step_oracle: CecaStepOracle,
    cfg: CecaConfig,
    initial_manifest_bids: Dict[str, XorBid] | None = None,
) -> CecaState:
    """Run the CECA round loop until a competitive equilibrium is reached.

    Each round: solve the tentative welfare-maximizing WDP allocation over
    current manifest bids, derive each bidder's XOR-induced Lindahl prices
    from it, and ask each bidder (via the oracle) whether they're satisfied.
    A bidder who isn't satisfied has their demanded bundle folded into their
    own manifest bid as a new (or replacement) atomic bid -- the mechanism
    owns this bookkeeping, not the proxy, so ``ceca_step_oracle`` can be a
    bare callable with no proxy object required. Terminates when every
    bidder is simultaneously satisfied, or at ``cfg.max_rounds`` (recording
    ``converged=False``).

    ``initial_manifest_bids``: optional per-bidder seed bids (e.g. the
    proxy's pre-computed inferred XOR bid).  Consistent with Section 2.1
    of the paper, the auctioneer asks each proxy for an initial XOR bid
    before the first allocation round; passing it here avoids a wasted
    round where everyone is trivially unsatisfied against an empty manifest.
    """
    if initial_manifest_bids is not None:
        manifest_bids: Dict[str, XorBid] = {
            bidder_id: XorBid(
                bidder_id=bidder_id,
                atoms=list(initial_manifest_bids[bidder_id].atoms),
            )
            for bidder_id in bidder_ids
        }
    else:
        manifest_bids = {
            bidder_id: XorBid(bidder_id=bidder_id, atoms=[])
            for bidder_id in bidder_ids
        }
    state = CecaState(
        round_idx=0,
        manifest_bids=manifest_bids,
        allocation={bidder_id: frozenset() for bidder_id in bidder_ids},
    )

    stall_rounds = 0
    for round_idx in range(cfg.max_rounds):
        state.round_idx = round_idx

        tentative = solve_wdp_xor_ilp(items, list(state.manifest_bids.values()))
        state.allocation = tentative.allocation

        satisfied_by_bidder: Dict[str, bool] = {}
        manifest_changed_this_round = False
        round_diagnostics: Dict[str, CecaBidderDiagnostic] = {}

        for bidder_id in bidder_ids:
            manifest_bid = state.manifest_bids[bidder_id]
            allocated_bundle = state.allocation.get(bidder_id, frozenset())
            prices = build_lindahl_prices(manifest_bid)

            response = ceca_step_oracle(
                bidder_id, prices, allocated_bundle, round_idx
            )
            satisfied_by_bidder[bidder_id] = response.satisfied

            changed = False
            if not response.satisfied and response.demanded_bundle:
                changed = _upsert_atom(
                    manifest_bid,
                    response.demanded_bundle,
                    response.value if response.value is not None else 0.0,
                )
                if changed:
                    manifest_changed_this_round = True

            if response.diagnostic is not None:
                response.diagnostic.demand_produced_new_info = changed
                round_diagnostics[bidder_id] = response.diagnostic

        state.history.append(
            CecaRoundRecord(
                round_idx=round_idx,
                allocation=dict(state.allocation),
                satisfied_by_bidder=satisfied_by_bidder,
                diagnostics=round_diagnostics if round_diagnostics else None,
            )
        )

        if all(satisfied_by_bidder.values()):
            state.converged = True
            state.stopped_reason = "converged"
            break

        if cfg.stop_on_round_no_useful_counterexamples:
            if not manifest_changed_this_round and not all(satisfied_by_bidder.values()):
                state.stopped_reason = "no_useful_counterexamples"
                break

        if cfg.stop_on_no_new_information:
            if not manifest_changed_this_round:
                stall_rounds += 1
            else:
                stall_rounds = 0
            if stall_rounds >= cfg.stall_patience:
                state.stopped_reason = "no_new_information"
                break

    return state


def finalize_ceca(
    items: List[Item],
    manifest_bids: Dict[str, XorBid],
) -> Tuple[WdpResult, WdpResult]:
    """Two-stage final allocation: welfare-max, then winner-count-max.

    Returns ``(stage1, stage2)``; ``stage2.allocation`` is the actual final
    allocation. ``stage2.welfare`` should equal ``stage1.welfare`` --
    callers keep both visible since a mismatch beyond tolerance would
    itself be diagnostic of a constraint bug.
    """
    bids = list(manifest_bids.values())
    stage1 = solve_wdp_xor_ilp(items, bids)
    stage2 = solve_wdp_xor_ilp_max_winners(
        items, bids, min_welfare=stage1.welfare
    )
    return stage1, stage2


def finalize_ceca_pay_as_bid(
    items: List[Item],
    manifest_bids: Dict[str, XorBid],
) -> CecaOutcome:
    """Faithful CECA payments: pay your own reported value for what you win.

    No Clarke-pivot computation, matching the source mechanism. The
    economic justification is the competitive-equilibrium argument: at a
    true CE fixed point, a bidder's price for their bundle equals their own
    willingness-to-pay by construction.
    """
    _stage1, stage2 = finalize_ceca(items, manifest_bids)
    payments = {
        bidder_id: bid.value_of(stage2.allocation.get(bidder_id, frozenset()))
        for bidder_id, bid in manifest_bids.items()
    }
    return CecaOutcome(
        allocation=stage2.allocation, welfare=stage2.welfare, payments=payments
    )


def finalize_ceca_vcg(
    items: List[Item],
    manifest_bids: Dict[str, XorBid],
) -> CecaOutcome:
    """Alternate finalization: same two-stage allocation, VCG payments.

    For apples-to-apples comparison against auctionlab's other VCG-based
    mechanisms -- not part of the original CECA mechanism.
    """
    _stage1, stage2 = finalize_ceca(items, manifest_bids)
    payments = compute_vcg_payments(items, list(manifest_bids.values()), full=stage2)
    return CecaOutcome(
        allocation=stage2.allocation, welfare=stage2.welfare, payments=payments
    )
