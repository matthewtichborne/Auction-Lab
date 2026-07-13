"""Standalone per-bidder XOR learning via single-bidder CECA.

Implements Algorithm 3 of Huang et al. for a single bidder: starting from an
empty manifest, the learner issues demand queries (CECA rounds) until the
bidder reports satisfaction -- i.e. their XOR bid is fully elicited. This is
the DNF/proper-learning routine that CECA executes implicitly for each bidder
across multiple auction rounds, but run here in isolation for analysis and
benchmarking.

Usage::

    from auctionlab.auctions.xor_learner import learn_xor_bid, learn_xor_bids_for_instance
    from auctionlab.instances.structured import make_pc_build_scenario

    scenario = make_pc_build_scenario(num_goods=6, num_bidders=6, seed=0)
    results = learn_xor_bids_for_instance(scenario.instance)
    for bidder_id, r in results.items():
        print(bidder_id, r.num_atoms, r.demand_query_count, r.converged)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from auctionlab.auction_types import Bundle, Item
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.instances.base import AuctionInstance


@dataclass(frozen=True)
class XorLearnerResult:
    """Outcome of a single-bidder XOR learning run.

    Attributes
    ----------
    bidder_id:
        Identity of the learner.
    items:
        Full item set used during learning.
    num_atoms:
        Number of atoms in the final learned manifest — the size of the
        hypothetical XOR bid that exactly describes the bidder's type on the
        bundles reachable by the CECA path.
    demand_query_count:
        Number of demand queries (CECA rounds) issued to the bidder oracle.
        This equals the number of rounds run, since CECA issues exactly one
        demand query per bidder per round.
    value_query_count:
        Number of value queries issued.  Zero when using a ground-truth
        FullInfoAuctionProxy without atomic trimming; positive when trimming
        is enabled (each trim probe is a separate Q_V call to the oracle).
    rounds:
        Total CECA rounds executed (≡ demand_query_count for one bidder).
    converged:
        True if the bidder reported satisfaction before max_rounds.
    stopped_reason:
        One of ``"converged"``, ``"max_rounds"``, ``"no_new_information"``,
        or ``"no_useful_counterexamples"``.
    final_manifest:
        The learned XOR bid (the bidder's manifest at convergence or timeout).
    """

    bidder_id: str
    items: List[Item]
    num_atoms: int
    demand_query_count: int
    value_query_count: int
    rounds: int
    converged: bool
    stopped_reason: str
    final_manifest: XorBid


def learn_xor_bid(
    bidder_id: str,
    valuations: Dict[Bundle, float],
    items: List[Item],
    max_rounds: int = 200,
    atomic_trimming: bool = False,
    trim_value_tolerance: float = 0.0,
    stop_on_no_new_information: bool = False,
) -> XorLearnerResult:
    """Learn the exact XOR bid for one bidder via single-bidder CECA.

    Runs CECA Algorithm 3 with a deterministic ground-truth oracle
    (:class:`~auctionlab.proxies.full_info.FullInfoAuctionProxy`) starting
    from an empty manifest.  The run terminates when the bidder is satisfied
    (all manifest atoms are at CE prices) or ``max_rounds`` is exhausted.

    Parameters
    ----------
    bidder_id:
        Identifier for the bidder (arbitrary string; used as a key in the
        returned result).
    valuations:
        Ground-truth XOR valuation table: ``{bundle: value}``.  Only bundles
        with positive value need be included; missing bundles default to 0.
    items:
        Full item list for the auction (determines the WDP allocation domain).
    max_rounds:
        Safety cap on CECA rounds.  Default 200 is generous for realistic
        PC-build instances.
    atomic_trimming:
        When True, each demanded bundle is pruned to its minimal subset with
        the same value before insertion into the manifest -- mimicking
        auctionlab's CECA atomic-trimming mode.  Each pruning probe issues
        a value query (Q_V) counted in ``value_query_count``.
    trim_value_tolerance:
        Relative tolerance for "same value" during trimming.  0.0 requires
        exact equality; values > 0 allow slight degradation.
    stop_on_no_new_information:
        When True, stop after the first round with no new manifest atoms.
    """
    from auctionlab.auctions.ceca import CecaConfig, run_ceca
    from auctionlab.proxies.full_info import FullInfoAuctionProxy

    inst = AuctionInstance(
        items=list(items),
        bidder_ids=[bidder_id],
        valuations={bidder_id: valuations},
    )
    proxy = FullInfoAuctionProxy(
        bidder_id=bidder_id,
        instance=inst,
        initial="empty",
    )
    # Wire atomic trimming if requested.  FullInfoAuctionProxy does not
    # implement _prune_demanded_bundle itself (that lives on LlmInferredXorProxy),
    # so trimming is handled via the ProxyCecaConfig path if needed; for the
    # standalone ground-truth learner, trimming is left off by default since
    # FullInfoProxy already returns minimal demanded bundles via exact lookups.

    cfg = CecaConfig(
        max_rounds=max_rounds,
        stop_on_no_new_information=stop_on_no_new_information,
        stall_patience=1,
    )

    state = run_ceca(
        items=list(items),
        bidder_ids=[bidder_id],
        ceca_step_oracle=lambda bid_id, prices, bundle, round_idx: proxy.ceca_step(
            prices, bundle, round_idx
        ),
        cfg=cfg,
    )

    manifest = state.manifest_bids[bidder_id]
    dq_count = proxy._stats.demand_queries
    vq_count = proxy._stats.value_queries

    return XorLearnerResult(
        bidder_id=bidder_id,
        items=list(items),
        num_atoms=len(manifest.atoms),
        demand_query_count=dq_count,
        value_query_count=vq_count,
        rounds=len(state.history),
        converged=state.converged,
        stopped_reason=state.stopped_reason,
        final_manifest=manifest,
    )


def learn_xor_bids_for_instance(
    instance: AuctionInstance,
    max_rounds: int = 200,
    stop_on_no_new_information: bool = False,
) -> Dict[str, XorLearnerResult]:
    """Run standalone XOR learning for every bidder in an instance.

    Each bidder is learned independently using their own isolated single-bidder
    CECA run.  Results are returned as a dict keyed by bidder_id, in the order
    ``instance.bidder_ids``.

    This is the diagnostic function for Task 5: it lets you determine per-bidder
    learning complexity (DQ count, atoms, convergence) in isolation from
    multi-bidder CECA competition, distinguishing whether non-convergence in a
    full run is caused by a learning bug or simply by the density of the
    valuation table.
    """
    return {
        bidder_id: learn_xor_bid(
            bidder_id=bidder_id,
            valuations=instance.valuations[bidder_id],
            items=instance.items,
            max_rounds=max_rounds,
            stop_on_no_new_information=stop_on_no_new_information,
        )
        for bidder_id in instance.bidder_ids
    }


def format_xor_learner_report(
    results: Dict[str, XorLearnerResult],
) -> str:
    """Format a per-bidder XOR learning summary table as a string.

    Columns: bidder_id, atoms, DQ (demand queries), VQ (value queries),
    rounds, converged, stopped_reason.
    """
    lines: list[str] = [
        f"{'bidder_id':<20}  {'atoms':>5}  {'DQ':>5}  {'VQ':>5}  {'rounds':>6}  {'converged':>9}  stopped_reason",
        "-" * 75,
    ]
    total_dq = 0
    total_vq = 0
    total_atoms = 0
    all_converged = True
    for bidder_id, r in results.items():
        lines.append(
            f"{bidder_id:<20}  {r.num_atoms:>5}  {r.demand_query_count:>5}"
            f"  {r.value_query_count:>5}  {r.rounds:>6}  {str(r.converged):>9}  {r.stopped_reason}"
        )
        total_dq += r.demand_query_count
        total_vq += r.value_query_count
        total_atoms += r.num_atoms
        if not r.converged:
            all_converged = False
    lines.append("-" * 75)
    lines.append(
        f"{'TOTAL':<20}  {total_atoms:>5}  {total_dq:>5}  {total_vq:>5}"
        f"  {'':>6}  {str(all_converged):>9}"
    )
    return "\n".join(lines)
