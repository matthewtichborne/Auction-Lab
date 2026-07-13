"""Winner determination for combinatorial auctions with XOR bids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from auctionlab.auction_types import Bundle, Item
from auctionlab.bids.xor import XorBid


@dataclass(frozen=True)
class WdpResult:
    """A welfare-maximizing allocation and its reported welfare."""

    allocation: Dict[str, Bundle]
    welfare: float


# CP-SAT's Add() constraints require integer coefficients and bounds (unlike
# Maximize/Minimize, which tolerate floats internally); this scales monetary
# values into that integer space for the welfare-floor constraint below.
_WELFARE_SCALE = 1_000_000


def _build_candidates(
    items: List[Item], bids: List[XorBid]
) -> Tuple[List[Tuple[str, Bundle, float]], Dict[str, List[int]], Dict[Item, List[int]]]:
    """Flatten atoms once so bidder and item constraints share variable indices."""
    candidates: List[Tuple[str, Bundle, float]] = []
    cand_by_bidder: Dict[str, List[int]] = {}
    cand_by_item: Dict[Item, List[int]] = {it: [] for it in items}

    for bid in bids:
        for atom in bid.atoms:
            current_index = len(candidates)
            candidates.append((bid.bidder_id, atom.bundle, atom.value))

            if bid.bidder_id in cand_by_bidder:
                cand_by_bidder[bid.bidder_id].append(current_index)
            else:
                cand_by_bidder[bid.bidder_id] = [current_index]

            for item in atom.bundle:
                if item not in cand_by_item:
                    raise ValueError(
                        f"Bundle contains item {item} not in items universe"
                    )
                cand_by_item[item].append(current_index)

    return candidates, cand_by_bidder, cand_by_item


def _extract_allocation(
    bids: List[XorBid],
    candidates: List[Tuple[str, Bundle, float]],
    x: List[cp_model.IntVar],
    solver: cp_model.CpSolver,
) -> WdpResult:
    allocation: Dict[str, Bundle] = {b.bidder_id: frozenset() for b in bids}
    welfare = 0.0

    for candidate_idx in range(len(candidates)):
        if solver.Value(x[candidate_idx]) == 1:
            bidder_id, bundle, value = candidates[candidate_idx]
            allocation[bidder_id] = bundle
            welfare += value

    return WdpResult(allocation=allocation, welfare=welfare)


def solve_wdp_xor_ilp(items: List[Item], bids: List[XorBid]) -> WdpResult:
    """Solve the XOR winner-determination problem with a binary CP-SAT model."""
    model = cp_model.CpModel()
    candidates, cand_by_bidder, cand_by_item = _build_candidates(items, bids)

    # x[t] selects one candidate atomic bid.
    x: List[cp_model.IntVar] = []
    for t in range(len(candidates)):
        x.append(model.NewBoolVar(f"x_{t}"))

    # XOR semantics permit at most one winning atom from each bidder's bid.
    for bidder_id in cand_by_bidder:
        model.Add(sum(x[t] for t in cand_by_bidder[bidder_id]) <= 1)

    # Unit supply permits each item to appear in at most one winning bundle.
    for item in cand_by_item:
        model.Add(sum(x[t] for t in cand_by_item[item]) <= 1)

    model.Maximize(sum(candidates[t][2] * x[t] for t in range(len(candidates))))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return WdpResult(allocation={}, welfare=0.0)

    return _extract_allocation(bids, candidates, x, solver)


def solve_wdp_xor_ilp_max_winners(
    items: List[Item],
    bids: List[XorBid],
    min_welfare: float,
    *,
    tolerance: float = 1e-6,
) -> WdpResult:
    """Among welfare-maximizing allocations, prefer the one serving the most bidders.

    Re-solves the same XOR/unit-supply WDP with welfare held at or above
    ``min_welfare`` (normally the result of a prior :func:`solve_wdp_xor_ilp`
    call), but maximizes the count of bidders who win a non-empty bundle
    instead of total value. This is CECA's two-stage allocation rule: stage
    1 finds the optimal welfare, stage 2 breaks ties among
    equally-welfare-maximizing allocations by preferring broader ones.
    """
    model = cp_model.CpModel()
    candidates, cand_by_bidder, cand_by_item = _build_candidates(items, bids)

    x: List[cp_model.IntVar] = []
    for t in range(len(candidates)):
        x.append(model.NewBoolVar(f"x_{t}"))

    for bidder_id in cand_by_bidder:
        model.Add(sum(x[t] for t in cand_by_bidder[bidder_id]) <= 1)

    for item in cand_by_item:
        model.Add(sum(x[t] for t in cand_by_item[item]) <= 1)

    # Hold welfare at or above the stage-1 optimum (tolerance absorbs solver
    # floating-point artifacts; scaled to an integer bound for CP-SAT).
    model.Add(
        sum(
            round(candidates[t][2] * _WELFARE_SCALE) * x[t]
            for t in range(len(candidates))
        )
        >= round((min_welfare - tolerance) * _WELFARE_SCALE)
    )

    # Each XOR constraint already caps a bidder's contribution to this sum at
    # 1, so summing over non-empty-bundle candidates directly counts winners.
    model.Maximize(
        sum(x[t] for t in range(len(candidates)) if candidates[t][1])
    )
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return WdpResult(allocation={}, welfare=0.0)

    return _extract_allocation(bids, candidates, x, solver)
