"""Proxy-mediated CECA experiments.

``run_ceca`` remains the pure CECA engine: it only knows about a
``ceca_step_oracle`` callable and owns its own growing ``manifest_bids``
state. This module builds that oracle on top of
:class:`~auctionlab.proxies.base.CecaAuctionProxy` proxies, runs the round
loop once, then finalizes payments via whichever of
:func:`~auctionlab.auctions.ceca.finalize_ceca_pay_as_bid` /
:func:`~auctionlab.auctions.ceca.finalize_ceca_vcg`
``proxy_config.payment_rule`` selects.

The two-phase API is preferred for multi-payment-rule comparisons:

    shared = run_proxy_ceca_elicitation(instance, proxies, ceca_cfg, proxy_cfg)
    result_pab = finalize_proxy_ceca_result(instance, shared, "pay_as_bid")
    result_vcg = finalize_proxy_ceca_result(instance, shared, "vcg")

This guarantees that both payment reports come from the same CECA run
(same manifest, same allocation, same rounds), differing only in payment
arithmetic. The legacy ``run_proxy_ceca_experiment`` wrapper is kept for
backward compatibility with single-payment callers.

``ProxyCecaConfig.initial_bid_mode`` controls how the CECA manifest is seeded:

- ``full_proxy`` (default / "prior"): full current proxy bid.
- ``singletons``: singleton atoms only; complements must be discovered.
- ``empty``: no atoms; CECA discovers everything from scratch.
"""

from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from dataclasses import dataclass

from auctionlab.auction_types import Bundle, CecaBidderDiagnostic, validate_bidder_keys
from auctionlab.llm.proxies import CecaTrimResult
from auctionlab.auctions.ceca import (
    CecaConfig,
    CecaState,
    finalize_ceca,
    run_ceca,
)
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.experiments.runner import MechanismResult
from auctionlab.instances.base import AuctionInstance, CecaStepResponse
from auctionlab.payments.vcg import compute_vcg_payments
from auctionlab.proxies.base import CecaAuctionProxy, clone_xor_bid
from auctionlab.solvers.wdp_ilp import WdpResult


MECHANISM_NAME = "proxy_ceca"

_VALID_PAYMENT_RULES = {"pay_as_bid", "vcg"}
_VALID_INITIAL_BID_MODES = {"full_proxy", "singletons", "empty"}
_VALID_DEMAND_UNIVERSES = {
    "all_items",
    "interested_items",
    "candidate_bundles",
    "manifest_plus_candidates",
}

_MODE_TO_VARIANT: dict[str, str] = {
    "full_proxy": "prior",
    "singletons": "singletons",
    "empty": "empty",
}

_PROXY_TYPE_TO_ARCHITECTURE: dict[str, str] = {
    "LlmAuctionProxyAdapter": "modular_llm",
    "NvdCecaProxy": "nvd",
    "Vd2CecaProxy": "vd2",
    "Vd1CecaProxy": "vd1",
    "DnfLearningProxy": "dnf",
    "HybridProxy": "hybrid",
    "FullInfoAuctionProxy": "full_info",
}


def _proxy_architecture(proxy: CecaAuctionProxy) -> str:
    return _PROXY_TYPE_TO_ARCHITECTURE.get(type(proxy).__name__, "unknown")


@dataclass(frozen=True)
class ProxyCecaConfig:
    """Configuration for the proxy-mediated CECA elicitation phase."""

    payment_rule: str = "pay_as_bid"
    initial_bid_mode: str = "full_proxy"
    atomic_trimming: bool = True
    trim_value_tolerance: float = 0.0
    stop_on_no_new_information: bool = False
    stall_patience: int = 1
    stop_on_round_no_useful_counterexamples: bool = False
    exhaust_repeated_bidders: bool = False
    bidder_stall_patience: int = 3
    ceca_demand_universe: str = "all_items"
    max_bundle_size: int | None = None

    def __post_init__(self) -> None:
        if self.payment_rule not in _VALID_PAYMENT_RULES:
            raise ValueError(
                f"payment_rule must be one of {sorted(_VALID_PAYMENT_RULES)}, "
                f"got {self.payment_rule!r}"
            )
        if self.initial_bid_mode not in _VALID_INITIAL_BID_MODES:
            raise ValueError(
                f"initial_bid_mode must be one of {sorted(_VALID_INITIAL_BID_MODES)}, "
                f"got {self.initial_bid_mode!r}"
            )
        if self.trim_value_tolerance < 0:
            raise ValueError("trim_value_tolerance must be >= 0")
        if self.ceca_demand_universe not in _VALID_DEMAND_UNIVERSES:
            raise ValueError(
                f"ceca_demand_universe must be one of {sorted(_VALID_DEMAND_UNIVERSES)}, "
                f"got {self.ceca_demand_universe!r}"
            )

    @property
    def ceca_variant(self) -> str:
        """Human-readable variant label: 'prior', 'singletons', or 'empty'."""
        return _MODE_TO_VARIANT[self.initial_bid_mode]


def _filter_initial_bid(bid: XorBid, mode: str) -> XorBid:
    """Return a new XorBid filtered according to ``mode``."""
    if mode == "full_proxy":
        atoms: list[XorAtomicBid] = list(bid.atoms)
    elif mode == "singletons":
        atoms = [a for a in bid.atoms if a.bundle and len(a.bundle) == 1]
    else:  # empty
        atoms = []
    return XorBid(bidder_id=bid.bidder_id, atoms=atoms)


@dataclass
class _DemandRecord:
    """One bidder's response to one CECA round's price step."""

    round_idx: int
    bidder_id: str
    allocated_bundle: Bundle
    satisfied: bool
    demanded_bundle: Bundle | None
    demanded_value: float | None
    trim_result: CecaTrimResult | None = None
    no_new_information: bool = False
    out_of_universe: bool = False
    projected: bool = False


@dataclass(frozen=True)
class AtomInsertionRecord:
    """One atom that was genuinely new or updated in the CECA manifest.

    Repeated demands (``no_new_information=True``) are NOT recorded here —
    only demands that caused ``_upsert_atom`` to return ``True``.
    ``raw_outside_interest`` and ``inserted_outside_interest`` are ``None``
    when no interest map is available for the bidder.
    """

    round_idx: int
    bidder_id: str
    raw_demanded_bundle: Bundle
    inserted_bundle: Bundle
    inserted_value: float
    is_new: bool
    is_update: bool
    raw_outside_interest: bool | None
    inserted_outside_interest: bool | None


@dataclass(frozen=True)
class DemandTraceRecord:
    """Full trace for one bidder-round demand event.

    Includes Lindahl prices for both the allocated bundle and the demanded
    bundle, raw vs trimmed bundles, and same-as-previous flags to identify
    stalled bidders.  ``bidder_exhausted=True`` means the demand query was
    skipped because the bidder was exhausted; in that case most fields are None.
    """

    round_idx: int
    bidder_id: str
    allocated_bundle: Bundle
    allocated_price: float
    demanded_bundle: Bundle | None
    demanded_price: float | None
    raw_demanded_bundle: Bundle | None
    trimmed_atom: Bundle | None
    satisfied: bool
    inserted_or_updated: bool
    no_new_information: bool
    same_as_previous_demand: bool
    same_as_previous_trimmed_atom: bool
    bidder_exhausted: bool
    out_of_universe: bool = False
    projected: bool = False


def _compute_allowed_bundles_by_bidder(
    universe_mode: str,
    original_bids: "dict[str, XorBid]",
    interest_items_by_bidder: "dict[str, frozenset | None]",
    max_bundle_size: "int | None",
) -> "dict[str, frozenset[frozenset] | None]":
    """Return per-bidder allowed bundle sets for the given universe mode.

    Returns ``None`` for a bidder when there is no restriction on that bidder
    (``all_items`` mode, or ``interested_items`` with no interest map).
    For ``manifest_plus_candidates`` the returned sets are the static candidate
    portion; the dynamic manifest portion is added inside the oracle per round.
    """
    if universe_mode == "all_items":
        return {b: None for b in original_bids}

    result: dict[str, frozenset | None] = {}
    for bidder_id, bid in original_bids.items():
        if universe_mode in ("candidate_bundles", "manifest_plus_candidates"):
            bundles = frozenset(a.bundle for a in bid.atoms if a.bundle)
            result[bidder_id] = bundles if bundles else None
        elif universe_mode == "interested_items":
            items = interest_items_by_bidder.get(bidder_id)
            if not items:
                result[bidder_id] = None
            else:
                max_sz = max_bundle_size if max_bundle_size is not None else len(items)
                subsets: list[frozenset] = []
                for size in range(1, min(max_sz, len(items)) + 1):
                    for combo in itertools.combinations(sorted(items), size):
                        subsets.append(frozenset(combo))
                result[bidder_id] = frozenset(subsets) if subsets else None
        else:
            result[bidder_id] = None
    return result


def _enforce_universe_constraint(
    atom: "Bundle",
    value: float,
    current_bundle: "Bundle",
    allowed: "frozenset[frozenset]",
    original_bid: "XorBid | None",
    prices,
) -> "tuple[Bundle | None, float, bool]":
    """Project an out-of-universe demand to the best admissible bundle.

    Returns ``(projected_atom, projected_value, was_projected)``.  If no
    admissible bundle is strictly preferred over ``current_bundle`` under the
    ``original_bid``'s reported values, returns ``(None, 0.0, False)`` to
    signal that this demand round should be skipped (no insertion).
    """
    if not allowed or original_bid is None:
        return None, 0.0, False

    alloc_util = original_bid.value_of(current_bundle) - prices(current_bundle)
    best_atom: "Bundle | None" = None
    best_util = alloc_util  # only project when strictly better than current allocation

    for b in allowed:
        util = original_bid.value_of(b) - prices(b)
        if util > best_util:
            best_util = util
            best_atom = b

    if best_atom is None:
        return None, 0.0, False

    return best_atom, original_bid.value_of(best_atom), True


def _make_tracked_oracle(
    proxies_by_bidder: dict[str, CecaAuctionProxy],
    records: list[_DemandRecord],
    initial_bids: dict[str, "XorBid"],
    trim_tolerance: float = 0.0,
    *,
    proxy_config: "ProxyCecaConfig | None" = None,
    interest_items_by_bidder: "dict[str, frozenset | None] | None" = None,
    atom_insertion_log: "list[AtomInsertionRecord] | None" = None,
    demand_trace: "list[DemandTraceRecord] | None" = None,
    exhaustion_events: "list[dict] | None" = None,
    universe_mode: str = "all_items",
    allowed_bundles_by_bidder: "dict[str, frozenset | None] | None" = None,
    original_bids_snap: "dict[str, XorBid] | None" = None,
):
    """Wrap the CECA step oracle to record every demand response.

    Maintains a local ``manifest_state`` mirror to detect when a demanded
    bundle is already in the bidder's manifest at approximately the same
    value — recorded as ``no_new_information=True`` in the demand record.

    Optional keyword arguments activate additional diagnostics:

    ``atom_insertion_log``: list — appended with ``AtomInsertionRecord``
    for each demand that genuinely changes the manifest (new or updated atom).
    Repeated no-new-information demands are NOT appended.

    ``demand_trace``: list — appended with ``DemandTraceRecord`` for every
    demand event (including satisfied bidders and exhausted-bidder skips).

    ``exhaustion_events``: list — appended with a dict each time a bidder is
    first marked exhausted (requires ``proxy_config.exhaust_repeated_bidders``).

    ``interest_items_by_bidder``: used to populate ``raw_outside_interest``
    and ``inserted_outside_interest`` on ``AtomInsertionRecord``.

    ``proxy_config.exhaust_repeated_bidders``: when True, bidders that return
    the same trimmed atom ``bidder_stall_patience`` consecutive times (with
    unchanged allocation) are skipped — the oracle returns satisfied=True for
    them to avoid wasted LLM queries.
    """
    manifest_state: dict[str, dict[frozenset, float]] = {
        bidder_id: {a.bundle: a.value for a in bid.atoms}
        for bidder_id, bid in initial_bids.items()
    }

    _exhaust_enabled = (
        proxy_config is not None
        and getattr(proxy_config, "exhaust_repeated_bidders", False)
    )
    _bidder_patience = (
        getattr(proxy_config, "bidder_stall_patience", 3)
        if proxy_config is not None
        else 3
    )

    # Per-bidder mutable state for same-as-previous and exhaustion tracking.
    last_demanded: dict[str, Bundle | None] = {b: None for b in proxies_by_bidder}
    last_trimmed: dict[str, Bundle | None] = {b: None for b in proxies_by_bidder}
    last_alloc: dict[str, Bundle] = {b: frozenset() for b in proxies_by_bidder}
    consecutive_same: dict[str, int] = {b: 0 for b in proxies_by_bidder}
    exhausted: set[str] = set()

    def oracle(bidder_id: str, prices, current_bundle: Bundle, round_idx: int):
        current_bundle = frozenset(current_bundle)

        # Reset exhaustion when the bidder's allocated bundle changes.
        if current_bundle != last_alloc[bidder_id]:
            exhausted.discard(bidder_id)
            consecutive_same[bidder_id] = 0
            last_alloc[bidder_id] = current_bundle

        # Exhausted bidder: skip the proxy call, return satisfied.
        if _exhaust_enabled and bidder_id in exhausted:
            records.append(_DemandRecord(
                round_idx=round_idx,
                bidder_id=bidder_id,
                allocated_bundle=current_bundle,
                satisfied=True,
                demanded_bundle=None,
                demanded_value=None,
                trim_result=None,
                no_new_information=False,
            ))
            if demand_trace is not None:
                demand_trace.append(DemandTraceRecord(
                    round_idx=round_idx,
                    bidder_id=bidder_id,
                    allocated_bundle=current_bundle,
                    allocated_price=prices(current_bundle),
                    demanded_bundle=None,
                    demanded_price=None,
                    raw_demanded_bundle=None,
                    trimmed_atom=None,
                    satisfied=True,
                    inserted_or_updated=False,
                    no_new_information=False,
                    same_as_previous_demand=True,
                    same_as_previous_trimmed_atom=True,
                    bidder_exhausted=True,
                ))
            return CecaStepResponse(satisfied=True)

        proxy = proxies_by_bidder[bidder_id]
        response = proxy.ceca_step(prices, current_bundle, round_idx)

        trim_result: CecaTrimResult | None = None
        if not response.satisfied:
            trim_result = getattr(proxy, "last_trim_result", None)
            if trim_result is None:
                inner = getattr(proxy, "proxy", None)
                trim_result = getattr(inner, "_last_trim_result", None)

        no_new_info = False
        is_new = False
        is_update = False
        out_of_universe = False
        projected = False
        allocated_price = prices(current_bundle)
        demanded_price: float | None = None

        if not response.satisfied and response.demanded_bundle is not None:
            # inserted_atom is the trimmed bundle (or raw bundle if no trimming).
            raw_bundle_pre = trim_result.raw_bundle if trim_result else response.demanded_bundle
            inserted_atom_pre = (
                trim_result.trimmed_bundle if trim_result else response.demanded_bundle
            )
            demanded_pre = response.demanded_bundle
            val_pre = response.value or 0.0
            demanded_price = prices(demanded_pre)

            # --- Universe enforcement ---
            # Compute the allowed set for this bidder this round.
            current_allowed: frozenset | None = None
            if universe_mode != "all_items" and allowed_bundles_by_bidder is not None:
                static_allowed = allowed_bundles_by_bidder.get(bidder_id)
                if universe_mode == "manifest_plus_candidates":
                    # Dynamic: union of pre-CECA candidates and current manifest.
                    current_allowed = (static_allowed or frozenset()) | frozenset(
                        manifest_state[bidder_id].keys()
                    )
                else:
                    current_allowed = static_allowed

            if current_allowed is not None and inserted_atom_pre not in current_allowed:
                out_of_universe = True
                orig_bid = (
                    original_bids_snap.get(bidder_id)
                    if original_bids_snap is not None
                    else None
                )
                proj_atom, proj_val, _ = _enforce_universe_constraint(
                    inserted_atom_pre, val_pre, current_bundle,
                    current_allowed, orig_bid, prices,
                )
                if proj_atom is None:
                    # No viable admissible alternative — skip this demand.
                    records.append(_DemandRecord(
                        round_idx=round_idx,
                        bidder_id=bidder_id,
                        allocated_bundle=current_bundle,
                        satisfied=False,
                        demanded_bundle=None,
                        demanded_value=None,
                        trim_result=trim_result,
                        no_new_information=False,
                        out_of_universe=True,
                        projected=False,
                    ))
                    same_demand_skip = last_demanded[bidder_id] == demanded_pre
                    same_trim_skip = last_trimmed[bidder_id] == inserted_atom_pre
                    if demand_trace is not None:
                        demand_trace.append(DemandTraceRecord(
                            round_idx=round_idx,
                            bidder_id=bidder_id,
                            allocated_bundle=current_bundle,
                            allocated_price=allocated_price,
                            demanded_bundle=None,
                            demanded_price=demanded_price,
                            raw_demanded_bundle=raw_bundle_pre,
                            trimmed_atom=inserted_atom_pre,
                            satisfied=False,
                            inserted_or_updated=False,
                            no_new_information=False,
                            same_as_previous_demand=same_demand_skip,
                            same_as_previous_trimmed_atom=same_trim_skip,
                            bidder_exhausted=False,
                            out_of_universe=True,
                            projected=False,
                        ))
                    last_demanded[bidder_id] = demanded_pre
                    last_trimmed[bidder_id] = inserted_atom_pre
                    return CecaStepResponse(satisfied=False, demanded_bundle=None, value=None)
                else:
                    # Successfully projected to an admissible bundle.
                    projected = True
                    response = CecaStepResponse(
                        satisfied=False, demanded_bundle=proj_atom, value=proj_val
                    )
                    demanded_price = prices(proj_atom)
                    # Proceed below with proj_atom / proj_val as the effective demand.
                    raw_bundle_pre = raw_bundle_pre  # keep original raw for logging
                    inserted_atom_pre = proj_atom
                    demanded_pre = proj_atom
                    val_pre = proj_val

            demanded = demanded_pre
            val = val_pre
            inserted_atom_eff = inserted_atom_pre  # may have been overwritten by projection

            existing = manifest_state[bidder_id].get(demanded)
            if existing is None:
                is_new = True
            elif abs(existing - val) <= trim_tolerance:
                no_new_info = True
            else:
                is_update = True
            manifest_state[bidder_id][demanded] = val

            # Atom insertion log: only new or updated atoms.
            if atom_insertion_log is not None and (is_new or is_update):
                int_items = (
                    interest_items_by_bidder.get(bidder_id)
                    if interest_items_by_bidder is not None
                    else None
                )
                raw_out = (
                    None if int_items is None
                    else not raw_bundle_pre.issubset(int_items)
                )
                ins_out = (
                    None if int_items is None else not demanded.issubset(int_items)
                )
                atom_insertion_log.append(AtomInsertionRecord(
                    round_idx=round_idx,
                    bidder_id=bidder_id,
                    raw_demanded_bundle=raw_bundle_pre,
                    inserted_bundle=demanded,
                    inserted_value=val,
                    is_new=is_new,
                    is_update=is_update,
                    raw_outside_interest=raw_out,
                    inserted_outside_interest=ins_out,
                ))
        else:
            inserted_atom_eff = None
            raw_bundle_pre = None

        # same-as-previous tracking.
        same_demand = last_demanded[bidder_id] == (
            response.demanded_bundle if not response.satisfied else None
        )
        same_trim = last_trimmed[bidder_id] == inserted_atom_eff
        last_demanded[bidder_id] = (
            response.demanded_bundle if not response.satisfied else None
        )
        last_trimmed[bidder_id] = inserted_atom_eff

        # Exhaustion tracking: K consecutive same trimmed atom.
        if _exhaust_enabled and not response.satisfied and inserted_atom_eff is not None:
            if same_trim:
                consecutive_same[bidder_id] += 1
            else:
                consecutive_same[bidder_id] = 1
            if (
                consecutive_same[bidder_id] >= _bidder_patience
                and bidder_id not in exhausted
            ):
                exhausted.add(bidder_id)
                if exhaustion_events is not None:
                    exhaustion_events.append({
                        "round_idx": round_idx,
                        "bidder_id": bidder_id,
                        "trimmed_atom": inserted_atom_eff,
                        "consecutive_count": consecutive_same[bidder_id],
                    })

        records.append(_DemandRecord(
            round_idx=round_idx,
            bidder_id=bidder_id,
            allocated_bundle=current_bundle,
            satisfied=response.satisfied,
            demanded_bundle=response.demanded_bundle if not response.satisfied else None,
            demanded_value=response.value if not response.satisfied else None,
            trim_result=trim_result if not response.satisfied else None,
            no_new_information=no_new_info,
            out_of_universe=out_of_universe,
            projected=projected,
        ))

        if demand_trace is not None:
            demand_trace.append(DemandTraceRecord(
                round_idx=round_idx,
                bidder_id=bidder_id,
                allocated_bundle=current_bundle,
                allocated_price=allocated_price,
                demanded_bundle=response.demanded_bundle if not response.satisfied else None,
                demanded_price=demanded_price,
                raw_demanded_bundle=(
                    raw_bundle_pre if not response.satisfied else None
                ),
                trimmed_atom=inserted_atom_eff,
                satisfied=response.satisfied,
                inserted_or_updated=is_new or is_update,
                no_new_information=no_new_info,
                same_as_previous_demand=same_demand,
                same_as_previous_trimmed_atom=same_trim,
                bidder_exhausted=False,
                out_of_universe=out_of_universe,
                projected=projected,
            ))

        return response

    return oracle


def _pruning_query_count(proxy: CecaAuctionProxy) -> int:
    return getattr(proxy, "pruning_query_count", 0)


@dataclass
class ProxyCecaSharedResult:
    """CECA state after elicitation, before payment finalization.

    Returned by :func:`run_proxy_ceca_elicitation`. Pass to
    :func:`finalize_proxy_ceca_result` once per payment rule to get
    ``MechanismResult`` objects that share identical allocation and
    manifest state, differing only in payments.
    """

    ceca_state: CecaState
    initial_bids: dict[str, XorBid]
    initial_manifest_sizes: dict[str, int]
    final_manifest_sizes: dict[str, int]
    manifest_growth_by_bidder: dict[str, int]
    manifest_growth_total: int
    demand_records: list[_DemandRecord]
    demand_query_count_by_bidder: dict[str, int]
    pruning_query_count_by_bidder: dict[str, int]
    # Duplicate demand diagnostics.
    demanded_bundle_count_by_bidder: dict[str, int]
    unique_demanded_bundle_count_by_bidder: dict[str, int]
    duplicate_demand_count_by_bidder: dict[str, int]
    unchanged_demand_count_by_bidder: dict[str, int]
    proxy_architecture: str
    ceca_variant: str
    ceca_initial_bid_mode: str
    ceca_atomic_trimming: bool
    ceca_trim_value_tolerance: float
    ceca_trimmed_demand_count: int
    ceca_total_trim_items_removed: int
    ceca_total_trim_value_queries: int
    ceca_avg_raw_demand_size: float
    ceca_avg_trimmed_atom_size: float
    ceca_repeated_raw_demand_count: int
    ceca_repeated_trimmed_atom_count: int
    # Corrected trim diagnostics (Feature 1).
    num_trim_attempts: int
    num_demands_trimmed_to_smaller_atom: int
    total_raw_demand_items: int
    total_inserted_atom_items: int
    total_net_items_removed: int
    avg_raw_demand_size: float
    avg_inserted_atom_size: float
    # No-new-information diagnostics (Feature 2).
    no_new_information_count_by_bidder: dict[str, int]
    repeated_trimmed_atom_count_by_bidder: dict[str, int]
    total_no_new_information: int
    # Useful counterexamples = demanded - no_new_information (demands that changed manifest).
    useful_counterexample_count_by_bidder: dict[str, int]
    total_useful_counterexamples: int
    # Stopped reason (Feature 3).
    stopped_reason: str
    # Detailed insertion diagnostics (Tasks 1–2).
    atom_insertion_log: list[AtomInsertionRecord]
    outside_interest_insertion_count_by_bidder: dict[str, int]
    # Per-demand trace (Task 3).
    demand_trace: list[DemandTraceRecord]
    # Per-round no-info stats (Task 4).
    no_info_round_count: int
    no_info_bidder_count_by_round: dict[int, int]
    bidders_exhausted_by_repetition: list[str]
    # Bidder exhaustion (Task 5).
    exhaustion_events: list[dict]
    # Demand universe constraint (Task 6).
    ceca_demand_universe: str
    out_of_universe_demand_count: int
    rejected_out_of_universe_count: int
    projected_demand_count: int
    allowed_bundle_count_by_bidder: dict[str, int]
    # Pre-computed final allocation — shared across payment rules so PAB and
    # VCG always report the same allocation/welfare regardless of ILP tie-breaking.
    final_stage1: WdpResult
    final_stage2: WdpResult


def _print_initial_manifest_summary(
    mode: str,
    initial_bids: dict[str, XorBid],
    original_bids: dict[str, XorBid],
) -> None:
    print(f"  CECA initial manifest  (mode={mode})", flush=True)
    header = (
        f"    {'bidder_id':<20}"
        f"  {'init_atoms':>10}"
        f"  {'singleton':>9}"
        f"  {'multi_item':>10}"
        f"  {'proxy_total':>11}"
    )
    print(header, flush=True)
    for bidder_id in sorted(initial_bids):
        bid = initial_bids[bidder_id]
        orig = original_bids[bidder_id]
        n_init = len(bid.atoms)
        n_sing = sum(1 for a in bid.atoms if a.bundle and len(a.bundle) == 1)
        n_multi = n_init - n_sing
        n_proxy = len(orig.atoms)
        if n_multi > 0 and mode == "singletons":
            print(
                f"  WARNING: singletons mode still has {n_multi} multi-item atoms "
                f"for {bidder_id} — filtering bug",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"    {bidder_id:<20}"
            f"  {n_init:>10}"
            f"  {n_sing:>9}"
            f"  {n_multi:>10}"
            f"  {n_proxy:>11}",
            flush=True,
        )


def _print_ceca_summary(
    mode: str,
    rounds: int,
    converged: bool,
    initial_sizes: dict[str, int],
    final_sizes: dict[str, int],
    demanded_counts: dict[str, int],
    duplicate_counts: dict[str, int] | None = None,
    unchanged_counts: dict[str, int] | None = None,
    *,
    atomic_trimming: bool = True,
    trim_value_tolerance: float = 0.0,
    ceca_total_trim_value_queries: int = 0,
    ceca_total_trim_items_removed: int = 0,
    ceca_avg_raw_demand_size: float = 0.0,
    ceca_avg_trimmed_atom_size: float = 0.0,
    ceca_repeated_raw_demand_count: int = 0,
    ceca_repeated_trimmed_atom_count: int = 0,
    # Feature 1: corrected trim diagnostics.
    num_trim_attempts: int = 0,
    num_demands_trimmed_to_smaller_atom: int = 0,
    total_raw_demand_items: int = 0,
    total_inserted_atom_items: int = 0,
    total_net_items_removed: int = 0,
    avg_raw_demand_size: float = 0.0,
    avg_inserted_atom_size: float = 0.0,
    trim_examples: list | None = None,
    # Feature 2: no-new-information diagnostics.
    total_no_new_information: int = 0,
    no_new_information_count_by_bidder: dict[str, int] | None = None,
    # Useful counterexamples (demanded - no_new_information).
    total_useful_counterexamples: int | None = None,
    useful_counterexample_count_by_bidder: dict[str, int] | None = None,
    # Feature 3: stopped reason.
    stopped_reason: str = "",
    stall_patience: int = 1,
    max_rounds: int = 0,
) -> None:
    init_total = sum(initial_sizes.values())
    final_total = sum(final_sizes.values())
    growth_total = final_total - init_total
    growth_by = {b: final_sizes[b] - initial_sizes.get(b, 0) for b in final_sizes}
    growth_str = "  ".join(
        f"{b}:+{g}" for b, g in sorted(growth_by.items()) if g > 0
    ) or "none"
    demanded_str = "  ".join(
        f"{b}:{c}" for b, c in sorted(demanded_counts.items()) if c > 0
    ) or "none"
    _max_str = f"{max_rounds}" if max_rounds else f"{rounds}"
    if stopped_reason == "converged":
        status_str = "converged"
    elif stopped_reason == "no_new_information":
        status_str = f"stopped: no_new_information (stall_patience={stall_patience})"
    elif stopped_reason == "no_useful_counterexamples":
        status_str = "stopped: no_useful_counterexamples"
    elif stopped_reason == "bidder_exhaustion":
        status_str = "stopped: bidder_exhaustion"
    elif stopped_reason == "max_rounds":
        status_str = f"NOT converged: max_rounds={_max_str}"
    else:
        # Fallback for callers that don't pass stopped_reason.
        status_str = "converged" if converged else f"NOT converged: max_rounds={_max_str}"
    lines = [
        f"  CECA done  mode={mode}  rounds={rounds}  stopped_reason={stopped_reason or ('converged' if converged else 'max_rounds')}  {status_str}",
        f"    init atoms total={init_total}  final atoms total={final_total}  "
        f"growth={growth_total}",
        f"    growth by bidder: {growth_str}",
        f"    demanded bundles: {demanded_str}",
    ]
    if duplicate_counts:
        dup_total = sum(duplicate_counts.values())
        if dup_total > 0:
            dup_str = "  ".join(
                f"{b}:{c}" for b, c in sorted(duplicate_counts.items()) if c > 0
            )
            lines.append(f"    duplicate demands: {dup_str}  (same bundle demanded again)")
    if unchanged_counts:
        unch_total = sum(unchanged_counts.values())
        if unch_total > 0:
            unch_str = "  ".join(
                f"{b}:{c}" for b, c in sorted(unchanged_counts.items()) if c > 0
            )
            lines.append(f"    consecutive same demand: {unch_str}")
    if total_useful_counterexamples is not None and useful_counterexample_count_by_bidder:
        uc_str = "  ".join(
            f"{b}:{c}"
            for b, c in sorted(useful_counterexample_count_by_bidder.items())
            if c > 0
        ) or "none"
        lines.append(
            f"    useful counterexamples: {total_useful_counterexamples}  by bidder: {uc_str}"
        )
    if total_no_new_information > 0 and no_new_information_count_by_bidder:
        nni_str = "  ".join(
            f"{b}:{c}"
            for b, c in sorted(no_new_information_count_by_bidder.items())
            if c > 0
        )
        lines.append(
            f"    no-new-info demands: {total_no_new_information}  by bidder: {nni_str}"
        )
    if atomic_trimming:
        lines.append(
            f"    atomic trimming: on  tolerance={trim_value_tolerance}"
        )
        lines.append(
            f"    trim attempts: {num_trim_attempts}"
            f"  reduced: {num_demands_trimmed_to_smaller_atom}"
            f"  net items removed: {total_net_items_removed}"
        )
        lines.append(
            f"    avg raw demand size: {avg_raw_demand_size:.2f}"
            f"  avg inserted atom size: {avg_inserted_atom_size:.2f}"
        )
        lines.append(
            f"    trim value queries: {ceca_total_trim_value_queries}"
        )
        if ceca_repeated_raw_demand_count > 0 or ceca_repeated_trimmed_atom_count > 0:
            lines.append(
                f"    repeated raw demands: {ceca_repeated_raw_demand_count}"
                f"  repeated trimmed atoms: {ceca_repeated_trimmed_atom_count}"
            )
        if trim_examples:
            lines.append("    trimming examples:")
            for ex in trim_examples[:5]:
                lines.append(
                    f"      {ex['bidder']}"
                    f"  {sorted(ex['raw_bundle'])} -> {sorted(ex['trimmed_bundle'])}"
                    f"  val={ex['raw_demanded_value']:.0f}"
                    f"  removed={ex['trim_items_removed']}"
                )
    else:
        lines.append("    atomic trimming: off")
    print("\n".join(lines), flush=True)


def _print_atom_insertion_log(
    atom_insertion_log: list[AtomInsertionRecord],
    bidder_ids: list[str],
) -> None:
    """Print per-bidder atom insertion log — new atoms, updates, and outside-interest flags."""
    by_bidder: dict[str, list[AtomInsertionRecord]] = defaultdict(list)
    for rec in atom_insertion_log:
        by_bidder[rec.bidder_id].append(rec)

    print("  CECA atom insertion log (new/updated atoms only):")
    for bidder_id in sorted(bidder_ids):
        recs = by_bidder.get(bidder_id, [])
        if not recs:
            print(f"    {bidder_id}: no insertions")
            continue
        n_new = sum(1 for r in recs if r.is_new)
        n_upd = sum(1 for r in recs if r.is_update)
        n_out_ins = sum(1 for r in recs if r.inserted_outside_interest)
        n_out_raw = sum(1 for r in recs if r.raw_outside_interest)
        flag = "  [OUTSIDE-INTEREST]" if n_out_ins > 0 else ""
        print(
            f"    {bidder_id:<22}  {n_new} new  {n_upd} upd"
            f"  raw_outside={n_out_raw}  ins_outside={n_out_ins}{flag}"
        )
        for r in recs:
            raw_str = "{" + ",".join(sorted(str(i) for i in r.raw_demanded_bundle)) + "}"
            ins_str = "{" + ",".join(sorted(str(i) for i in r.inserted_bundle)) + "}"
            kind = "NEW" if r.is_new else "UPD"
            out_flag = "  [outside-interest]" if r.inserted_outside_interest else ""
            print(
                f"      r{r.round_idx:02d}  {kind}"
                f"  raw={raw_str} -> ins={ins_str}"
                f"  val={r.inserted_value:.0f}{out_flag}"
            )
    print(flush=True)


def run_proxy_ceca_elicitation(
    instance: AuctionInstance,
    proxies: list[CecaAuctionProxy],
    ceca_config: CecaConfig,
    proxy_config: ProxyCecaConfig,
) -> ProxyCecaSharedResult:
    """Run the CECA elicitation phase without finalizing payments.

    The ``proxy_config.payment_rule`` field is **ignored** by this function
    -- finalization is deferred to :func:`finalize_proxy_ceca_result`.
    Call that function once per desired payment rule to obtain
    ``MechanismResult`` objects that share the same CECA state.
    """
    proxies_by_bidder = {proxy.bidder_id: proxy for proxy in proxies}
    validate_bidder_keys(
        bidder_ids=instance.bidder_ids,
        values=proxies_by_bidder,
        label="proxies",
    )

    original_bids = {
        bidder_id: clone_xor_bid(proxies_by_bidder[bidder_id].current_bid())
        for bidder_id in instance.bidder_ids
    }
    initial_bids = {
        bidder_id: _filter_initial_bid(bid, proxy_config.initial_bid_mode)
        for bidder_id, bid in original_bids.items()
    }
    initial_manifest_sizes = {
        bidder_id: len(bid.atoms) for bidder_id, bid in initial_bids.items()
    }

    _print_initial_manifest_summary(proxy_config.initial_bid_mode, initial_bids, original_bids)

    if proxy_config.initial_bid_mode in ("singletons", "empty"):
        empty_bidders = [b for b, bid in initial_bids.items() if not bid.atoms]
        if empty_bidders and proxy_config.initial_bid_mode == "singletons":
            print(
                f"  NOTE: no singleton atoms in proxy bid for: "
                f"{', '.join(sorted(empty_bidders))}; "
                "CECA starts these bidders from an empty manifest.",
                flush=True,
            )

    # Configure atomic trimming on each proxy that supports it.
    for proxy in proxies:
        inner = getattr(proxy, 'proxy', proxy)  # unwrap LlmAuctionProxyAdapter → LlmInferredXorProxy
        if hasattr(inner, 'ceca_atomic_trimming'):
            inner.ceca_atomic_trimming = proxy_config.atomic_trimming
            inner.ceca_trim_value_tolerance = proxy_config.trim_value_tolerance

    # Extract per-bidder interest-map items for outside-interest detection.
    interest_items_by_bidder: dict[str, frozenset | None] = {}
    for bidder_id, proxy in proxies_by_bidder.items():
        inner = getattr(proxy, "proxy", proxy)
        im = getattr(inner, "interest_map", None)
        interest_items_by_bidder[bidder_id] = getattr(im, "interested_items", None)

    # Compute per-bidder allowed bundle universe before CECA starts.
    allowed_bundles_by_bidder = _compute_allowed_bundles_by_bidder(
        universe_mode=proxy_config.ceca_demand_universe,
        original_bids=original_bids,
        interest_items_by_bidder=interest_items_by_bidder,
        max_bundle_size=proxy_config.max_bundle_size,
    )

    demand_records: list[_DemandRecord] = []
    atom_insertion_log: list[AtomInsertionRecord] = []
    demand_trace: list[DemandTraceRecord] = []
    exhaustion_events: list[dict] = []
    oracle = _make_tracked_oracle(
        proxies_by_bidder,
        demand_records,
        initial_bids,
        proxy_config.trim_value_tolerance,
        proxy_config=proxy_config,
        interest_items_by_bidder=interest_items_by_bidder,
        atom_insertion_log=atom_insertion_log,
        demand_trace=demand_trace,
        exhaustion_events=exhaustion_events,
        universe_mode=proxy_config.ceca_demand_universe,
        allowed_bundles_by_bidder=allowed_bundles_by_bidder,
        original_bids_snap=original_bids,
    )

    effective_ceca_cfg = CecaConfig(
        max_rounds=ceca_config.max_rounds,
        stop_on_no_new_information=proxy_config.stop_on_no_new_information,
        stall_patience=proxy_config.stall_patience,
        stop_on_round_no_useful_counterexamples=(
            proxy_config.stop_on_round_no_useful_counterexamples
        ),
    )

    state = run_ceca(
        items=instance.items,
        bidder_ids=instance.bidder_ids,
        ceca_step_oracle=oracle,
        cfg=effective_ceca_cfg,
        initial_manifest_bids=initial_bids,
    )

    demand_query_count_by_bidder = {
        bidder_id: proxies_by_bidder[bidder_id].stats().demand_queries
        for bidder_id in instance.bidder_ids
    }
    pruning_query_count_by_bidder = {
        bidder_id: _pruning_query_count(proxies_by_bidder[bidder_id])
        for bidder_id in instance.bidder_ids
    }
    final_manifest_sizes = {
        bidder_id: len(bid.atoms) for bidder_id, bid in state.manifest_bids.items()
    }
    manifest_growth_by_bidder = {
        b: final_manifest_sizes[b] - initial_manifest_sizes.get(b, 0)
        for b in instance.bidder_ids
    }
    manifest_growth_total = sum(manifest_growth_by_bidder.values())

    # Duplicate-demand diagnostics.
    _demanded_per_bidder: dict[str, list[Bundle]] = defaultdict(list)
    for rec in demand_records:
        if not rec.satisfied and rec.demanded_bundle is not None:
            _demanded_per_bidder[rec.bidder_id].append(rec.demanded_bundle)

    demanded_bundle_count_by_bidder: dict[str, int] = {b: 0 for b in instance.bidder_ids}
    unique_demanded_bundle_count_by_bidder: dict[str, int] = {b: 0 for b in instance.bidder_ids}
    duplicate_demand_count_by_bidder: dict[str, int] = {b: 0 for b in instance.bidder_ids}
    unchanged_demand_count_by_bidder: dict[str, int] = {b: 0 for b in instance.bidder_ids}

    for b, demanded_list in _demanded_per_bidder.items():
        n = len(demanded_list)
        u = len(set(demanded_list))
        demanded_bundle_count_by_bidder[b] = n
        unique_demanded_bundle_count_by_bidder[b] = u
        duplicate_demand_count_by_bidder[b] = n - u
        unchanged_demand_count_by_bidder[b] = sum(
            1 for i in range(1, n) if demanded_list[i] == demanded_list[i - 1]
        )

    # Aggregate atomic trim diagnostics from demand records.
    trim_records = [r for r in demand_records if r.trim_result is not None]
    raw_demand_sizes = [len(r.demanded_bundle) for r in demand_records if r.demanded_bundle]
    trimmed_atom_sizes = [
        len(r.trim_result.trimmed_bundle) for r in trim_records
    ]
    ceca_trimmed_demand_count = len(trim_records)
    ceca_total_trim_items_removed = sum(r.trim_result.trim_items_removed for r in trim_records)
    ceca_total_trim_value_queries = sum(r.trim_result.trim_value_queries for r in trim_records)
    ceca_avg_raw_demand_size = (
        sum(raw_demand_sizes) / len(raw_demand_sizes) if raw_demand_sizes else 0.0
    )
    ceca_avg_trimmed_atom_size = (
        sum(trimmed_atom_sizes) / len(trimmed_atom_sizes) if trimmed_atom_sizes else 0.0
    )
    # Repeated raw demands: same bidder demanded same raw bundle more than once.
    raw_demands_by_bidder: dict[str, list] = defaultdict(list)
    for r in demand_records:
        if r.demanded_bundle is not None:
            raw_bundle = r.trim_result.raw_bundle if r.trim_result else r.demanded_bundle
            raw_demands_by_bidder[r.bidder_id].append(raw_bundle)
    ceca_repeated_raw_demand_count = sum(
        len(bundles) - len(set(bundles))
        for bundles in raw_demands_by_bidder.values()
    )
    trimmed_demands_by_bidder: dict[str, list] = defaultdict(list)
    for r in trim_records:
        trimmed_demands_by_bidder[r.bidder_id].append(r.trim_result.trimmed_bundle)
    ceca_repeated_trimmed_atom_count = sum(
        len(bundles) - len(set(bundles))
        for bundles in trimmed_demands_by_bidder.values()
    )

    # Feature 1: corrected trim diagnostics — computed over trim_records only.
    num_trim_attempts = len(trim_records)
    num_demands_trimmed_to_smaller_atom = sum(
        1 for r in trim_records if r.trim_result.trim_items_removed > 0
    )
    total_raw_demand_items = sum(
        len(r.trim_result.raw_bundle) for r in trim_records
    )
    total_inserted_atom_items = sum(
        len(r.trim_result.trimmed_bundle) for r in trim_records
    )
    total_net_items_removed = total_raw_demand_items - total_inserted_atom_items
    avg_raw_demand_size = (
        total_raw_demand_items / num_trim_attempts if num_trim_attempts > 0 else 0.0
    )
    avg_inserted_atom_size = (
        total_inserted_atom_items / num_trim_attempts if num_trim_attempts > 0 else 0.0
    )

    # Trim examples: up to 5 records where items were actually removed.
    trim_examples = [
        {
            "bidder": r.bidder_id,
            "raw_bundle": r.trim_result.raw_bundle,
            "trimmed_bundle": r.trim_result.trimmed_bundle,
            "raw_demanded_value": r.trim_result.raw_demanded_value,
            "trim_items_removed": r.trim_result.trim_items_removed,
        }
        for r in trim_records
        if r.trim_result.trim_items_removed > 0
    ]

    # Feature 2: no-new-information diagnostics.
    no_new_info_records = [r for r in demand_records if r.no_new_information]
    no_new_information_count_by_bidder: dict[str, int] = {
        b: 0 for b in instance.bidder_ids
    }
    for r in no_new_info_records:
        no_new_information_count_by_bidder[r.bidder_id] += 1
    repeated_trimmed_atom_count_by_bidder = dict(no_new_information_count_by_bidder)
    total_no_new_information = len(no_new_info_records)

    # Useful counterexamples = demanded - no_new_information.
    useful_counterexample_count_by_bidder: dict[str, int] = {
        b: demanded_bundle_count_by_bidder.get(b, 0) - no_new_information_count_by_bidder.get(b, 0)
        for b in instance.bidder_ids
    }
    total_useful_counterexamples = sum(useful_counterexample_count_by_bidder.values())

    # Task 1/2: outside-interest insertion counts.
    outside_interest_insertion_count_by_bidder: dict[str, int] = {
        b: 0 for b in instance.bidder_ids
    }
    for rec in atom_insertion_log:
        if rec.inserted_outside_interest:
            outside_interest_insertion_count_by_bidder[rec.bidder_id] += 1

    # Task 4: per-round no-info stats.
    no_info_bidder_count_by_round: dict[int, int] = {}
    for r in demand_records:
        if r.no_new_information:
            no_info_bidder_count_by_round[r.round_idx] = (
                no_info_bidder_count_by_round.get(r.round_idx, 0) + 1
            )
    rounds_with_insertions = {r.round_idx for r in atom_insertion_log}
    rounds_with_unsatisfied = {r.round_idx for r in demand_records if not r.satisfied}
    stall_round_set = rounds_with_unsatisfied - rounds_with_insertions
    no_info_round_count = len(stall_round_set)

    # bidders_exhausted_by_repetition: all their unsatisfied demands were no-new-info.
    bidders_exhausted_by_repetition: list[str] = []
    for b in instance.bidder_ids:
        b_unsatisfied = [r for r in demand_records if r.bidder_id == b and not r.satisfied]
        if b_unsatisfied and all(r.no_new_information for r in b_unsatisfied):
            bidders_exhausted_by_repetition.append(b)

    # Demand universe stats.
    out_of_universe_demand_count = sum(1 for r in demand_records if r.out_of_universe)
    rejected_out_of_universe_count = sum(
        1 for r in demand_records if r.out_of_universe and not r.projected
    )
    projected_demand_count = sum(1 for r in demand_records if r.projected)
    allowed_bundle_count_by_bidder: dict[str, int] = {
        b: len(allowed_bundles_by_bidder[b]) if allowed_bundles_by_bidder.get(b) is not None else -1
        for b in instance.bidder_ids
    }

    _print_ceca_summary(
        mode=proxy_config.initial_bid_mode,
        rounds=len(state.history),
        converged=state.converged,
        initial_sizes=initial_manifest_sizes,
        final_sizes=final_manifest_sizes,
        demanded_counts=demanded_bundle_count_by_bidder,
        duplicate_counts=duplicate_demand_count_by_bidder,
        unchanged_counts=unchanged_demand_count_by_bidder,
        atomic_trimming=proxy_config.atomic_trimming,
        trim_value_tolerance=proxy_config.trim_value_tolerance,
        ceca_total_trim_value_queries=ceca_total_trim_value_queries,
        ceca_total_trim_items_removed=ceca_total_trim_items_removed,
        ceca_avg_raw_demand_size=ceca_avg_raw_demand_size,
        ceca_avg_trimmed_atom_size=ceca_avg_trimmed_atom_size,
        ceca_repeated_raw_demand_count=ceca_repeated_raw_demand_count,
        ceca_repeated_trimmed_atom_count=ceca_repeated_trimmed_atom_count,
        num_trim_attempts=num_trim_attempts,
        num_demands_trimmed_to_smaller_atom=num_demands_trimmed_to_smaller_atom,
        total_raw_demand_items=total_raw_demand_items,
        total_inserted_atom_items=total_inserted_atom_items,
        total_net_items_removed=total_net_items_removed,
        avg_raw_demand_size=avg_raw_demand_size,
        avg_inserted_atom_size=avg_inserted_atom_size,
        trim_examples=trim_examples,
        total_no_new_information=total_no_new_information,
        no_new_information_count_by_bidder=no_new_information_count_by_bidder,
        total_useful_counterexamples=total_useful_counterexamples,
        useful_counterexample_count_by_bidder=useful_counterexample_count_by_bidder,
        stopped_reason=state.stopped_reason,
        stall_patience=proxy_config.stall_patience,
        max_rounds=ceca_config.max_rounds,
    )

    if atom_insertion_log:
        _print_atom_insertion_log(atom_insertion_log, instance.bidder_ids)

    if proxy_config.ceca_demand_universe != "all_items" and out_of_universe_demand_count > 0:
        print(
            f"  CECA demand universe: {proxy_config.ceca_demand_universe}"
            f"  out_of_universe={out_of_universe_demand_count}"
            f"  rejected={rejected_out_of_universe_count}"
            f"  projected={projected_demand_count}",
            flush=True,
        )
        allowed_str = "  ".join(
            f"{b}:{n}" if n >= 0 else f"{b}:all"
            for b, n in sorted(allowed_bundle_count_by_bidder.items())
        )
        print(f"    allowed bundles by bidder: {allowed_str}", flush=True)

    final_stage1, final_stage2 = finalize_ceca(instance.items, state.manifest_bids)

    return ProxyCecaSharedResult(
        ceca_state=state,
        initial_bids=initial_bids,
        initial_manifest_sizes=initial_manifest_sizes,
        final_manifest_sizes=final_manifest_sizes,
        manifest_growth_by_bidder=manifest_growth_by_bidder,
        manifest_growth_total=manifest_growth_total,
        demand_records=demand_records,
        demand_query_count_by_bidder=demand_query_count_by_bidder,
        pruning_query_count_by_bidder=pruning_query_count_by_bidder,
        demanded_bundle_count_by_bidder=demanded_bundle_count_by_bidder,
        unique_demanded_bundle_count_by_bidder=unique_demanded_bundle_count_by_bidder,
        duplicate_demand_count_by_bidder=duplicate_demand_count_by_bidder,
        unchanged_demand_count_by_bidder=unchanged_demand_count_by_bidder,
        proxy_architecture=_proxy_architecture(proxies[0]) if proxies else "unknown",
        ceca_variant=proxy_config.ceca_variant,
        ceca_initial_bid_mode=proxy_config.initial_bid_mode,
        ceca_atomic_trimming=proxy_config.atomic_trimming,
        ceca_trim_value_tolerance=proxy_config.trim_value_tolerance,
        ceca_trimmed_demand_count=ceca_trimmed_demand_count,
        ceca_total_trim_items_removed=ceca_total_trim_items_removed,
        ceca_total_trim_value_queries=ceca_total_trim_value_queries,
        ceca_avg_raw_demand_size=ceca_avg_raw_demand_size,
        ceca_avg_trimmed_atom_size=ceca_avg_trimmed_atom_size,
        ceca_repeated_raw_demand_count=ceca_repeated_raw_demand_count,
        ceca_repeated_trimmed_atom_count=ceca_repeated_trimmed_atom_count,
        num_trim_attempts=num_trim_attempts,
        num_demands_trimmed_to_smaller_atom=num_demands_trimmed_to_smaller_atom,
        total_raw_demand_items=total_raw_demand_items,
        total_inserted_atom_items=total_inserted_atom_items,
        total_net_items_removed=total_net_items_removed,
        avg_raw_demand_size=avg_raw_demand_size,
        avg_inserted_atom_size=avg_inserted_atom_size,
        no_new_information_count_by_bidder=no_new_information_count_by_bidder,
        repeated_trimmed_atom_count_by_bidder=repeated_trimmed_atom_count_by_bidder,
        total_no_new_information=total_no_new_information,
        useful_counterexample_count_by_bidder=useful_counterexample_count_by_bidder,
        total_useful_counterexamples=total_useful_counterexamples,
        stopped_reason=state.stopped_reason,
        atom_insertion_log=atom_insertion_log,
        outside_interest_insertion_count_by_bidder=outside_interest_insertion_count_by_bidder,
        demand_trace=demand_trace,
        no_info_round_count=no_info_round_count,
        no_info_bidder_count_by_round=dict(no_info_bidder_count_by_round),
        bidders_exhausted_by_repetition=bidders_exhausted_by_repetition,
        exhaustion_events=exhaustion_events,
        ceca_demand_universe=proxy_config.ceca_demand_universe,
        out_of_universe_demand_count=out_of_universe_demand_count,
        rejected_out_of_universe_count=rejected_out_of_universe_count,
        projected_demand_count=projected_demand_count,
        allowed_bundle_count_by_bidder=allowed_bundle_count_by_bidder,
        final_stage1=final_stage1,
        final_stage2=final_stage2,
    )


def finalize_proxy_ceca_result(
    instance: AuctionInstance,
    shared: ProxyCecaSharedResult,
    payment_rule: str,
) -> MechanismResult:
    """Apply ``payment_rule`` to a completed CECA shared state.

    Call this once per desired payment rule after
    :func:`run_proxy_ceca_elicitation`. Both ``"pay_as_bid"`` and
    ``"vcg"`` results will have identical allocation, welfare, rounds,
    and manifest metrics; they differ only in ``payments`` / ``revenue``.
    """
    if payment_rule not in _VALID_PAYMENT_RULES:
        raise ValueError(
            f"payment_rule must be one of {sorted(_VALID_PAYMENT_RULES)}, "
            f"got {payment_rule!r}"
        )

    state = shared.ceca_state
    stage2 = shared.final_stage2

    # Reuse the pre-computed allocation so PAB and VCG always report the same
    # allocation/welfare regardless of ILP tie-breaking.
    if payment_rule == "pay_as_bid":
        payments = {
            bidder_id: bid.value_of(stage2.allocation.get(bidder_id, frozenset()))
            for bidder_id, bid in state.manifest_bids.items()
        }
    else:
        payments = compute_vcg_payments(
            instance.items, list(state.manifest_bids.values()), full=stage2
        )

    return MechanismResult(
        mechanism=f"{MECHANISM_NAME}_{payment_rule}",
        allocation=stage2.allocation,
        welfare=stage2.welfare,
        payments=payments,
        revenue=sum(payments.values()),
        rounds=len(state.history),
        query_count=(
            sum(shared.demand_query_count_by_bidder.values())
            + sum(shared.pruning_query_count_by_bidder.values())
        ),
        metadata={
            "payment_rule": payment_rule,
            "ceca_initial_bid_mode": shared.ceca_initial_bid_mode,
            "ceca_variant": shared.ceca_variant,
            "ceca_internal_price_rule": "lindahl_manifest",
            "converged": state.converged,
            "ceca_rounds": len(state.history),
            "demand_query_count_by_bidder": shared.demand_query_count_by_bidder,
            "pruning_query_count_by_bidder": shared.pruning_query_count_by_bidder,
            "initial_bids": shared.initial_bids,
            "final_bids": dict(state.manifest_bids),
            "stage1_welfare": shared.final_stage1.welfare,
            "stage2_welfare": stage2.welfare,
            "final_manifest_sizes": shared.final_manifest_sizes,
            "final_manifest_total_atoms": sum(shared.final_manifest_sizes.values()),
            "initial_manifest_sizes": shared.initial_manifest_sizes,
            "initial_manifest_total_atoms": sum(shared.initial_manifest_sizes.values()),
            "manifest_growth_by_bidder": shared.manifest_growth_by_bidder,
            "manifest_growth_total": shared.manifest_growth_total,
            "demanded_bundle_count_by_bidder": shared.demanded_bundle_count_by_bidder,
            "unique_demanded_bundle_count_by_bidder": shared.unique_demanded_bundle_count_by_bidder,
            "duplicate_demand_count_by_bidder": shared.duplicate_demand_count_by_bidder,
            "unchanged_demand_count_by_bidder": shared.unchanged_demand_count_by_bidder,
            "proxy_architecture": shared.proxy_architecture,
            "ceca_atomic_trimming": shared.ceca_atomic_trimming,
            "ceca_trim_value_tolerance": shared.ceca_trim_value_tolerance,
            "ceca_trimmed_demand_count": shared.ceca_trimmed_demand_count,
            "ceca_total_trim_items_removed": shared.ceca_total_trim_items_removed,
            "ceca_total_trim_value_queries": shared.ceca_total_trim_value_queries,
            "ceca_avg_raw_demand_size": shared.ceca_avg_raw_demand_size,
            "ceca_avg_trimmed_atom_size": shared.ceca_avg_trimmed_atom_size,
            "ceca_repeated_raw_demand_count": shared.ceca_repeated_raw_demand_count,
            "ceca_repeated_trimmed_atom_count": shared.ceca_repeated_trimmed_atom_count,
            # Feature 1: corrected trim diagnostics.
            "num_trim_attempts": shared.num_trim_attempts,
            "num_demands_trimmed_to_smaller_atom": shared.num_demands_trimmed_to_smaller_atom,
            "total_raw_demand_items": shared.total_raw_demand_items,
            "total_inserted_atom_items": shared.total_inserted_atom_items,
            "total_net_items_removed": shared.total_net_items_removed,
            "avg_raw_demand_size": shared.avg_raw_demand_size,
            "avg_inserted_atom_size": shared.avg_inserted_atom_size,
            # Feature 2: no-new-information diagnostics.
            "no_new_information_count_by_bidder": shared.no_new_information_count_by_bidder,
            "total_no_new_information": shared.total_no_new_information,
            "useful_counterexample_count_by_bidder": shared.useful_counterexample_count_by_bidder,
            "total_useful_counterexamples": shared.total_useful_counterexamples,
            # Feature 3: stopped reason.
            "stopped_reason": shared.stopped_reason,
            # Tasks 1–2: atom insertion diagnostics.
            "atom_insertion_log": shared.atom_insertion_log,
            "outside_interest_insertion_count_by_bidder": shared.outside_interest_insertion_count_by_bidder,
            "outside_interest_insertion_total": sum(
                shared.outside_interest_insertion_count_by_bidder.values()
            ),
            # Task 3: demand trace.
            "demand_trace": shared.demand_trace,
            # Task 4: per-round no-info stats.
            "no_info_round_count": shared.no_info_round_count,
            "no_info_bidder_count_by_round": shared.no_info_bidder_count_by_round,
            "bidders_exhausted_by_repetition": shared.bidders_exhausted_by_repetition,
            # Task 5: bidder exhaustion events.
            "exhaustion_events": shared.exhaustion_events,
            # Demand universe constraint.
            "ceca_demand_universe": shared.ceca_demand_universe,
            "out_of_universe_demand_count": shared.out_of_universe_demand_count,
            "rejected_out_of_universe_count": shared.rejected_out_of_universe_count,
            "projected_demand_count": shared.projected_demand_count,
            "allowed_bundle_count_by_bidder": shared.allowed_bundle_count_by_bidder,
        },
    )


def ceca_satisfaction_diagnostic_rows(
    shared: ProxyCecaSharedResult,
) -> list[dict]:
    """Extract per-round per-bidder satisfaction diagnostics as flat dicts.

    Each row corresponds to one (round_idx, bidder_id) pair for which a
    :class:`~auctionlab.auction_types.CecaBidderDiagnostic` was recorded.
    Rows are sorted by round_idx then bidder_id.

    Columns:
      round_idx, bidder_id,
      allocated_bundle, allocated_lindahl_price, allocated_manifest_value,
      allocated_utility,
      best_bundle, best_value, best_price, best_utility, utility_gap,
      satisfied, demand_produced_new_info
    """
    def _fmt_bundle(b) -> str:
        if b is None:
            return ""
        return "{" + ",".join(sorted(b)) + "}" if b else "∅"

    rows: list[dict] = []
    for record in shared.ceca_state.history:
        if record.diagnostics is None:
            continue
        for bidder_id in sorted(record.diagnostics):
            d: CecaBidderDiagnostic = record.diagnostics[bidder_id]
            rows.append({
                "round_idx": record.round_idx,
                "bidder_id": bidder_id,
                "allocated_bundle": _fmt_bundle(d.allocated_bundle),
                "allocated_lindahl_price": f"{d.allocated_lindahl_price:.4f}",
                "allocated_manifest_value": f"{d.allocated_manifest_value:.4f}",
                "allocated_utility": f"{d.allocated_utility:.4f}",
                "best_bundle": _fmt_bundle(d.best_bundle),
                "best_value": f"{d.best_value:.4f}" if d.best_value is not None else "",
                "best_price": f"{d.best_price:.4f}" if d.best_price is not None else "",
                "best_utility": f"{d.best_utility:.4f}" if d.best_utility is not None else "",
                "utility_gap": f"{d.utility_gap:.4f}" if d.utility_gap is not None else "",
                "satisfied": d.satisfied,
                "demand_produced_new_info": d.demand_produced_new_info,
            })
    return rows


def run_proxy_ceca_experiment(
    instance: AuctionInstance,
    proxies: list[CecaAuctionProxy],
    ceca_config: CecaConfig,
    proxy_config: ProxyCecaConfig,
) -> MechanismResult:
    """Run CECA elicitation and finalize with ``proxy_config.payment_rule``.

    Backward-compatible single-payment wrapper around the two-phase API.
    For multi-payment-rule comparisons, prefer calling
    :func:`run_proxy_ceca_elicitation` once and then
    :func:`finalize_proxy_ceca_result` once per payment rule so CECA runs
    exactly once and both results share the same manifest/allocation.
    """
    shared = run_proxy_ceca_elicitation(instance, proxies, ceca_config, proxy_config)
    return finalize_proxy_ceca_result(instance, shared, proxy_config.payment_rule)
