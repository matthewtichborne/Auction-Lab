"""Round-by-round diagnostic trajectory for the proxy-mediated clock.

Mirrors :mod:`auctionlab.experiments.proxy_sealed_runner`'s trajectory
recording for the sealed mechanism, but for the ascending clock. This module
adds *diagnostic logging only* on top of the existing clock machinery in
:mod:`auctionlab.auctions.clock` and :mod:`auctionlab.experiments.proxy_clock_runner`:
it does not change the price-update rule, the WDP solver, VCG payments, the
proxy refinement policy, event trigger thresholds, or top-k demand
semantics. The unmodified :func:`run_ascending_clock_with_supplementary` is
still the sole driver of the clock; this module only observes it via the
``on_bidder_round`` diagnostic hook in ``_make_demand_oracle`` and, at each
round boundary, additionally solves a "finalise-at-round" supplementary WDP
(the exact same :func:`finalize_from_supplementary_vcg` used for the real
final outcome, just run early against a snapshot of atoms accumulated so
far) to build an efficiency-over-time trajectory.

Round numbers in every row/column produced here are 1-based (round 1 is the
clock's first price round), matching the terminal round headers the clock
already prints (``round {round_idx + 1}``) and the worked examples in the
diagnostics spec this module implements.

TODO (explicitly deferred, not blocking): a proxy-vs-full-information clock
demand-path comparison (``curated_proxy_clock_vs_fullinfo_top_{k}.csv``) is
not implemented. Doing it well requires re-running (or replaying) a
full-information clock in lockstep with the proxy clock so per-round demand
sets, price paths, and overdemanded-goods sets can be diffed round-by-round.
Without it, the ``price_path_divergence`` failure-classification label is
also unreachable -- ``classify_clock_failure`` falls back to ``"unknown"``
for any failure that isn't an omitted bundle or a large value-calibration
error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from auctionlab.auction_types import Bundle, validate_bidder_keys
from auctionlab.auctions.clock import (
    ClockConfig,
    compute_excess_demand,
    finalize_from_supplementary_vcg,
    has_no_excess_demand,
    record_supplementary_bids,
    run_ascending_clock_with_supplementary,
)
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.experiments._trajectory_util import (
    aggregate_query_counts,
    logger_total_tokens,
    true_welfare_for_allocation,
)
from auctionlab.experiments.export import allocation_to_str
from auctionlab.experiments.proxy_clock_runner import (
    BidderRoundObservation,
    ProxyClockConfig,
    _ClockAuditState,
    _BidderElicitationState,
    _build_clock_mechanism_result,
    _make_demand_oracle,
    _terminal_stability_audit,
)
from auctionlab.experiments.runner import (
    MechanismResult,
    run_clock_supplementary_vcg_experiment,
    run_sealed_vcg_experiment,
)
from auctionlab.instances.base import AuctionInstance, bundle_price
from auctionlab.payments.vcg import vcg_witness_count
from auctionlab.proxies.base import ClockAuctionProxy, clone_xor_bid

# Possible values of `failure_classification` (see `classify_clock_failure`).
# Not every label is reachable by the current heuristic -- `price_path_divergence`
# needs the not-yet-implemented full-info demand-path comparison (see module
# docstring), and `supplementary_wdp_tie_or_payment_issue` is not distinguished
# by the heuristic requested for this pass.
FAILURE_LABELS = (
    "no_failure_global",
    "no_failure_mechanism_relative",
    "late_refinement_worsened_final_allocation",
    "hit_max_rounds_unconverged",
    "candidate_omission",
    "seen_but_not_refined",
    "refined_but_miscalibrated",
    "price_path_divergence",
    "supplementary_wdp_tie_or_payment_issue",
    "unknown",
)

# Efficiency-point gap (relative to the global full-info optimum) between the
# best true welfare seen at any round and the final round's true welfare,
# above which the final allocation is considered a late-refinement
# regression. Matches the 5-percentage-point threshold used for the
# best-vs-final console warning.
LATE_REGRESSION_EFFICIENCY_THRESHOLD = 0.05


def _bundle_repr(bundle: Bundle | None) -> str:
    if not bundle:
        return "∅"
    return "[" + ",".join(sorted(bundle)) + "]"


def _stats_snapshot(
    proxies_by_bidder: dict[str, ClockAuctionProxy],
) -> dict[str, dict[str, int]]:
    snapshot: dict[str, dict[str, int]] = {}
    for bidder_id, proxy in proxies_by_bidder.items():
        stats = proxy.stats()
        snapshot[bidder_id] = {
            "value_queries": stats.value_queries,
            "demand_queries": stats.demand_queries,
            "nl_queries": stats.nl_queries,
            "refinement_queries": stats.refinement_queries,
        }
    return snapshot


@dataclass(frozen=True)
class ProxyClockTrajectory:
    """Full diagnostic output of :func:`run_proxy_clock_trajectory`."""

    final_result: MechanismResult
    round_rows: list[dict[str, Any]]
    bidder_round_rows: list[dict[str, Any]]
    coverage_rows: list[dict[str, Any]]
    event_rows: list[dict[str, Any]]
    failure_classification: str
    best_final: dict[str, Any]


class _ClockTrajectoryRecorder:
    """Accumulates per-round/per-bidder diagnostics via ``on_bidder_round``.

    Round boundaries are detected by counting distinct bidders observed for
    the current round: ``run_ascending_clock_with_supplementary`` always
    calls the demand oracle once per bidder, in ``bidder_ids`` order, before
    advancing to the next round, so the round is complete exactly when every
    bidder has reported.
    """

    def __init__(
        self,
        *,
        instance: AuctionInstance,
        bidder_ids: list[str],
        proxies_by_bidder: dict[str, ClockAuctionProxy],
        proxy_config: ProxyClockConfig,
        clock_config: ClockConfig,
        logger: Any | None,
        full_info: MechanismResult,
        mechanism_full_info_welfare: float,
        scenario_name: str,
        scenario_seed: Any,
        num_goods: int,
        num_bidders: int,
    ) -> None:
        self.instance = instance
        self.bidder_ids = bidder_ids
        self.proxies_by_bidder = proxies_by_bidder
        self.proxy_config = proxy_config
        self.clock_config = clock_config
        self.logger = logger
        self.full_info = full_info
        self.mechanism_full_info_welfare = mechanism_full_info_welfare
        self._row_context = {
            "scenario": scenario_name,
            "scenario_seed": scenario_seed,
            "num_goods": num_goods,
            "num_bidders": num_bidders,
            "top_k": proxy_config.top_k,
        }

        self._pending: dict[str, BidderRoundObservation] = {}
        self.supplementary: dict[str, list[XorAtomicBid]] = {
            bidder_id: [] for bidder_id in bidder_ids
        }
        self._prev_query_stats = {
            bidder_id: {
                "value_queries": 0,
                "demand_queries": 0,
                "nl_queries": 0,
                "refinement_queries": 0,
            }
            for bidder_id in bidder_ids
        }
        self._prev_tok_in, self._prev_tok_out = logger_total_tokens(logger)
        self._cum_tok_in = 0
        self._cum_tok_out = 0
        self._prev_vq = 0
        self._prev_dq = 0
        self._prev_supp_total = 0
        # Static (computed once, not accumulated round-by-round): total size
        # of each proxy's *candidate* bundle universe, when the proxy type
        # exposes one (e.g. LlmAuctionProxyAdapter.candidate_bundles). This
        # is a different concept from `supplementary_atoms_total` (below):
        # candidate bundles are what the proxy is configured to consider at
        # all; supplementary atoms are what has actually been reported to
        # the mechanism's finalise-at-round WDP so far. With provisional
        # valuations enabled, a proxy's initial bid already has a value for
        # every candidate bundle, so `supplementary_atoms_total` legitimately
        # equals `candidate_atoms_total` from round 1 onward -- that is
        # correct accumulation behaviour, not a bug, but the two numbers
        # answer different questions, hence two separate columns.
        self.candidate_atoms_total = sum(
            len(getattr(proxy, "candidate_bundles", None) or [])
            for proxy in proxies_by_bidder.values()
        )
        self._prev_finalise_true_welfare: float | None = None
        self._prev_finalise_allocation: dict[str, Bundle] | None = None

        self.round_rows: list[dict[str, Any]] = []
        self.bidder_round_rows: list[dict[str, Any]] = []
        self.event_rows: list[dict[str, Any]] = []
        self._event_row_refs: list[tuple[str, Bundle | None]] = []

        # Coverage bookkeeping.
        self._fi_bundle_by_bidder = {
            bidder_id: full_info.allocation.get(bidder_id, frozenset())
            for bidder_id in bidder_ids
        }
        self._seen_bundles: dict[str, set[Bundle]] = {b: set() for b in bidder_ids}
        self._first_seen_round: dict[tuple[str, Bundle], int] = {}
        self._first_seen_as_primary_round: dict[tuple[str, Bundle], int] = {}
        self._reported_value_when_first_seen: dict[str, float] = {}

    # -- observation intake -------------------------------------------------

    def observe(self, obs: BidderRoundObservation) -> None:
        self._track_coverage(obs)
        for event in obs.fired_events:
            self._record_event_row(obs, event)

        self._pending[obs.bidder_id] = obs
        if len(self._pending) == len(self.bidder_ids):
            self._finalize_round()

    def _track_coverage(self, obs: BidderRoundObservation) -> None:
        response = obs.response
        top_k_bundles = response.primary_bundles or (
            [response.primary_bundle] if response.primary_bundle else []
        )
        for bundle in top_k_bundles:
            key = (obs.bidder_id, bundle)
            if key not in self._first_seen_round:
                self._first_seen_round[key] = obs.round_idx
            self._seen_bundles[obs.bidder_id].add(bundle)

        if response.primary_bundle is not None:
            key = (obs.bidder_id, response.primary_bundle)
            if key not in self._first_seen_as_primary_round:
                self._first_seen_as_primary_round[key] = obs.round_idx

        target = self._fi_bundle_by_bidder[obs.bidder_id]
        if target and obs.bidder_id not in self._reported_value_when_first_seen:
            if target in top_k_bundles:
                for atom in response.supplementary_atoms:
                    if atom.bundle == target:
                        self._reported_value_when_first_seen[obs.bidder_id] = atom.value
                        break

    def _record_event_row(self, obs: BidderRoundObservation, event) -> None:
        proxy = self.proxies_by_bidder[obs.bidder_id]
        match = None
        for rec in reversed(proxy.refinement_records()):
            if (
                rec.bundle == event.bundle
                and rec.round_idx == event.round_idx
                and rec.event_type == event.event_type
            ):
                match = rec
                break

        old_value = match.old_value if match is not None else None
        new_value = match.new_value if match is not None else None
        true_value = (
            self.instance.value_of(obs.bidder_id, event.bundle) if event.bundle else 0.0
        )
        signed_correction: float | str = ""
        if new_value is not None:
            signed_correction = new_value - (old_value if old_value is not None else 0.0)
        abs_correction = abs(signed_correction) if signed_correction != "" else ""

        price_of_bundle = bundle_price(event.bundle, obs.prices) if event.bundle else 0.0
        surplus_before: float | str = ""
        for atom, surplus in obs.ranked:
            if atom.bundle == event.bundle:
                surplus_before = surplus
                break
        surplus_after: float | str = (
            (new_value - price_of_bundle) if new_value is not None else ""
        )

        self.event_rows.append({
            **self._row_context,
            "round": obs.round_idx + 1,
            "bidder_id": obs.bidder_id,
            "event_type": event.event_type,
            "target_bundle": _bundle_repr(event.bundle),
            "old_reported_value": "" if old_value is None else old_value,
            "new_reported_value": "" if new_value is None else new_value,
            "true_value": true_value,
            "signed_correction": signed_correction,
            "abs_correction": abs_correction,
            "price_of_bundle": price_of_bundle,
            "surplus_before": surplus_before,
            "surplus_after": surplus_after,
            "reported_allocation_gap_before_query": (
                "" if event.allocation_gap is None else event.allocation_gap
            ),
            # Backfilled once the final/full-info allocations are known --
            # see `_backfill_event_allocation_columns`.
            "appears_in_final_allocation": None,
            "reported_vcg_counterfactual_count": None,
            "appears_in_reported_vcg_counterfactual": None,
            "appears_in_any_reported_vcg_witness": None,
            "appears_in_full_info_allocation": None,
            "full_info_vcg_counterfactual_count": None,
            "appears_in_full_info_vcg_counterfactual": None,
            "appears_in_any_full_info_vcg_witness": None,
        })
        self._event_row_refs.append((obs.bidder_id, event.bundle))

    # -- round boundary -------------------------------------------------

    def _finalize_round(self) -> None:
        observations = self._pending
        self._pending = {}
        round_idx = next(iter(observations.values())).round_idx

        supplementary_this_round = {
            bidder_id: [
                XorAtomicBid(bundle=frozenset(atom.bundle), value=atom.value)
                for atom in obs.response.supplementary_atoms
            ]
            for bidder_id, obs in observations.items()
        }
        record_supplementary_bids(
            SimpleNamespace(supplementary=self.supplementary),
            supplementary_this_round,
        )
        supp_total = sum(len(v) for v in self.supplementary.values())
        supp_new = supp_total - self._prev_supp_total
        self._prev_supp_total = supp_total

        primary_demands: dict[str, list[Bundle]] = {}
        for bidder_id, obs in observations.items():
            response = obs.response
            if response.primary_bundles:
                primary_demands[bidder_id] = [
                    frozenset(bundle) for bundle in response.primary_bundles
                ]
            elif response.primary_bundle is None:
                primary_demands[bidder_id] = []
            else:
                primary_demands[bidder_id] = [frozenset(response.primary_bundle)]

        excess = compute_excess_demand(self.instance.items, primary_demands)
        clock_terminated = has_no_excess_demand(excess)
        overdemanded_goods = sorted(item for item, v in excess.items() if v > 0)

        termination_reason = ""
        if clock_terminated:
            termination_reason = "no_excess_demand"
        elif round_idx == self.clock_config.max_rounds - 1:
            termination_reason = "max_rounds_reached"

        finalise_outcome = finalize_from_supplementary_vcg(
            items=self.instance.items,
            supplementary_atoms=self.supplementary,
        )
        finalise_true_welfare = true_welfare_for_allocation(
            self.instance, finalise_outcome.allocation
        )
        full_info_welfare = self.full_info.welfare
        finalise_global_efficiency = (
            finalise_true_welfare / full_info_welfare
            if full_info_welfare > 1e-9
            else 1.0
        )
        finalise_mechanism_relative_efficiency = (
            finalise_true_welfare / self.mechanism_full_info_welfare
            if self.mechanism_full_info_welfare > 1e-9
            else 1.0
        )
        finalise_revenue = sum(finalise_outcome.payments.values())
        finalise_true_surplus = finalise_true_welfare - finalise_revenue
        finalise_allocation_repr = allocation_to_str(finalise_outcome.allocation)

        allocation_changed = (
            False
            if self._prev_finalise_allocation is None
            else finalise_outcome.allocation != self._prev_finalise_allocation
        )
        welfare_delta = (
            0.0
            if self._prev_finalise_true_welfare is None
            else finalise_true_welfare - self._prev_finalise_true_welfare
        )
        self._prev_finalise_allocation = finalise_outcome.allocation
        self._prev_finalise_true_welfare = finalise_true_welfare

        stats_now = _stats_snapshot(self.proxies_by_bidder)
        new_by_bidder = {
            bidder_id: {
                key: stats_now[bidder_id][key] - self._prev_query_stats[bidder_id][key]
                for key in stats_now[bidder_id]
            }
            for bidder_id in self.bidder_ids
        }
        cum_value_queries, cum_demand_queries, _cum_nl_queries = (
            aggregate_query_counts(self.logger, stats_now)
        )
        new_value_queries = cum_value_queries - self._prev_vq
        new_demand_queries = cum_demand_queries - self._prev_dq
        self._prev_vq, self._prev_dq = cum_value_queries, cum_demand_queries

        tok_in_total, tok_out_total = logger_total_tokens(self.logger)
        new_tok_in = tok_in_total - self._prev_tok_in
        new_tok_out = tok_out_total - self._prev_tok_out
        self._cum_tok_in += new_tok_in
        self._cum_tok_out += new_tok_out
        self._prev_tok_in, self._prev_tok_out = tok_in_total, tok_out_total

        events_this_round = [e for obs in observations.values() for e in obs.fired_events]
        num_near_tie = sum(
            1 for e in events_this_round if e.event_type.endswith("near_tie")
        )
        num_near_zero = sum(
            1 for e in events_this_round if e.event_type == "near_zero_surplus"
        )
        num_demand_changed = sum(
            1
            for e in events_this_round
            if e.event_type.endswith("demand_changed")
        )
        num_top_k_frontier = sum(
            1
            for e in events_this_round
            if e.event_type in {
                "top_k_frontier",
                "allocation_pivotal_top_k_frontier",
            }
        )
        num_allocation_counterfactual = sum(
            1
            for e in events_this_round
            if e.event_type == "allocation_counterfactual"
        )
        num_allocation_pivotal = sum(
            1
            for e in events_this_round
            if e.event_type.startswith("allocation_pivotal_")
        )

        prices = next(iter(observations.values())).prices

        self.round_rows.append({
            **self._row_context,
            "round": round_idx + 1,
            "prices_json": json.dumps(
                {item: prices[item] for item in sorted(prices)}
            ),
            "overdemanded_goods_json": json.dumps(overdemanded_goods),
            "num_overdemanded_goods": len(overdemanded_goods),
            "price_step": self.clock_config.price_step,
            "clock_terminated": clock_terminated,
            "termination_reason": termination_reason,
            "num_events_this_round": len(events_this_round),
            "num_near_tie_events": num_near_tie,
            "num_near_zero_surplus_events": num_near_zero,
            "num_demand_changed_events": num_demand_changed,
            "num_top_k_frontier_events": num_top_k_frontier,
            "num_allocation_counterfactual_events": (
                num_allocation_counterfactual
            ),
            "num_allocation_pivotal_events": num_allocation_pivotal,
            "num_refinements_this_round": len(events_this_round),
            "terminal_stability_audit_queries": 0,
            "terminal_stability_audit_iterations": 0,
            "new_value_queries": new_value_queries,
            "cumulative_value_queries": cum_value_queries,
            "new_demand_queries": new_demand_queries,
            "cumulative_demand_queries": cum_demand_queries,
            "new_tokens_in": new_tok_in,
            "cumulative_tokens_in": self._cum_tok_in,
            "new_tokens_out": new_tok_out,
            "cumulative_tokens_out": self._cum_tok_out,
            "supplementary_atoms_total": supp_total,
            "supplementary_atoms_new_this_round": supp_new,
            "candidate_atoms_total": self.candidate_atoms_total,
            "finalise_true_welfare": finalise_true_welfare,
            "finalise_reported_welfare": finalise_outcome.welfare,
            "full_info_welfare": full_info_welfare,
            "finalise_global_efficiency": finalise_global_efficiency,
            "mechanism_full_info_welfare": self.mechanism_full_info_welfare,
            "finalise_mechanism_relative_efficiency": (
                finalise_mechanism_relative_efficiency
            ),
            "finalise_revenue": finalise_revenue,
            "finalise_true_surplus": finalise_true_surplus,
            "finalise_allocation_repr": finalise_allocation_repr,
            "allocation_changed_from_previous_round": allocation_changed,
            "welfare_delta_from_previous_round": welfare_delta,
        })

        for bidder_id, obs in observations.items():
            self._append_bidder_round_row(bidder_id, obs, stats_now, new_by_bidder)

        self._prev_query_stats = stats_now

    def _append_bidder_round_row(
        self,
        bidder_id: str,
        obs: BidderRoundObservation,
        stats_now: dict[str, dict[str, int]],
        new_by_bidder: dict[str, dict[str, int]],
    ) -> None:
        response = obs.response
        ranked = obs.ranked
        primary_bundle = response.primary_bundle
        top_k_bundles = response.primary_bundles or (
            [primary_bundle] if primary_bundle else []
        )

        primary_surplus: float | str = ranked[0][1] if ranked else ""
        runner_up_bundle = ranked[1][0].bundle if len(ranked) > 1 else None
        runner_up_surplus: float | str = ranked[1][1] if len(ranked) > 1 else ""
        surplus_gap: float | str = (
            (primary_surplus - runner_up_surplus)
            if (ranked and len(ranked) > 1)
            else ""
        )

        demand_changed = (
            obs.previous_primary_bundle is not None
            and obs.previous_primary_bundle != obs.current_primary_bundle
        )
        near_tie = any(c[1].endswith("near_tie") for c in obs.candidates)
        near_zero_surplus = any(c[1] == "near_zero_surplus" for c in obs.candidates)
        event_types = sorted({e.event_type for e in obs.fired_events})
        refined_bundles = [sorted(e.bundle) for e in obs.fired_events if e.bundle]

        primary_true_value: float | str = (
            self.instance.value_of(bidder_id, primary_bundle) if primary_bundle else ""
        )
        primary_reported_value: float | None = None
        if primary_bundle is not None:
            for atom, _surplus in ranked:
                if atom.bundle == primary_bundle:
                    primary_reported_value = atom.value
                    break
        primary_price: float | str = (
            bundle_price(primary_bundle, obs.prices) if primary_bundle else ""
        )
        primary_value_error: float | str = (
            (primary_reported_value - primary_true_value)
            if (primary_reported_value is not None and primary_bundle is not None)
            else ""
        )

        self.bidder_round_rows.append({
            **self._row_context,
            "round": obs.round_idx + 1,
            "bidder_id": bidder_id,
            "primary_demand_bundle": _bundle_repr(primary_bundle),
            "top_k_demand_bundles_json": json.dumps(
                [sorted(b) for b in top_k_bundles]
            ),
            "primary_surplus": primary_surplus,
            "runner_up_bundle": _bundle_repr(runner_up_bundle)
            if runner_up_bundle is not None
            else "",
            "runner_up_surplus": runner_up_surplus,
            "surplus_gap": surplus_gap,
            "demand_changed": demand_changed,
            "near_tie": near_tie,
            "near_zero_surplus": near_zero_surplus,
            "event_triggered": bool(obs.fired_events),
            "event_types_json": json.dumps(event_types),
            "refined_bundles_json": json.dumps(refined_bundles),
            "num_refinements_this_round": len(obs.fired_events),
            "new_value_queries": new_by_bidder[bidder_id]["value_queries"],
            "cumulative_value_queries_for_bidder": stats_now[bidder_id][
                "value_queries"
            ],
            "supplementary_atoms_for_bidder": len(self.supplementary[bidder_id]),
            "primary_demand_true_value": primary_true_value,
            "primary_demand_reported_value": (
                "" if primary_reported_value is None else primary_reported_value
            ),
            "primary_demand_price": primary_price,
            "primary_demand_value_error": primary_value_error,
        })

    # -- post-run -------------------------------------------------------

    def apply_terminal_audit(
        self,
        *,
        state,
        audit_state: _ClockAuditState,
    ) -> None:
        """Reconcile the last diagnostic row with post-clock stability queries."""
        if not self.round_rows or not audit_state.terminal_events:
            return

        self.supplementary = {
            bidder_id: list(atoms)
            for bidder_id, atoms in state.supplementary.items()
        }
        outcome = finalize_from_supplementary_vcg(
            items=self.instance.items,
            supplementary_atoms=self.supplementary,
        )
        true_welfare = true_welfare_for_allocation(
            self.instance, outcome.allocation
        )
        previous_allocation = (
            self._prev_finalise_allocation
            if self._prev_finalise_allocation is not None
            else {}
        )
        previous_welfare = self._prev_finalise_true_welfare
        row = self.round_rows[-1]

        stats_now = _stats_snapshot(self.proxies_by_bidder)
        cum_value_queries, cum_demand_queries, _ = aggregate_query_counts(
            self.logger, stats_now
        )
        extra_vq = cum_value_queries - self._prev_vq
        row["num_events_this_round"] += len(audit_state.terminal_events)
        row["num_refinements_this_round"] += len(audit_state.terminal_events)
        row["new_value_queries"] += extra_vq
        row["cumulative_value_queries"] = cum_value_queries
        row["cumulative_demand_queries"] = cum_demand_queries
        row["supplementary_atoms_total"] = sum(
            len(atoms) for atoms in self.supplementary.values()
        )
        row["finalise_true_welfare"] = true_welfare
        row["finalise_reported_welfare"] = outcome.welfare
        row["finalise_global_efficiency"] = (
            true_welfare / self.full_info.welfare
            if self.full_info.welfare > 1e-9
            else 1.0
        )
        row["finalise_mechanism_relative_efficiency"] = (
            true_welfare / self.mechanism_full_info_welfare
            if self.mechanism_full_info_welfare > 1e-9
            else 1.0
        )
        revenue = sum(outcome.payments.values())
        row["finalise_revenue"] = revenue
        row["finalise_true_surplus"] = true_welfare - revenue
        row["finalise_allocation_repr"] = allocation_to_str(
            outcome.allocation
        )
        row["allocation_changed_from_previous_round"] = (
            outcome.allocation != previous_allocation
        )
        row["welfare_delta_from_previous_round"] = (
            0.0
            if previous_welfare is None
            else true_welfare - previous_welfare
        )
        row["terminal_stability_audit_queries"] = audit_state.terminal_queries
        row["terminal_stability_audit_iterations"] = (
            audit_state.terminal_iterations
        )
        for event in audit_state.terminal_events:
            proxy = self.proxies_by_bidder[event.bidder_id]
            match = next(
                (
                    record
                    for record in reversed(proxy.refinement_records())
                    if record.bundle == event.bundle
                    and record.round_idx == event.round_idx
                    and record.event_type == event.event_type
                ),
                None,
            )
            old_value = match.old_value if match is not None else None
            new_value = match.new_value if match is not None else None
            true_value = self.instance.value_of(
                event.bidder_id, event.bundle
            )
            correction: float | str = (
                new_value - (old_value if old_value is not None else 0.0)
                if new_value is not None
                else ""
            )
            bundle_cost = bundle_price(event.bundle, state.prices)
            self.event_rows.append({
                **self._row_context,
                "round": len(state.history) + 1,
                "bidder_id": event.bidder_id,
                "event_type": event.event_type,
                "target_bundle": _bundle_repr(event.bundle),
                "old_reported_value": "" if old_value is None else old_value,
                "new_reported_value": "" if new_value is None else new_value,
                "true_value": true_value,
                "signed_correction": correction,
                "abs_correction": (
                    abs(correction) if correction != "" else ""
                ),
                "price_of_bundle": bundle_cost,
                "surplus_before": (
                    old_value - bundle_cost
                    if old_value is not None
                    else ""
                ),
                "surplus_after": (
                    new_value - bundle_cost
                    if new_value is not None
                    else ""
                ),
                "appears_in_final_allocation": None,
                "reported_vcg_counterfactual_count": None,
                "appears_in_reported_vcg_counterfactual": None,
                "appears_in_any_reported_vcg_witness": None,
                "appears_in_full_info_allocation": None,
                "full_info_vcg_counterfactual_count": None,
                "appears_in_full_info_vcg_counterfactual": None,
                "appears_in_any_full_info_vcg_witness": None,
            })
            self._event_row_refs.append((event.bidder_id, event.bundle))
        self._prev_vq = cum_value_queries
        self._prev_dq = cum_demand_queries
        self._prev_query_stats = stats_now
        self._prev_finalise_allocation = dict(outcome.allocation)
        self._prev_finalise_true_welfare = true_welfare

    def backfill_event_allocation_columns(
        self,
        *,
        final_allocation: dict[str, Bundle],
        full_info_allocation: dict[str, Bundle],
        reported_vcg_counterfactuals: dict[str, Any],
        full_info_vcg_counterfactuals: dict[str, Any],
    ) -> None:
        for row, (bidder_id, bundle) in zip(self.event_rows, self._event_row_refs):
            final_hit = bool(bundle) and (
                final_allocation.get(bidder_id, frozenset()) == bundle
            )
            reported_count = vcg_witness_count(
                reported_vcg_counterfactuals,
                bidder_id,
                bundle,
            )
            full_info_hit = bool(bundle) and (
                full_info_allocation.get(bidder_id, frozenset()) == bundle
            )
            full_info_count = vcg_witness_count(
                full_info_vcg_counterfactuals,
                bidder_id,
                bundle,
            )
            row["appears_in_final_allocation"] = final_hit
            row["reported_vcg_counterfactual_count"] = reported_count
            row["appears_in_reported_vcg_counterfactual"] = reported_count > 0
            row["appears_in_any_reported_vcg_witness"] = (
                final_hit or reported_count > 0
            )
            row["appears_in_full_info_allocation"] = full_info_hit
            row["full_info_vcg_counterfactual_count"] = full_info_count
            row["appears_in_full_info_vcg_counterfactual"] = (
                full_info_count > 0
            )
            row["appears_in_any_full_info_vcg_witness"] = (
                full_info_hit or full_info_count > 0
            )

    def build_coverage_rows(
        self,
        *,
        final_allocation: dict[str, Bundle],
        final_supplementary: dict[str, list[XorAtomicBid]],
        final_true_welfare: float,
        welfare_tolerance: float = 1e-6,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        full_info_welfare = self.full_info.welfare
        for bidder_id in self.bidder_ids:
            fi_bundle = self._fi_bundle_by_bidder[bidder_id]
            fi_value = (
                self.instance.value_of(bidder_id, fi_bundle) if fi_bundle else 0.0
            )
            seen = bool(fi_bundle) and fi_bundle in self._seen_bundles.get(
                bidder_id, set()
            )
            first_seen = self._first_seen_round.get((bidder_id, fi_bundle))
            first_seen_as_primary = self._first_seen_as_primary_round.get(
                (bidder_id, fi_bundle)
            )

            proxy = self.proxies_by_bidder[bidder_id]
            first_refined: int | None = None
            for rec in proxy.refinement_records():
                if rec.bundle == fi_bundle and rec.round_idx is not None:
                    if first_refined is None or rec.round_idx < first_refined:
                        first_refined = rec.round_idx

            final_atom = next(
                (
                    atom
                    for atom in final_supplementary.get(bidder_id, [])
                    if atom.bundle == fi_bundle
                ),
                None,
            )
            included_in_supp = bool(fi_bundle) and final_atom is not None
            final_reported_value = final_atom.value if final_atom is not None else None
            final_value_error = (
                (final_reported_value - fi_value)
                if final_reported_value is not None
                else ""
            )
            allocated_in_proxy_final = bool(fi_bundle) and (
                final_allocation.get(bidder_id, frozenset()) == fi_bundle
            )
            omitted_bundle_failure = (
                bool(fi_bundle)
                and not included_in_supp
                and final_true_welfare < full_info_welfare - welfare_tolerance
            )

            rows.append({
                **self._row_context,
                "bidder_id": bidder_id,
                "full_info_winning_bundle": _bundle_repr(fi_bundle),
                "full_info_winning_value": fi_value,
                "full_info_winning_bundle_nonempty": bool(fi_bundle),
                "seen_in_top_k_by_final": seen,
                "first_seen_round": "" if first_seen is None else first_seen + 1,
                "first_seen_as_primary_round": (
                    "" if first_seen_as_primary is None else first_seen_as_primary + 1
                ),
                "first_refined_round": (
                    "" if first_refined is None else first_refined + 1
                ),
                "included_in_supplementary_by_final": included_in_supp,
                "reported_value_when_first_seen": self._reported_value_when_first_seen.get(
                    bidder_id, ""
                ),
                "final_reported_value": (
                    "" if final_reported_value is None else final_reported_value
                ),
                "final_value_error": final_value_error,
                "allocated_in_proxy_final": allocated_in_proxy_final,
                "omitted_bundle_failure": omitted_bundle_failure,
            })
        return rows


def _compute_best_final_round_diagnostics(
    round_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the final round against the best true-welfare round seen.

    ``best`` is the earliest round achieving the maximum
    ``finalise_true_welfare`` seen anywhere in the trajectory (ties keep the
    earliest round, since that is the round the clock could have stopped at
    without losing welfare). Comparing it against the final round surfaces
    late-refinement regressions: cases where a correct reported-value
    update in a late round changes the finalise-at-round WDP's winner for
    the worse, rather than for the better.
    """
    final_row = round_rows[-1]
    best_row = max(round_rows, key=lambda row: row["finalise_true_welfare"])

    final_true_welfare = final_row["finalise_true_welfare"]
    best_true_welfare = best_row["finalise_true_welfare"]

    return {
        "final_round": final_row["round"],
        "termination_reason": final_row["termination_reason"],
        "final_true_welfare": final_true_welfare,
        "final_reported_welfare": final_row["finalise_reported_welfare"],
        "final_true_efficiency": final_row["finalise_global_efficiency"],
        "final_true_surplus": final_row["finalise_true_surplus"],
        "final_revenue": final_row["finalise_revenue"],
        "best_round": best_row["round"],
        "best_true_welfare": best_true_welfare,
        "best_true_efficiency": best_row["finalise_global_efficiency"],
        "best_reported_welfare_at_best_round": best_row["finalise_reported_welfare"],
        "best_true_surplus_at_best_round": best_row["finalise_true_surplus"],
        "cumulative_value_queries_at_best_round": best_row["cumulative_value_queries"],
        "final_round_events": final_row["num_events_this_round"],
        "final_round_demand_changed_events": final_row["num_demand_changed_events"],
        "final_round_refinements": final_row["num_refinements_this_round"],
        "final_welfare_minus_best_welfare": final_true_welfare - best_true_welfare,
    }


def classify_clock_failure(
    *,
    final_true_welfare: float,
    global_full_info_welfare: float,
    mechanism_full_info_welfare: float | None,
    best_true_welfare: float,
    termination_reason: str,
    coverage_rows: list[dict[str, Any]],
    welfare_tolerance: float = 1e-6,
    large_value_error_relative_threshold: float = 0.2,
    late_regression_efficiency_threshold: float = LATE_REGRESSION_EFFICIENCY_THRESHOLD,
) -> str:
    """Heuristic failure label for one proxy-clock run (see module docstring).

    Distinguishes *global* efficiency (relative to the sealed-VCG full-
    information optimum across all mechanisms) from *mechanism-relative*
    efficiency (relative to what an all-knowing bidder would achieve through
    this same clock mechanism, at the same top_k -- the clock's own
    ceiling, which can sit below the global optimum purely from clock/top-k
    structural limits that have nothing to do with proxy elicitation
    quality).

    1. Proxy reaches the global optimum -> ``no_failure_global``.
    2. The final round's true welfare is materially below the *best* true
       welfare seen at any earlier round (by more than
       ``late_regression_efficiency_threshold`` of the global optimum) ->
       ``late_refinement_worsened_final_allocation``. This is the pattern
       from the 8x8 live-run regression this label was added for: a
       same-round batch of ``demand_changed`` refinements corrected reported
       values late, changing the finalise-at-round WDP's winner for the
       worse (best-round efficiency 98.6%, final-round efficiency 74.7%).
    3. Proxy reaches the mechanism's own full-information benchmark (but not
       necessarily the global optimum) -> ``no_failure_mechanism_relative``.
    4. The clock never reached "no excess demand" and instead hit the round
       cap -> ``hit_max_rounds_unconverged``.
    5. Any non-empty FI-winning bundle missing from final supplementary
       atoms -> ``candidate_omission``.
    6. A FI-winning bundle was seen in top-k but never explicitly refined,
       and still has a large relative value error -> ``seen_but_not_refined``.
    7. All FI-winning bundles present (refined or not) but at least one has
       a large relative value error -> ``refined_but_miscalibrated``.
    8. ``price_path_divergence`` is not reachable yet (needs the
       full-info-vs-proxy demand-path comparison; see module TODO).
       Otherwise -> ``unknown``.
    """
    global_efficiency = (
        final_true_welfare / global_full_info_welfare
        if global_full_info_welfare > welfare_tolerance
        else 1.0
    )
    if final_true_welfare >= global_full_info_welfare - welfare_tolerance or (
        global_efficiency >= 1.0 - 1e-9
    ):
        return "no_failure_global"

    if global_full_info_welfare > welfare_tolerance:
        late_regression = (
            (best_true_welfare - final_true_welfare) / global_full_info_welfare
        )
    else:
        late_regression = 0.0
    if late_regression > late_regression_efficiency_threshold:
        return "late_refinement_worsened_final_allocation"

    if (
        mechanism_full_info_welfare is not None
        and final_true_welfare >= mechanism_full_info_welfare - welfare_tolerance
    ):
        return "no_failure_mechanism_relative"

    if termination_reason == "max_rounds_reached":
        return "hit_max_rounds_unconverged"

    for row in coverage_rows:
        if row["full_info_winning_bundle_nonempty"] and not row[
            "included_in_supplementary_by_final"
        ]:
            return "candidate_omission"

    def _large_error(row: dict[str, Any]) -> bool:
        if not row["full_info_winning_bundle_nonempty"]:
            return False
        fi_value = row["full_info_winning_value"]
        err = row["final_value_error"]
        return bool(
            fi_value > welfare_tolerance
            and err != ""
            and abs(err) / fi_value > large_value_error_relative_threshold
        )

    for row in coverage_rows:
        if row["seen_in_top_k_by_final"] and row["first_refined_round"] == "" and (
            _large_error(row)
        ):
            return "seen_but_not_refined"

    for row in coverage_rows:
        if _large_error(row):
            return "refined_but_miscalibrated"

    return "unknown"


def run_proxy_clock_trajectory(
    instance: AuctionInstance,
    proxies: list[ClockAuctionProxy],
    clock_config: ClockConfig,
    proxy_config: ProxyClockConfig,
    *,
    scenario_name: str = "unnamed_scenario",
    scenario_seed: Any = "",
    num_goods: int | None = None,
    num_bidders: int | None = None,
    logger: Any | None = None,
) -> ProxyClockTrajectory:
    """Run the proxy clock once, recording full round-by-round diagnostics.

    Runs the exact same clock (same price update rule, WDP solver, VCG
    payments, refinement policy, and event thresholds as
    :func:`~auctionlab.experiments.proxy_clock_runner.run_proxy_clock_experiment`)
    a single time; ``final_result`` is bit-for-bit what that function would
    return for the same inputs. Diagnostics are captured as a side effect via
    the ``on_bidder_round`` hook in ``_make_demand_oracle``.
    """
    proxies_by_bidder = {proxy.bidder_id: proxy for proxy in proxies}
    validate_bidder_keys(
        bidder_ids=instance.bidder_ids,
        values=proxies_by_bidder,
        label="proxies",
    )

    full_info = run_sealed_vcg_experiment(instance)
    # Ground-truth (no LLM/proxy involved) clock run at the same top_k and
    # ClockConfig: the mechanism's own achievable ceiling, which can sit
    # below the global sealed-VCG optimum purely from clock/top-k structural
    # limits. Used to tell "elicitation could be better" apart from
    # "this mechanism/top_k just can't reach the global optimum here" in
    # `classify_clock_failure`.
    mechanism_full_info = run_clock_supplementary_vcg_experiment(
        instance, clock_config, top_k=proxy_config.top_k
    )

    states = {
        bidder_id: _BidderElicitationState() for bidder_id in instance.bidder_ids
    }
    initial_bids: dict[str, XorBid] = {
        bidder_id: clone_xor_bid(proxies_by_bidder[bidder_id].current_bid())
        for bidder_id in instance.bidder_ids
    }

    recorder = _ClockTrajectoryRecorder(
        instance=instance,
        bidder_ids=instance.bidder_ids,
        proxies_by_bidder=proxies_by_bidder,
        proxy_config=proxy_config,
        clock_config=clock_config,
        logger=logger,
        full_info=full_info,
        mechanism_full_info_welfare=mechanism_full_info.welfare,
        scenario_name=scenario_name,
        scenario_seed=scenario_seed,
        num_goods=(
            num_goods if num_goods is not None else len(instance.items)
        ),
        num_bidders=(
            num_bidders if num_bidders is not None else len(instance.bidder_ids)
        ),
    )

    audit_state = _ClockAuditState()
    demand_oracle = _make_demand_oracle(
        proxies_by_bidder=proxies_by_bidder,
        states=states,
        proxy_config=proxy_config,
        on_bidder_round=recorder.observe,
        instance=instance,
        audit_state=audit_state,
    )

    state, _provisional = run_ascending_clock_with_supplementary(
        items=instance.items,
        bidder_ids=instance.bidder_ids,
        demand_oracle=demand_oracle,
        cfg=clock_config,
    )
    _terminal_stability_audit(
        instance=instance,
        proxies_by_bidder=proxies_by_bidder,
        states=states,
        proxy_config=proxy_config,
        state=state,
        audit_state=audit_state,
    )
    recorder.apply_terminal_audit(state=state, audit_state=audit_state)

    final_result = _build_clock_mechanism_result(
        instance=instance,
        proxies_by_bidder=proxies_by_bidder,
        proxy_config=proxy_config,
        state=state,
        initial_bids=initial_bids,
        audit_state=audit_state,
    )

    final_true_welfare = true_welfare_for_allocation(instance, final_result.allocation)

    recorder.backfill_event_allocation_columns(
        final_allocation=final_result.allocation,
        full_info_allocation=full_info.allocation,
        reported_vcg_counterfactuals=final_result.metadata[
            "vcg_counterfactuals"
        ],
        full_info_vcg_counterfactuals=full_info.metadata[
            "vcg_counterfactuals"
        ],
    )
    coverage_rows = recorder.build_coverage_rows(
        final_allocation=final_result.allocation,
        final_supplementary=state.supplementary,
        final_true_welfare=final_true_welfare,
    )
    best_final = _compute_best_final_round_diagnostics(recorder.round_rows)
    failure_classification = classify_clock_failure(
        final_true_welfare=final_true_welfare,
        global_full_info_welfare=full_info.welfare,
        mechanism_full_info_welfare=mechanism_full_info.welfare,
        best_true_welfare=best_final["best_true_welfare"],
        termination_reason=best_final["termination_reason"],
        coverage_rows=coverage_rows,
    )

    final_result = replace(
        final_result,
        metadata={
            **final_result.metadata,
            **best_final,
            "failure_classification": failure_classification,
            "full_info_welfare": full_info.welfare,
            "mechanism_full_info_welfare": mechanism_full_info.welfare,
            "final_true_welfare": final_true_welfare,
        },
    )

    return ProxyClockTrajectory(
        final_result=final_result,
        round_rows=recorder.round_rows,
        bidder_round_rows=recorder.bidder_round_rows,
        coverage_rows=coverage_rows,
        event_rows=recorder.event_rows,
        failure_classification=failure_classification,
        best_final=best_final,
    )
