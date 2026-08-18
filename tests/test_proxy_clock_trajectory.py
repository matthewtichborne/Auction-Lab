"""Round-level clock trajectory records.

Covers one row per round, cumulative query and atom counts that never
decrease, agreement between the final row and the run result, and that the
round cap behaves as a safeguard rather than a termination rule when the
market never clears.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from auctionlab.auctions.clock import ClockConfig
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.experiments.proxy_clock_runner import (
    ProxyClockConfig,
    run_proxy_clock_experiment,
)
from auctionlab.experiments.proxy_clock_trajectory import run_proxy_clock_trajectory
from auctionlab.instances.base import AuctionInstance, DemandResponse
from auctionlab.proxies.base import ProxyStats, RefinementRecord
from auctionlab.proxies.full_info import FullInfoAuctionProxy


def make_proxies(toy_instance, *, initial: str) -> list[FullInfoAuctionProxy]:
    return [
        FullInfoAuctionProxy(bidder_id=bidder_id, instance=toy_instance, initial=initial)
        for bidder_id in toy_instance.bidder_ids
    ]


@dataclass
class ScriptedRoundClockProxy:
    """A test double whose demand for a fixed round schedule is fully scripted.

    ``schedule`` maps round_idx -> (bundle, reported_value); ``default`` is
    used for any round_idx not in ``schedule``. A ``None`` bundle means "no
    demand this round" -- used to control exactly when a bundle first
    becomes visible to the clock, independent of surplus/price.

    ``true_values``, if given, makes ``refine`` a real (not no-op) update:
    refining a bundle present in ``true_values`` overrides its reported
    value for all later rounds (mirroring
    :class:`~auctionlab.proxies.full_info.FullInfoAuctionProxy`, which counts
    every value lookup -- including refinement-triggered ones -- as a value
    query). Bundles absent from ``true_values`` make ``refine`` a no-op, so
    existing tests that never set ``true_values`` are unaffected.
    """

    bidder_id: str
    schedule: dict[int, tuple[frozenset | None, float]] = field(default_factory=dict)
    default: tuple[frozenset | None, float] = (None, 0.0)
    true_values: dict[frozenset, float] = field(default_factory=dict)
    _stats: ProxyStats = field(default_factory=ProxyStats, init=False)
    _records: list = field(default_factory=list, init=False)
    _overrides: dict[frozenset, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_round_idx: int = field(default=-1, init=False, repr=False)

    def _scheduled(self, round_idx: int) -> tuple[frozenset | None, float]:
        bundle, value = self.schedule.get(round_idx, self.default)
        if bundle is not None and bundle in self._overrides:
            value = self._overrides[bundle]
        return bundle, value

    def current_bid(self) -> XorBid:
        bundle, value = self._scheduled(self._last_round_idx)
        if bundle is None:
            return XorBid(bidder_id=self.bidder_id, atoms=[])
        return XorBid(bidder_id=self.bidder_id, atoms=[XorAtomicBid(bundle=bundle, value=value)])

    def submit_bid(self) -> XorBid:
        return self.current_bid()

    def demand_at_prices(
        self, prices, round_idx: int = 0, top_k: int = 1
    ) -> DemandResponse:
        self._last_round_idx = round_idx
        self._stats.demand_queries += 1
        bundle, value = self._scheduled(round_idx)
        if bundle is None:
            return DemandResponse(
                primary_bundle=None, supplementary_atoms=[], primary_bundles=[]
            )
        atom = XorAtomicBid(bundle=bundle, value=value)
        return DemandResponse(
            primary_bundle=bundle,
            supplementary_atoms=[atom],
            primary_bundles=[bundle],
        )

    def refine(self, event) -> None:
        if event.bundle is None:
            return
        true_value = self.true_values.get(event.bundle)
        if true_value is None:
            return
        old_value = self._overrides.get(event.bundle)
        if old_value is None:
            _, old_value = self._scheduled(self._last_round_idx)
        self._stats.refinement_queries += 1
        self._stats.value_queries += 1
        self._overrides[event.bundle] = true_value
        self._records.append(
            RefinementRecord(
                bidder_id=self.bidder_id,
                mechanism=event.mechanism,
                event_type=event.event_type,
                round_idx=event.round_idx,
                bundle=event.bundle,
                old_value=old_value,
                new_value=true_value,
                reason=event.reason,
            )
        )

    def receive_provisional_feedback(self, event) -> None:
        self.refine(event)

    def stats(self) -> ProxyStats:
        return self._stats

    def refinement_records(self) -> list:
        return list(self._records)


# ---------------------------------------------------------------------------
# 1. Clock trajectory rows
# ---------------------------------------------------------------------------


def _tampered_full_info_proxies(toy_instance):
    """FullInfoAuctionProxy set for i1 tampered so refinement has work to do."""
    proxies = make_proxies(toy_instance, initial="all_atoms")
    proxy_by_bidder = {proxy.bidder_id: proxy for proxy in proxies}
    i1 = proxy_by_bidder["i1"]
    for idx, atom in enumerate(i1.current_bid().atoms):
        if atom.bundle == frozenset({"A", "B"}):
            i1.current_bid().atoms[idx] = XorAtomicBid(bundle=atom.bundle, value=999.0)
    return proxies


def test_trajectory_has_one_row_per_round(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(
        top_k=1,
        elicited=True,
        margin_threshold=10_000.0,
        tie_threshold=10_000.0,
        max_refinements_per_bidder=10,
    )

    trajectory = run_proxy_clock_trajectory(
        toy_instance,
        _tampered_full_info_proxies(toy_instance),
        cfg,
        proxy_cfg,
        scenario_name="toy",
        scenario_seed=0,
        num_goods=3,
        num_bidders=3,
    )

    assert trajectory.final_result.rounds == len(trajectory.round_rows)
    assert [row["round"] for row in trajectory.round_rows] == list(
        range(1, trajectory.final_result.rounds + 1)
    )
    # Every bidder reports every round.
    assert len(trajectory.bidder_round_rows) == 3 * len(trajectory.round_rows)


def test_trajectory_cumulative_queries_and_atoms_are_nondecreasing(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(
        top_k=1,
        elicited=True,
        margin_threshold=10_000.0,
        tie_threshold=10_000.0,
        max_refinements_per_bidder=10,
    )

    trajectory = run_proxy_clock_trajectory(
        toy_instance,
        _tampered_full_info_proxies(toy_instance),
        cfg,
        proxy_cfg,
        scenario_name="toy",
        scenario_seed=0,
        num_goods=3,
        num_bidders=3,
    )

    for key in (
        "cumulative_value_queries",
        "cumulative_demand_queries",
        "cumulative_tokens_in",
        "cumulative_tokens_out",
        "supplementary_atoms_total",
    ):
        values = [row[key] for row in trajectory.round_rows]
        assert values == sorted(values), f"{key} is not nondecreasing: {values}"

    # At least one refinement happened (i1's tampered {A,B} atom gets corrected).
    assert trajectory.round_rows[-1]["cumulative_value_queries"] > 0


def test_trajectory_final_row_matches_existing_proxy_clock_result(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(
        top_k=1,
        elicited=True,
        margin_threshold=10_000.0,
        tie_threshold=10_000.0,
        max_refinements_per_bidder=10,
    )

    trajectory = run_proxy_clock_trajectory(
        toy_instance,
        _tampered_full_info_proxies(toy_instance),
        cfg,
        proxy_cfg,
        scenario_name="toy",
        scenario_seed=0,
        num_goods=3,
        num_bidders=3,
    )

    classic_result = run_proxy_clock_experiment(
        toy_instance,
        _tampered_full_info_proxies(toy_instance),
        cfg,
        proxy_cfg,
    )

    assert trajectory.final_result.allocation == classic_result.allocation
    assert trajectory.final_result.welfare == classic_result.welfare
    assert trajectory.final_result.payments == classic_result.payments
    assert trajectory.final_result.rounds == classic_result.rounds

    final_round_row = trajectory.round_rows[-1]
    assert final_round_row["finalise_reported_welfare"] == classic_result.welfare
    assert (
        final_round_row["finalise_allocation_repr"]
        == trajectory.round_rows[-1]["finalise_allocation_repr"]
    )
    # The last round's finalise allocation is the final allocation.
    from auctionlab.experiments.export import allocation_to_str

    assert final_round_row["finalise_allocation_repr"] == allocation_to_str(
        classic_result.allocation
    )


# ---------------------------------------------------------------------------
# 1b. --max-rounds propagation
# ---------------------------------------------------------------------------


def test_proxy_clock_honours_max_rounds_beyond_twenty_when_never_clearing():
    """Two bidders permanently overdemand the same item, so the clock can
    only ever stop via the round cap. With ``max_rounds=30`` it must run
    past round 20, not silently stop there.
    """
    instance = AuctionInstance(
        items=["X"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"X"}): 1000.0},
            "i2": {frozenset({"X"}): 1000.0},
        },
    )
    i1 = ScriptedRoundClockProxy(bidder_id="i1", default=(frozenset({"X"}), 1000.0))
    i2 = ScriptedRoundClockProxy(bidder_id="i2", default=(frozenset({"X"}), 1000.0))

    cfg = ClockConfig(max_rounds=30, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(top_k=1, elicited=False)

    trajectory = run_proxy_clock_trajectory(
        instance,
        [i1, i2],
        cfg,
        proxy_cfg,
        scenario_name="never_clears",
        scenario_seed=0,
        num_goods=1,
        num_bidders=2,
    )

    assert len(trajectory.round_rows) > 20
    assert trajectory.final_result.rounds == 30
    assert trajectory.round_rows[-1]["termination_reason"] == "max_rounds_reached"


def test_cli_max_rounds_flag_reaches_clock_config(monkeypatch):
    """``--max-rounds`` must reach the ``ClockConfig`` used for the proxy
    clock, exactly as ``main()`` builds it (``examples/run_live_llm_curated_batch.py``):
    ``ClockConfig(max_rounds=args.max_rounds, price_step=args.price_step,
    reserve=args.reserve)``.
    """
    import sys

    from examples.run_live_llm_curated_batch import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live_llm_curated_batch.py", "--max-rounds", "40"],
    )
    args = parse_args()
    assert args.max_rounds == 40

    cfg = ClockConfig(
        max_rounds=args.max_rounds,
        price_step=args.price_step,
        reserve=args.reserve,
    )
    assert cfg.max_rounds == 40

    instance = AuctionInstance(
        items=["X"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"X"}): 1000.0},
            "i2": {frozenset({"X"}): 1000.0},
        },
    )
    i1 = ScriptedRoundClockProxy(bidder_id="i1", default=(frozenset({"X"}), 1000.0))
    i2 = ScriptedRoundClockProxy(bidder_id="i2", default=(frozenset({"X"}), 1000.0))
    proxy_cfg = ProxyClockConfig(top_k=1, elicited=False)

    trajectory = run_proxy_clock_trajectory(
        instance,
        [i1, i2],
        cfg,
        proxy_cfg,
        scenario_name="cli_max_rounds",
        scenario_seed=0,
        num_goods=1,
        num_bidders=2,
    )

    assert trajectory.final_result.rounds == 40


# ---------------------------------------------------------------------------
# 2. Finalise-at-round welfare
# ---------------------------------------------------------------------------


def test_finalise_at_round_only_sees_atoms_accumulated_so_far():
    """i1's reported value for {X} grows each round; i2's is fixed at 40.

    The finalise-at-round WDP re-solved after every round must reflect only
    the atoms known as of that round -- so the reported (winner-declared)
    welfare should track i1's *current* reported value once i1 overtakes,
    not jump straight to the final value on round 1.
    """
    instance = AuctionInstance(
        items=["X"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"X"}): 50.0},
            "i2": {frozenset({"X"}): 40.0},
        },
    )
    i1 = ScriptedRoundClockProxy(
        bidder_id="i1",
        schedule={
            0: (frozenset({"X"}), 10.0),
            1: (frozenset({"X"}), 30.0),
            2: (frozenset({"X"}), 50.0),
        },
        default=(frozenset({"X"}), 50.0),
    )
    i2 = ScriptedRoundClockProxy(
        bidder_id="i2",
        default=(frozenset({"X"}), 40.0),
    )

    cfg = ClockConfig(max_rounds=3, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(top_k=1, elicited=False)

    trajectory = run_proxy_clock_trajectory(
        instance,
        [i1, i2],
        cfg,
        proxy_cfg,
        scenario_name="two_bidder_race",
        scenario_seed=0,
        num_goods=1,
        num_bidders=2,
    )

    assert len(trajectory.round_rows) == 3

    # Round 1: only i1's round-0 atom (10.0) known -- i2 (40.0) wins.
    assert trajectory.round_rows[0]["finalise_reported_welfare"] == pytest.approx(40.0)
    assert trajectory.round_rows[0]["finalise_true_welfare"] == pytest.approx(40.0)
    # Round 2: i1's atom is now 30.0 -- i2 (40.0) still wins.
    assert trajectory.round_rows[1]["finalise_reported_welfare"] == pytest.approx(40.0)
    # Round 3: i1's atom is now 50.0 -- i1 overtakes.
    assert trajectory.round_rows[2]["finalise_reported_welfare"] == pytest.approx(50.0)
    assert trajectory.round_rows[2]["finalise_true_welfare"] == pytest.approx(50.0)

    assert trajectory.round_rows[0]["allocation_changed_from_previous_round"] is False
    assert trajectory.round_rows[1]["allocation_changed_from_previous_round"] is False
    assert trajectory.round_rows[2]["allocation_changed_from_previous_round"] is True

    assert trajectory.round_rows[2]["full_info_welfare"] == pytest.approx(50.0)
    assert trajectory.round_rows[2]["finalise_global_efficiency"] == pytest.approx(1.0)
    assert trajectory.round_rows[0]["finalise_global_efficiency"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 3. Top-k support
# ---------------------------------------------------------------------------


def test_topk_1_2_3_each_produce_a_separate_trajectory(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)

    trajectories = {}
    for k in (1, 2, 3):
        trajectories[k] = run_proxy_clock_trajectory(
            toy_instance,
            make_proxies(toy_instance, initial="empty"),
            cfg,
            ProxyClockConfig(top_k=k, elicited=False),
            scenario_name="toy",
            scenario_seed=0,
            num_goods=3,
            num_bidders=3,
        )

    assert set(trajectories) == {1, 2, 3}
    for k, trajectory in trajectories.items():
        assert trajectory.round_rows, f"top_k={k} produced no rounds"
        assert all(row["top_k"] == k for row in trajectory.round_rows)
        assert all(row["top_k"] == k for row in trajectory.bidder_round_rows)
        assert all(row["top_k"] == k for row in trajectory.coverage_rows)


def test_cli_parses_three_top_k_values(monkeypatch):
    import sys

    from examples.run_live_llm_curated_batch import parse_args

    monkeypatch.setattr(
        sys, "argv", ["run_live_llm_curated_batch.py", "--top-k", "1", "2", "3"]
    )
    args = parse_args()
    assert args.top_k == [1, 2, 3]


# ---------------------------------------------------------------------------
# 4. Coverage diagnostics
# ---------------------------------------------------------------------------


def _coverage_instance_and_proxies(*, appears_from_round: int | None):
    instance = AuctionInstance(
        items=["X"],
        bidder_ids=["i1", "i2", "i3"],
        valuations={
            "i1": {frozenset({"X"}): 50.0},
            "i2": {frozenset({"X"}): 1.0},
            "i3": {frozenset({"X"}): 0.5},
        },
    )
    schedule = {}
    if appears_from_round is not None:
        for round_idx in range(appears_from_round, appears_from_round + 3):
            schedule[round_idx] = (frozenset({"X"}), 50.0)
    i1 = ScriptedRoundClockProxy(bidder_id="i1", schedule=schedule, default=(None, 0.0))
    i2 = ScriptedRoundClockProxy(bidder_id="i2", default=(frozenset({"X"}), 1.0))
    # Filler bidder: also demands X for rounds 0-1 only, so excess demand
    # stays positive (2 bidders wanting 1 unit) and the clock doesn't clear
    # before round_idx=2, where i1 (in the "appears later" case) shows up.
    i3 = ScriptedRoundClockProxy(
        bidder_id="i3",
        schedule={
            0: (frozenset({"X"}), 0.5),
            1: (frozenset({"X"}), 0.5),
        },
        default=(None, 0.0),
    )
    return instance, [i1, i2, i3]


def test_coverage_first_seen_round_when_bundle_appears_later():
    instance, proxies = _coverage_instance_and_proxies(appears_from_round=2)
    cfg = ClockConfig(max_rounds=5, price_step=1.0, reserve=0.0)

    trajectory = run_proxy_clock_trajectory(
        instance,
        proxies,
        cfg,
        ProxyClockConfig(top_k=1, elicited=False),
        scenario_name="coverage_seen_later",
        scenario_seed=0,
        num_goods=1,
        num_bidders=3,
    )

    coverage_by_bidder = {row["bidder_id"]: row for row in trajectory.coverage_rows}
    i1_row = coverage_by_bidder["i1"]
    assert i1_row["full_info_winning_bundle_nonempty"] is True
    assert i1_row["seen_in_top_k_by_final"] is True
    # appears_from_round=2 (0-based) -> round 3 in the 1-based CSV convention.
    assert i1_row["first_seen_round"] == 3
    assert i1_row["first_seen_as_primary_round"] == 3
    assert i1_row["included_in_supplementary_by_final"] is True
    assert i1_row["omitted_bundle_failure"] is False


def test_coverage_omitted_bundle_failure_when_never_seen():
    instance, proxies = _coverage_instance_and_proxies(appears_from_round=None)
    cfg = ClockConfig(max_rounds=5, price_step=1.0, reserve=0.0)

    trajectory = run_proxy_clock_trajectory(
        instance,
        proxies,
        cfg,
        ProxyClockConfig(top_k=1, elicited=False),
        scenario_name="coverage_never_seen",
        scenario_seed=0,
        num_goods=1,
        num_bidders=3,
    )

    coverage_by_bidder = {row["bidder_id"]: row for row in trajectory.coverage_rows}
    i1_row = coverage_by_bidder["i1"]
    assert i1_row["full_info_winning_bundle_nonempty"] is True
    assert i1_row["seen_in_top_k_by_final"] is False
    assert i1_row["first_seen_round"] == ""
    assert i1_row["included_in_supplementary_by_final"] is False
    assert i1_row["omitted_bundle_failure"] is True
    assert trajectory.failure_classification == "candidate_omission"


# ---------------------------------------------------------------------------
# 5. Event usefulness
# ---------------------------------------------------------------------------


def test_event_usefulness_signed_and_abs_correction(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(
        top_k=1,
        elicited=True,
        margin_threshold=10_000.0,
        tie_threshold=10_000.0,
        max_refinements_per_bidder=10,
    )

    trajectory = run_proxy_clock_trajectory(
        toy_instance,
        _tampered_full_info_proxies(toy_instance),
        cfg,
        proxy_cfg,
        scenario_name="toy",
        scenario_seed=0,
        num_goods=3,
        num_bidders=3,
    )

    matches = [
        row
        for row in trajectory.event_rows
        if row["bidder_id"] == "i1" and row["target_bundle"] == "[A,B]"
    ]
    assert len(matches) == 1
    row = matches[0]
    assert row["old_reported_value"] == 999.0
    assert row["new_reported_value"] == 15.0
    assert row["true_value"] == 15.0
    assert row["signed_correction"] == pytest.approx(15.0 - 999.0)
    assert row["abs_correction"] == pytest.approx(984.0)
    # {A,B} is not i1's full-info winning bundle (i1 wins nothing in the toy
    # instance's full-info allocation) or the proxy-clock final allocation.
    assert row["appears_in_full_info_allocation"] is False
    assert row["appears_in_final_allocation"] is False


def test_event_usefulness_event_types_present(toy_instance):
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(
        top_k=1,
        elicited=True,
        margin_threshold=10_000.0,
        tie_threshold=10_000.0,
        max_refinements_per_bidder=10,
    )

    trajectory = run_proxy_clock_trajectory(
        toy_instance,
        _tampered_full_info_proxies(toy_instance),
        cfg,
        proxy_cfg,
        scenario_name="toy",
        scenario_seed=0,
        num_goods=3,
        num_bidders=3,
    )

    event_types = {row["event_type"] for row in trajectory.event_rows}
    assert event_types <= {"near_tie", "near_zero_surplus", "demand_changed"}
    assert event_types  # at least one event fired


# ---------------------------------------------------------------------------
# Query-accounting fix regression tests
# ---------------------------------------------------------------------------


def _scripted_race_with_refinement():
    """i1 under-reports (10.0) until refined to its true value (50.0); i2 is
    static at its true value (40.0). elicited=True with a huge margin
    threshold fires a near_zero_surplus event for both bidders in round 1,
    but only i1 has a `true_values` entry, so only i1's refine() does
    anything -- exactly one real refinement, deterministically in round 1
    (state.refined_bundles then permanently excludes it from future
    candidates)."""
    instance = AuctionInstance(
        items=["X"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"X"}): 50.0},
            "i2": {frozenset({"X"}): 40.0},
        },
    )
    i1 = ScriptedRoundClockProxy(
        bidder_id="i1",
        default=(frozenset({"X"}), 10.0),
        true_values={frozenset({"X"}): 50.0},
    )
    i2 = ScriptedRoundClockProxy(bidder_id="i2", default=(frozenset({"X"}), 40.0))
    return instance, [i1, i2]


def test_clock_trajectory_cumulative_value_queries_matches_final_result_total():
    instance, proxies = _scripted_race_with_refinement()
    cfg = ClockConfig(max_rounds=5, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(
        top_k=1,
        elicited=True,
        margin_threshold=10_000.0,
        tie_threshold=10_000.0,
        max_refinements_per_bidder=10,
    )

    trajectory = run_proxy_clock_trajectory(
        instance,
        proxies,
        cfg,
        proxy_cfg,
        scenario_name="race",
        scenario_seed=0,
        num_goods=1,
        num_bidders=2,
    )

    independent_total = sum(p.stats().value_queries for p in proxies)
    assert independent_total > 0
    assert trajectory.round_rows[-1]["cumulative_value_queries"] == independent_total


def test_clock_trajectory_query_counts_nondecreasing_and_positive_on_refinement():
    instance, proxies = _scripted_race_with_refinement()
    cfg = ClockConfig(max_rounds=5, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(
        top_k=1,
        elicited=True,
        margin_threshold=10_000.0,
        tie_threshold=10_000.0,
        max_refinements_per_bidder=10,
    )

    trajectory = run_proxy_clock_trajectory(
        instance,
        proxies,
        cfg,
        proxy_cfg,
        scenario_name="race",
        scenario_seed=0,
        num_goods=1,
        num_bidders=2,
    )

    cum_vq = [row["cumulative_value_queries"] for row in trajectory.round_rows]
    cum_dq = [row["cumulative_demand_queries"] for row in trajectory.round_rows]
    assert cum_vq == sorted(cum_vq)
    assert cum_dq == sorted(cum_dq)

    # The one real refinement (i1's {X}, round 1) must show up as a positive
    # new_value_queries in round 1, and nowhere else.
    assert trajectory.round_rows[0]["new_value_queries"] == 1
    assert all(row["new_value_queries"] == 0 for row in trajectory.round_rows[1:])
    assert trajectory.round_rows[-1]["cumulative_value_queries"] == 1


# ---------------------------------------------------------------------------
# Top-k comparison: true welfare, not reported welfare
# ---------------------------------------------------------------------------


def test_topk_comparison_uses_true_welfare_not_reported(capsys):
    from examples.run_live_llm_curated_batch import print_topk_comparison

    # Regression case straight from the bug report: the proxy *over*-reports
    # welfare (4730) above the full-info optimum (4290), while its *true*
    # (ground-truth) welfare (3270) is actually below it. The efficiency
    # numerator must be the true figure, never the reported one.
    rows = [{
        "top_k": 1,
        "true_welfare": 3270.0,
        "reported_welfare": 4730.0,
        "full_info_welfare": 4290.0,
        "efficiency": 3270.0 / 4290.0,
        "cumulative_value_queries": 13,
        "supplementary_atoms_total": 92,
        "fi_winner_coverage_hits": 2,
        "fi_winner_coverage_total": 3,
        "failure_classification": "refined_but_miscalibrated",
    }]

    print_topk_comparison(rows)
    captured = capsys.readouterr().out

    assert "3270/4290" in captured
    assert "4730" in captured
    # The bug: reported welfare (4730) must never appear as the numerator
    # over full_info_welfare.
    assert "4730/4290" not in captured


def test_topk_comparison_header_disambiguates_welfare_and_atoms(capsys):
    """Column headers must not use bare 'welfare'/'atoms' labels -- both are
    overloaded elsewhere in the codebase (reported vs true welfare;
    candidate vs supplementary atom counts)."""
    from examples.run_live_llm_curated_batch import print_topk_comparison

    print_topk_comparison([{
        "top_k": 1,
        "true_welfare": 100.0,
        "reported_welfare": 100.0,
        "full_info_welfare": 100.0,
        "efficiency": 1.0,
        "cumulative_value_queries": 1,
        "supplementary_atoms_total": 5,
        "fi_winner_coverage_hits": 1,
        "fi_winner_coverage_total": 1,
        "failure_classification": "no_failure_global",
    }])
    header = capsys.readouterr().out.splitlines()[2]
    assert "true welfare" in header
    assert "supp atoms" in header
    assert "  atoms" not in header


def test_proxy_clock_trajectory_header_disambiguates_welfare_and_atoms(capsys):
    from examples.run_live_llm_curated_batch import print_proxy_clock_trajectory

    print_proxy_clock_trajectory(1, [{
        "round": 1,
        "finalise_true_welfare": 100.0,
        "full_info_welfare": 100.0,
        "finalise_global_efficiency": 1.0,
        "new_value_queries": 0,
        "cumulative_value_queries": 0,
        "supplementary_atoms_total": 5,
        "num_events_this_round": 0,
        "allocation_changed_from_previous_round": False,
    }])
    header = capsys.readouterr().out.splitlines()[2]
    assert "true welfare" in header
    assert "supp atoms" in header
    assert "  atoms" not in header


def test_results_table_header_and_legend_disambiguate_welfare_and_surplus(capsys):
    from examples.run_live_llm_curated_batch import _print_results_table

    _print_results_table([{
        "scenario": "s1",
        "arm": "sealed",
        "efficiency": 1.0,
        "true_welfare": 100.0,
        "full_info_welfare": 100.0,
        "revenue": 50.0,
        "surplus": 50.0,
        "tok_in": 0,
        "tok_out": 0,
    }])
    out = capsys.readouterr().out
    assert "true welfare" in out
    assert "reported" in out.lower() and "REPORTED bids" in out
    assert "true surplus" in out.lower() or "true welfare − revenue" in out


# ---------------------------------------------------------------------------
# Failure classification: global vs mechanism-relative tiers
# ---------------------------------------------------------------------------


def _coverage_row(
    *,
    nonempty=True,
    included=True,
    seen=True,
    first_refined_round="",
    fi_value=100.0,
    error=0.0,
):
    return {
        "full_info_winning_bundle_nonempty": nonempty,
        "included_in_supplementary_by_final": included,
        "seen_in_top_k_by_final": seen,
        "first_refined_round": first_refined_round,
        "full_info_winning_value": fi_value,
        "final_value_error": error,
    }


def test_failure_classification_case_a_reaches_global_optimum():
    from auctionlab.experiments.proxy_clock_trajectory import classify_clock_failure

    label = classify_clock_failure(
        final_true_welfare=100.0,
        global_full_info_welfare=100.0,
        mechanism_full_info_welfare=90.0,
        best_true_welfare=100.0,
        termination_reason="no_excess_demand",
        coverage_rows=[],
    )
    assert label == "no_failure_global"


def test_failure_classification_case_b_reaches_mechanism_benchmark_only():
    from auctionlab.experiments.proxy_clock_trajectory import classify_clock_failure

    label = classify_clock_failure(
        final_true_welfare=80.0,
        global_full_info_welfare=100.0,
        mechanism_full_info_welfare=80.0,
        best_true_welfare=80.0,
        termination_reason="no_excess_demand",
        coverage_rows=[],
    )
    assert label == "no_failure_mechanism_relative"


def test_failure_classification_case_c_omitted_winning_bundle():
    from auctionlab.experiments.proxy_clock_trajectory import classify_clock_failure

    label = classify_clock_failure(
        final_true_welfare=50.0,
        global_full_info_welfare=100.0,
        mechanism_full_info_welfare=90.0,
        best_true_welfare=50.0,
        termination_reason="no_excess_demand",
        coverage_rows=[_coverage_row(included=False)],
    )
    assert label == "candidate_omission"


def test_failure_classification_case_d_refined_but_miscalibrated():
    from auctionlab.experiments.proxy_clock_trajectory import classify_clock_failure

    label = classify_clock_failure(
        final_true_welfare=50.0,
        global_full_info_welfare=100.0,
        mechanism_full_info_welfare=90.0,
        best_true_welfare=50.0,
        termination_reason="no_excess_demand",
        coverage_rows=[
            _coverage_row(
                included=True,
                seen=True,
                first_refined_round=2,
                fi_value=100.0,
                error=50.0,
            )
        ],
    )
    assert label == "refined_but_miscalibrated"


def test_failure_classification_seen_but_not_refined():
    from auctionlab.experiments.proxy_clock_trajectory import classify_clock_failure

    label = classify_clock_failure(
        final_true_welfare=50.0,
        global_full_info_welfare=100.0,
        mechanism_full_info_welfare=90.0,
        best_true_welfare=50.0,
        termination_reason="no_excess_demand",
        coverage_rows=[
            _coverage_row(
                included=True,
                seen=True,
                first_refined_round="",
                fi_value=100.0,
                error=50.0,
            )
        ],
    )
    assert label == "seen_but_not_refined"


def test_failure_classification_late_refinement_worsened_final_allocation():
    from auctionlab.experiments.proxy_clock_trajectory import classify_clock_failure

    # Best round reached 98.6% of the global optimum; the final round
    # regressed to 74.7% -- a >5pp drop despite no coverage/calibration
    # issue by itself (mirrors the 8x8 live-run bug report).
    label = classify_clock_failure(
        final_true_welfare=747.0,
        global_full_info_welfare=1000.0,
        mechanism_full_info_welfare=1000.0,
        best_true_welfare=986.0,
        termination_reason="no_excess_demand",
        coverage_rows=[],
    )
    assert label == "late_refinement_worsened_final_allocation"


def test_failure_classification_hit_max_rounds_unconverged():
    from auctionlab.experiments.proxy_clock_trajectory import classify_clock_failure

    label = classify_clock_failure(
        final_true_welfare=50.0,
        global_full_info_welfare=100.0,
        mechanism_full_info_welfare=90.0,
        best_true_welfare=50.0,
        termination_reason="max_rounds_reached",
        coverage_rows=[],
    )
    assert label == "hit_max_rounds_unconverged"


def test_failure_classification_k3_reaches_mechanism_not_global(toy_instance):
    """Integration-level sibling of the unit cases above: a real clock run
    (top_k=1, ground-truth proxies) where the mechanism itself is the
    binding constraint, not proxy elicitation -- true welfare should equal
    the *mechanism's* full-info benchmark for this top_k, and the failure
    label should reflect that distinction rather than blaming refinement."""
    cfg = ClockConfig(max_rounds=20, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(top_k=1, elicited=True, margin_threshold=100.0, tie_threshold=100.0)

    trajectory = run_proxy_clock_trajectory(
        toy_instance,
        make_proxies(toy_instance, initial="empty"),
        cfg,
        proxy_cfg,
        scenario_name="toy",
        scenario_seed=0,
        num_goods=3,
        num_bidders=3,
    )

    # FullInfoAuctionProxy always reports truthfully, so the proxy clock's
    # true welfare must equal (not just approximate) the same top_k's
    # full-info clock benchmark -- whatever that mechanism ceiling is.
    assert trajectory.final_result.metadata["mechanism_full_info_welfare"] is not None
    final_true_welfare = trajectory.final_result.metadata["final_true_welfare"]
    mechanism_welfare = trajectory.final_result.metadata["mechanism_full_info_welfare"]
    assert final_true_welfare == pytest.approx(mechanism_welfare)
    assert trajectory.failure_classification in (
        "no_failure_global",
        "no_failure_mechanism_relative",
    )


# ---------------------------------------------------------------------------
# 6. Best-vs-final round diagnostics
# ---------------------------------------------------------------------------


def _late_regression_instance_and_proxies():
    """i1 truly values {X} at 100 and reports it correctly for two rounds --
    the clock would finalize efficiently (welfare 100) if it stopped there.
    In the third and final round i1's reported value drops to 30 (a bad
    late correction), so the finalise-at-round WDP flips the winner to i2
    (whose report of 40 has been correct and static throughout), producing
    a final allocation worse than the best one seen mid-run. i1 always
    demands {X} regardless of its reported value, so excess demand for the
    single unit never clears and the clock runs the full round cap
    deterministically.
    """
    instance = AuctionInstance(
        items=["X"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"X"}): 100.0},
            "i2": {frozenset({"X"}): 40.0},
        },
    )
    i1 = ScriptedRoundClockProxy(
        bidder_id="i1",
        schedule={
            0: (frozenset({"X"}), 100.0),
            1: (frozenset({"X"}), 100.0),
            2: (frozenset({"X"}), 30.0),
        },
        default=(frozenset({"X"}), 30.0),
    )
    i2 = ScriptedRoundClockProxy(bidder_id="i2", default=(frozenset({"X"}), 40.0))
    return instance, [i1, i2]


def test_best_final_diagnostics_when_late_report_change_worsens_final_allocation():
    instance, proxies = _late_regression_instance_and_proxies()
    cfg = ClockConfig(max_rounds=3, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(top_k=1, elicited=False)

    trajectory = run_proxy_clock_trajectory(
        instance,
        proxies,
        cfg,
        proxy_cfg,
        scenario_name="late_regression",
        scenario_seed=0,
        num_goods=1,
        num_bidders=2,
    )

    assert len(trajectory.round_rows) == 3
    # Rounds 1-2: i1's correctly-reported 100 wins -- true welfare 100.
    assert trajectory.round_rows[0]["finalise_true_welfare"] == pytest.approx(100.0)
    assert trajectory.round_rows[1]["finalise_true_welfare"] == pytest.approx(100.0)
    # Round 3: i1's late (wrong) report of 30 loses to i2's honest 40.
    assert trajectory.round_rows[2]["finalise_true_welfare"] == pytest.approx(40.0)

    best_final = trajectory.best_final
    assert best_final["best_round"] == 1
    assert best_final["best_true_welfare"] == pytest.approx(100.0)
    assert best_final["final_round"] == 3
    assert best_final["final_true_welfare"] == pytest.approx(40.0)
    assert best_final["final_welfare_minus_best_welfare"] == pytest.approx(-60.0)
    assert best_final["best_true_welfare"] > best_final["final_true_welfare"]

    assert trajectory.failure_classification == (
        "late_refinement_worsened_final_allocation"
    )
    assert trajectory.final_result.metadata["best_round"] == 1
    assert trajectory.final_result.metadata["final_round"] == 3
    assert trajectory.final_result.metadata[
        "failure_classification"
    ] == "late_refinement_worsened_final_allocation"


def test_proxy_clock_result_to_row_surfaces_best_final_fields():
    from auctionlab.experiments.llm_comparison import proxy_clock_result_to_row

    instance, proxies = _late_regression_instance_and_proxies()
    cfg = ClockConfig(max_rounds=3, price_step=1.0, reserve=0.0)
    proxy_cfg = ProxyClockConfig(top_k=1, elicited=False)

    trajectory = run_proxy_clock_trajectory(
        instance,
        proxies,
        cfg,
        proxy_cfg,
        scenario_name="late_regression",
        scenario_seed=0,
        num_goods=1,
        num_bidders=2,
    )

    row = proxy_clock_result_to_row(
        instance_name="late_regression",
        instance=instance,
        result=trajectory.final_result,
    )

    assert row["best_round"] == 1
    assert row["final_round"] == 3
    assert row["best_true_welfare"] == pytest.approx(100.0)
    assert row["final_true_welfare"] == pytest.approx(40.0)
    assert row["final_welfare_minus_best_welfare"] == pytest.approx(-60.0)
    assert row["failure_classification"] == (
        "late_refinement_worsened_final_allocation"
    )
