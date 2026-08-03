from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from auctionlab.auctions.sealed_vcg import run_sealed_xor_vcg
from auctionlab.bids.xor import Bundle, XorAtomicBid, XorBid
from auctionlab.experiments.llm_comparison import (
    proxy_sealed_result_to_row,
    proxy_sealed_trajectory_to_rows,
)
from auctionlab.experiments.proxy_sealed_runner import (
    ProxySealedConfig,
    _competitive_frontier,
    run_proxy_sealed_vcg_experiment,
    run_proxy_sealed_vcg_trajectory,
)
from auctionlab.solvers.wdp_ilp import solve_wdp_xor_ilp
from auctionlab.instances.base import AuctionInstance
from auctionlab.proxies.base import ElicitationEvent, ProxyStats, RefinementRecord
from auctionlab.proxies.full_info import FullInfoAuctionProxy


def _upsert(bid: XorBid, bundle: Bundle, value: float) -> None:
    for idx, atom in enumerate(bid.atoms):
        if atom.bundle == bundle:
            bid.atoms[idx] = XorAtomicBid(bundle=bundle, value=value)
            return
    bid.atoms.append(XorAtomicBid(bundle=bundle, value=value))


@dataclass
class ScriptedSealedProxy:
    """A test double whose ``refine`` replaces an atom's value on cue."""

    bidder_id: str
    initial_atoms: list[XorAtomicBid]
    refined_value_by_bundle: dict[Bundle, float] = field(default_factory=dict)
    _bid: XorBid = field(init=False, repr=False)
    _stats: ProxyStats = field(default_factory=ProxyStats, init=False)

    def __post_init__(self) -> None:
        self._bid = XorBid(bidder_id=self.bidder_id, atoms=list(self.initial_atoms))

    def current_bid(self) -> XorBid:
        return self._bid

    def submit_bid(self) -> XorBid:
        return self._bid

    def refine(self, event: ElicitationEvent) -> None:
        if event.bundle is None:
            return
        value = self.refined_value_by_bundle.get(event.bundle)
        if value is None:
            return
        self._stats.refinement_queries += 1
        _upsert(self._bid, event.bundle, value)

    def receive_provisional_feedback(self, event: ElicitationEvent) -> None:
        self.refine(event)

    def stats(self) -> ProxyStats:
        return self._stats


@dataclass
class RecordingScriptedSealedProxy(ScriptedSealedProxy):
    """``ScriptedSealedProxy`` that also tracks ``RefinementRecord``\\ s."""

    _records: list[RefinementRecord] = field(default_factory=list, init=False, repr=False)

    def refine(self, event: ElicitationEvent) -> None:
        if event.bundle is None:
            return
        new_value = self.refined_value_by_bundle.get(event.bundle)
        if new_value is None:
            return

        old_value = None
        for atom in self._bid.atoms:
            if atom.bundle == event.bundle:
                old_value = atom.value
                break

        self._stats.refinement_queries += 1
        _upsert(self._bid, event.bundle, new_value)
        self._records.append(
            RefinementRecord(
                bidder_id=self.bidder_id,
                mechanism=event.mechanism,
                event_type=event.event_type,
                round_idx=event.round_idx,
                bundle=event.bundle,
                old_value=old_value,
                new_value=new_value,
                reason=event.reason,
            )
        )

    def refinement_records(self) -> list[RefinementRecord]:
        return list(self._records)


@dataclass
class MultiStepScriptedSealedProxy:
    """A test double that steps through a sequence of values on each refine.

    Unlike :class:`ScriptedSealedProxy` (which snaps a bundle's value to a
    single fixed target), each call to ``refine`` for a given bundle
    consumes the next value in ``value_sequence_by_bundle[bundle]``. This
    lets tests simulate a proxy that only gradually converges on its true
    value across several elicitation rounds.
    """

    bidder_id: str
    initial_atoms: list[XorAtomicBid]
    value_sequence_by_bundle: dict[Bundle, list[float]] = field(default_factory=dict)
    _bid: XorBid = field(init=False, repr=False)
    _stats: ProxyStats = field(default_factory=ProxyStats, init=False)
    _call_count_by_bundle: dict[Bundle, int] = field(default_factory=dict, init=False, repr=False)
    _records: list[RefinementRecord] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._bid = XorBid(bidder_id=self.bidder_id, atoms=list(self.initial_atoms))

    def current_bid(self) -> XorBid:
        return self._bid

    def submit_bid(self) -> XorBid:
        return self._bid

    def refine(self, event: ElicitationEvent) -> None:
        if event.bundle is None:
            return
        sequence = self.value_sequence_by_bundle.get(event.bundle)
        if not sequence:
            return
        idx = self._call_count_by_bundle.get(event.bundle, 0)
        if idx >= len(sequence):
            return

        old_value = None
        for atom in self._bid.atoms:
            if atom.bundle == event.bundle:
                old_value = atom.value
                break

        new_value = sequence[idx]
        self._call_count_by_bundle[event.bundle] = idx + 1
        self._stats.refinement_queries += 1
        # Mirror FullInfoAuctionProxy: a refinement resolves one value
        # lookup, so it counts as a value query too (this is the no-logger
        # fallback path -- see aggregate_query_counts).
        self._stats.value_queries += 1
        _upsert(self._bid, event.bundle, new_value)
        self._records.append(
            RefinementRecord(
                bidder_id=self.bidder_id,
                mechanism=event.mechanism,
                event_type=event.event_type,
                round_idx=event.round_idx,
                bundle=event.bundle,
                old_value=old_value,
                new_value=new_value,
                reason=event.reason,
            )
        )

    def receive_provisional_feedback(self, event: ElicitationEvent) -> None:
        self.refine(event)

    def stats(self) -> ProxyStats:
        return self._stats

    def refinement_records(self) -> list[RefinementRecord]:
        return list(self._records)


def make_two_bidder_instance() -> AuctionInstance:
    return AuctionInstance(
        items=["A"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"A"}): 5.0},
            "i2": {frozenset({"A"}): 8.0},
        },
    )


def test_static_config_reproduces_sealed_vcg_on_initial_bids():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {
                frozenset({"A"}): 10.0,
                frozenset({"B"}): 5.0,
            },
            "i2": {
                frozenset({"A"}): 4.0,
                frozenset({"B"}): 9.0,
            },
        },
    )
    proxies = [
        FullInfoAuctionProxy(bidder_id="i1", instance=instance, initial="all_atoms"),
        FullInfoAuctionProxy(bidder_id="i2", instance=instance, initial="all_atoms"),
    ]

    result = run_proxy_sealed_vcg_experiment(
        instance,
        proxies,
        ProxySealedConfig(),
    )

    expected = run_sealed_xor_vcg(
        items=instance.items,
        bids=[proxy.submit_bid() for proxy in proxies],
    )

    assert result.mechanism == "proxy_sealed_vcg_static"
    assert result.allocation == expected.allocation
    assert result.welfare == expected.welfare
    assert result.payments == expected.payments
    assert result.metadata["elicitation_rounds"] == 0
    assert result.metadata["feedback_rule"] == "none"
    assert result.metadata["refinement_query_count_by_bidder"] == {
        "i1": 0,
        "i2": 0,
    }


def test_elicitation_round_sends_feedback_and_improves_allocation():
    instance = make_two_bidder_instance()
    i1 = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
        refined_value_by_bundle={frozenset({"A"}): 20.0},
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )

    # Static: i2 wins (8 > 5).
    static_result = run_proxy_sealed_vcg_experiment(
        instance,
        [i1, i2],
        ProxySealedConfig(),
    )
    assert static_result.allocation["i2"] == frozenset({"A"})
    assert static_result.allocation["i1"] == frozenset()

    # Reset for the elicited run.
    i1 = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
        refined_value_by_bundle={frozenset({"A"}): 20.0},
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )

    elicited_result = run_proxy_sealed_vcg_experiment(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=1,
            feedback_rule="lost_interested_bundle",
            max_refinements_per_bidder=1,
        ),
    )

    assert elicited_result.mechanism == (
        "proxy_sealed_vcg_elicited_lost_interested_bundle_1"
    )
    # After refinement, i1's value for A (20) beats i2's (8).
    assert elicited_result.allocation["i1"] == frozenset({"A"})
    assert elicited_result.allocation["i2"] == frozenset()
    assert elicited_result.metadata["refinement_query_count_by_bidder"] == {
        "i1": 1,
        "i2": 0,
    }
    assert elicited_result.metadata["feedback_rule"] == "lost_interested_bundle"
    assert elicited_result.metadata["elicitation_rounds"] == 1


def test_max_refinements_per_bidder_zero_is_unlimited():
    instance = make_two_bidder_instance()
    i1 = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
        refined_value_by_bundle={frozenset({"A"}): 20.0},
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )

    result = run_proxy_sealed_vcg_experiment(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=3,
            feedback_rule="lost_interested_bundle",
            max_refinements_per_bidder=0,
        ),
    )

    # A cap of 0 means unlimited, so i1's refinement goes through and i1
    # ends up winning with its corrected value of 20.
    assert result.allocation["i1"] == frozenset({"A"})
    assert result.metadata["refinement_query_count_by_bidder"] == {
        "i1": 1,
        "i2": 0,
    }


def test_proxy_sealed_row_exports_final_updated_bid_and_refinement_records():
    instance = make_two_bidder_instance()
    i1 = RecordingScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
        refined_value_by_bundle={frozenset({"A"}): 20.0},
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )

    result = run_proxy_sealed_vcg_experiment(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=1,
            feedback_rule="lost_interested_bundle",
            max_refinements_per_bidder=1,
        ),
    )

    # The proxy's own bid is updated in place.
    assert i1.current_bid().value_of(frozenset({"A"})) == 20.0
    assert i1.submit_bid().value_of(frozenset({"A"})) == 20.0

    records = result.metadata["refinement_records_by_bidder"]["i1"]
    assert len(records) == 1
    assert records[0].old_value == 5.0
    assert records[0].new_value == 20.0

    row = proxy_sealed_result_to_row(
        instance_name="two_bidder",
        instance=instance,
        result=result,
    )

    # Initial bid snapshot is unaffected by the later refinement.
    assert "i1={[A]:5.0}" in row["initial_bids"]
    assert "i1={[A]:5.0}" in row["initial_reported_bids"]
    # Final bid reflects the post-refinement value.
    assert "i1={[A]:20.0}" in row["final_bids"]
    assert "i1={[A]:20.0}" in row["final_reported_bids"]
    assert "i1:[A] 5.0->20.0" in row["refinement_records"]


def test_proxies_must_match_instance_bidder_ids():
    instance = make_two_bidder_instance()
    i1 = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
    )

    with pytest.raises(ValueError, match="proxies must contain"):
        run_proxy_sealed_vcg_experiment(instance, [i1], ProxySealedConfig())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"elicitation_rounds": -1},
        {"feedback_rule": "bogus"},
        {"stopping_rule": "bogus"},
        {"max_refinements_per_bidder": -1},
        {"max_total_refinements": -1},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ProxySealedConfig(**kwargs)


def make_three_bidder_instance() -> AuctionInstance:
    return AuctionInstance(
        items=["A"],
        bidder_ids=["i1", "i2", "i3"],
        valuations={
            "i1": {frozenset({"A"}): 5.0},
            "i2": {frozenset({"A"}): 8.0},
            "i3": {frozenset({"A"}): 6.0},
        },
    )


def test_max_total_refinements_caps_across_bidders():
    """A global cap stops firing further events once the total is reached,
    even for a bidder whose own per-bidder count is still zero."""
    instance = make_three_bidder_instance()
    i1 = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
        refined_value_by_bundle={frozenset({"A"}): 1.0},
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )
    i3 = ScriptedSealedProxy(
        bidder_id="i3",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=6.0)],
        refined_value_by_bundle={frozenset({"A"}): 1.0},
    )

    result = run_proxy_sealed_vcg_experiment(
        instance,
        [i1, i2, i3],
        ProxySealedConfig(
            elicitation_rounds=1,
            feedback_rule="lost_interested_bundle",
            max_total_refinements=1,
        ),
    )

    # i2 wins provisionally, so i1 and i3 (both losers) get refinement
    # events -- but the global cap of 1 only lets the first (i1) through.
    counts = result.metadata["refinement_query_count_by_bidder"]
    assert counts["i1"] == 1
    assert counts["i3"] == 0
    assert result.metadata["total_refinement_queries"] == 1
    assert result.metadata["safety_cap_hit"] is True
    assert result.metadata["cap_binding_indicator"] is False  # per-bidder cap unset


def test_refinement_cap_metadata_fields_present_and_unhit_by_default():
    instance = make_two_bidder_instance()
    i1 = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
        refined_value_by_bundle={frozenset({"A"}): 20.0},
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )

    result = run_proxy_sealed_vcg_experiment(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=1,
            feedback_rule="lost_interested_bundle",
        ),
    )

    assert result.metadata["total_refinement_queries"] == 1
    assert result.metadata["per_bidder_refinement_queries"] == {"i1": 1, "i2": 0}
    assert result.metadata["cap_binding_indicator"] is False
    assert result.metadata["safety_cap_hit"] is False

    row = proxy_sealed_result_to_row(
        instance_name="two_bidder",
        instance=instance,
        result=result,
    )
    assert row["total_refinement_queries"] == 1
    assert row["per_bidder_refinement_queries"] == "i1:1;i2:0"
    assert row["cap_binding_indicator"] is False
    assert row["safety_cap_hit"] is False
    assert row["max_total_refinements"] == 0


def _make_trajectory_instance_and_proxies() -> tuple[
    AuctionInstance, MultiStepScriptedSealedProxy, ScriptedSealedProxy
]:
    """A toy instance where i2 wins early, then i1 overtakes as it refines.

    Ground truth: i1's true value for {A} is 12.0, i2's is 8.0 (so the
    full-info-efficient outcome allocates {A} to i1). i1's proxy starts by
    underreporting (bid 5.0) and only reaches the truth after two rounds of
    refinement (6.0 -> 10.0 -> 12.0); i2's proxy never refines. This makes
    round 0-1 inefficient (i2 wins on the higher initial bid) and rounds 2-3
    efficient (i1 overtakes once its bid exceeds i2's static 8.0).
    """
    instance = AuctionInstance(
        items=["A"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"A"}): 12.0},
            "i2": {frozenset({"A"}): 8.0},
        },
    )
    i1 = MultiStepScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
        value_sequence_by_bundle={frozenset({"A"}): [6.0, 10.0, 12.0]},
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )
    return instance, i1, i2


def test_trajectory_records_one_row_per_round():
    instance, i1, i2 = _make_trajectory_instance_and_proxies()

    trajectory = run_proxy_sealed_vcg_trajectory(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=3,
            feedback_rule="all_provisional",
            max_refinements_per_bidder=0,
        ),
    )

    assert len(trajectory) == 4
    assert [r.metadata["elicitation_rounds"] for r in trajectory] == [0, 1, 2, 3]

    # Round 0: i2 wins on the higher initial bid (8 > 5).
    assert trajectory[0].allocation["i2"] == frozenset({"A"})
    assert trajectory[0].allocation["i1"] == frozenset()
    # Round 1: i1's bid rose to 6, still below i2's 8 -- no change.
    assert trajectory[1].allocation["i2"] == frozenset({"A"})
    # Round 2: i1's bid rose to 10, now above i2's 8 -- i1 takes over.
    assert trajectory[2].allocation["i1"] == frozenset({"A"})
    # Round 3: i1's bid rises further to 12 -- still i1.
    assert trajectory[3].allocation["i1"] == frozenset({"A"})


def test_no_new_refinements_stopping_rule_ends_at_first_fixed_point():
    instance, i1, i2 = _make_trajectory_instance_and_proxies()

    trajectory = run_proxy_sealed_vcg_trajectory(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=50,
            feedback_rule="all_provisional",
            stopping_rule="no_new_refinements",
        ),
    )

    # i1 has three scripted refinements. Round 4 is the first completed
    # cycle in which the same feedback produces no new query.
    assert [r.metadata["elicitation_rounds"] for r in trajectory] == [
        0, 1, 2, 3, 4
    ]
    final = trajectory[-1]
    assert final.metadata["requested_elicitation_rounds"] == 50
    assert final.metadata["stopping_rule"] == "no_new_refinements"
    assert final.metadata["termination_reason"] == "no_new_refinements"
    assert sum(
        final.metadata["new_refinement_query_count_by_bidder"].values()
    ) == 0


def test_competitive_stopping_advances_to_novel_bundle_before_converging():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1"],
        valuations={
            "i1": {
                frozenset({"A"}): 10.0,
                frozenset({"B"}): 9.0,
            }
        },
    )
    proxy = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[
            XorAtomicBid(frozenset({"A"}), 10.0),
            XorAtomicBid(frozenset({"B"}), 9.0),
        ],
        refined_value_by_bundle={
            frozenset({"A"}): 10.0,
            frozenset({"B"}): 9.0,
        },
    )

    trajectory = run_proxy_sealed_vcg_trajectory(
        instance,
        [proxy],
        ProxySealedConfig(
            elicitation_rounds=10,
            feedback_rule="competitive",
            stopping_rule="no_new_refinements",
        ),
    )

    assert proxy.stats().refinement_queries == 2
    assert trajectory[-1].metadata["elicitation_rounds"] == 3
    assert (
        trajectory[-1].metadata["termination_reason"]
        == "no_eligible_refinements"
    )


def test_fixed_rounds_does_not_stop_when_refinements_are_exhausted():
    instance, i1, i2 = _make_trajectory_instance_and_proxies()

    trajectory = run_proxy_sealed_vcg_trajectory(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=5,
            feedback_rule="all_provisional",
            stopping_rule="fixed_rounds",
        ),
    )

    assert len(trajectory) == 6
    assert trajectory[-1].metadata["elicitation_rounds"] == 5
    assert trajectory[-1].metadata["termination_reason"] == "max_rounds_reached"


def test_trajectory_round_print_labels_reported_and_true_welfare_separately(capsys):
    """Regression test for the "sealed round N welfare X" ambiguity bug.

    i1's reported bid (50.0) wildly overstates its true value (5.0), so it
    wins the provisional round on the inflated report. The per-round print
    must show both figures, explicitly labeled, rather than a single bare
    "welfare" figure that reads as ground truth but is actually the
    reported/WDP objective.
    """
    instance = AuctionInstance(
        items=["A"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"A"}): 5.0},
            "i2": {frozenset({"A"}): 8.0},
        },
    )
    i1 = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=50.0)],
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )

    run_proxy_sealed_vcg_trajectory(
        instance,
        [i1, i2],
        ProxySealedConfig(elicitation_rounds=1, feedback_rule="none"),
    )

    out = capsys.readouterr().out
    round_lines = [line for line in out.splitlines() if "sealed round" in line]
    assert len(round_lines) == 1
    line = round_lines[0]

    # The old bug printed a single unqualified "welfare {X}"; the fix must
    # only ever say "welfare" as part of an explicit "reported welfare" or
    # "true welfare" phrase -- never bare.
    assert "reported welfare 50" in line
    assert "true welfare 5" in line
    assert line.count("welfare") == 2


def test_trajectory_does_not_reset_proxy_state_between_rounds():
    instance, i1, i2 = _make_trajectory_instance_and_proxies()

    run_proxy_sealed_vcg_trajectory(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=3,
            feedback_rule="all_provisional",
            max_refinements_per_bidder=0,
        ),
    )

    # i1's live proxy state reflects all 3 accumulated refinements, not a
    # fresh proxy re-initialised each round.
    assert i1.stats().refinement_queries == 3
    assert i1.current_bid().value_of(frozenset({"A"})) == 12.0
    assert len(i1.refinement_records()) == 3


def test_trajectory_cumulative_queries_are_nondecreasing():
    instance, i1, i2 = _make_trajectory_instance_and_proxies()

    trajectory = run_proxy_sealed_vcg_trajectory(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=3,
            feedback_rule="all_provisional",
            max_refinements_per_bidder=0,
        ),
    )
    rows = proxy_sealed_trajectory_to_rows(
        scenario_name="toy",
        scenario_seed=0,
        num_goods=1,
        num_bidders=2,
        instance=instance,
        trajectory=trajectory,
    )

    for key in (
        "cumulative_value_queries",
        "cumulative_demand_queries",
        "cumulative_nl_queries",
        "cumulative_refinements",
        "cumulative_tokens_in",
        "cumulative_tokens_out",
    ):
        values = [row[key] for row in rows]
        assert values == sorted(values), f"{key} is not nondecreasing: {values}"

    # Refinements: 0 at round 0, then +1 per round (i1 refines every round).
    assert [row["cumulative_refinements"] for row in rows] == [0, 1, 2, 3]
    assert [row["num_refinements_this_round"] for row in rows] == [0, 1, 1, 1]


def test_trajectory_final_row_matches_existing_proxy_sealed_result():
    instance, i1, i2 = _make_trajectory_instance_and_proxies()
    config = ProxySealedConfig(
        elicitation_rounds=3,
        feedback_rule="all_provisional",
        max_refinements_per_bidder=0,
    )

    trajectory = run_proxy_sealed_vcg_trajectory(instance, [i1, i2], config)

    # Fresh, independently-scripted proxies for the classic single-shot API.
    instance2, i1b, i2b = _make_trajectory_instance_and_proxies()
    assert instance2 == instance
    final_result = run_proxy_sealed_vcg_experiment(instance, [i1b, i2b], config)

    final_round = trajectory[-1]
    assert final_round.mechanism == final_result.mechanism
    assert final_round.allocation == final_result.allocation
    assert final_round.welfare == final_result.welfare
    assert final_round.payments == final_result.payments
    assert final_round.revenue == final_result.revenue
    assert (
        final_round.metadata["elicitation_rounds"]
        == final_result.metadata["elicitation_rounds"]
    )
    assert (
        final_round.metadata["feedback_rule"]
        == final_result.metadata["feedback_rule"]
    )
    assert (
        final_round.metadata["refinement_query_count_by_bidder"]
        == final_result.metadata["refinement_query_count_by_bidder"]
    )


def test_trajectory_allocation_changed_from_previous_round():
    instance, i1, i2 = _make_trajectory_instance_and_proxies()

    trajectory = run_proxy_sealed_vcg_trajectory(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=3,
            feedback_rule="all_provisional",
            max_refinements_per_bidder=0,
        ),
    )
    rows = proxy_sealed_trajectory_to_rows(
        scenario_name="toy",
        scenario_seed=0,
        num_goods=1,
        num_bidders=2,
        instance=instance,
        trajectory=trajectory,
    )

    # Round 0 has no previous round to compare against.
    assert rows[0]["allocation_changed_from_previous_round"] is False
    # Round 1: still i2 -- unchanged.
    assert rows[1]["allocation_changed_from_previous_round"] is False
    # Round 2: i1 overtakes i2 -- changed.
    assert rows[2]["allocation_changed_from_previous_round"] is True
    # Round 3: still i1 -- unchanged.
    assert rows[3]["allocation_changed_from_previous_round"] is False

    assert rows[0]["welfare_delta_from_previous_round"] == 0.0
    assert rows[1]["welfare_delta_from_previous_round"] == 0.0
    assert rows[2]["welfare_delta_from_previous_round"] == pytest.approx(4.0)
    assert rows[3]["welfare_delta_from_previous_round"] == 0.0

    assert rows[0]["true_welfare"] == pytest.approx(8.0)
    assert rows[2]["true_welfare"] == pytest.approx(12.0)
    assert rows[2]["global_efficiency"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Query-accounting fix regression tests
# ---------------------------------------------------------------------------


def test_trajectory_cumulative_value_queries_matches_final_result_total():
    """cumulative_value_queries in the final round must match a fresh,
    independent recomputation of the proxies' own value-query totals --
    i.e. the trajectory's aggregate isn't silently stuck at zero while the
    proxies themselves recorded real value queries (the reported bug)."""
    instance, i1, i2 = _make_trajectory_instance_and_proxies()

    trajectory = run_proxy_sealed_vcg_trajectory(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=3,
            feedback_rule="all_provisional",
            max_refinements_per_bidder=0,
        ),
    )

    final_row = trajectory[-1]
    independent_total = i1.stats().value_queries + i2.stats().value_queries
    assert independent_total > 0
    assert final_row.metadata["cumulative_value_queries"] == independent_total


def test_trajectory_new_value_queries_positive_when_refinements_occur():
    instance, i1, i2 = _make_trajectory_instance_and_proxies()

    trajectory = run_proxy_sealed_vcg_trajectory(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=3,
            feedback_rule="all_provisional",
            max_refinements_per_bidder=0,
        ),
    )

    # i1 refines every round (rounds 1-3), so rounds 1-3 must each show a
    # positive new_value_queries, matching their positive
    # new_refinement_query_count_by_bidder.
    for result in trajectory[1:]:
        assert result.metadata["new_value_queries"] > 0
        assert (
            sum(result.metadata["new_refinement_query_count_by_bidder"].values())
            > 0
        )

    values = [row.metadata["cumulative_value_queries"] for row in trajectory]
    assert values == sorted(values)
def test_competitive_loser_challenger_policy_is_optional():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1", "i2", "i3"],
        valuations={
            "i1": {frozenset({"A"}): 10.0},
            "i2": {frozenset({"B"}): 10.0},
            "i3": {frozenset({"A", "B"}): 5.0},
        },
    )
    bids = {
        bidder_id: XorBid(
            bidder_id=bidder_id,
            atoms=[
                XorAtomicBid(bundle=bundle, value=value)
                for bundle, value in values.items()
            ],
        )
        for bidder_id, values in instance.valuations.items()
    }
    provisional = solve_wdp_xor_ilp(instance.items, list(bids.values()))

    without_challengers = _competitive_frontier(
        instance=instance,
        bids_by_bidder=bids,
        provisional=provisional,
        loser_challenger_policy="off",
    )
    with_challengers = _competitive_frontier(
        instance=instance,
        bids_by_bidder=bids,
        provisional=provisional,
        loser_challenger_policy="shadow_price",
    )

    assert without_challengers["i3"] == []
    assert with_challengers["i3"] == [
        (frozenset({"A", "B"}), "loser_challenger")
    ]


def test_invalid_loser_challenger_policy_is_rejected():
    with pytest.raises(ValueError, match="loser_challenger_policy"):
        ProxySealedConfig(loser_challenger_policy="unknown")
