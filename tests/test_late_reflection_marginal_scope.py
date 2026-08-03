from __future__ import annotations

import json

import pytest

from auctionlab.auctions.clock import ClockConfig
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.experiments.proxy_clock_runner import (
    ProxyClockConfig,
    run_proxy_clock_experiment,
)
from auctionlab.experiments.proxy_sealed_runner import (
    ProxySealedConfig,
    run_proxy_sealed_vcg_trajectory,
)
from auctionlab.experiments.run_config import (
    late_reflection_candidates_to_rows,
    late_reflection_records_to_rows,
    late_reflection_summary_fields,
)
from auctionlab.instances.base import AuctionInstance
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.late_reflection import (
    LateReflectionConfig,
    clock_marginality_scores,
    rank_and_select_marginal_bidders,
    run_late_reflection_trigger,
    sealed_marginality_scores,
)
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmAuctionProxyAdapter, LlmInferredXorProxy
from auctionlab.proxies.base import ElicitationEvent


ITEM_DESCRIPTIONS = {g: f"Item {g}" for g in "ABCDEFGXYZW"}


def _make_proxy(bidder_id: str, responses: list[str], *, epsilon: float = 1.0) -> LlmInferredXorProxy:
    person = LlmPersonSimulator(
        bidder_id=bidder_id,
        scenario_description="A test auction.",
        person_seed="Values items directly.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=MockLlmClient(responses),
    )
    return LlmInferredXorProxy(bidder_id=bidder_id, person=person, epsilon=epsilon)


_REFLECTION_JSON = json.dumps({
    "question": "Q?",
    "reflection_mode": "bundle_comparison",
    "primary_bundle": ["A"],
    "suggested_followup": "value_query",
    "followup_bundles": [["A"]],
})
_ANSWER_JSON = json.dumps({"answer": "Yes."})


def _xor_bid(bidder_id: str, atoms: list[tuple[frozenset, float]]) -> XorBid:
    return XorBid(
        bidder_id=bidder_id,
        atoms=[XorAtomicBid(bundle=b, value=v) for b, v in atoms],
    )


# ---------------------------------------------------------------------------
# A. Config/CLI
# ---------------------------------------------------------------------------

def test_allocation_marginal_accepted_as_scope():
    cfg = LateReflectionConfig(scope="allocation_marginal")
    assert cfg.scope == "allocation_marginal"


def test_late_reflection_max_bidders_accepted_by_config_and_cli(monkeypatch):
    cfg = LateReflectionConfig(late_reflection_max_bidders=3)
    assert cfg.late_reflection_max_bidders == 3
    assert LateReflectionConfig().late_reflection_max_bidders is None

    import sys

    sys.path.insert(0, "examples")
    import run_live_llm_curated_batch as cli

    monkeypatch.setattr(
        sys, "argv",
        ["run_live_llm_curated_batch.py", "--late-reflection",
         "--late-reflection-scope", "allocation_marginal",
         "--late-reflection-max-bidders", "3"],
    )
    args = cli.parse_args()
    assert args.late_reflection_scope == "allocation_marginal"
    assert args.late_reflection_max_bidders == 3


def test_negative_max_bidders_is_rejected():
    with pytest.raises(ValueError):
        LateReflectionConfig(late_reflection_max_bidders=-1)


def test_zero_max_bidders_caps_at_zero_not_unlimited():
    # Documented convention: unlike some other 0-means-unlimited flags in
    # this project, None (not 0) means "no cap" here -- 0 is a valid,
    # literal "select nobody" cap.
    scores = sealed_marginality_scores(
        bidder_ids=["b1"],
        provisional_allocation={"b1": frozenset({"A"})},
        previous_allocation=None,
        bids_by_bidder={"b1": _xor_bid("b1", [(frozenset({"A"}), 100.0)])},
        events=[],
    )
    _ranked, selected = rank_and_select_marginal_bidders(scores, max_bidders=0)
    assert selected == {}


def test_existing_scopes_allocation_relevant_and_all_bidders_still_accepted():
    LateReflectionConfig(scope="allocation_relevant")
    LateReflectionConfig(scope="all_bidders")


# ---------------------------------------------------------------------------
# B. Sealed selection -- shared fixture
# ---------------------------------------------------------------------------

def _sealed_fixture():
    """7 bidders exercising every sealed marginality signal.

    b1: currently allocated only                     -> score 100
    b2: currently allocated AND allocation changed    -> score 200 (highest)
    b3/b4/b5: losing, all inside the top-3 losing     -> score 80 each (tie)
              (raw values deliberately NOT in bidder_id order: b5 has the
              highest bid (300, rank 1), b3 the lowest of the trio (200,
              rank 3) -- so a rank tie-break that silently fell back to bid
              value would sort them b5,b4,b3, not b3,b4,b5.)
    b6: losing, NOT in the top-3, but has a large bundle overlapping the
        allocation AND received sealed feedback                -> score 60
    b7: losing, NOT in the top-3, feedback only ("ordinary loser")
                                                                -> score 20
    """
    provisional_allocation = {
        "b1": frozenset({"A"}), "b2": frozenset({"B"}),
    }
    previous_allocation = {
        "b1": frozenset({"A"}), "b2": frozenset(),
    }
    bids_by_bidder = {
        "b1": _xor_bid("b1", [(frozenset({"A"}), 100.0)]),
        "b2": _xor_bid("b2", [(frozenset({"B"}), 90.0)]),
        "b3": _xor_bid("b3", [(frozenset({"C"}), 200.0)]),
        "b4": _xor_bid("b4", [(frozenset({"D"}), 250.0)]),
        "b5": _xor_bid("b5", [(frozenset({"E"}), 300.0)]),
        "b6": _xor_bid("b6", [
            (frozenset({"F"}), 50.0),
            (frozenset({"A", "C", "D"}), 40.0),
        ]),
        "b7": _xor_bid("b7", [(frozenset({"G"}), 10.0)]),
    }
    events = [
        ElicitationEvent(
            mechanism="proxy_sealed_vcg", event_type="lost_interested_bundle",
            bidder_id="b6", bundle=frozenset({"A", "C", "D"}),
        ),
        ElicitationEvent(
            mechanism="proxy_sealed_vcg", event_type="lost_interested_bundle",
            bidder_id="b7", bundle=frozenset({"G"}),
        ),
    ]
    bidder_ids = ["b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    return bidder_ids, provisional_allocation, previous_allocation, bids_by_bidder, events


def test_currently_allocated_bidder_receives_high_score():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    assert scores["b1"].score >= 100
    assert "currently_allocated" in scores["b1"].reasons


def test_allocation_changed_last_round_receives_high_score():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    assert scores["b2"].score == 200.0
    assert "currently_allocated" in scores["b2"].reasons
    assert "allocation_changed_last_round" in scores["b2"].reasons


def test_losing_bidder_with_top_k_bundle_is_scored():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    for bidder_id in ("b3", "b4", "b5"):
        assert scores[bidder_id].score == 80.0
        assert any(
            r.startswith("top_losing_bundle_rank=") for r in scores[bidder_id].reasons
        )
    # b5 has the highest raw bid (300) -> rank 1; b3 the lowest of the
    # trio (200) -> rank 3.
    assert "top_losing_bundle_rank=1" in scores["b5"].reasons
    assert "top_losing_bundle_rank=3" in scores["b3"].reasons


def test_ordinary_low_score_loser_not_selected_when_cap_binds():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    assert scores["b7"].score == 20.0
    assert scores["b7"].reasons == ("received_sealed_feedback",)

    ranked, selected = rank_and_select_marginal_bidders(scores, max_bidders=5)
    assert "b7" not in selected
    assert "b6" not in selected  # score 60, also excluded by the 5-bidder cap
    assert set(selected) == {"b1", "b2", "b3", "b4", "b5"}


def test_max_bidders_cap_selects_top_ranked_bidders_deterministically():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    ranked, selected = rank_and_select_marginal_bidders(scores, max_bidders=3)
    assert set(selected) == {"b2", "b1", "b3"}
    rank_by_bidder = {bidder_id: rank for bidder_id, _s, rank in ranked}
    assert rank_by_bidder["b2"] == 1
    assert rank_by_bidder["b1"] == 2
    assert rank_by_bidder["b3"] == 3


def test_tie_break_by_bidder_id_is_deterministic_not_by_bid_value():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    ranked, _selected = rank_and_select_marginal_bidders(scores, max_bidders=None)
    tied_ranks = {
        bidder_id: rank for bidder_id, s, rank in ranked
        if bidder_id in ("b3", "b4", "b5")
    }
    # Ascending bidder_id order (b3 < b4 < b5), NOT descending-bid-value
    # order (which would be b5, b4, b3).
    assert tied_ranks["b3"] < tied_ranks["b4"] < tied_ranks["b5"]


def test_max_bidders_none_selects_all_positive_score_bidders():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    _ranked, selected = rank_and_select_marginal_bidders(scores, max_bidders=None)
    assert set(selected) == set(bidder_ids)  # every bidder here has score > 0


def test_no_positive_score_bidders_means_no_calls():
    bidder_ids = ["b1", "b2"]
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids,
        provisional_allocation={},
        previous_allocation=None,
        bids_by_bidder={
            "b1": _xor_bid("b1", []),
            "b2": _xor_bid("b2", []),
        },
        events=[],
    )
    assert all(s.score == 0.0 for s in scores.values())
    _ranked, selected = rank_and_select_marginal_bidders(scores, max_bidders=None)
    assert selected == {}

    instance = AuctionInstance(items=["A"], bidder_ids=bidder_ids, valuations={
        "b1": {}, "b2": {},
    })
    config = LateReflectionConfig(enabled=True, scope="allocation_marginal")
    result = run_late_reflection_trigger(
        instance=instance,
        proxies_by_bidder={
            "b1": _make_proxy("b1", []),
            "b2": _make_proxy("b2", []),
        },
        bids_by_bidder={"b1": _xor_bid("b1", []), "b2": _xor_bid("b2", [])},
        config=config,
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        allocation_relevant_bidders={},
        marginality_scores=scores,
    )
    assert result.records == []
    # Candidates are still reported (both scored zero), just none selected.
    assert len(result.candidates) == 2
    assert all(not c.marginality_selected for c in result.candidates)


# ---------------------------------------------------------------------------
# C. Clock selection
# ---------------------------------------------------------------------------

def _clock_fixture():
    current_demand_by_bidder = {
        "c1": frozenset({"X"}),
        "c2": frozenset({"Y"}),
        "c3": frozenset({"Z"}),
        "c4": None,
        "c5": frozenset({"W"}),
    }
    positive_excess_demand_goods = {"X"}
    contested_goods = {"Y"}
    recent_events_by_bidder = {
        "c2": [ElicitationEvent(
            mechanism="proxy_clock_vcg", event_type="near_tie", bidder_id="c2",
        )],
        "c3": [ElicitationEvent(
            mechanism="proxy_clock_vcg", event_type="near_zero_surplus", bidder_id="c3",
        )],
        "c4": [ElicitationEvent(
            mechanism="proxy_clock_vcg", event_type="demand_changed", bidder_id="c4",
        )],
    }
    old_rule_relevant_bidders = {"c5": "contested_good_in_current_demand"}
    return (
        ["c1", "c2", "c3", "c4", "c5"],
        current_demand_by_bidder,
        positive_excess_demand_goods,
        contested_goods,
        recent_events_by_bidder,
        old_rule_relevant_bidders,
    )


def test_positive_excess_demand_good_receives_high_score():
    ids, demand, pos_excess, contested, events, old_rule = _clock_fixture()
    scores = clock_marginality_scores(
        bidder_ids=ids, current_demand_by_bidder=demand,
        positive_excess_demand_goods=pos_excess, contested_goods=contested,
        recent_events_by_bidder=events, old_rule_relevant_bidders=old_rule,
    )
    assert scores["c1"].score == 100.0
    assert "current_demand_positive_excess_demand_good" in scores["c1"].reasons


def test_recent_near_tie_bidder_receives_score():
    ids, demand, pos_excess, contested, events, old_rule = _clock_fixture()
    scores = clock_marginality_scores(
        bidder_ids=ids, current_demand_by_bidder=demand,
        positive_excess_demand_goods=pos_excess, contested_goods=contested,
        recent_events_by_bidder=events, old_rule_relevant_bidders=old_rule,
    )
    assert "recent_near_tie" in scores["c2"].reasons
    assert scores["c2"].score == 120.0  # near_tie (80) + contested demand (40)


def test_recent_near_zero_surplus_bidder_receives_score():
    ids, demand, pos_excess, contested, events, old_rule = _clock_fixture()
    scores = clock_marginality_scores(
        bidder_ids=ids, current_demand_by_bidder=demand,
        positive_excess_demand_goods=pos_excess, contested_goods=contested,
        recent_events_by_bidder=events, old_rule_relevant_bidders=old_rule,
    )
    assert scores["c3"].score == 70.0
    assert scores["c3"].reasons == ("recent_near_zero_surplus",)


def test_recent_demand_changed_bidder_receives_score():
    ids, demand, pos_excess, contested, events, old_rule = _clock_fixture()
    scores = clock_marginality_scores(
        bidder_ids=ids, current_demand_by_bidder=demand,
        positive_excess_demand_goods=pos_excess, contested_goods=contested,
        recent_events_by_bidder=events, old_rule_relevant_bidders=old_rule,
    )
    assert scores["c4"].score == 50.0
    assert scores["c4"].reasons == ("recent_demand_changed",)


def test_stale_event_outside_window_does_not_score():
    # Simulate "the bidder's only event was outside the recent window" by
    # simply not including it in recent_events_by_bidder -- this function
    # never re-windows, it trusts the caller's already-windowed input.
    ids, demand, pos_excess, contested, _events, old_rule = _clock_fixture()
    scores = clock_marginality_scores(
        bidder_ids=ids, current_demand_by_bidder=demand,
        positive_excess_demand_goods=pos_excess, contested_goods=contested,
        recent_events_by_bidder={},  # nothing in the window
        old_rule_relevant_bidders=old_rule,
    )
    # c2's demand still overlaps the (event-independent) contested-goods
    # signal (+40), but the near_tie event bonus (+80) is gone entirely --
    # 40, not the 120 it scored with the event included.
    assert scores["c2"].score == 40.0
    assert "recent_near_tie" not in scores["c2"].reasons
    assert scores["c3"].score == 0.0  # near_zero_surplus bonus gone
    assert scores["c4"].score == 0.0  # demand_changed bonus gone


def test_clock_max_bidders_cap_selects_top_ranked_deterministically():
    ids, demand, pos_excess, contested, events, old_rule = _clock_fixture()
    scores = clock_marginality_scores(
        bidder_ids=ids, current_demand_by_bidder=demand,
        positive_excess_demand_goods=pos_excess, contested_goods=contested,
        recent_events_by_bidder=events, old_rule_relevant_bidders=old_rule,
    )
    ranked, selected = rank_and_select_marginal_bidders(scores, max_bidders=2)
    assert set(selected) == {"c2", "c1"}  # scores 120, 100 -- the top two
    rank_by_bidder = {bidder_id: rank for bidder_id, _s, rank in ranked}
    assert rank_by_bidder["c2"] == 1
    assert rank_by_bidder["c1"] == 2


def test_old_rule_relevant_but_low_marginality_bidder_excluded_by_cap():
    ids, demand, pos_excess, contested, events, old_rule = _clock_fixture()
    scores = clock_marginality_scores(
        bidder_ids=ids, current_demand_by_bidder=demand,
        positive_excess_demand_goods=pos_excess, contested_goods=contested,
        recent_events_by_bidder=events, old_rule_relevant_bidders=old_rule,
    )
    assert scores["c5"].score == 20.0
    assert scores["c5"].reasons == ("old_rule_allocation_relevant",)
    _ranked, selected = rank_and_select_marginal_bidders(scores, max_bidders=3)
    assert "c5" not in selected  # lowest score, excluded once the cap binds


# ---------------------------------------------------------------------------
# D. Runtime/logging
# ---------------------------------------------------------------------------

def test_allocation_marginal_with_max_bidders_issues_at_most_n_nl_calls():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    # Expected top-3 selection: b2, b1, b3 (see test_max_bidders_cap_...).
    proxies = {}
    for bidder_id in bidder_ids:
        if bidder_id in ("b1", "b2", "b3"):
            proxy = _make_proxy(bidder_id, [_REFLECTION_JSON, _ANSWER_JSON, '{"bundle_value": 105}'])
        else:
            # Never queried -- an empty response list would raise if touched.
            proxy = _make_proxy(bidder_id, [])
        proxy.set_provisional_bid({atom.bundle: atom.value for atom in bids[bidder_id].atoms})
        proxies[bidder_id] = proxy

    instance = AuctionInstance(
        items=["A", "B", "C", "D", "E", "F", "G"],
        bidder_ids=bidder_ids,
        valuations={b: {} for b in bidder_ids},
    )
    config = LateReflectionConfig(
        enabled=True, scope="allocation_marginal", late_reflection_max_bidders=3,
        followup="value_query", followups_per_bidder=1,
    )
    result = run_late_reflection_trigger(
        instance=instance,
        proxies_by_bidder=proxies,
        bids_by_bidder=bids,
        config=config,
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        allocation_relevant_bidders={},
        marginality_scores=scores,
        allocated_bundle_by_bidder=prov,
    )
    # 3 selected bidders, 1 followup each (followups_per_bidder=1) -> 3 rows.
    assert len(result.records) == 3
    assert {r.bidder_id for r in result.records} == {"b1", "b2", "b3"}
    assert all(r.actual_followup_type == "value_query" for r in result.records)
    # Every bidder is still reported as a candidate for inspection.
    assert len(result.candidates) == 7
    selected_candidates = {c.bidder_id for c in result.candidates if c.marginality_selected}
    assert selected_candidates == {"b1", "b2", "b3"}


def test_followups_per_bidder_one_issues_at_most_one_vq_per_selected_bidder():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    multi_bundle_reflection = json.dumps({
        "question": "Q?",
        "reflection_mode": "bundle_comparison",
        "primary_bundle": ["A"],
        "comparison_bundle": ["B"],
        "suggested_followup": "value_query",
        "followup_bundles": [["A"], ["B"]],
    })
    proxies = {}
    for bidder_id in bidder_ids:
        if bidder_id == "b2":
            proxy = _make_proxy(
                bidder_id, [multi_bundle_reflection, _ANSWER_JSON, '{"bundle_value": 95}']
            )
        elif bidder_id in ("b1", "b3"):
            proxy = _make_proxy(bidder_id, [_REFLECTION_JSON, _ANSWER_JSON, '{"bundle_value": 105}'])
        else:
            proxy = _make_proxy(bidder_id, [])
        proxy.set_provisional_bid({atom.bundle: atom.value for atom in bids[bidder_id].atoms})
        proxies[bidder_id] = proxy

    instance = AuctionInstance(
        items=["A", "B", "C", "D", "E", "F", "G"],
        bidder_ids=bidder_ids,
        valuations={b: {} for b in bidder_ids},
    )
    config = LateReflectionConfig(
        enabled=True, scope="allocation_marginal", late_reflection_max_bidders=3,
        followup="value_query", followups_per_bidder=1,
    )
    result = run_late_reflection_trigger(
        instance=instance,
        proxies_by_bidder=proxies,
        bids_by_bidder=bids,
        config=config,
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        allocation_relevant_bidders={},
        marginality_scores=scores,
        allocated_bundle_by_bidder=prov,
    )
    # Even though b2's reflection named two followup bundles, the cap of 1
    # limits it to a single followup row.
    b2_records = [r for r in result.records if r.bidder_id == "b2"]
    assert len(b2_records) == 1


def test_records_csv_includes_marginality_diagnostics():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    proxies = {"b1": _make_proxy("b1", [_REFLECTION_JSON, _ANSWER_JSON, '{"bundle_value": 105}'])}
    for bidder_id in bidder_ids:
        if bidder_id == "b1":
            continue
        proxies[bidder_id] = _make_proxy(bidder_id, [])
    for bidder_id in bidder_ids:
        proxies[bidder_id].set_provisional_bid(
            {atom.bundle: atom.value for atom in bids[bidder_id].atoms}
        )

    instance = AuctionInstance(
        items=["A", "B", "C", "D", "E", "F", "G"],
        bidder_ids=bidder_ids,
        valuations={b: {} for b in bidder_ids},
    )
    config = LateReflectionConfig(
        enabled=True, scope="allocation_marginal", late_reflection_max_bidders=1,
        followup="value_query", followups_per_bidder=1,
    )
    result = run_late_reflection_trigger(
        instance=instance,
        proxies_by_bidder=proxies,
        bids_by_bidder=bids,
        config=config,
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        allocation_relevant_bidders={},
        marginality_scores=scores,
        allocated_bundle_by_bidder=prov,
    )
    assert len(result.records) == 1
    rows = late_reflection_records_to_rows(result.records)
    assert len(rows) == 1
    row = rows[0]
    for column in (
        "marginality_score", "marginality_rank", "marginality_selected",
        "marginality_reasons",
    ):
        assert column in row


def test_candidates_csv_written_with_selected_and_non_selected():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    proxies = {}
    for bidder_id in bidder_ids:
        if bidder_id in ("b1", "b2", "b3"):
            proxies[bidder_id] = _make_proxy(
                bidder_id, [_REFLECTION_JSON, _ANSWER_JSON, '{"bundle_value": 105}']
            )
        else:
            proxies[bidder_id] = _make_proxy(bidder_id, [])
        proxies[bidder_id].set_provisional_bid(
            {atom.bundle: atom.value for atom in bids[bidder_id].atoms}
        )

    instance = AuctionInstance(
        items=["A", "B", "C", "D", "E", "F", "G"],
        bidder_ids=bidder_ids,
        valuations={b: {} for b in bidder_ids},
    )
    config = LateReflectionConfig(
        enabled=True, scope="allocation_marginal", late_reflection_max_bidders=3,
        followup="value_query", followups_per_bidder=1,
    )
    result = run_late_reflection_trigger(
        instance=instance,
        proxies_by_bidder=proxies,
        bids_by_bidder=bids,
        config=config,
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        allocation_relevant_bidders={},
        marginality_scores=scores,
        allocated_bundle_by_bidder=prov,
    )
    rows = late_reflection_candidates_to_rows(result.candidates)
    assert len(rows) == 7
    selected_rows = [r for r in rows if r["marginality_selected"]]
    non_selected_rows = [r for r in rows if not r["marginality_selected"]]
    assert {r["bidder_id"] for r in selected_rows} == {"b1", "b2", "b3"}
    assert len(non_selected_rows) == 4
    required_columns = {
        "scenario", "mechanism", "arm", "round", "bidder_id", "scope_rule",
        "marginality_score", "marginality_rank", "marginality_selected",
        "marginality_reasons", "current_allocation", "current_demand",
        "recent_events", "best_losing_bundle", "best_losing_bundle_reported_value",
    }
    assert required_columns.issubset(rows[0].keys())
    # A losing bidder outside the top-3 (b6) should carry its best losing
    # bundle diagnostics.
    b6_row = next(r for r in rows if r["bidder_id"] == "b6")
    assert b6_row["best_losing_bundle"] != ""


def test_run_summary_includes_candidate_and_selected_bidder_counts():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    _ranked, selected = rank_and_select_marginal_bidders(scores, max_bidders=3)
    candidates = [
        type("C", (), {"marginality_selected": bidder_id in selected})()
        for bidder_id in bidder_ids
    ]
    summary = late_reflection_summary_fields(
        [], enabled=True, scope="allocation_marginal", max_bidders=3,
        candidates=candidates,
    )
    assert summary["late_reflection_scope"] == "allocation_marginal"
    assert summary["late_reflection_max_bidders"] == 3
    assert summary["late_reflection_candidate_bidders"] == 7
    assert summary["late_reflection_selected_bidders"] == 3


def test_marginal_scope_parse_failure_does_not_abort_others():
    bidder_ids, prov, prev, bids, events = _sealed_fixture()
    scores = sealed_marginality_scores(
        bidder_ids=bidder_ids, provisional_allocation=prov,
        previous_allocation=prev, bids_by_bidder=bids, events=events,
    )
    proxies = {}
    for bidder_id in bidder_ids:
        if bidder_id == "b2":
            proxies[bidder_id] = _make_proxy(bidder_id, ["not valid json"])
        elif bidder_id in ("b1", "b3"):
            proxies[bidder_id] = _make_proxy(
                bidder_id, [_REFLECTION_JSON, _ANSWER_JSON, '{"bundle_value": 105}']
            )
        else:
            proxies[bidder_id] = _make_proxy(bidder_id, [])
        proxies[bidder_id].set_provisional_bid(
            {atom.bundle: atom.value for atom in bids[bidder_id].atoms}
        )

    instance = AuctionInstance(
        items=["A", "B", "C", "D", "E", "F", "G"],
        bidder_ids=bidder_ids,
        valuations={b: {} for b in bidder_ids},
    )
    config = LateReflectionConfig(
        enabled=True, scope="allocation_marginal", late_reflection_max_bidders=3,
        followup="value_query", followups_per_bidder=1,
    )
    result = run_late_reflection_trigger(
        instance=instance,
        proxies_by_bidder=proxies,
        bids_by_bidder=bids,
        config=config,
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        allocation_relevant_bidders={},
        marginality_scores=scores,
        allocated_bundle_by_bidder=prov,
    )
    by_bidder = {r.bidder_id: r for r in result.records}
    assert by_bidder["b2"].parse_success is False
    assert by_bidder["b1"].parse_success is True
    assert by_bidder["b3"].parse_success is True


def test_allocation_relevant_scope_behaviour_not_regressed_end_to_end():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["b1", "b2"],
        valuations={
            "b1": {frozenset({"A"}): 100.0},
            "b2": {frozenset({"B"}): 90.0},
        },
    )
    proxy_b1 = _make_proxy("b1", [
        '{"bundle_value": 100}', _REFLECTION_JSON, _ANSWER_JSON, '{"bundle_value": 110}',
    ])
    proxy_b2 = _make_proxy("b2", [
        '{"bundle_value": 90}',
        json.dumps({
            "question": "Q2?", "reflection_mode": "bundle_comparison",
            "primary_bundle": ["B"], "suggested_followup": "value_query",
            "followup_bundles": [["B"]],
        }),
        _ANSWER_JSON, '{"bundle_value": 95}',
    ])
    adapter_b1 = LlmAuctionProxyAdapter(
        bidder_id="b1", proxy=proxy_b1, candidate_bundles=[frozenset({"A"})],
    )
    adapter_b2 = LlmAuctionProxyAdapter(
        bidder_id="b2", proxy=proxy_b2, candidate_bundles=[frozenset({"B"})],
    )
    sealed_config = ProxySealedConfig(elicitation_rounds=1, feedback_rule="allocated_bundle")
    lr_config = LateReflectionConfig(enabled=True, scope="allocation_relevant")
    trajectory = run_proxy_sealed_vcg_trajectory(
        instance, [adapter_b1, adapter_b2], sealed_config,
        late_reflection_config=lr_config, scenario_name="test",
    )
    final = trajectory[-1]
    records = final.metadata["late_reflection_records"]
    candidates = final.metadata["late_reflection_candidates"]
    assert len(records) == 2
    assert {r.bidder_id for r in records} == {"b1", "b2"}
    # allocation_relevant never produces marginality candidates -- that's
    # allocation_marginal-only diagnostic output.
    assert candidates == []


def test_clock_allocation_marginal_end_to_end_through_runner():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["c1", "c2", "c3"],
        valuations={
            "c1": {frozenset({"A"}): 100.0},
            "c2": {frozenset({"A"}): 80.0},
            "c3": {frozenset({"B"}): 50.0},
        },
    )
    dq_response = None
    proxy_c1 = _make_proxy("c1", [
        '{"bundle_value": 100}', _REFLECTION_JSON, _ANSWER_JSON, '{"bundle_value": 105}',
    ])
    proxy_c2 = _make_proxy("c2", [
        '{"bundle_value": 80}',
        json.dumps({
            "question": "Q?", "reflection_mode": "bundle_comparison",
            "primary_bundle": ["A"], "suggested_followup": "value_query",
            "followup_bundles": [["A"]],
        }),
        _ANSWER_JSON, '{"bundle_value": 78}',
    ])
    # c3 still gets its ordinary initial-bid value query (every proxy does,
    # regardless of late-reflection scope) but no late-reflection question --
    # c1/c2 outscore it under allocation_marginal.
    proxy_c3 = _make_proxy("c3", ['{"bundle_value": 50}'])
    adapters = [
        LlmAuctionProxyAdapter(bidder_id="c1", proxy=proxy_c1, candidate_bundles=[frozenset({"A"})]),
        LlmAuctionProxyAdapter(bidder_id="c2", proxy=proxy_c2, candidate_bundles=[frozenset({"A"})]),
        LlmAuctionProxyAdapter(bidder_id="c3", proxy=proxy_c3, candidate_bundles=[frozenset({"B"})]),
    ]
    lr_config = LateReflectionConfig(
        enabled=True, scope="allocation_marginal", late_reflection_max_bidders=2,
        near_clearing_threshold=1, followup="value_query", followups_per_bidder=1,
    )
    result = run_proxy_clock_experiment(
        instance, adapters,
        clock_config=ClockConfig(max_rounds=15, price_step=10.0, reserve=0.0),
        proxy_config=ProxyClockConfig(top_k=1, elicited=False),
        late_reflection_config=lr_config,
        scenario_name="test",
    )
    records = result.metadata["late_reflection_records"]
    candidates = result.metadata["late_reflection_candidates"]
    # Cap of 2 -> at most c1 and c2 (both demand A, the overdemanded good);
    # c3 (demand B, never overdemanded) should not be selected.
    assert len(records) <= 2
    assert all(r.bidder_id in ("c1", "c2") for r in records)
    assert len(candidates) == 3
    assert not any(c.bidder_id == "c3" and c.marginality_selected for c in candidates)


# ---------------------------------------------------------------------------
# E. No live calls
# ---------------------------------------------------------------------------

def test_no_live_llm_client_used_anywhere_in_this_module():
    import sys

    module_globals = vars(sys.modules[__name__])
    live_client_names = {"OpenAICompatibleLlmClient", "CachingLlmClient"}
    assert not (live_client_names & set(module_globals))
