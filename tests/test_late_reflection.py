from __future__ import annotations

import json

import pytest

from auctionlab.auction_types import Bundle
from auctionlab.experiments.proxy_clock_runner import (
    ProxyClockConfig,
    run_proxy_clock_experiment,
)
from auctionlab.experiments.proxy_sealed_runner import (
    ProxySealedConfig,
    run_proxy_sealed_vcg_trajectory,
)
from auctionlab.experiments.run_config import (
    late_reflection_records_to_rows,
    late_reflection_summary_fields,
)
from auctionlab.auctions.clock import ClockConfig
from auctionlab.instances.base import AuctionInstance
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.late_reflection import (
    LateReflectionConfig,
    LateReflectionRecord,
    build_late_reflection_context,
    clock_allocation_relevant_bidders,
    run_late_reflection_for_bidder,
    run_late_reflection_trigger,
    sealed_allocation_relevant_bidders,
)
from auctionlab.llm.logging import LlmCallLogger
from auctionlab.llm.parsing import (
    LateReflectionParseError,
    derive_late_reflection_followup_bundles,
    filter_late_reflection_bundles,
    parse_late_reflection_response,
    raw_response_excerpt,
)
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.prompts import build_late_reflection_prompt
from auctionlab.llm.proxies import LlmAuctionProxyAdapter, LlmInferredXorProxy
from auctionlab.proxies.base import ElicitationEvent
from auctionlab.proxies.events import LATE_REFLECTION, ProxyElicitationEvent, ProxyResponse


ITEM_DESCRIPTIONS = {"A": "Item A", "B": "Item B", "C": "Item C"}


def _make_proxy(
    bidder_id: str,
    responses: list[str],
    *,
    epsilon: float = 1.0,
    logger: LlmCallLogger | None = None,
) -> LlmInferredXorProxy:
    person = LlmPersonSimulator(
        bidder_id=bidder_id,
        scenario_description="A test auction.",
        person_seed="Values items directly.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=MockLlmClient(responses),
        logger=logger,
        model_name="mock-model",
    )
    return LlmInferredXorProxy(bidder_id=bidder_id, person=person, epsilon=epsilon)


_VALID_REFLECTION_JSON = json.dumps({
    "question": "Would you rather have just [A] or the fuller [A, B]?",
    "reason": "checking large bundle premium",
    "reflection_mode": "bundle_comparison",
    "target_type": "large_bundle_check",
    "primary_bundle": ["A"],
    "comparison_bundle": ["A", "B"],
    "target_bundles": [["A"], ["A", "B"]],
    "suggested_followup": "value_query",
    "followup_bundles": [["A"], ["A", "B"]],
})

_MARGINAL_REFLECTION_JSON = json.dumps({
    "question": "Would adding B to [A] materially improve things?",
    "reason": "testing marginal item importance",
    "reflection_mode": "marginal_item_test",
    "target_type": "large_bundle_check",
    "primary_bundle": ["A"],
    "comparison_bundle": ["A", "B"],
    "marginal_item": "B",
    "suggested_followup": "value_query",
    "followup_bundles": [["A"], ["A", "B"]],
})

_VALID_ANSWER_JSON = json.dumps({"answer": "Yes, A alone is still valuable to me."})


# ---------------------------------------------------------------------------
# event type exists and serialises cleanly
# ---------------------------------------------------------------------------

def test_late_reflection_event_type_exists_and_serialises():
    assert LATE_REFLECTION == "late_reflection"
    event = ProxyElicitationEvent(
        event_type=LATE_REFLECTION,
        bidder_id="b1",
        mechanism="sealed",
        round_idx=3,
        payload={"foo": "bar"},
    )
    assert event.event_type == LATE_REFLECTION
    response = ProxyResponse(response_type="late_reflection", payload={"ok": True})
    assert response.response_type == "late_reflection"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_rejects_invalid_scope_and_followup():
    with pytest.raises(ValueError):
        LateReflectionConfig(scope="bogus")
    with pytest.raises(ValueError):
        LateReflectionConfig(followup="bogus")
    with pytest.raises(ValueError):
        LateReflectionConfig(followups_per_bidder=-1)
    with pytest.raises(ValueError):
        LateReflectionConfig(near_clearing_threshold=-1)
    with pytest.raises(ValueError):
        LateReflectionConfig(recent_window_rounds=-1)


# ---------------------------------------------------------------------------
# 8/9. mechanism_default now resolves to value_query for BOTH mechanisms
# ---------------------------------------------------------------------------

def test_resolved_followup_mechanism_default_is_value_query_for_clock():
    cfg = LateReflectionConfig(followup="mechanism_default")
    assert cfg.resolved_followup("clock") == "value_query"


def test_resolved_followup_mechanism_default_is_value_query_for_sealed():
    cfg = LateReflectionConfig(followup="mechanism_default")
    assert cfg.resolved_followup("sealed") == "value_query"


# ---------------------------------------------------------------------------
# 10. explicit demand_query still works
# ---------------------------------------------------------------------------

def test_resolved_followup_explicit_demand_query_passes_through():
    fixed = LateReflectionConfig(followup="demand_query")
    assert fixed.resolved_followup("clock") == "demand_query"
    assert fixed.resolved_followup("sealed") == "demand_query"


# ---------------------------------------------------------------------------
# structured context construction (no mechanism / no LLM call)
# ---------------------------------------------------------------------------

def test_build_late_reflection_context_includes_required_fields():
    proxy = _make_proxy("b1", [])
    proxy.knowledge_base.add_qa("What do you want?", "I mostly want A.")
    proxy.nl_transcript.append(("What do you want?", "I mostly want A."))
    proxy.set_provisional_bid({
        frozenset({"A"}): 100.0,
        frozenset({"A", "B"}): 150.0,
    })

    recent_event = ElicitationEvent(
        mechanism="proxy_clock_vcg",
        event_type="near_tie",
        bidder_id="b1",
        bundle=frozenset({"A"}),
        round_idx=2,
        reason="near tie",
    )

    context = build_late_reflection_context(
        bidder_id="b1",
        mechanism="clock",
        round_idx=3,
        proxy=proxy,
        demanded_bundle=frozenset({"A"}),
        current_prices={"A": 10.0, "B": 5.0},
        recent_events=[recent_event],
        contested_goods={"A"},
    )

    assert context.initial_nl_summary is not None
    assert "I mostly want A." in context.initial_nl_summary
    assert context.demanded_bundle == frozenset({"A"})
    assert context.current_prices == {"A": 10.0, "B": 5.0}
    assert len(context.recent_events) == 1
    assert context.recent_events[0]["event_type"] == "near_tie"
    assert context.contested_goods == ("A",)
    assert frozenset({"A", "B"}) in [b for b, _v in context.top_reported_bundles]

    prompt_dict = context.as_prompt_dict()
    assert prompt_dict["mechanism"] == "clock"
    assert prompt_dict["demanded_bundle"] == ["A"]
    assert prompt_dict["contested_goods"] == ["A"]


def test_context_resolved_hints_flag_already_resolved_distinctions():
    # Bidder's only bundle (of size >= 3) IS the current demand, and there is
    # no near-tie bundle -- both distinctions are "resolved"/unavailable.
    proxy = _make_proxy("b1", [])
    proxy.set_provisional_bid({frozenset({"A", "B", "C"}): 100.0})
    context = build_late_reflection_context(
        bidder_id="b1",
        mechanism="clock",
        round_idx=1,
        proxy=proxy,
        demanded_bundle=frozenset({"A", "B", "C"}),
    )
    hints = " ".join(context.resolved_hints)
    assert "near-tie" in hints
    assert "core subset" in hints or "moot" in hints


# ---------------------------------------------------------------------------
# 1. LlmLateReflectionResponse parses reflection_mode/primary/comparison/marginal_item
# ---------------------------------------------------------------------------

def test_parse_late_reflection_response_accepts_valid_bundle_comparison():
    parsed = parse_late_reflection_response(_VALID_REFLECTION_JSON)
    assert parsed.question
    assert parsed.reflection_mode == "bundle_comparison"
    assert parsed.target_type == "large_bundle_check"
    assert parsed.primary_bundle == ["A"]
    assert parsed.comparison_bundle == ["A", "B"]
    assert parsed.marginal_item is None
    assert parsed.suggested_followup == "value_query"
    assert parsed.followup_bundles == [["A"], ["A", "B"]]


def test_parse_late_reflection_response_accepts_marginal_item_test():
    parsed = parse_late_reflection_response(_MARGINAL_REFLECTION_JSON)
    assert parsed.reflection_mode == "marginal_item_test"
    assert parsed.primary_bundle == ["A"]
    assert parsed.comparison_bundle == ["A", "B"]
    assert parsed.marginal_item == "B"


# ---------------------------------------------------------------------------
# 2/3 (this round). reflection_mode is recoverable, not fatal -- "strict
# prompt, forgiving parser": an unrecognised/missing reflection_mode, an
# unrecognised target_type/suggested_followup, and malformed JSON with no
# recoverable question are all handled without ever hard-failing on
# reflection_mode alone.
# ---------------------------------------------------------------------------

def test_parse_late_reflection_response_recovers_bad_reflection_mode():
    bad = json.dumps({
        "question": "Do you want A?",
        "reflection_mode": "not_a_real_mode",
        "target_type": "other",
        "suggested_followup": "none",
    })
    parsed = parse_late_reflection_response(bad)
    assert parsed.reflection_mode == "bundle_comparison"
    assert parsed.reflection_mode_inferred is True


def test_parse_late_reflection_response_recovers_missing_reflection_mode():
    bad = json.dumps({
        "question": "Do you want A?",
        "target_type": "other",
        "suggested_followup": "none",
    })
    parsed = parse_late_reflection_response(bad)
    assert parsed.reflection_mode == "bundle_comparison"
    assert parsed.reflection_mode_inferred is True


def test_parse_late_reflection_response_rejects_empty_question():
    bad = json.dumps({
        "question": "",
        "reflection_mode": "bundle_comparison",
        "target_type": "other",
        "suggested_followup": "none",
    })
    with pytest.raises(ValueError):
        parse_late_reflection_response(bad)


def test_parse_late_reflection_response_keeps_unrecognised_target_type_and_followup():
    # target_type/suggested_followup are informational (never drive control
    # flow), so an unrecognised value is kept as-is rather than rejected --
    # only reflection_mode has real recovery logic behind it.
    raw = json.dumps({
        "question": "Do you want A?",
        "reflection_mode": "bundle_comparison",
        "target_type": "not_a_real_category",
        "suggested_followup": "not_a_real_followup",
    })
    parsed = parse_late_reflection_response(raw)
    assert parsed.target_type == "not_a_real_category"
    assert parsed.suggested_followup == "not_a_real_followup"


def test_parse_late_reflection_response_rejects_malformed_json():
    with pytest.raises(ValueError):
        parse_late_reflection_response("not json at all")


# ---------------------------------------------------------------------------
# 3. followup_bundles derived from primary_bundle/comparison_bundle when omitted
# ---------------------------------------------------------------------------

def test_derive_late_reflection_followup_bundles_from_comparison_pair():
    parsed = parse_late_reflection_response(json.dumps({
        "question": "Core [A] or fuller [A, B]?",
        "reflection_mode": "bundle_comparison",
        "target_type": "large_bundle_check",
        "primary_bundle": ["A"],
        "comparison_bundle": ["A", "B"],
        "suggested_followup": "value_query",
        # followup_bundles deliberately omitted.
    }))
    derived = derive_late_reflection_followup_bundles(parsed)
    assert derived == [["A"], ["A", "B"]]


def test_filter_late_reflection_bundles_derives_when_omitted_after_filtering():
    parsed = parse_late_reflection_response(json.dumps({
        "question": "Core [A] or fuller [A, Z]?",
        "reflection_mode": "bundle_comparison",
        "target_type": "large_bundle_check",
        "primary_bundle": ["A"],
        "comparison_bundle": ["A", "Z"],
    }))
    filtered = filter_late_reflection_bundles(parsed, {"A", "B"})
    # "Z" is unknown and dropped; the derived pair reflects the filtered bundles.
    assert filtered.comparison_bundle == ["A"]
    assert filtered.followup_bundles == [["A"]]


def test_filter_late_reflection_bundles_drops_unknown_items_from_target_bundles():
    parsed = parse_late_reflection_response(json.dumps({
        "question": "Do you want A and Z?",
        "reflection_mode": "bundle_comparison",
        "target_type": "other",
        "target_bundles": [["A", "Z"]],
        "suggested_followup": "value_query",
        "followup_bundles": [["A"]],
    }))
    filtered = filter_late_reflection_bundles(parsed, {"A", "B"})
    assert filtered.target_bundles == [["A"]]
    assert filtered.followup_bundles == [["A"]]


# ---------------------------------------------------------------------------
# 6/7. prompt content: avoid generic/resolved questions, good examples
# ---------------------------------------------------------------------------

def test_prompt_instructs_to_avoid_generic_and_resolved_questions():
    prompt = build_late_reflection_prompt(
        scenario_description="A test scenario.",
        item_descriptions=ITEM_DESCRIPTIONS,
        context={"mechanism": "sealed", "round_idx": 1},
    )
    assert "generic" in prompt
    assert "already resolved" in prompt
    assert "reflection_mode" in prompt
    assert "bundle_comparison" in prompt
    assert "marginal_item_test" in prompt


def test_prompt_includes_one_concise_good_and_bad_example():
    prompt = build_late_reflection_prompt(
        scenario_description="A test scenario.",
        item_descriptions=ITEM_DESCRIPTIONS,
        context={"mechanism": "clock", "round_idx": 1},
    )
    assert "Good example" in prompt
    assert "core" in prompt and "subset" in prompt
    assert "Bad example" in prompt
    assert "what else do you care about" in prompt


def test_prompt_says_return_only_valid_json_no_markdown():
    prompt = build_late_reflection_prompt(
        scenario_description="A test scenario.",
        item_descriptions=ITEM_DESCRIPTIONS,
        context={"mechanism": "sealed", "round_idx": 1},
    )
    assert "Return only valid JSON" in prompt
    assert "Do not wrap it in markdown" in prompt


def test_prompt_has_concise_examples_not_the_full_long_set():
    # The earlier iteration of this prompt included 3 fully-worked examples
    # (large-bundle-vs-core, marginal item, substitute/near-tie) plus 4 bad
    # examples -- this revision keeps exactly one of each to reduce prompt
    # size and token pressure on the reflection call's own budget.
    prompt = build_late_reflection_prompt(
        scenario_description="A test scenario.",
        item_descriptions=ITEM_DESCRIPTIONS,
        context={"mechanism": "clock", "round_idx": 1},
    )
    assert prompt.count("Good example") == 1
    assert prompt.count("Bad example") == 1
    # The old substitute/near-tie worked example is gone entirely.
    assert "CPU_LO" not in prompt
    assert len(prompt) < 3500


def test_prompt_surfaces_resolved_hints_from_context():
    prompt = build_late_reflection_prompt(
        scenario_description="A test scenario.",
        item_descriptions=ITEM_DESCRIPTIONS,
        context={
            "mechanism": "clock",
            "round_idx": 1,
            "resolved_hints": ["no near-tie bundle currently observed"],
        },
    )
    assert "no near-tie bundle currently observed" in prompt
    assert "do NOT ask about these" in prompt


# ---------------------------------------------------------------------------
# allocation relevance scope selection
# ---------------------------------------------------------------------------

def test_sealed_allocation_relevant_selects_bidders_with_feedback():
    events = [
        ElicitationEvent(
            mechanism="proxy_sealed_vcg",
            event_type="allocated_bundle",
            bidder_id="b1",
            bundle=frozenset({"A"}),
        ),
        ElicitationEvent(
            mechanism="proxy_sealed_vcg",
            event_type="lost_interested_bundle",
            bidder_id="b2",
            bundle=frozenset({"B"}),
        ),
    ]
    relevant = sealed_allocation_relevant_bidders(events)
    assert set(relevant) == {"b1", "b2"}
    assert relevant["b1"] == "sealed_feedback:allocated_bundle"
    assert relevant["b2"] == "sealed_feedback:lost_interested_bundle"
    assert "b3" not in relevant


def test_clock_allocation_relevant_selects_recent_events_and_contested_demand():
    relevant = clock_allocation_relevant_bidders(
        bidder_ids=["b1", "b2", "b3"],
        recent_event_bidders={"b1"},
        current_demand_by_bidder={
            "b1": frozenset({"A"}),
            "b2": frozenset({"B"}),
            "b3": frozenset({"C"}),
        },
        recently_contested_goods={"B"},
    )
    assert relevant["b1"] == "recent_clock_event"
    assert relevant["b2"] == "contested_good_in_current_demand"
    assert "b3" not in relevant


def test_all_bidders_scope_selects_every_bidder():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["b1", "b2"],
        valuations={
            "b1": {frozenset({"A"}): 100.0},
            "b2": {frozenset({"B"}): 80.0},
        },
    )
    proxy_b1 = _make_proxy("b1", [])
    proxy_b1.set_provisional_bid({frozenset({"A"}): 100.0})
    proxy_b2 = _make_proxy("b2", [])
    proxy_b2.set_provisional_bid({frozenset({"B"}): 80.0})
    proxies_by_bidder = {"b1": proxy_b1, "b2": proxy_b2}

    config = LateReflectionConfig(enabled=True, scope="all_bidders", followup="none")
    records = run_late_reflection_trigger(
        instance=instance,
        proxies_by_bidder=proxies_by_bidder,
        bids_by_bidder={"b1": proxy_b1._cached_bid, "b2": proxy_b2._cached_bid},
        config=config,
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        allocation_relevant_bidders={},
    ).records
    assert {r.bidder_id for r in records} == {"b1", "b2"}
    assert all(r.scope_rule == "all_bidders" for r in records)
    assert all(r.allocation_relevant_reason == "all_bidders_scope" for r in records)


# ---------------------------------------------------------------------------
# 4. followup_bundles capped by followups_per_bidder
# ---------------------------------------------------------------------------

def test_followups_per_bidder_caps_bundles():
    reflection_multi = json.dumps({
        "question": "Which bundle matters most?",
        "reflection_mode": "bundle_comparison",
        "target_type": "missing_bundle",
        "suggested_followup": "value_query",
        "followup_bundles": [["A"], ["B"], ["A", "B"]],
    })
    proxy = _make_proxy("b1", [
        reflection_multi,
        _VALID_ANSWER_JSON,
        '{"bundle_value": 60}',
    ])
    proxy.set_provisional_bid({
        frozenset({"A"}): 50.0,
        frozenset({"B"}): 40.0,
        frozenset({"A", "B"}): 100.0,
    })
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    config = LateReflectionConfig(followups_per_bidder=1)
    records = run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="value_query",
        prices=None,
        instance=None,
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )
    assert len(records) == 1
    assert records[0].followup_bundle == frozenset({"A"})
    assert records[0].followup_bundle_rank == 1


# ---------------------------------------------------------------------------
# 5. marginal_item_test with bundle B and B-plus-item yields correct pair
# ---------------------------------------------------------------------------

def test_marginal_item_test_yields_correct_followup_pair():
    proxy = _make_proxy("b1", [
        _MARGINAL_REFLECTION_JSON,
        _VALID_ANSWER_JSON,
        '{"bundle_value": 55}',   # for [A]
        '{"bundle_value": 95}',   # for [A, B]
    ])
    proxy.set_provisional_bid({
        frozenset({"A"}): 50.0,
        frozenset({"A", "B"}): 90.0,
    })
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    config = LateReflectionConfig(followup="value_query", followups_per_bidder=2)
    records = run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="value_query",
        prices=None,
        instance=None,
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )
    assert len(records) == 2
    bundles = {r.followup_bundle for r in records}
    assert bundles == {frozenset({"A"}), frozenset({"A", "B"})}
    for rec in records:
        assert rec.reflection_mode == "marginal_item_test"
        assert rec.primary_bundle == frozenset({"A"})
        assert rec.comparison_bundle == frozenset({"A", "B"})
        assert rec.marginal_item == "B"
        assert rec.comparison_pair_available is True


# ---------------------------------------------------------------------------
# 12. multiple follow-up bundles -> multiple rows, followup_bundle_rank
# ---------------------------------------------------------------------------

def test_multiple_followups_produce_ranked_rows():
    proxy = _make_proxy("b1", [
        _VALID_REFLECTION_JSON,
        _VALID_ANSWER_JSON,
        '{"bundle_value": 55}',
        '{"bundle_value": 95}',
    ])
    proxy.set_provisional_bid({
        frozenset({"A"}): 50.0,
        frozenset({"A", "B"}): 90.0,
    })
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    config = LateReflectionConfig(followup="value_query", followups_per_bidder=2)
    records = run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="value_query",
        prices=None,
        instance=None,
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )
    assert len(records) == 2
    ranks = sorted(r.followup_bundle_rank for r in records)
    assert ranks == [1, 2]
    rank_by_bundle = {r.followup_bundle: r.followup_bundle_rank for r in records}
    assert rank_by_bundle[frozenset({"A"})] == 1
    assert rank_by_bundle[frozenset({"A", "B"})] == 2


# ---------------------------------------------------------------------------
# value-query and demand-query follow-up correctness (unchanged from v1)
# ---------------------------------------------------------------------------

def test_value_query_followup_records_old_new_and_pricing_error():
    proxy = _make_proxy("b1", [
        json.dumps({
            "question": "Just [A] or the fuller bundle?",
            "reflection_mode": "bundle_comparison",
            "target_type": "large_bundle_check",
            "primary_bundle": ["A"],
            "suggested_followup": "value_query",
            "followup_bundles": [["A"]],
        }),
        _VALID_ANSWER_JSON,
        '{"bundle_value": 120}',
    ])
    proxy.set_provisional_bid({frozenset({"A"}): 90.0})
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["b1"],
        valuations={"b1": {frozenset({"A"}): 125.0}},
    )
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    config = LateReflectionConfig(followup="value_query")
    records = run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="value_query",
        prices=None,
        instance=instance,
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )
    assert len(records) == 1
    rec = records[0]
    assert rec.parse_success is True
    assert rec.actual_followup_type == "value_query"
    assert rec.old_reported_value == 90.0
    assert rec.new_reported_value == 120.0
    assert rec.true_value == 125.0
    assert rec.old_abs_error == pytest.approx(35.0)
    assert rec.new_abs_error == pytest.approx(5.0)
    assert rec.pricing_error_improved is True
    assert rec.absolute_correction == pytest.approx(30.0)


def test_demand_query_followup_uses_existing_dq_path_for_clock():
    reflection = json.dumps({
        "question": "Are you satisfied with A at these prices?",
        "reflection_mode": "bundle_comparison",
        "target_type": "final_allocation_acceptability",
        "primary_bundle": ["A"],
        "suggested_followup": "demand_query",
        "followup_bundles": [["A"]],
    })
    proxy = _make_proxy("b1", [
        reflection,
        _VALID_ANSWER_JSON,
        '{"satisfied": true, "preferred_bundle": null}',
    ])
    proxy.set_provisional_bid({frozenset({"A"}): 90.0})
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="clock", round_idx=4, proxy=proxy,
        current_prices={"A": 50.0, "B": 10.0},
    )
    # Explicit demand_query request (not mechanism_default).
    config = LateReflectionConfig(followup="demand_query")
    records = run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="demand_query",
        prices={"A": 50.0, "B": 10.0},
        instance=None,
        bidder_id="b1",
        scenario="s",
        arm="proxy_clock_top_1",
        mechanism="clock",
        round_idx=4,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )
    assert len(records) == 1
    assert records[0].actual_followup_type == "demand_query"
    assert records[0].demand_changed is False


def test_demand_query_followup_falls_back_to_value_query_for_sealed():
    reflection = json.dumps({
        "question": "Are you satisfied with A?",
        "reflection_mode": "bundle_comparison",
        "target_type": "final_allocation_acceptability",
        "primary_bundle": ["A"],
        "suggested_followup": "demand_query",
        "followup_bundles": [["A"]],
    })
    proxy = _make_proxy("b1", [
        reflection,
        _VALID_ANSWER_JSON,
        '{"bundle_value": 95}',
    ])
    proxy.set_provisional_bid({frozenset({"A"}): 90.0})
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    config = LateReflectionConfig(followup="demand_query")
    records = run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="demand_query",
        prices=None,
        instance=None,
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )
    assert len(records) == 1
    assert records[0].actual_followup_type == "value_query"
    assert records[0].new_reported_value == 95.0


# ---------------------------------------------------------------------------
# 13/14. pairwise pricing-error summary fields
# ---------------------------------------------------------------------------

def test_pairwise_pricing_error_fields_computed_when_both_true_values_available():
    proxy = _make_proxy("b1", [
        _VALID_REFLECTION_JSON,
        _VALID_ANSWER_JSON,
        '{"bundle_value": 60}',    # new value for [A]     (true 50)
        '{"bundle_value": 130}',   # new value for [A, B]  (true 120)
    ])
    proxy.set_provisional_bid({
        frozenset({"A"}): 40.0,       # old error |40-50|=10
        frozenset({"A", "B"}): 150.0,  # old error |150-120|=30
    })
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["b1"],
        valuations={
            "b1": {
                frozenset({"A"}): 50.0,
                frozenset({"A", "B"}): 120.0,
            },
        },
    )
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    config = LateReflectionConfig(followup="value_query", followups_per_bidder=2)
    records = run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="value_query",
        prices=None,
        instance=instance,
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )
    assert len(records) == 2
    # old abs errors: |40-50|=10, |150-120|=30 -> sum=40
    # new abs errors: |60-50|=10, |130-120|=10 -> sum=20
    for rec in records:
        assert rec.pair_old_abs_error_sum == pytest.approx(40.0)
        assert rec.pair_new_abs_error_sum == pytest.approx(20.0)
        assert rec.pair_pricing_error_improved is True


def test_pairwise_pricing_error_fields_blank_without_true_values():
    proxy = _make_proxy("b1", [
        _VALID_REFLECTION_JSON,
        _VALID_ANSWER_JSON,
        '{"bundle_value": 60}',
        '{"bundle_value": 130}',
    ])
    proxy.set_provisional_bid({
        frozenset({"A"}): 40.0,
        frozenset({"A", "B"}): 150.0,
    })
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    config = LateReflectionConfig(followup="value_query", followups_per_bidder=2)
    records = run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="value_query",
        prices=None,
        instance=None,  # no ground truth available
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )
    assert len(records) == 2
    for rec in records:
        assert rec.pair_old_abs_error_sum is None
        assert rec.pair_new_abs_error_sum is None
        assert rec.pair_pricing_error_improved is None


def test_pairwise_pricing_error_fields_blank_with_only_one_bundle_queried():
    proxy = _make_proxy("b1", [
        _VALID_REFLECTION_JSON,
        _VALID_ANSWER_JSON,
        '{"bundle_value": 60}',
    ])
    proxy.set_provisional_bid({
        frozenset({"A"}): 40.0,
        frozenset({"A", "B"}): 150.0,
    })
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["b1"],
        valuations={
            "b1": {frozenset({"A"}): 50.0, frozenset({"A", "B"}): 120.0},
        },
    )
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    # Cap at 1 -> only the primary bundle gets queried, comparison_bundle does not.
    config = LateReflectionConfig(followup="value_query", followups_per_bidder=1)
    records = run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="value_query",
        prices=None,
        instance=instance,
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )
    assert len(records) == 1
    assert records[0].pair_old_abs_error_sum is None
    assert records[0].pair_pricing_error_improved is None


# ---------------------------------------------------------------------------
# 15. one bidder's failure does not abort the rest
# ---------------------------------------------------------------------------

def test_bidder_failure_does_not_abort_the_others():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["b1", "b2"],
        valuations={
            "b1": {frozenset({"A"}): 100.0},
            "b2": {frozenset({"B"}): 80.0},
        },
    )
    # b1's client has no queued responses -> raises on .complete().
    proxy_b1 = _make_proxy("b1", [])
    proxy_b1.set_provisional_bid({frozenset({"A"}): 100.0})
    # b2's client has a full valid sequence -> succeeds.
    proxy_b2 = _make_proxy("b2", [
        json.dumps({
            "question": "Do you still want B?",
            "reflection_mode": "bundle_comparison",
            "target_type": "other",
            "primary_bundle": ["B"],
            "suggested_followup": "none",
        }),
        json.dumps({"answer": "Yes."}),
    ])
    proxy_b2.set_provisional_bid({frozenset({"B"}): 80.0})

    config = LateReflectionConfig(enabled=True, scope="all_bidders", followup="none")
    records = run_late_reflection_trigger(
        instance=instance,
        proxies_by_bidder={"b1": proxy_b1, "b2": proxy_b2},
        bids_by_bidder={"b1": proxy_b1._cached_bid, "b2": proxy_b2._cached_bid},
        config=config,
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        allocation_relevant_bidders={},
    ).records
    by_bidder = {r.bidder_id: r for r in records}
    assert set(by_bidder) == {"b1", "b2"}
    assert by_bidder["b1"].parse_success is False
    assert by_bidder["b1"].error_message
    assert by_bidder["b2"].parse_success is True
    assert by_bidder["b2"].question == "Do you still want B?"


# ---------------------------------------------------------------------------
# 11. late_reflection_records rows include the new comparison columns
# ---------------------------------------------------------------------------

def test_late_reflection_records_csv_has_required_columns():
    record = LateReflectionRecord(
        scenario="s1",
        mechanism="sealed",
        arm="proxy_sealed_allocated_bundle",
        round_idx=2,
        bidder_id="b1",
        trigger_reason="sealed_pre_final_round",
        scope_rule="allocation_relevant",
        allocation_relevant_reason="sealed_feedback:allocated_bundle",
        question="Do you still want A?",
        person_response="Yes.",
        parse_success=True,
        reflection_mode="bundle_comparison",
        target_type="large_bundle_check",
        primary_bundle=frozenset({"A"}),
        comparison_bundle=frozenset({"A", "B"}),
        marginal_item="B",
        comparison_pair_available=True,
        target_bundles=[frozenset({"A"})],
        suggested_followup="value_query",
        actual_followup_type="value_query",
        followup_bundle=frozenset({"A"}),
        followup_bundle_rank=1,
        old_reported_value=90.0,
        new_reported_value=100.0,
        true_value=100.0,
        absolute_correction=10.0,
        signed_correction=10.0,
        old_abs_error=10.0,
        new_abs_error=0.0,
        pricing_error_improved=True,
        pair_old_abs_error_sum=15.0,
        pair_new_abs_error_sum=5.0,
        pair_pricing_error_improved=True,
        allocation_before={"b1": frozenset({"A"})},
        allocation_after={"b1": frozenset({"A"})},
        allocation_changed_after_reflection=False,
        true_welfare_before=100.0,
        true_welfare_after=100.0,
        welfare_delta_after_reflection=0.0,
        tokens_in=5,
        tokens_out=7,
    )
    rows = late_reflection_records_to_rows([record])
    assert len(rows) == 1
    row = rows[0]

    required_columns = {
        "scenario", "mechanism", "arm", "round", "bidder_id", "trigger_reason",
        "scope_rule", "allocation_relevant_reason", "question", "person_response",
        "parse_success", "reflection_mode", "target_type", "primary_bundle",
        "comparison_bundle", "marginal_item", "comparison_pair_available",
        "target_bundles", "suggested_followup", "actual_followup_type",
        "followup_bundle", "followup_bundle_rank", "old_reported_value",
        "new_reported_value", "true_value", "absolute_correction",
        "signed_correction", "old_abs_error", "new_abs_error",
        "old_signed_error", "new_signed_error", "pricing_error_improved",
        "pair_old_abs_error_sum", "pair_new_abs_error_sum",
        "pair_pricing_error_improved", "pair_old_signed_error_sum",
        "pair_new_signed_error_sum",
        "demand_before", "demand_after", "demand_changed", "allocation_before",
        "allocation_after", "allocation_changed_after_reflection",
        "true_welfare_before", "true_welfare_after",
        "welfare_delta_after_reflection", "reported_welfare_before",
        "reported_welfare_after", "reported_welfare_delta_after_reflection",
        "revenue_before", "revenue_after", "surplus_before", "surplus_after",
        "tokens_in", "tokens_out", "cache_hit", "error_message",
    }
    assert required_columns.issubset(row.keys())
    assert row["reflection_mode"] == "bundle_comparison"
    assert row["primary_bundle"] == "{A}"
    assert row["comparison_bundle"] == "{A,B}"
    assert row["marginal_item"] == "B"
    assert row["comparison_pair_available"] is True
    assert row["followup_bundle_rank"] == 1
    assert row["pair_pricing_error_improved"] is True


def test_late_reflection_records_csv_writes_to_disk(tmp_path):
    from auctionlab.experiments.export import write_csv

    record = LateReflectionRecord(
        scenario="s1", mechanism="sealed", arm="a", round_idx=1,
        bidder_id="b1", trigger_reason="t", scope_rule="allocation_relevant",
        allocation_relevant_reason="r", question="Q?", person_response="A.",
        parse_success=True,
    )
    rows = late_reflection_records_to_rows([record])
    out_path = tmp_path / "curated_late_reflection_records.csv"
    write_csv(rows, out_path)
    text = out_path.read_text()
    assert "reflection_mode" in text.splitlines()[0]
    assert "b1" in text


def test_late_reflection_summary_fields_counts():
    records = [
        LateReflectionRecord(
            scenario="s", mechanism="sealed", arm="a", round_idx=1,
            bidder_id="b1", trigger_reason="t", scope_rule="all_bidders",
            allocation_relevant_reason="r", parse_success=True,
            question="q1", actual_followup_type="value_query",
            pricing_error_improved=True,
            allocation_changed_after_reflection=True,
            welfare_delta_after_reflection=5.0,
            tokens_in=10, tokens_out=20,
        ),
        LateReflectionRecord(
            scenario="s", mechanism="sealed", arm="a", round_idx=1,
            bidder_id="b2", trigger_reason="t", scope_rule="all_bidders",
            allocation_relevant_reason="r", parse_success=True,
            question="q2", actual_followup_type="demand_query",
            pricing_error_improved=False,
            allocation_changed_after_reflection=True,
            welfare_delta_after_reflection=5.0,
            tokens_in=3, tokens_out=4,
        ),
    ]
    summary = late_reflection_summary_fields(records, enabled=True)
    assert summary["late_reflection_enabled"] is True
    assert summary["late_reflection_nl_queries"] == 2
    assert summary["late_reflection_followup_vq"] == 1
    assert summary["late_reflection_followup_dq"] == 1
    assert summary["late_reflection_total_followups"] == 2
    assert summary["late_reflection_pricing_error_improvements"] == 1
    assert summary["late_reflection_allocation_changes"] == 1
    assert summary["late_reflection_welfare_delta_total"] == 5.0
    assert summary["late_reflection_token_in"] == 13
    assert summary["late_reflection_token_out"] == 24


# ---------------------------------------------------------------------------
# sealed trigger timing (fires once before final round)
# ---------------------------------------------------------------------------

def _sealed_setup(*, elicitation_rounds: int):
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["b1", "b2"],
        valuations={
            "b1": {frozenset({"A"}): 100.0},
            "b2": {frozenset({"B"}): 90.0},
        },
    )
    proxy_b1 = _make_proxy("b1", [
        '{"bundle_value": 100}',   # initial value query (round 0 bid)
        _VALID_REFLECTION_JSON,    # late reflection question (2 followups)
        _VALID_ANSWER_JSON,        # person's NL answer
        '{"bundle_value": 105}',   # followup value query for [A]
        '{"bundle_value": 150}',   # followup value query for [A, B]
    ])
    proxy_b2 = _make_proxy("b2", [
        '{"bundle_value": 90}',
        json.dumps({
            "question": "Just [B] or the fuller set?",
            "reflection_mode": "bundle_comparison",
            "target_type": "large_bundle_check",
            "primary_bundle": ["B"],
            "comparison_bundle": ["A", "B"],
            "suggested_followup": "value_query",
            "followup_bundles": [["B"], ["A", "B"]],
        }),
        json.dumps({"answer": "Yes, B is still important."}),
        '{"bundle_value": 95}',
        '{"bundle_value": 140}',
    ])
    adapter_b1 = LlmAuctionProxyAdapter(
        bidder_id="b1", proxy=proxy_b1, candidate_bundles=[frozenset({"A"})],
    )
    adapter_b2 = LlmAuctionProxyAdapter(
        bidder_id="b2", proxy=proxy_b2, candidate_bundles=[frozenset({"B"})],
    )
    config = ProxySealedConfig(
        elicitation_rounds=elicitation_rounds,
        feedback_rule="allocated_bundle",
    )
    return instance, [adapter_b1, adapter_b2], config


def test_sealed_late_reflection_fires_once_before_final_round_with_two_followups():
    instance, proxies, config = _sealed_setup(elicitation_rounds=2)
    lr_config = LateReflectionConfig(
        enabled=True, scope="allocation_relevant", followups_per_bidder=2,
    )
    trajectory = run_proxy_sealed_vcg_trajectory(
        instance, proxies, config,
        late_reflection_config=lr_config,
        scenario_name="test_scenario",
    )
    final = trajectory[-1]
    records = final.metadata["late_reflection_records"]
    # 2 bidders x 2 followup bundles each = 4 rows.
    assert len(records) == 4
    assert {r.bidder_id for r in records} == {"b1", "b2"}
    assert all(r.trigger_reason == "sealed_pre_final_round" for r in records)
    assert all(r.round_idx == config.elicitation_rounds for r in records)
    assert all(r.actual_followup_type == "value_query" for r in records)
    ranks_by_bidder: dict[str, list[int]] = {}
    for rec in records:
        ranks_by_bidder.setdefault(rec.bidder_id, []).append(rec.followup_bundle_rank)
    assert sorted(ranks_by_bidder["b1"]) == [1, 2]
    assert sorted(ranks_by_bidder["b2"]) == [1, 2]
    for round_result in trajectory[:-1]:
        assert round_result.metadata.get("late_reflection_records", []) == []


def test_sealed_late_reflection_does_not_fire_when_disabled():
    instance, proxies, config = _sealed_setup(elicitation_rounds=2)
    for proxy in proxies:
        proxy.proxy.person.client.responses = proxy.proxy.person.client.responses[:1]

    trajectory = run_proxy_sealed_vcg_trajectory(
        instance, proxies, config, late_reflection_config=None,
    )
    final = trajectory[-1]
    assert final.metadata.get("late_reflection_records", []) == []


# ---------------------------------------------------------------------------
# clock trigger timing: mechanism_default now uses value_query
# ---------------------------------------------------------------------------

def _clock_proxy(bidder_id: str, bundle: Bundle, value: float, responses: list[str]):
    proxy = _make_proxy(bidder_id, responses)
    return LlmAuctionProxyAdapter(
        bidder_id=bidder_id, proxy=proxy, candidate_bundles=[bundle],
    )


def test_clock_late_reflection_mechanism_default_uses_value_query():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["b1", "b2", "b3"],
        valuations={
            "b1": {frozenset({"A"}): 100.0},
            "b2": {frozenset({"A"}): 80.0},
            "b3": {frozenset({"B"}): 50.0},
        },
    )
    b1 = _clock_proxy("b1", frozenset({"A"}), 100.0, [
        '{"bundle_value": 100}',
        json.dumps({
            "question": "Still want A over the alternative?",
            "reflection_mode": "bundle_comparison",
            "target_type": "final_allocation_acceptability",
            "primary_bundle": ["A"],
            "suggested_followup": "value_query",
            "followup_bundles": [["A"]],
        }),
        _VALID_ANSWER_JSON,
        '{"bundle_value": 105}',
    ])
    b2 = _clock_proxy("b2", frozenset({"A"}), 80.0, [
        '{"bundle_value": 80}',
        json.dumps({
            "question": "Still want A over the alternative?",
            "reflection_mode": "bundle_comparison",
            "target_type": "final_allocation_acceptability",
            "primary_bundle": ["A"],
            "suggested_followup": "value_query",
            "followup_bundles": [["A"]],
        }),
        _VALID_ANSWER_JSON,
        '{"bundle_value": 78}',
    ])
    b3 = _clock_proxy("b3", frozenset({"B"}), 50.0, [
        '{"bundle_value": 50}',
        json.dumps({
            "question": "Still want B over the alternative?",
            "reflection_mode": "bundle_comparison",
            "target_type": "final_allocation_acceptability",
            "primary_bundle": ["B"],
            "suggested_followup": "value_query",
            "followup_bundles": [["B"]],
        }),
        _VALID_ANSWER_JSON,
        '{"bundle_value": 52}',
    ])

    lr_config = LateReflectionConfig(
        enabled=True, scope="all_bidders", near_clearing_threshold=1,
        followup="mechanism_default",
    )
    result = run_proxy_clock_experiment(
        instance,
        [b1, b2, b3],
        clock_config=ClockConfig(max_rounds=15, price_step=10.0, reserve=0.0),
        proxy_config=ProxyClockConfig(top_k=1, elicited=False),
        late_reflection_config=lr_config,
        scenario_name="test",
    )
    records = result.metadata["late_reflection_records"]
    assert len(records) == 3
    assert {r.bidder_id for r in records} == {"b1", "b2", "b3"}
    assert all(r.trigger_reason == "clock_near_clearing" for r in records)
    assert all(r.round_idx == 0 for r in records)
    # Core of this fix: clock's mechanism_default now dispatches value_query,
    # not demand_query.
    assert all(r.actual_followup_type == "value_query" for r in records)


def test_clock_late_reflection_does_not_fire_without_reaching_threshold():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["b1", "b2"],
        valuations={
            "b1": {frozenset({"A"}): 1000.0},
            "b2": {frozenset({"A"}): 900.0},
        },
    )
    b1 = _clock_proxy("b1", frozenset({"A"}), 1000.0, ['{"bundle_value": 1000}'])
    b2 = _clock_proxy("b2", frozenset({"A"}), 900.0, ['{"bundle_value": 900}'])

    lr_config = LateReflectionConfig(
        enabled=True, scope="all_bidders", near_clearing_threshold=0,
    )
    result = run_proxy_clock_experiment(
        instance,
        [b1, b2],
        clock_config=ClockConfig(max_rounds=2, price_step=1.0, reserve=0.0),
        proxy_config=ProxyClockConfig(top_k=1, elicited=False),
        late_reflection_config=lr_config,
        scenario_name="test",
    )
    assert result.metadata["late_reflection_records"] == []


# ---------------------------------------------------------------------------
# A.8-A.12 (this round). robustness: markdown fences, prose, trailing
# commas, bundle-as-string, invalid bundle goods, raw-excerpt logging.
# ---------------------------------------------------------------------------

def test_markdown_fenced_json_parses():
    raw = (
        "Sure, here is my answer:\n"
        "```json\n"
        '{"question": "Core [A] or fuller [A,B]?", '
        '"reflection_mode": "bundle_comparison", '
        '"primary_bundle": ["A"], "comparison_bundle": ["A", "B"]}\n'
        "```\n"
    )
    parsed = parse_late_reflection_response(raw)
    assert parsed.question == "Core [A] or fuller [A,B]?"
    assert parsed.primary_bundle == ["A"]
    assert parsed.comparison_bundle == ["A", "B"]


def test_json_with_short_prose_before_and_after_parses():
    raw = (
        "Here is the structured reflection you requested:\n"
        '{"question": "Do you still want A?", "reflection_mode": "bundle_comparison", '
        '"primary_bundle": ["A"]}\n'
        "Let me know if you need anything else."
    )
    parsed = parse_late_reflection_response(raw)
    assert parsed.question == "Do you still want A?"
    assert parsed.primary_bundle == ["A"]


def test_bundle_given_as_brace_string_parses_into_goods():
    raw = json.dumps({
        "question": "Core or fuller?",
        "primary_bundle": "{A,B}",
        "comparison_bundle": "{A}",
    })
    parsed = parse_late_reflection_response(raw)
    assert parsed.primary_bundle == ["A", "B"]
    assert parsed.comparison_bundle == ["A"]


def test_single_bundle_given_as_flat_list_of_bundles_is_treated_as_one_bundle():
    raw = json.dumps({
        "question": "Do you want A and B together?",
        "followup_bundles": ["A", "B"],
    })
    parsed = parse_late_reflection_response(raw)
    # A flat list of plain item strings is ONE bundle mistakenly given as a
    # list of bundles, not two singleton bundles.
    assert parsed.followup_bundles == [["A", "B"]]


def test_trailing_comma_is_safely_repaired():
    raw = '{"question": "Do you want A?", "primary_bundle": ["A",],}'
    parsed = parse_late_reflection_response(raw)
    assert parsed.question == "Do you want A?"
    assert parsed.primary_bundle == ["A"]


def test_invalid_bundle_goods_are_dropped_gracefully_by_filter():
    parsed = parse_late_reflection_response(json.dumps({
        "question": "Core or fuller with Z?",
        "primary_bundle": ["A"],
        "comparison_bundle": ["A", "Z"],
    }))
    filtered = filter_late_reflection_bundles(parsed, {"A", "B"})
    # "Z" is not a known good -- dropped, not left dangling or crashing.
    assert filtered.comparison_bundle == ["A"]


def test_malformed_json_raises_with_raw_excerpt_and_error_type():
    raw = "The model just said something in plain English, not JSON at all."
    with pytest.raises(LateReflectionParseError) as exc_info:
        parse_late_reflection_response(raw)
    assert exc_info.value.error_type == "invalid_json"
    assert exc_info.value.raw_excerpt == raw


def test_truncated_output_is_classified_and_excerpt_is_bounded():
    long_prefix = "x" * 600
    raw = '{"' + long_prefix  # unbalanced brace, no recoverable question
    with pytest.raises(LateReflectionParseError) as exc_info:
        parse_late_reflection_response(raw)
    assert exc_info.value.error_type == "truncated_output"
    assert len(exc_info.value.raw_excerpt) <= 520  # bounded, not the full 601 chars
    assert "truncated" in exc_info.value.raw_excerpt


def test_raw_response_excerpt_bounds_long_text():
    long_raw = "word " * 200
    excerpt = raw_response_excerpt(long_raw, max_len=50)
    assert len(excerpt) <= 50 + len("…[truncated]")
    assert excerpt.startswith("word word")


# ---------------------------------------------------------------------------
# C.16/17 (this round). parse failure at the orchestration layer: record
# fields populated, does not raise, does not abort other bidders.
# ---------------------------------------------------------------------------

def test_parse_failure_populates_record_with_error_type_and_excerpt():
    proxy = _make_proxy("b1", ["this is not json, just plain text"])
    proxy.set_provisional_bid({frozenset({"A"}): 50.0})
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    config = LateReflectionConfig()
    records = run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="value_query",
        prices=None,
        instance=None,
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )
    assert len(records) == 1
    rec = records[0]
    assert rec.parse_success is False
    assert rec.error_message
    assert rec.parse_error_type == "invalid_json"
    assert rec.raw_reflection_response_excerpt == "this is not json, just plain text"


def test_trigger_level_parse_failure_for_one_bidder_does_not_abort_others():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["b1", "b2"],
        valuations={
            "b1": {frozenset({"A"}): 100.0},
            "b2": {frozenset({"B"}): 80.0},
        },
    )
    # b1's client returns garbage text (a PARSE failure, not a call failure).
    proxy_b1 = _make_proxy("b1", ["garbled non-JSON output"])
    proxy_b1.set_provisional_bid({frozenset({"A"}): 100.0})
    proxy_b2 = _make_proxy("b2", [
        json.dumps({
            "question": "Do you still want B?",
            "reflection_mode": "bundle_comparison",
            "primary_bundle": ["B"],
            "suggested_followup": "none",
        }),
        json.dumps({"answer": "Yes."}),
    ])
    proxy_b2.set_provisional_bid({frozenset({"B"}): 80.0})

    config = LateReflectionConfig(enabled=True, scope="all_bidders", followup="none")
    records = run_late_reflection_trigger(
        instance=instance,
        proxies_by_bidder={"b1": proxy_b1, "b2": proxy_b2},
        bids_by_bidder={"b1": proxy_b1._cached_bid, "b2": proxy_b2._cached_bid},
        config=config,
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        allocation_relevant_bidders={},
    ).records
    by_bidder = {r.bidder_id: r for r in records}
    assert by_bidder["b1"].parse_success is False
    assert by_bidder["b1"].parse_error_type == "invalid_json"
    assert by_bidder["b2"].parse_success is True
    assert by_bidder["b2"].question == "Do you still want B?"


# ---------------------------------------------------------------------------
# C.18/19 (this round). old-style responses parse and still drive follow-ups.
# ---------------------------------------------------------------------------

def test_old_style_response_produces_followup_value_queries():
    # Pre-pairwise schema: question/target_type/target_bundles/
    # suggested_followup/followup_bundles only -- no reflection_mode,
    # primary_bundle, or comparison_bundle at all.
    old_style = json.dumps({
        "question": "Do you still want the core bundle or the fuller one?",
        "reason": "checking large bundle premium",
        "target_type": "large_bundle_check",
        "target_bundles": [["A"], ["A", "B"]],
        "suggested_followup": "value_query",
        "followup_bundles": [["A"], ["A", "B"]],
    })
    proxy = _make_proxy("b1", [
        old_style,
        _VALID_ANSWER_JSON,
        '{"bundle_value": 55}',
        '{"bundle_value": 95}',
    ])
    proxy.set_provisional_bid({
        frozenset({"A"}): 50.0,
        frozenset({"A", "B"}): 90.0,
    })
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    config = LateReflectionConfig(followup="value_query", followups_per_bidder=2)
    records = run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="value_query",
        prices=None,
        instance=None,
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )
    assert len(records) == 2
    assert all(r.parse_success for r in records)
    assert all(r.actual_followup_type == "value_query" for r in records)
    # ["A"] and ["A", "B"] are one item apart -> inferred as marginal_item_test.
    assert all(r.reflection_mode == "marginal_item_test" for r in records)
    assert all(r.reflection_mode_inferred is True for r in records)
    assert all(r.comparison_pair_available is True for r in records)
    values_by_bundle = {r.followup_bundle: r.new_reported_value for r in records}
    assert values_by_bundle[frozenset({"A"})] == 55.0
    assert values_by_bundle[frozenset({"A", "B"})] == 95.0
    ranks = sorted(r.followup_bundle_rank for r in records)
    assert ranks == [1, 2]


# ---------------------------------------------------------------------------
# calls.jsonl clearly identifies late_reflection calls and includes raw output
# ---------------------------------------------------------------------------

def test_reflection_call_is_logged_to_calls_jsonl_with_raw_output(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    logger = LlmCallLogger(log_path)
    proxy = _make_proxy(
        "b1",
        [_VALID_REFLECTION_JSON, _VALID_ANSWER_JSON],
        logger=logger,
    )
    proxy.set_provisional_bid({frozenset({"A"}): 50.0, frozenset({"A", "B"}): 90.0})
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    config = LateReflectionConfig(followup="none")
    run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="value_query",
        prices=None,
        instance=None,
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )

    lines = log_path.read_text().splitlines()
    entries = [json.loads(line) for line in lines]
    reflection_entries = [e for e in entries if e["prompt_type"] == "proxy_late_reflection"]
    assert len(reflection_entries) == 1
    entry = reflection_entries[0]
    assert entry["bidder_id"] == "b1"
    assert entry["success"] is True
    assert entry["raw_response"] == _VALID_REFLECTION_JSON
    assert entry["parsed_response"] is not None


def test_reflection_call_parse_failure_is_logged_to_calls_jsonl(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    logger = LlmCallLogger(log_path)
    proxy = _make_proxy("b1", ["not valid json"], logger=logger)
    proxy.set_provisional_bid({frozenset({"A"}): 50.0})
    context = build_late_reflection_context(
        bidder_id="b1", mechanism="sealed", round_idx=1, proxy=proxy,
    )
    config = LateReflectionConfig()
    run_late_reflection_for_bidder(
        proxy=proxy,
        context=context,
        config=config,
        resolved_followup="value_query",
        prices=None,
        instance=None,
        bidder_id="b1",
        scenario="s",
        arm="proxy_sealed",
        mechanism="sealed",
        round_idx=1,
        trigger_reason="test",
        scope_rule="all_bidders",
        allocation_relevant_reason="all_bidders_scope",
    )

    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    reflection_entries = [e for e in entries if e["prompt_type"] == "proxy_late_reflection"]
    assert len(reflection_entries) == 1
    assert reflection_entries[0]["success"] is False
    assert reflection_entries[0]["raw_response"] == "not valid json"
    assert reflection_entries[0]["error"]


# ---------------------------------------------------------------------------
# C.20 (this round). attempted/successful/failed reflection counts
# ---------------------------------------------------------------------------

def test_summary_fields_recover_attempted_successful_and_failed_counts():
    records = [
        LateReflectionRecord(
            scenario="s", mechanism="sealed", arm="a", round_idx=1,
            bidder_id="b1", trigger_reason="t", scope_rule="all_bidders",
            allocation_relevant_reason="r", parse_success=True, question="q1",
        ),
        LateReflectionRecord(
            scenario="s", mechanism="sealed", arm="a", round_idx=1,
            bidder_id="b2", trigger_reason="t", scope_rule="all_bidders",
            allocation_relevant_reason="r", parse_success=False,
            parse_error_type="invalid_json", error_message="boom",
        ),
        LateReflectionRecord(
            scenario="s", mechanism="sealed", arm="a", round_idx=1,
            bidder_id="b3", trigger_reason="t", scope_rule="all_bidders",
            allocation_relevant_reason="r", parse_success=False,
            parse_error_type="truncated_output", error_message="boom2",
        ),
    ]
    summary = late_reflection_summary_fields(records, enabled=True)
    assert summary["late_reflection_attempted_nl_queries"] == 3
    assert summary["late_reflection_successful_nl_queries"] == 1
    assert summary["late_reflection_parse_failures"] == 2
    # Backward-compatible alias still present and consistent.
    assert summary["late_reflection_nl_queries"] == 1

    rows = late_reflection_records_to_rows(records)
    failed_rows = [r for r in rows if r["parse_success"] is False]
    assert len(failed_rows) == 2
    assert {r["parse_error_type"] for r in failed_rows} == {
        "invalid_json", "truncated_output",
    }


# ---------------------------------------------------------------------------
# C.21 (this round). --late-reflection-max-tokens is accepted by the CLI
# ---------------------------------------------------------------------------

def test_cli_accepts_late_reflection_max_tokens_flag(monkeypatch):
    import sys

    sys.path.insert(0, "examples")
    import run_live_llm_curated_batch as cli

    monkeypatch.setattr(
        sys, "argv",
        ["run_live_llm_curated_batch.py", "--late-reflection",
         "--late-reflection-max-tokens", "1500"],
    )
    args = cli.parse_args()
    assert args.late_reflection_max_tokens == 1500


def test_cli_late_reflection_max_tokens_default_is_1000(monkeypatch):
    import sys

    sys.path.insert(0, "examples")
    import run_live_llm_curated_batch as cli

    monkeypatch.setattr(sys, "argv", ["run_live_llm_curated_batch.py"])
    args = cli.parse_args()
    assert args.late_reflection_max_tokens == 1000


def test_late_reflection_config_threads_max_tokens_and_validates():
    cfg = LateReflectionConfig(max_tokens=1500)
    assert cfg.max_tokens == 1500
    with pytest.raises(ValueError):
        LateReflectionConfig(max_tokens=0)
    with pytest.raises(ValueError):
        LateReflectionConfig(max_tokens=-1)


# ---------------------------------------------------------------------------
# 16. no live LLM/API client is invoked anywhere in this file
# ---------------------------------------------------------------------------

def test_no_live_llm_client_used_anywhere_in_this_module():
    import sys

    module_globals = vars(sys.modules[__name__])
    live_client_names = {"OpenAICompatibleLlmClient", "CachingLlmClient"}
    assert not (live_client_names & set(module_globals))
