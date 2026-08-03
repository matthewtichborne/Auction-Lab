"""Deterministic tests for interest-map derivation and candidate generation."""

from __future__ import annotations

import json

import pytest

from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.interest_map import (
    InterestMapDerivationError,
    derive_interest_map,
    generate_candidate_bundles_from_interest_map,
    interest_map_accuracy,
    interest_map_candidate_counts,
    normalise_interest_map,
)
from auctionlab.llm.logging import LlmCallLogger
from auctionlab.llm.schemas import LlmInterestMap, LlmSubstituteGroup
from auctionlab.llm.parsing import parse_interest_map_response


def _sg(
    items: list[str],
    mode: str = "choose_one",
    evidence: str = "Explicit test evidence.",
) -> LlmSubstituteGroup:
    return LlmSubstituteGroup(
        items=items,
        acquisition_mode=mode,
        evidence=evidence,
    )


def _map(**overrides) -> LlmInterestMap:
    values = {
        "interested_items": [],
        "excluded_items": [],
        "complementary_groups": [],
        "substitute_groups": [],
        "budget_hint": None,
        "reasoning": "test",
    }
    values.update(overrides)
    return LlmInterestMap(**values)


def _derive(client, **overrides):
    kwargs = {
        "client": client,
        "scenario_description": "A generic auction.",
        "item_descriptions": {"A": "Item A", "B": "Item B"},
        "nl_question": "What do you want?",
        "nl_answer": "I want A.",
        "bidder_id": "bidder_1",
        "scenario_id": "scenario_1",
    }
    kwargs.update(overrides)
    return derive_interest_map(**kwargs)


def test_parse_failure_raise_policy_exhausts_retries():
    client = MockLlmClient(["not json"] * 3)
    with pytest.warns(UserWarning), pytest.raises(
        InterestMapDerivationError,
        match=r"bidder_id='bidder_1'.*scenario_id='scenario_1'.*attempts=3",
    ) as caught:
        _derive(client, failure_policy="raise")
    assert len(client.calls) == 3
    assert isinstance(caught.value.original_exception, ValueError)


def test_new_interest_map_parser_discards_complement_without_true_evidence():
    parsed = parse_interest_map_response(
        json.dumps({
            "interested_items": ["A", "B"],
            "excluded_items": [],
            "complementary_groups": [["A", "B"]],
            "complementary_group_evidence": [{
                "items": ["A", "B"],
                "evidence": "my ideal setup",
                "explicit_extra_joint_value": False,
            }],
            "substitute_groups": [],
            "budget_hint": None,
            "reasoning": "Preferred setup only.",
        }),
        {"A", "B"},
    )

    assert parsed.complementary_groups == []
    assert parsed.complementary_group_evidence == []


def test_new_interest_map_parser_retains_explicit_joint_value_evidence():
    parsed = parse_interest_map_response(
        json.dumps({
            "interested_items": ["A", "B"],
            "excluded_items": [],
            "complementary_groups": [["A", "B"]],
            "complementary_group_evidence": [{
                "items": ["B", "A"],
                "evidence": "worth more together than separately",
                "explicit_extra_joint_value": True,
            }],
            "substitute_groups": [],
            "budget_hint": None,
            "reasoning": "Explicit joint value.",
        }),
        {"A", "B"},
    )

    assert parsed.complementary_groups == [["A", "B"]]
    assert parsed.complementary_group_evidence[0].items == ["A", "B"]


def test_parser_downgrades_substitute_mode_without_explicit_evidence():
    parsed = parse_interest_map_response(
        json.dumps({
            "interested_items": ["A", "B"],
            "excluded_items": [],
            "complementary_groups": [],
            "substitute_groups": [{
                "items": ["A", "B"],
                "acquisition_mode": "choose_one",
                "evidence": "A is preferred and B is a fallback",
                "mode_explicitly_stated": False,
            }],
            "budget_hint": None,
            "reasoning": "The answer ranks the alternatives.",
        }),
        {"A", "B"},
    )

    assert parsed.substitute_groups[0].acquisition_mode == "unclear"
    assert parsed.substitute_groups[0].mode_explicitly_stated is False


def test_complement_entailment_second_pass_rejects_ideal_stack(tmp_path):
    interest_response = json.dumps({
        "interested_items": ["A", "B"],
        "excluded_items": [],
        "complementary_groups": [["A", "B"]],
        "complementary_group_evidence": [{
            "items": ["A", "B"],
            "evidence": "my ideal stack",
            "explicit_extra_joint_value": True,
        }],
        "substitute_groups": [],
        "budget_hint": None,
        "reasoning": "Proposed ideal stack.",
    })
    entailment_response = json.dumps({
        "judgments": [{
            "items": ["A", "B"],
            "entailed": False,
            "evidence": "my ideal stack",
            "reason": "Preference does not establish extra joint value.",
        }]
    })
    logger = LlmCallLogger(tmp_path / "calls.jsonl")

    result = _derive(
        MockLlmClient([interest_response, entailment_response]),
        nl_answer="A and B are my ideal stack.",
        logger=logger,
    )

    assert result.complementary_groups == []
    records = [
        json.loads(line)
        for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert records[-1]["prompt_type"] == (
        "proxy_interest_map_complement_entailment"
    )


def test_parse_failure_all_items_policy_is_marked_in_map_and_log(tmp_path):
    client = MockLlmClient(["not json"] * 3)
    logger = LlmCallLogger(tmp_path / "calls.jsonl")
    with pytest.warns(UserWarning):
        result = _derive(client, failure_policy="all_items", logger=logger)

    assert result.interested_items == ["A", "B"]
    assert result.substitute_groups == []
    assert result.reasoning.startswith("Fallback:")
    record = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[-1])
    diagnostics = record["parsed_response"]["_interest_map_diagnostics"]
    assert diagnostics["fallback_used"] is True
    assert "fallback_used" in diagnostics["quality_flags"]
    assert record["success"] is False


def test_semantic_conflict_retries_instead_of_silent_normalisation(tmp_path):
    conflicting = json.dumps({
        "interested_items": ["A", "B"],
        "excluded_items": ["B"],
        "complementary_groups": [],
        "substitute_groups": [{
            "items": ["A", "B"],
            "acquisition_mode": "choose_one",
            "evidence": "at most one",
        }],
        "budget_hint": None,
        "reasoning": "B was placed in both arrays.",
    })
    repaired = json.dumps({
        "interested_items": ["A", "B"],
        "excluded_items": [],
        "complementary_groups": [],
        "substitute_groups": [{
            "items": ["A", "B"],
            "acquisition_mode": "choose_one",
            "evidence": "at most one",
        }],
        "budget_hint": None,
        "reasoning": "Both are positive alternatives.",
    })
    client = MockLlmClient([conflicting, repaired])
    logger = LlmCallLogger(tmp_path / "calls.jsonl")

    with pytest.warns(UserWarning, match="both interested and excluded"):
        result = _derive(
            client,
            nl_answer="I would choose at most one of A and B.",
            failure_policy="raise",
            logger=logger,
        )

    assert result.interested_items == ["A", "B"]
    assert len(client.calls) == 2
    assert "VALIDATION ERROR TO FIX" in client.calls[1]
    records = [
        json.loads(line)
        for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert [record["success"] for record in records] == [False, True]
    assert "semantic_validation_failed" in (
        records[0]["parsed_response"]["_interest_map_diagnostics"][
            "quality_flags"
        ]
    )


def test_normalise_unknown_items_filters_or_raises_strict():
    im = _map(
        interested_items=["UNKNOWN", "A"],
        substitute_groups=[_sg(["A", "UNKNOWN"])],
    )
    flags: list[str] = []
    result = normalise_interest_map(im, {"A", "B"}, diagnostics=flags)
    assert result.interested_items == ["A"]
    assert result.substitute_groups == []
    assert "unknown_items_removed" in flags
    with pytest.raises(ValueError, match="UNKNOWN"):
        normalise_interest_map(im, {"A", "B"}, strict=True)


def test_normalise_deduplicates_and_sorts_items_and_groups():
    result = normalise_interest_map(
        _map(
            interested_items=["B", "A", "A"],
            complementary_groups=[["B", "A", "A"], ["A", "B"]],
            substitute_groups=[
                _sg(["D", "C"]),
                _sg(["C", "D"]),
                _sg(["C", "C"]),
            ],
        ),
        {"A", "B", "C", "D"},
    )
    assert result.interested_items == ["A", "B", "C", "D"]
    assert result.complementary_groups == [["A", "B"]]
    assert result.substitute_groups == [_sg(["C", "D"])]


def test_excluded_wins_and_is_removed_from_groups():
    flags: list[str] = []
    result = normalise_interest_map(
        _map(
            interested_items=["A", "B", "C"],
            excluded_items=["B"],
            complementary_groups=[["A", "B"], ["A", "C"]],
            substitute_groups=[_sg(["B", "C"])],
        ),
        {"A", "B", "C"},
        diagnostics=flags,
    )
    assert result.interested_items == ["A", "C"]
    assert result.excluded_items == ["B"]
    assert result.complementary_groups == [["A", "C"]]
    assert result.substitute_groups == []
    assert "excluded_interested_conflict_normalised" in flags


def test_group_members_are_added_to_interested_items():
    flags: list[str] = []
    result = normalise_interest_map(
        _map(interested_items=["A"], substitute_groups=[_sg(["A", "B"])]),
        {"A", "B"},
        diagnostics=flags,
    )
    assert result.interested_items == ["A", "B"]
    assert "group_members_added_to_interested" in flags


def test_complement_substitute_conflict_raises_strict():
    im = _map(
        interested_items=["A", "B"],
        complementary_groups=[["A", "B"]],
        substitute_groups=[_sg(["A", "B"])],
    )
    with pytest.raises(ValueError, match="complementary group"):
        normalise_interest_map(im, {"A", "B"}, strict=True)


def test_complement_substitute_conflict_drops_complement_and_flags():
    flags: list[str] = []
    result = normalise_interest_map(
        _map(
            interested_items=["A", "B"],
            complementary_groups=[["A", "B"]],
            substitute_groups=[_sg(["A", "B"])],
        ),
        {"A", "B"},
        diagnostics=flags,
    )
    assert result.complementary_groups == []
    assert result.substitute_groups == [_sg(["A", "B"])]
    assert "conflicting_complement_substitute_group_dropped" in flags


def test_candidates_enforce_substitutes_and_exclusions():
    with pytest.warns(
        UserWarning,
        match="excluded_interested_conflict_normalised",
    ):
        bundles = generate_candidate_bundles_from_interest_map(
            _map(
                interested_items=["A", "B", "C"],
                excluded_items=["C"],
                substitute_groups=[_sg(["A", "B"])],
            ),
            ["A", "B", "C"],
        )
    assert frozenset({"A", "B"}) not in bundles
    assert all("C" not in bundle for bundle in bundles)
    assert bundles == [frozenset({"A"}), frozenset({"B"})]


def test_candidate_priority_order_and_cap_after_prioritisation():
    im = _map(
        interested_items=["A", "B", "C", "D"],
        complementary_groups=[["B", "C"]],
        substitute_groups=[_sg(["A", "D"])],
    )
    bundles = generate_candidate_bundles_from_interest_map(im, ["A", "B", "C", "D"])
    assert bundles[0] == frozenset({"B", "C"})
    assert bundles[1:5] == [
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"D"}),
    ]
    remaining = bundles[5:]
    assert remaining == sorted(
        remaining, key=lambda bundle: (len(bundle), tuple(sorted(bundle)))
    )
    assert generate_candidate_bundles_from_interest_map(
        im, ["A", "B", "C", "D"], max_candidate_bundles=3
    ) == bundles[:3]
    assert interest_map_candidate_counts(im, ["A", "B", "C", "D"]) == (
        15,
        len(bundles),
    )


def test_broad_map_warning_reports_pre_and_post_filter_counts():
    im = _map(
        interested_items=list("ABCDEFGH"),
        substitute_groups=[_sg(["A", "B"])],
    )
    with pytest.warns(
        UserWarning,
        match=(
            r"interested_count=8.*candidate_count_before_filter=255.*"
            r"candidate_count_after_substitute_filter=191.*"
            r"substitute_group_count=1.*missed substitute relations"
        ),
    ):
        generate_candidate_bundles_from_interest_map(im, list("ABCDEFGH"))


@pytest.mark.parametrize("mode", ["can_use_multiple", "unclear"])
def test_nonexclusive_or_unclear_groups_do_not_filter_joint_bundles(mode):
    im = _map(
        interested_items=["A", "B"],
        substitute_groups=[_sg(["A", "B"], mode=mode)],
    )

    bundles = generate_candidate_bundles_from_interest_map(im, ["A", "B"])

    assert frozenset({"A", "B"}) in bundles
    assert interest_map_candidate_counts(im, ["A", "B"]) == (3, 3)


@pytest.mark.parametrize("mode", ["can_use_multiple", "unclear"])
def test_nonexclusive_or_unclear_group_does_not_delete_complement(mode):
    result = normalise_interest_map(
        _map(
            interested_items=["A", "B"],
            complementary_groups=[["A", "B"]],
            substitute_groups=[_sg(["A", "B"], mode=mode)],
        ),
        {"A", "B"},
    )

    assert result.complementary_groups == [["A", "B"]]


def test_conflicting_duplicate_modes_become_unclear_and_do_not_filter():
    flags: list[str] = []
    result = normalise_interest_map(
        _map(
            interested_items=["A", "B"],
            substitute_groups=[
                _sg(["A", "B"], mode="choose_one"),
                _sg(["B", "A"], mode="can_use_multiple"),
            ],
        ),
        {"A", "B"},
        diagnostics=flags,
    )

    assert result.substitute_groups[0].acquisition_mode == "unclear"
    assert "conflicting_substitute_modes_made_unclear" in flags
    assert frozenset({"A", "B"}) in generate_candidate_bundles_from_interest_map(
        result, ["A", "B"]
    )


def test_accuracy_reports_dangerous_false_exclusivity():
    result = interest_map_accuracy(
        _map(
            interested_items=["A", "B"],
            substitute_groups=[_sg(["A", "B"], mode="choose_one")],
        ),
        true_interested_items={"A", "B"},
        true_substitute_groups=[
            {
                "items": ["A", "B"],
                "acquisition_mode": "can_use_multiple",
            }
        ],
        available_items={"A", "B"},
    )

    assert result["item_recall"] == 1.0
    assert result["item_precision"] == 1.0
    assert result["item_f1"] == 1.0
    assert result["group_item_set_recall"] == 1.0
    assert result["mode_accuracy_on_matched_groups"] == 0.0
    assert result["dangerous_false_exclusivity_count"] == 1
    assert result["dangerous_false_exclusivity_groups"] == [["A", "B"]]


def test_accuracy_detects_partial_false_exclusivity_overlap():
    result = interest_map_accuracy(
        _map(
            interested_items=["A", "B", "C"],
            substitute_groups=[_sg(["A", "B"], mode="choose_one")],
        ),
        true_interested_items={"A", "B", "C"},
        true_substitute_groups=[
            {
                "items": ["A", "B", "C"],
                "acquisition_mode": "can_use_multiple",
            }
        ],
        available_items={"A", "B", "C"},
    )

    assert result["group_item_set_recall"] == 0.0
    assert result["dangerous_false_exclusivity_count"] == 1
    assert result["dangerous_false_exclusivity_details"] == [
        {
            "inferred_choose_one_group": ["A", "B"],
            "true_can_use_multiple_group": ["A", "B", "C"],
            "conflicting_items": ["A", "B"],
        }
    ]


def test_accuracy_scores_complements_and_candidate_coverage():
    result = interest_map_accuracy(
        _map(
            interested_items=["A", "B", "C"],
            complementary_groups=[["A", "B"]],
            substitute_groups=[_sg(["B", "C"], mode="choose_one")],
        ),
        true_interested_items={"A", "B", "C"},
        true_substitute_groups=[{
            "items": ["B", "C"],
            "acquisition_mode": "choose_one",
        }],
        true_complement_groups=[],
        available_items={"A", "B", "C"},
        nl_answer=(
            "A and B are parts of my setup. "
            "B and C are alternatives."
        ),
    )

    assert result["complement_group_precision"] == 0.0
    assert result["complement_group_recall"] == 1.0
    assert result["extra_inferred_complement_groups"] == [["A", "B"]]
    assert result["substitute_evidence_coverage"] == 1.0
    assert result["complement_evidence_coverage"] == 0.0
    assert result["candidate_set_precision"] == 1.0
    assert result["candidate_set_recall"] == 1.0


def test_accuracy_exposes_candidate_loss_from_omitted_fallback():
    available = {
        "CPU_HI", "CPU_MID", "CPU_LO",
        "GPU_HI", "GPU_VALUE",
        "RAM_64", "RAM_32", "MB",
    }
    result = interest_map_accuracy(
        _map(
            interested_items=[
                "CPU_HI", "CPU_MID",
                "GPU_HI", "GPU_VALUE",
                "RAM_64", "RAM_32", "MB",
            ],
            excluded_items=["CPU_LO"],
            substitute_groups=[
                _sg(["CPU_HI", "CPU_MID"]),
                _sg(["GPU_HI", "GPU_VALUE"]),
                _sg(["RAM_64", "RAM_32"]),
            ],
        ),
        true_interested_items=available,
        true_substitute_groups=[
            {
                "items": ["CPU_HI", "CPU_MID", "CPU_LO"],
                "acquisition_mode": "choose_one",
            },
            {
                "items": ["GPU_HI", "GPU_VALUE"],
                "acquisition_mode": "choose_one",
            },
            {
                "items": ["RAM_64", "RAM_32"],
                "acquisition_mode": "choose_one",
            },
        ],
        available_items=available,
        singleton_values={"CPU_LO": 120},
    )

    assert result["oracle_candidate_count"] == 71
    assert result["inferred_candidate_count"] == 53
    assert result["candidate_set_recall"] == pytest.approx(53 / 71)
    assert result["missed_positive_singleton_value_total"] == 120
