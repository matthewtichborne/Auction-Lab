"""The simulated person and its answer contract.

Covers value and demand queries, the prompt context each records, and the
validation applied to answers: an answer is checked against the latent
profile before acceptance, and reasoning that mentions items outside the
queried bundle is tolerated rather than treated as failure.
"""

from __future__ import annotations

import json

import pytest

from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.logging import LlmCallLogger
from auctionlab.llm.person_simulator import (
    LlmPersonSimulator,
    compare_person_answer_extraction,
)
from auctionlab.llm.schemas import LlmPersonAnswerSemanticExtraction


ITEM_DESCRIPTIONS = {
    "IPAD": "Apple iPad",
    "PENCIL": "Apple Pencil",
}


def make_simulator(client: MockLlmClient) -> LlmPersonSimulator:
    return LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
    )


def test_value_query_returns_value_and_records_contextual_prompt():
    client = MockLlmClient(
        [
            (
                '{"bundle_value": 612, "confidence": 0.8, '
                '"reasoning_summary": "iPad plus pencil bundle"}'
            )
        ]
    )
    simulator = make_simulator(client)

    value = simulator.value_query(frozenset({"IPAD", "PENCIL"}))

    assert value == 612.0
    assert len(client.calls) == 1
    assert "A technology auction." in client.calls[0]
    assert "Prefers portable creative tools." in client.calls[0]
    assert "Apple iPad" in client.calls[0]
    assert "Apple Pencil" in client.calls[0]


def test_value_query_minimal_response_binds_expected_bundle_in_log(tmp_path):
    # The preferred, minimal schema: no queried_bundle in the model response.
    # The logged parsed_response must still show the mechanism's expected
    # bundle, bound by the caller -- never left to the model to report.
    log_path = tmp_path / "calls.jsonl"
    client = MockLlmClient(
        ['{"bundle_value": 612, "confidence": 0.8, '
         '"reasoning_summary": "iPad plus pencil bundle"}']
    )
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
        logger=LlmCallLogger(log_path),
    )

    value = simulator.value_query(frozenset({"IPAD", "PENCIL"}))

    assert value == 612.0
    record = json.loads(log_path.read_text().strip())
    assert record["parsed_response"]["queried_bundle"] == ["IPAD", "PENCIL"]


def test_value_query_prompt_no_longer_requests_queried_bundle():
    client = MockLlmClient(['{"bundle_value": 500}'])
    simulator = make_simulator(client)

    simulator.value_query(frozenset({"IPAD"}))

    prompt = client.calls[0]
    header_index = prompt.index("Return JSON only in exactly this schema")
    json_body = prompt[header_index:].split("{", 1)[1].split("}", 1)[0]
    assert "queried_bundle" not in json_body
    assert "Do NOT output a queried_bundle field" in prompt


def test_value_query_reasoning_mentioning_extra_items_does_not_fail():
    # Extra item IDs mentioned only in prose (reasoning_summary) are not a
    # bundle-identity violation -- only a literal queried_bundle field is
    # validated against the expected bundle.
    client = MockLlmClient(
        [
            '{"bundle_value": 300, "confidence": 0.6, '
            '"reasoning_summary": "Redundant with RAM_64 and SSD_2TB, '
            'which this person already values highly."}'
        ]
    )
    simulator = make_simulator(client)

    value = simulator.value_query(frozenset({"IPAD", "PENCIL"}))

    assert value == 300.0


def test_value_query_accepts_matching_queried_bundle():
    simulator = make_simulator(
        MockLlmClient(
            [
                (
                    '{"queried_bundle": ["PENCIL", "IPAD"], '
                    '"bundle_value": 612}'
                )
            ]
        )
    )

    assert simulator.value_query(
        frozenset({"IPAD", "PENCIL"})
    ) == 612.0


def test_value_query_rejects_wrong_queried_bundle_without_retry():
    simulator = make_simulator(
        MockLlmClient(
            [
                (
                    '{"queried_bundle": ["IPAD", "PENCIL"], '
                    '"bundle_value": 650}'
                )
            ]
        )
    )

    with pytest.raises(ValueError, match="does not match"):
        simulator.value_query(frozenset({"IPAD"}))


def test_value_query_mock_response_echoes_exact_bundle_and_parses():
    # Mirrors the live-run failure mode: mechanism queries {GPU_AI, GPU_GAM}
    # and the mock person response must echo exactly those two item IDs.
    item_descriptions = {
        "GPU_AI": "AI-focused GPU",
        "GPU_GAM": "Gaming-focused GPU",
    }
    client = MockLlmClient(
        ['{"queried_bundle": ["GPU_AI", "GPU_GAM"], "bundle_value": 900}']
    )
    simulator = LlmPersonSimulator(
        bidder_id="enthusiast_gamer",
        scenario_description="A PC build auction.",
        person_seed="Wants a complete gaming and AI workstation.",
        item_descriptions=item_descriptions,
        client=client,
    )

    value = simulator.value_query(frozenset({"GPU_AI", "GPU_GAM"}))

    assert value == 900.0


def test_value_query_malformed_response_with_extra_item_raises_clear_error():
    # Mirrors the live-run failure: model adds RAM_64 to a two-GPU query.
    item_descriptions = {
        "GPU_AI": "AI-focused GPU",
        "GPU_GAM": "Gaming-focused GPU",
        "RAM_64": "64GB RAM kit",
    }
    client = MockLlmClient(
        [
            '{"queried_bundle": ["GPU_AI", "GPU_GAM", "RAM_64"], '
            '"bundle_value": 900}'
        ]
    )
    simulator = LlmPersonSimulator(
        bidder_id="enthusiast_gamer",
        scenario_description="A PC build auction.",
        person_seed="Wants a complete gaming and AI workstation.",
        item_descriptions=item_descriptions,
        client=client,
    )

    with pytest.raises(ValueError, match="does not match") as exc_info:
        simulator.value_query(frozenset({"GPU_AI", "GPU_GAM"}))

    message = str(exc_info.value)
    assert "expected=['GPU_AI', 'GPU_GAM']" in message
    assert "actual=['GPU_AI', 'GPU_GAM', 'RAM_64']" in message
    assert "added=['RAM_64']" in message


def test_wrong_queried_bundle_retries_and_succeeds():
    client = MockLlmClient(
        [
            (
                '{"queried_bundle": ["IPAD", "PENCIL"], '
                '"bundle_value": 650}'
            ),
            (
                '{"queried_bundle": ["IPAD"], '
                '"bundle_value": 500}'
            ),
        ]
    )
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
        max_parse_retries=1,
    )

    assert simulator.value_query(frozenset({"IPAD"})) == 500.0
    assert len(client.calls) == 2
    assert "EXPECTED_BUNDLE_ITEM_IDS = ['IPAD']" in client.calls[1]
    assert "queried_bundle" in client.calls[1]


def test_value_queries_uses_queued_responses():
    client = MockLlmClient(
        [
            '{"bundle_value": 500}',
            '{"bundle_value": 100}',
        ]
    )
    simulator = make_simulator(client)
    ipad = frozenset({"IPAD"})
    pencil = frozenset({"PENCIL"})

    values = simulator.value_queries([ipad, pencil])

    assert values == {
        ipad: 500.0,
        pencil: 100.0,
    }
    assert len(client.calls) == 2


def test_value_query_propagates_invalid_response():
    simulator = make_simulator(MockLlmClient(["not a value"]))

    with pytest.raises(ValueError):
        simulator.value_query(frozenset({"IPAD"}))


def test_successful_value_query_writes_success_log(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    client = MockLlmClient(['{"bundle_value": 612}'])
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
        logger=LlmCallLogger(log_path),
        model_name="mock-model",
    )

    assert simulator.value_query(frozenset({"IPAD"})) == 612.0

    record = json.loads(log_path.read_text().strip())
    assert record["success"] is True
    assert record["parsed_response"]["bundle_value"] == 612.0
    assert record["model"] == "mock-model"
    assert record["attempt"] == 1


def test_failed_parse_writes_failure_log_and_raises(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=MockLlmClient(["not a value"]),
        logger=LlmCallLogger(log_path),
        max_parse_retries=0,
    )

    with pytest.raises(ValueError):
        simulator.value_query(frozenset({"IPAD"}))

    record = json.loads(log_path.read_text().strip())
    assert record["success"] is False
    assert record["raw_response"] == "not a value"
    assert record["error"]
    assert record["attempt"] == 1


def test_failed_parse_then_repair_succeeds_and_logs_both_attempts(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    client = MockLlmClient(
        [
            "I think it is worth roughly two hundred dollars.",
            (
                '{"bundle_value": 200, "confidence": 0.7, '
                '"reasoning_summary": "Reformatted previous estimate"}'
            ),
        ]
    )
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
        logger=LlmCallLogger(log_path),
        max_parse_retries=1,
    )

    value = simulator.value_query(frozenset({"IPAD"}))

    assert value == 200.0
    assert len(client.calls) == 2
    assert "I think it is worth roughly two hundred dollars." in client.calls[1]
    assert '"bundle_value"' in client.calls[1]
    assert '"queried_bundle"' in client.calls[1]
    assert '"confidence"' in client.calls[1]
    assert '"reasoning_summary"' in client.calls[1]

    records = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
    ]
    assert [record["success"] for record in records] == [False, True]
    assert [record["attempt"] for record in records] == [1, 2]


def test_value_query_includes_transcript_context_when_provided():
    client = MockLlmClient(['{"bundle_value": 612}'])
    simulator = make_simulator(client)

    simulator.value_query(
        frozenset({"IPAD"}),
        transcript_context="PRIOR_PREFERENCE_QA:\nQ: foo?\nA: bar.\n",
    )

    assert "PRIOR_PREFERENCE_QA:" in client.calls[0]
    assert "Q: foo?" in client.calls[0]
    assert "A: bar." in client.calls[0]


def test_demand_query_satisfied_returns_parsed_response():
    client = MockLlmClient(['{"satisfied": true, "preferred_bundle": null}'])
    simulator = make_simulator(client)

    response = simulator.demand_query(
        frozenset({"IPAD"}), prices={"IPAD": 100.0, "PENCIL": 50.0}
    )

    assert response.satisfied is True
    assert response.preferred_bundle is None
    assert "IPAD: 100.0" in client.calls[0]


def test_demand_query_unsatisfied_returns_preferred_bundle():
    client = MockLlmClient(
        ['{"satisfied": false, "preferred_bundle": ["PENCIL"]}']
    )
    simulator = make_simulator(client)

    response = simulator.demand_query(
        frozenset({"IPAD"}), prices={"IPAD": 100.0, "PENCIL": 50.0}
    )

    assert response.satisfied is False
    assert response.preferred_bundle == ["PENCIL"]


def test_demand_query_retries_on_parse_failure():
    client = MockLlmClient(
        [
            "not json",
            '{"satisfied": true, "preferred_bundle": null}',
        ]
    )
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
        max_parse_retries=1,
    )

    response = simulator.demand_query(
        frozenset({"IPAD"}), prices={"IPAD": 100.0, "PENCIL": 50.0}
    )

    assert response.satisfied is True
    assert len(client.calls) == 2


def test_answer_question_returns_answer_text():
    client = MockLlmClient(['{"answer": "I mostly care about portability."}'])
    simulator = make_simulator(client)

    answer = simulator.answer_question("What matters most to you?")

    assert answer == "I mostly care about portability."
    assert "What matters most to you?" in client.calls[0]


def test_answer_question_writes_log(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    client = MockLlmClient(['{"answer": "Portability matters most."}'])
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="Prefers portable creative tools.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
        logger=LlmCallLogger(log_path),
    )

    simulator.answer_question("What matters most to you?")

    record = json.loads(log_path.read_text().strip())
    assert record["prompt_type"] == "nl_question"
    assert record["parsed_response"]["answer"] == "Portability matters most."


def test_answer_question_retry_uses_compact_repair_prompt_and_logs_raw_failure(
    tmp_path,
):
    log_path = tmp_path / "calls.jsonl"
    client = MockLlmClient([
        '{"answer":"truncated',
        '{"answer":"CPU_LO is a fallback for CPU_HI."}',
    ])
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="CPU_HI is primary and CPU_LO is a fallback.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
        logger=LlmCallLogger(log_path),
        max_parse_retries=1,
    )

    answer = simulator.answer_question("What alternatives would you accept?")

    assert answer == "CPU_LO is a fallback for CPU_HI."
    assert len(client.calls) == 2
    assert "previous simulated-person response" in client.calls[1]
    assert "never exceed 60 words" in client.calls[1]
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[0]["raw_response"] == '{"answer":"truncated'
    assert records[0]["response_char_count"] == len('{"answer":"truncated')
    assert records[0]["success"] is False
    assert records[1]["success"] is True


def test_answer_question_retries_when_structural_disclosure_is_omitted(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    client = MockLlmClient([
        json.dumps({
            "answer": (
                "A is my preference and B is a fallback. "
                "I am not interested in C."
            )
        }),
        json.dumps({
            "answer": (
                "I would choose at most one of A and B. "
                "I am not interested in C."
            )
        }),
    ])
    verifier = MockLlmClient([
        json.dumps({
            "positive_items": [
                {"item_id": "A", "evidence": "A is my preference"},
                {"item_id": "B", "evidence": "B is a fallback"},
            ],
            "excluded_items": [
                {"item_id": "C", "evidence": "not interested in C"},
            ],
            "substitute_groups": [],
            "complementary_groups": [],
            "budget_hint": None,
            "other_numeric_valuation_details": [],
        }),
        json.dumps({
            "positive_items": [
                {"item_id": "A", "evidence": "one of A and B"},
                {"item_id": "B", "evidence": "one of A and B"},
            ],
            "excluded_items": [
                {"item_id": "C", "evidence": "not interested in C"},
            ],
            "substitute_groups": [{
                "items": ["A", "B"],
                "acquisition_mode": "choose_one",
                "evidence": "choose at most one of A and B",
                "mode_explicitly_stated": True,
            }],
            "complementary_groups": [],
            "budget_hint": None,
            "other_numeric_valuation_details": [],
        }),
        json.dumps({
            "judgments": [{
                "items": ["A", "B"],
                "acquisition_mode": "choose_one",
                "entailed": True,
                "evidence": "choose at most one of A and B",
                "reason": "The answer explicitly limits acquisition.",
            }],
        }),
    ])
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="A is primary, B is a choose-one fallback, and C is excluded.",
        item_descriptions={"A": "Primary", "B": "Fallback", "C": "Excluded"},
        client=client,
        verifier_client=verifier,
        logger=LlmCallLogger(log_path),
        max_parse_retries=1,
        expected_interested_items={"A", "B"},
        expected_excluded_items={"C"},
        expected_substitute_groups=[{
            "items": ["A", "B"],
            "acquisition_mode": "choose_one",
        }],
        expected_complement_groups=[],
    )

    answer = simulator.answer_question("What do you want?")

    assert answer.startswith("I would choose at most one")
    assert len(client.calls) == 2
    assert len(verifier.calls) == 3
    assert "MANDATORY CORRECTIONS" in client.calls[1]
    assert "one choose-at-most-one alternative group" in client.calls[1]
    assert client.calls[1].index("MANDATORY CORRECTIONS") > client.calls[1].index(
        "PREVIOUS INVALID RESPONSE"
    )
    records = [
        json.loads(line) for line in log_path.read_text().splitlines()
    ]
    person_records = [
        record for record in records
        if record["prompt_type"] == "nl_question"
    ]
    assert [record["success"] for record in person_records] == [False, True]
    assert [row["passed"] for row in simulator.answer_verification_history] == [
        False,
        True,
    ]


def test_focused_entailment_rejects_fallback_only_choose_one_claim(tmp_path):
    person = MockLlmClient([
        json.dumps({
            "answer": "I prefer A, with B as a fallback."
        }),
        json.dumps({
            "answer": (
                "I prefer A, with B as a fallback, and I only need one "
                "from that pair."
            )
        }),
    ])
    extraction = {
        "positive_items": [
            {"item_id": "A", "evidence": "prefer A"},
            {"item_id": "B", "evidence": "B as a fallback"},
        ],
        "excluded_items": [],
        "substitute_groups": [{
            "items": ["A", "B"],
            "acquisition_mode": "choose_one",
            "evidence": "A, with B as a fallback",
            "mode_explicitly_stated": True,
        }],
        "complementary_groups": [],
        "budget_hint": None,
        "other_numeric_valuation_details": [],
    }
    verifier = MockLlmClient([
        json.dumps(extraction),
        json.dumps({
            "judgments": [{
                "items": ["A", "B"],
                "acquisition_mode": "choose_one",
                "entailed": False,
                "evidence": "",
                "reason": "Fallback language does not limit joint ownership.",
            }]
        }),
        json.dumps(extraction),
        json.dumps({
            "judgments": [{
                "items": ["A", "B"],
                "acquisition_mode": "choose_one",
                "entailed": True,
                "evidence": "I only need one from that pair",
                "reason": "The acquisition limit is explicit.",
            }]
        }),
    ])
    log_path = tmp_path / "calls.jsonl"
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A generic auction.",
        person_seed="A is preferred to B; only one is useful.",
        item_descriptions={"A": "Primary", "B": "Fallback"},
        client=person,
        verifier_client=verifier,
        logger=LlmCallLogger(log_path),
        max_parse_retries=1,
        expected_interested_items={"A", "B"},
        expected_excluded_items=set(),
        expected_substitute_groups=[{
            "items": ["A", "B"],
            "acquisition_mode": "choose_one",
        }],
        expected_complement_groups=[],
    )

    answer = simulator.answer_question("What do you want?")

    assert "only need one" in answer
    assert simulator.answer_attempt_count == 2
    assert simulator.answer_verification_history[0]["passed"] is False
    assert simulator.answer_verification_history[1]["passed"] is True
    assert "choose-at-most-one alternative group" in person.calls[1]
    records = [
        json.loads(line) for line in log_path.read_text().splitlines()
    ]
    entailment_records = [
        row for row in records
        if row["prompt_type"] == "person_answer_substitute_entailment"
    ]
    assert len(entailment_records) == 2


def test_answer_question_accepts_natural_paraphrase_when_verifier_passes():
    person = MockLlmClient([
        json.dumps({
            "answer": (
                "For drawing on the move, I'd prefer the iPad with its "
                "Pencil, and I can spend about $600 overall."
            )
        })
    ])
    verifier = MockLlmClient([
        json.dumps({
            "positive_items": [
                {"item_id": "IPAD", "evidence": "prefer the iPad"},
                {"item_id": "PENCIL", "evidence": "with its Pencil"},
            ],
            "excluded_items": [],
            "substitute_groups": [],
            "complementary_groups": [{
                "items": ["IPAD", "PENCIL"],
                "evidence": "iPad with its Pencil",
                "explicit_extra_joint_value": True,
            }],
            "budget_hint": 600,
            "other_numeric_valuation_details": [],
        })
    ])
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="Wants IPAD and PENCIL together. Overall budget $600.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=person,
        verifier_client=verifier,
        expected_interested_items={"IPAD", "PENCIL"},
        expected_excluded_items=set(),
        expected_substitute_groups=[],
        expected_complement_groups=[["IPAD", "PENCIL"]],
    )

    answer = simulator.answer_question("What are you looking for?")

    assert "iPad with its Pencil" in answer
    assert simulator.answer_attempt_count == 1
    assert simulator.last_answer_verification["passed"] is True
    assert len(simulator.answer_verification_history) == 1


def test_blind_extraction_comparison_catches_omitted_weak_fallback():
    extraction = LlmPersonAnswerSemanticExtraction.model_validate({
        "positive_items": [
            {"item_id": "CPU_HI", "evidence": "high-performance CPU"},
            {"item_id": "CPU_MID", "evidence": "mid-range fallback"},
        ],
        "excluded_items": [],
        "substitute_groups": [{
            "items": ["CPU_HI", "CPU_MID"],
            "acquisition_mode": "choose_one",
            "evidence": "step back to mid-range",
            "mode_explicitly_stated": False,
        }],
        "complementary_groups": [],
        "budget_hint": 1850,
        "other_numeric_valuation_details": [],
    })

    result = compare_person_answer_extraction(
        extraction,
        expected_interested_items={"CPU_HI", "CPU_MID", "CPU_LO"},
        expected_excluded_items={"GPU_AI"},
        expected_substitute_groups=[{
            "items": ["CPU_HI", "CPU_MID", "CPU_LO"],
            "acquisition_mode": "choose_one",
        }],
        expected_complement_groups=[],
        available_items={"CPU_HI", "CPU_MID", "CPU_LO", "GPU_AI"},
        expected_budget_hint=1850,
    )

    assert result.passed is False
    assert result.missing_positive_items == ["CPU_LO"]
    assert "CPU_LO" in result.repair_instructions
    assert result.substitute_group_issues
    assert any(
        "invented unclear substitute group" in issue
        for issue in result.substitute_group_issues
    )


def test_blind_extraction_requires_explicit_acquisition_mode():
    extraction = LlmPersonAnswerSemanticExtraction.model_validate({
        "positive_items": [
            {"item_id": "A", "evidence": "A is my priority"},
            {"item_id": "B", "evidence": "B is my fallback"},
        ],
        "excluded_items": [],
        "substitute_groups": [{
            "items": ["A", "B"],
            "acquisition_mode": "choose_one",
            "evidence": "A is my priority and B is my fallback",
            "mode_explicitly_stated": False,
        }],
        "complementary_groups": [],
        "budget_hint": None,
        "other_numeric_valuation_details": [],
    })

    result = compare_person_answer_extraction(
        extraction,
        expected_interested_items={"A", "B"},
        expected_excluded_items=set(),
        expected_substitute_groups=[{
            "items": ["A", "B"],
            "acquisition_mode": "choose_one",
        }],
        expected_complement_groups=[],
        available_items={"A", "B"},
        expected_budget_hint=None,
    )

    assert result.passed is False
    assert result.substitute_group_issues == [
        "group ['A', 'B'] has mode unclear, expected choose_one"
    ]
    assert "choose-at-most-one alternative group" in result.repair_instructions


def test_blind_extraction_repair_explains_how_to_remove_invented_group():
    extraction = LlmPersonAnswerSemanticExtraction.model_validate({
        "positive_items": [
            {"item_id": "CPU", "evidence": "lower priority"},
            {"item_id": "SSD", "evidence": "main priority"},
        ],
        "excluded_items": [],
        "substitute_groups": [{
            "items": ["CPU", "SSD"],
            "acquisition_mode": "unclear",
            "evidence": "SSD is primary; CPU is lower priority",
            "mode_explicitly_stated": False,
        }],
        "complementary_groups": [],
        "budget_hint": 225,
        "other_numeric_valuation_details": [],
    })

    result = compare_person_answer_extraction(
        extraction,
        expected_interested_items={"CPU", "SSD"},
        expected_excluded_items=set(),
        expected_substitute_groups=[],
        expected_complement_groups=[],
        available_items={"CPU", "SSD"},
        expected_budget_hint=225,
    )

    assert result.passed is False
    assert "separate independent interests" in result.repair_instructions
    assert "relative priority alone" in result.repair_instructions


def test_answer_question_retries_when_ten_good_hard_word_limit_is_exceeded():
    item_descriptions = {
        f"G{i}": f"Good {i}" for i in range(10)
    }
    too_long = " ".join(["word"] * 101)
    person = MockLlmClient([
        json.dumps({"answer": too_long}),
        json.dumps({"answer": "I mainly want Good 0."}),
    ])
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A ten-good auction.",
        person_seed="Wants G0.",
        item_descriptions=item_descriptions,
        client=person,
        max_parse_retries=1,
    )

    assert simulator.answer_question("What do you want?") == "I mainly want Good 0."
    assert simulator.first_answer_word_count == 101
    assert simulator.final_answer_word_count == 5
    assert "never exceed 100 words" in person.calls[1]


def test_answer_target_expands_for_structurally_dense_disclosure():
    item_descriptions = {f"G{i}": f"Good {i}" for i in range(10)}
    client = MockLlmClient([
        json.dumps({"answer": "A concise complete answer."})
    ])
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A ten-good auction.",
        person_seed="Dense qualitative seed.",
        item_descriptions=item_descriptions,
        client=client,
        expected_interested_items={f"G{i}" for i in range(8)},
        expected_substitute_groups=[
            {"items": ["G0", "G1"], "acquisition_mode": "choose_one"},
            {"items": ["G2", "G3"], "acquisition_mode": "choose_one"},
            {"items": ["G4", "G5"], "acquisition_mode": "choose_one"},
        ],
    )

    simulator.answer_question("What do you want?")

    assert "Aim for about 87 words" in client.calls[0]
    assert "never exceed 100 words" in client.calls[0]


def test_value_query_sets_last_prompt_and_response_summary():
    client = MockLlmClient(
        ['{"bundle_value": 500, "reasoning_summary": "iPad is core item"}']
    )
    simulator = make_simulator(client)

    simulator.value_query(frozenset({"IPAD"}))

    assert simulator._last_prompt is not None
    assert "PROPOSED_BUNDLE_ITEM_IDS" in simulator._last_prompt
    assert simulator._last_response_summary == "iPad is core item"


def test_value_query_last_prompt_updates_on_each_call():
    client = MockLlmClient(
        [
            '{"bundle_value": 500, "reasoning_summary": "iPad alone"}',
            '{"bundle_value": 100, "reasoning_summary": "Pencil alone"}',
        ]
    )
    simulator = make_simulator(client)

    simulator.value_query(frozenset({"IPAD"}))
    first_prompt = simulator._last_prompt

    simulator.value_query(frozenset({"PENCIL"}))
    second_prompt = simulator._last_prompt

    assert "IPAD" in first_prompt
    assert "PENCIL" in second_prompt
    assert first_prompt != second_prompt
    assert simulator._last_response_summary == "Pencil alone"


def test_demand_query_sets_last_prompt_and_response_summary():
    client = MockLlmClient(
        ['{"satisfied": false, "preferred_bundle": ["PENCIL"]}']
    )
    simulator = make_simulator(client)

    simulator.demand_query(frozenset({"IPAD"}), prices={"IPAD": 200.0})

    assert simulator._last_prompt is not None
    assert "PROPOSED_BUNDLE_ITEM_IDS" in simulator._last_prompt
    assert "satisfied=False" in simulator._last_response_summary
    assert "PENCIL" in simulator._last_response_summary


def test_demand_query_satisfied_records_response_summary():
    client = MockLlmClient(
        ['{"satisfied": true, "preferred_bundle": null}']
    )
    simulator = make_simulator(client)

    simulator.demand_query(frozenset({"IPAD"}), prices={"IPAD": 50.0})

    assert "satisfied=True" in simulator._last_response_summary


def test_value_query_passes_elicitation_context_to_prompt():
    client = MockLlmClient(['{"bundle_value": 400}'])
    simulator = make_simulator(client)

    simulator.value_query(
        frozenset({"IPAD"}),
        elicitation_context="This bundle is near-zero surplus at current prices.",
    )

    assert "ELICITATION_CONTEXT:" in client.calls[0]
    assert "near-zero surplus" in client.calls[0]


def test_value_query_without_elicitation_context_has_no_elicitation_section():
    client = MockLlmClient(['{"bundle_value": 400}'])
    simulator = make_simulator(client)

    simulator.value_query(frozenset({"IPAD"}))

    assert "ELICITATION_CONTEXT:" not in client.calls[0]


class _TokenAwareClient:
    """Minimal LlmClient stub that exposes token-count attributes."""

    def __init__(self, response: str, input_tokens: int, output_tokens: int):
        self._response = response
        self._last_input_tokens = input_tokens
        self._last_output_tokens = output_tokens
        self._last_total_tokens = input_tokens + output_tokens

    def complete(self, prompt: str) -> str:
        return self._response


def test_log_attempt_includes_token_counts_when_client_provides_them(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    client = _TokenAwareClient(
        '{"bundle_value": 300, "reasoning_summary": "good deal"}',
        input_tokens=120,
        output_tokens=35,
    )
    simulator = LlmPersonSimulator(
        bidder_id="b1",
        scenario_description="Test auction.",
        person_seed="Values IPAD.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
        logger=LlmCallLogger(log_path),
    )

    simulator.value_query(frozenset({"IPAD"}))

    import json
    record = json.loads(log_path.read_text().strip())
    assert record["input_tokens"] == 120
    assert record["output_tokens"] == 35
    assert record["total_tokens"] == 155


def test_log_attempt_token_counts_are_none_for_mock_client(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    client = MockLlmClient(['{"bundle_value": 300}'])
    simulator = LlmPersonSimulator(
        bidder_id="b1",
        scenario_description="Test auction.",
        person_seed="Values IPAD.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=client,
        logger=LlmCallLogger(log_path),
    )

    simulator.value_query(frozenset({"IPAD"}))

    import json
    record = json.loads(log_path.read_text().strip())
    assert record["input_tokens"] is None
    assert record["output_tokens"] is None
    assert record["total_tokens"] is None


def test_deterministic_value_query_uses_private_table_without_llm_call():
    client = MockLlmClient([])
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="A brief qualitative disclosure with a $100 ceiling.",
        item_descriptions={
            "CPU_HI": "High-performance CPU.",
            "CPU_LO": "Entry-level CPU.",
        },
        client=client,
        ground_truth_valuations={
            frozenset({"CPU_HI"}): 100.0,
            frozenset({"CPU_HI", "CPU_LO"}): 125.0,
        },
    )

    assert simulator.value_query(frozenset({"CPU_HI", "CPU_LO"})) == 125.0
    assert client.calls == []


def test_deterministic_demand_query_maximizes_true_surplus_without_llm_call():
    client = MockLlmClient([])
    simulator = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A technology auction.",
        person_seed="A brief qualitative disclosure with a $100 ceiling.",
        item_descriptions={
            "CPU_HI": "High-performance CPU.",
            "CPU_LO": "Entry-level CPU.",
        },
        client=client,
        ground_truth_valuations={
            frozenset({"CPU_HI"}): 100.0,
            frozenset({"CPU_LO"}): 70.0,
            frozenset({"CPU_HI", "CPU_LO"}): 125.0,
        },
    )

    response = simulator.demand_query(
        frozenset({"CPU_LO"}),
        prices={"CPU_HI": 20.0, "CPU_LO": 60.0},
    )

    assert response.satisfied is False
    assert response.preferred_bundle == ["CPU_HI"]
    assert client.calls == []
