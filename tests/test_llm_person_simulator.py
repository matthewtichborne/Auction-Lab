from __future__ import annotations

import json

import pytest

from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.logging import LlmCallLogger
from auctionlab.llm.person_simulator import LlmPersonSimulator


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
