from __future__ import annotations

import json

from auctionlab.llm.logging import LlmCallLogger, LlmCallRecord
from auctionlab.llm.schemas import LlmValueResponse


def make_record(parsed_response=None) -> LlmCallRecord:
    return LlmCallRecord(
        timestamp="2026-06-10T12:00:00+00:00",
        bidder_id="i1",
        prompt_type="value_query",
        prompt="prompt",
        raw_response='{"bundle_value": 10}',
        parsed_response=parsed_response,
        success=True,
        error=None,
        latency_seconds=0.1,
        model="test-model",
        attempt=1,
    )


def test_logger_creates_parent_and_writes_jsonl_record(tmp_path):
    path = tmp_path / "nested" / "calls.jsonl"
    logger = LlmCallLogger(path)

    logger.log(make_record({"bundle_value": 10}))

    loaded = json.loads(path.read_text().strip())
    assert loaded["bidder_id"] == "i1"
    assert loaded["parsed_response"] == {"bundle_value": 10}


def test_logger_appends_multiple_records(tmp_path):
    path = tmp_path / "calls.jsonl"
    logger = LlmCallLogger(path)

    logger.log(make_record({"bundle_value": 10}))
    logger.log(make_record({"bundle_value": 20}))

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["parsed_response"]["bundle_value"] == 10
    assert json.loads(lines[1])["parsed_response"]["bundle_value"] == 20


def test_logger_serializes_pydantic_response(tmp_path):
    path = tmp_path / "calls.jsonl"
    logger = LlmCallLogger(path)

    logger.log(make_record(LlmValueResponse(bundle_value=12.5)))

    loaded = json.loads(path.read_text().strip())
    assert loaded["parsed_response"]["bundle_value"] == 12.5


def test_record_token_fields_default_to_none():
    record = make_record()
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.total_tokens is None


def test_record_with_token_counts_serialises_to_json(tmp_path):
    path = tmp_path / "calls.jsonl"
    logger = LlmCallLogger(path)
    record = LlmCallRecord(
        timestamp="2026-06-10T12:00:00+00:00",
        bidder_id="i1",
        prompt_type="value_query",
        prompt="prompt",
        raw_response='{"bundle_value": 10}',
        parsed_response=None,
        success=True,
        error=None,
        latency_seconds=0.1,
        model="test-model",
        attempt=1,
        input_tokens=150,
        output_tokens=42,
        total_tokens=192,
    )

    logger.log(record)

    payload = json.loads(path.read_text().strip())
    assert payload["input_tokens"] == 150
    assert payload["output_tokens"] == 42
    assert payload["total_tokens"] == 192
