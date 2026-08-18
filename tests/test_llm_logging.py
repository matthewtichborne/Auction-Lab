"""Call-record logging.

Covers JSONL record writing, parent-directory creation, appending across
calls, starting a fresh run without inheriting stale records, and
serialisation of structured Pydantic responses.
"""

from __future__ import annotations

import json

from auctionlab.llm.logging import (
    LlmCallLogger,
    LlmCallRecord,
    call_stats_from_records,
)
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


def test_logger_can_start_a_fresh_run_without_stale_records(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text('{"stale": true}\n', encoding="utf-8")

    logger = LlmCallLogger(path, append=False)

    assert path.read_text(encoding="utf-8") == ""
    logger.log(make_record({"bundle_value": 30}))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["parsed_response"]["bundle_value"] == 30


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


def test_call_stats_from_records_uses_logical_cached_tokens():
    stats = call_stats_from_records(
        [
            {
                "prompt_type": "nl_question",
                "success": True,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 700,
                "cached_output_tokens": 120,
            },
            {
                "prompt_type": "nl_question",
                "success": False,
                "input_tokens": 999,
                "output_tokens": 999,
            },
        ],
        logical_cached_tokens=True,
    )

    assert stats["nl_question"].calls == 1
    assert stats["nl_question"].input_tokens == 700
    assert stats["nl_question"].output_tokens == 120
