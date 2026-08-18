"""Model client behaviour and retry policy.

Covers the mock client used throughout the suite, which returns queued
responses and records prompts so tests need no credentials, and the retry
decorator: transient failures such as rate limits are retried with back-off
while permanent 4xx errors fail immediately rather than consuming the
budget.
"""

from __future__ import annotations

import pytest

from auctionlab.llm.clients import MockLlmClient, api_retry


def test_mock_client_returns_queued_responses_and_records_prompts():
    client = MockLlmClient(responses=["first", "second"])

    assert client.complete("prompt one") == "first"
    assert client.complete("prompt two") == "second"
    assert client.prompts == ["prompt one", "prompt two"]


def test_mock_client_raises_when_no_responses_remain():
    client = MockLlmClient(responses=[])

    with pytest.raises(RuntimeError, match="No mock LLM responses remain"):
        client.complete("prompt")

    assert client.prompts == ["prompt"]


def test_api_retry_fails_fast_for_permanent_400_error():
    calls = 0

    class PermanentError(RuntimeError):
        status_code = 400

    @api_retry(max_attempts=8)
    def fail():
        nonlocal calls
        calls += 1
        raise PermanentError("invalid request")

    with pytest.raises(PermanentError):
        fail()

    assert calls == 1


def test_api_retry_still_retries_rate_limits():
    calls = 0

    class RateLimitError(RuntimeError):
        status_code = 429

    @api_retry(max_attempts=3, base_delay=0)
    def eventually_succeed():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RateLimitError("try again")
        return "ok"

    assert eventually_succeed() == "ok"
    assert calls == 3
