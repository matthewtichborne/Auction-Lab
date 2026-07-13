from __future__ import annotations

import pytest

from auctionlab.llm.clients import MockLlmClient


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
