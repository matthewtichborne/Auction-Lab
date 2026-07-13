from __future__ import annotations

from types import SimpleNamespace

import pytest

from auctionlab.llm.clients import OpenAICompatibleLlmClient


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAIClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(response)
        )


def make_response(content: str | None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content)
            )
        ]
    )


def test_complete_returns_content_and_sends_expected_request():
    fake = FakeOpenAIClient(make_response('{"bundle_value": 10}'))
    client = OpenAICompatibleLlmClient(
        model="test-model",
        _client=fake,
    )

    result = client.complete("hello")

    assert result == '{"bundle_value": 10}'
    assert fake.chat.completions.calls == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.0,
        }
    ]


def test_complete_includes_max_tokens_when_configured():
    fake = FakeOpenAIClient(make_response("response"))
    client = OpenAICompatibleLlmClient(
        model="test-model",
        max_tokens=300,
        _client=fake,
    )

    client.complete("hello")

    assert fake.chat.completions.calls[0]["max_tokens"] == 300


def test_complete_rejects_response_without_choices():
    fake = FakeOpenAIClient(SimpleNamespace(choices=[]))
    client = OpenAICompatibleLlmClient(
        model="test-model",
        _client=fake,
    )

    with pytest.raises(RuntimeError, match="no choices"):
        client.complete("hello")


@pytest.mark.parametrize("content", [None, "", "   "])
def test_complete_rejects_empty_message_content(content):
    fake = FakeOpenAIClient(make_response(content))
    client = OpenAICompatibleLlmClient(
        model="test-model",
        _client=fake,
    )

    with pytest.raises(RuntimeError, match="no message content"):
        client.complete("hello")


def test_for_ollama_uses_expected_connection_settings():
    client = OpenAICompatibleLlmClient.for_ollama()

    assert client.model == "llama3.1:8b"
    assert client.base_url == "http://localhost:11434/v1"
    assert client.api_key == "ollama"
