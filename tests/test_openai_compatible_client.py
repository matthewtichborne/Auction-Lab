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
                message=SimpleNamespace(content=content),
                finish_reason="length",
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
    assert client._last_finish_reason == "length"
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


def test_complete_records_provider_usage_including_reasoning_tokens():
    response = make_response("response")
    response.usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=45,
        total_tokens=165,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
    )
    fake = FakeOpenAIClient(response)
    client = OpenAICompatibleLlmClient(
        model="test-model",
        _client=fake,
    )

    client.complete("hello")

    assert client._last_input_tokens == 120
    assert client._last_output_tokens == 45
    assert client._last_total_tokens == 165
    assert client._last_reasoning_tokens == 12


def test_gpt5_mini_uses_model_appropriate_request_fields(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    fake = FakeOpenAIClient(make_response("response"))
    client = OpenAICompatibleLlmClient.for_openai(
        model="gpt-5-mini-2025-08-07",
        max_tokens=12000,
    )
    client._client = fake

    client.complete("hello")

    request = fake.chat.completions.calls[0]
    assert request["max_completion_tokens"] == 12000
    assert request["reasoning_effort"] == "low"
    assert "max_tokens" not in request
    assert "temperature" not in request
    assert client.base_url == "https://api.openai.com/v1"
    assert client.api_key == "test-openai-key"


def test_anthropic_factory_uses_compatibility_endpoint(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    fake = FakeOpenAIClient(make_response("response"))
    client = OpenAICompatibleLlmClient.for_anthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.0,
        max_tokens=12000,
    )
    client._client = fake

    client.complete("hello")

    request = fake.chat.completions.calls[0]
    assert request["max_tokens"] == 12000
    assert request["temperature"] == 0.0
    assert "max_completion_tokens" not in request
    assert "reasoning_effort" not in request
    assert client.base_url == "https://api.anthropic.com/v1/"
    assert client.api_key == "test-anthropic-key"


@pytest.mark.parametrize(
    "model",
    ["gemini-3.5-flash-lite", "gemini-3.6-flash"],
)
def test_new_gemini_models_omit_deprecated_temperature(monkeypatch, model):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    fake = FakeOpenAIClient(make_response("response"))
    client = OpenAICompatibleLlmClient.for_gemini(
        model=model,
        temperature=0.0,
        max_tokens=12000,
    )
    client._client = fake

    client.complete("hello")

    request = fake.chat.completions.calls[0]
    assert request["max_tokens"] == 12000
    assert "temperature" not in request
    assert client.temperature is None


def test_gpt41_mini_keeps_temperature_and_omits_reasoning(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    fake = FakeOpenAIClient(make_response("response"))
    client = OpenAICompatibleLlmClient.for_openai(
        model="gpt-4.1-mini-2025-04-14",
        temperature=0.0,
        max_tokens=12000,
    )
    client._client = fake

    client.complete("hello")

    request = fake.chat.completions.calls[0]
    assert request["temperature"] == 0.0
    assert request["max_completion_tokens"] == 12000
    assert "reasoning_effort" not in request


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
