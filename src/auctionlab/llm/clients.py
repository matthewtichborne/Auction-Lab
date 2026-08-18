"""Model provider clients and retry behaviour.

All clients share one interface so an experiment can swap provider without
touching calling code, and the mock client lets the whole test suite run
without credentials or network access. Retries use exponential back-off with
jitter and distinguish transient failures from permanent ones: retrying a
malformed request would consume budget without any prospect of success.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Protocol


class LlmClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


def api_retry(
    max_attempts: int = 8,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
):
    """Retry a function on transient errors with exponential back-off + jitter.

    Adapted from alpha-main's retry utility. Default parameters suit LLM API
    calls where transient rate-limit and timeout errors are common. Permanent
    client errors (ordinary 4xx responses other than 408/409/429) fail
    immediately rather than wasting time and quota on identical retries.
    """
    def is_non_retryable(exc: BaseException) -> bool:
        status_code = getattr(exc, "status_code", None)
        return (
            isinstance(status_code, int)
            and 400 <= status_code < 500
            and status_code not in {408, 409, 429}
        )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= max_attempts or is_non_retryable(exc):
                        raise
                    jitter = random.random() * delay
                    print(
                        f"[api_retry] {func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed ({exc!r}); retrying in {delay + jitter:.2f}s"
                    )
                    time.sleep(delay + jitter)
                    delay *= backoff
        return wrapper
    return decorator


@dataclass
class MockLlmClient:
    responses: list[str]
    prompts: list[str] = field(default_factory=list)

    @property
    def calls(self) -> list[str]:
        return self.prompts

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)

        if not self.responses:
            raise RuntimeError("No mock LLM responses remain")

        return self.responses.pop(0)


@dataclass
class OpenAICompatibleLlmClient:
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = 0.0
    max_tokens: int | None = None
    max_tokens_parameter: str = "max_tokens"
    reasoning_effort: str | None = None
    required_api_key_env: str | None = None
    timeout: float | None = 60.0
    extra_body: dict[str, Any] | None = None
    _client: Any | None = field(default=None, repr=False)
    _last_input_tokens: int | None = field(default=None, init=False, repr=False)
    _last_output_tokens: int | None = field(default=None, init=False, repr=False)
    _last_total_tokens: int | None = field(default=None, init=False, repr=False)
    _last_reasoning_tokens: int | None = field(
        default=None, init=False, repr=False
    )
    _last_finish_reason: str | None = field(default=None, init=False, repr=False)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            if self.required_api_key_env and not self.api_key:
                raise ValueError(
                    f"API key not provided. Pass api_key= or set "
                    f"{self.required_api_key_env}."
                )
            client_kwargs: dict[str, Any] = {}
            if self.base_url is not None:
                client_kwargs["base_url"] = self.base_url
            if self.api_key is not None:
                client_kwargs["api_key"] = self.api_key
            if self.timeout is not None:
                client_kwargs["timeout"] = self.timeout

            self._client = OpenAI(**client_kwargs)

        return self._client

    @api_retry()
    def complete(self, prompt: str) -> str:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            request_kwargs[self.max_tokens_parameter] = self.max_tokens
        if self.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body is not None:
            request_kwargs["extra_body"] = self.extra_body

        response = self._get_client().chat.completions.create(
            **request_kwargs
        )
        choices = response.choices
        if not choices:
            raise RuntimeError("OpenAI-compatible response contained no choices")

        self._last_finish_reason = getattr(choices[0], "finish_reason", None)
        content = choices[0].message.content
        if content is None or not str(content).strip():
            raise RuntimeError(
                "OpenAI-compatible response contained no message content"
            )

        usage = getattr(response, "usage", None)
        if usage is not None:
            self._last_input_tokens = getattr(usage, "prompt_tokens", None)
            self._last_output_tokens = getattr(usage, "completion_tokens", None)
            self._last_total_tokens = getattr(usage, "total_tokens", None)
            completion_details = getattr(
                usage, "completion_tokens_details", None
            )
            self._last_reasoning_tokens = getattr(
                completion_details, "reasoning_tokens", None
            )
        else:
            self._last_input_tokens = None
            self._last_output_tokens = None
            self._last_total_tokens = None
            self._last_reasoning_tokens = None

        return str(content)

    @classmethod
    def for_ollama(
        cls,
        model: str = "llama3.1:8b",
        *,
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = 60.0,
    ) -> OpenAICompatibleLlmClient:
        return cls(
            model=model,
            base_url=base_url,
            api_key="ollama",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    @classmethod
    def for_groq(
        cls,
        model: str = "llama-3.1-8b-instant",
        *,
        base_url: str = "https://api.groq.com/openai/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = 60.0,
    ) -> OpenAICompatibleLlmClient:
        resolved_api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Groq API key not provided. Pass api_key= or set GROQ_API_KEY."
            )
        return cls(
            model=model,
            base_url=base_url,
            api_key=resolved_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    @classmethod
    def for_gemini(
        cls,
        model: str = "gemini-2.0-flash",
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = 60.0,
        thinking_budget: int | None = None,
    ) -> OpenAICompatibleLlmClient:
        resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Gemini API key not provided. Pass api_key= or set GEMINI_API_KEY."
            )
        # The Gemini OpenAI-compatible endpoint does not expose a thinking-budget
        # field. Truncated responses caused by thinking tokens are handled by the
        # parser's "bundle_value" key regex fallback instead.
        extra_body = (
            {"thinking": {"budget_tokens": thinking_budget}}
            if thinking_budget is not None
            else None
        )
        # Gemini 3.5 Flash-Lite, 3.6 Flash, and later model releases
        # deprecate sampling parameters. Omit temperature entirely rather
        # than relying on the endpoint to ignore it (or eventually reject it).
        effective_temperature = (
            None
            if model.startswith(("gemini-3.5-", "gemini-3.6-"))
            else temperature
        )
        return cls(
            model=model,
            base_url=base_url,
            api_key=resolved_api_key,
            temperature=effective_temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_body,
        )

    @classmethod
    def for_openai(
        cls,
        model: str = "gpt-5-mini-2025-08-07",
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = 60.0,
        reasoning_effort: str | None = None,
    ) -> OpenAICompatibleLlmClient:
        """Build a first-party OpenAI client with GPT-5-safe parameters.

        GPT-5 mini uses ``max_completion_tokens`` on Chat Completions.
        Sampling temperature is omitted by default because GPT-5 reasoning
        models do not provide the same temperature-zero control as the
        non-reasoning models used elsewhere in this project.
        """
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        effective_reasoning_effort = reasoning_effort
        if (
            effective_reasoning_effort is None
            and model.startswith("gpt-5-mini")
        ):
            effective_reasoning_effort = "low"
        return cls(
            model=model,
            base_url=base_url,
            api_key=resolved_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_tokens_parameter="max_completion_tokens",
            reasoning_effort=effective_reasoning_effort,
            required_api_key_env="OPENAI_API_KEY",
            timeout=timeout,
        )

    @classmethod
    def for_anthropic(
        cls,
        model: str = "claude-haiku-4-5-20251001",
        *,
        base_url: str = "https://api.anthropic.com/v1/",
        api_key: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = 60.0,
    ) -> OpenAICompatibleLlmClient:
        """Build a Claude client through Anthropic's compatibility endpoint."""
        resolved_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        return cls(
            model=model,
            base_url=base_url,
            api_key=resolved_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_tokens_parameter="max_tokens",
            required_api_key_env="ANTHROPIC_API_KEY",
            timeout=timeout,
        )


MISTRAL_DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
"""Mistral's ``/v1/chat/completions`` endpoint is OpenAI-SDK-compatible, so
:class:`MistralLlmClient` reuses :class:`OpenAICompatibleLlmClient`'s request/
response handling wholesale rather than a bespoke client."""


@dataclass
class MistralLlmClient(OpenAICompatibleLlmClient):
    """:class:`OpenAICompatibleLlmClient` pointed at Mistral's chat-completions API.

    Unlike :meth:`OpenAICompatibleLlmClient.for_groq`/``for_gemini``, a
    missing API key is deliberately **not** raised at construction time: the
    key is resolved (``api_key=`` or ``MISTRAL_API_KEY``) in
    :meth:`for_mistral`, but only *enforced* here, in :meth:`complete`, which
    runs solely on a live call. This means a client can be built and wrapped
    in a :class:`~auctionlab.llm.cache.CachingLlmClient` with no API key at
    all and still serve ``llm-cache-mode read-only`` cache hits (which never
    reach :meth:`complete`) -- only an actual cache-miss live call raises.
    """

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError(
                "Mistral API key not provided. Pass --api-key or set "
                "MISTRAL_API_KEY. (Only required to make a live call -- a "
                "--llm-cache-mode read-only cache hit does not need one.)"
            )
        return super().complete(prompt)

    @classmethod
    def for_mistral(
        cls,
        model: str = "mistral-large-latest",
        *,
        base_url: str = MISTRAL_DEFAULT_BASE_URL,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = 60.0,
    ) -> "MistralLlmClient":
        resolved_api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        return cls(
            model=model,
            base_url=base_url,
            api_key=resolved_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
