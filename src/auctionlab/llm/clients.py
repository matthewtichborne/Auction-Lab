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
    calls where transient rate-limit and timeout errors are common.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= max_attempts:
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
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float | None = 60.0
    extra_body: dict[str, Any] | None = None
    _client: Any | None = field(default=None, repr=False)
    _last_input_tokens: int | None = field(default=None, init=False, repr=False)
    _last_output_tokens: int | None = field(default=None, init=False, repr=False)
    _last_total_tokens: int | None = field(default=None, init=False, repr=False)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

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
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens
        if self.extra_body is not None:
            request_kwargs["extra_body"] = self.extra_body

        response = self._get_client().chat.completions.create(
            **request_kwargs
        )
        choices = response.choices
        if not choices:
            raise RuntimeError("OpenAI-compatible response contained no choices")

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
        else:
            self._last_input_tokens = None
            self._last_output_tokens = None
            self._last_total_tokens = None

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
        temperature: float = 0.0,
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
        return cls(
            model=model,
            base_url=base_url,
            api_key=resolved_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_body,
        )

