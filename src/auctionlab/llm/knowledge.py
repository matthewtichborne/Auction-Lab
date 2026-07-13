"""Knowledge-base abstractions for proxy preference state.

A KnowledgeBase tracks what the proxy currently knows about a person's
preferences. The context_for_prompt() method serialises that knowledge into
a string suitable for inclusion in an LLM prompt; add_qa() records a new
natural-language question/answer interaction.

Implementations (in increasing sophistication):

- TranscriptKnowledgeBase: raw Q/A list; simple and token-linear.
- SummaryKnowledgeBase: rolling LLM-generated summary + recent Q/A window;
  token cost stays bounded regardless of elicitation depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from auctionlab.llm.prompts import format_summary_context, format_transcript_context

if TYPE_CHECKING:
    from auctionlab.llm.clients import LlmClient


@runtime_checkable
class KnowledgeBase(Protocol):
    """Interface for proxy knowledge about a person's preferences."""

    def context_for_prompt(self) -> str:
        """Return a string to embed in LLM prompts as preference context."""
        ...

    def add_qa(self, question: str, answer: str) -> None:
        """Record a natural-language question/answer exchange."""
        ...

    def __bool__(self) -> bool:
        """Return True if the knowledge base contains any information."""
        ...


@dataclass
class SummaryKnowledgeBase:
    """Knowledge base backed by a rolling LLM-generated summary.

    After each Q/A pair, the LLM compresses the full preference history into a
    short summary (2–5 sentences).  Only the most recent ``max_recent`` raw
    pairs are also kept in-band so very fresh information is never lost in
    translation.  Prompt-token cost is therefore O(1) in elicitation depth
    rather than O(N).

    ``client`` must be a callable LlmClient (same interface used by proxies).
    """

    client: LlmClient
    max_recent: int = 3
    _summary: str = field(default="", init=False, repr=False)
    _recent_qa: list[tuple[str, str]] = field(default_factory=list, init=False, repr=False)

    def context_for_prompt(self) -> str:
        return format_summary_context(self._summary, self._recent_qa)

    def add_qa(self, question: str, answer: str) -> None:
        from auctionlab.llm.parsing import parse_summary_update_response
        from auctionlab.llm.prompts import build_summary_update_prompt

        prompt = build_summary_update_prompt(
            current_summary=self._summary,
            question=question,
            answer=answer,
        )
        raw = self.client.complete(prompt)
        parsed = parse_summary_update_response(raw)
        self._summary = parsed.summary

        self._recent_qa.append((question, answer))
        if len(self._recent_qa) > self.max_recent:
            self._recent_qa = self._recent_qa[-self.max_recent :]

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def recent_qa(self) -> list[tuple[str, str]]:
        return list(self._recent_qa)

    def __bool__(self) -> bool:
        return bool(self._summary) or bool(self._recent_qa)


@dataclass
class TranscriptKnowledgeBase:
    """Knowledge base backed by a list of raw Q/A pairs.

    Each pair is a (question, answer) tuple recorded during proxy/person
    interaction. The knowledge base renders them as PRIOR_PREFERENCE_QA
    context for value-query prompts.

    ``_entries`` can be passed in as a reference to share the backing list
    with the proxy's ``nl_transcript`` attribute, keeping both in sync.
    """

    _entries: list[tuple[str, str]] = field(default_factory=list)

    def context_for_prompt(self) -> str:
        return format_transcript_context(self._entries)

    def add_qa(self, question: str, answer: str) -> None:
        self._entries.append((question, answer))

    @property
    def entries(self) -> list[tuple[str, str]]:
        """Read-only view of the recorded Q/A pairs."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)
