"""Per-call records and aggregate call statistics.

Every model call is written as a JSONL record so a result can be traced back
to the exact request that produced it. Statistics are kept separately by call
type, because preparation-time inference and person-side queries are
different costs and must not be summed into a single figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass
class CallTypeStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LlmCallRecord:
    timestamp: str
    bidder_id: str | None
    prompt_type: str
    prompt: str
    raw_response: str | None
    parsed_response: Any | None
    success: bool
    error: str | None
    latency_seconds: float | None
    model: str | None = None
    provider: str | None = None
    llm_role: str | None = None
    attempt: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    cached_output_tokens: int | None = None
    cache_hit: bool | None = None
    finish_reason: str | None = None
    response_char_count: int | None = None


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _to_jsonable(value.dict())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


class LlmCallLogger:
    def __init__(self, path: str | Path, *, append: bool = True):
        self.path = Path(path)
        self._stats: dict[str, CallTypeStats] = {}
        self._mark: dict[str, CallTypeStats] = {}
        self._records: list[dict[str, Any]] = []
        if not append:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def log(self, record: LlmCallRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _to_jsonable(record)
        self._records.append(payload)

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True))
            f.write("\n")

        if record.success:
            key = record.prompt_type or "unknown"
            s = self._stats.setdefault(key, CallTypeStats())
            s.calls += 1
            s.input_tokens += record.input_tokens or 0
            s.output_tokens += record.output_tokens or 0

    def mark(self) -> None:
        """Snapshot current stats so subsequent calls to ``stats_since_mark``
        return only the delta from this point forward."""
        self._mark = {
            k: CallTypeStats(v.calls, v.input_tokens, v.output_tokens)
            for k, v in self._stats.items()
        }

    def total_stats(self) -> dict[str, CallTypeStats]:
        """Return cumulative per-prompt-type stats since logger creation.

        Unlike :meth:`stats_since_mark`, this is unaffected by ``mark()``,
        so callers that need their own independent running baseline (e.g.
        per-round token deltas nested inside a per-arm ``mark()`` window)
        can snapshot totals without disturbing the outer watermark.
        """
        return dict(self._stats)

    def total_tokens(self) -> tuple[int, int]:
        """Cumulative (input, output) tokens logged so far, ignoring ``mark()``."""
        stats = self._stats.values()
        return (
            sum(s.input_tokens for s in stats),
            sum(s.output_tokens for s in stats),
        )

    def stats_since_mark(self) -> dict[str, CallTypeStats]:
        """Return per-prompt-type stats accumulated since the last ``mark``."""
        result: dict[str, CallTypeStats] = {}
        for key in set(self._stats) | set(self._mark):
            current = self._stats.get(key, CallTypeStats())
            baseline = self._mark.get(key, CallTypeStats())
            delta = CallTypeStats(
                calls=current.calls - baseline.calls,
                input_tokens=current.input_tokens - baseline.input_tokens,
                output_tokens=current.output_tokens - baseline.output_tokens,
            )
            if delta.calls > 0:
                result[key] = delta
        return result

    def records(self) -> list[dict[str, Any]]:
        """Return JSON-safe records logged by this logger instance.

        Unlike reading :attr:`path`, this never includes rows left by an
        earlier run that reused the same append-only ``calls.jsonl`` file.
        Frozen-elicitation artefacts use this method to embed exactly the
        generation calls belonging to the current run.
        """
        return [dict(record) for record in self._records]


def call_stats_from_records(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    logical_cached_tokens: bool = False,
) -> dict[str, CallTypeStats]:
    """Reconstruct per-prompt statistics from serialized call records.

    Frozen elicitation replay makes no live initial calls, but scalability
    reporting still needs the logical cost of the opening interaction. When
    ``logical_cached_tokens`` is true, cached token counts take precedence
    over the zero live-call counts recorded during a cache replay.
    """
    stats: dict[str, CallTypeStats] = {}
    for record in records:
        if not record.get("success"):
            continue
        key = str(record.get("prompt_type") or "unknown")
        input_tokens = record.get("input_tokens")
        output_tokens = record.get("output_tokens")
        if logical_cached_tokens:
            if record.get("cached_input_tokens") is not None:
                input_tokens = record["cached_input_tokens"]
            if record.get("cached_output_tokens") is not None:
                output_tokens = record["cached_output_tokens"]
        row = stats.setdefault(key, CallTypeStats())
        row.calls += 1
        row.input_tokens += int(input_tokens or 0)
        row.output_tokens += int(output_tokens or 0)
    return stats
