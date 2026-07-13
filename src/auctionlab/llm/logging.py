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
    attempt: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


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
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stats: dict[str, CallTypeStats] = {}
        self._mark: dict[str, CallTypeStats] = {}

    def log(self, record: LlmCallRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _to_jsonable(record)

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
