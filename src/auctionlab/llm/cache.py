"""First-class SQLite-backed caching for LLM responses.

Parameter grids repeatedly ask the same LLM questions (NL person responses,
interest-map extraction, provisional valuations, explicit value/demand
queries). This module lets large grids replay identical LLM responses
instead of re-issuing the same call, which is faster, cheaper, and removes
avoidable stochastic variation across mechanism-comparison arms.

Cache key
---------
``cache_key = sha256(canonical_json(request_envelope))`` -- the envelope
built by :func:`build_request_envelope` includes every field that could
plausibly affect the response (call type, provider, model, temperature,
max_tokens, a hash of the exact rendered prompt text, prompt/parser
versions, scenario/bidder identifiers, and bundle-related hashes). Hashing
the *rendered* prompt text is deliberate: this codebase's prompt builders
already render the scenario description, item descriptions, person seed,
and queried bundle directly into the prompt, so hashing the final prompt
text automatically captures all of those without having to separately
thread each one through the cache key.

Cache modes
-----------
- ``off``: never read or write the cache; always call the provider.
- ``read-write``: use a cached response if present, otherwise call the
  provider and write the response.
- ``read-only``: require a cached response; raise :class:`CacheMissError`
  and never call the provider on a miss.
- ``refresh``: ignore any existing cache entry, always call the provider,
  then write/overwrite the cache entry.

See ``README.md`` ("Frozen elicitation and caching") for the recommended
read-write/read-only/refresh workflow across a parameter grid.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

LlmCacheMode = str
"""One of ``"off"``, ``"read-write"``, ``"read-only"``, ``"refresh"``."""

VALID_CACHE_MODES: tuple[str, ...] = ("off", "read-write", "read-only", "refresh")

DEFAULT_CACHE_PATH = "cache/llm_responses.sqlite"

CALL_TYPE_PERSON_NL_RESPONSE = "person_nl_response"
CALL_TYPE_INTEREST_MAP = "interest_map"
CALL_TYPE_PROVISIONAL_VALUATIONS = "provisional_valuations"
CALL_TYPE_VALUE_QUERY = "value_query"
CALL_TYPE_DEMAND_QUERY = "demand_query"

KNOWN_CALL_TYPES = (
    CALL_TYPE_PERSON_NL_RESPONSE,
    CALL_TYPE_INTEREST_MAP,
    CALL_TYPE_PROVISIONAL_VALUATIONS,
    CALL_TYPE_VALUE_QUERY,
    CALL_TYPE_DEMAND_QUERY,
)


def validate_cache_mode(mode: str) -> str:
    if mode not in VALID_CACHE_MODES:
        raise ValueError(
            f"Invalid llm cache mode {mode!r}; must be one of {VALID_CACHE_MODES}"
        )
    return mode


class CacheMissError(RuntimeError):
    """Raised on a ``read-only`` cache miss. Never calls the provider."""

    def __init__(
        self,
        *,
        call_type: str,
        cache_key: str,
        bidder_id: str | None = None,
        bundle_key: str | None = None,
    ) -> None:
        self.call_type = call_type
        self.bidder_id = bidder_id
        self.bundle_key = bundle_key
        self.cache_key = cache_key
        self.cache_key_prefix = cache_key[:12]
        message = (
            f"llm cache read-only miss: call_type={call_type!r} "
            f"bidder_id={bidder_id!r} bundle_key={bundle_key!r} "
            f"cache_key_prefix={self.cache_key_prefix}..."
        )
        super().__init__(message)


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialisation: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_cache_key(request_envelope: dict[str, Any]) -> str:
    """``sha256(canonical_json(request_envelope))`` -- stable under key order."""
    return sha256_hex(canonical_json(request_envelope))


def bundle_key_str(bundle: Iterable[str] | None) -> str | None:
    """Canonical ``{item1,item2}``-style string for one bundle, or ``None``."""
    if bundle is None:
        return None
    return "{" + ",".join(sorted(bundle)) + "}"


def bundle_set_hash(bundles: Iterable[Iterable[str]] | None) -> str | None:
    """Hash of a whole candidate-bundle set, order-independent."""
    if bundles is None:
        return None
    normalized = sorted(bundle_key_str(b) or "" for b in bundles)
    return sha256_hex(canonical_json(normalized))


def build_request_envelope(
    *,
    call_type: str,
    provider: str,
    model: str,
    prompt: str | None = None,
    prompt_hash: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    prompt_version: str | None = None,
    scenario_id: str | None = None,
    bidder_id: str | None = None,
    person_seed_hash: str | None = None,
    item_list_hash: str | None = None,
    bundle_set_hash: str | None = None,
    queried_bundle: str | None = None,
    valuation_calibration: dict[str, Any] | None = None,
    parser_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical dict whose hash is the cache key.

    Every field that could plausibly affect the response is included
    explicitly (as ``None`` when not applicable) so the envelope's shape is
    stable regardless of which optional fields a given caller supplies --
    this is what makes :func:`compute_cache_key` stable under dict/kwarg
    ordering. Either ``prompt`` (hashed here) or a precomputed ``prompt_hash``
    must be supplied.
    """
    if prompt_hash is None:
        if prompt is None:
            raise ValueError("Either prompt or prompt_hash must be provided")
        prompt_hash = sha256_hex(prompt)

    envelope: dict[str, Any] = {
        "call_type": call_type,
        "provider": provider,
        "model": model,
        "prompt_hash": prompt_hash,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_version": prompt_version,
        "scenario_id": scenario_id,
        "bidder_id": bidder_id,
        "person_seed_hash": person_seed_hash,
        "item_list_hash": item_list_hash,
        "bundle_set_hash": bundle_set_hash,
        "queried_bundle": queried_bundle,
        "valuation_calibration": valuation_calibration,
        "parser_version": parser_version,
    }
    if extra:
        envelope["extra"] = extra
    return envelope


@dataclass
class CacheEntry:
    """One row of the ``llm_cache`` table."""

    cache_key: str
    call_type: str
    provider: str
    model: str
    prompt_hash: str
    request_json: str
    raw_response_text: str
    parsed_response_json: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_seconds: float | None = None
    created_at: str = ""
    code_version: str | None = None
    prompt_version: str | None = None
    scenario_id: str | None = None
    bidder_id: str | None = None
    bundle_key: str | None = None
    metadata_json: str | None = None


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    call_type TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    request_json TEXT NOT NULL,
    raw_response_text TEXT NOT NULL,
    parsed_response_json TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_seconds REAL,
    created_at TEXT NOT NULL,
    code_version TEXT,
    prompt_version TEXT,
    scenario_id TEXT,
    bidder_id TEXT,
    bundle_key TEXT,
    metadata_json TEXT
);
"""

_INDEX_SQL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_llm_cache_call_type "
    "ON llm_cache(call_type);",
    "CREATE INDEX IF NOT EXISTS idx_llm_cache_scenario_bidder "
    "ON llm_cache(scenario_id, bidder_id);",
    "CREATE INDEX IF NOT EXISTS idx_llm_cache_bundle_key "
    "ON llm_cache(bundle_key);",
)

_ENTRY_COLUMNS: tuple[str, ...] = (
    "cache_key", "call_type", "provider", "model", "prompt_hash",
    "request_json", "raw_response_text", "parsed_response_json",
    "tokens_in", "tokens_out", "latency_seconds", "created_at",
    "code_version", "prompt_version", "scenario_id", "bidder_id",
    "bundle_key", "metadata_json",
)


class LlmResponseCache:
    """SQLite-backed store for :class:`CacheEntry` rows, keyed by cache_key."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(_SCHEMA_SQL)
            for stmt in _INDEX_SQL:
                self._conn.execute(stmt)

    def get(self, cache_key: str) -> CacheEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM llm_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        return CacheEntry(**{col: row[col] for col in _ENTRY_COLUMNS})

    def put(self, entry: CacheEntry) -> None:
        values = {col: getattr(entry, col) for col in _ENTRY_COLUMNS}
        placeholders = ", ".join(f":{col}" for col in _ENTRY_COLUMNS)
        columns = ", ".join(_ENTRY_COLUMNS)
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO llm_cache ({columns}) VALUES ({placeholders})",
                values,
            )

    def delete(self, cache_key: str) -> None:
        """Remove one response, used when downstream parsing rejects it."""
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "LlmResponseCache":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass
class CacheStats:
    """Run-level accounting of cache activity, for the run-summary CSVs."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    read_only_misses: int = 0
    cached_tokens_in: int = 0
    cached_tokens_out: int = 0
    actual_live_tokens_in: int = 0
    actual_live_tokens_out: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "llm_cache_hits": self.hits,
            "llm_cache_misses": self.misses,
            "llm_cache_writes": self.writes,
            "llm_cache_read_only_misses": self.read_only_misses,
            "cached_tokens_in": self.cached_tokens_in,
            "cached_tokens_out": self.cached_tokens_out,
            "actual_live_tokens_in": self.actual_live_tokens_in,
            "actual_live_tokens_out": self.actual_live_tokens_out,
        }

    def merge(self, other: "CacheStats") -> None:
        self.hits += other.hits
        self.misses += other.misses
        self.writes += other.writes
        self.read_only_misses += other.read_only_misses
        self.cached_tokens_in += other.cached_tokens_in
        self.cached_tokens_out += other.cached_tokens_out
        self.actual_live_tokens_in += other.actual_live_tokens_in
        self.actual_live_tokens_out += other.actual_live_tokens_out


@dataclass
class LlmCallOutcome:
    """What a live provider call produced, independent of caching."""

    raw_response_text: str
    parsed_response_json: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_seconds: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class CachedResult:
    """What :func:`get_or_call` returns to its caller."""

    raw_response_text: str
    parsed_response_json: str | None
    cache_hit: bool
    cache_key: str
    tokens_in: int | None
    tokens_out: int | None
    cached_tokens_in: int | None
    cached_tokens_out: int | None
    latency_seconds: float | None


def get_or_call(
    *,
    cache: LlmResponseCache | None,
    mode: str,
    call_type: str,
    request_envelope: dict[str, Any],
    call_fn: Callable[[], LlmCallOutcome],
    scenario_id: str | None = None,
    bidder_id: str | None = None,
    bundle_key: str | None = None,
    code_version: str | None = None,
    prompt_version: str | None = None,
    metadata: dict[str, Any] | None = None,
    stats: CacheStats | None = None,
) -> CachedResult:
    """Apply cache-mode semantics around one logical LLM call.

    ``call_fn`` is only invoked when the provider actually needs to be
    called (cache miss in ``read-write``, or any call in ``refresh``/``off``).
    On a cache hit, ``tokens_in``/``tokens_out`` on the returned
    :class:`CachedResult` are ``0`` (no tokens charged this run); the
    original call's token counts are still available via
    ``cached_tokens_in``/``cached_tokens_out``.
    """
    validate_cache_mode(mode)
    cache_key = compute_cache_key(request_envelope)

    if mode == "off":
        outcome = call_fn()
        if stats is not None:
            stats.actual_live_tokens_in += outcome.tokens_in or 0
            stats.actual_live_tokens_out += outcome.tokens_out or 0
        return CachedResult(
            raw_response_text=outcome.raw_response_text,
            parsed_response_json=outcome.parsed_response_json,
            cache_hit=False,
            cache_key=cache_key,
            tokens_in=outcome.tokens_in,
            tokens_out=outcome.tokens_out,
            cached_tokens_in=None,
            cached_tokens_out=None,
            latency_seconds=outcome.latency_seconds,
        )

    if cache is None:
        raise ValueError(f"llm cache mode {mode!r} requires a cache instance")

    if mode in ("read-write", "read-only"):
        entry = cache.get(cache_key)
        if entry is not None:
            if stats is not None:
                stats.hits += 1
                stats.cached_tokens_in += entry.tokens_in or 0
                stats.cached_tokens_out += entry.tokens_out or 0
            return CachedResult(
                raw_response_text=entry.raw_response_text,
                parsed_response_json=entry.parsed_response_json,
                cache_hit=True,
                cache_key=cache_key,
                tokens_in=0,
                tokens_out=0,
                cached_tokens_in=entry.tokens_in,
                cached_tokens_out=entry.tokens_out,
                latency_seconds=0.0,
            )
        if mode == "read-only":
            if stats is not None:
                stats.misses += 1
                stats.read_only_misses += 1
            raise CacheMissError(
                call_type=call_type,
                cache_key=cache_key,
                bidder_id=bidder_id,
                bundle_key=bundle_key,
            )
        if stats is not None:
            stats.misses += 1

    # mode is "refresh", or "read-write" with a miss: call the provider and
    # write/overwrite the cache entry.
    outcome = call_fn()
    entry = CacheEntry(
        cache_key=cache_key,
        call_type=call_type,
        provider=str(request_envelope.get("provider", "")),
        model=str(request_envelope.get("model", "")),
        prompt_hash=str(request_envelope.get("prompt_hash", "")),
        request_json=canonical_json(request_envelope),
        raw_response_text=outcome.raw_response_text,
        parsed_response_json=outcome.parsed_response_json,
        tokens_in=outcome.tokens_in,
        tokens_out=outcome.tokens_out,
        latency_seconds=outcome.latency_seconds,
        created_at=current_timestamp(),
        code_version=code_version,
        prompt_version=prompt_version,
        scenario_id=scenario_id,
        bidder_id=bidder_id,
        bundle_key=bundle_key,
        metadata_json=canonical_json(metadata) if metadata is not None else None,
    )
    cache.put(entry)
    if stats is not None:
        stats.writes += 1
        stats.actual_live_tokens_in += outcome.tokens_in or 0
        stats.actual_live_tokens_out += outcome.tokens_out or 0
    return CachedResult(
        raw_response_text=outcome.raw_response_text,
        parsed_response_json=outcome.parsed_response_json,
        cache_hit=False,
        cache_key=cache_key,
        tokens_in=outcome.tokens_in,
        tokens_out=outcome.tokens_out,
        cached_tokens_in=None,
        cached_tokens_out=None,
        latency_seconds=outcome.latency_seconds,
    )


@dataclass
class CachingLlmClient:
    """Wraps any :class:`~auctionlab.llm.clients.LlmClient` with caching.

    Conforms to the plain ``complete(prompt) -> str`` protocol (so it drops
    into any existing call site unchanged), but also accepts optional
    keyword-only cache-context fields on each call -- callers that know the
    call type/bidder/bundle (see :func:`call_client`) get richer, more
    filterable cache rows; callers that don't still get correct caching
    keyed on prompt text, provider, and model alone.
    """

    inner: Any
    cache: LlmResponseCache
    mode: str
    provider: str
    model: str
    call_type: str = "generic"
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    prompt_version: str | None = None
    code_version: str | None = None
    stats: CacheStats = field(default_factory=CacheStats)
    _last_input_tokens: int | None = field(default=None, init=False, repr=False)
    _last_output_tokens: int | None = field(default=None, init=False, repr=False)
    _last_total_tokens: int | None = field(default=None, init=False, repr=False)
    _last_cache_hit: bool | None = field(default=None, init=False, repr=False)
    _last_cached_input_tokens: int | None = field(
        default=None, init=False, repr=False
    )
    _last_cached_output_tokens: int | None = field(
        default=None, init=False, repr=False
    )
    _last_finish_reason: str | None = field(default=None, init=False, repr=False)
    _last_cache_key: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_cache_mode(self.mode)

    def complete(
        self,
        prompt: str,
        *,
        call_type: str | None = None,
        scenario_id: str | None = None,
        bidder_id: str | None = None,
        bundle_key: str | None = None,
        extra_key_fields: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        resolved_call_type = call_type or self.call_type
        resolved_extra = dict(extra_key_fields or {})
        if self.reasoning_effort is not None:
            resolved_extra["reasoning_effort"] = self.reasoning_effort
        envelope = build_request_envelope(
            call_type=resolved_call_type,
            provider=self.provider,
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            prompt_version=self.prompt_version,
            scenario_id=scenario_id,
            bidder_id=bidder_id,
            queried_bundle=bundle_key,
            extra=resolved_extra or None,
        )

        def _call_fn() -> LlmCallOutcome:
            started = time.perf_counter()
            text = self.inner.complete(prompt)
            latency = time.perf_counter() - started
            return LlmCallOutcome(
                raw_response_text=text,
                tokens_in=getattr(self.inner, "_last_input_tokens", None),
                tokens_out=getattr(self.inner, "_last_output_tokens", None),
                latency_seconds=latency,
            )

        result = get_or_call(
            cache=self.cache,
            mode=self.mode,
            call_type=resolved_call_type,
            request_envelope=envelope,
            call_fn=_call_fn,
            scenario_id=scenario_id,
            bidder_id=bidder_id,
            bundle_key=bundle_key,
            code_version=self.code_version,
            prompt_version=self.prompt_version,
            metadata=metadata,
            stats=self.stats,
        )
        self._last_cache_hit = result.cache_hit
        self._last_cache_key = result.cache_key
        self._last_cached_input_tokens = result.cached_tokens_in
        self._last_cached_output_tokens = result.cached_tokens_out
        self._last_finish_reason = (
            None
            if result.cache_hit
            else getattr(self.inner, "_last_finish_reason", None)
        )
        self._last_input_tokens = result.tokens_in
        self._last_output_tokens = result.tokens_out
        self._last_total_tokens = (
            None
            if result.tokens_in is None and result.tokens_out is None
            else (result.tokens_in or 0) + (result.tokens_out or 0)
        )
        return result.raw_response_text

    def invalidate_last(self) -> None:
        """Discard the last raw response after a downstream parse failure."""
        if self._last_cache_key is not None:
            self.cache.delete(self._last_cache_key)
            self._last_cache_key = None


def call_client(client: Any, prompt: str, **cache_kwargs: Any) -> str:
    """Call ``client.complete(prompt)``, passing cache context if supported.

    Plain :class:`~auctionlab.llm.clients.LlmClient` implementations only
    accept ``prompt``; :class:`CachingLlmClient` additionally accepts the
    keyword-only cache-context fields. This lets call sites (person
    simulator, interest map, provisional valuations) pass rich cache
    context unconditionally without needing to know which kind of client
    they were given.
    """
    if isinstance(client, CachingLlmClient):
        return client.complete(prompt, **cache_kwargs)
    return client.complete(prompt)


# ---------------------------------------------------------------------------
# Run-summary aggregation helpers
# ---------------------------------------------------------------------------

CACHE_NUMERIC_SUMMARY_FIELDS: tuple[str, ...] = (
    "llm_cache_hits",
    "llm_cache_misses",
    "llm_cache_writes",
    "llm_cache_read_only_misses",
    "cached_tokens_in",
    "cached_tokens_out",
    "actual_live_tokens_in",
    "actual_live_tokens_out",
)

CACHE_SUMMARY_FIELDS: tuple[str, ...] = (
    "llm_cache_mode",
    "llm_cache_path",
    *CACHE_NUMERIC_SUMMARY_FIELDS,
)


def aggregate_cache_fields_from_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sum numeric cache-accounting fields across CSV-fixture rows.

    ``rows`` are dict rows (e.g. from ``csv.DictReader``) that may carry
    string values for the numeric fields; blank/missing values count as 0.
    ``llm_cache_mode``/``llm_cache_path`` are per-run settings, not summed --
    the first row that sets a non-blank value wins.
    """
    totals: dict[str, int] = {field_name: 0 for field_name in CACHE_NUMERIC_SUMMARY_FIELDS}
    mode = ""
    path = ""
    for row in rows:
        for field_name in CACHE_NUMERIC_SUMMARY_FIELDS:
            value = row.get(field_name)
            if value in (None, ""):
                continue
            totals[field_name] += int(float(value))
        if not mode and row.get("llm_cache_mode"):
            mode = row["llm_cache_mode"]
        if not path and row.get("llm_cache_path"):
            path = row["llm_cache_path"]
    return {"llm_cache_mode": mode, "llm_cache_path": path, **totals}
