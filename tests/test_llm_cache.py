"""Tests for the LLM response cache (auctionlab.llm.cache).

All deterministic: no live LLM/API calls anywhere in this file. Provider
calls are simulated with small in-process fake "call_fn"/client objects
that record how many times they were invoked.
"""

from __future__ import annotations

import pytest

from auctionlab.llm.cache import (
    CacheEntry,
    CacheMissError,
    CacheStats,
    CachingLlmClient,
    LlmCallOutcome,
    LlmResponseCache,
    aggregate_cache_fields_from_rows,
    bundle_key_str,
    bundle_set_hash,
    build_request_envelope,
    call_client,
    compute_cache_key,
    get_or_call,
)
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.interest_map import derive_interest_map
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.provisional_valuations import generate_provisional_valuations


# ---------------------------------------------------------------------------
# 1-5: cache key construction
# ---------------------------------------------------------------------------


class TestCacheKeyConstruction:
    def test_cache_key_stable_under_dict_ordering(self):
        env_a = {"call_type": "value_query", "model": "m", "provider": "p", "temperature": 0.0}
        env_b = {"temperature": 0.0, "provider": "p", "model": "m", "call_type": "value_query"}
        assert compute_cache_key(env_a) == compute_cache_key(env_b)

    def test_build_request_envelope_stable_under_kwarg_ordering(self):
        key_a = compute_cache_key(build_request_envelope(
            call_type="value_query", provider="p", model="m", prompt="hello",
            temperature=0.0, bidder_id="b1",
        ))
        key_b = compute_cache_key(build_request_envelope(
            bidder_id="b1", temperature=0.0, prompt="hello",
            model="m", provider="p", call_type="value_query",
        ))
        assert key_a == key_b

    def test_changing_prompt_text_changes_key(self):
        base = dict(call_type="value_query", provider="p", model="m")
        key_1 = compute_cache_key(build_request_envelope(prompt="prompt one", **base))
        key_2 = compute_cache_key(build_request_envelope(prompt="prompt two", **base))
        assert key_1 != key_2

    def test_changing_model_changes_key(self):
        base = dict(call_type="value_query", provider="p", prompt="same prompt")
        key_1 = compute_cache_key(build_request_envelope(model="model-a", **base))
        key_2 = compute_cache_key(build_request_envelope(model="model-b", **base))
        assert key_1 != key_2

    def test_changing_provider_changes_key(self):
        base = dict(call_type="value_query", model="m", prompt="same prompt")
        key_1 = compute_cache_key(build_request_envelope(provider="gemini", **base))
        key_2 = compute_cache_key(build_request_envelope(provider="groq", **base))
        assert key_1 != key_2

    def test_changing_temperature_changes_key(self):
        base = dict(call_type="value_query", provider="p", model="m", prompt="same prompt")
        key_1 = compute_cache_key(build_request_envelope(temperature=0.0, **base))
        key_2 = compute_cache_key(build_request_envelope(temperature=0.7, **base))
        assert key_1 != key_2

    def test_changing_candidate_bundle_set_changes_pv_key(self):
        base = dict(call_type="provisional_valuations", provider="p", model="m", prompt="pv prompt")
        set_a = bundle_set_hash([frozenset({"A"}), frozenset({"A", "B"})])
        set_b = bundle_set_hash([frozenset({"A"}), frozenset({"A", "C"})])
        key_1 = compute_cache_key(build_request_envelope(bundle_set_hash=set_a, **base))
        key_2 = compute_cache_key(build_request_envelope(bundle_set_hash=set_b, **base))
        assert key_1 != key_2

    def test_bundle_set_hash_is_order_independent(self):
        bundles_a = [frozenset({"A"}), frozenset({"A", "B"})]
        bundles_b = [frozenset({"A", "B"}), frozenset({"A"})]
        assert bundle_set_hash(bundles_a) == bundle_set_hash(bundles_b)

    def test_value_query_cache_key_distinguishes_bundles(self):
        base = dict(call_type="value_query", provider="p", model="m", prompt="value query prompt")
        key_1 = compute_cache_key(build_request_envelope(
            queried_bundle=bundle_key_str(frozenset({"A"})), **base
        ))
        key_2 = compute_cache_key(build_request_envelope(
            queried_bundle=bundle_key_str(frozenset({"A", "B"})), **base
        ))
        assert key_1 != key_2

    def test_requires_prompt_or_prompt_hash(self):
        with pytest.raises(ValueError):
            build_request_envelope(call_type="value_query", provider="p", model="m")


# ---------------------------------------------------------------------------
# 6-10: cache mode semantics via get_or_call
# ---------------------------------------------------------------------------


def _envelope(**overrides):
    base = dict(call_type="value_query", provider="p", model="m", prompt="prompt text")
    base.update(overrides)
    return build_request_envelope(**base)


class _CountingCallFn:
    def __init__(self, text: str = "raw response", tokens_in: int = 10, tokens_out: int = 5):
        self.calls = 0
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out

    def __call__(self) -> LlmCallOutcome:
        self.calls += 1
        return LlmCallOutcome(
            raw_response_text=self.text,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            latency_seconds=0.01,
        )


class TestCacheModeSemantics:
    def test_read_write_miss_calls_fake_client_and_writes_cache(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        call_fn = _CountingCallFn()
        envelope = _envelope()

        result = get_or_call(
            cache=cache, mode="read-write", call_type="value_query",
            request_envelope=envelope, call_fn=call_fn,
        )

        assert call_fn.calls == 1
        assert result.cache_hit is False
        assert result.raw_response_text == "raw response"
        assert cache.get(compute_cache_key(envelope)) is not None

    def test_read_write_hit_does_not_call_fake_client(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        call_fn = _CountingCallFn()
        envelope = _envelope()

        get_or_call(
            cache=cache, mode="read-write", call_type="value_query",
            request_envelope=envelope, call_fn=call_fn,
        )
        assert call_fn.calls == 1

        result = get_or_call(
            cache=cache, mode="read-write", call_type="value_query",
            request_envelope=envelope, call_fn=call_fn,
        )
        assert call_fn.calls == 1  # not called again
        assert result.cache_hit is True
        assert result.raw_response_text == "raw response"

    def test_read_only_hit_returns_cached_response(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        envelope = _envelope()
        write_call_fn = _CountingCallFn(text="pre-populated response")
        get_or_call(
            cache=cache, mode="read-write", call_type="value_query",
            request_envelope=envelope, call_fn=write_call_fn,
        )

        read_only_call_fn = _CountingCallFn(text="should never be used")
        result = get_or_call(
            cache=cache, mode="read-only", call_type="value_query",
            request_envelope=envelope, call_fn=read_only_call_fn,
        )

        assert read_only_call_fn.calls == 0
        assert result.cache_hit is True
        assert result.raw_response_text == "pre-populated response"

    def test_read_only_miss_raises_cache_miss_error_and_does_not_call_fake_client(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        call_fn = _CountingCallFn()
        envelope = _envelope()

        with pytest.raises(CacheMissError) as exc_info:
            get_or_call(
                cache=cache, mode="read-only", call_type="value_query",
                request_envelope=envelope, call_fn=call_fn,
                bidder_id="bidder_01", bundle_key="{A,B}",
            )

        assert call_fn.calls == 0
        err = exc_info.value
        assert err.call_type == "value_query"
        assert err.bidder_id == "bidder_01"
        assert err.bundle_key == "{A,B}"
        assert err.cache_key_prefix == err.cache_key[:12]
        assert cache.get(compute_cache_key(envelope)) is None

    def test_refresh_calls_fake_client_and_overwrites_cache(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        envelope = _envelope()
        first_call_fn = _CountingCallFn(text="first response")
        get_or_call(
            cache=cache, mode="read-write", call_type="value_query",
            request_envelope=envelope, call_fn=first_call_fn,
        )

        second_call_fn = _CountingCallFn(text="refreshed response")
        result = get_or_call(
            cache=cache, mode="refresh", call_type="value_query",
            request_envelope=envelope, call_fn=second_call_fn,
        )

        assert second_call_fn.calls == 1
        assert result.cache_hit is False
        assert result.raw_response_text == "refreshed response"

        entry = cache.get(compute_cache_key(envelope))
        assert entry.raw_response_text == "refreshed response"

    def test_off_mode_always_calls_and_never_touches_cache(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        call_fn = _CountingCallFn()
        envelope = _envelope()

        get_or_call(cache=cache, mode="off", call_type="value_query", request_envelope=envelope, call_fn=call_fn)
        get_or_call(cache=cache, mode="off", call_type="value_query", request_envelope=envelope, call_fn=call_fn)

        assert call_fn.calls == 2
        assert cache.get(compute_cache_key(envelope)) is None

    def test_off_mode_requires_no_cache_instance(self):
        call_fn = _CountingCallFn()
        result = get_or_call(cache=None, mode="off", call_type="value_query", request_envelope=_envelope(), call_fn=call_fn)
        assert call_fn.calls == 1
        assert result.cache_hit is False

    def test_non_off_mode_requires_cache_instance(self):
        with pytest.raises(ValueError):
            get_or_call(
                cache=None, mode="read-write", call_type="value_query",
                request_envelope=_envelope(), call_fn=_CountingCallFn(),
            )

    def test_invalid_mode_rejected(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        with pytest.raises(ValueError):
            get_or_call(
                cache=cache, mode="bogus-mode", call_type="value_query",
                request_envelope=_envelope(), call_fn=_CountingCallFn(),
            )


# ---------------------------------------------------------------------------
# 11: token accounting
# ---------------------------------------------------------------------------


class TestTokenAccounting:
    def test_token_accounting_distinguishes_cached_from_live_tokens(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        stats = CacheStats()
        envelope = _envelope()

        write_call_fn = _CountingCallFn(tokens_in=100, tokens_out=40)
        first = get_or_call(
            cache=cache, mode="read-write", call_type="value_query",
            request_envelope=envelope, call_fn=write_call_fn, stats=stats,
        )
        assert first.tokens_in == 100
        assert first.tokens_out == 40
        assert stats.actual_live_tokens_in == 100
        assert stats.actual_live_tokens_out == 40
        assert stats.cached_tokens_in == 0

        hit_call_fn = _CountingCallFn(tokens_in=999, tokens_out=999)
        second = get_or_call(
            cache=cache, mode="read-write", call_type="value_query",
            request_envelope=envelope, call_fn=hit_call_fn, stats=stats,
        )

        assert hit_call_fn.calls == 0
        # actual tokens charged *this run* are zero on a cache hit ...
        assert second.tokens_in == 0
        assert second.tokens_out == 0
        # ... but the original call's token counts are still reported.
        assert second.cached_tokens_in == 100
        assert second.cached_tokens_out == 40

        assert stats.hits == 1
        assert stats.writes == 1
        # live totals are unaffected by the hit; cached totals accumulate it.
        assert stats.actual_live_tokens_in == 100
        assert stats.actual_live_tokens_out == 40
        assert stats.cached_tokens_in == 100
        assert stats.cached_tokens_out == 40

    def test_caching_llm_client_reports_zero_tokens_on_hit(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        inner = MockLlmClient(responses=["resp-1"])
        inner._last_input_tokens = 50
        inner._last_output_tokens = 20

        client = CachingLlmClient(
            inner=inner, cache=cache, mode="read-write", provider="p", model="m",
        )

        text_1 = client.complete("prompt A")
        assert text_1 == "resp-1"
        assert client._last_input_tokens == 50
        assert client._last_output_tokens == 20
        assert client._last_cache_hit is False

        text_2 = client.complete("prompt A")
        assert text_2 == "resp-1"
        assert client._last_input_tokens == 0
        assert client._last_output_tokens == 0
        assert client._last_cache_hit is True
        # the inner client was only ever asked once.
        assert inner.prompts == ["prompt A"]

    def test_caching_llm_client_can_invalidate_rejected_raw_response(
        self, tmp_path
    ):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        inner = MockLlmClient(responses=["truncated", "repaired"])
        client = CachingLlmClient(
            inner=inner,
            cache=cache,
            mode="read-write",
            provider="p",
            model="m",
        )

        assert client.complete("prompt A") == "truncated"
        rejected_key = client._last_cache_key
        assert rejected_key is not None
        assert cache.get(rejected_key) is not None

        client.invalidate_last()

        assert cache.get(rejected_key) is None
        assert client.complete("prompt A") == "repaired"
        assert inner.prompts == ["prompt A", "prompt A"]


# ---------------------------------------------------------------------------
# call_client() passthrough + integration with real call sites
# ---------------------------------------------------------------------------


class TestCallClientPassthrough:
    def test_plain_client_ignores_cache_kwargs(self):
        client = MockLlmClient(responses=["ok"])
        result = call_client(client, "hello", call_type="value_query", bidder_id="b1")
        assert result == "ok"
        assert client.prompts == ["hello"]

    def test_caching_client_receives_cache_kwargs(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        inner = MockLlmClient(responses=["a", "b"])
        client = CachingLlmClient(inner=inner, cache=cache, mode="read-write", provider="p", model="m")

        result_1 = call_client(client, "prompt", call_type="value_query", bidder_id="b1", bundle_key="{A}")
        result_2 = call_client(client, "prompt", call_type="value_query", bidder_id="b1", bundle_key="{A}")

        assert result_1 == "a"
        assert result_2 == "a"  # replayed from cache, not "b"
        assert inner.prompts == ["prompt"]


class TestPersonSimulatorIntegration:
    def _make_simulator(self, cache, responses):
        inner = MockLlmClient(responses=list(responses))
        client = CachingLlmClient(inner=inner, cache=cache, mode="read-write", provider="p", model="m")
        person = LlmPersonSimulator(
            bidder_id="bidder_01",
            scenario_description="A simple scenario.",
            person_seed="Wants item A a lot.",
            item_descriptions={"A": "Item A", "B": "Item B"},
            client=client,
            scenario_id="scenario_1",
        )
        return person, inner

    def test_value_query_cache_hit_skips_inner_client(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        responses = [
            '{"queried_bundle": ["A"], "bundle_value": 100, "confidence": 0.9, "reasoning_summary": "ok"}',
            '{"queried_bundle": ["A"], "bundle_value": 999, "confidence": 0.9, "reasoning_summary": "should not be used"}',
        ]
        person, inner = self._make_simulator(cache, responses)

        value_1 = person.value_query(frozenset({"A"}))
        value_2 = person.value_query(frozenset({"A"}))

        assert value_1 == 100.0
        assert value_2 == 100.0
        assert len(inner.prompts) == 1  # second call was a cache hit

    def test_value_query_different_bundles_are_not_conflated(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        responses = [
            '{"queried_bundle": ["A"], "bundle_value": 100, "confidence": 0.9, "reasoning_summary": "ok"}',
            '{"queried_bundle": ["B"], "bundle_value": 200, "confidence": 0.9, "reasoning_summary": "ok"}',
        ]
        person, inner = self._make_simulator(cache, responses)

        value_a = person.value_query(frozenset({"A"}))
        value_b = person.value_query(frozenset({"B"}))

        assert value_a == 100.0
        assert value_b == 200.0
        assert len(inner.prompts) == 2  # distinct bundles, both are misses

    def test_answer_question_cache_hit_skips_inner_client(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        responses = [
            '{"answer": "I love item A."}',
            '{"answer": "should not be used"}',
        ]
        person, inner = self._make_simulator(cache, responses)

        answer_1 = person.answer_question("What do you want?")
        answer_2 = person.answer_question("What do you want?")

        assert answer_1 == "I love item A."
        assert answer_2 == "I love item A."
        assert len(inner.prompts) == 1

    def test_answer_question_parse_retry_does_not_replay_cached_failure(
        self, tmp_path
    ):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        person, inner = self._make_simulator(
            cache,
            [
                '{"answer":"truncated',
                '{"answer":"A complete repaired answer."}',
            ],
        )
        person.max_parse_retries = 1

        answer = person.answer_question("What do you want?")

        assert answer == "A complete repaired answer."
        assert len(inner.prompts) == 2
        assert inner.prompts[0] != inner.prompts[1]

    def test_rejected_answer_is_evicted_before_resumed_attempt(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        person, inner = self._make_simulator(
            cache,
            [
                '{"answer":"truncated',
                '{"answer":"A complete answer."}',
            ],
        )
        person.max_parse_retries = 0

        with pytest.raises(ValueError):
            person.answer_question("What do you want?")

        assert person.answer_question("What do you want?") == "A complete answer."
        assert inner.prompts == [inner.prompts[0], inner.prompts[0]]

    def test_read_only_miss_raises_cache_miss_error(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        inner = MockLlmClient(responses=[])
        client = CachingLlmClient(inner=inner, cache=cache, mode="read-only", provider="p", model="m")
        person = LlmPersonSimulator(
            bidder_id="bidder_01",
            scenario_description="A simple scenario.",
            person_seed="Wants item A a lot.",
            item_descriptions={"A": "Item A"},
            client=client,
        )

        with pytest.raises(CacheMissError):
            person.value_query(frozenset({"A"}))
        assert inner.prompts == []


class TestInterestMapAndProvisionalValuationsIntegration:
    def test_interest_map_parse_retry_does_not_replay_cached_failure(
        self, tmp_path
    ):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        inner = MockLlmClient(responses=[
            '{"interested_items":["A"]',
            (
                '{"interested_items":["A"],"excluded_items":["B"],'
                '"complementary_groups":[],"substitute_groups":[],'
                '"budget_hint":null,"reasoning":"A only"}'
            ),
        ])
        client = CachingLlmClient(
            inner=inner,
            cache=cache,
            mode="read-write",
            provider="p",
            model="m",
        )

        with pytest.warns(UserWarning, match="Retrying"):
            result = derive_interest_map(
                client=client,
                scenario_description="A scenario.",
                item_descriptions={"A": "Item A", "B": "Item B"},
                nl_question="What do you want?",
                nl_answer="A.",
                bidder_id="bidder_01",
                scenario_id="scenario_1",
                failure_policy="raise",
            )

        assert result.interested_items == ["A"]
        assert len(inner.prompts) == 2
        assert inner.prompts[0] != inner.prompts[1]

    def test_interest_map_cache_hit_skips_inner_client(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        responses = [
            (
                '{"interested_items": ["A"], "excluded_items": ["B"], '
                '"complementary_groups": [], "substitute_groups": [], '
                '"budget_hint": null, "reasoning": "likes A"}'
            ),
        ]
        inner = MockLlmClient(responses=list(responses))
        client = CachingLlmClient(inner=inner, cache=cache, mode="read-write", provider="p", model="m")

        kwargs = dict(
            client=client,
            scenario_description="A scenario.",
            item_descriptions={"A": "Item A", "B": "Item B"},
            nl_question="What do you want?",
            nl_answer="I want A.",
            bidder_id="bidder_01",
            scenario_id="scenario_1",
        )
        im_1 = derive_interest_map(**kwargs)
        im_2 = derive_interest_map(**kwargs)

        assert im_1.interested_items == ["A"]
        assert im_2.interested_items == ["A"]
        assert len(inner.prompts) == 1

    def test_provisional_valuations_cache_key_differs_by_candidate_bundle_set(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        responses = [
            '{"valuations": [{"bundle": ["A"], "value": 100}], "reasoning": "r1"}',
            '{"valuations": [{"bundle": ["A"], "value": 100}, {"bundle": ["A", "B"], "value": 250}], "reasoning": "r2"}',
        ]
        inner = MockLlmClient(responses=list(responses))
        client = CachingLlmClient(inner=inner, cache=cache, mode="read-write", provider="p", model="m")

        common = dict(
            client=client,
            scenario_description="A scenario.",
            item_descriptions={"A": "Item A", "B": "Item B"},
            nl_question="What do you want?",
            nl_answer="I want A and maybe B.",
            bidder_id="bidder_01",
            scenario_id="scenario_1",
        )

        result_1 = generate_provisional_valuations(
            candidate_bundles=[frozenset({"A"})], **common
        )
        result_2 = generate_provisional_valuations(
            candidate_bundles=[frozenset({"A"}), frozenset({"A", "B"})], **common
        )

        assert result_1 == {frozenset({"A"}): 100.0}
        assert result_2[frozenset({"A", "B"})] == 250.0
        # Different candidate bundle sets -> different cache keys -> both
        # calls actually reached the inner client.
        assert len(inner.prompts) == 2

        # Re-running the second (larger) candidate set is now a cache hit.
        result_2_again = generate_provisional_valuations(
            candidate_bundles=[frozenset({"A"}), frozenset({"A", "B"})], **common
        )
        assert result_2_again == result_2
        assert len(inner.prompts) == 2

    def test_provisional_valuation_parse_failure_is_not_left_in_cache(
        self, tmp_path
    ):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        inner = MockLlmClient(
            responses=[
                '{"values":[100',
                '{"values":[100]}',
            ]
        )
        client = CachingLlmClient(
            inner=inner,
            cache=cache,
            mode="read-write",
            provider="p",
            model="m",
        )
        common = dict(
            client=client,
            scenario_description="A scenario.",
            item_descriptions={"A": "Item A"},
            nl_question="What do you want?",
            nl_answer="I want A.",
            bidder_id="bidder_01",
            scenario_id="scenario_1",
            candidate_bundles=[frozenset({"A"})],
        )

        with pytest.raises(ValueError):
            generate_provisional_valuations(**common)

        # A resumed attempt with the same request must call the provider,
        # rather than replaying the malformed first response.
        result = generate_provisional_valuations(**common)
        assert result == {frozenset({"A"}): 100.0}
        assert inner.prompts == [inner.prompts[0], inner.prompts[0]]


# ---------------------------------------------------------------------------
# CacheEntry / LlmResponseCache round trip
# ---------------------------------------------------------------------------


class TestLlmResponseCacheStorage:
    def test_put_and_get_round_trip(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        entry = CacheEntry(
            cache_key="abc123",
            call_type="value_query",
            provider="p",
            model="m",
            prompt_hash="hash",
            request_json="{}",
            raw_response_text="response text",
            tokens_in=10,
            tokens_out=5,
            created_at="2026-01-01T00:00:00+00:00",
            bidder_id="bidder_01",
            bundle_key="{A}",
        )
        cache.put(entry)

        fetched = cache.get("abc123")
        assert fetched is not None
        assert fetched.raw_response_text == "response text"
        assert fetched.bidder_id == "bidder_01"
        assert fetched.bundle_key == "{A}"

    def test_get_missing_key_returns_none(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        assert cache.get("does-not-exist") is None

    def test_put_overwrites_existing_entry(self, tmp_path):
        cache = LlmResponseCache(tmp_path / "cache.sqlite")
        entry = CacheEntry(
            cache_key="k1", call_type="value_query", provider="p", model="m",
            prompt_hash="h", request_json="{}", raw_response_text="first",
            created_at="t",
        )
        cache.put(entry)
        entry2 = CacheEntry(
            cache_key="k1", call_type="value_query", provider="p", model="m",
            prompt_hash="h", request_json="{}", raw_response_text="second",
            created_at="t",
        )
        cache.put(entry2)
        assert cache.get("k1").raw_response_text == "second"

    def test_persists_across_reopen(self, tmp_path):
        path = tmp_path / "cache.sqlite"
        cache_1 = LlmResponseCache(path)
        cache_1.put(CacheEntry(
            cache_key="k1", call_type="value_query", provider="p", model="m",
            prompt_hash="h", request_json="{}", raw_response_text="persisted",
            created_at="t",
        ))
        cache_1.close()

        cache_2 = LlmResponseCache(path)
        assert cache_2.get("k1").raw_response_text == "persisted"


# ---------------------------------------------------------------------------
# 14: run-summary cache aggregation from CSV fixtures
# ---------------------------------------------------------------------------


class TestAggregateCacheFieldsFromRows:
    def test_sums_numeric_fields_across_rows(self):
        rows = [
            {
                "llm_cache_mode": "read-write", "llm_cache_path": "cache/a.sqlite",
                "llm_cache_hits": "3", "llm_cache_misses": "1", "llm_cache_writes": "1",
                "llm_cache_read_only_misses": "0", "cached_tokens_in": "300",
                "cached_tokens_out": "120", "actual_live_tokens_in": "50",
                "actual_live_tokens_out": "20",
            },
            {
                "llm_cache_mode": "read-write", "llm_cache_path": "cache/a.sqlite",
                "llm_cache_hits": "2", "llm_cache_misses": "0", "llm_cache_writes": "0",
                "llm_cache_read_only_misses": "0", "cached_tokens_in": "150",
                "cached_tokens_out": "60", "actual_live_tokens_in": "0",
                "actual_live_tokens_out": "0",
            },
        ]

        totals = aggregate_cache_fields_from_rows(rows)

        assert totals["llm_cache_mode"] == "read-write"
        assert totals["llm_cache_path"] == "cache/a.sqlite"
        assert totals["llm_cache_hits"] == 5
        assert totals["llm_cache_misses"] == 1
        assert totals["llm_cache_writes"] == 1
        assert totals["cached_tokens_in"] == 450
        assert totals["cached_tokens_out"] == 180
        assert totals["actual_live_tokens_in"] == 50
        assert totals["actual_live_tokens_out"] == 20

    def test_blank_and_missing_values_count_as_zero(self):
        rows = [
            {"llm_cache_hits": "", "llm_cache_misses": "2"},
            {"llm_cache_hits": "4"},
        ]
        totals = aggregate_cache_fields_from_rows(rows)
        assert totals["llm_cache_hits"] == 4
        assert totals["llm_cache_misses"] == 2
        assert totals["llm_cache_writes"] == 0

    def test_empty_rows_returns_zeroed_totals_with_blank_mode_and_path(self):
        totals = aggregate_cache_fields_from_rows([])
        assert totals["llm_cache_mode"] == ""
        assert totals["llm_cache_path"] == ""
        assert totals["llm_cache_hits"] == 0

    def test_first_nonblank_mode_and_path_win(self):
        rows = [
            {"llm_cache_mode": "", "llm_cache_path": ""},
            {"llm_cache_mode": "refresh", "llm_cache_path": "cache/b.sqlite"},
            {"llm_cache_mode": "read-only", "llm_cache_path": "cache/c.sqlite"},
        ]
        totals = aggregate_cache_fields_from_rows(rows)
        assert totals["llm_cache_mode"] == "refresh"
        assert totals["llm_cache_path"] == "cache/b.sqlite"
