u"""Tests for deterministic provisional-valuation (PV) chunking.

Covers ``chunk_candidate_bundles`` (pure chunk-boundary logic),
``generate_provisional_valuations_chunked`` (per-chunk calls, merge
invariants, retry/duplicate/missing-bundle handling), and the
``LlmInferredXorProxy``/``LlmAuctionProxyAdapter`` integration points that
thread ``pv_chunk_size``/``max_parse_retries`` through. No live LLM/API
calls -- everything uses ``MockLlmClient``, a deterministic in-memory fake.
"""

from __future__ import annotations

import json

import pytest

from auctionlab.auction_types import Bundle
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.logging import LlmCallLogger
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmInferredXorProxy
from auctionlab.llm.provisional_valuations import (
    PvChunkStats,
    chunk_candidate_bundles,
    generate_provisional_valuations,
    generate_provisional_valuations_chunked,
)
from auctionlab.proxies.events import INFER_PROVISIONAL_VALUES, ProxyElicitationEvent


ITEM_DESCRIPTIONS = {f"ITEM_{i}": f"Item number {i}." for i in range(10)}


def _bundles(*names: str) -> list[Bundle]:
    return [frozenset({n}) for n in names]


def _pv_response(entries: dict[str, float], reasoning: str = "r") -> str:
    return json.dumps({
        "valuations": [
            {"bundle": [item], "value": value} for item, value in entries.items()
        ],
        "reasoning": reasoning,
    })


def _compact_pv_response(values: list[float], reasoning: str = "r") -> str:
    return json.dumps({"values": values, "reasoning": reasoning})


COMMON_KWARGS = dict(
    scenario_description="A test auction.",
    item_descriptions=ITEM_DESCRIPTIONS,
    nl_question="What do you want?",
    nl_answer="I want stuff.",
    bidder_id="bidder_1",
)


# ---------------------------------------------------------------------------
# chunk_candidate_bundles: pure, deterministic split
# ---------------------------------------------------------------------------

class TestChunkCandidateBundles:
    def test_empty_input_returns_no_chunks(self):
        assert chunk_candidate_bundles([], 3) == []

    def test_unset_chunk_size_returns_one_chunk(self):
        bundles = _bundles("A", "B", "C")
        assert chunk_candidate_bundles(bundles, None) == [bundles]

    def test_zero_chunk_size_returns_one_chunk(self):
        bundles = _bundles("A", "B", "C")
        assert chunk_candidate_bundles(bundles, 0) == [bundles]

    def test_negative_chunk_size_returns_one_chunk(self):
        bundles = _bundles("A", "B", "C")
        assert chunk_candidate_bundles(bundles, -5) == [bundles]

    def test_count_at_or_below_chunk_size_returns_one_chunk(self):
        bundles = _bundles("A", "B", "C")
        assert chunk_candidate_bundles(bundles, 3) == [bundles]
        assert chunk_candidate_bundles(bundles, 10) == [bundles]

    def test_count_above_chunk_size_splits_deterministically(self):
        bundles = _bundles("A", "B", "C", "D", "E")
        chunks = chunk_candidate_bundles(bundles, 2)
        assert chunks == [
            [frozenset({"A"}), frozenset({"B"})],
            [frozenset({"C"}), frozenset({"D"})],
            [frozenset({"E"})],
        ]

    def test_chunk_boundaries_are_stable_across_repeated_calls(self):
        bundles = _bundles(*[f"ITEM_{i}" for i in range(10)])
        first = chunk_candidate_bundles(bundles, 4)
        second = chunk_candidate_bundles(bundles, 4)
        assert first == second
        assert [len(c) for c in first] == [4, 4, 2]

    def test_preserves_bundle_identity_and_order(self):
        bundles = _bundles("Z", "A", "M")  # deliberately not sorted
        chunks = chunk_candidate_bundles(bundles, 2)
        flattened = [b for chunk in chunks for b in chunk]
        assert flattened == bundles


# ---------------------------------------------------------------------------
# generate_provisional_valuations_chunked: single-call passthrough
# ---------------------------------------------------------------------------

class TestChunkedSingleCallPassthrough:
    def test_compact_values_are_mapped_by_requested_bundle_order(self):
        client = MockLlmClient([
            _compact_pv_response([200.0, 100.0])
        ])

        result, stats = generate_provisional_valuations_chunked(
            client=client,
            candidate_bundles=_bundles("B", "A"),
            pv_chunk_size=100,
            **COMMON_KWARGS,
        )

        assert result == {
            frozenset({"B"}): 200.0,
            frozenset({"A"}): 100.0,
        }
        assert stats.pv_chunks == 1

    def test_compact_value_count_mismatch_fails_loudly(self):
        client = MockLlmClient([
            _compact_pv_response([100.0])
        ])

        with pytest.raises(ValueError, match="value count.*received=1.*requested=2"):
            generate_provisional_valuations_chunked(
                client=client,
                candidate_bundles=_bundles("A", "B"),
                pv_chunk_size=100,
                **COMMON_KWARGS,
            )

    def test_count_at_or_below_chunk_size_uses_one_pv_call(self):
        client = MockLlmClient([_pv_response({"A": 100.0, "B": 200.0})])
        result, stats = generate_provisional_valuations_chunked(
            client=client,
            candidate_bundles=_bundles("A", "B"),
            pv_chunk_size=10,
            **COMMON_KWARGS,
        )
        assert result == {frozenset({"A"}): 100.0, frozenset({"B"}): 200.0}
        assert len(client.calls) == 1
        assert stats == PvChunkStats(
            pv_chunk_size=10,
            pv_chunks=1,
            candidate_count=2,
            per_chunk_bundle_counts=(2,),
            chunking_used=False,
        )

    def test_unset_chunk_size_is_identical_to_unchunked_call(self):
        """Regression: chunk_size unset/0 must reproduce the exact single-call
        behaviour generate_provisional_valuations already had."""
        response = _pv_response({"A": 100.0, "B": 200.0})

        direct = generate_provisional_valuations(
            client=MockLlmClient([response]),
            candidate_bundles=_bundles("A", "B"),
            **COMMON_KWARGS,
        )
        chunked, stats = generate_provisional_valuations_chunked(
            client=MockLlmClient([response]),
            candidate_bundles=_bundles("A", "B"),
            pv_chunk_size=0,
            **COMMON_KWARGS,
        )
        assert chunked == direct
        assert stats.chunking_used is False
        assert stats.pv_chunks == 1

    def test_empty_candidate_bundles_makes_no_call(self):
        client = MockLlmClient([])
        result, stats = generate_provisional_valuations_chunked(
            client=client,
            candidate_bundles=[],
            pv_chunk_size=5,
            **COMMON_KWARGS,
        )
        assert result == {}
        assert stats.pv_chunks == 0
        assert stats.chunking_used is False
        assert len(client.calls) == 0


# ---------------------------------------------------------------------------
# generate_provisional_valuations_chunked: multi-chunk merge
# ---------------------------------------------------------------------------

class TestChunkedMerge:
    def test_splits_into_expected_number_of_chunks_and_merges_all_bundles(self):
        client = MockLlmClient([
            _pv_response({"A": 10.0, "B": 20.0}),
            _pv_response({"C": 30.0, "D": 40.0}),
            _pv_response({"E": 50.0}),
        ])
        result, stats = generate_provisional_valuations_chunked(
            client=client,
            candidate_bundles=_bundles("A", "B", "C", "D", "E"),
            pv_chunk_size=2,
            **COMMON_KWARGS,
        )
        assert len(client.calls) == 3
        assert stats.pv_chunks == 3
        assert stats.chunking_used is True
        assert stats.per_chunk_bundle_counts == (2, 2, 1)
        assert result == {
            frozenset({"A"}): 10.0,
            frozenset({"B"}): 20.0,
            frozenset({"C"}): 30.0,
            frozenset({"D"}): 40.0,
            frozenset({"E"}): 50.0,
        }

    def test_merged_output_preserves_correct_per_chunk_values(self):
        """Values must come from the chunk that actually valued that bundle,
        not accidentally overwritten by a later chunk's response."""
        client = MockLlmClient([
            _pv_response({"A": 111.0}),
            _pv_response({"B": 222.0}),
        ])
        result, _ = generate_provisional_valuations_chunked(
            client=client,
            candidate_bundles=_bundles("A", "B"),
            pv_chunk_size=1,
            **COMMON_KWARGS,
        )
        assert result[frozenset({"A"})] == 111.0
        assert result[frozenset({"B"})] == 222.0

    def test_prompt_is_annotated_with_chunk_index_and_count(self):
        client = MockLlmClient([
            _pv_response({"A": 1.0}),
            _pv_response({"B": 2.0}),
        ])
        generate_provisional_valuations_chunked(
            client=client,
            candidate_bundles=_bundles("A", "B"),
            pv_chunk_size=1,
            **COMMON_KWARGS,
        )
        assert "chunk 1 of 2" in client.calls[0]
        assert "chunk 2 of 2" in client.calls[1]

    def test_duplicate_bundle_across_chunks_raises(self):
        """A bundle that appears in two chunks (e.g. duplicated in the input
        candidate list) must raise rather than silently overwrite."""
        client = MockLlmClient([
            _pv_response({"A": 1.0, "B": 2.0}),
            _pv_response({"A": 999.0}),
        ])
        with pytest.raises(ValueError, match="duplicate valuation"):
            generate_provisional_valuations_chunked(
                client=client,
                candidate_bundles=_bundles("A", "B", "A"),
                pv_chunk_size=2,
                **COMMON_KWARGS,
            )

    def test_missing_valuation_in_a_chunk_raises_clear_error(self):
        client = MockLlmClient([
            _pv_response({"A": 1.0}),  # missing "B" from this chunk's response
        ])
        with pytest.raises(ValueError) as exc_info:
            generate_provisional_valuations_chunked(
                client=client,
                candidate_bundles=_bundles("A", "B"),
                pv_chunk_size=5,  # single chunk containing both A and B
                **COMMON_KWARGS,
            )
        message = str(exc_info.value)
        assert "bidder_1" in message
        assert "B" in message

    def test_missing_valuation_names_chunk_index(self):
        client = MockLlmClient([
            _pv_response({"A": 1.0}),
            _pv_response({}),  # missing "B" in the second chunk
        ])
        with pytest.raises(ValueError) as exc_info:
            generate_provisional_valuations_chunked(
                client=client,
                candidate_bundles=_bundles("A", "B"),
                pv_chunk_size=1,
                **COMMON_KWARGS,
            )
        assert "chunk_index=1" in str(exc_info.value)

    def test_extra_valuations_beyond_the_requested_chunk_are_ignored(self):
        """Extras naming a bundle outside the requesting chunk are dropped,
        matching the base parser's existing (pre-chunking) behaviour."""
        client = MockLlmClient([
            _pv_response({"A": 1.0, "Z_NOT_REQUESTED": 999.0}),
        ])
        result, _ = generate_provisional_valuations_chunked(
            client=client,
            candidate_bundles=_bundles("A"),
            pv_chunk_size=5,
            **COMMON_KWARGS,
        )
        assert result == {frozenset({"A"}): 1.0}

    def test_the_whole_candidate_set_is_never_reduced_by_chunking(self):
        n = 25
        bundles = _bundles(*[f"ITEM_{i}" for i in range(n)])
        responses = [
            _pv_response({b: float(i) for i, b in enumerate(chunk_items)})
            for chunk_items in (
                [f"ITEM_{i}" for i in range(0, 10)],
                [f"ITEM_{i}" for i in range(10, 20)],
                [f"ITEM_{i}" for i in range(20, 25)],
            )
        ]
        client = MockLlmClient(responses)
        result, stats = generate_provisional_valuations_chunked(
            client=client,
            candidate_bundles=bundles,
            pv_chunk_size=10,
            **COMMON_KWARGS,
        )
        assert len(result) == n
        assert stats.pv_chunks == 3
        assert stats.candidate_count == n


# ---------------------------------------------------------------------------
# Retry behaviour (shared by single-call and chunked paths)
# ---------------------------------------------------------------------------

class TestParseRetry:
    def test_malformed_response_is_retried_and_recovers(self):
        client = MockLlmClient([
            "not json at all",
            _pv_response({"A": 5.0}),
        ])
        result = generate_provisional_valuations(
            client=client,
            candidate_bundles=_bundles("A"),
            max_parse_retries=1,
            **COMMON_KWARGS,
        )
        assert result == {frozenset({"A"}): 5.0}
        assert len(client.calls) == 2

    def test_malformed_response_raises_after_retries_exhausted(self):
        client = MockLlmClient(["not json", "still not json"])
        with pytest.raises(ValueError):
            generate_provisional_valuations(
                client=client,
                candidate_bundles=_bundles("A"),
                max_parse_retries=1,
                **COMMON_KWARGS,
            )
        assert len(client.calls) == 2

    def test_no_retry_by_default(self):
        client = MockLlmClient(["not json"])
        with pytest.raises(ValueError):
            generate_provisional_valuations(
                client=client,
                candidate_bundles=_bundles("A"),
                **COMMON_KWARGS,
            )
        assert len(client.calls) == 1

    def test_failing_chunk_is_retried_independently_within_chunked_call(self):
        client = MockLlmClient([
            _pv_response({"A": 1.0}),  # chunk 1 succeeds first try
            "malformed",  # chunk 2 first attempt fails
            _pv_response({"B": 2.0}),  # chunk 2 retry succeeds
        ])
        result, stats = generate_provisional_valuations_chunked(
            client=client,
            candidate_bundles=_bundles("A", "B"),
            pv_chunk_size=1,
            max_parse_retries=1,
            **COMMON_KWARGS,
        )
        assert result == {frozenset({"A"}): 1.0, frozenset({"B"}): 2.0}
        assert len(client.calls) == 3
        assert stats.pv_chunks == 2

    def test_chunk_failure_after_retries_propagates_from_chunked_call(self):
        client = MockLlmClient([
            _pv_response({"A": 1.0}),
            "malformed",
            "still malformed",
        ])
        with pytest.raises(ValueError):
            generate_provisional_valuations_chunked(
                client=client,
                candidate_bundles=_bundles("A", "B"),
                pv_chunk_size=1,
                max_parse_retries=1,
                **COMMON_KWARGS,
            )

    def test_call_level_exception_is_not_retried(self):
        """A raised exception from the client itself (network/timeout-style)
        propagates immediately -- only parse failures are retried."""

        class _RaisingClient:
            def complete(self, prompt: str) -> str:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            generate_provisional_valuations(
                client=_RaisingClient(),
                candidate_bundles=_bundles("A"),
                max_parse_retries=3,
                **COMMON_KWARGS,
            )


# ---------------------------------------------------------------------------
# calls.jsonl metadata: chunk_index/chunk_count are logged
# ---------------------------------------------------------------------------

def _read_log_records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines()]


class TestChunkLogMetadata:
    def test_chunk_index_and_count_appear_in_call_log(self, tmp_path):
        log_path = tmp_path / "calls.jsonl"
        logger = LlmCallLogger(log_path)
        client = MockLlmClient([
            _pv_response({"A": 1.0}),
            _pv_response({"B": 2.0}),
        ])
        generate_provisional_valuations_chunked(
            client=client,
            candidate_bundles=_bundles("A", "B"),
            pv_chunk_size=1,
            logger=logger,
            **COMMON_KWARGS,
        )
        records = _read_log_records(log_path)
        assert len(records) == 2
        assert records[0]["parsed_response"]["chunk_index"] == 0
        assert records[0]["parsed_response"]["chunk_count"] == 2
        assert records[1]["parsed_response"]["chunk_index"] == 1
        assert records[1]["parsed_response"]["chunk_count"] == 2

    def test_single_chunk_logs_null_chunk_index(self, tmp_path):
        log_path = tmp_path / "calls.jsonl"
        logger = LlmCallLogger(log_path)
        client = MockLlmClient([_pv_response({"A": 1.0})])
        generate_provisional_valuations_chunked(
            client=client,
            candidate_bundles=_bundles("A"),
            pv_chunk_size=5,
            logger=logger,
            **COMMON_KWARGS,
        )
        records = _read_log_records(log_path)
        assert records[0]["parsed_response"]["chunk_index"] is None

    def test_failed_chunk_attempt_is_logged(self, tmp_path):
        log_path = tmp_path / "calls.jsonl"
        logger = LlmCallLogger(log_path)
        client = MockLlmClient(["malformed"])
        with pytest.raises(ValueError):
            generate_provisional_valuations(
                client=client,
                candidate_bundles=_bundles("A"),
                logger=logger,
                **COMMON_KWARGS,
            )
        records = _read_log_records(log_path)
        assert records[0]["success"] is False
        assert records[0]["error"]


# ---------------------------------------------------------------------------
# Proxy-level integration: LlmInferredXorProxy.build_provisional_valuations
# ---------------------------------------------------------------------------

def _make_proxy_with_nl_transcript() -> LlmInferredXorProxy:
    person = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A test auction.",
        person_seed="Values useful combinations.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=MockLlmClient([]),
    )
    proxy = LlmInferredXorProxy(bidder_id="bidder_1", person=person)
    proxy.nl_transcript.append(("What do you want?", "I want stuff."))
    return proxy


class TestProxyChunkingIntegration:
    def test_build_provisional_valuations_uses_chunking_and_records_stats(self):
        proxy = _make_proxy_with_nl_transcript()
        pv_client = MockLlmClient([
            _pv_response({"ITEM_0": 10.0}),
            _pv_response({"ITEM_1": 20.0}),
        ])
        raw_values = proxy.build_provisional_valuations(
            _bundles("ITEM_0", "ITEM_1"),
            client=pv_client,
            pv_chunk_size=1,
        )
        assert raw_values == {frozenset({"ITEM_0"}): 10.0, frozenset({"ITEM_1"}): 20.0}
        assert proxy.last_pv_chunk_stats is not None
        assert proxy.last_pv_chunk_stats.pv_chunks == 2
        assert proxy.last_pv_chunk_stats.chunking_used is True

    def test_build_provisional_valuations_default_chunk_size_is_unchunked(self):
        proxy = _make_proxy_with_nl_transcript()
        pv_client = MockLlmClient([_pv_response({"ITEM_0": 10.0, "ITEM_1": 20.0})])
        raw_values = proxy.build_provisional_valuations(
            _bundles("ITEM_0", "ITEM_1"),
            client=pv_client,
        )
        assert len(pv_client.calls) == 1
        assert proxy.last_pv_chunk_stats.chunking_used is False
        assert proxy.last_pv_chunk_stats.pv_chunks == 1
        assert raw_values == {
            frozenset({"ITEM_0"}): 10.0,
            frozenset({"ITEM_1"}): 20.0,
        }

    def test_handle_event_infer_provisional_values_threads_pv_chunk_size(self):
        proxy = _make_proxy_with_nl_transcript()
        pv_client = MockLlmClient([
            _pv_response({"ITEM_0": 1.0}),
            _pv_response({"ITEM_1": 2.0}),
        ])
        response = proxy.handle_event(
            ProxyElicitationEvent(
                event_type=INFER_PROVISIONAL_VALUES,
                bidder_id="bidder_1",
                mechanism="init",
                payload={
                    "candidate_bundles": _bundles("ITEM_0", "ITEM_1"),
                    "client": pv_client,
                    "pv_chunk_size": 1,
                },
            )
        )
        assert response.state_delta["pv_chunks"] == 2
        assert response.state_delta["pv_chunking_used"] is True
        assert len(response.payload["raw_values"]) == 2
