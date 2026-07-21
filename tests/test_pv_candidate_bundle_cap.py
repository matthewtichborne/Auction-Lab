"""Tests for the PV candidate-bundle truncation behavior.

Covers the bug where candidate bundles were silently truncated based on an
auto-derived token-budget estimate even when the caller never asked for a
cap. All truncation must now be explicit (via ``max_bundles`` /
``max_candidate_bundles``); an uncapped run must send every candidate
bundle to the LLM regardless of the client's ``max_tokens``. No live
LLM/API calls -- everything here uses ``MockLlmClient``, a deterministic
in-memory fake.
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations

import pytest

from auctionlab.auction_types import Bundle
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.logging import LlmCallLogger
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmInferredXorProxy
from auctionlab.llm.provisional_valuations import (
    PvCandidateBundleStats,
    compute_pv_candidate_bundle_stats,
    generate_provisional_valuations,
)
from auctionlab.proxies.events import INFER_PROVISIONAL_VALUES, ProxyElicitationEvent

ITEM_DESCRIPTIONS = {f"ITEM_{i}": f"Item number {i}." for i in range(10)}

_EMPTY_PV_RESPONSE = json.dumps({"valuations": [], "reasoning": "test"})


def _candidate_bundles(n: int) -> list[Bundle]:
    """``n`` distinct non-empty bundles, deterministic and stable in order.

    Singletons first, then pairs, then triples, etc. -- guaranteed distinct
    since each is built from :func:`itertools.combinations`, unlike a naive
    prefix-repeat scheme which can silently produce duplicate frozensets
    (and thus fewer dict entries than bundles requested).
    """
    items = list(ITEM_DESCRIPTIONS)
    bundles: list[Bundle] = []
    for size in range(1, len(items) + 1):
        for combo in combinations(items, size):
            bundles.append(frozenset(combo))
            if len(bundles) >= n:
                return bundles[:n]
    if len(bundles) < n:
        raise ValueError(f"only {len(bundles)} distinct bundles available from {len(items)} items, need {n}")
    return bundles[:n]


def _call_generate(
    *,
    n_generated: int,
    max_bundles: int | None,
    client_max_tokens: int | None = None,
    logger: LlmCallLogger | None = None,
    bidder_id: str = "bidder_1",
) -> tuple[dict[Bundle, float], list]:
    client = MockLlmClient([_EMPTY_PV_RESPONSE])
    if client_max_tokens is not None:
        client.max_tokens = client_max_tokens

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = generate_provisional_valuations(
            client=client,
            scenario_description="A test auction.",
            item_descriptions=ITEM_DESCRIPTIONS,
            nl_question="What do you want?",
            nl_answer="I want stuff.",
            candidate_bundles=_candidate_bundles(n_generated),
            logger=logger,
            bidder_id=bidder_id,
            max_bundles=max_bundles,
        )
    return result, caught


# ---------------------------------------------------------------------------
# compute_pv_candidate_bundle_stats: pure truncation-decision tests
# ---------------------------------------------------------------------------

def test_stats_no_cap_means_no_truncation():
    bundles = _candidate_bundles(63)
    stats = compute_pv_candidate_bundle_stats(bundles, None)

    assert stats == PvCandidateBundleStats(
        candidate_bundles_generated=63,
        candidate_bundles_sent_to_pv=63,
        candidate_bundles_truncated=False,
        candidate_truncation_reason=None,
        max_candidate_bundles=None,
    )


def test_stats_explicit_cap_below_generated_truncates():
    bundles = _candidate_bundles(63)
    stats = compute_pv_candidate_bundle_stats(bundles, 56)

    assert stats.candidate_bundles_generated == 63
    assert stats.candidate_bundles_sent_to_pv == 56
    assert stats.candidate_bundles_truncated is True
    assert stats.max_candidate_bundles == 56
    assert "56" in stats.candidate_truncation_reason
    assert "63" in stats.candidate_truncation_reason


def test_stats_explicit_cap_above_generated_no_truncation():
    bundles = _candidate_bundles(10)
    stats = compute_pv_candidate_bundle_stats(bundles, 50)

    assert stats.candidate_bundles_generated == 10
    assert stats.candidate_bundles_sent_to_pv == 10
    assert stats.candidate_bundles_truncated is False
    assert stats.candidate_truncation_reason is None
    assert stats.max_candidate_bundles == 50


def test_stats_explicit_cap_equal_to_generated_no_truncation():
    bundles = _candidate_bundles(56)
    stats = compute_pv_candidate_bundle_stats(bundles, 56)

    assert stats.candidate_bundles_sent_to_pv == 56
    assert stats.candidate_bundles_truncated is False


# ---------------------------------------------------------------------------
# generate_provisional_valuations: no explicit cap -> everything is sent
# ---------------------------------------------------------------------------

def test_no_explicit_cap_sends_all_generated_candidates_even_with_small_token_budget():
    """63 generated candidates must remain 63 sent candidates when uncapped.

    Uses a deliberately small ``max_tokens`` (500) so the old auto-derived
    cap (`(500-200)//50 = 6`) would previously have silently truncated this
    down to 6 bundles. It must not anymore.
    """
    result, caught = _call_generate(n_generated=63, max_bundles=None, client_max_tokens=500)

    assert len(result) == 63

    messages = [str(w.message) for w in caught]
    assert any("no truncation applied" in m for m in messages)
    assert any("--max-candidate-bundles was not set" in m for m in messages)
    assert any("candidate bundle count is 63" in m for m in messages)
    # Must never claim to have truncated anything.
    assert not any("Truncating" in m for m in messages)


def test_no_explicit_cap_and_no_client_max_tokens_is_silent():
    """No cap, no max_tokens on the client at all -> no warning, full send."""
    result, caught = _call_generate(n_generated=63, max_bundles=None, client_max_tokens=None)

    assert len(result) == 63
    assert len(caught) == 0


def test_no_explicit_cap_small_candidate_list_is_silent():
    """A small candidate list under a tight token budget shouldn't warn at all."""
    result, caught = _call_generate(n_generated=3, max_bundles=None, client_max_tokens=500)

    assert len(result) == 3
    assert len(caught) == 0


# ---------------------------------------------------------------------------
# generate_provisional_valuations: explicit cap -> deterministic truncation
# ---------------------------------------------------------------------------

def test_explicit_cap_truncates_deterministically_to_56():
    result, caught = _call_generate(n_generated=63, max_bundles=56)

    assert len(result) == 56

    messages = [str(w.message) for w in caught]
    assert any("truncating 63" in m.lower() for m in messages)
    assert any("56" in m for m in messages)


def test_explicit_cap_larger_than_generated_count_no_truncation():
    result, caught = _call_generate(n_generated=63, max_bundles=100)

    assert len(result) == 63
    messages = [str(w.message) for w in caught]
    assert not any("truncat" in m.lower() for m in messages)


def test_explicit_cap_with_tight_token_budget_warns_without_lying_about_the_flag():
    """If an explicit cap is set but still exceeds the token estimate, the

    warning must not claim --max-candidate-bundles was unset.
    """
    result, caught = _call_generate(
        n_generated=63, max_bundles=60, client_max_tokens=500
    )

    assert len(result) == 60  # 63 > 60, so the explicit cap does truncate to 60
    messages = [str(w.message) for w in caught]
    assert any("truncating 63" in m.lower() and "60" in m for m in messages)


# ---------------------------------------------------------------------------
# Log metadata: candidate_bundles_* fields land in the LlmCallRecord
# ---------------------------------------------------------------------------

def _read_last_log_record(path) -> dict:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


def test_log_metadata_records_no_truncation_when_uncapped(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    logger = LlmCallLogger(log_path)

    _call_generate(n_generated=63, max_bundles=None, client_max_tokens=500, logger=logger)

    record = _read_last_log_record(log_path)
    parsed = record["parsed_response"]
    assert parsed["candidate_bundles_generated"] == 63
    assert parsed["candidate_bundles_sent_to_pv"] == 63
    assert parsed["candidate_bundles_truncated"] is False
    assert parsed["candidate_truncation_reason"] is None
    assert parsed["max_candidate_bundles"] is None


def test_log_metadata_records_truncation_when_capped(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    logger = LlmCallLogger(log_path)

    _call_generate(n_generated=63, max_bundles=56, logger=logger)

    record = _read_last_log_record(log_path)
    parsed = record["parsed_response"]
    assert parsed["candidate_bundles_generated"] == 63
    assert parsed["candidate_bundles_sent_to_pv"] == 56
    assert parsed["candidate_bundles_truncated"] is True
    assert "56" in parsed["candidate_truncation_reason"]
    assert parsed["max_candidate_bundles"] == 56


# ---------------------------------------------------------------------------
# Proxy-level integration: LlmInferredXorProxy.build_provisional_valuations
# ---------------------------------------------------------------------------

def _make_proxy_with_nl_transcript() -> LlmInferredXorProxy:
    person = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A test auction.",
        person_seed="Values useful combinations.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=MockLlmClient([_EMPTY_PV_RESPONSE]),
    )
    proxy = LlmInferredXorProxy(bidder_id="bidder_1", person=person)
    proxy.nl_transcript.append(("What do you want?", "I want stuff."))
    return proxy


def test_proxy_build_provisional_valuations_uncapped_keeps_all_candidates():
    proxy = _make_proxy_with_nl_transcript()
    proxy.person.client.max_tokens = 500  # would have auto-capped to 6 previously

    with pytest.warns(UserWarning, match="no truncation applied"):
        raw_values = proxy.build_provisional_valuations(_candidate_bundles(63))

    assert len(raw_values) == 63
    assert proxy.last_pv_candidate_stats.candidate_bundles_generated == 63
    assert proxy.last_pv_candidate_stats.candidate_bundles_sent_to_pv == 63
    assert proxy.last_pv_candidate_stats.candidate_bundles_truncated is False


def test_proxy_build_provisional_valuations_respects_explicit_cap():
    proxy = _make_proxy_with_nl_transcript()

    with pytest.warns(UserWarning, match="truncating 63"):
        raw_values = proxy.build_provisional_valuations(
            _candidate_bundles(63), max_candidate_bundles=56
        )

    assert len(raw_values) == 56
    assert proxy.last_pv_candidate_stats.candidate_bundles_sent_to_pv == 56
    assert proxy.last_pv_candidate_stats.candidate_bundles_truncated is True


def test_proxy_handle_event_infer_provisional_values_threads_max_candidate_bundles():
    proxy = _make_proxy_with_nl_transcript()

    with pytest.warns(UserWarning, match="truncating 63"):
        response = proxy.handle_event(
            ProxyElicitationEvent(
                event_type=INFER_PROVISIONAL_VALUES,
                bidder_id="bidder_1",
                mechanism="init",
                payload={
                    "candidate_bundles": _candidate_bundles(63),
                    "max_candidate_bundles": 56,
                },
            )
        )

    assert len(response.payload["raw_values"]) == 56
    assert response.state_delta["candidate_bundles_generated"] == 63
    assert response.state_delta["candidate_bundles_sent_to_pv"] == 56
    assert response.state_delta["candidate_bundles_truncated"] is True


def test_proxy_handle_event_infer_provisional_values_uncapped_by_default():
    proxy = _make_proxy_with_nl_transcript()

    response = proxy.handle_event(
        ProxyElicitationEvent(
            event_type=INFER_PROVISIONAL_VALUES,
            bidder_id="bidder_1",
            mechanism="init",
            payload={"candidate_bundles": _candidate_bundles(63)},
        )
    )

    assert len(response.payload["raw_values"]) == 63
    assert response.state_delta["candidate_bundles_sent_to_pv"] == 63
    assert response.state_delta["candidate_bundles_truncated"] is False
