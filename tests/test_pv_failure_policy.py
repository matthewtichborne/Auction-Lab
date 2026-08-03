"""Tests for ``--pv-failure-policy`` and the mass-direct-value-query
regression it fixes.

Exercises ``compute_elicitation_cache``/``PvGenerationFailedError`` from
``examples/run_live_llm_curated_batch.py`` (imported by file path, mirroring
``tests/test_run_live_llm_curated_batch_cli.py``'s loader, since the script
is not an importable package module). No live LLM/API calls -- everything
uses ``MockLlmClient``, a deterministic in-memory fake.

The core regression under test: before this fix, a failed provisional-
valuation (PV) call left a bidder's cached XOR bid empty, so
``LlmAuctionProxyAdapter.current_bid()`` fell through to querying every
candidate bundle individually (a live 8x8 run saw ~191 such value queries
for one bidder) -- silently switching the elicitation regime from bulk PV
to mass direct querying and contaminating VQ counts. ``--pv-failure-policy
raise`` (the new default) aborts instead; ``--pv-failure-policy zero``
(the old debugging behaviour) now zero-initialises the bidder's cached bid
so the same mass-query fallback path is never reached either.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.logging import LlmCallLogger
from auctionlab.llm.frozen_elicitation import BidderElicitationData
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmAuctionProxyAdapter, LlmInferredXorProxy

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "run_live_llm_curated_batch.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_run_live_llm_curated_batch_pv_failure_test_module", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_module = _load_module()
compute_elicitation_cache = _module.compute_elicitation_cache
PvGenerationFailedError = _module.PvGenerationFailedError
provisional_value_scale_diagnostics = (
    _module.provisional_value_scale_diagnostics
)


ITEM_DESCRIPTIONS = {"A": "Item A", "B": "Item B"}

_QUESTION_RESPONSE = json.dumps({"question": "What do you want?"})
_ANSWER_RESPONSE = json.dumps({"answer": "I want A and B together."})
_INTEREST_MAP_RESPONSE = json.dumps({
    "interested_items": ["A", "B"],
    "excluded_items": [],
    "complementary_groups": [],
    "substitute_groups": [],
    "budget_hint": None,
    "reasoning": "test",
})
_MALFORMED_PV_RESPONSE = "not valid json at all"


def _pv_response(entries: dict[str, float]) -> str:
    return json.dumps({
        "valuations": [{"bundle": list(b), "value": v} for b, v in entries.items()],
        "reasoning": "test",
    })


def _make_scenario():
    return SimpleNamespace(instance=SimpleNamespace(items=["A", "B"]), name="test_scenario")


def _make_scenario_with_values():
    return SimpleNamespace(
        instance=SimpleNamespace(
            items=["A", "B"],
            valuations={
                "bidder_1": {
                    frozenset({"A"}): 100.0,
                    frozenset({"B"}): 50.0,
                    frozenset({"A", "B"}): 150.0,
                }
            },
        ),
        name="test_scenario",
    )


def _make_persons(person_client: MockLlmClient | None = None) -> dict[str, LlmPersonSimulator]:
    person_client = person_client or MockLlmClient([
        _QUESTION_RESPONSE, _ANSWER_RESPONSE, _INTEREST_MAP_RESPONSE,
    ])
    person = LlmPersonSimulator(
        bidder_id="bidder_1",
        scenario_description="A test auction.",
        person_seed="Wants A and B together.",
        item_descriptions=ITEM_DESCRIPTIONS,
        client=person_client,
    )
    return {"bidder_1": person}


def test_pv_scale_diagnostic_flags_gross_bundle_and_singleton_inflation():
    report = provisional_value_scale_diagnostics(
        scenario=_make_scenario_with_values(),
        bidder_id="bidder_1",
        raw_pv_values={
            frozenset({"A"}): 400.0,
            frozenset({"B"}): 100.0,
            frozenset({"A", "B"}): 600.0,
        },
    )

    assert report["max_value_ratio"] == 4.0
    assert report["max_singleton_ratio"] == 4.0
    assert set(report["quality_flags"]) == {
        "pv_max_value_scale_exceeds_ground_truth",
        "pv_singleton_scale_exceeds_ground_truth",
    }


def test_pv_scale_diagnostic_does_not_flag_values_within_threshold():
    report = provisional_value_scale_diagnostics(
        scenario=_make_scenario_with_values(),
        bidder_id="bidder_1",
        raw_pv_values={
            frozenset({"A"}): 120.0,
            frozenset({"A", "B"}): 180.0,
        },
    )

    assert report["quality_flags"] == ()


def test_fixed_disclosure_regenerates_proxy_state_without_person_call():
    person_client = MockLlmClient([])
    persons = _make_persons(person_client)
    cache = compute_elicitation_cache(
        scenario=_make_scenario_with_values(),
        persons=persons,
        use_provisional_valuations=True,
        max_candidate_bundles=None,
        pv_client=MockLlmClient([
            _pv_response({"A": 100.0, "B": 50.0, "AB": 150.0})
        ]),
        opening_question=None,
        interest_map_client=MockLlmClient([_INTEREST_MAP_RESPONSE]),
        pv_failure_policy="raise",
        fixed_disclosures={
            "bidder_1": BidderElicitationData(
                nl_question="What do you want?",
                nl_answer="I want A and B together.",
                interest_map=None,
                candidate_bundles=[],
                raw_pv_values=None,
            )
        },
    )

    assert cache["bidder_1"].nl_answer == "I want A and B together."
    assert cache["bidder_1"].raw_pv_values is not None


# ---------------------------------------------------------------------------
# pv_failure_policy == "raise" (the default)
# ---------------------------------------------------------------------------

class TestPvFailurePolicyRaise:
    def test_raises_pv_generation_failed_error_with_diagnostic_context(self):
        persons = _make_persons()
        pv_client = MockLlmClient([_MALFORMED_PV_RESPONSE])

        with pytest.raises(PvGenerationFailedError) as exc_info:
            compute_elicitation_cache(
                scenario=_make_scenario(),
                persons=persons,
                use_provisional_valuations=True,
                max_candidate_bundles=None,
                pv_client=pv_client,
                pv_chunk_size=0,
                pv_failure_policy="raise",
                pv_max_tokens=6000,
            )

        err = exc_info.value
        assert err.bidder_id == "bidder_1"
        assert err.candidate_count == 3  # {A}, {B}, {A,B}
        assert err.pv_chunk_size == 0
        assert err.pv_max_tokens == 6000
        assert err.original_exception is not None
        message = str(err)
        assert "bidder_1" in message
        assert "pv_max_tokens=6000" in message
        assert "pv_chunk_size=0" in message
        assert "--pv-max-tokens" in message or "--pv-chunk-size" in message

    def test_default_policy_is_raise_not_zero(self):
        """compute_elicitation_cache's own default keyword must match the
        CLI's default (raise) -- a caller relying on defaults must never
        silently fall into the debugging zero-fallback."""
        persons = _make_persons()
        pv_client = MockLlmClient([_MALFORMED_PV_RESPONSE])

        with pytest.raises(PvGenerationFailedError):
            compute_elicitation_cache(
                scenario=_make_scenario(),
                persons=persons,
                use_provisional_valuations=True,
                max_candidate_bundles=None,
                pv_client=pv_client,
            )

    def test_raise_never_touches_value_queries(self):
        """No value_query must ever be issued as part of the raise path."""
        person_client = MockLlmClient([
            _QUESTION_RESPONSE, _ANSWER_RESPONSE, _INTEREST_MAP_RESPONSE,
        ])
        persons = _make_persons(person_client)
        pv_client = MockLlmClient([_MALFORMED_PV_RESPONSE])

        with pytest.raises(PvGenerationFailedError):
            compute_elicitation_cache(
                scenario=_make_scenario(),
                persons=persons,
                use_provisional_valuations=True,
                max_candidate_bundles=None,
                pv_client=pv_client,
            )
        # Only the 3 elicitation-phase calls (question, answer, interest
        # map) -- no direct value queries.
        assert len(person_client.calls) == 3


# ---------------------------------------------------------------------------
# pv_failure_policy == "zero" (debugging fallback)
# ---------------------------------------------------------------------------

class TestPvFailurePolicyZero:
    def test_zero_initialises_bidder_and_marks_degraded(self):
        persons = _make_persons()
        pv_client = MockLlmClient([_MALFORMED_PV_RESPONSE])

        cache = compute_elicitation_cache(
            scenario=_make_scenario(),
            persons=persons,
            use_provisional_valuations=True,
            max_candidate_bundles=None,
            pv_client=pv_client,
            pv_failure_policy="zero",
        )

        entry = cache["bidder_1"]
        assert entry.pv_degraded is True
        assert entry.raw_pv_values is not None
        assert set(entry.raw_pv_values.values()) == {0.0}
        assert set(entry.raw_pv_values.keys()) == set(entry.candidate_bundles)

    def test_zero_policy_does_not_trigger_mass_direct_value_queries(self):
        """The regression this feature exists to fix: a PV failure must
        never fall through to LlmAuctionProxyAdapter.current_bid() issuing
        a value_query per candidate bundle."""
        person_client = MockLlmClient([
            _QUESTION_RESPONSE, _ANSWER_RESPONSE, _INTEREST_MAP_RESPONSE,
        ])
        persons = _make_persons(person_client)
        pv_client = MockLlmClient([_MALFORMED_PV_RESPONSE])

        cache = compute_elicitation_cache(
            scenario=_make_scenario(),
            persons=persons,
            use_provisional_valuations=True,
            max_candidate_bundles=None,
            pv_client=pv_client,
            pv_failure_policy="zero",
        )
        entry = cache["bidder_1"]
        assert len(person_client.calls) == 3  # no VQs yet, only elicitation phase

        # Replay into a fresh proxy exactly like make_elicited_proxies() does.
        proxy = LlmInferredXorProxy(bidder_id="bidder_1", person=persons["bidder_1"])
        proxy.replay_elicitation(
            nl_question=entry.nl_question,
            nl_answer=entry.nl_answer,
            interest_map=entry.interest_map,
            provisional_raw_values=entry.raw_pv_values,
        )
        adapter = LlmAuctionProxyAdapter(
            bidder_id="bidder_1",
            proxy=proxy,
            candidate_bundles=entry.candidate_bundles,
        )

        bid = adapter.current_bid()

        # No new LLM calls at all: the cached (zero) bid was already
        # populated by replay_elicitation, so current_bid() must not fall
        # through to querying every candidate bundle individually.
        assert len(person_client.calls) == 3
        assert adapter.stats().value_queries == 0
        assert len(bid.atoms) == len(entry.candidate_bundles)
        assert all(atom.value == 0.0 for atom in bid.atoms)

    def test_zero_policy_records_chunk_metadata_as_none(self):
        """A failed PV call never reaches the chunked generator successfully,
        so there is no chunk_stats to report -- degraded status alone
        signals the debug fallback."""
        persons = _make_persons()
        pv_client = MockLlmClient([_MALFORMED_PV_RESPONSE])

        cache = compute_elicitation_cache(
            scenario=_make_scenario(),
            persons=persons,
            use_provisional_valuations=True,
            max_candidate_bundles=None,
            pv_client=pv_client,
            pv_failure_policy="zero",
        )
        assert cache["bidder_1"].pv_chunk_stats is None


# ---------------------------------------------------------------------------
# Successful PV: not degraded, chunk stats recorded, no mass fallback either
# ---------------------------------------------------------------------------

class TestPvSuccess:
    def test_question_person_answer_interest_map_and_pv_use_separate_clients(
        self, tmp_path
    ):
        question_client = MockLlmClient([_QUESTION_RESPONSE])
        person_client = MockLlmClient([_ANSWER_RESPONSE])
        interest_map_client = MockLlmClient([_INTEREST_MAP_RESPONSE])
        pv_client = MockLlmClient([
            _pv_response({
                frozenset({"A"}): 10.0,
                frozenset({"B"}): 20.0,
                frozenset({"A", "B"}): 40.0,
            })
        ])
        for client, role, model in (
            (question_client, "proxy", "proxy-question-model"),
            (person_client, "person", "person-model"),
            (interest_map_client, "proxy", "proxy-im-model"),
            (pv_client, "proxy", "proxy-pv-model"),
        ):
            client._auctionlab_llm_role = role
            client._auctionlab_provider = "test-provider"
            client._auctionlab_model = model

        logger = LlmCallLogger(tmp_path / "calls.jsonl")
        person = LlmPersonSimulator(
            bidder_id="bidder_1",
            scenario_description="A test auction.",
            person_seed="Wants A and B together.",
            item_descriptions=ITEM_DESCRIPTIONS,
            client=person_client,
            logger=logger,
            model_name="person-model",
            provider_name="test-provider",
            scenario_id="test_scenario",
        )

        compute_elicitation_cache(
            scenario=_make_scenario(),
            persons={"bidder_1": person},
            use_provisional_valuations=True,
            max_candidate_bundles=None,
            question_client=question_client,
            interest_map_client=interest_map_client,
            pv_client=pv_client,
        )

        assert len(question_client.calls) == 1
        assert len(person_client.calls) == 1
        assert len(interest_map_client.calls) == 1
        assert len(pv_client.calls) == 1

        records = [
            json.loads(line)
            for line in (tmp_path / "calls.jsonl").read_text().splitlines()
        ]
        assert [(r["prompt_type"], r["llm_role"]) for r in records] == [
            ("proxy_nl_gen", "proxy"),
            ("nl_question", "person"),
            ("proxy_interest_map", "proxy"),
            ("proxy_provisional_valuations", "proxy"),
        ]
        assert [r["model"] for r in records] == [
            "proxy-question-model",
            "person-model",
            "proxy-im-model",
            "proxy-pv-model",
        ]

    def test_successful_pv_is_not_degraded_and_records_chunk_stats(self):
        persons = _make_persons()
        pv_client = MockLlmClient([
            _pv_response({
                frozenset({"A"}): 100.0,
                frozenset({"B"}): 200.0,
                frozenset({"A", "B"}): 400.0,
            })
        ])

        cache = compute_elicitation_cache(
            scenario=_make_scenario(),
            persons=persons,
            use_provisional_valuations=True,
            max_candidate_bundles=None,
            pv_client=pv_client,
            pv_failure_policy="raise",
        )

        entry = cache["bidder_1"]
        assert entry.pv_degraded is False
        assert entry.raw_pv_values == {
            frozenset({"A"}): 100.0,
            frozenset({"B"}): 200.0,
            frozenset({"A", "B"}): 400.0,
        }
        assert entry.pv_chunk_stats is not None
        assert entry.pv_chunk_stats.chunking_used is False
        assert entry.pv_chunk_stats.pv_chunks == 1

    def test_existing_call_without_new_pv_kwargs_still_works(self):
        """Regression: a caller that omits pv_chunk_size/pv_failure_policy/
        pv_max_tokens/max_parse_retries entirely (pre-chunking call shape)
        must still succeed using the documented defaults."""
        persons = _make_persons()
        pv_client = MockLlmClient([
            _pv_response({
                frozenset({"A"}): 10.0,
                frozenset({"B"}): 20.0,
                frozenset({"A", "B"}): 40.0,
            })
        ])

        cache = compute_elicitation_cache(
            scenario=_make_scenario(),
            persons=persons,
            use_provisional_valuations=True,
            max_candidate_bundles=None,
            pv_client=pv_client,
        )
        assert cache["bidder_1"].raw_pv_values is not None
        assert cache["bidder_1"].pv_degraded is False
