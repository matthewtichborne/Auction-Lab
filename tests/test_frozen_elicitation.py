from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from auctionlab.instances.base import AuctionInstance
from auctionlab.instances.nl_types import NaturalLanguageAuctionScenario
from auctionlab.llm.frozen_elicitation import (
    BidderElicitationData,
    ModelProvenance,
    build_frozen_elicitation_pack,
    load_frozen_elicitation_pack,
    project_frozen_pack_to_bidders,
    validate_pack_for_scenario,
    write_frozen_elicitation_pack,
)
from auctionlab.llm.provisional_valuations import (
    PvCandidateBundleStats,
    PvChunkStats,
)
from auctionlab.llm.schemas import LlmInterestMap


def _scenario() -> NaturalLanguageAuctionScenario:
    return NaturalLanguageAuctionScenario(
        name="toy_frozen",
        instance=AuctionInstance(
            items=["A", "B"],
            bidder_ids=["b1"],
            valuations={
                "b1": {
                    frozenset({"A"}): 10.0,
                    frozenset({"B"}): 0.0,
                    frozenset({"A", "B"}): 10.0,
                }
            },
        ),
        scenario_description="A toy auction.",
        item_descriptions={"A": "Useful A", "B": "Unwanted B"},
        person_seeds={
            "b1": "Interested in A, not B. Maximum total WTP is $10."
        },
        seed_type="structured",
        metadata={
            "scenario_seed": 3,
            "environment_generation_provider": "gemini",
            "environment_generation_model": "environment-model",
        },
    )


def _entries() -> dict[str, BidderElicitationData]:
    return {
        "b1": BidderElicitationData(
            nl_question="What do you want?",
            nl_answer="I want A, not B, up to $10.",
            interest_map=LlmInterestMap(
                interested_items=["A"],
                excluded_items=["B"],
                complementary_groups=[],
                substitute_groups=[],
                budget_hint=10.0,
                reasoning="A positive, B excluded.",
            ),
            candidate_bundles=[frozenset({"A"})],
            raw_pv_values={frozenset({"A"}): 9.0},
            pv_candidate_stats=PvCandidateBundleStats(
                candidate_bundles_generated=1,
                candidate_bundles_sent_to_pv=1,
                candidate_bundles_truncated=False,
                candidate_truncation_reason=None,
                max_candidate_bundles=None,
            ),
            pv_chunk_stats=PvChunkStats(
                pv_chunk_size=0,
                pv_chunks=1,
                candidate_count=1,
                per_chunk_bundle_counts=(1,),
                chunking_used=False,
            ),
        )
    }


def _pack(scenario: NaturalLanguageAuctionScenario | None = None):
    scenario = scenario or _scenario()
    return build_frozen_elicitation_pack(
        scenario=scenario,
        entries=_entries(),
        scenario_spec_path=None,
        selection_policy="coverage_stratified",
        person_model=ModelProvenance("anthropic", "person-model", 0.0),
        proxy_model=ModelProvenance("openai-compatible", "proxy-model", 0.0),
        generation_settings={
            "use_interest_map": True,
            "use_provisional_valuations": True,
        },
        generation_calls=[
            {
                "prompt_type": "proxy_interest_map",
                "raw_response": '{"interested_items":["A"]}',
                "parsed_response": {"interested_items": ["A"]},
            }
        ],
    )


def test_round_trip_preserves_raw_values_models_and_calls(tmp_path):
    path = tmp_path / "pack.json"
    write_frozen_elicitation_pack(_pack(), path)
    loaded = load_frozen_elicitation_pack(path)

    assert loaded.environment_model.model == "environment-model"
    assert loaded.person_model.model == "person-model"
    assert loaded.proxy_model.model == "proxy-model"
    assert loaded.bidders["b1"].raw_pv_values == {
        frozenset({"A"}): 9.0
    }
    assert loaded.generation_calls[0]["raw_response"].startswith("{")
    validate_pack_for_scenario(loaded, _scenario())


def test_pack_json_does_not_store_hidden_valuation_table(tmp_path):
    path = tmp_path / "pack.json"
    write_frozen_elicitation_pack(_pack(), path)
    payload = json.loads(path.read_text())

    assert "valuations" not in payload["scenario"]
    assert payload["bidders"]["b1"]["raw_provisional_values"] == [
        {"bundle": ["A"], "value": 9.0}
    ]


def test_mismatched_environment_fingerprint_is_rejected():
    scenario = _scenario()
    changed = NaturalLanguageAuctionScenario(
        **{
            **scenario.__dict__,
            "person_seeds": {"b1": "Changed qualitative disclosure."},
        }
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_pack_for_scenario(_pack(), changed)


def test_imperfect_person_interest_inference_is_recorded_not_truth_corrected():
    entries = _entries()
    entries["b1"].interest_map = LlmInterestMap(
        interested_items=["A", "B"],
        excluded_items=[],
        reasoning="Incorrectly makes B positive.",
    )
    entries["b1"].interest_map_accuracy = {
        "item_precision": 0.5,
        "item_recall": 1.0,
    }
    pack = build_frozen_elicitation_pack(
        scenario=_scenario(),
        entries=entries,
        scenario_spec_path=None,
        selection_policy="coverage_stratified",
        person_model=ModelProvenance("x", "person", 0.0),
        proxy_model=ModelProvenance("y", "proxy", 0.0),
        generation_settings={},
        generation_calls=[],
    )

    assert pack.bidders["b1"].interest_map.interested_items == ["A", "B"]
    assert pack.bidders["b1"].interest_map_accuracy["item_precision"] == 0.5


def test_closed_world_pack_rejects_unclassified_item():
    entries = _entries()
    entries["b1"].interest_map = LlmInterestMap(
        interested_items=["A"],
        excluded_items=[],
        reasoning="B omitted.",
    )
    with pytest.raises(ValueError, match="classify every item"):
        build_frozen_elicitation_pack(
            scenario=_scenario(),
            entries=entries,
            scenario_spec_path=None,
            selection_policy="coverage_stratified",
            person_model=ModelProvenance("x", "person", 0.0),
            proxy_model=ModelProvenance("y", "proxy", 0.0),
            generation_settings={},
            generation_calls=[],
        )


def _two_bidder_projection_fixture():
    scenario = NaturalLanguageAuctionScenario(
        name="toy_parent",
        instance=AuctionInstance(
            items=["A", "B"],
            bidder_ids=["b1", "b2"],
            valuations={
                "b1": {frozenset({"A"}): 10.0},
                "b2": {frozenset({"B"}): 8.0},
            },
        ),
        scenario_description="Projection test.",
        item_descriptions={"A": "A", "B": "B"},
        person_seeds={"b1": "A person", "b2": "B person"},
        seed_type="structured",
        metadata={"scenario_seed": 7},
    )
    entries = {
        "b1": BidderElicitationData(
            nl_question="What do you want?",
            nl_answer="A only.",
            interest_map=LlmInterestMap(
                interested_items=["A"],
                excluded_items=["B"],
                reasoning="A only.",
            ),
            candidate_bundles=[frozenset({"A"})],
            raw_pv_values={frozenset({"A"}): 9.0},
        ),
        "b2": BidderElicitationData(
            nl_question="What do you want?",
            nl_answer="B only.",
            interest_map=LlmInterestMap(
                interested_items=["B"],
                excluded_items=["A"],
                reasoning="B only.",
            ),
            candidate_bundles=[frozenset({"B"})],
            raw_pv_values={frozenset({"B"}): 7.0},
        ),
    }
    pack = build_frozen_elicitation_pack(
        scenario=scenario,
        entries=entries,
        scenario_spec_path=None,
        selection_policy="coverage_stratified",
        person_model=ModelProvenance("x", "person", 0.0),
        proxy_model=ModelProvenance("y", "proxy", 0.0),
        generation_settings={},
        generation_calls=[
            {"bidder_id": "b1", "total_tokens": 10},
            {"bidder_id": "b2", "total_tokens": 20},
            {"prompt_type": "shared", "total_tokens": 3},
        ],
    )
    return scenario, pack


def test_project_pack_to_ordered_bidder_subset_filters_calls_and_validates():
    parent_scenario, parent = _two_bidder_projection_fixture()
    target = NaturalLanguageAuctionScenario(
        **{
            **parent_scenario.__dict__,
            "name": "toy_subset",
            "instance": AuctionInstance(
                items=["A", "B"],
                bidder_ids=["b1"],
                valuations={"b1": {frozenset({"A"}): 10.0}},
            ),
            "person_seeds": {"b1": "A person"},
        }
    )

    projected = project_frozen_pack_to_bidders(parent, target)

    assert projected.bidder_ids == ("b1",)
    assert set(projected.bidders) == {"b1"}
    assert [row.get("bidder_id") for row in projected.generation_calls] == [
        "b1",
        None,
    ]
    assert projected.generation_settings["projection"]["kind"] == (
        "ordered_bidder_subset"
    )
    validate_pack_for_scenario(projected, target)


def test_project_pack_rejects_goods_projection():
    parent_scenario, parent = _two_bidder_projection_fixture()
    target = NaturalLanguageAuctionScenario(
        **{
            **parent_scenario.__dict__,
            "name": "toy_changed_goods",
            "instance": AuctionInstance(
                items=["A"],
                bidder_ids=["b1"],
                valuations={"b1": {frozenset({"A"}): 10.0}},
            ),
            "item_descriptions": {"A": "A"},
            "person_seeds": {"b1": "A person"},
        }
    )
    with pytest.raises(ValueError, match="only change bidders"):
        project_frozen_pack_to_bidders(parent, target)


def test_cli_replays_pack_without_initial_llm_calls(tmp_path):
    from auctionlab.instances.structured import make_pc_build_scenario
    from auctionlab.llm.interest_map import (
        generate_candidate_bundles_from_interest_map,
    )

    scenario = make_pc_build_scenario(4, 4, seed=0)
    entries: dict[str, BidderElicitationData] = {}
    for bidder_id in scenario.instance.bidder_ids:
        interested = [
            item
            for item in scenario.instance.items
            if scenario.instance.valuations[bidder_id].get(
                frozenset({item}), 0.0
            ) > 0
        ]
        excluded = [
            item
            for item in scenario.instance.items
            if item not in interested
        ]
        interest_map = LlmInterestMap(
            interested_items=interested,
            excluded_items=excluded,
            reasoning="Frozen test map.",
        )
        candidates = generate_candidate_bundles_from_interest_map(
            interest_map, list(scenario.instance.items)
        )
        raw = {
            bundle: scenario.instance.valuations[bidder_id].get(bundle, 0.0)
            for bundle in candidates
        }
        entries[bidder_id] = BidderElicitationData(
            nl_question="Describe all positive interests and exclusions.",
            nl_answer="Frozen answer.",
            interest_map=interest_map,
            candidate_bundles=candidates,
            raw_pv_values=raw,
        )

    pack = build_frozen_elicitation_pack(
        scenario=scenario,
        entries=entries,
        scenario_spec_path=None,
        selection_policy="prefix",
        person_model=ModelProvenance("gemini", "person-model", 0.0),
        proxy_model=ModelProvenance("gemini", "proxy-model", 0.0),
        generation_settings={
            "use_interest_map": True,
            "use_provisional_valuations": True,
        },
        generation_calls=[],
    )
    pack_path = tmp_path / "pack.json"
    write_frozen_elicitation_pack(pack, pack_path)
    log_dir = tmp_path / "run"
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "examples/run_live_llm_curated_batch.py",
            "--scenario", "pc_build",
            "--num-goods", "4",
            "--num-bidders", "4",
            "--scenario-seed", "0",
            "--seed-type", "structured",
            "--skip-baselines",
            "--sealed-elicitation-rounds", "1",
            "--sealed-feedback-rule", "competitive",
            "--elicitation-pack", str(pack_path),
            "--person-query-mode", "deterministic",
            "--llm-cache-mode", "off",
            "--log-dir", str(log_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "frozen elicitation replay" in completed.stdout
    calls_path = log_dir / "calls.jsonl"
    call_rows = (
        [
            json.loads(line)
            for line in calls_path.read_text().splitlines()
        ]
        if calls_path.exists()
        else []
    )
    assert all(
        row["prompt_type"] not in {
            "nl_question",
            "proxy_initial_question",
            "proxy_interest_map",
            "proxy_provisional_valuations",
        }
        for row in call_rows
    )
