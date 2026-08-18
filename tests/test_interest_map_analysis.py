"""Offline analysis of interest maps and candidate support.

Covers the support-reduction and reliability metrics, aggregation across
seeds with the shared anchor counted once, and that packs projected from a
master pack are not double-counted in the totals.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from auctionlab.experiments.interest_map_analysis import (
    aggregate_cases,
    aggregate_summary,
    discover_frozen_packs,
    load_interest_map_results,
    write_interest_map_tables,
)
from auctionlab.llm.frozen_elicitation import (
    BidderElicitationData,
    FrozenElicitationPack,
    ModelProvenance,
    write_frozen_elicitation_pack,
)
from auctionlab.llm.provisional_valuations import (
    PvCandidateBundleStats,
    PvChunkStats,
)
from auctionlab.llm.schemas import LlmInterestMap


def _write_fixture(path: Path) -> None:
    entry = BidderElicitationData(
        nl_question="What do you want?",
        nl_answer="A or B, but only one. I do not want C.",
        interest_map=LlmInterestMap(
            interested_items=["A", "B"],
            excluded_items=["C"],
            substitute_groups=[
                {
                    "items": ["A", "B"],
                    "acquisition_mode": "choose_one",
                    "evidence": "only one",
                    "mode_explicitly_stated": True,
                }
            ],
            reasoning="A and B are single-choice alternatives.",
        ),
        candidate_bundles=[
            frozenset({"A"}),
            frozenset({"B"}),
        ],
        raw_pv_values={
            frozenset({"A"}): 10.0,
            frozenset({"B"}): 8.0,
        },
        pv_candidate_stats=PvCandidateBundleStats(
            candidate_bundles_generated=2,
            candidate_bundles_sent_to_pv=2,
            candidate_bundles_truncated=False,
            candidate_truncation_reason=None,
            max_candidate_bundles=None,
        ),
        pv_chunk_stats=PvChunkStats(
            pv_chunk_size=1,
            pv_chunks=2,
            candidate_count=2,
            per_chunk_bundle_counts=(1, 1),
            chunking_used=True,
        ),
        interest_map_candidate_count_before_filter=3,
        interest_map_candidate_count_after_filter=2,
        interest_map_accuracy={
            "item_precision": 1.0,
            "item_recall": 1.0,
            "item_f1": 1.0,
            "group_item_set_precision": 1.0,
            "group_item_set_recall": 1.0,
            "mode_accuracy_on_matched_groups": 1.0,
            "exact_group_and_mode_recall": 1.0,
            "choose_one_precision": 1.0,
            "choose_one_recall": 1.0,
            "complement_group_precision": 1.0,
            "complement_group_recall": 1.0,
            "candidate_set_precision": 1.0,
            "candidate_set_recall": 1.0,
            "oracle_candidate_count": 2,
            "missed_positive_items": [],
            "false_positive_items": [],
            "missed_oracle_candidate_count": 0,
            "extra_candidate_count": 0,
            "dangerous_false_exclusivity_count": 0,
        },
        person_answer_verification={"passed": True},
        person_answer_verification_history=[
            {"passed": False},
            {"passed": True},
        ],
        person_answer_attempt_count=2,
    )
    pack = FrozenElicitationPack(
        format="auctionlab.frozen_elicitation",
        version=1,
        scenario_name="fixture",
        scenario_fingerprint="fixture-fingerprint",
        scenario_spec_path=None,
        scenario_spec_sha256=None,
        scenario_seed=2,
        selection_policy="coverage_stratified",
        items=("A", "B", "C"),
        bidder_ids=("b1",),
        environment_model=ModelProvenance("test", "environment"),
        person_model=ModelProvenance("test", "person"),
        proxy_model=ModelProvenance("test", "proxy"),
        generation_settings={},
        bidders={"b1": entry},
        generation_calls=(
            {
                "bidder_id": "b1",
                "prompt_type": "proxy_interest_map",
                "attempt": 1,
                "success": False,
            },
            {
                "bidder_id": "b1",
                "prompt_type": "proxy_interest_map",
                "attempt": 2,
                "success": True,
            },
        ),
    )
    write_frozen_elicitation_pack(pack, path)


def test_support_reduction_and_reliability_metrics(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "seed_2"
        / "joint_3x1"
        / "frozen_elicitation.json"
    )
    _write_fixture(path)

    bidders, invalid = load_interest_map_results(tmp_path)
    cases = aggregate_cases(bidders)

    assert invalid == []
    assert len(bidders) == 1
    bidder = bidders[0]
    assert bidder["full_powerset_count"] == 7
    assert bidder["interested_item_powerset_count"] == 3
    assert bidder["inferred_candidate_count"] == 2
    assert bidder["exclusion_reduction_count"] == 4
    assert bidder["substitute_reduction_count"] == 1
    assert bidder["total_reduction_pct"] == pytest.approx(100 * 5 / 7)
    assert bidder["pv_api_call_count"] == 2
    assert bidder["person_answer_first_attempt_success"] is False
    assert bidder["person_answer_repair_count"] == 1
    assert bidder["interest_map_first_attempt_success"] is False
    assert bidder["interest_map_final_success"] is True
    assert bidder["interest_map_retry_count"] == 1

    assert cases[0]["full_powerset_count"] == 7
    assert cases[0]["inferred_candidate_count"] == 2
    assert cases[0]["mean_item_f1"] == 1.0
    assert cases[0]["person_answer_first_attempt_success_rate"] == 0.0


def test_summary_aggregates_seed_range_and_anchor(tmp_path: Path) -> None:
    case_rows = [
        {
            "seed": seed,
            "series": "anchor",
            "x_value": 3,
            "num_goods": 3,
            "num_bidders": 3,
            **{metric: 1.0 for metric in (
                "full_powerset_count",
                "interested_item_powerset_count",
                "inferred_candidate_count",
                "oracle_candidate_count",
                "pv_candidate_count_sent",
                "pv_api_call_count",
                "exclusion_reduction_pct",
                "substitute_reduction_pct_of_full",
                "substitute_reduction_pct_of_interested_support",
                "total_reduction_pct",
                "total_missed_positive_items",
                "total_missed_oracle_candidates",
                "total_dangerous_false_exclusivity",
                "person_answer_first_attempt_success_rate",
                "mean_person_answer_repairs",
                "interest_map_first_attempt_success_rate",
                "mean_interest_map_retries",
                *(f"mean_{field}" for field in (
                    "item_precision",
                    "item_recall",
                    "item_f1",
                    "group_item_set_precision",
                    "group_item_set_recall",
                    "mode_accuracy_on_matched_groups",
                    "exact_group_and_mode_recall",
                    "choose_one_precision",
                    "choose_one_recall",
                    "complement_group_precision",
                    "complement_group_recall",
                    "candidate_set_precision",
                    "candidate_set_recall",
                )),
            )},
        }
        for seed in (0, 1)
    ]
    summary = aggregate_summary(case_rows)

    assert {(row["series"], row["x_value"]) for row in summary} == {
        ("goods", 3),
        ("bidders", 3),
        ("joint", 3),
    }
    assert all(row["num_cases"] == 2 for row in summary)
    assert all(row["num_seeds"] == 2 for row in summary)


def test_projected_packs_prevent_master_double_counting(
    tmp_path: Path,
) -> None:
    projected = (
        tmp_path
        / "seed_0"
        / "goods_3x1"
        / "frozen_elicitation.json"
    )
    master = (
        tmp_path
        / "seed_0"
        / "masters"
        / "goods_3_master_3x1.json"
    )
    _write_fixture(projected)
    _write_fixture(master)

    assert discover_frozen_packs(tmp_path) == [projected]
    assert set(discover_frozen_packs(tmp_path, include_masters=True)) == {
        projected,
        master,
    }


def test_writes_all_analysis_tables(tmp_path: Path) -> None:
    pack_path = (
        tmp_path
        / "packs"
        / "seed_2"
        / "goods_3x1"
        / "frozen_elicitation.json"
    )
    _write_fixture(pack_path)
    bidder_rows, invalid = load_interest_map_results(tmp_path / "packs")
    case_rows = aggregate_cases(bidder_rows)
    summary_rows = aggregate_summary(case_rows)

    paths = write_interest_map_tables(
        tmp_path / "analysis",
        bidder_rows,
        case_rows,
        summary_rows,
        invalid,
    )

    assert all(path.exists() for path in paths)
    with paths[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["full_powerset_count"] == "7"
    assert rows[0]["inferred_candidate_count"] == "2"

