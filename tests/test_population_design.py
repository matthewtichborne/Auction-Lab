"""Population design and sample validation.

Covers the committed 16x16 design and the checks a sampled cell must pass:
every selected good wants at least one bidder, declared structural groups
survive selection, and category, stratum and exclusion diversity are
enforced, so no accepted cell is trivially solvable.
"""

from __future__ import annotations

import json
from pathlib import Path

from auctionlab.instances.population_design import (
    bidder_monetary_semantic_violations,
    coverage_aware_nested_order,
    freeze_validated_nested_orders,
    population_coverage_report,
    sample_economic_report,
    sample_structure_report,
)
from auctionlab.instances.scenario_spec import scenario_profile_spec_from_dict


DESIGN_PATH = Path(
    "scenarios/pc_build_v2/population_design_16x16.json"
)


def _synthetic_population():
    goods = [
        {"id": "CPU_H", "description": "", "category": "cpu"},
        {"id": "CPU_M", "description": "", "category": "cpu"},
        {"id": "CPU_L", "description": "", "category": "cpu"},
        {"id": "GPU_H", "description": "", "category": "gpu"},
        {"id": "GPU_M", "description": "", "category": "gpu"},
        {"id": "MB", "description": "", "category": "motherboard"},
        {"id": "RAM", "description": "", "category": "ram"},
        {"id": "SSD", "description": "", "category": "storage"},
    ]
    good_ids = [good["id"] for good in goods]
    bidders = []
    for index in range(8):
        bidders.append(
            {
                "bidder_id": f"bidder_{index}",
                "role": f"Bidder {index}.",
                "archetype_category": f"stratum_{index % 4}",
                "budget_range": [100, 2000],
                "base_values": {
                    good_id: 100 + 10 * index for good_id in good_ids
                },
                "substitute_groups": [
                    {
                        "items": ["CPU_H", "CPU_M", "CPU_L"],
                        "acquisition_mode": "choose_one",
                        "backup_factor": 0.0,
                    },
                    {
                        "items": ["GPU_H", "GPU_M"],
                        "acquisition_mode": "choose_one",
                        "backup_factor": 0.0,
                    },
                ],
                "complement_groups": [
                    {"items": ["CPU_H", "MB"], "bonus": 50},
                    {"items": ["GPU_H", "RAM"], "bonus": 50},
                    {"items": ["MB", "SSD"], "bonus": 50},
                ],
                "core_items": good_ids[:2],
                "secondary_items": good_ids[2:],
                "low_interest_items": [],
            }
        )
    return scenario_profile_spec_from_dict(
        {
            "schema_version": "pc_build_profile_spec_v1",
            "domain": "pc_build",
            "goods": goods,
            "bidder_profiles": bidders,
        }
    )


def test_committed_design_has_variable_category_cardinalities():
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for good in design["goods"]:
        counts[good["category"]] = counts.get(good["category"], 0) + 1

    assert len(design["goods"]) == 16
    assert len(design["bidder_archetypes"]) == 16
    assert sorted(counts.values(), reverse=True) == [3, 3, 3, 2, 2, 1, 1, 1]


def test_sample_report_detects_uninterested_good_and_missing_groups():
    spec = _synthetic_population()
    report = sample_structure_report(
        spec,
        ["CPU_H", "CPU_M", "CPU_L", "SSD"],
        ["bidder_0"],
    )

    assert report["passed"] is False
    assert any("interested bidders" in item for item in report["violations"])
    assert any("complement groups" in item for item in report["violations"])


def test_sample_report_enforces_good_category_and_bidder_stratum_diversity():
    spec = _synthetic_population()
    report = sample_structure_report(
        spec,
        ["CPU_H", "CPU_M", "CPU_L", "GPU_H"],
        ["bidder_0", "bidder_4", "bidder_1", "bidder_5"],
        constraints={
            "min_positive_bidders_per_good": 1,
            "min_positive_goods_per_bidder": 1,
            "min_substitute_groups_by_goods": {},
            "min_complement_groups_by_goods": {},
            "min_distinct_substitute_groups_by_goods": {},
            "min_distinct_complement_groups_by_goods": {},
            "min_bidders_with_substitute_groups_by_goods": {},
            "min_bidders_with_complement_groups_by_goods": {},
            "min_good_categories_by_goods": {"4": 3},
            "min_bidder_strata_by_bidders": {"4": 3},
        },
    )

    assert report["passed"] is False
    assert report["selected_good_categories"] == ["cpu", "gpu"]
    assert report["selected_bidder_strata"] == ["stratum_0", "stratum_1"]
    assert any("good categories" in item for item in report["violations"])
    assert any("bidder strata" in item for item in report["violations"])


def test_sample_report_enforces_exclusion_density():
    spec = _synthetic_population()
    report = sample_structure_report(
        spec,
        [good.id for good in spec.goods[:4]],
        [bidder.bidder_id for bidder in spec.bidder_profiles[:4]],
        constraints={
            "max_mean_interest_density_by_goods": {"4": 0.8},
            "min_fraction_bidders_with_exclusions": 0.5,
        },
    )

    assert report["mean_interest_density"] == 1.0
    assert report["bidders_with_exclusions"] == []
    assert any("interest density" in item for item in report["violations"])
    assert any("exclude at least one" in item for item in report["violations"])


def test_monetary_semantics_reject_fractional_penalty_and_outside_cap():
    profile = _synthetic_population().bidder_profiles[0].model_copy(
        update={
            "budget_cap": 2500.0,
            "saturation_start": 4,
            "saturation_penalty": 0.18,
        }
    )

    violations = bidder_monetary_semantic_violations(
        profile,
        num_goods=8,
    )

    assert any("budget_cap" in item for item in violations)
    assert any("absolute-dollar" in item for item in violations)


def test_population_report_detects_multiway_substitute_categories():
    spec = _synthetic_population()
    report = population_coverage_report(
        spec,
        constraints={
            "min_positive_bidders_per_good": 2,
            "min_core_bidders_per_good": 0,
            "min_strata_with_positive_value_per_good": 2,
            "min_profiles_with_substitute_groups": 4,
            "min_profiles_with_complement_groups": 4,
            "required_multiway_substitute_categories": ["cpu"],
        },
    )

    assert report["passed"] is True
    assert report["multiway_substitute_categories"] == ["cpu"]


def test_population_report_enforces_interest_density_limits():
    spec = _synthetic_population()
    report = population_coverage_report(
        spec,
        constraints={
            "max_positive_bidders_per_good": 7,
            "max_population_interest_density": 0.8,
            "max_positive_goods_per_bidder_by_stratum": {
                "stratum_0": 6,
                "stratum_1": 6,
                "stratum_2": 6,
                "stratum_3": 6,
            },
        },
    )

    assert report["passed"] is False
    assert report["population_interest_density"] == 1.0
    assert len(report["interest_density_violations"]) == 8
    assert any("maximum 7" in item for item in report["violations"])


def test_population_report_enforces_backup_factor_roles():
    spec = _synthetic_population()
    for group in spec.bidder_profiles[0].substitute_groups:
        group.acquisition_mode = "can_use_multiple"
        group.backup_factor = 0.6
    spec.bidder_profiles[1].substitute_groups[0].backup_factor = 0.1
    report = population_coverage_report(
        spec,
        constraints={
            "choose_one_backup_factor_max": 0.0,
            "can_use_multiple_backup_factor_min": 0.0,
            "reseller_substitute_backup_factor_min": 0.7,
            "high_backup_bidder_ids": ["bidder_0"],
        },
    )

    assert report["passed"] is False
    assert any("choose_one group" in item for item in report["violations"])
    assert any("high-backup bidder bidder_0" in item for item in report["violations"])


def test_population_report_rejects_nonpositive_substitute_members():
    spec = _synthetic_population()
    spec.bidder_profiles[0].base_values.pop("CPU_L")

    report = population_coverage_report(spec)

    assert report["passed"] is False
    assert report["nonpositive_substitute_items"]["bidder_0"] == ["CPU_L"]


def test_coverage_order_is_deterministic_nested_and_passes_cells():
    spec = _synthetic_population()
    constraints = {
        "min_positive_bidders_per_good": 2,
        "min_positive_goods_per_bidder": 1,
        "min_substitute_groups_by_goods": {"4": 1},
        "min_complement_groups_by_goods": {"4": 1},
        "min_distinct_substitute_groups_by_goods": {"4": 1},
        "min_distinct_complement_groups_by_goods": {"4": 1},
        "min_bidders_with_substitute_groups_by_goods": {"4": 2},
        "min_bidders_with_complement_groups_by_goods": {"4": 2},
    }
    first = coverage_aware_nested_order(
        spec,
        seed=7,
        sizes=[4, 5],
        fixed_size=4,
        constraints=constraints,
    )
    second = coverage_aware_nested_order(
        spec,
        seed=7,
        sizes=[4, 5],
        fixed_size=4,
        constraints=constraints,
    )

    assert first == second
    assert set(first.goods[:4]) < set(first.goods[:5])
    assert set(first.bidders[:4]) < set(first.bidders[:5])
    for goods, bidders in ((4, 4), (5, 4), (4, 5), (5, 5)):
        assert sample_structure_report(
            spec,
            first.goods[:goods],
            first.bidders[:bidders],
            constraints=constraints,
        )["passed"]


def test_validated_orders_are_frozen_as_complete_population_orders():
    spec = _synthetic_population()
    frozen = freeze_validated_nested_orders(
        spec,
        {
            "passed": True,
            "cases": [
                {
                    "seed": 3,
                    "selected_goods": ["CPU_M", "CPU_H", "GPU_H", "MB"],
                    "selected_bidders": [
                        "bidder_3",
                        "bidder_2",
                        "bidder_1",
                        "bidder_0",
                    ],
                    "order_attempts": 17,
                },
                {
                    "seed": 3,
                    "selected_goods": [
                        "CPU_M", "CPU_H", "GPU_H", "MB", "RAM",
                    ],
                    "selected_bidders": [
                        "bidder_3",
                        "bidder_2",
                        "bidder_1",
                        "bidder_0",
                        "bidder_4",
                    ],
                    "order_attempts": 17,
                },
            ],
        },
    )["3"]

    assert frozen["goods"][:5] == [
        "CPU_M", "CPU_H", "GPU_H", "MB", "RAM",
    ]
    assert frozen["bidders"][:5] == [
        "bidder_3", "bidder_2", "bidder_1", "bidder_0", "bidder_4",
    ]
    assert len(frozen["goods"]) == 8
    assert len(frozen["bidders"]) == 8
    assert frozen["attempts"] == 17


def test_economic_report_uses_size_aware_welfare_share_limit():
    spec = _synthetic_population()
    report = sample_economic_report(
        spec,
        [good.id for good in spec.goods[:4]],
        [bidder.bidder_id for bidder in spec.bidder_profiles[:4]],
        constraints={
            "min_full_information_winners": 1,
            "max_largest_winner_welfare_share_by_goods": {
                "4": 0.9,
                "6": 0.8,
            },
        },
    )

    assert report["max_largest_winner_welfare_share"] == 0.9
