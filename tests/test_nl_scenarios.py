from __future__ import annotations

import pytest

from examples.run_live_llm_curated_batch import select_scenarios
from auctionlab.instances.nl_scenarios import (
    curated_natural_language_scenarios,
    legacy_natural_language_scenarios,
)
from auctionlab.solvers.wdp_ilp import solve_wdp_xor_ilp


# ---------------------------------------------------------------------------
# curated_natural_language_scenarios — PC-build structured scenarios
# ---------------------------------------------------------------------------

def test_curated_scenarios_are_structurally_consistent():
    scenarios = curated_natural_language_scenarios()
    assert len(scenarios) == 4
    names = {s.name for s in scenarios}
    assert names == {
        "pc_build_4x4_calibrated",
        "pc_build_6x6_calibrated",
        "pc_build_8x8_calibrated",
        "pc_build_10x10_calibrated",
    }
    for scenario in scenarios:
        instance = scenario.instance
        assert scenario.seed_type == "structured"
        assert set(scenario.person_seeds) == set(instance.bidder_ids)
        assert set(instance.items).issubset(scenario.item_descriptions)
        assert all(instance.valuations[bid] for bid in instance.bidder_ids)
        assert scenario.candidate_bundles_by_bidder is None


def test_curated_scenarios_have_full_valuation_tables():
    for scenario in curated_natural_language_scenarios():
        n = len(scenario.instance.items)
        expected = 2 ** n - 1
        for bidder_id in scenario.instance.bidder_ids:
            assert len(scenario.instance.valuations[bidder_id]) == expected, (
                f"{scenario.name} / {bidder_id}: expected {expected} bundles"
            )


def test_curated_scenario_seeds_have_no_forbidden_strings():
    forbidden = ["Single Item Bundles", "Complementary Bundles", "Total Value:"]
    for scenario in curated_natural_language_scenarios():
        for bidder_id, seed in scenario.person_seeds.items():
            for term in forbidden:
                assert term not in seed, (
                    f"{scenario.name} / {bidder_id} seed contains forbidden string: {term!r}"
                )


def test_curated_scenario_seeds_contain_budget_language():
    for scenario in curated_natural_language_scenarios():
        for bidder_id, seed in scenario.person_seeds.items():
            assert "$" in seed, (
                f"{scenario.name} / {bidder_id}: seed should mention dollar amounts"
            )


def test_curated_scenario_metadata_fields():
    scenarios = curated_natural_language_scenarios()
    for scenario in scenarios:
        md = scenario.metadata
        assert "num_goods" in md
        assert "num_bidders" in md
        assert "scenario_seed" in md
        assert "full_valuation_table_size" in md
        assert md["full_valuation_table_size"] == 2 ** md["num_goods"] - 1


def test_curated_pc_build_4x4_wdp_smoke():
    scenario = next(
        s for s in curated_natural_language_scenarios()
        if s.name == "pc_build_4x4_calibrated"
    )
    instance = scenario.instance
    result = solve_wdp_xor_ilp(instance.items, instance.to_xor_bids())
    assert result.welfare > 0


def test_curated_pc_build_8x8_wdp_smoke():
    scenario = next(
        s for s in curated_natural_language_scenarios()
        if s.name == "pc_build_8x8_calibrated"
    )
    instance = scenario.instance
    result = solve_wdp_xor_ilp(instance.items, instance.to_xor_bids())
    assert result.welfare > 0


# ---------------------------------------------------------------------------
# legacy_natural_language_scenarios — old explicit/implicit scenarios
# ---------------------------------------------------------------------------

def test_legacy_scenarios_exist_and_have_expected_names():
    scenarios = legacy_natural_language_scenarios()
    names = {s.name for s in scenarios}
    assert "home_office_5x5" in names
    assert "home_studio_6x6" in names


def test_legacy_home_studio_benchmark_welfare():
    scenario = next(
        s for s in legacy_natural_language_scenarios()
        if s.name == "home_studio_6x6"
    )
    instance = scenario.instance
    result = solve_wdp_xor_ilp(instance.items, instance.to_xor_bids())
    assert result.welfare == 2200.0


def test_legacy_home_office_benchmark_welfare():
    scenario = next(
        s for s in legacy_natural_language_scenarios()
        if s.name == "home_office_5x5"
    )
    instance = scenario.instance
    result = solve_wdp_xor_ilp(instance.items, instance.to_xor_bids())
    assert result.welfare == 1700.0


# ---------------------------------------------------------------------------
# select_scenarios
# ---------------------------------------------------------------------------

def test_select_scenarios_all_returns_curated():
    selected = select_scenarios(None, "all")
    curated = curated_natural_language_scenarios()
    assert len(selected) == len(curated)
    assert {s.name for s in selected} == {s.name for s in curated}


def test_select_scenarios_by_name_pc_build_4x4():
    selected = select_scenarios(["pc_build_4x4_calibrated"], "all")
    assert len(selected) == 1
    assert selected[0].name == "pc_build_4x4_calibrated"


def test_select_scenarios_seed_type_structured():
    selected = select_scenarios(None, "structured")
    assert selected
    assert all(s.seed_type == "structured" for s in selected)


def test_select_scenarios_dynamic_pc_build():
    selected = select_scenarios(
        ["pc_build"], "all",
        num_goods=6, num_bidders=4, scenario_seed=1,
    )
    assert len(selected) == 1
    scenario = selected[0]
    assert scenario.name == "pc_build_6x4_calibrated"
    assert len(scenario.instance.items) == 6
    assert len(scenario.instance.bidder_ids) == 4


def test_select_scenarios_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown scenario names"):
        select_scenarios(["nonexistent_scenario"], "all")


def test_select_scenarios_rejects_empty_filtered_selection():
    with pytest.raises(ValueError, match="No scenarios matched"):
        select_scenarios(None, "explicit")
