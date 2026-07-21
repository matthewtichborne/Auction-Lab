"""Tests for the spec-based PC-build scenario factory.

See scenarios/pc_build_v1/README.md ("A note on exact numeric equivalence
with the hard-coded factory") for why these tests check *structural*
equivalence with the hard-coded factory rather than exact value equality,
and check exact determinism within the spec-based path alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auctionlab.instances.scenario_spec import (
    BidderProfileSpec,
    GoodSpec,
    load_scenario_profile_spec,
    scenario_profile_spec_from_dict,
    write_scenario_profile_spec,
)
from auctionlab.instances.structured import make_pc_build_scenario
from auctionlab.instances.structured_spec import (
    categorize_bidder,
    categorize_good,
    make_pc_build_scenario_from_spec,
)

V0_SPEC_PATH = Path("scenarios/pc_build_v1/pc_build_profiles_v0_manual.json")


@pytest.fixture(scope="module")
def v0_spec_path(tmp_path_factory) -> Path:
    """Use the committed v0 spec if present; otherwise export a fresh one."""
    if V0_SPEC_PATH.exists():
        return V0_SPEC_PATH
    from scripts.export_current_pc_build_profiles import build_manual_spec

    spec = build_manual_spec(seed=0)
    path = tmp_path_factory.mktemp("spec") / "pc_build_profiles_v0_manual.json"
    write_scenario_profile_spec(spec, path)
    return path


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------

def test_from_spec_basic_shape(v0_spec_path):
    s = make_pc_build_scenario_from_spec(v0_spec_path, num_goods=6, num_bidders=6)
    assert len(s.instance.items) == 6
    assert len(s.instance.bidder_ids) == 6
    for bid in s.instance.bidder_ids:
        assert len(s.instance.valuations[bid]) == 2 ** 6 - 1
    assert s.seed_type == "structured"
    assert s.candidate_bundles_by_bidder is None


def test_from_spec_name_default(v0_spec_path):
    s = make_pc_build_scenario_from_spec(v0_spec_path, num_goods=4, num_bidders=3)
    assert s.name == "pc_build_4x3_from_spec"


def test_from_spec_name_override(v0_spec_path):
    s = make_pc_build_scenario_from_spec(
        v0_spec_path, num_goods=4, num_bidders=3, name="custom"
    )
    assert s.name == "custom"


def test_from_spec_rejects_out_of_range_sizes(v0_spec_path):
    with pytest.raises(ValueError):
        make_pc_build_scenario_from_spec(v0_spec_path, num_goods=100, num_bidders=3)
    with pytest.raises(ValueError):
        make_pc_build_scenario_from_spec(v0_spec_path, num_goods=4, num_bidders=100)


def test_from_spec_accepts_preloaded_spec(v0_spec_path):
    spec = load_scenario_profile_spec(v0_spec_path)
    s = make_pc_build_scenario_from_spec(spec, num_goods=4, num_bidders=4)
    assert len(s.instance.items) == 4


# ---------------------------------------------------------------------------
# Structural equivalence with the hard-coded factory
# ---------------------------------------------------------------------------

def test_structural_equivalence_with_hardcoded_factory(v0_spec_path):
    hardcoded = make_pc_build_scenario(6, 6, seed=0)
    from_spec = make_pc_build_scenario_from_spec(
        v0_spec_path, num_goods=6, num_bidders=6, selection_policy="prefix"
    )

    assert list(from_spec.instance.items) == list(hardcoded.instance.items)
    assert list(from_spec.instance.bidder_ids) == list(hardcoded.instance.bidder_ids)
    assert set(from_spec.person_seeds) == set(hardcoded.person_seeds)
    for bid in hardcoded.instance.bidder_ids:
        assert len(from_spec.instance.valuations[bid]) == len(
            hardcoded.instance.valuations[bid]
        )


def test_monotonicity_holds_for_spec_based_valuations(v0_spec_path):
    s = make_pc_build_scenario_from_spec(v0_spec_path, num_goods=6, num_bidders=6)
    items = s.instance.items
    tol = 1e-6
    for bid in s.instance.bidder_ids:
        table = s.instance.valuations[bid]
        for bundle_a, value_a in table.items():
            for item in items:
                if item not in bundle_a:
                    bundle_b = bundle_a | {item}
                    if bundle_b in table:
                        assert value_a <= table[bundle_b] + tol


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_from_spec_is_deterministic(v0_spec_path):
    s1 = make_pc_build_scenario_from_spec(v0_spec_path, num_goods=6, num_bidders=4, seed=0)
    s2 = make_pc_build_scenario_from_spec(v0_spec_path, num_goods=6, num_bidders=4, seed=0)
    assert s1.person_seeds == s2.person_seeds
    assert s1.instance.valuations == s2.instance.valuations
    assert list(s1.instance.items) == list(s2.instance.items)
    assert list(s1.instance.bidder_ids) == list(s2.instance.bidder_ids)


def test_from_spec_is_deterministic_across_reloaded_spec(v0_spec_path, tmp_path):
    """Loading the same spec twice (fresh objects each time) still gives
    byte-identical valuations and person seeds."""
    spec_a = load_scenario_profile_spec(v0_spec_path)
    spec_b = load_scenario_profile_spec(v0_spec_path)
    assert spec_a == spec_b

    s1 = make_pc_build_scenario_from_spec(spec_a, num_goods=8, num_bidders=8, seed=7)
    s2 = make_pc_build_scenario_from_spec(spec_b, num_goods=8, num_bidders=8, seed=7)
    assert s1.instance.valuations == s2.instance.valuations
    assert s1.person_seeds == s2.person_seeds


def test_from_spec_seed_does_not_affect_prefix_policy(v0_spec_path):
    """Under selection_policy='prefix', seed only matters for seeded_sample;
    varying it should not change the resulting scenario."""
    s1 = make_pc_build_scenario_from_spec(v0_spec_path, num_goods=6, num_bidders=6, seed=0)
    s2 = make_pc_build_scenario_from_spec(v0_spec_path, num_goods=6, num_bidders=6, seed=123)
    assert list(s1.instance.items) == list(s2.instance.items)
    assert list(s1.instance.bidder_ids) == list(s2.instance.bidder_ids)
    assert s1.instance.valuations == s2.instance.valuations
    assert s1.person_seeds == s2.person_seeds


# ---------------------------------------------------------------------------
# Selection policies
# ---------------------------------------------------------------------------

def test_seeded_sample_selection_is_deterministic(v0_spec_path):
    s1 = make_pc_build_scenario_from_spec(
        v0_spec_path, num_goods=6, num_bidders=6, seed=5, selection_policy="seeded_sample"
    )
    s2 = make_pc_build_scenario_from_spec(
        v0_spec_path, num_goods=6, num_bidders=6, seed=5, selection_policy="seeded_sample"
    )
    assert list(s1.instance.items) == list(s2.instance.items)
    assert list(s1.instance.bidder_ids) == list(s2.instance.bidder_ids)
    assert s1.instance.valuations == s2.instance.valuations


def test_seeded_sample_differs_from_prefix_for_some_seed(v0_spec_path):
    prefix = make_pc_build_scenario_from_spec(
        v0_spec_path, num_goods=5, num_bidders=5, selection_policy="prefix"
    )
    found_difference = False
    for seed in range(10):
        sampled = make_pc_build_scenario_from_spec(
            v0_spec_path,
            num_goods=5,
            num_bidders=5,
            seed=seed,
            selection_policy="seeded_sample",
        )
        if list(sampled.instance.items) != list(prefix.instance.items):
            found_difference = True
            break
    assert found_difference


def test_unknown_selection_policy_rejected(v0_spec_path):
    with pytest.raises(ValueError):
        make_pc_build_scenario_from_spec(
            v0_spec_path, num_goods=4, num_bidders=4, selection_policy="bogus"
        )


# ---------------------------------------------------------------------------
# Stratified selection
# ---------------------------------------------------------------------------

def _synthetic_platform_spec_dict() -> dict:
    """A spec deliberately shaped so a plain prefix misses whole categories.

    Declaration order is CPU, CPU, GPU, GPU, motherboard, RAM, storage,
    power/cooling -- a prefix of 4 goods captures only CPUs and GPUs, never
    a complete platform.
    """
    return {
        "schema_version": "pc_build_profile_spec_v1",
        "domain": "pc_build",
        "description": "Synthetic spec for stratified-selection tests.",
        "goods": [
            {"id": "CPU_A", "description": "A processor.", "category": "CPU"},
            {"id": "CPU_B", "description": "Another processor.", "category": "CPU"},
            {"id": "GPU_A", "description": "A graphics card with 24GB VRAM.", "category": "GPU"},
            {"id": "GPU_B", "description": "Another graphics card.", "category": "GPU"},
            {"id": "MB_A", "description": "A motherboard.", "category": "Motherboard"},
            {"id": "RAM_A", "description": "A memory kit.", "category": "RAM"},
            {"id": "SSD_A", "description": "A storage drive.", "category": "Storage"},
            {"id": "PSU_A", "description": "A power supply.", "category": "PSU"},
        ],
        "bidder_profiles": [
            {
                "bidder_id": "enthusiast_gamer",
                "role": "A competitive gamer who wants top performance.",
                "budget_range": [500.0, 2000.0],
                "base_values": {"CPU_A": 300.0, "GPU_A": 400.0},
                "substitute_groups": [],
                "complement_groups": [],
                "budget_cap": None,
                "saturation_start": None,
                "saturation_penalty": 0.0,
                "notes": "",
                "core_items": ["CPU_A", "GPU_A"],
                "secondary_items": [],
                "low_interest_items": [],
            },
            {
                "bidder_id": "office_user",
                "role": "An office worker upgrading a productivity machine.",
                "budget_range": [200.0, 600.0],
                "base_values": {"SSD_A": 150.0},
                "substitute_groups": [],
                "complement_groups": [],
                "budget_cap": None,
                "saturation_start": None,
                "saturation_penalty": 0.0,
                "notes": "",
                "core_items": ["SSD_A"],
                "secondary_items": [],
                "low_interest_items": [],
            },
            {
                "bidder_id": "ai_professional",
                "role": "A professional AI/ML practitioner building a workstation.",
                "budget_range": [1000.0, 3000.0],
                "base_values": {"GPU_B": 500.0, "RAM_A": 200.0},
                "substitute_groups": [],
                "complement_groups": [],
                "budget_cap": None,
                "saturation_start": None,
                "saturation_penalty": 0.0,
                "notes": "",
                "core_items": ["GPU_B", "RAM_A"],
                "secondary_items": [],
                "low_interest_items": [],
            },
            {
                "bidder_id": "component_reseller",
                "role": "A procurement buyer who resells components.",
                "budget_range": [1000.0, 4000.0],
                "base_values": {"CPU_B": 250.0, "PSU_A": 100.0},
                "substitute_groups": [],
                "complement_groups": [],
                "budget_cap": None,
                "saturation_start": None,
                "saturation_penalty": 0.0,
                "notes": "",
                "core_items": ["CPU_B", "PSU_A"],
                "secondary_items": [],
                "low_interest_items": [],
            },
        ],
        "generation": {"source": "test fixture"},
        "notes": None,
    }


@pytest.fixture
def synthetic_platform_spec_path(tmp_path) -> Path:
    write_scenario_profile_spec(
        scenario_profile_spec_from_dict(_synthetic_platform_spec_dict()), tmp_path / "synthetic.json"
    )
    return tmp_path / "synthetic.json"


def test_prefix_misses_categories_that_stratified_covers(synthetic_platform_spec_path):
    """Demonstrates the exact failure mode stratified selection exists to fix:
    a naive prefix of the first 4 goods captures only CPU/GPU, no
    motherboard/RAM/storage/power."""
    prefix = make_pc_build_scenario_from_spec(
        synthetic_platform_spec_path, num_goods=4, num_bidders=4, selection_policy="prefix"
    )
    assert set(prefix.instance.items) == {"CPU_A", "CPU_B", "GPU_A", "GPU_B"}

    stratified = make_pc_build_scenario_from_spec(
        synthetic_platform_spec_path,
        num_goods=4,
        num_bidders=4,
        seed=0,
        selection_policy="stratified",
    )
    categories = {
        categorize_good(g)[0]
        for g in load_scenario_profile_spec(synthetic_platform_spec_path).goods
        if g.id in set(stratified.instance.items)
    }
    assert categories == {"cpu", "gpu", "motherboard", "ram"}


def test_stratified_goods_selection_is_deterministic(synthetic_platform_spec_path):
    s1 = make_pc_build_scenario_from_spec(
        synthetic_platform_spec_path, num_goods=6, num_bidders=4, seed=3, selection_policy="stratified"
    )
    s2 = make_pc_build_scenario_from_spec(
        synthetic_platform_spec_path, num_goods=6, num_bidders=4, seed=3, selection_policy="stratified"
    )
    assert list(s1.instance.items) == list(s2.instance.items)
    assert list(s1.instance.bidder_ids) == list(s2.instance.bidder_ids)
    assert s1.instance.valuations == s2.instance.valuations


def test_stratified_bidder_selection_covers_archetype_buckets(synthetic_platform_spec_path):
    s = make_pc_build_scenario_from_spec(
        synthetic_platform_spec_path, num_goods=8, num_bidders=4, seed=0, selection_policy="stratified"
    )
    assert set(s.instance.bidder_ids) == {
        "enthusiast_gamer", "office_user", "ai_professional", "component_reseller",
    }


def test_stratified_selection_all_goods_covers_every_category(synthetic_platform_spec_path):
    """With enough slots, stratified selection should include every populated category."""
    s = make_pc_build_scenario_from_spec(
        synthetic_platform_spec_path, num_goods=6, num_bidders=4, seed=0, selection_policy="stratified"
    )
    spec = load_scenario_profile_spec(synthetic_platform_spec_path)
    goods_by_id = {g.id: g for g in spec.goods}
    covered = {
        cat
        for item_id in s.instance.items
        for cat in categorize_good(goods_by_id[item_id])
    }
    assert covered == {"cpu", "gpu", "motherboard", "ram", "storage", "power_cooling"}


def test_categorize_good_does_not_match_vram_as_ram():
    """Regression test: a GPU described with '24GB VRAM' must not be
    miscategorized as RAM via a naive substring match on 'ram'."""
    gpu = GoodSpec(id="GPU_AI", description="Professional GPU with 24GB VRAM for AI workloads.", category="GPU")
    assert categorize_good(gpu) == ["gpu"]


def test_categorize_good_matches_underscore_ids():
    mb = GoodSpec(id="MB_PRO", description="A professional motherboard.")
    assert "motherboard" in categorize_good(mb)


def test_categorize_bidder_matches_role_keywords():
    reseller = BidderProfileSpec(
        bidder_id="reseller_pro",
        role="A procurement buyer who resells components.",
        budget_range=(100.0, 200.0),
        base_values={"X": 1.0},
    )
    assert "reseller_procurement" in categorize_bidder(reseller)


@pytest.mark.parametrize("num_goods,num_bidders", [(4, 4), (6, 4)])
def test_stratified_selection_within_bounds(synthetic_platform_spec_path, num_goods, num_bidders):
    s = make_pc_build_scenario_from_spec(
        synthetic_platform_spec_path,
        num_goods=num_goods,
        num_bidders=num_bidders,
        seed=1,
        selection_policy="stratified",
    )
    assert len(s.instance.items) == num_goods
    assert len(s.instance.bidder_ids) == num_bidders
    assert len(set(s.instance.items)) == num_goods
    assert len(set(s.instance.bidder_ids)) == num_bidders


def test_stratified_fixes_complement_availability_on_real_generated_spec():
    """Integration-level regression test for the reported bug: under prefix
    selection, a generated spec's first-N goods can miss a complete
    complement platform even though the full spec has one."""
    v2_path = Path("scenarios/pc_build_v1/pc_build_profiles_v2_gemini_trial.json")
    if not v2_path.exists():
        pytest.skip("v2 gemini trial spec not present in this checkout")

    from auctionlab.instances.structured import all_nonempty_bundles, value_bundle
    from scripts.validate_scenario_spec import compute_size_diagnostics

    spec = load_scenario_profile_spec(v2_path)
    prefix_diag = compute_size_diagnostics(spec, 6, 6, selection_policy="prefix")
    stratified_diag = compute_size_diagnostics(spec, 6, 6, seed=0, selection_policy="stratified")

    # Not a strict inequality requirement for every possible spec, but true
    # for the specific v2 trial spec this was reported against.
    assert stratified_diag["complement_groups_available"] >= prefix_diag["complement_groups_available"]


def test_no_llm_import_in_structured_spec_module():
    import auctionlab.instances.structured_spec as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "auctionlab.llm" not in source
