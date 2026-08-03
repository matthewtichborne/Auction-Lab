"""Tests for the structured PC-build scenario factory."""

from __future__ import annotations

from itertools import combinations

import pytest

from auctionlab.instances.structured import (
    PC_GOOD_CATALOG,
    all_nonempty_bundles,
    enforce_monotonicity,
    make_pc_build_scenario,
    value_bundle,
    make_competitive_gamer_profile,
    make_budget_gamer_profile,
    make_pc_reseller_profile,
)
from auctionlab.solvers.wdp_ilp import solve_wdp_xor_ilp
import random


# ---------------------------------------------------------------------------
# 1. Basic shape tests
# ---------------------------------------------------------------------------

def test_pc_build_4x4_shape():
    s = make_pc_build_scenario(4, 4, seed=0)
    assert s.name == "pc_build_4x4_calibrated"
    assert len(s.instance.items) == 4
    assert len(s.instance.bidder_ids) == 4
    for bid in s.instance.bidder_ids:
        assert len(s.instance.valuations[bid]) == 2 ** 4 - 1


def test_pc_build_5x4_shape():
    s = make_pc_build_scenario(5, 4, seed=1)
    assert len(s.instance.items) == 5
    assert len(s.instance.bidder_ids) == 4
    for bid in s.instance.bidder_ids:
        assert len(s.instance.valuations[bid]) == 2 ** 5 - 1


def test_pc_build_8x6_shape():
    s = make_pc_build_scenario(8, 6, seed=2)
    assert len(s.instance.items) == 8
    assert len(s.instance.bidder_ids) == 6
    for bid in s.instance.bidder_ids:
        assert len(s.instance.valuations[bid]) == 2 ** 8 - 1


def test_pc_build_10x10_shape():
    s = make_pc_build_scenario(10, 10, seed=3)
    assert len(s.instance.items) == 10
    assert len(s.instance.bidder_ids) == 10
    for bid in s.instance.bidder_ids:
        assert len(s.instance.valuations[bid]) == 2 ** 10 - 1


def test_pc_build_name_override():
    s = make_pc_build_scenario(4, 3, seed=0, name="my_custom_name")
    assert s.name == "my_custom_name"


def test_pc_build_default_name():
    s = make_pc_build_scenario(6, 5, seed=0)
    assert s.name == "pc_build_6x5_calibrated"


def test_pc_build_uses_catalog_order():
    s = make_pc_build_scenario(6, 3, seed=0)
    assert list(s.instance.items) == PC_GOOD_CATALOG[:6]


def test_pc_build_seed_type():
    s = make_pc_build_scenario(4, 3, seed=0)
    assert s.seed_type == "structured"


def test_pc_build_no_candidate_bundles():
    s = make_pc_build_scenario(6, 4, seed=0)
    assert s.candidate_bundles_by_bidder is None


def test_pc_build_metadata_fields():
    s = make_pc_build_scenario(8, 6, seed=2)
    md = s.metadata
    assert md["num_goods"] == 8
    assert md["num_bidders"] == 6
    assert md["scenario_seed"] == 2
    assert md["full_valuation_table_size"] == 255
    assert md["domain"] == "pc_build"
    assert md["valuation_model"] == "structured_substitutes_complements"
    assert md["seed_style"] == "brief_qualitative"


# ---------------------------------------------------------------------------
# 2. Validation errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_goods", [0, 1, 3, 11, 100])
def test_invalid_num_goods_rejected(num_goods):
    with pytest.raises(ValueError, match="num_goods"):
        make_pc_build_scenario(num_goods, 4, seed=0)


@pytest.mark.parametrize("num_bidders", [0, 1, 2, 11, 100])
def test_invalid_num_bidders_rejected(num_bidders):
    with pytest.raises(ValueError, match="num_bidders"):
        make_pc_build_scenario(4, num_bidders, seed=0)


# ---------------------------------------------------------------------------
# 3. Monotonicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_size", [(4, 4), (6, 6), (8, 6)])
def test_monotonicity(scenario_size):
    num_goods, num_bidders = scenario_size
    s = make_pc_build_scenario(num_goods, num_bidders, seed=0)
    items = s.instance.items
    tol = 1e-6
    for bid in s.instance.bidder_ids:
        table = s.instance.valuations[bid]
        for size_a in range(1, len(items)):
            for bundle_a in (b for b in table if len(b) == size_a):
                for item in items:
                    if item not in bundle_a:
                        bundle_b = bundle_a | {item}
                        if bundle_b in table:
                            assert table[bundle_a] <= table[bundle_b] + tol, (
                                f"{bid}: v({sorted(bundle_a)})={table[bundle_a]:.2f} > "
                                f"v({sorted(bundle_b)})={table[bundle_b]:.2f}"
                            )


# ---------------------------------------------------------------------------
# 4. Substitute behaviour
# ---------------------------------------------------------------------------

def test_substitute_cpu_pair_in_6x6():
    """In 6x6, competitive_gamer should have substitute behaviour for CPUs."""
    s = make_pc_build_scenario(6, 3, seed=0)
    table = s.instance.valuations["competitive_gamer"]
    v_high = table[frozenset({"CPU_HIGH"})]
    v_budget = table[frozenset({"CPU_BUDGET"})]
    v_both = table[frozenset({"CPU_HIGH", "CPU_BUDGET"})]

    # Having both costs less than the sum (substitute discount)
    assert v_both < v_high + v_budget - 1.0, (
        f"v_both={v_both:.0f} should be < v_high+v_budget={v_high+v_budget:.0f}"
    )
    # But still worth at least the better one
    assert v_both >= max(v_high, v_budget) - 1e-6


def test_substitute_gpu_pair_in_6x6():
    s = make_pc_build_scenario(6, 3, seed=0)
    table = s.instance.valuations["competitive_gamer"]
    v_high = table[frozenset({"GPU_HIGH"})]
    v_mid = table[frozenset({"GPU_MID"})]
    v_both = table[frozenset({"GPU_HIGH", "GPU_MID"})]

    assert v_both < v_high + v_mid - 1.0
    assert v_both >= max(v_high, v_mid) - 1e-6


def test_reseller_has_high_backup_factor():
    """Reseller should retain most value from duplicate items."""
    s = make_pc_build_scenario(6, 7, seed=0)
    table = s.instance.valuations["pc_reseller"]
    v_high = table[frozenset({"CPU_HIGH"})]
    v_budget = table[frozenset({"CPU_BUDGET"})]
    v_both = table[frozenset({"CPU_HIGH", "CPU_BUDGET"})]

    # Reseller has backup_factor=0.88, so v_both ≈ max + 0.88 * other
    # i.e., much closer to the sum than for other bidders
    additive_sum = v_high + v_budget
    # v_both should be at least 80% of the additive sum
    assert v_both >= 0.80 * additive_sum


# ---------------------------------------------------------------------------
# 5. Complement behaviour
# ---------------------------------------------------------------------------

def test_complement_in_4x4_competitive_gamer():
    """Full gaming core in 4x4 should exceed sum of singletons."""
    s = make_pc_build_scenario(4, 3, seed=0)
    table = s.instance.valuations["competitive_gamer"]
    core = frozenset({"CPU_HIGH", "GPU_HIGH", "MOTHERBOARD", "RAM_32GB"})
    sum_singles = sum(table[frozenset({item})] for item in core)
    assert table[core] > sum_singles, (
        f"core bundle value {table[core]:.0f} should exceed sum of singles {sum_singles:.0f}"
    )


def test_complement_in_6x6_video_editor():
    """Video editor CPU+RAM+MOTHERBOARD complement should hold in 6x6."""
    s = make_pc_build_scenario(6, 3, seed=0)
    table = s.instance.valuations["video_editor"]
    items = set(s.instance.items)
    # RAM item depends on availability: 4-item catalog only has RAM_32GB
    # 6-item catalog has RAM_32GB (item 4)
    ram = "RAM_32GB"
    triple = frozenset({"CPU_HIGH", ram, "MOTHERBOARD"})
    if triple.issubset(items):
        sum_singles = sum(table[frozenset({item})] for item in triple)
        assert table[triple] > sum_singles, (
            f"video_editor triple value {table[triple]:.0f} should exceed "
            f"sum of singles {sum_singles:.0f}"
        )


# ---------------------------------------------------------------------------
# 6. Person seed content
# ---------------------------------------------------------------------------

FORBIDDEN_SEED_STRINGS = [
    "Single Item Bundles",
    "Complementary Bundles",
    "Total Value:",
    "1. Single",
    "2. Compl",
]


@pytest.mark.parametrize("scenario_size", [(4, 4), (6, 6), (8, 6)])
def test_seeds_have_no_forbidden_strings(scenario_size):
    num_goods, num_bidders = scenario_size
    s = make_pc_build_scenario(num_goods, num_bidders, seed=0)
    for bid, seed in s.person_seeds.items():
        for term in FORBIDDEN_SEED_STRINGS:
            assert term not in seed, f"{bid} seed contains forbidden: {term!r}"


def test_seeds_contain_dollar_amounts():
    s = make_pc_build_scenario(6, 6, seed=0)
    for bid, seed in s.person_seeds.items():
        assert "$" in seed, f"{bid}: seed should contain dollar amounts"


def test_seeds_contain_budget_range_language():
    s = make_pc_build_scenario(4, 4, seed=0)
    for bid, seed in s.person_seeds.items():
        assert "maximum total willingness to pay" in seed.lower(), (
            f"{bid}: seed should mention one total willingness-to-pay ceiling"
        )


def test_seeds_do_not_enumerate_all_bundles():
    """Seeds should not list out every bundle line-by-line."""
    s = make_pc_build_scenario(6, 6, seed=0)
    items = s.instance.items
    # If a seed enumerated all bundles, it would have ~63 lines with item names.
    # We check that no seed has more than 30 lines.
    for bid, seed in s.person_seeds.items():
        lines = [l for l in seed.splitlines() if l.strip()]
        assert len(lines) <= 30, (
            f"{bid}: seed has {len(lines)} lines — likely enumerating bundles"
        )


def test_reseller_seed_does_not_say_only_one_can_be_installed():
    """Reseller substitute groups have high backup factor — seed should not
    claim 'only one can be installed at a time', which contradicts resale logic."""
    s = make_pc_build_scenario(6, 7, seed=0)
    reseller_seed = s.person_seeds["pc_reseller"]
    assert "only one can be installed at a time" not in reseller_seed
    assert "only one GPU can be installed" not in reseller_seed
    assert "only one CPU can be installed" not in reseller_seed


def test_seeds_are_distinct_across_bidders():
    s = make_pc_build_scenario(6, 6, seed=0)
    seeds = list(s.person_seeds.values())
    assert len(seeds) == len(set(seeds)), "all bidder seeds should be distinct"


def test_seed_reproducibility():
    s1 = make_pc_build_scenario(6, 4, seed=42)
    s2 = make_pc_build_scenario(6, 4, seed=42)
    assert s1.person_seeds == s2.person_seeds
    assert s1.instance.valuations == s2.instance.valuations


def test_different_seeds_give_different_valuations():
    s1 = make_pc_build_scenario(6, 4, seed=0)
    s2 = make_pc_build_scenario(6, 4, seed=99)
    # At least one bidder should have different valuations
    diffs = 0
    for bid in s1.instance.bidder_ids:
        if s1.instance.valuations[bid] != s2.instance.valuations[bid]:
            diffs += 1
    assert diffs > 0


# ---------------------------------------------------------------------------
# 7. Full-info WDP smoke
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(4, 4), (8, 6)])
def test_wdp_smoke(size):
    num_goods, num_bidders = size
    s = make_pc_build_scenario(num_goods, num_bidders, seed=0)
    instance = s.instance
    result = solve_wdp_xor_ilp(instance.items, instance.to_xor_bids())
    assert result.welfare > 0


# ---------------------------------------------------------------------------
# 8. enforce_monotonicity helper
# ---------------------------------------------------------------------------

def test_enforce_monotonicity_repairs_violation():
    items = ["A", "B", "C"]
    table = {
        frozenset({"A"}): 10.0,
        frozenset({"B"}): 5.0,
        frozenset({"C"}): 3.0,
        frozenset({"A", "B"}): 8.0,   # violates: less than v({A})
        frozenset({"A", "C"}): 12.0,
        frozenset({"B", "C"}): 6.0,
        frozenset({"A", "B", "C"}): 9.0,  # violates: less than v({A,C})
    }
    repaired = enforce_monotonicity(items, table)
    assert repaired[frozenset({"A", "B"})] >= repaired[frozenset({"A"})]
    assert repaired[frozenset({"A", "B"})] >= repaired[frozenset({"B"})]
    assert repaired[frozenset({"A", "B", "C"})] >= repaired[frozenset({"A", "C"})]


def test_enforce_monotonicity_preserves_valid_values():
    items = ["A", "B"]
    table = {frozenset({"A"}): 10.0, frozenset({"B"}): 5.0, frozenset({"A", "B"}): 20.0}
    repaired = enforce_monotonicity(items, table)
    assert repaired[frozenset({"A", "B"})] == 20.0


# ---------------------------------------------------------------------------
# 9. CLI scenario selection helper
# ---------------------------------------------------------------------------

def test_select_scenarios_pc_build_dynamic():
    from examples.run_live_llm_curated_batch import select_scenarios
    selected = select_scenarios(
        ["pc_build"], "all",
        num_goods=5, num_bidders=3, scenario_seed=7,
    )
    assert len(selected) == 1
    assert selected[0].name == "pc_build_5x3_calibrated"
    assert len(selected[0].instance.items) == 5
    assert len(selected[0].instance.bidder_ids) == 3


def test_select_scenarios_rejects_pc_build_with_invalid_size():
    from examples.run_live_llm_curated_batch import select_scenarios
    with pytest.raises((ValueError, SystemExit)):
        select_scenarios(["pc_build"], "all", num_goods=3, num_bidders=3)


# ---------------------------------------------------------------------------
# 10. Per-profile metadata
# ---------------------------------------------------------------------------

def test_per_profile_metadata_present_8x8():
    s = make_pc_build_scenario(8, 8, seed=0)
    profiles = s.metadata.get("profiles", {})
    assert set(profiles.keys()) == set(s.instance.bidder_ids)
    for bidder_id, pmd in profiles.items():
        assert "budget_cap" in pmd
        assert "saturation_start" in pmd
        assert "saturation_penalty" in pmd
        assert "core_items" in pmd
        assert "secondary_items" in pmd
        assert "low_interest_items" in pmd
        assert "substitute_groups" in pmd
        assert "complement_groups" in pmd
        for sg in pmd["substitute_groups"]:
            assert "items" in sg and "backup_factor" in sg
        for cg in pmd["complement_groups"]:
            assert "items" in cg and "bonus" in cg


def test_per_profile_metadata_budget_cap_uncapped_bidders():
    s = make_pc_build_scenario(8, 8, seed=0)
    profiles = s.metadata["profiles"]
    for bidder_id in ["budget_gamer", "office_upgrader", "student_builder"]:
        assert profiles[bidder_id]["budget_cap"] is None, (
            f"{bidder_id} should have no hard cap"
        )


# ---------------------------------------------------------------------------
# 11. Cap plateau / top-value concentration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bidder_id", ["budget_gamer", "office_upgrader", "student_builder"])
def test_top_value_concentration_below_25pct_in_8x8(bidder_id):
    """For formerly-capped bidders, the most common valuation level in 8x8
    must cover fewer than 25% of all bundles."""
    s = make_pc_build_scenario(8, 8, seed=0)
    table = s.instance.valuations[bidder_id]
    values = list(table.values())
    max_val = max(values)
    top_count = sum(1 for v in values if abs(v - max_val) < 0.01)
    total = len(values)
    assert top_count / total < 0.25, (
        f"{bidder_id}: {top_count}/{total} ({top_count/total:.1%}) bundles at top "
        f"value {max_val:.0f} — plateau too large"
    )


# ---------------------------------------------------------------------------
# 12. Backup factor constraints
# ---------------------------------------------------------------------------

def test_reseller_substitute_backup_factors_high():
    """All pc_reseller substitute groups should have backup_factor >= 0.80."""
    s = make_pc_build_scenario(8, 8, seed=0)
    profiles = s.metadata["profiles"]
    for sg in profiles["pc_reseller"]["substitute_groups"]:
        assert sg["backup_factor"] >= 0.80, (
            f"reseller sub group {sg['items']} has backup_factor={sg['backup_factor']:.2f} < 0.80"
        )


def test_ordinary_cpu_gpu_backup_factors_low():
    """Non-reseller bidders with CPU or GPU substitute groups should have
    backup_factor < 0.15."""
    s = make_pc_build_scenario(8, 8, seed=0)
    profiles = s.metadata["profiles"]
    cpu_gpu_items = {"CPU_HIGH", "CPU_BUDGET", "GPU_HIGH", "GPU_MID"}
    for bidder_id, pmd in profiles.items():
        if bidder_id == "pc_reseller":
            continue
        for sg in pmd["substitute_groups"]:
            if set(sg["items"]) & cpu_gpu_items:
                assert sg["backup_factor"] < 0.15, (
                    f"{bidder_id} CPU/GPU sub group {sg['items']} has "
                    f"backup_factor={sg['backup_factor']:.2f} >= 0.15"
                )


# ---------------------------------------------------------------------------
# 13. Full-info WDP smoke — all four default sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_goods,num_bidders", [(4, 4), (6, 6), (8, 8), (10, 10)])
def test_wdp_smoke_all_sizes(num_goods, num_bidders):
    s = make_pc_build_scenario(num_goods, num_bidders, seed=0)
    instance = s.instance
    result = solve_wdp_xor_ilp(instance.items, instance.to_xor_bids())
    assert result.welfare > 0
