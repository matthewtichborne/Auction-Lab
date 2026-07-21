"""Tests for scripts/calibrate_generated_pc_build_spec.py.

Covers only the deterministic calibration transform and grid expansion on a
tiny synthetic spec -- no LLM/API call, no WDP solve, no dependency on the
real generated spec file.
"""

from __future__ import annotations

import pytest

from auctionlab.instances.scenario_spec import (
    BidderProfileSpec,
    ComplementGroupSpec,
    GoodSpec,
    ScenarioProfileSpec,
)
from scripts.calibrate_generated_pc_build_spec import (
    DEFAULT_GRID,
    apply_calibration,
    iter_default_grid,
)

RESELLER_ID = "reseller_pro"


def _tiny_spec() -> ScenarioProfileSpec:
    """A 2-good, 2-bidder spec: one ordinary bidder, one reseller."""
    goods = [GoodSpec(id="A", description="a"), GoodSpec(id="B", description="b")]
    bidder_profiles = [
        BidderProfileSpec(
            bidder_id="alice",
            role="tester",
            budget_range=(0.0, 1000.0),
            base_values={"A": 100.0, "B": 50.0},
            complement_groups=[ComplementGroupSpec(items=["A", "B"], bonus=20.0)],
            saturation_start=1,
            saturation_penalty=0.1,
        ),
        BidderProfileSpec(
            bidder_id=RESELLER_ID,
            role="reseller",
            budget_range=(0.0, 1000.0),
            base_values={"A": 40.0, "B": 40.0},
            complement_groups=[],
            saturation_start=None,
            saturation_penalty=0.0,
        ),
    ]
    return ScenarioProfileSpec(
        schema_version="pc_build_profile_spec_v1",
        domain="pc_build",
        goods=goods,
        bidder_profiles=bidder_profiles,
    )


def _identity_kwargs(**overrides) -> dict:
    kwargs = {
        "non_reseller_value_multiplier": 1.0,
        "reseller_value_multiplier": 1.0,
        "complement_multiplier": 1.0,
        "saturation_penalty_multiplier": 1.0,
    }
    kwargs.update(overrides)
    return kwargs


def test_identity_calibration_leaves_spec_unchanged():
    spec = _tiny_spec()
    calibrated = apply_calibration(spec, reseller_id=RESELLER_ID, **_identity_kwargs())

    for orig, cal in zip(spec.bidder_profiles, calibrated.bidder_profiles):
        assert cal.base_values == orig.base_values
        assert [cg.bonus for cg in cal.complement_groups] == [cg.bonus for cg in orig.complement_groups]
        assert cal.saturation_penalty == orig.saturation_penalty


def test_non_reseller_multiplier_scales_only_non_reseller_base_values():
    spec = _tiny_spec()
    calibrated = apply_calibration(
        spec, reseller_id=RESELLER_ID, **_identity_kwargs(non_reseller_value_multiplier=1.5)
    )

    alice = next(bp for bp in calibrated.bidder_profiles if bp.bidder_id == "alice")
    reseller = next(bp for bp in calibrated.bidder_profiles if bp.bidder_id == RESELLER_ID)

    assert alice.base_values == {"A": 150.0, "B": 75.0}
    assert reseller.base_values == {"A": 40.0, "B": 40.0}  # unaffected


def test_reseller_multiplier_scales_only_reseller_base_values():
    spec = _tiny_spec()
    calibrated = apply_calibration(
        spec, reseller_id=RESELLER_ID, **_identity_kwargs(reseller_value_multiplier=0.5)
    )

    alice = next(bp for bp in calibrated.bidder_profiles if bp.bidder_id == "alice")
    reseller = next(bp for bp in calibrated.bidder_profiles if bp.bidder_id == RESELLER_ID)

    assert alice.base_values == {"A": 100.0, "B": 50.0}  # unaffected
    assert reseller.base_values == {"A": 20.0, "B": 20.0}


def test_complement_multiplier_scales_bonus_for_all_bidders():
    spec = _tiny_spec()
    calibrated = apply_calibration(
        spec, reseller_id=RESELLER_ID, **_identity_kwargs(complement_multiplier=2.0)
    )

    alice = next(bp for bp in calibrated.bidder_profiles if bp.bidder_id == "alice")
    assert [cg.bonus for cg in alice.complement_groups] == [40.0]


def test_saturation_multiplier_only_affects_bidders_with_saturation_start():
    spec = _tiny_spec()
    calibrated = apply_calibration(
        spec, reseller_id=RESELLER_ID, **_identity_kwargs(saturation_penalty_multiplier=3.0)
    )

    alice = next(bp for bp in calibrated.bidder_profiles if bp.bidder_id == "alice")
    reseller = next(bp for bp in calibrated.bidder_profiles if bp.bidder_id == RESELLER_ID)

    assert alice.saturation_penalty == pytest.approx(0.3)
    # reseller has saturation_start=None, so the multiplier is a no-op.
    assert reseller.saturation_penalty == 0.0


def test_calibration_leaves_budgets_and_classifications_untouched():
    spec = _tiny_spec()
    calibrated = apply_calibration(
        spec,
        reseller_id=RESELLER_ID,
        **_identity_kwargs(
            non_reseller_value_multiplier=2.0,
            reseller_value_multiplier=2.0,
            complement_multiplier=2.0,
            saturation_penalty_multiplier=2.0,
        ),
    )

    for orig, cal in zip(spec.bidder_profiles, calibrated.bidder_profiles):
        assert cal.budget_range == orig.budget_range
        assert cal.budget_cap == orig.budget_cap
        assert cal.core_items == orig.core_items
        assert cal.secondary_items == orig.secondary_items
        assert cal.low_interest_items == orig.low_interest_items


def test_original_spec_is_not_mutated():
    spec = _tiny_spec()
    original_alice_values = dict(spec.bidder_profiles[0].base_values)
    original_bonus = spec.bidder_profiles[0].complement_groups[0].bonus

    apply_calibration(
        spec, reseller_id=RESELLER_ID,
        **_identity_kwargs(non_reseller_value_multiplier=5.0, complement_multiplier=5.0),
    )

    assert spec.bidder_profiles[0].base_values == original_alice_values
    assert spec.bidder_profiles[0].complement_groups[0].bonus == original_bonus


def test_calibrated_spec_still_validates():
    """A calibrated spec must remain a valid ScenarioProfileSpec (invariants hold)."""
    spec = _tiny_spec()
    calibrated = apply_calibration(
        spec, reseller_id=RESELLER_ID,
        **_identity_kwargs(non_reseller_value_multiplier=1.35, reseller_value_multiplier=0.7),
    )
    # model_copy(update=...) skips re-validation; explicitly re-validate here.
    ScenarioProfileSpec.model_validate(calibrated.model_dump())


def test_iter_default_grid_covers_full_cartesian_product():
    combos = list(iter_default_grid())
    expected_count = 1
    for values in DEFAULT_GRID.values():
        expected_count *= len(values)

    assert len(combos) == expected_count
    # every combo distinct
    assert len({tuple(sorted(c.items())) for c in combos}) == expected_count
    # every combo has exactly the grid's keys
    assert all(set(c.keys()) == set(DEFAULT_GRID.keys()) for c in combos)


def test_iter_default_grid_respects_custom_grid():
    custom_grid = {"non_reseller_value_multiplier": [1.0, 2.0], "reseller_value_multiplier": [0.5]}
    combos = list(iter_default_grid(custom_grid))
    assert len(combos) == 2
    assert {c["non_reseller_value_multiplier"] for c in combos} == {1.0, 2.0}
    assert all(c["reseller_value_multiplier"] == 0.5 for c in combos)
