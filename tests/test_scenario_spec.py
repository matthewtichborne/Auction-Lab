"""Tests for the frozen scenario-spec schema, loader, writer, and export script."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from auctionlab.instances.scenario_spec import (
    ScenarioProfileSpec,
    load_scenario_profile_spec,
    scenario_profile_spec_from_dict,
    scenario_profile_spec_to_dict,
    validate_spec_dict,
    write_scenario_profile_spec,
)


def _minimal_valid_spec_dict() -> dict:
    return {
        "schema_version": "pc_build_profile_spec_v1",
        "domain": "pc_build",
        "description": "A tiny test profile universe.",
        "goods": [
            {"id": "CPU", "description": "A processor."},
            {"id": "GPU", "description": "A graphics card."},
            {"id": "RAM", "description": "Memory."},
        ],
        "bidder_profiles": [
            {
                "bidder_id": "gamer",
                "role": "A gamer.",
                "budget_range": [500.0, 1000.0],
                "base_values": {"CPU": 300.0, "GPU": 400.0, "RAM": 100.0},
                "substitute_groups": [
                    {"items": ["CPU", "GPU"], "backup_factor": 0.1, "description": "n/a"}
                ],
                "complement_groups": [
                    {"items": ["CPU", "GPU", "RAM"], "bonus": 50.0, "description": "platform"}
                ],
                "budget_cap": None,
                "saturation_start": None,
                "saturation_penalty": 0.0,
                "notes": "",
                "core_items": ["CPU", "GPU"],
                "secondary_items": ["RAM"],
                "low_interest_items": [],
            },
            {
                "bidder_id": "office_user",
                "role": "An office worker.",
                "budget_range": [100.0, 300.0],
                "base_values": {"RAM": 80.0},
                "substitute_groups": [],
                "complement_groups": [],
                "budget_cap": 250.0,
                "saturation_start": 2,
                "saturation_penalty": 10.0,
                "notes": "",
                "core_items": ["RAM"],
                "secondary_items": [],
                "low_interest_items": ["CPU", "GPU"],
            },
        ],
        "generation": {"source": "test fixture"},
        "notes": None,
    }


# ---------------------------------------------------------------------------
# Valid spec
# ---------------------------------------------------------------------------

def test_minimal_valid_spec_parses():
    spec = scenario_profile_spec_from_dict(_minimal_valid_spec_dict())
    assert spec.schema_version == "pc_build_profile_spec_v1"
    assert spec.domain == "pc_build"
    assert len(spec.goods) == 3
    assert len(spec.bidder_profiles) == 2


def test_validate_spec_dict_valid():
    result = validate_spec_dict(_minimal_valid_spec_dict())
    assert result.valid
    assert result.errors == []
    assert result.spec is not None


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_to_dict_from_dict_round_trip():
    data = _minimal_valid_spec_dict()
    spec = scenario_profile_spec_from_dict(data)
    round_tripped = scenario_profile_spec_from_dict(scenario_profile_spec_to_dict(spec))
    assert round_tripped == spec


def test_write_and_load_round_trip(tmp_path: Path):
    spec = scenario_profile_spec_from_dict(_minimal_valid_spec_dict())
    path = tmp_path / "spec.json"
    write_scenario_profile_spec(spec, path)
    loaded = load_scenario_profile_spec(path)
    assert loaded == spec


def test_write_scenario_profile_spec_is_stable_indented_json(tmp_path: Path):
    spec = scenario_profile_spec_from_dict(_minimal_valid_spec_dict())
    path = tmp_path / "spec.json"
    write_scenario_profile_spec(spec, path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("{\n")
    assert '"schema_version": "pc_build_profile_spec_v1"' in text
    # re-parses to identical structure
    assert json.loads(text) == scenario_profile_spec_to_dict(spec)


# ---------------------------------------------------------------------------
# Schema validation failures
# ---------------------------------------------------------------------------

def test_missing_good_reference_fails():
    data = _minimal_valid_spec_dict()
    data["bidder_profiles"][0]["base_values"]["SSD"] = 100.0  # SSD is not a declared good
    result = validate_spec_dict(data)
    assert not result.valid
    assert any("SSD" in e for e in result.errors)


def test_duplicate_bidder_id_fails():
    data = _minimal_valid_spec_dict()
    data["bidder_profiles"][1]["bidder_id"] = "gamer"
    result = validate_spec_dict(data)
    assert not result.valid
    assert any("duplicate bidder ids" in e for e in result.errors)


def test_duplicate_good_id_fails():
    data = _minimal_valid_spec_dict()
    data["goods"][1]["id"] = "CPU"
    result = validate_spec_dict(data)
    assert not result.valid
    assert any("duplicate good ids" in e for e in result.errors)


def test_invalid_backup_factor_fails():
    data = _minimal_valid_spec_dict()
    data["bidder_profiles"][0]["substitute_groups"][0]["backup_factor"] = 1.5
    result = validate_spec_dict(data)
    assert not result.valid


def test_negative_backup_factor_fails():
    data = _minimal_valid_spec_dict()
    data["bidder_profiles"][0]["substitute_groups"][0]["backup_factor"] = -0.1
    result = validate_spec_dict(data)
    assert not result.valid


def test_negative_complement_bonus_fails():
    data = _minimal_valid_spec_dict()
    data["bidder_profiles"][0]["complement_groups"][0]["bonus"] = -5.0
    result = validate_spec_dict(data)
    assert not result.valid


def test_wrong_schema_version_fails():
    data = _minimal_valid_spec_dict()
    data["schema_version"] = "pc_build_profile_spec_v2"
    result = validate_spec_dict(data)
    assert not result.valid
    assert any("schema_version" in e for e in result.errors)


def test_wrong_domain_fails():
    data = _minimal_valid_spec_dict()
    data["domain"] = "laptops"
    result = validate_spec_dict(data)
    assert not result.valid
    assert any("domain" in e for e in result.errors)


def test_budget_range_low_gt_high_fails():
    data = _minimal_valid_spec_dict()
    data["bidder_profiles"][0]["budget_range"] = [1000.0, 500.0]
    result = validate_spec_dict(data)
    assert not result.valid


def test_negative_base_value_fails():
    data = _minimal_valid_spec_dict()
    data["bidder_profiles"][0]["base_values"]["CPU"] = -1.0
    result = validate_spec_dict(data)
    assert not result.valid


def test_all_zero_base_values_fails():
    data = _minimal_valid_spec_dict()
    data["bidder_profiles"][1]["base_values"] = {"RAM": 0.0}
    result = validate_spec_dict(data)
    assert not result.valid
    assert any("at least one positive base value" in e for e in result.errors)


def test_negative_saturation_penalty_fails():
    data = _minimal_valid_spec_dict()
    data["bidder_profiles"][1]["saturation_penalty"] = -5.0
    result = validate_spec_dict(data)
    assert not result.valid


def test_core_item_references_unknown_good_fails():
    data = _minimal_valid_spec_dict()
    data["bidder_profiles"][0]["core_items"] = ["NOT_A_GOOD"]
    result = validate_spec_dict(data)
    assert not result.valid


def test_empty_role_fails():
    data = _minimal_valid_spec_dict()
    data["bidder_profiles"][0]["role"] = ""
    result = validate_spec_dict(data)
    assert not result.valid


def test_load_scenario_profile_spec_raises_on_invalid_file(tmp_path: Path):
    data = _minimal_valid_spec_dict()
    data["domain"] = "not_pc_build"
    path = tmp_path / "bad_spec.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(Exception):
        load_scenario_profile_spec(path)


# ---------------------------------------------------------------------------
# Export script
# ---------------------------------------------------------------------------

def test_export_current_pc_build_profiles_builds_valid_spec():
    from scripts.export_current_pc_build_profiles import build_manual_spec

    spec = build_manual_spec(seed=0)
    assert isinstance(spec, ScenarioProfileSpec)
    assert len(spec.goods) == 10
    assert len(spec.bidder_profiles) == 10

    # Round-trips through validate_spec_dict cleanly.
    result = validate_spec_dict(scenario_profile_spec_to_dict(spec))
    assert result.valid, result.errors


def test_export_current_pc_build_profiles_writes_loadable_json(tmp_path: Path):
    from scripts.export_current_pc_build_profiles import build_manual_spec

    spec = build_manual_spec(seed=0)
    path = tmp_path / "exported.json"
    write_scenario_profile_spec(spec, path)
    loaded = load_scenario_profile_spec(path)
    assert loaded == spec


def test_export_is_deterministic_for_fixed_seed():
    from scripts.export_current_pc_build_profiles import build_manual_spec

    spec_a = build_manual_spec(seed=0)
    spec_b = build_manual_spec(seed=0)
    assert spec_a == spec_b


def test_v0_manual_spec_file_is_valid_if_present():
    """If the committed v0 manual spec exists, it must validate cleanly."""
    path = Path("scenarios/pc_build_v1/pc_build_profiles_v0_manual.json")
    if not path.exists():
        pytest.skip("v0 manual spec not generated in this checkout")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = validate_spec_dict(data)
    assert result.valid, result.errors
