"""The frozen final-experiment specification.

Covers the case grid (the shared 8x8 anchor is emitted once, not three
times), the content hashes recorded for every frozen input, detection of a
changed input on reload, and that the replay command carries only the frozen
primary configuration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auctionlab.experiments.final_pipeline import (
    build_final_experiment_spec,
    load_final_experiment_spec,
    scalability_case_names,
    write_final_experiment_spec,
)
from auctionlab.llm.value_calibration import (
    ValueCalibration,
    write_calibration_config,
)
from scripts.run_frozen_final_experiment import build_command


def _fixture(tmp_path: Path):
    scenario = tmp_path / "scenario.json"
    scenario.write_text('{"scenario": true}\n')
    calibration = tmp_path / "calibration.json"
    write_calibration_config(
        ValueCalibration(
            family="uniform",
            scale=1.2,
            provenance={"accepted": True},
        ),
        calibration,
    )
    pack = (
        tmp_path
        / "packs"
        / "seed_0"
        / "anchor_1x1"
        / "frozen_elicitation.json"
    )
    pack.parent.mkdir(parents=True)
    pack.write_text('{"pack": true}\n')
    return scenario, calibration, pack


def _spec(tmp_path: Path):
    scenario, calibration, _pack = _fixture(tmp_path)
    return build_final_experiment_spec(
        scenario_spec=scenario,
        elicitation_pack_dir=tmp_path / "packs",
        calibration_config=calibration,
        seeds=[0],
        sizes=[1],
        fixed_size=1,
        sealed_max_rounds=40,
        correction_threshold=0.25,
        clock_max_rounds=50,
        clock_price_step=50,
        clock_top_k=3,
        clock_tie_threshold=100,
        robustness_models=[{"provider": "openai", "model": "gpt-test"}],
    )


def test_scalability_names_emit_shared_anchor_once():
    names = scalability_case_names([4, 8, 10], 8)
    assert names == [
        "goods_4x8",
        "goods_10x8",
        "bidders_8x4",
        "bidders_8x10",
        "joint_4x4",
        "anchor_8x8",
        "joint_10x10",
    ]


def test_frozen_spec_records_hashes_and_primary_design(tmp_path):
    spec = _spec(tmp_path)
    assert spec["status"] == "frozen"
    assert spec["dataset"]["case_count"] == 1
    assert spec["version"] == 3
    assert spec["event_policy"]["name"] == "final-v3"
    assert spec["event_policy"]["sealed"]["large_correction_followup"] is True
    clock = spec["event_policy"]["clock"]
    assert clock["framework"] == "frontier_v1"
    assert clock["frontier_vcg_single_pass"] is True
    assert clock["frontier_vcg_revealed_only"] is True
    assert clock["frontier_winner_closure"] is True
    assert clock["frontier_staged_revealed_vcg_closure"] is True
    assert spec["calibration"]["config"]["scale"] == pytest.approx(1.2)
    assert spec["model_robustness"]["models"][0]["model"] == "gpt-test"


def test_load_detects_changed_frozen_input(tmp_path):
    spec = _spec(tmp_path)
    path = write_final_experiment_spec(tmp_path / "final.json", spec)
    load_final_experiment_spec(path)
    Path(spec["dataset"]["packs"][0]["path"]).write_text("changed\n")
    with pytest.raises(ValueError, match="frozen input changed"):
        load_final_experiment_spec(path)


def test_final_command_contains_only_frozen_primary_configuration(tmp_path):
    spec = _spec(tmp_path)
    command = build_command(spec, output_dir=tmp_path / "results")
    joined = " ".join(command)
    assert "--event-policy final-v3" in joined
    assert "--sealed-stopping-rule no_new_refinements" in joined
    assert "--price-step 50.0" in joined
    assert "--clock-tie-threshold 100.0" in joined
    assert "--llm-cache-mode off" in joined


def test_clock_only_final_command_disables_sealed_arm(tmp_path):
    command = build_command(
        _spec(tmp_path),
        output_dir=tmp_path / "clock-results",
        mechanisms="clock",
    )
    joined = " ".join(command)
    assert "--sealed-elicitation-rounds 0" in joined
    assert "--elicited-clock" in command


def test_sealed_only_final_command_disables_clock_arm(tmp_path):
    command = build_command(
        _spec(tmp_path),
        output_dir=tmp_path / "sealed-results",
        mechanisms="sealed",
    )
    assert "--elicited-clock" not in command
    assert "--sealed-elicitation-rounds 40" in " ".join(command)


def test_rejects_unaccepted_calibration(tmp_path):
    scenario, calibration, _pack = _fixture(tmp_path)
    write_calibration_config(
        ValueCalibration(
            family="uniform",
            scale=1.2,
            provenance={"accepted": False},
        ),
        calibration,
    )
    with pytest.raises(ValueError, match="accepted=true"):
        build_final_experiment_spec(
            scenario_spec=scenario,
            elicitation_pack_dir=tmp_path / "packs",
            calibration_config=calibration,
            seeds=[0],
            sizes=[1],
            fixed_size=1,
            sealed_max_rounds=40,
            correction_threshold=0.25,
            clock_max_rounds=50,
            clock_price_step=50,
            clock_top_k=3,
            clock_tie_threshold=100,
        )
