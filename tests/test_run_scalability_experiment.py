"""Scalability grid construction.

Covers the default grid across sizes with one anchor per seed, repetition of
every case for each seed, rejection of a fixed size outside the grid, and the
scenario arguments the generated commands supply.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.run_scalability_experiment import (
    ScalabilityRun,
    _validate_live_args,
    build_command,
    build_scalability_runs,
)


def test_default_style_grid_includes_odd_sizes_and_one_anchor_per_seed():
    runs = build_scalability_runs(
        sizes=[4, 5, 6, 7, 8, 9, 10],
        fixed_size=8,
        seeds=[0],
    )

    assert len(runs) == 19
    assert sum(run.series == "anchor" for run in runs) == 1
    assert any(run.case_name == "goods_5x8" for run in runs)
    assert any(run.case_name == "bidders_8x7" for run in runs)
    assert any(run.case_name == "joint_9x9" for run in runs)


def test_grid_repeats_all_cases_for_each_seed():
    runs = build_scalability_runs(
        sizes=[4, 8, 10],
        fixed_size=8,
        seeds=[0, 1],
    )

    assert len(runs) == 14
    assert {run.seed for run in runs} == {0, 1}
    assert sum(run.series == "anchor" for run in runs) == 2


def test_fixed_size_must_be_in_grid():
    with pytest.raises(ValueError, match="fixed_size"):
        build_scalability_runs(sizes=[4, 5], fixed_size=8, seeds=[0])


def test_build_command_supplies_managed_scenario_arguments(tmp_path: Path):
    run = ScalabilityRun(3, "goods", 5, 8, "goods_5x8")
    command = build_command(
        run,
        scenario_spec=Path("scenario.json"),
        selection_policy="stratified",
        case_dir=tmp_path / "goods_5x8",
        live_args=["--provider", "gemini"],
    )

    assert command[0] == sys.executable
    assert command[1] == "examples/run_live_llm_curated_batch.py"
    assert command[command.index("--num-goods") + 1] == "5"
    assert command[command.index("--num-bidders") + 1] == "8"
    assert command[command.index("--scenario-seed") + 1] == "3"
    assert command[command.index("--selection-policy") + 1] == "stratified"
    assert command[-2:] == ["--provider", "gemini"]


def test_build_command_can_supply_case_specific_frozen_pack(tmp_path: Path):
    run = ScalabilityRun(3, "goods", 5, 8, "goods_5x8")
    pack = tmp_path / "frozen_elicitation.json"
    command = build_command(
        run,
        scenario_spec=Path("scenario.json"),
        selection_policy="stratified",
        case_dir=tmp_path / "goods_5x8",
        live_args=["--skip-baselines"],
        elicitation_pack=pack,
    )

    assert command[command.index("--elicitation-pack") + 1] == str(pack)


def test_forwarded_args_cannot_override_managed_flags():
    with pytest.raises(ValueError, match="--num-goods"):
        _validate_live_args(["--provider", "gemini", "--num-goods=6"])
