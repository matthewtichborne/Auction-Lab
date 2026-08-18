"""Planning of elicitation-pack preparation.

Covers the projection scheme in which seven goods-specific master packs
cover the 19-cell grid for a seed, repetition across seeds, and that the
generated commands carry the required preparation flags while preserving any
explicit safety override.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from scripts.prepare_scalability_elicitation_packs import (
    _validate_live_args,
    build_master_command,
    build_master_pack_plans,
    master_for_run,
    redact_command,
)
from scripts.run_scalability_experiment import build_scalability_runs


def test_seven_master_plans_cover_19_case_grid_for_one_seed(tmp_path: Path):
    sizes = [4, 5, 6, 7, 8, 9, 10]
    plans = build_master_pack_plans(
        sizes=sizes,
        fixed_size=8,
        seeds=[0],
        output_dir=tmp_path,
    )
    runs = build_scalability_runs(
        sizes=sizes,
        fixed_size=8,
        seeds=[0],
    )

    assert len(plans) == 7
    assert len(runs) == 19
    assert [plan.num_bidders for plan in plans] == [8, 8, 8, 8, 10, 9, 10]
    for run in runs:
        master = master_for_run(run, plans)
        assert master.num_goods == run.num_goods
        assert master.num_bidders >= run.num_bidders


def test_master_plans_repeat_per_seed(tmp_path: Path):
    plans = build_master_pack_plans(
        sizes=[4, 8, 10],
        fixed_size=8,
        seeds=[0, 1],
        output_dir=tmp_path,
    )

    assert len(plans) == 6
    assert {plan.seed for plan in plans} == {0, 1}


def test_master_command_adds_required_preparation_flags(tmp_path: Path):
    plan = build_master_pack_plans(
        sizes=[8],
        fixed_size=8,
        seeds=[2],
        output_dir=tmp_path,
    )[0]
    command = build_master_command(
        plan,
        scenario_spec=Path("population.json"),
        selection_policy="coverage_stratified",
        live_args=["--provider", "gemini"],
    )

    assert command[0] == sys.executable
    assert command[1] == "scripts/generate_frozen_elicitation.py"
    assert command[command.index("--num-goods") + 1] == "8"
    assert command[command.index("--num-bidders") + 1] == "8"
    assert "--ask-initial-question" in command
    assert "--use-interest-map" in command
    assert "--use-provisional-valuations" in command
    assert command[command.index("--pv-max-tokens") + 1] == "12000"
    assert command[command.index("--max-parse-retries") + 1] == "2"
    assert command[command.index("--timeout") + 1] == "240"
    assert command[command.index("--llm-cache-mode") + 1] == "read-write"
    assert (
        command[command.index("--llm-cache-path") + 1]
        == str(tmp_path / "preparation_cache.sqlite")
    )
    assert "--log-dir" in command


def test_master_command_preserves_explicit_safety_overrides(tmp_path: Path):
    plan = build_master_pack_plans(
        sizes=[8],
        fixed_size=8,
        seeds=[0],
        output_dir=tmp_path,
    )[0]
    command = build_master_command(
        plan,
        scenario_spec=Path("population.json"),
        selection_policy="coverage_stratified",
        live_args=[
            "--pv-max-tokens=9000",
            "--max-parse-retries",
            "4",
            "--timeout",
            "300",
            "--llm-cache-mode",
            "refresh",
            "--llm-cache-path",
            "cache/custom.sqlite",
        ],
    )

    assert "--pv-max-tokens=9000" in command
    assert "--pv-max-tokens" not in command
    assert command.count("--max-parse-retries") == 1
    assert command[command.index("--max-parse-retries") + 1] == "4"
    assert command[command.index("--timeout") + 1] == "300"
    assert command[command.index("--llm-cache-mode") + 1] == "refresh"
    assert command[command.index("--llm-cache-path") + 1] == "cache/custom.sqlite"


def test_forwarded_args_cannot_override_master_dimensions():
    with pytest.raises(ValueError, match="--num-bidders"):
        _validate_live_args(
            ["--proxy-provider", "gemini", "--num-bidders=4"]
        )


def test_redact_command_removes_separate_and_equals_form_api_keys():
    command = [
        "python",
        "script.py",
        "--person-api-key",
        "person-secret",
        "--proxy-api-key=proxy-secret",
        "--model",
        "model-name",
    ]

    redacted = redact_command(command)

    assert redacted == [
        "python",
        "script.py",
        "--person-api-key",
        "<redacted>",
        "--proxy-api-key=<redacted>",
        "--model",
        "model-name",
    ]
    assert "person-secret" not in redacted
    assert all("proxy-secret" not in token for token in redacted)
