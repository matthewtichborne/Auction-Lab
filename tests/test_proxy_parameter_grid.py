"""Tests for the offline experiment-grid runner/aggregator.

Covers dry-run command construction, no-overwrite run-directory
allocation, and aggregation from fake CSV/manifest fixtures. Normal run
mode (which invokes ``examples/run_live_llm_curated_batch.py`` as a
subprocess) is exercised here only with a monkeypatched subprocess runner
that never spawns a real process -- no live LLM/API calls are made
anywhere in this file.
"""

from __future__ import annotations

import csv
import json

import pytest

from auctionlab.experiments.grid_config import ExperimentArm, ExperimentGrid
from scripts.run_proxy_parameter_grid import (
    RESULT_COLUMNS,
    aggregate,
    allocate_run_dir,
    arm_rows_from_run_dir,
    build_command,
    discover_run_dirs,
    dispatch_run,
    dry_run,
    load_run_manifest,
    planned_run_dir_label,
    write_run_manifest,
)


def _simple_grid(name="g", grid_type="final", n_arms=1, **arm_kwargs):
    arms = [ExperimentArm(name=f"arm{i}", **arm_kwargs) for i in range(n_arms)]
    return ExperimentGrid(grid_name=name, grid_type=grid_type, arms=arms)


class TestBuildCommand:
    def test_includes_python_and_curated_batch_script(self, tmp_path):
        arm = ExperimentArm(name="a1")
        cmd = build_command(arm, tmp_path / "run1")
        assert cmd[0].endswith("python") or "python" in cmd[0]
        assert cmd[1].endswith("run_live_llm_curated_batch.py")
        assert "--log-dir" in cmd
        assert cmd[cmd.index("--log-dir") + 1] == str(tmp_path / "run1")

    def test_overrides_provider_and_model(self, tmp_path):
        arm = ExperimentArm(name="a1", provider="ollama", model="llama3.1:8b")
        cmd = build_command(arm, tmp_path / "run1", provider="groq", model="other-model")
        assert cmd[cmd.index("--provider") + 1] == "groq"
        assert cmd[cmd.index("--model") + 1] == "other-model"

    def test_appends_api_key_and_base_url_when_given(self, tmp_path):
        arm = ExperimentArm(name="a1")
        cmd = build_command(arm, tmp_path / "run1", api_key="secret", base_url="http://x")
        assert cmd[cmd.index("--api-key") + 1] == "secret"
        assert cmd[cmd.index("--base-url") + 1] == "http://x"

    def test_no_api_key_or_base_url_when_not_given(self, tmp_path):
        arm = ExperimentArm(name="a1")
        cmd = build_command(arm, tmp_path / "run1")
        assert "--api-key" not in cmd
        assert "--base-url" not in cmd


class TestDryRun:
    def test_creates_no_directories_and_no_subprocess(self, tmp_path, monkeypatch):
        output_root = tmp_path / "out"
        grid = _simple_grid(n_arms=2)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("subprocess must never be invoked in dry-run mode")

        monkeypatch.setattr("subprocess.run", _fail_if_called)

        plans = dry_run(grid, output_root)

        assert not output_root.exists()
        assert len(plans) == 2
        for plan in plans:
            assert "<timestamp>" in plan["run_dir"]
            assert plan["command"][1].endswith("run_live_llm_curated_batch.py")

    def test_planned_run_dir_label_is_estimated(self, tmp_path):
        label = planned_run_dir_label(tmp_path, "g", "arm1")
        assert str(label).startswith(str(tmp_path / "g" / "arm1__"))

    def test_api_key_is_redacted_in_printed_and_returned_command(self, tmp_path, capsys):
        grid = _simple_grid(n_arms=1)
        plans = dry_run(grid, tmp_path, api_key="super-secret-key")

        assert "super-secret-key" not in " ".join(plans[0]["command"])
        assert "<redacted>" in plans[0]["command"]

        captured = capsys.readouterr()
        assert "super-secret-key" not in captured.out
        assert "<redacted>" in captured.out

    def test_no_api_key_flag_when_not_given(self, tmp_path, capsys):
        grid = _simple_grid(n_arms=1)
        plans = dry_run(grid, tmp_path)
        assert "--api-key" not in plans[0]["command"]
        assert "<redacted>" not in capsys.readouterr().out


class TestAllocateRunDir:
    def test_creates_directory(self, tmp_path):
        run_dir = allocate_run_dir(tmp_path, "g", "arm1")
        assert run_dir.exists()
        assert run_dir.is_dir()

    def test_never_overwrites_existing_directory(self, tmp_path, monkeypatch):
        # Freeze time so both allocations collide on the same timestamp.
        monkeypatch.setattr(
            "scripts.run_proxy_parameter_grid._timestamp", lambda: "20260101T000000Z"
        )
        first = allocate_run_dir(tmp_path, "g", "arm1")
        second = allocate_run_dir(tmp_path, "g", "arm1")
        assert first != second
        assert first.exists()
        assert second.exists()
        assert second.name == "arm1__20260101T000000Z_1"

    def test_third_collision_gets_next_suffix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_proxy_parameter_grid._timestamp", lambda: "20260101T000000Z"
        )
        allocate_run_dir(tmp_path, "g", "arm1")
        allocate_run_dir(tmp_path, "g", "arm1")
        third = allocate_run_dir(tmp_path, "g", "arm1")
        assert third.name == "arm1__20260101T000000Z_2"


class TestDispatchRun:
    def test_real_api_key_is_passed_unredacted_to_the_actual_subprocess_command(self, tmp_path):
        # Redaction is display-only (dry-run); a real dispatched run must
        # still authenticate correctly.
        grid = _simple_grid(n_arms=1)
        calls = []

        class _FakeCompleted:
            returncode = 0

        def _fake_runner(command):
            calls.append(command)
            return _FakeCompleted()

        dispatch_run(grid, tmp_path, api_key="super-secret-key", subprocess_runner=_fake_runner)

        assert "super-secret-key" in calls[0]
        assert "<redacted>" not in calls[0]

    def test_writes_manifest_and_never_calls_real_subprocess(self, tmp_path):
        grid = _simple_grid(n_arms=2)
        calls = []

        class _FakeCompleted:
            returncode = 0

        def _fake_runner(command):
            calls.append(command)
            return _FakeCompleted()

        results = dispatch_run(grid, tmp_path, subprocess_runner=_fake_runner)

        assert len(calls) == 2
        assert len(results) == 2
        for result in results:
            manifest_path = result["run_dir"] / "run_manifest.json"
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text())
            assert manifest["grid_name"] == grid.grid_name
            assert manifest["command"][1].endswith("run_live_llm_curated_batch.py")

    def test_reports_nonzero_returncode_without_raising(self, tmp_path, capsys):
        grid = _simple_grid(n_arms=1)

        class _FailedCompleted:
            returncode = 1

        results = dispatch_run(grid, tmp_path, subprocess_runner=lambda cmd: _FailedCompleted())
        assert results[0]["returncode"] == 1
        assert "WARNING" in capsys.readouterr().err


def _write_manifest(run_dir, *, config_name, top_k=None, safety_limits=None, **env_overrides):
    env = {
        "scenario_spec": "scenarios/pc_build_v1/x.json",
        "num_goods": 6,
        "num_bidders": 6,
        "scenario_seed": 0,
    }
    env.update(env_overrides)
    manifest = {
        "run_label": run_dir.name,
        "config_name": config_name,
        "grid_name": "g",
        "grid_type": "final",
        "label": "final",
        "environment": env,
        "arm_config": {
            "proxy_candidate_policy": {"pv_max_tokens": 1500, "max_bundle_size": None},
            "clock_treatment": {
                "clock_tie_threshold": 50.0,
                "clock_margin_threshold": 50.0,
                "price_step": 25.0,
                "max_rounds": 40,
                "top_k": [top_k] if top_k else [1],
            },
            "sealed_treatment": {"sealed_feedback_rule": "none", "sealed_elicitation_rounds": 0},
            "safety_limits": safety_limits if safety_limits is not None else {
                "per_bidder_refinement_query_limit": None,
                "global_refinement_query_safety_limit": 200,
            },
        },
        "command": ["python", "examples/run_live_llm_curated_batch.py"],
        "timestamp": "2026-01-01T00:00:00Z",
        "git_commit": "abc123",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestAggregation:
    def test_arm_rows_from_run_dir_surfaces_refinement_limit_and_cap_fields(self, tmp_path):
        run_dir = tmp_path / "g" / "arm1__ts"
        run_dir.mkdir(parents=True)
        _write_manifest(
            run_dir, config_name="arm1", top_k=1,
            safety_limits={
                "per_bidder_refinement_query_limit": 3,
                "global_refinement_query_safety_limit": 200,
            },
        )
        _write_csv(
            run_dir / "curated_run_summary.csv",
            ["scenario", "arm", "efficiency", "true_welfare", "full_info_welfare",
             "revenue", "surplus", "vq", "dq", "nl", "tok_in", "tok_out"],
            [{"scenario": "env1", "arm": "proxy clock k=1", "efficiency": "0.9",
              "true_welfare": "900", "full_info_welfare": "1000", "revenue": "400",
              "surplus": "500", "vq": "20", "dq": "0", "nl": "6", "tok_in": "1000",
              "tok_out": "500"}],
        )
        _write_csv(
            run_dir / "curated_clock_proxy_elicited_top_1.csv",
            ["instance_name", "proxy_reported_welfare", "rounds", "termination_reason",
             "failure_classification", "supplementary_atoms_total", "best_true_welfare",
             "best_true_efficiency", "final_true_welfare", "final_true_efficiency",
             "total_refinement_queries", "per_bidder_refinement_queries",
             "cap_binding_indicator", "safety_cap_hit"],
            [{"instance_name": "env1", "proxy_reported_welfare": "930", "rounds": "12",
              "termination_reason": "converged", "failure_classification": "no_failure_global",
              "supplementary_atoms_total": "40", "best_true_welfare": "960",
              "best_true_efficiency": "0.96", "final_true_welfare": "950",
              "final_true_efficiency": "0.95", "total_refinement_queries": "6",
              "per_bidder_refinement_queries": "b1:3;b2:3",
              "cap_binding_indicator": "True", "safety_cap_hit": "False"}],
        )

        rows = arm_rows_from_run_dir(run_dir)
        row = rows[0]
        assert row["per_bidder_refinement_query_limit"] == 3
        assert row["global_refinement_query_safety_limit"] == 200
        assert row["total_refinement_queries"] == "6"
        assert row["per_bidder_refinement_queries"] == "b1:3;b2:3"
        assert row["cap_binding_indicator"] == "True"
        assert row["safety_cap_hit"] == "False"

    def test_arm_rows_from_run_dir_null_limits_surface_as_blank(self, tmp_path):
        run_dir = tmp_path / "g" / "arm2__ts"
        run_dir.mkdir(parents=True)
        _write_manifest(
            run_dir, config_name="arm2", top_k=1,
            safety_limits={
                "per_bidder_refinement_query_limit": None,
                "global_refinement_query_safety_limit": None,
            },
        )
        _write_csv(
            run_dir / "curated_run_summary.csv",
            ["scenario", "arm", "efficiency", "true_welfare", "full_info_welfare",
             "revenue", "surplus", "vq", "dq", "nl", "tok_in", "tok_out"],
            [{"scenario": "env1", "arm": "sealed", "efficiency": "0.9",
              "true_welfare": "900", "full_info_welfare": "1000", "revenue": "400",
              "surplus": "500", "vq": "20", "dq": "0", "nl": "6", "tok_in": "1000",
              "tok_out": "500"}],
        )

        rows = arm_rows_from_run_dir(run_dir)
        row = rows[0]
        assert row["per_bidder_refinement_query_limit"] == ""
        assert row["global_refinement_query_safety_limit"] == ""
        # No mechanism-specific cap CSV was written for a plain "sealed" arm,
        # so these stay blank too, never a stray 0.
        assert row["total_refinement_queries"] == ""

    def test_arm_rows_from_run_dir_reads_summary_and_enriches(self, tmp_path):
        run_dir = tmp_path / "g" / "arm1__ts"
        run_dir.mkdir(parents=True)
        _write_manifest(run_dir, config_name="arm1", top_k=1)

        _write_csv(
            run_dir / "curated_run_summary.csv",
            ["scenario", "arm", "efficiency", "true_welfare", "full_info_welfare",
             "revenue", "surplus", "vq", "dq", "nl", "tok_in", "tok_out"],
            [
                {"scenario": "env1", "arm": "shared initial (nl+im+pv)", "efficiency": "",
                 "true_welfare": "", "full_info_welfare": "", "revenue": "", "surplus": "",
                 "vq": "0", "dq": "0", "nl": "6", "tok_in": "500", "tok_out": "200"},
                {"scenario": "env1", "arm": "sealed", "efficiency": "0.9", "true_welfare": "900",
                 "full_info_welfare": "1000", "revenue": "400", "surplus": "500",
                 "vq": "20", "dq": "0", "nl": "6", "tok_in": "1000", "tok_out": "500"},
                {"scenario": "env1", "arm": "proxy clock k=1", "efficiency": "0.95",
                 "true_welfare": "950", "full_info_welfare": "1000", "revenue": "420",
                 "surplus": "530", "vq": "10", "dq": "30", "nl": "6", "tok_in": "1200",
                 "tok_out": "600"},
            ],
        )
        _write_csv(
            run_dir / "curated_sealed_llm_comparison.csv",
            ["instance_name", "llm_proxy_reported_welfare"],
            [{"instance_name": "env1", "llm_proxy_reported_welfare": "880"}],
        )
        _write_csv(
            run_dir / "curated_clock_proxy_elicited_top_1.csv",
            ["instance_name", "proxy_reported_welfare", "rounds", "termination_reason",
             "failure_classification", "supplementary_atoms_total", "best_true_welfare",
             "best_true_efficiency", "final_true_welfare", "final_true_efficiency"],
            [{"instance_name": "env1", "proxy_reported_welfare": "930", "rounds": "12",
              "termination_reason": "converged", "failure_classification": "no_failure_global",
              "supplementary_atoms_total": "40", "best_true_welfare": "960",
              "best_true_efficiency": "0.96", "final_true_welfare": "950",
              "final_true_efficiency": "0.95"}],
        )
        _write_csv(
            run_dir / "curated_pv_candidate_bundle_stats.csv",
            ["scenario", "bidder_id", "candidate_bundles_generated", "candidate_bundles_sent_to_pv",
             "candidate_bundles_truncated", "candidate_truncation_reason", "max_candidate_bundles"],
            [{"scenario": "env1", "bidder_id": "b1", "candidate_bundles_generated": "30",
              "candidate_bundles_sent_to_pv": "20", "candidate_bundles_truncated": "True",
              "candidate_truncation_reason": "explicit cap", "max_candidate_bundles": "20"}],
        )

        rows = arm_rows_from_run_dir(run_dir)

        # The "shared initial" bookkeeping row is not itself an arm result.
        arms = {r["arm"] for r in rows}
        assert arms == {"sealed", "proxy clock k=1"}
        assert set(RESULT_COLUMNS) <= set(rows[0].keys())

        sealed_row = next(r for r in rows if r["arm"] == "sealed")
        assert sealed_row["final_true_welfare"] == 900
        assert sealed_row["reported_welfare"] == 880
        assert sealed_row["candidate_bundles_generated"] == 30
        assert sealed_row["candidate_bundles_truncated"] is True

        clock_row = next(r for r in rows if r["arm"] == "proxy clock k=1")
        assert clock_row["top_k"] == 1
        assert clock_row["reported_welfare"] == 930
        assert clock_row["clock_rounds"] == 12
        assert clock_row["termination_reason"] == "converged"
        assert clock_row["failure_classification"] == "no_failure_global"
        assert clock_row["best_true_welfare"] == 960
        assert clock_row["best_efficiency"] == 0.96
        # final_true_efficiency from the mechanism CSV overrides the plain
        # "efficiency" already present from curated_run_summary.csv.
        assert clock_row["final_efficiency"] == 0.95

    def test_discover_run_dirs_finds_only_dirs_with_manifest(self, tmp_path):
        good = tmp_path / "g" / "arm1__ts"
        good.mkdir(parents=True)
        _write_manifest(good, config_name="arm1")
        (tmp_path / "g" / "not_a_run").mkdir(parents=True)

        found = discover_run_dirs(tmp_path, "g")
        assert found == [good]

    def test_discover_run_dirs_empty_when_root_missing(self, tmp_path):
        assert discover_run_dirs(tmp_path / "nope") == []

    def test_load_run_manifest_returns_none_when_absent(self, tmp_path):
        assert load_run_manifest(tmp_path) is None

    def test_write_and_load_run_manifest_round_trip(self, tmp_path):
        grid = _simple_grid(n_arms=1)
        arm = grid.arms[0]
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        write_run_manifest(run_dir, grid=grid, arm=arm, command=["python", "x.py"])
        manifest = load_run_manifest(run_dir)
        assert manifest["config_name"] == arm.name
        assert manifest["grid_name"] == grid.grid_name
        assert manifest["command"] == ["python", "x.py"]

    def test_aggregate_writes_full_width_rows_across_multiple_runs(self, tmp_path):
        for i, (config_name, arm_label) in enumerate([
            ("arm1", "sealed"), ("arm2", "proxy clock k=2"),
        ]):
            run_dir = tmp_path / "g" / f"{config_name}__ts{i}"
            run_dir.mkdir(parents=True)
            _write_manifest(run_dir, config_name=config_name, top_k=2, scenario_seed=i)
            _write_csv(
                run_dir / "curated_run_summary.csv",
                ["scenario", "arm", "efficiency", "true_welfare", "full_info_welfare",
                 "revenue", "surplus", "vq", "dq", "nl", "tok_in", "tok_out"],
                [{"scenario": f"env{i}", "arm": arm_label, "efficiency": "0.8",
                  "true_welfare": "800", "full_info_welfare": "1000", "revenue": "300",
                  "surplus": "500", "vq": "15", "dq": "5", "nl": "4", "tok_in": "900",
                  "tok_out": "400"}],
            )

        results, by_config, by_env, report = aggregate(tmp_path, grid_name="g")

        assert len(results) == 2
        assert {r["config_name"] for r in results} == {"arm1", "arm2"}
        assert len(by_config) == 2
        assert len(by_env) == 2  # different scenario_seed => different environment
        assert "Parameter grid report" in report
        assert "Effect of clock top_k" in report

    def test_aggregate_empty_output_root_returns_empty(self, tmp_path):
        results, by_config, by_env, report = aggregate(tmp_path, grid_name="missing")
        assert results == []
        assert by_config == []
        assert by_env == []
        assert "0 arm-result rows" in report
