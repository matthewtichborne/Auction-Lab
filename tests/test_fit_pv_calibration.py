"""Offline fitting of the PV calibration: recovery, CV, and CLI outputs.

The fitter must never touch the network. These tests install a guard that
fails the test if any socket is opened, so a regression that adds a live call
is caught here rather than in a bill.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import socket
import sys

import pytest

from auctionlab.experiments.pv_calibration import (
    BENCHMARK_DOMAINS,
    artefact_file_name,
    build_benchmark_scenario,
    evaluate_predictions,
    fit_calibration,
    leave_one_domain_out,
    load_observations,
    objective_value,
    size_gamma_is_identifiable,
    synthesize_observations,
    synthetic_artefact,
    write_benchmark_artefact,
)
from auctionlab.llm.value_calibration import (
    ValueCalibration,
    load_calibration_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if anything in this module opens a socket."""

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "the offline fitter must not make network calls"
        )

    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)


def _observations(true_calibration, *, noise_scale=0.0, domains=None):
    observations = []
    for index, domain in enumerate(domains or BENCHMARK_DOMAINS):
        scenario = build_benchmark_scenario(domain, seed=0)
        budgets = {
            bidder_id: max(scenario.instance.valuations[bidder_id].values())
            for bidder_id in scenario.instance.bidder_ids
        }
        observations.extend(
            synthesize_observations(
                scenario,
                true_calibration=true_calibration,
                noise_scale=noise_scale,
                seed=index,
                disclosed_budgets=budgets,
            )
        )
    return observations


@pytest.fixture
def benchmark_dir(tmp_path):
    """Five synthetic artefacts generated from a known calibration."""
    truth = ValueCalibration(
        family="exponential",
        scale=1.6,
        size_gamma=0.95,
        size_threshold=3,
        budget_cap=False,
    )
    directory = tmp_path / "benchmark"
    for index, domain in enumerate(BENCHMARK_DOMAINS):
        write_benchmark_artefact(
            synthetic_artefact(
                domain,
                seed=0,
                true_calibration=truth,
                noise_scale=0.08,
                noise_seed=index,
            ),
            directory / artefact_file_name(domain, 0),
        )
    return directory


class TestParameterRecovery:
    def test_recovers_uniform_scale(self):
        truth = ValueCalibration(family="uniform", scale=1.83, budget_cap=False)
        result = fit_calibration(
            _observations(truth),
            family="uniform",
            budget_cap=False,
            objective="robust_log_error",
        )
        assert result.calibration.scale == pytest.approx(1.83, rel=0.02)

    def test_recovers_scale_and_size_gamma(self):
        truth = ValueCalibration(
            family="exponential",
            scale=1.75,
            size_gamma=0.92,
            size_threshold=3,
            budget_cap=False,
        )
        result = fit_calibration(
            _observations(truth),
            family="exponential",
            budget_cap=False,
            objective="robust_log_error",
        )
        assert result.calibration.scale == pytest.approx(1.75, rel=0.03)
        assert result.calibration.size_gamma == pytest.approx(0.92, rel=0.03)

    def test_recovers_scale_above_one_under_noise(self):
        truth = ValueCalibration(family="uniform", scale=1.5, budget_cap=False)
        result = fit_calibration(
            _observations(truth, noise_scale=0.15),
            family="uniform",
            budget_cap=False,
            objective="budget_normalized_mae",
        )
        assert result.calibration.scale == pytest.approx(1.5, rel=0.08)
        assert result.calibration.scale > 1.0

    def test_recovered_calibration_beats_raw(self):
        truth = ValueCalibration(family="uniform", scale=1.6, budget_cap=False)
        observations = _observations(truth, noise_scale=0.1)
        result = fit_calibration(
            observations, family="uniform", budget_cap=False
        )
        raw = evaluate_predictions(observations, ValueCalibration(family="none"))
        calibrated = evaluate_predictions(observations, result.calibration)
        assert calibrated["mae"] < raw["mae"]
        assert abs(calibrated["signed_bias"]) < abs(raw["signed_bias"])


class TestIdentifiability:
    def test_gamma_pinned_to_one_when_no_bundle_exceeds_threshold(self):
        truth = ValueCalibration(family="uniform", scale=1.5, budget_cap=False)
        observations = [
            observation
            for observation in _observations(truth)
            if observation.bundle_size <= 3
        ]
        assert not size_gamma_is_identifiable(observations, 3)
        result = fit_calibration(
            observations,
            family="exponential",
            size_threshold=3,
            budget_cap=False,
        )
        assert result.size_gamma_identifiable is False
        assert result.calibration.size_gamma == pytest.approx(1.0)

    def test_identifiable_when_larger_bundles_present(self):
        observations = _observations(
            ValueCalibration(family="uniform", scale=1.5, budget_cap=False)
        )
        assert size_gamma_is_identifiable(observations, 3)


class TestObjectives:
    def test_objective_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="objective"):
            objective_value([], ValueCalibration(family="none"), "l4_norm")

    def test_zero_true_values_are_handled_safely(self):
        truth = ValueCalibration(family="uniform", scale=2.0, budget_cap=False)
        observations = _observations(truth, domains=["home_office"])
        # Every domain has excluded goods, so zero-valued bundles exist.
        assert any(o.true_value == 0.0 for o in observations)
        metrics = evaluate_predictions(observations, truth)
        assert all(
            value is None or value == value  # not NaN
            for value in metrics.values()
            if isinstance(value, float)
        )


class TestLeaveOneDomainOut:
    def test_produces_one_fold_per_domain(self):
        truth = ValueCalibration(family="uniform", scale=1.6, budget_cap=False)
        folds = leave_one_domain_out(
            _observations(truth, noise_scale=0.05),
            family="uniform",
            budget_cap=False,
        )
        assert [fold.held_out_domain for fold in folds] == sorted(
            BENCHMARK_DOMAINS
        )
        for fold in folds:
            assert fold.fit.n_observations > 0
            assert fold.test_metrics_raw["n"] > 0
            assert fold.test_metrics_calibrated["n"] > 0
            # The held-out domain contributed nothing to its own fit.
            assert fold.fit.n_observations < len(
                _observations(truth, noise_scale=0.05)
            )

    def test_calibration_transfers_out_of_domain_when_bias_is_shared(self):
        truth = ValueCalibration(family="uniform", scale=1.6, budget_cap=False)
        folds = leave_one_domain_out(
            _observations(truth, noise_scale=0.05),
            family="uniform",
            budget_cap=False,
        )
        for fold in folds:
            assert (
                fold.test_metrics_calibrated["budget_normalized_mae"]
                < fold.test_metrics_raw["budget_normalized_mae"]
            )

    def test_requires_at_least_two_domains(self):
        observations = _observations(
            ValueCalibration(family="uniform", scale=1.5, budget_cap=False),
            domains=["home_office"],
        )
        with pytest.raises(ValueError, match="at least two domains"):
            leave_one_domain_out(observations, family="uniform")


class TestFitCli:
    def _run(self, argv):
        from scripts.fit_pv_calibration import main

        return main(argv)

    def test_writes_every_expected_artefact(self, benchmark_dir, tmp_path):
        output = tmp_path / "fit"
        assert (
            self._run(
                [
                    "--benchmark-dir", str(benchmark_dir),
                    "--output-dir", str(output),
                    "--objective", "robust_log_error",
                    "--no-plots",
                    "--grid-steps", "13",
                    "--grid-passes", "3",
                ]
            )
            == 0
        )
        for name in (
            "pv_calibration.json",
            "pv_calibration_observations.csv",
            "pv_calibration_folds.csv",
            "pv_calibration_metrics.csv",
            "pv_calibration_parameters.csv",
            "pv_calibration_by_bundle_size.csv",
        ):
            assert (output / name).exists(), name

    def test_output_config_is_directly_consumable(self, benchmark_dir, tmp_path):
        output = tmp_path / "fit"
        self._run(
            [
                "--benchmark-dir", str(benchmark_dir),
                "--output-dir", str(output),
                "--no-plots",
                "--grid-steps", "13",
                "--grid-passes", "3",
            ]
        )
        calibration = load_calibration_config(output / "pv_calibration.json")
        assert calibration.family == "exponential"
        assert calibration.scale == pytest.approx(1.6, rel=0.06)

    def test_provenance_records_the_required_fields(self, benchmark_dir, tmp_path):
        output = tmp_path / "fit"
        self._run(
            [
                "--benchmark-dir", str(benchmark_dir),
                "--output-dir", str(output),
                "--no-plots",
                "--grid-steps", "13",
                "--grid-passes", "3",
            ]
        )
        provenance = json.loads(
            (output / "pv_calibration.json").read_text()
        )["provenance"]
        for key in (
            "proxy_models",
            "benchmark_domains",
            "benchmark_seeds",
            "fitting_objective",
            "cross_validation",
            "benchmark_artefact_sha256",
            "created_at",
        ):
            assert key in provenance, key
        assert "git_revision" in provenance
        assert set(provenance["benchmark_domains"]) == set(BENCHMARK_DOMAINS)
        assert (
            provenance["cross_validation"]["folds_total"]
            == len(BENCHMARK_DOMAINS)
        )

    def test_fold_csv_has_one_pair_of_rows_per_domain(
        self, benchmark_dir, tmp_path
    ):
        output = tmp_path / "fit"
        self._run(
            [
                "--benchmark-dir", str(benchmark_dir),
                "--output-dir", str(output),
                "--no-plots",
                "--grid-steps", "13",
                "--grid-passes", "3",
            ]
        )
        with (output / "pv_calibration_folds.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2 * len(BENCHMARK_DOMAINS)
        assert {row["variant"] for row in rows} == {"raw", "calibrated"}
        assert {row["label"] for row in rows} == set(BENCHMARK_DOMAINS)

    def test_missing_artefacts_fail_clearly(self, tmp_path):
        with pytest.raises(SystemExit, match="no benchmark artefacts"):
            self._run(
                [
                    "--benchmark-dir", str(tmp_path / "empty"),
                    "--output-dir", str(tmp_path / "fit"),
                    "--no-plots",
                ]
            )

    def test_observation_csv_reports_raw_and_calibrated(
        self, benchmark_dir, tmp_path
    ):
        output = tmp_path / "fit"
        self._run(
            [
                "--benchmark-dir", str(benchmark_dir),
                "--output-dir", str(output),
                "--no-plots",
                "--grid-steps", "13",
                "--grid-passes", "3",
            ]
        )
        with (output / "pv_calibration_observations.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        for column in (
            "raw_value",
            "calibrated_value",
            "raw_abs_error",
            "calibrated_abs_error",
            "bundle_size",
            "true_value",
        ):
            assert column in rows[0]


class TestPrepareCliIsOffline:
    def test_dry_run_makes_no_calls(self, tmp_path, capsys):
        from scripts.prepare_pv_calibration_benchmark import main

        assert (
            main(
                [
                    "--dry-run",
                    "--domains", "all",
                    "--seeds", "0", "1",
                    "--output-dir", str(tmp_path / "bench"),
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "Would prepare 10 benchmark artefact(s)" in out
        assert "no LLM calls made" in out

    def test_resume_skips_completed_artefacts(self, benchmark_dir, capsys):
        from scripts.prepare_pv_calibration_benchmark import main

        main(
            [
                "--dry-run",
                "--domains", "home_office",
                "--seeds", "0",
                "--output-dir", str(benchmark_dir),
            ]
        )
        assert "skip (resume)" in capsys.readouterr().out

    def test_live_run_requires_provider_and_model(self, tmp_path):
        from scripts.prepare_pv_calibration_benchmark import main

        with pytest.raises(SystemExit):
            main(["--domains", "home_office", "--output-dir", str(tmp_path)])


class TestArtefactHashesAreStable:
    def test_loading_reports_content_hashes(self, benchmark_dir):
        paths = sorted(benchmark_dir.glob("pv_calibration_*.json"))
        _, hashes = load_observations(paths)
        assert len(hashes) == len(paths)
        assert all(len(value) == 64 for value in hashes.values())
        _, again = load_observations(paths)
        assert hashes == again
