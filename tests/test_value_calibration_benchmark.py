"""Tests for the out-of-sample value-calibration benchmark script.

Covers deterministic benchmark generation, candidate-bundle construction,
PV-vs-ground-truth diagnostics on synthetic/fake PV data, and config
parsing. No live LLM/API calls -- ``--synthesize-fake-pv`` is a
deterministic noise model over the benchmark's own hidden ground truth,
never a network call.
"""

from __future__ import annotations

import json

import pytest

from scripts.run_value_calibration_benchmark import (
    DOMAIN_CATALOGS,
    ValueCalibrationBenchmarkConfig,
    build_benchmark,
    build_bundle_level_rows,
    build_by_bundle_size_rows,
    build_summary_rows,
    bundle_key,
    generate_candidate_bundles,
    load_config,
    parse_bundle_key,
    run_evaluate,
    run_generate,
    spearman_rank_correlation,
    synthesize_fake_pv,
    topk_recall,
    write_benchmark,
)


def test_all_documented_domains_are_registered():
    for domain in [
        "home_office",
        "travel_package",
        "camera_video_kit",
        "kitchen_appliance_bundle",
        "gaming_peripherals",
    ]:
        assert domain in DOMAIN_CATALOGS


def test_build_benchmark_is_deterministic():
    a = build_benchmark("home_office", num_goods=6, num_bidders=4, seed=0)
    b = build_benchmark("home_office", num_goods=6, num_bidders=4, seed=0)
    assert a == b


def test_build_benchmark_different_seed_changes_values_not_structure():
    a = build_benchmark("home_office", num_goods=6, num_bidders=4, seed=0)
    b = build_benchmark("home_office", num_goods=6, num_bidders=4, seed=1)
    assert a["candidate_bundles"] == b["candidate_bundles"]
    assert a["bidder_ids"] == b["bidder_ids"]
    assert a["ground_truth_valuations"] != b["ground_truth_valuations"]


def test_build_benchmark_rejects_out_of_range_sizes():
    with pytest.raises(ValueError):
        build_benchmark("home_office", num_goods=100, num_bidders=4, seed=0)
    with pytest.raises(ValueError):
        build_benchmark("home_office", num_goods=6, num_bidders=100, seed=0)


def test_build_benchmark_rejects_unknown_domain():
    with pytest.raises(ValueError):
        build_benchmark("not_a_real_domain", num_goods=4, num_bidders=2, seed=0)


def test_ground_truth_covers_every_candidate_bundle():
    benchmark = build_benchmark("gaming_peripherals", num_goods=6, num_bidders=4, seed=0)
    for bidder_id in benchmark["bidder_ids"]:
        assert set(benchmark["ground_truth_valuations"][bidder_id]) == set(
            benchmark["candidate_bundles"]
        )


def test_ground_truth_values_are_non_negative():
    benchmark = build_benchmark("kitchen_appliance_bundle", num_goods=6, num_bidders=4, seed=0)
    for values in benchmark["ground_truth_valuations"].values():
        assert all(v >= 0.0 for v in values.values())


class TestCandidateBundleGeneration:
    def test_includes_all_singletons_and_pairs(self):
        items = ["A", "B", "C", "D"]
        bundles = generate_candidate_bundles(items, max_candidate_bundle_size=2)
        singles = {frozenset({i}) for i in items}
        pairs = {frozenset({items[i], items[j]}) for i in range(4) for j in range(i + 1, 4)}
        assert singles <= set(bundles)
        assert pairs <= set(bundles)

    def test_includes_grand_bundle_and_all_but_one(self):
        items = ["A", "B", "C", "D", "E"]
        bundles = generate_candidate_bundles(items, max_candidate_bundle_size=2)
        assert frozenset(items) in bundles
        for item in items:
            assert frozenset(items) - {item} in bundles

    def test_no_duplicates(self):
        items = ["A", "B", "C"]
        # max_candidate_bundle_size covers everything already, so the
        # grand-bundle/all-but-one additions must not duplicate entries.
        bundles = generate_candidate_bundles(items, max_candidate_bundle_size=3)
        assert len(bundles) == len(set(bundles))

    def test_deterministic_order(self):
        items = ["A", "B", "C", "D"]
        a = generate_candidate_bundles(items, max_candidate_bundle_size=2)
        b = generate_candidate_bundles(items, max_candidate_bundle_size=2)
        assert a == b


def test_bundle_key_round_trip():
    bundle = frozenset({"B", "A", "C"})
    key = bundle_key(bundle)
    assert key == "A+B+C"
    assert parse_bundle_key(key) == bundle


def test_perfect_pv_yields_zero_error_and_perfect_rank_correlation():
    benchmark = build_benchmark("home_office", num_goods=6, num_bidders=4, seed=0)
    pv_by_bidder = benchmark["ground_truth_valuations"]

    bundle_rows = build_bundle_level_rows(benchmark, pv_by_bidder)
    assert all(row["signed_error"] == 0.0 for row in bundle_rows)
    assert all(row["abs_error"] == 0.0 for row in bundle_rows)

    summary_rows = build_summary_rows(benchmark, pv_by_bidder, bundle_rows)
    per_bidder_rows = [r for r in summary_rows if r["bidder_id"] != "ALL"]
    assert per_bidder_rows  # sanity: at least one bidder row
    for row in per_bidder_rows:
        assert row["mean_signed_error"] == 0.0
        assert row["mean_ratio"] == 1.0
        assert row["rank_correlation"] == pytest.approx(1.0)
        assert row["topk_recall_k3"] == 1.0


def test_synthesize_fake_pv_is_deterministic_and_bias_shows_up():
    benchmark = build_benchmark("home_office", num_goods=6, num_bidders=4, seed=0)

    a = synthesize_fake_pv(benchmark, noise_scale=0.1, bias_per_size=0.05, seed=1)
    b = synthesize_fake_pv(benchmark, noise_scale=0.1, bias_per_size=0.05, seed=1)
    assert a == b

    bundle_rows = build_bundle_level_rows(benchmark, a)
    summary_rows = build_summary_rows(
        benchmark, a, bundle_rows, large_bundle_size_threshold=4
    )
    for row in summary_rows:
        if row["bidder_id"] == "ALL":
            continue
        # A positive per-size bias must show up as large bundles being
        # over-valued more than small ones on average.
        assert row["large_bundle_overvaluation_bias"] > 0


def test_synthesize_fake_pv_zero_bias_and_noise_matches_ground_truth():
    benchmark = build_benchmark("home_office", num_goods=6, num_bidders=4, seed=0)
    pv = synthesize_fake_pv(benchmark, noise_scale=0.0, bias_per_size=0.0, seed=0)
    assert pv == benchmark["ground_truth_valuations"]


def test_topk_recall_perfect_and_imperfect():
    gt = {"A": 10.0, "B": 5.0, "C": 1.0}
    perfect_pv = dict(gt)
    assert topk_recall(list(gt), gt, perfect_pv, 2) == 1.0

    scrambled_pv = {"A": 1.0, "B": 5.0, "C": 10.0}
    # true top-2 is {A,B}; proxy top-2 (by scrambled_pv) is {C,B}: overlap 1/2
    assert topk_recall(list(gt), gt, scrambled_pv, 2) == 0.5


def test_topk_recall_none_when_k_exceeds_bundle_count():
    gt = {"A": 10.0}
    assert topk_recall(list(gt), gt, gt, 5) is None


def test_spearman_rank_correlation_basic_cases():
    assert spearman_rank_correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman_rank_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_rank_correlation([1, 1, 1], [1, 2, 3]) is None
    assert spearman_rank_correlation([1.0], [2.0]) is None


def test_by_bundle_size_rows_include_all_domain_aggregate():
    benchmark = build_benchmark("home_office", num_goods=6, num_bidders=4, seed=0)
    bundle_rows = build_bundle_level_rows(benchmark, benchmark["ground_truth_valuations"])
    by_size = build_by_bundle_size_rows(bundle_rows)
    domains = {row["domain"] for row in by_size}
    assert "home_office" in domains
    assert "ALL" in domains


class TestConfig:
    def test_load_config_round_trips_example_file(self):
        config = load_config("configs/value_calibration_example.json")
        assert isinstance(config, ValueCalibrationBenchmarkConfig)
        assert set(config.domains) <= set(DOMAIN_CATALOGS)
        assert config.num_goods >= 1
        assert config.num_bidders >= 1

    def test_default_config_is_valid(self):
        config = ValueCalibrationBenchmarkConfig()
        assert config.domains == ["home_office"]
        assert config.output_dir == "benchmarks/value_calibration"


class TestCliIntegration:
    def test_generate_writes_one_file_per_domain(self, tmp_path):
        args = _ns(
            config=None,
            domains=["home_office", "travel_package"],
            num_goods=6,
            num_bidders=4,
            seed=0,
            max_candidate_bundle_size=2,
            output_dir=str(tmp_path),
        )
        written = run_generate(args)
        assert len(written) == 2
        for path in written:
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["schema_version"] == "value_calibration_benchmark_v1"

    def test_generate_from_config_file(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "domains": ["camera_video_kit"],
            "num_goods": 5,
            "num_bidders": 3,
            "seed": 2,
            "max_candidate_bundle_size": 1,
            "output_dir": str(tmp_path / "out"),
        }))
        args = _ns(
            config=str(config_path),
            domains=None,
            num_goods=None,
            num_bidders=None,
            seed=None,
            max_candidate_bundle_size=None,
            output_dir=None,
        )
        written = run_generate(args)
        assert len(written) == 1
        data = json.loads(written[0].read_text())
        assert data["domain"] == "camera_video_kit"
        assert data["num_goods"] == 5
        assert data["num_bidders"] == 3

    def test_generate_domains_all_expands_to_every_registered_domain(self, tmp_path):
        args = _ns(
            config=None,
            domains=["all"],
            num_goods=6,
            num_bidders=4,
            seed=0,
            max_candidate_bundle_size=2,
            output_dir=str(tmp_path),
        )
        written = run_generate(args)
        domains_written = {json.loads(p.read_text())["domain"] for p in written}
        assert domains_written == set(DOMAIN_CATALOGS)

    def test_generate_from_example_config_writes_all_five_domains(self, tmp_path):
        args = _ns(
            config="configs/value_calibration_example.json",
            domains=None,
            num_goods=None,
            num_bidders=None,
            seed=None,
            max_candidate_bundle_size=None,
            output_dir=str(tmp_path),
        )
        written = run_generate(args)
        domains_written = {json.loads(p.read_text())["domain"] for p in written}
        assert domains_written == set(DOMAIN_CATALOGS)
        assert len(written) == len(DOMAIN_CATALOGS)

    def test_evaluate_requires_pv_source(self, tmp_path):
        benchmark = build_benchmark("home_office", num_goods=6, num_bidders=4, seed=0)
        bpath = tmp_path / "bench.json"
        bpath.write_text(json.dumps(benchmark))

        args = _ns(
            benchmark_file=[str(bpath)],
            pv_file=None,
            synthesize_fake_pv=False,
            noise_scale=0.15,
            bias_per_size=0.0,
            seed=0,
            large_bundle_size_threshold=4,
            output_dir=str(tmp_path / "out"),
        )
        with pytest.raises(SystemExit):
            run_evaluate(args)

    def test_evaluate_writes_all_four_outputs(self, tmp_path):
        benchmark = build_benchmark("home_office", num_goods=6, num_bidders=4, seed=0)
        bpath = tmp_path / "bench.json"
        write_benchmark(benchmark, tmp_path)
        # write_benchmark uses its own generated filename; find it.
        [bpath] = list(tmp_path.glob("value_calibration_benchmark_*.json"))

        out_dir = tmp_path / "reports"
        args = _ns(
            benchmark_file=[str(bpath)],
            pv_file=None,
            synthesize_fake_pv=True,
            noise_scale=0.1,
            bias_per_size=0.0,
            seed=0,
            large_bundle_size_threshold=4,
            output_dir=str(out_dir),
        )
        run_evaluate(args)

        assert (out_dir / "value_calibration_bundle_level.csv").exists()
        assert (out_dir / "value_calibration_summary.csv").exists()
        assert (out_dir / "value_calibration_by_bundle_size.csv").exists()
        assert (out_dir / "value_calibration_report.md").exists()
        report = (out_dir / "value_calibration_report.md").read_text()
        assert "value-calibration benchmark" in report.lower()


class _ns:
    """Minimal argparse.Namespace-like stand-in for direct function tests."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
