"""Out-of-domain PV calibration benchmark: environments and artefacts.

Every test here is offline. Nothing in this file constructs an LLM client.
"""

from __future__ import annotations

import re

import pytest

from auctionlab.experiments.pv_calibration import (
    BENCHMARK_DOMAINS,
    DOMAIN_CATALOGS,
    PV_CALIBRATION_BENCHMARK_FORMAT,
    artefact_file_name,
    build_benchmark_scenario,
    bundle_key,
    load_benchmark_artefact,
    observations_from_artefact,
    synthetic_artefact,
    write_benchmark_artefact,
)
from auctionlab.llm.value_calibration import ValueCalibration


MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d+)?")


@pytest.fixture(params=list(BENCHMARK_DOMAINS))
def domain(request):
    return request.param


class TestDomainCoverage:
    def test_five_registered_domains(self):
        assert set(DOMAIN_CATALOGS) == set(BENCHMARK_DOMAINS)
        assert len(BENCHMARK_DOMAINS) == 5

    def test_domains_are_disjoint_from_pc_build(self):
        """Calibration must stay out of domain relative to the experiment."""
        from auctionlab.instances.structured import _ITEM_DISCLOSURE_LABELS

        pc_goods = set(_ITEM_DISCLOSURE_LABELS)
        for catalog in DOMAIN_CATALOGS.values():
            assert not set(catalog["goods"]) & pc_goods


class TestDeterminism:
    def test_same_seed_gives_identical_scenario(self, domain):
        first = build_benchmark_scenario(domain, seed=0)
        second = build_benchmark_scenario(domain, seed=0)
        assert first.person_seeds == second.person_seeds
        assert first.instance.items == second.instance.items
        assert first.instance.valuations == second.instance.valuations

    def test_different_seeds_give_different_valuations(self, domain):
        first = build_benchmark_scenario(domain, seed=0)
        second = build_benchmark_scenario(domain, seed=1)
        assert first.instance.valuations != second.instance.valuations
        # Jitter must not silently change the catalogue itself.
        assert first.instance.items == second.instance.items

    def test_synthetic_artefact_is_deterministic(self, domain):
        calibration = ValueCalibration(family="uniform", scale=1.5, budget_cap=False)
        first = synthetic_artefact(
            domain, seed=0, true_calibration=calibration, noise_scale=0.1
        )
        second = synthetic_artefact(
            domain, seed=0, true_calibration=calibration, noise_scale=0.1
        )
        assert first == second


class TestPublicSeedsLeakNoNumbers:
    def test_exactly_one_monetary_figure_per_seed(self, domain):
        scenario = build_benchmark_scenario(domain, seed=0)
        for bidder_id, seed_text in scenario.person_seeds.items():
            amounts = MONEY_RE.findall(seed_text)
            assert len(amounts) == 1, (
                f"{domain}/{bidder_id} disclosed {amounts}; the seed must "
                "contain exactly one overall budget and no item-level prices"
            )

    def test_seed_contains_no_bare_numbers(self, domain):
        """No base values, synergy bonuses, factors, or lookup tables."""
        scenario = build_benchmark_scenario(domain, seed=0)
        for bidder_id, seed_text in scenario.person_seeds.items():
            without_budget = MONEY_RE.sub("", seed_text)
            # Item ids like MONITOR_144HZ render as "monitor 144hz", so allow
            # digits that are part of a good's own name.
            for label in scenario.item_descriptions:
                without_budget = without_budget.replace(
                    label.replace("_", " ").lower(), ""
                )
            leftovers = re.findall(r"\d+", without_budget)
            assert not leftovers, (
                f"{domain}/{bidder_id} leaked numbers {leftovers}"
            )

    def test_seed_never_quotes_a_hidden_valuation(self, domain):
        scenario = build_benchmark_scenario(domain, seed=0)
        for bidder_id, seed_text in scenario.person_seeds.items():
            disclosed = {
                float(a.lstrip("$").replace(",", ""))
                for a in MONEY_RE.findall(seed_text)
            }
            truth = scenario.instance.valuations[bidder_id]
            ceiling = max(truth.values())
            # The single disclosed figure is the overall ceiling, rounded for
            # display -- never a singleton price or a bundle value.
            assert len(disclosed) == 1
            assert abs(next(iter(disclosed)) - ceiling) <= 1.0

    def test_seed_describes_priorities_and_exclusions(self, domain):
        scenario = build_benchmark_scenario(domain, seed=0)
        joined = " ".join(scenario.person_seeds.values())
        assert "mainly interested in" in joined
        assert "not interested in" in joined
        assert "maximum total willingness to pay" in joined

    def test_every_domain_discloses_alternatives_and_complements(self, domain):
        scenario = build_benchmark_scenario(domain, seed=0)
        joined = " ".join(scenario.person_seeds.values())
        assert (
            "choosing at most one" in joined or "related alternatives" in joined
        ), f"{domain} discloses no substitutes"
        assert "together" in joined, f"{domain} discloses no complements"


class TestGroundTruth:
    def test_valuations_are_monotone_and_non_negative(self, domain):
        scenario = build_benchmark_scenario(domain, seed=0)
        for bidder_id in scenario.instance.bidder_ids:
            table = scenario.instance.valuations[bidder_id]
            assert all(value >= 0 for value in table.values())
            for bundle, value in table.items():
                for item in bundle:
                    sub = bundle - {item}
                    if sub:
                        assert table[sub] <= value + 1e-9

    def test_profile_metadata_records_disclosed_budget(self, domain):
        scenario = build_benchmark_scenario(domain, seed=0)
        for bidder_id, profile in scenario.metadata["profiles"].items():
            ceiling = max(scenario.instance.valuations[bidder_id].values())
            assert profile["disclosed_budget_hint"] == pytest.approx(ceiling)

    def test_subset_selection_is_validated(self):
        with pytest.raises(ValueError, match="num_goods"):
            build_benchmark_scenario("home_office", num_goods=99)
        with pytest.raises(ValueError, match="num_bidders"):
            build_benchmark_scenario("home_office", num_bidders=0)

    def test_unknown_domain_rejected(self):
        with pytest.raises(ValueError, match="unknown benchmark domain"):
            build_benchmark_scenario("space_elevator")


class TestArtefacts:
    def test_round_trip_and_observation_extraction(self, tmp_path, domain):
        calibration = ValueCalibration(
            family="uniform", scale=2.0, budget_cap=False
        )
        payload = synthetic_artefact(
            domain, seed=0, true_calibration=calibration
        )
        path = write_benchmark_artefact(
            payload, tmp_path / artefact_file_name(domain, 0)
        )
        loaded = load_benchmark_artefact(path)
        assert loaded["format"] == PV_CALIBRATION_BENCHMARK_FORMAT

        observations = observations_from_artefact(loaded)
        assert observations
        for observation in observations:
            # synthetic raw = truth / scale, so truth = raw * scale exactly.
            assert observation.true_value == pytest.approx(
                observation.raw_value * 2.0
            )
            assert observation.domain == domain

    def test_rejects_a_foreign_json_file(self, tmp_path):
        path = tmp_path / "not_a_benchmark.json"
        path.write_text('{"format": "something else"}')
        with pytest.raises(ValueError, match="not a pv-calibration benchmark"):
            load_benchmark_artefact(path)

    def test_rejects_an_artefact_without_bidder_records(self, tmp_path):
        path = tmp_path / "missing_bidders.json"
        path.write_text(
            '{"format": "auctionlab.pv_calibration_benchmark", '
            '"version": 1, "bidder_ids": ["missing"]}'
        )
        with pytest.raises(ValueError, match="no non-empty bidders mapping"):
            load_benchmark_artefact(path)

    def test_missing_predictions_are_dropped_not_zeroed(self, tmp_path, domain):
        payload = synthetic_artefact(
            domain,
            seed=0,
            true_calibration=ValueCalibration(family="uniform", scale=2.0),
        )
        bidder_id = next(iter(payload["bidders"]))
        entry = payload["bidders"][bidder_id]
        dropped = entry["raw_provisional_values"].pop(0)
        observations = observations_from_artefact(payload)
        keys = {
            (o.bidder_id, bundle_key(o.bundle))
            for o in observations
        }
        assert (bidder_id, bundle_key(dropped["bundle"])) not in keys
        assert all(o.raw_value > 0 for o in observations if o.true_value > 0)
