"""Out-of-domain calibration environments.

Covers construction of the synthetic scenarios used to fit the provisional-value
scale, and the structural requirements they must satisfy: a dense bidder must
still exclude something, and both substitute acquisition modes must be
present, so the fitted scale is not tuned on degenerate cases.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from auctionlab.experiments.pv_calibration_environments import (
    GENERATED_CALIBRATION_DOMAINS,
    build_environment_prompt,
    build_generated_environment_scenario,
    environment_design,
    environment_file_name,
    validate_environment_payload,
)
from auctionlab.llm.value_calibration import ValueCalibration
from scripts.validate_pv_calibration import _mechanism_rows, _pricing_metrics


def _valid_payload(domain: str) -> dict:
    design = environment_design(domain)
    goods = list(design["goods"])
    bidders = []
    for index, role in enumerate(design["bidder_roles"]):
        positive = goods[: 5 if index == 1 else 4]
        row = {
            "bidder_id": role["bidder_id"],
            "identity_text": role["role"],
            "budget_range": [400, 1800],
            "budget_cap": 1500,
            "base_values": {
                good: (100 + 20 * offset if good in positive else 0)
                for offset, good in enumerate(goods)
            },
            "substitute_groups": [],
            "complement_groups": [],
            "saturation_start": None,
            "saturation_penalty": 0,
            "core_items": positive[:2],
            "secondary_items": positive[2:3],
            "low_interest_items": positive[3:],
        }
        bidders.append(row)
    bidders[0]["substitute_groups"] = [
        {
            "items": goods[:2],
            "backup_factor": 0.1,
            "acquisition_mode": "choose_one",
            "description": "one is enough",
        }
    ]
    bidders[1]["substitute_groups"] = [
        {
            "items": goods[:2],
            "backup_factor": 0.8,
            "acquisition_mode": "can_use_multiple",
            "description": "both remain useful",
        }
    ]
    bidders[2]["complement_groups"] = [
        {
            "items": goods[:2],
            "bonus": 50,
            "description": "work better together",
        }
    ]
    return {"domain": domain, "bidders": bidders}


@pytest.mark.parametrize("domain", GENERATED_CALIBRATION_DOMAINS)
def test_generated_environment_builds_full_six_by_three_scenario(domain: str):
    payload = validate_environment_payload(
        _valid_payload(domain), expected_domain=domain
    )
    scenario = build_generated_environment_scenario(payload)

    assert len(scenario.instance.items) == 6
    assert len(scenario.instance.bidder_ids) == 3
    assert all(
        len(values) == 2**6 - 1
        for values in scenario.instance.valuations.values()
    )
    assert scenario.metadata["benchmark"] == "pv_calibration_generated"


def test_environment_rejects_dense_bidder_with_no_exclusion():
    domain = GENERATED_CALIBRATION_DOMAINS[0]
    payload = _valid_payload(domain)
    bidder = payload["bidders"][0]
    bidder["base_values"] = {
        good: 100 for good in environment_design(domain)["goods"]
    }
    bidder["low_interest_items"] = list(
        dict.fromkeys(
            bidder["low_interest_items"]
            + list(environment_design(domain)["goods"])[4:]
        )
    )

    with pytest.raises(ValueError, match="3--5 positive goods"):
        validate_environment_payload(payload, expected_domain=domain)


def test_environment_requires_both_substitute_acquisition_modes():
    domain = GENERATED_CALIBRATION_DOMAINS[0]
    payload = _valid_payload(domain)
    payload["bidders"][1]["substitute_groups"] = []

    with pytest.raises(ValueError, match="can_use_multiple"):
        validate_environment_payload(payload, expected_domain=domain)


def test_environment_instances_preserve_legacy_zero_filename():
    domain = GENERATED_CALIBRATION_DOMAINS[0]
    assert environment_file_name(domain, 0) == (
        f"pv_calibration_environment_{domain}.json"
    )
    assert environment_file_name(domain, 2) == (
        f"pv_calibration_environment_{domain}_instance2.json"
    )


def test_environment_instance_is_frozen_into_prompt_and_scenario():
    domain = GENERATED_CALIBRATION_DOMAINS[0]
    payload = _valid_payload(domain)
    payload["instance_index"] = 2
    validated = validate_environment_payload(
        payload,
        expected_domain=domain,
        expected_instance_index=2,
    )
    scenario = build_generated_environment_scenario(validated)

    assert '"instance_index": 2' in build_environment_prompt(
        domain, instance_index=2
    )
    assert scenario.metadata["scenario_seed"] == 2
    assert scenario.metadata["environment_instance_index"] == 2
    assert "instance2" in scenario.name


def test_environment_rejects_wrong_instance_index():
    domain = GENERATED_CALIBRATION_DOMAINS[0]
    payload = _valid_payload(domain)
    payload["instance_index"] = 1
    with pytest.raises(ValueError, match="instance_index must be 2"):
        validate_environment_payload(
            payload,
            expected_domain=domain,
            expected_instance_index=2,
        )


def test_generated_environment_supports_offline_mechanism_replay():
    domain = GENERATED_CALIBRATION_DOMAINS[0]
    environment = validate_environment_payload(
        _valid_payload(domain), expected_domain=domain
    )
    scenario = build_generated_environment_scenario(environment)
    bidders = {}
    for bidder_id in scenario.instance.bidder_ids:
        truth = scenario.instance.valuations[bidder_id]
        positive = sorted(
            item
            for item in scenario.instance.items
            if truth[frozenset({item})] > 0
        )
        excluded = sorted(set(scenario.instance.items) - set(positive))
        bidders[bidder_id] = {
            "nl_question": "What are you interested in and what is your budget?",
            "nl_answer": "A concise frozen answer.",
            "interest_map": {
                "interested_items": positive,
                "excluded_items": excluded,
                "complementary_groups": [],
                "complementary_group_evidence": [],
                "substitute_groups": [],
                "budget_hint": max(truth.values()),
                "reasoning": "Synthetic replay fixture.",
            },
            "candidate_bundles": [
                sorted(bundle)
                for bundle in sorted(
                    truth, key=lambda bundle: (len(bundle), sorted(bundle))
                )
            ],
            "raw_provisional_values": [
                {"bundle": sorted(bundle), "value": value}
                for bundle, value in truth.items()
            ],
        }
    artefact = {
        "domain": domain,
        "environment": environment,
        "bidders": bidders,
    }
    args = argparse.Namespace(
        sealed_max_rounds=2,
        clock_max_rounds=5,
        clock_price_step=50.0,
        clock_top_k=3,
    )

    rows = _mechanism_rows(
        artefact,
        variant="raw",
        calibration=ValueCalibration(family="none"),
        args=args,
    )

    assert {row["mechanism"] for row in rows} == {
        "initial",
        "sealed",
        "clock",
    }
    assert all(0 <= row["efficiency"] <= 1 for row in rows)


def test_pricing_metric_uses_bidder_payment_vector_and_welfare_normalisation():
    truth = SimpleNamespace(
        payments={"a": 100.0, "b": 50.0},
        revenue=150.0,
        welfare=500.0,
    )
    result = SimpleNamespace(
        payments={"a": 80.0, "b": 70.0},
        revenue=150.0,
    )

    metrics = _pricing_metrics(result, truth)

    assert metrics["payment_absolute_error"] == pytest.approx(40.0)
    assert metrics["payment_error_over_optimum_welfare"] == pytest.approx(
        0.08
    )
    assert metrics["revenue_absolute_error"] == pytest.approx(0.0)
    assert metrics["revenue_loss"] == pytest.approx(0.0)
