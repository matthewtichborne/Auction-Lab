"""Command-line surface of the population generator.

Covers argument parsing for batch regeneration, resumption, provider
selection, repair passes and budget guards, and that the committed design
loads as exactly 16 goods by 16 bidders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_pc_build_population import (
    _load_design,
    _parse_args,
    build_batch_prompt,
    build_batch_repair_prompt,
    build_client,
    conservative_request_cost,
    effective_generation_temperature,
    estimated_token_cost,
    parse_batch_response,
    resolve_token_prices,
)


DESIGN_PATH = Path(
    "scenarios/pc_build_v2/population_design_16x16.json"
)


def _valid_profile(archetype: dict, goods: list[dict]) -> dict:
    good_ids = [good["id"] for good in goods]
    return {
        "bidder_id": archetype["bidder_id"],
        "archetype_category": archetype["stratum"],
        "role": archetype["brief"],
        "identity_text": "An individual describing their background and intended work.",
        "budget_range": [500, 2000],
        "base_values": {good_id: 100 for good_id in good_ids},
        "substitute_groups": [
            {
                "items": ["CPU_HI", "CPU_MID", "CPU_LO"],
                "acquisition_mode": "choose_one",
                "backup_factor": 0.0,
            }
        ],
        "complement_groups": [
            {"items": ["CPU_HI", "MB_PRO"], "bonus": 100}
        ],
        "budget_cap": None,
        "saturation_start": 6,
        "saturation_penalty": 10,
        "notes": "",
        "core_items": good_ids[:4],
        "secondary_items": good_ids[4:10],
        "low_interest_items": good_ids[10:],
    }


def test_design_loads_exactly_16_by_16():
    design = _load_design(DESIGN_PATH)
    assert len(design["goods"]) == 16
    assert len(design["bidder_archetypes"]) == 16


def test_regenerate_batches_cli_parses_with_resume():
    args = _parse_args(
        [
            "--output", "out.json",
            "--raw-output-dir", "raw",
            "--manifest", "manifest.json",
            "--provider", "gemini",
            "--model", "model",
            "--resume",
            "--regenerate-batches", "2", "4",
        ]
    )
    assert args.resume is True
    assert args.regenerate_batches == [2, 4]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_environment_cli_accepts_additional_model_providers(provider):
    args = _parse_args([
        "--output", "out.json",
        "--raw-output-dir", "raw",
        "--manifest", "manifest.json",
        "--provider", provider,
        "--model", "test-model",
    ])
    assert args.provider == provider


def test_environment_cli_accepts_reasoning_repairs_and_budget_guard():
    args = _parse_args([
        "--output", "out.json",
        "--raw-output-dir", "raw",
        "--manifest", "manifest.json",
        "--provider", "openai",
        "--model", "gpt-5.6-sol",
        "--reasoning-effort", "high",
        "--max-batch-repair-retries", "3",
        "--max-estimated-cost-usd", "9",
    ])
    assert args.reasoning_effort == "high"
    assert args.max_batch_repair_retries == 3
    assert args.max_estimated_cost_usd == 9


def test_known_openai_price_and_cost_estimates():
    prices = resolve_token_prices(
        provider="openai",
        model="gpt-5.6-sol",
        input_cost_per_million=None,
        output_cost_per_million=None,
    )
    assert prices == (5.0, 30.0)
    assert estimated_token_cost(1_000_000, 1_000_000, prices) == 35.0
    assert conservative_request_cost(
        prompt="abc" * 100,
        max_tokens=1000,
        prices=prices,
    ) == pytest.approx(0.0305)


def test_batch_repair_prompt_includes_error_and_complete_invalid_response():
    prompt = build_batch_repair_prompt(
        original_prompt="Generate bidder A and bidder B.",
        invalid_response='{"bidder_profiles": [{"bidder_id": "A"}]}',
        validation_error="A: RAM_32 requires a positive base value",
    )
    assert "RAM_32 requires a positive base value" in prompt
    assert '"bidder_id": "A"' in prompt
    assert "complete corrected JSON object" in prompt
    assert "minimum economic changes" in prompt


def test_new_gemini_provenance_records_omitted_temperature():
    assert effective_generation_temperature(
        "gemini", "gemini-3.6-flash", 0.0
    ) is None
    assert effective_generation_temperature(
        "gemini", "gemini-3.5-flash-lite", 0.0
    ) is None
    assert effective_generation_temperature(
        "gemini", "gemini-3.1-flash-lite", 0.0
    ) == 0.0


def test_environment_client_honours_base_url():
    client = build_client(
        "anthropic",
        "claude-haiku-4-5-20251001",
        "test-key",
        0.0,
        12000,
        base_url="https://example.test/v1/",
        timeout=240.0,
    )
    assert client.base_url == "https://example.test/v1/"
    assert client.timeout == 240.0


def test_batch_prompt_requests_identity_and_multiway_substitutes():
    design = _load_design(DESIGN_PATH)
    prompt = build_batch_prompt(
        design,
        design["bidder_archetypes"][:4],
        [],
    )
    assert "identity_text" in prompt
    assert "three alternatives" in prompt
    assert '"acquisition_mode"' in prompt
    assert "can_use_multiple" in prompt
    assert "CURRENT COVERAGE" in prompt
    assert "required_positive_goods" in prompt
    assert "absolute dollar" in prompt
    assert "ACCEPTED MONETARY SCALES" in prompt


def test_parse_batch_accepts_exact_requested_profiles():
    design = _load_design(DESIGN_PATH)
    requested = design["bidder_archetypes"][:2]
    raw = json.dumps(
        {
            "bidder_profiles": [
                _valid_profile(archetype, design["goods"])
                for archetype in requested
            ]
        }
    )

    profiles = parse_batch_response(
        raw,
        expected_archetypes=requested,
        goods=design["goods"],
    )

    assert [profile.bidder_id for profile in profiles] == [
        archetype["bidder_id"] for archetype in requested
    ]
    assert profiles[0].identity_text == requested[0]["identity_text"]


def test_parse_batch_rejects_overlapping_classifications():
    design = _load_design(DESIGN_PATH)
    requested = design["bidder_archetypes"][:1]
    profile = _valid_profile(requested[0], design["goods"])
    profile["secondary_items"].append(profile["core_items"][0])

    with pytest.raises(ValueError, match="disjoint"):
        parse_batch_response(
            json.dumps({"bidder_profiles": [profile]}),
            expected_archetypes=requested,
            goods=design["goods"],
        )


def test_parse_batch_repairs_missing_classification_from_base_value():
    design = _load_design(DESIGN_PATH)
    requested = design["bidder_archetypes"][:1]
    profile = _valid_profile(requested[0], design["goods"])
    missing_positive = profile["secondary_items"].pop()
    missing_zero = profile["low_interest_items"].pop()
    profile["base_values"].pop(missing_zero)

    parsed = parse_batch_response(
        json.dumps({"bidder_profiles": [profile]}),
        expected_archetypes=requested,
        goods=design["goods"],
    )[0]

    assert missing_positive in parsed.secondary_items
    assert missing_zero in parsed.low_interest_items


def test_parse_batch_rejects_unknown_group_good_immediately():
    design = _load_design(DESIGN_PATH)
    requested = design["bidder_archetypes"][:1]
    profile = _valid_profile(requested[0], design["goods"])
    profile["substitute_groups"][0]["items"].append("CPU_IMAGINARY")

    with pytest.raises(ValueError, match="unknown goods"):
        parse_batch_response(
            json.dumps({"bidder_profiles": [profile]}),
            expected_archetypes=requested,
            goods=design["goods"],
        )


def test_parse_batch_rejects_nonpositive_substitute_good():
    design = _load_design(DESIGN_PATH)
    requested = design["bidder_archetypes"][:1]
    profile = _valid_profile(requested[0], design["goods"])
    profile["base_values"].pop("CPU_LO")

    with pytest.raises(ValueError, match="substitute-group goods"):
        parse_batch_response(
            json.dumps({"bidder_profiles": [profile]}),
            expected_archetypes=requested,
            goods=design["goods"],
        )


def test_parse_batch_rejects_fractional_saturation_penalty():
    design = _load_design(DESIGN_PATH)
    requested = design["bidder_archetypes"][:1]
    profile = _valid_profile(requested[0], design["goods"])
    profile["saturation_start"] = 4
    profile["saturation_penalty"] = 0.18

    with pytest.raises(ValueError, match="absolute-dollar"):
        parse_batch_response(
            json.dumps({"bidder_profiles": [profile]}),
            expected_archetypes=requested,
            goods=design["goods"],
        )


def test_parse_batch_rejects_bidder_interest_density_above_stratum_limit():
    design = _load_design(DESIGN_PATH)
    requested = design["bidder_archetypes"][:1]
    profile = _valid_profile(requested[0], design["goods"])

    with pytest.raises(ValueError, match="interest density"):
        parse_batch_response(
            json.dumps({"bidder_profiles": [profile]}),
            expected_archetypes=requested,
            goods=design["goods"],
            population_constraints=design["population_constraints"],
        )


def test_parse_batch_replaces_model_authored_identity_with_frozen_identity():
    design = _load_design(DESIGN_PATH)
    requested = design["bidder_archetypes"][:1]
    profile = _valid_profile(requested[0], design["goods"])
    profile["identity_text"] = "A buyer with a $2,000 budget."

    profiles = parse_batch_response(
        json.dumps({"bidder_profiles": [profile]}),
        expected_archetypes=requested,
        goods=design["goods"],
    )

    assert profiles[0].identity_text == requested[0]["identity_text"]


def test_parse_batch_enforces_required_positive_goods():
    design = _load_design(DESIGN_PATH)
    requested = [
        bidder
        for bidder in design["bidder_archetypes"]
        if bidder["bidder_id"] == "streamer_creator"
    ]
    profile = _valid_profile(requested[0], design["goods"])
    profile["base_values"].pop("GPU_AI")

    with pytest.raises(ValueError, match="required positive goods"):
        parse_batch_response(
            json.dumps({"bidder_profiles": [profile]}),
            expected_archetypes=requested,
            goods=design["goods"],
        )
