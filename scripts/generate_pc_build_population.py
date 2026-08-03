#!/usr/bin/env python3
"""Generate a validated bidder population against a frozen goods catalogue.

Goods and bidder briefs come from a population-design JSON file. Bidder
profiles are generated in small resumable batches; raw responses and prompts
are retained for provenance. No final ScenarioProfileSpec is published unless
schema validation and master-population coverage checks both pass.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from auctionlab.instances.population_design import (
    bidder_interest_density_violations,
    bidder_monetary_semantic_violations,
    freeze_validated_nested_orders,
    population_coverage_report,
    validate_nested_scalability_samples,
)
from auctionlab.instances.scenario_spec import (
    BidderProfileSpec,
    GoodSpec,
    ScenarioProfileSpec,
    write_scenario_profile_spec,
)
from auctionlab.llm.clients import OpenAICompatibleLlmClient


_OPENAI_PRICING_PER_MILLION_USD: dict[str, tuple[float, float]] = {
    "gpt-5.6": (5.0, 30.0),
    "gpt-5.6-sol": (5.0, 30.0),
    "gpt-5.6-terra": (2.5, 15.0),
    "gpt-5.6-luna": (1.0, 6.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-mini-2026-03-17": (0.75, 4.5),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-mini-2025-08-07": (0.25, 2.0),
}
_OPENAI_PRICING_AS_OF = "2026-07-29"
_OPENAI_PRICING_SOURCE = "https://developers.openai.com/api/docs/models"


def build_client(
    provider: str,
    model: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int | None,
    *,
    base_url: str | None = None,
    timeout: float = 180.0,
    reasoning_effort: str | None = None,
) -> OpenAICompatibleLlmClient:
    """Construct the environment-generation client used by this command."""
    if provider == "gemini":
        return OpenAICompatibleLlmClient.for_gemini(
            model=model,
            base_url=base_url
            or "https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    if provider == "groq":
        return OpenAICompatibleLlmClient.for_groq(
            model=model,
            base_url=base_url or "https://api.groq.com/openai/v1",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    if provider == "ollama":
        return OpenAICompatibleLlmClient.for_ollama(
            model=model,
            base_url=base_url or "http://localhost:11434/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    if provider == "mistral":
        return OpenAICompatibleLlmClient.for_mistral(
            model=model,
            base_url=base_url
            or "https://api.mistral.ai/v1",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    if provider == "openai":
        return OpenAICompatibleLlmClient.for_openai(
            model=model,
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            temperature=(
                None if model.startswith("gpt-5") else temperature
            ),
            max_tokens=max_tokens,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
    if provider == "anthropic":
        return OpenAICompatibleLlmClient.for_anthropic(
            model=model,
            base_url=base_url or "https://api.anthropic.com/v1/",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    raise ValueError(
        f"Unsupported provider: {provider!r} "
        "(expected gemini/groq/ollama/mistral/openai/anthropic)"
    )


def effective_generation_temperature(
    provider: str,
    model: str,
    requested_temperature: float,
) -> float | None:
    """Return the sampling temperature actually sent to the provider."""
    if (
        provider == "gemini"
        and model.startswith(("gemini-3.5-", "gemini-3.6-"))
    ):
        return None
    if provider == "openai" and model.startswith("gpt-5"):
        return None
    return requested_temperature


def effective_generation_reasoning_effort(
    provider: str,
    requested_reasoning_effort: str | None,
) -> str | None:
    return requested_reasoning_effort if provider == "openai" else None


def resolve_token_prices(
    *,
    provider: str,
    model: str,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> tuple[float, float] | None:
    """Resolve explicit prices first, then known current OpenAI prices."""
    if (
        input_cost_per_million is None
        and output_cost_per_million is None
    ):
        return (
            _OPENAI_PRICING_PER_MILLION_USD.get(model)
            if provider == "openai"
            else None
        )
    if (
        input_cost_per_million is None
        or output_cost_per_million is None
    ):
        raise ValueError(
            "input and output token prices must be supplied together"
        )
    if input_cost_per_million < 0 or output_cost_per_million < 0:
        raise ValueError("token prices must be non-negative")
    return input_cost_per_million, output_cost_per_million


def estimated_token_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    prices: tuple[float, float] | None,
) -> float | None:
    if prices is None or input_tokens is None or output_tokens is None:
        return None
    input_price, output_price = prices
    return (
        input_tokens * input_price + output_tokens * output_price
    ) / 1_000_000


def conservative_request_cost(
    *,
    prompt: str,
    max_tokens: int | None,
    prices: tuple[float, float] | None,
) -> float | None:
    """Estimate the upper-bound request cost used by the budget guard."""
    if prices is None:
        return None
    # UTF-8 bytes / 3 deliberately overestimates typical English/JSON tokens.
    estimated_input_tokens = max(1, (len(prompt.encode("utf-8")) + 2) // 3)
    return estimated_token_cost(
        estimated_input_tokens,
        max_tokens or 0,
        prices,
    )


def call_with_usage(
    client: OpenAICompatibleLlmClient,
    prompt: str,
    *,
    prices: tuple[float, float] | None,
) -> tuple[str, dict[str, Any]]:
    """Make one provider call and capture its usage and elapsed time."""
    started = time.monotonic()
    response = client.complete(prompt)
    elapsed = time.monotonic() - started
    input_tokens = client._last_input_tokens
    output_tokens = client._last_output_tokens
    return response, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": client._last_reasoning_tokens,
        "total_tokens": client._last_total_tokens,
        "finish_reason": client._last_finish_reason,
        "latency_seconds": round(elapsed, 3),
        "estimated_cost_usd": estimated_token_cost(
            input_tokens,
            output_tokens,
            prices,
        ),
    }


def build_batch_repair_prompt(
    *,
    original_prompt: str,
    invalid_response: str,
    validation_error: str,
) -> str:
    """Ask for the smallest complete correction after semantic validation."""
    return f"""Repair the previous bidder-profile batch response.

DETERMINISTIC VALIDATION ERROR:
{validation_error}

Return the complete corrected JSON object for all requested bidders, not a
patch and not only the affected bidder. Make the minimum economic changes
needed to resolve the error while preserving every already-valid bidder and
field.

Cross-field consistency requirements:
- every substitute-group and complement-group item must use a frozen good ID;
- every substitute-group member must have a positive base_values entry;
- choose_one groups require backup_factor 0;
- can_use_multiple groups require a positive backup_factor;
- positive core/secondary classifications require positive base values;
- all frozen goods must remain covered exactly once by the three
  classification arrays.
- budget_cap must lie within budget_range;
- saturation_penalty is an absolute dollar amount, not a fraction, and must
  be meaningful relative to the profile's positive base values.
- respect bidder/stratum positive-good limits. Prefer removing marginal
  standalone positive values and classify those goods as low interest. If a
  relationship must be narrowed, update its group and base values together;
  never leave a group member with zero value or remove a required-positive
  good.

Do not invent a value mechanically if removing an unsupported group member is
more faithful to the bidder description. Conversely, when the prose and
classification clearly retain positive interest, add a plausible positive
value consistent with the profile's existing scale.

ORIGINAL REQUEST:
{original_prompt}

PREVIOUS INVALID RESPONSE:
{invalid_response}

Return JSON only with exactly one top-level field named "bidder_profiles".
Do not include markdown fences or explanatory prose."""


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object while tolerating surrounding markdown fences."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Could not locate a JSON object in the model output"
            )
        return json.loads(text[start : end + 1])


def _load_design(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    goods = data.get("goods", [])
    bidders = data.get("bidder_archetypes", [])
    if len(goods) != 16 or len(bidders) != 16:
        raise ValueError(
            "the 16x16 design must contain exactly 16 goods and 16 bidder archetypes"
        )
    for good in goods:
        GoodSpec.model_validate(good)
    if len({good["id"] for good in goods}) != len(goods):
        raise ValueError("population design contains duplicate good IDs")
    if len({bidder["bidder_id"] for bidder in bidders}) != len(bidders):
        raise ValueError("population design contains duplicate bidder IDs")
    missing_archetype_fields = [
        index
        for index, bidder in enumerate(bidders)
        if not all(
            bidder.get(field)
            for field in ("bidder_id", "stratum", "brief", "identity_text")
        )
    ]
    if missing_archetype_fields:
        raise ValueError(
            "bidder archetypes require bidder_id, stratum, brief, and "
            "identity_text; "
            f"invalid indices={missing_archetype_fields}"
        )
    stratum_counts: dict[str, int] = {}
    for bidder in bidders:
        stratum = bidder["stratum"]
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
    if sorted(stratum_counts.values()) != [4, 4, 4, 4]:
        raise ValueError(
            "the 16x16 design must contain four bidder strata with four "
            f"archetypes each; got {stratum_counts}"
        )
    known_goods = {good["id"] for good in goods}
    for bidder in bidders:
        unknown_required = sorted(
            set(bidder.get("required_positive_goods", [])) - known_goods
        )
        if unknown_required:
            raise ValueError(
                f"{bidder['bidder_id']}: required_positive_goods reference "
                f"unknown goods {unknown_required}"
            )
    return data


def _coverage_snapshot(
    goods: list[dict[str, Any]],
    profiles: Sequence[BidderProfileSpec],
    strata: dict[str, str],
) -> dict[str, Any]:
    positive = {good["id"]: 0 for good in goods}
    core = {good["id"]: 0 for good in goods}
    positive_strata = {good["id"]: set() for good in goods}
    for profile in profiles:
        for good_id in positive:
            if profile.base_values.get(good_id, 0.0) > 0:
                positive[good_id] += 1
                positive_strata[good_id].add(strata[profile.bidder_id])
            if good_id in profile.core_items:
                core[good_id] += 1
    return {
        good_id: {
            "positive_bidders": positive[good_id],
            "core_bidders": core[good_id],
            "positive_strata": sorted(positive_strata[good_id]),
        }
        for good_id in positive
    }


def _economic_scale_snapshot(
    profiles: Sequence[BidderProfileSpec],
) -> dict[str, Any]:
    """Summarise accepted monetary scales for cross-batch calibration."""
    return {
        profile.bidder_id: {
            "budget_range": list(profile.budget_range),
            "budget_cap": profile.budget_cap,
            "positive_base_value_min": min(
                value for value in profile.base_values.values() if value > 0
            ),
            "positive_base_value_max": max(profile.base_values.values()),
            "complement_bonus_total": sum(
                group.bonus for group in profile.complement_groups
            ),
            "saturation_start": profile.saturation_start,
            "saturation_penalty_dollars": profile.saturation_penalty,
        }
        for profile in profiles
    }


def build_batch_prompt(
    design: dict[str, Any],
    archetypes: Sequence[dict[str, str]],
    existing_profiles: Sequence[BidderProfileSpec],
) -> str:
    """Build one self-contained profile-generation batch prompt."""
    goods = design["goods"]
    constraints = design["population_constraints"]
    strata = {
        bidder["bidder_id"]: bidder["stratum"]
        for bidder in design["bidder_archetypes"]
    }
    snapshot = _coverage_snapshot(goods, existing_profiles, strata)
    economic_snapshot = _economic_scale_snapshot(existing_profiles)
    return f"""You are generating one batch of structured bidder profiles for a
combinatorial-auction research population. Return JSON only, with exactly one
top-level field named "bidder_profiles". Do not use markdown fences.

FROZEN GOODS CATALOGUE (use only these IDs):
{json.dumps(goods, indent=2)}

GENERATE EXACTLY THESE BIDDER ARCHETYPES:
{json.dumps(list(archetypes), indent=2)}

CURRENT COVERAGE BEFORE THIS BATCH:
{json.dumps(snapshot, indent=2)}

ACCEPTED MONETARY SCALES FROM EARLIER BATCHES:
{json.dumps(economic_snapshot, indent=2)}

FINAL POPULATION CONSTRAINTS:
{json.dumps(constraints, indent=2)}

For every requested bidder return:
- bidder_id exactly as supplied;
- archetype_category exactly equal to the supplied stratum;
- role: concise 1-3 sentence role description;
- identity_text copied exactly from the supplied bidder archetype. Do not
  rewrite, expand, or paraphrase it;
- budget_range and optional budget_cap, all in dollars. A budget_cap must lie
  within its budget_range and represents the hard maximum total willingness
  to pay;
- base_values using plausible dollar values for every good they value
  positively; omit genuinely zero-value goods;
- every good in an archetype's required_positive_goods must have a positive
  base value. This is a population-coverage obligation, though the good may
  be classified as core, secondary, or low interest;
- semantically coherent substitute_groups. CPU, GPU, and RAM groups may
  contain three alternatives where appropriate. Every group must declare
  acquisition_mode:
  - choose_one only when owning multiple members gives no meaningful extra
    benefit; use backup_factor 0;
  - can_use_multiple when multiple related alternatives retain value for
    resale, inventory, redundancy, or deployment to different systems; use
    a positive backup_factor, with high factors for resellers/integrators.
  Functional similarity alone is not enough for choose_one. Every
  substitute-group member must have a positive base value;
- economically meaningful complete complement_groups of 2-5 goods;
- saturation_start/saturation_penalty. The penalty is an absolute dollar
  deduction per squared excess item, not a fraction or percentage. It should
  normally be at least 1% of the bidder's median positive item value so that
  it has a meaningful effect;
- monetary values must remain calibrated with earlier accepted batches:
  professional/reseller profiles may have larger budgets, but should not
  dominate merely because a later batch silently changes the dollar scale;
- core_items, secondary_items, and low_interest_items. Every frozen good must
  appear in exactly one of these three classifications for each bidder.

Ensure this batch helps close weak columns in CURRENT COVERAGE. Preserve
diversity: bidders must not all value every good, use identical groups, or
want the grand bundle. Singleton-category goods PSU_1000, COOL_AIO, and
CASE_ATX need broad positive interest and should participate in multiple
complementary systems across the final population.

Apply the interest-density limits in FINAL POPULATION CONSTRAINTS explicitly:
- a positive base_values entry means genuine positive acquisition interest;
- omit zero-interest goods from base_values and classify them as low interest;
- respect each stratum's maximum positive-good count and any bidder override;
- no good may exceed the population-wide maximum number of positive bidders;
- broad-interest reseller exceptions do not justify making ordinary gaming,
  office, or professional profiles interested in the entire catalogue.

Each profile must match this shape:
{{
  "bidder_id": "...",
  "archetype_category": "...",
  "role": "...",
  "identity_text": "...",
  "budget_range": [0, 0],
  "base_values": {{"GOOD_ID": 0}},
  "substitute_groups": [
    {{"items": ["A", "B"], "backup_factor": 0,
      "acquisition_mode": "choose_one", "description": "..."}}
  ],
  "complement_groups": [
    {{"items": ["A", "B"], "bonus": 100, "description": "..."}}
  ],
  "budget_cap": null,
  "saturation_start": null,
  "saturation_penalty": 0,
  "notes": "",
  "core_items": [],
  "secondary_items": [],
  "low_interest_items": []
}}
"""


def parse_batch_response(
    raw_text: str,
    *,
    expected_archetypes: Sequence[dict[str, str]],
    goods: Sequence[dict[str, Any]],
    population_constraints: dict[str, Any] | None = None,
) -> list[BidderProfileSpec]:
    data = extract_json_object(raw_text)
    raw_profiles = data.get("bidder_profiles")
    if not isinstance(raw_profiles, list):
        raise ValueError("response must contain bidder_profiles list")
    expected = {
        bidder["bidder_id"]: bidder
        for bidder in expected_archetypes
    }
    if {profile.get("bidder_id") for profile in raw_profiles} != set(expected):
        raise ValueError(
            "response bidder IDs do not exactly match the requested batch"
        )
    known_goods = {good["id"] for good in goods}
    parsed: list[BidderProfileSpec] = []
    for raw_profile in raw_profiles:
        # Identity is a frozen design input. Older retained responses may
        # contain model-authored identity text, so canonicalise it before
        # validation rather than requiring another paid generation call.
        canonical_raw_profile = {
            **raw_profile,
            "identity_text": expected[raw_profile["bidder_id"]]["identity_text"],
        }
        profile = BidderProfileSpec.model_validate(canonical_raw_profile)
        expected_stratum = expected[profile.bidder_id]["stratum"]
        if profile.archetype_category != expected_stratum:
            raise ValueError(
                f"{profile.bidder_id}: archetype_category must be "
                f"{expected_stratum!r}"
            )
        required_positive = set(
            expected[profile.bidder_id].get("required_positive_goods", [])
        )
        missing_required_positive = sorted(
            good_id
            for good_id in required_positive
            if profile.base_values.get(good_id, 0.0) <= 0
        )
        if missing_required_positive:
            raise ValueError(
                f"{profile.bidder_id}: required positive goods are absent or "
                f"non-positive: {missing_required_positive}"
            )
        referenced_goods = set(profile.base_values)
        for group in profile.substitute_groups:
            referenced_goods.update(group.items)
        for group in profile.complement_groups:
            referenced_goods.update(group.items)
        unknown_references = sorted(referenced_goods - known_goods)
        if unknown_references:
            raise ValueError(
                f"{profile.bidder_id}: profile references unknown goods "
                f"{unknown_references}"
            )
        if any(len(set(group.items)) < 2 for group in profile.substitute_groups):
            raise ValueError(
                f"{profile.bidder_id}: substitute groups need at least two "
                "distinct goods"
            )
        nonpositive_substitutes = sorted(
            {
                good_id
                for group in profile.substitute_groups
                for good_id in group.items
                if profile.base_values.get(good_id, 0.0) <= 0
            }
        )
        if nonpositive_substitutes:
            raise ValueError(
                f"{profile.bidder_id}: substitute-group goods need positive "
                f"base values: {nonpositive_substitutes}"
            )
        invalid_choose_one = [
            group.items
            for group in profile.substitute_groups
            if group.acquisition_mode == "choose_one"
            and group.backup_factor != 0
        ]
        if invalid_choose_one:
            raise ValueError(
                f"{profile.bidder_id}: choose_one groups require "
                f"backup_factor 0: {invalid_choose_one}"
            )
        invalid_multiple = [
            group.items
            for group in profile.substitute_groups
            if group.acquisition_mode == "can_use_multiple"
            and group.backup_factor <= 0
        ]
        if invalid_multiple:
            raise ValueError(
                f"{profile.bidder_id}: can_use_multiple groups require a "
                f"positive backup_factor: {invalid_multiple}"
            )
        if any(len(set(group.items)) < 2 for group in profile.complement_groups):
            raise ValueError(
                f"{profile.bidder_id}: complement groups need at least two "
                "distinct goods"
            )
        identity = profile.identity_text or ""
        identity_lower = identity.lower()
        forbidden_identity_terms = {
            "price",
            "budget",
            "value",
            "substitute",
            "complement",
        }
        mentioned_terms = sorted(
            term
            for term in forbidden_identity_terms
            if re.search(rf"\b{re.escape(term)}\w*\b", identity_lower)
        )
        mentioned_goods = sorted(
            good_id
            for good_id in known_goods
            if good_id.lower() in identity_lower
        )
        if "$" in identity or mentioned_terms or mentioned_goods:
            raise ValueError(
                f"{profile.bidder_id}: identity_text must be value-free; "
                f"terms={mentioned_terms}, goods={mentioned_goods}, "
                f"contains_dollar_sign={'$' in identity}"
            )
        classified = (
            set(profile.core_items)
            | set(profile.secondary_items)
            | set(profile.low_interest_items)
        )
        unknown_classifications = sorted(classified - known_goods)
        if unknown_classifications:
            raise ValueError(
                f"{profile.bidder_id}: classifications reference unknown "
                f"goods {unknown_classifications}"
            )
        if (
            set(profile.core_items) & set(profile.secondary_items)
            or set(profile.core_items) & set(profile.low_interest_items)
            or set(profile.secondary_items) & set(profile.low_interest_items)
        ):
            raise ValueError(
                f"{profile.bidder_id}: item classifications must be disjoint"
            )
        # Classification is descriptive metadata derived from the economic
        # fields. Repair a pure omission deterministically so a valid paid
        # response is not discarded for a clerical list error: positive-value
        # goods are secondary interest and zero-value goods are low interest.
        missing_classifications = sorted(known_goods - classified)
        if missing_classifications:
            secondary_items = list(profile.secondary_items)
            low_interest_items = list(profile.low_interest_items)
            for good_id in missing_classifications:
                target = (
                    secondary_items
                    if profile.base_values.get(good_id, 0.0) > 0
                    else low_interest_items
                )
                target.append(good_id)
            profile = profile.model_copy(
                update={
                    "secondary_items": secondary_items,
                    "low_interest_items": low_interest_items,
                }
            )
        nonpositive_interested = sorted(
            good_id
            for good_id in set(profile.core_items) | set(profile.secondary_items)
            if profile.base_values.get(good_id, 0.0) <= 0
        )
        if nonpositive_interested:
            raise ValueError(
                f"{profile.bidder_id}: core/secondary goods need positive "
                f"base values: {nonpositive_interested}"
            )
        monetary_violations = bidder_monetary_semantic_violations(
            profile,
            num_goods=len(goods),
        )
        if monetary_violations:
            raise ValueError(
                f"{profile.bidder_id}: monetary semantics invalid: "
                + "; ".join(monetary_violations)
            )
        density_violations = bidder_interest_density_violations(
            profile,
            num_goods=len(goods),
            constraints=population_constraints,
        )
        if density_violations:
            raise ValueError(
                f"{profile.bidder_id}: interest density invalid: "
                + "; ".join(density_violations)
            )
        parsed.append(profile)
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=Path(
            "scenarios/pc_build_v2/population_design_16x16.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=[
            "gemini",
            "groq",
            "ollama",
            "mistral",
            "openai",
            "anthropic",
        ],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="medium",
        help="OpenAI reasoning effort (ignored by non-OpenAI providers).",
    )
    parser.add_argument(
        "--max-batch-repair-retries",
        type=int,
        default=2,
        help="Maximum validation-aware repair calls after an invalid batch.",
    )
    parser.add_argument(
        "--input-cost-per-million",
        type=float,
        default=None,
        help="Override input-token price in USD per million tokens.",
    )
    parser.add_argument(
        "--output-cost-per-million",
        type=float,
        default=None,
        help="Override output-token price in USD per million tokens.",
    )
    parser.add_argument(
        "--max-estimated-cost-usd",
        type=float,
        default=None,
        help=(
            "Stop before a call whose conservative maximum would exceed this "
            "run's estimated API spend."
        ),
    )
    parser.add_argument("--profiles-per-call", type=int, default=4)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seeds whose complete 4-10 scalability grids must pass.",
    )
    parser.add_argument(
        "--skip-economic-sample-validation",
        action="store_true",
        help="Debugging only: omit full-information allocation checks.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse existing raw batch responses and call the provider only "
            "for missing batches."
        ),
    )
    parser.add_argument(
        "--regenerate-batches",
        type=int,
        nargs="+",
        default=[],
        help=(
            "With --resume, make fresh calls for these one-based batch "
            "numbers while reusing the other response files."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write batch prompts only; make no provider calls.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.profiles_per_call < 1:
        raise SystemExit("Error: --profiles-per-call must be positive")
    if args.max_batch_repair_retries < 0:
        raise SystemExit(
            "Error: --max-batch-repair-retries must be non-negative"
        )
    if (
        args.max_estimated_cost_usd is not None
        and args.max_estimated_cost_usd <= 0
    ):
        raise SystemExit("Error: --max-estimated-cost-usd must be positive")
    if args.regenerate_batches and not args.resume:
        raise SystemExit("Error: --regenerate-batches requires --resume")
    try:
        token_prices = resolve_token_prices(
            provider=args.provider,
            model=args.model,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    if args.max_estimated_cost_usd is not None and token_prices is None:
        raise SystemExit(
            "Error: a cost ceiling requires known model pricing; pass both "
            "--input-cost-per-million and --output-cost-per-million"
        )
    design = _load_design(args.design)
    goods = design["goods"]
    archetypes = design["bidder_archetypes"]
    num_batches = (
        len(archetypes) + args.profiles_per_call - 1
    ) // args.profiles_per_call
    invalid_batch_numbers = sorted(
        set(args.regenerate_batches) - set(range(1, num_batches + 1))
    )
    if invalid_batch_numbers:
        raise SystemExit(
            f"Error: invalid --regenerate-batches values "
            f"{invalid_batch_numbers}; valid range is 1..{num_batches}"
        )
    args.raw_output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    client = None

    profiles: list[BidderProfileSpec] = []
    batches: list[dict[str, Any]] = []
    cumulative_cost_usd = 0.0

    def manifest_payload(
        status: str,
        *,
        failure: str | None = None,
        population_coverage: dict[str, Any] | None = None,
        sample_validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pricing = None
        if token_prices is not None:
            pricing = {
                "input_cost_per_million_usd": token_prices[0],
                "output_cost_per_million_usd": token_prices[1],
                "as_of": _OPENAI_PRICING_AS_OF,
                "source": (
                    _OPENAI_PRICING_SOURCE
                    if args.input_cost_per_million is None
                    else "cli_override"
                ),
            }
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "failure": failure,
            "design": str(args.design),
            "output": str(args.output),
            "provider": args.provider,
            "model": args.model,
            "temperature": effective_generation_temperature(
                args.provider, args.model, args.temperature
            ),
            "requested_temperature": args.temperature,
            "reasoning_effort": effective_generation_reasoning_effort(
                args.provider, args.reasoning_effort
            ),
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_batch_repair_retries": args.max_batch_repair_retries,
            "pricing": pricing,
            "max_estimated_cost_usd": args.max_estimated_cost_usd,
            "cumulative_estimated_cost_usd": round(
                cumulative_cost_usd, 8
            ),
            "batches": batches,
            "population_coverage": population_coverage,
            "sample_validation": sample_validation,
        }

    def write_manifest(
        status: str,
        *,
        failure: str | None = None,
        population_coverage: dict[str, Any] | None = None,
        sample_validation: dict[str, Any] | None = None,
    ) -> None:
        args.manifest.write_text(
            json.dumps(
                manifest_payload(
                    status,
                    failure=failure,
                    population_coverage=population_coverage,
                    sample_validation=sample_validation,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    def enforce_budget(next_prompt: str) -> float | None:
        projected_cost = conservative_request_cost(
            prompt=next_prompt,
            max_tokens=args.max_tokens,
            prices=token_prices,
        )
        if (
            args.max_estimated_cost_usd is not None
            and projected_cost is not None
            and cumulative_cost_usd + projected_cost
            > args.max_estimated_cost_usd
        ):
            message = (
                "cost ceiling would be exceeded before the next call: "
                f"spent≈${cumulative_cost_usd:.4f}, "
                f"next-call upper bound≈${projected_cost:.4f}, "
                f"ceiling=${args.max_estimated_cost_usd:.4f}"
            )
            write_manifest("budget_stopped", failure=message)
            raise SystemExit(f"Error: {message}")
        return projected_cost

    for start in range(0, len(archetypes), args.profiles_per_call):
        batch_number = start // args.profiles_per_call + 1
        requested = archetypes[start : start + args.profiles_per_call]
        prompt = build_batch_prompt(design, requested, profiles)
        prompt_path = args.raw_output_dir / f"batch_{batch_number:02d}_prompt.txt"
        raw_path = args.raw_output_dir / f"batch_{batch_number:02d}_response.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        if args.dry_run:
            print(f"Wrote prompt: {prompt_path}")
            continue
        reused_response = (
            args.resume
            and raw_path.exists()
            and batch_number not in args.regenerate_batches
        )
        archived_raw_path = None
        attempts: list[dict[str, Any]] = []
        if reused_response:
            raw_text = raw_path.read_text(encoding="utf-8")
            attempts.append(
                {
                    "attempt": 1,
                    "kind": "initial",
                    "response_source": "reused",
                    "prompt": str(prompt_path),
                    "raw_response": str(raw_path),
                    "input_tokens": None,
                    "output_tokens": None,
                    "reasoning_tokens": None,
                    "total_tokens": None,
                    "finish_reason": None,
                    "latency_seconds": 0.0,
                    "estimated_cost_usd": 0.0,
                    "projected_max_cost_usd": 0.0,
                }
            )
        else:
            if raw_path.exists() and batch_number in args.regenerate_batches:
                timestamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%S%fZ"
                )
                archived_raw_path = raw_path.with_name(
                    f"{raw_path.stem}.previous_{timestamp}{raw_path.suffix}"
                )
                shutil.copy2(raw_path, archived_raw_path)
            if client is None:
                client = build_client(
                    args.provider,
                    args.model,
                    args.api_key,
                    args.temperature,
                    args.max_tokens,
                    base_url=args.base_url,
                    timeout=args.timeout,
                    reasoning_effort=effective_generation_reasoning_effort(
                        args.provider, args.reasoning_effort
                    ),
                )
            projected_cost = enforce_budget(prompt)
            try:
                raw_text, usage = call_with_usage(
                    client, prompt, prices=token_prices
                )
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": 1,
                        "kind": "initial",
                        "response_source": "generated",
                        "prompt": str(prompt_path),
                        "raw_response": None,
                        "error": repr(exc),
                        "projected_max_cost_usd": projected_cost,
                    }
                )
                batches.append(
                    {
                        "batch": batch_number,
                        "bidder_ids": [
                            archetype["bidder_id"] for archetype in requested
                        ],
                        "attempts": attempts,
                        "status": "api_error",
                    }
                )
                write_manifest("api_error", failure=repr(exc))
                raise
            raw_path.write_text(raw_text, encoding="utf-8")
            call_cost = usage["estimated_cost_usd"]
            if call_cost is None and token_prices is not None:
                call_cost = estimated_token_cost(
                    max(1, (len(prompt.encode("utf-8")) + 2) // 3),
                    max(1, (len(raw_text.encode("utf-8")) + 2) // 3),
                    token_prices,
                )
                usage["estimated_cost_source"] = "utf8_byte_estimate"
            else:
                usage["estimated_cost_source"] = "provider_usage"
            cumulative_cost_usd += call_cost or 0.0
            usage["estimated_cost_usd"] = call_cost
            attempts.append(
                {
                    "attempt": 1,
                    "kind": "initial",
                    "response_source": "generated",
                    "prompt": str(prompt_path),
                    "raw_response": str(raw_path),
                    "projected_max_cost_usd": projected_cost,
                    **usage,
                }
            )

        parsed = None
        last_error: Exception | None = None
        for repair_number in range(args.max_batch_repair_retries + 1):
            try:
                parsed = parse_batch_response(
                    raw_text,
                    expected_archetypes=requested,
                    goods=goods,
                    population_constraints=design[
                        "population_constraints"
                    ],
                )
                attempts[-1]["passed_validation"] = True
                break
            except Exception as exc:
                last_error = exc
                attempts[-1]["passed_validation"] = False
                attempts[-1]["validation_error"] = str(exc)
                if attempts[-1]["raw_response"] == str(raw_path):
                    timestamp = datetime.now(timezone.utc).strftime(
                        "%Y%m%dT%H%M%S%fZ"
                    )
                    invalid_path = raw_path.with_name(
                        f"{raw_path.stem}.invalid_{timestamp}{raw_path.suffix}"
                    )
                    shutil.copy2(raw_path, invalid_path)
                    attempts[-1]["raw_response"] = str(invalid_path)
                if repair_number >= args.max_batch_repair_retries:
                    break
                if client is None:
                    client = build_client(
                        args.provider,
                        args.model,
                        args.api_key,
                        args.temperature,
                        args.max_tokens,
                        base_url=args.base_url,
                        timeout=args.timeout,
                        reasoning_effort=(
                            effective_generation_reasoning_effort(
                                args.provider, args.reasoning_effort
                            )
                        ),
                    )
                repair_prompt = build_batch_repair_prompt(
                    original_prompt=prompt,
                    invalid_response=raw_text,
                    validation_error=str(exc),
                )
                repair_prompt_path = args.raw_output_dir / (
                    f"batch_{batch_number:02d}_repair_"
                    f"{repair_number + 1:02d}_prompt.txt"
                )
                repair_raw_path = args.raw_output_dir / (
                    f"batch_{batch_number:02d}_repair_"
                    f"{repair_number + 1:02d}_response.txt"
                )
                repair_prompt_path.write_text(
                    repair_prompt, encoding="utf-8"
                )
                projected_cost = enforce_budget(repair_prompt)
                try:
                    raw_text, usage = call_with_usage(
                        client, repair_prompt, prices=token_prices
                    )
                except Exception as repair_exc:
                    attempts.append(
                        {
                            "attempt": len(attempts) + 1,
                            "kind": "repair",
                            "repair_number": repair_number + 1,
                            "response_source": "generated",
                            "prompt": str(repair_prompt_path),
                            "raw_response": None,
                            "error": repr(repair_exc),
                            "projected_max_cost_usd": projected_cost,
                            "passed_validation": False,
                        }
                    )
                    batches.append(
                        {
                            "batch": batch_number,
                            "bidder_ids": [
                                archetype["bidder_id"]
                                for archetype in requested
                            ],
                            "prompt": str(prompt_path),
                            "raw_response": attempts[-2]["raw_response"],
                            "archived_raw_response": (
                                str(archived_raw_path)
                                if archived_raw_path is not None
                                else None
                            ),
                            "response_source": (
                                attempts[-2]["response_source"]
                            ),
                            "attempts": attempts,
                            "status": "api_error",
                        }
                    )
                    write_manifest("api_error", failure=repr(repair_exc))
                    raise
                repair_raw_path.write_text(raw_text, encoding="utf-8")
                call_cost = usage["estimated_cost_usd"]
                if call_cost is None and token_prices is not None:
                    call_cost = estimated_token_cost(
                        max(
                            1,
                            (
                                len(repair_prompt.encode("utf-8")) + 2
                            ) // 3,
                        ),
                        max(
                            1,
                            (len(raw_text.encode("utf-8")) + 2) // 3,
                        ),
                        token_prices,
                    )
                    usage["estimated_cost_source"] = "utf8_byte_estimate"
                else:
                    usage["estimated_cost_source"] = "provider_usage"
                cumulative_cost_usd += call_cost or 0.0
                usage["estimated_cost_usd"] = call_cost
                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "kind": "repair",
                        "repair_number": repair_number + 1,
                        "response_source": "generated",
                        "prompt": str(repair_prompt_path),
                        "raw_response": str(repair_raw_path),
                        "projected_max_cost_usd": projected_cost,
                        **usage,
                    }
                )
        if parsed is None:
            batch_record = {
                "batch": batch_number,
                "bidder_ids": [
                    archetype["bidder_id"] for archetype in requested
                ],
                "prompt": str(prompt_path),
                "raw_response": attempts[-1]["raw_response"],
                "archived_raw_response": (
                    str(archived_raw_path)
                    if archived_raw_path is not None
                    else None
                ),
                "response_source": attempts[-1]["response_source"],
                "attempts": attempts,
                "status": "validation_failed",
            }
            batches.append(batch_record)
            write_manifest("batch_validation_failed", failure=str(last_error))
            raise SystemExit(
                f"Error parsing batch {batch_number} after "
                f"{len(attempts)} attempt(s); raw responses and manifest "
                f"retained: {last_error}"
            ) from last_error

        # Canonical response always points at the accepted batch, so a later
        # --resume never gets trapped repeatedly parsing an obsolete failure.
        accepted_response_path = Path(attempts[-1]["raw_response"])
        if accepted_response_path != raw_path:
            shutil.copy2(accepted_response_path, raw_path)
        profiles.extend(parsed)
        batches.append(
            {
                "batch": batch_number,
                "bidder_ids": [profile.bidder_id for profile in parsed],
                "prompt": str(prompt_path),
                "raw_response": str(raw_path),
                "archived_raw_response": (
                    str(archived_raw_path)
                    if archived_raw_path is not None
                    else None
                ),
                "response_source": attempts[-1]["response_source"],
                "attempts": attempts,
                "accepted_attempt": attempts[-1]["attempt"],
                "status": "validated",
            }
        )
        write_manifest("generating")
        print(
            f"Validated batch {batch_number}: "
            f"{', '.join(profile.bidder_id for profile in parsed)} "
            f"(attempt {attempts[-1]['attempt']}, "
            f"cumulative estimated cost ${cumulative_cost_usd:.4f})"
        )

    if args.dry_run:
        print(
            f"Dry run complete: {len(archetypes)} profiles in "
            f"{(len(archetypes) + args.profiles_per_call - 1) // args.profiles_per_call} batches"
        )
        return 0

    spec = ScenarioProfileSpec(
        schema_version="pc_build_profile_spec_v1",
        domain="pc_build",
        description=design["description"],
        goods=[GoodSpec.model_validate(good) for good in goods],
        bidder_profiles=profiles,
        generation={
            "source": "batched_llm_population_generation",
            "design": str(args.design),
            "provider": args.provider,
            "model": args.model,
            "temperature": effective_generation_temperature(
                args.provider, args.model, args.temperature
            ),
            "requested_temperature": args.temperature,
            "reasoning_effort": effective_generation_reasoning_effort(
                args.provider, args.reasoning_effort
            ),
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_batch_repair_retries": args.max_batch_repair_retries,
            "profiles_per_call": args.profiles_per_call,
            "population_constraints": design["population_constraints"],
            "sample_constraints": design["sample_constraints"],
        },
    )
    strata = {
        bidder["bidder_id"]: bidder["stratum"] for bidder in archetypes
    }
    coverage = population_coverage_report(
        spec,
        bidder_strata=strata,
        constraints=design["population_constraints"],
    )
    sample_validation = None
    if coverage["passed"]:
        sample_validation = validate_nested_scalability_samples(
            spec,
            seeds=args.validation_seeds,
            sizes=list(range(4, 11)),
            fixed_size=8,
            constraints=design["sample_constraints"],
            include_economic=not args.skip_economic_sample_validation,
        )
    candidate_path = args.output.with_name(
        f"{args.output.stem}.candidate{args.output.suffix}"
    )
    report_path = args.output.with_name(
        f"{args.output.stem}.validation.json"
    )
    report_path.write_text(
        json.dumps(
            {
                "population_coverage": coverage,
                "sample_validation": sample_validation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not coverage["passed"] or not (
        sample_validation and sample_validation["passed"]
    ):
        write_scenario_profile_spec(spec, candidate_path)
        write_manifest(
            "population_validation_failed",
            population_coverage=coverage,
            sample_validation=sample_validation,
        )
        print(f"Candidate failed population/sample validation: {candidate_path}")
        print(f"Validation report: {report_path}")
        for violation in coverage["violations"]:
            print(f"  - population: {violation}")
        if sample_validation:
            for violation in sample_validation["violations"]:
                print(f"  - sample: {violation}")
        return 1

    assert spec.generation is not None
    spec.generation["coverage_orders"] = freeze_validated_nested_orders(
        spec,
        sample_validation,
    )
    spec.generation["coverage_order_validation"] = {
        "seeds": sample_validation["seeds"],
        "sizes": sample_validation["sizes"],
        "fixed_size": sample_validation["fixed_size"],
        "include_economic": sample_validation["include_economic"],
    }
    write_scenario_profile_spec(spec, args.output)
    write_manifest(
        "complete",
        population_coverage=coverage,
        sample_validation=sample_validation,
    )
    print(f"Wrote validated 16x16 population: {args.output}")
    print(f"Wrote validation report: {report_path}")
    print(f"Wrote manifest: {args.manifest}")
    print(f"Estimated API cost this run: ${cumulative_cost_usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
