"""Generated, out-of-domain environments for provisional-value calibration.

The main PC-build population must not be used to choose provisional-value
calibration parameters.  This module defines three deliberately small
six-good/three-bidder domains, validates an LLM-generated hidden preference
specification, and turns it into the same structured valuation machinery used
by the main experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from auctionlab.experiments.pv_calibration import DOMAIN_CATALOGS
from auctionlab.instances.base import AuctionInstance
from auctionlab.instances.nl_types import NaturalLanguageAuctionScenario
from auctionlab.instances.structured import (
    BidderPreferenceProfile,
    ComplementGroup,
    SubstituteGroup,
    generate_full_valuations,
    render_brief_qualitative_person_seed,
)


PV_CALIBRATION_ENVIRONMENT_FORMAT = "auctionlab.pv_calibration_environment"
PV_CALIBRATION_ENVIRONMENT_VERSION = 1

# The final calibration design uses three independently generated instances in
# each domain.  Instance zero is the original environment; instances one and
# two extend the sample without invalidating the already-frozen artefact.
GENERATED_CALIBRATION_DOMAINS: tuple[str, ...] = (
    "camera_video_kit",
    "travel_package",
    "kitchen_appliance_bundle",
)


def environment_design(domain: str) -> dict[str, Any]:
    """Return the fixed public catalogue and bidder roles for ``domain``."""
    if domain not in GENERATED_CALIBRATION_DOMAINS:
        raise ValueError(
            f"unknown generated calibration domain {domain!r}; expected one of "
            f"{list(GENERATED_CALIBRATION_DOMAINS)}"
        )
    catalog = DOMAIN_CATALOGS[domain]
    return {
        "domain": domain,
        "scenario_description": catalog["scenario_description"],
        "goods": dict(catalog["goods"]),
        "bidder_roles": [
            {
                "bidder_id": bidder["bidder_id"],
                "role": bidder["role"],
            }
            for bidder in catalog["bidders"][:3]
        ],
    }


def build_environment_prompt(
    domain: str,
    *,
    instance_index: int = 0,
    repair_error: str | None = None,
) -> str:
    """Build the single-call hidden-environment generation prompt."""
    design = environment_design(domain)
    repair = (
        ""
        if repair_error is None
        else (
            "\nYour previous answer failed validation:\n"
            f"{repair_error}\nReturn a corrected complete object.\n"
        )
    )
    if instance_index < 0:
        raise ValueError("instance_index must be non-negative")
    return f"""Create one small synthetic combinatorial-auction environment.
It will tune a provisional-value estimator outside the PC-component domain.
This is independent environment instance {instance_index}. Generate fresh
identities and hidden values rather than reproducing another instance.

The public design is fixed:
{json.dumps(design, indent=2)}

Generate hidden deterministic preferences for exactly the listed three
bidders and all six goods. Use monetary values on a broadly comparable scale:
positive singleton base values should normally be 50--900 and each bidder's
maximum bundle value should normally be 500--2500.

Return JSON only:
{{
  "domain": "{domain}",
  "instance_index": {instance_index},
  "bidders": [
    {{
      "bidder_id": "exact listed id",
      "identity_text": "one natural sentence expanding the listed role",
      "budget_range": [positive lower number, larger upper number],
      "budget_cap": positive number,
      "base_values": {{"EVERY_GOOD_ID": nonnegative number}},
      "substitute_groups": [
        {{
          "items": ["GOOD_A", "GOOD_B"],
          "backup_factor": number from 0 to 1,
          "acquisition_mode": "choose_one or can_use_multiple",
          "description": "brief reason"
        }}
      ],
      "complement_groups": [
        {{
          "items": ["GOOD_A", "GOOD_B"],
          "bonus": nonnegative number,
          "description": "brief reason"
        }}
      ],
      "saturation_start": null_or_integer,
      "saturation_penalty": nonnegative number,
      "core_items": ["positive-valued goods"],
      "secondary_items": ["positive-valued goods"],
      "low_interest_items": ["positive-valued goods"]
    }}
  ]
}}

Hard constraints:
- base_values must contain every good exactly once.
- Each bidder must value 3--5 goods positively, leaving at least one genuine
  zero-valued exclusion.
- The three priority lists must be disjoint and together cover every
  positive-valued good exactly once; zero-valued goods appear in none.
- A group may contain only positive-valued goods and no duplicate goods.
- Every scenario must contain at least one choose_one substitute group, one
  can_use_multiple substitute group, and one complement group across its
  bidders. Each group has at least two goods.
- choose_one backup_factor is at most 0.35; can_use_multiple backup_factor is
  at least 0.65.
- budget_range lower < upper, and budget_cap lies within that range.
- Preferences should create genuine competition and a non-trivial allocation;
  avoid giving one bidder overwhelmingly larger values than everyone else.
- Do not copy hidden numbers from any pre-existing example.
{repair}"""


def validate_environment_payload(
    payload: Mapping[str, Any],
    *,
    expected_domain: str,
    expected_instance_index: int | None = None,
) -> dict[str, Any]:
    """Validate and normalise one generated environment payload."""
    design = environment_design(expected_domain)
    goods = set(design["goods"])
    expected_ids = [row["bidder_id"] for row in design["bidder_roles"]]
    if payload.get("domain") != expected_domain:
        raise ValueError(
            f"domain must be {expected_domain!r}, got {payload.get('domain')!r}"
        )
    raw_instance_index = payload.get("instance_index", 0)
    if isinstance(raw_instance_index, bool):
        raise ValueError("instance_index must be a non-negative integer")
    try:
        instance_index = int(raw_instance_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "instance_index must be a non-negative integer"
        ) from exc
    if instance_index < 0 or instance_index != raw_instance_index:
        raise ValueError("instance_index must be a non-negative integer")
    if (
        expected_instance_index is not None
        and instance_index != expected_instance_index
    ):
        raise ValueError(
            f"instance_index must be {expected_instance_index}, got "
            f"{instance_index}"
        )
    bidders = payload.get("bidders")
    if not isinstance(bidders, list) or len(bidders) != 3:
        raise ValueError("bidders must contain exactly three objects")
    if [row.get("bidder_id") for row in bidders] != expected_ids:
        raise ValueError(f"bidder ids/order must be exactly {expected_ids}")

    saw_choose_one = False
    saw_multiple = False
    saw_complement = False
    normalised: list[dict[str, Any]] = []
    for row in bidders:
        bidder_id = row["bidder_id"]
        identity = str(row.get("identity_text", "")).strip()
        if len(identity.split()) < 5:
            raise ValueError(f"{bidder_id}: identity_text is too short")

        raw_values = row.get("base_values")
        if not isinstance(raw_values, dict) or set(raw_values) != goods:
            missing = sorted(goods - set(raw_values or {}))
            unknown = sorted(set(raw_values or {}) - goods)
            raise ValueError(
                f"{bidder_id}: base_values must cover every good; "
                f"missing={missing}, unknown={unknown}"
            )
        values = {item: float(raw_values[item]) for item in design["goods"]}
        if any(value < 0 for value in values.values()):
            raise ValueError(f"{bidder_id}: base values must be nonnegative")
        positive = {item for item, value in values.items() if value > 0}
        if not 3 <= len(positive) <= 5:
            raise ValueError(
                f"{bidder_id}: must have 3--5 positive goods, got {len(positive)}"
            )

        priority_names = ("core_items", "secondary_items", "low_interest_items")
        priority_sets = [set(row.get(name, [])) for name in priority_names]
        if any(not group <= goods for group in priority_sets):
            raise ValueError(f"{bidder_id}: priority list contains unknown goods")
        if any(
            priority_sets[i] & priority_sets[j]
            for i in range(3)
            for j in range(i + 1, 3)
        ):
            raise ValueError(f"{bidder_id}: priority lists must be disjoint")
        if set().union(*priority_sets) != positive:
            raise ValueError(
                f"{bidder_id}: priority lists must cover positive goods exactly"
            )

        substitute_groups: list[dict[str, Any]] = []
        used_substitute_items: set[str] = set()
        for group in row.get("substitute_groups", []):
            items = list(group.get("items", []))
            item_set = set(items)
            if len(items) < 2 or len(item_set) != len(items):
                raise ValueError(
                    f"{bidder_id}: substitute groups need distinct 2+ goods"
                )
            if not item_set <= positive:
                raise ValueError(
                    f"{bidder_id}: substitute groups require positive known goods"
                )
            if used_substitute_items & item_set:
                raise ValueError(
                    f"{bidder_id}: substitute groups may not overlap"
                )
            used_substitute_items |= item_set
            mode = group.get("acquisition_mode")
            backup = float(group.get("backup_factor"))
            if mode == "choose_one":
                saw_choose_one = True
                if not 0 <= backup <= 0.35:
                    raise ValueError(
                        f"{bidder_id}: choose_one backup_factor must be <= 0.35"
                    )
            elif mode == "can_use_multiple":
                saw_multiple = True
                if not 0.65 <= backup <= 1:
                    raise ValueError(
                        f"{bidder_id}: can_use_multiple backup_factor must be >= 0.65"
                    )
            else:
                raise ValueError(
                    f"{bidder_id}: invalid acquisition_mode {mode!r}"
                )
            substitute_groups.append(
                {
                    "items": sorted(item_set),
                    "backup_factor": backup,
                    "acquisition_mode": mode,
                    "description": str(group.get("description", "")).strip(),
                }
            )

        complement_groups: list[dict[str, Any]] = []
        for group in row.get("complement_groups", []):
            items = list(group.get("items", []))
            item_set = set(items)
            if len(items) < 2 or len(item_set) != len(items):
                raise ValueError(
                    f"{bidder_id}: complement groups need distinct 2+ goods"
                )
            if not item_set <= positive:
                raise ValueError(
                    f"{bidder_id}: complement groups require positive known goods"
                )
            bonus = float(group.get("bonus"))
            if bonus < 0:
                raise ValueError(f"{bidder_id}: complement bonus must be >= 0")
            saw_complement = True
            complement_groups.append(
                {
                    "items": sorted(item_set),
                    "bonus": bonus,
                    "description": str(group.get("description", "")).strip(),
                }
            )

        budget_range = [float(value) for value in row.get("budget_range", [])]
        if len(budget_range) != 2 or not 0 < budget_range[0] < budget_range[1]:
            raise ValueError(f"{bidder_id}: invalid budget_range")
        budget_cap = float(row.get("budget_cap"))
        if not budget_range[0] <= budget_cap <= budget_range[1]:
            raise ValueError(f"{bidder_id}: budget_cap must lie in budget_range")
        saturation_start = row.get("saturation_start")
        if saturation_start is not None:
            saturation_start = int(saturation_start)
            if not 1 <= saturation_start <= len(goods):
                raise ValueError(f"{bidder_id}: invalid saturation_start")
        saturation_penalty = float(row.get("saturation_penalty", 0))
        if saturation_penalty < 0:
            raise ValueError(f"{bidder_id}: saturation_penalty must be >= 0")

        normalised.append(
            {
                "bidder_id": bidder_id,
                "identity_text": identity,
                "budget_range": budget_range,
                "budget_cap": budget_cap,
                "base_values": values,
                "substitute_groups": substitute_groups,
                "complement_groups": complement_groups,
                "saturation_start": saturation_start,
                "saturation_penalty": saturation_penalty,
                **{
                    name: sorted(items)
                    for name, items in zip(priority_names, priority_sets)
                },
            }
        )

    missing_structures = [
        name
        for name, seen in (
            ("choose_one substitute", saw_choose_one),
            ("can_use_multiple substitute", saw_multiple),
            ("complement", saw_complement),
        )
        if not seen
    ]
    if missing_structures:
        raise ValueError(
            "scenario is missing required structures: "
            + ", ".join(missing_structures)
        )
    return {
        "format": PV_CALIBRATION_ENVIRONMENT_FORMAT,
        "version": PV_CALIBRATION_ENVIRONMENT_VERSION,
        "domain": expected_domain,
        "instance_index": instance_index,
        "scenario_description": design["scenario_description"],
        "goods": design["goods"],
        "bidders": normalised,
    }


def load_generated_environment(path: str | Path) -> dict[str, Any]:
    """Load and validate a generated calibration environment."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != PV_CALIBRATION_ENVIRONMENT_FORMAT:
        raise ValueError(f"{source}: unexpected environment format")
    if payload.get("version") != PV_CALIBRATION_ENVIRONMENT_VERSION:
        raise ValueError(f"{source}: unsupported environment version")
    validated = validate_environment_payload(
        payload,
        expected_domain=str(payload.get("domain")),
        expected_instance_index=int(payload.get("instance_index", 0)),
    )
    # Generation provenance is not part of the hidden preference validation.
    if "generation" in payload:
        validated["generation"] = payload["generation"]
    return validated


def build_generated_environment_scenario(
    environment: Mapping[str, Any],
) -> NaturalLanguageAuctionScenario:
    """Materialise full deterministic valuations from an environment payload."""
    validated = validate_environment_payload(
        environment, expected_domain=str(environment["domain"])
    )
    items = list(validated["goods"])
    profiles: list[BidderPreferenceProfile] = []
    for row in validated["bidders"]:
        profiles.append(
            BidderPreferenceProfile(
                bidder_id=row["bidder_id"],
                role=row["identity_text"],
                budget_range=tuple(row["budget_range"]),
                budget_cap=row["budget_cap"],
                base_values=dict(row["base_values"]),
                substitute_groups=[
                    SubstituteGroup(
                        items=frozenset(group["items"]),
                        backup_factor=group["backup_factor"],
                        acquisition_mode=group["acquisition_mode"],
                        description=group["description"],
                    )
                    for group in row["substitute_groups"]
                ],
                complement_groups=[
                    ComplementGroup(
                        items=frozenset(group["items"]),
                        bonus=group["bonus"],
                        description=group["description"],
                    )
                    for group in row["complement_groups"]
                ],
                saturation_start=row["saturation_start"],
                saturation_penalty=row["saturation_penalty"],
                core_items=frozenset(row["core_items"]),
                secondary_items=frozenset(row["secondary_items"]),
                low_interest_items=frozenset(row["low_interest_items"]),
            )
        )
    valuations = generate_full_valuations(items, profiles)
    person_seeds = {
        profile.bidder_id: render_brief_qualitative_person_seed(
            profile,
            identity_text=profile.role,
            available_goods=items,
        )
        for profile in profiles
    }
    profile_metadata = {
        profile.bidder_id: {
            "role": profile.role,
            "budget_range": list(profile.budget_range),
            "disclosed_budget_hint": max(
                valuations[profile.bidder_id].values(), default=0.0
            ),
            "disclosed_positive_items": sorted(
                item for item in items if profile.base_values.get(item, 0) > 0
            ),
            "core_items": sorted(profile.core_items),
            "secondary_items": sorted(profile.secondary_items),
            "low_interest_items": sorted(profile.low_interest_items),
            "substitute_groups": [
                {
                    "items": sorted(group.items),
                    "backup_factor": group.backup_factor,
                    "acquisition_mode": group.acquisition_mode,
                }
                for group in profile.substitute_groups
            ],
            "complement_groups": [
                {"items": sorted(group.items), "bonus": group.bonus}
                for group in profile.complement_groups
            ],
            "person_seed_source": "brief_qualitative_disclosure",
            "person_seed_identity_source": "generated_environment",
        }
        for profile in profiles
    }
    domain = validated["domain"]
    instance_index = validated["instance_index"]
    return NaturalLanguageAuctionScenario(
        name=f"pv_calib_generated_{domain}_instance{instance_index}_6x3",
        seed_type="structured",
        instance=AuctionInstance(
            items=items,
            bidder_ids=[profile.bidder_id for profile in profiles],
            valuations=valuations,
        ),
        scenario_description=validated["scenario_description"],
        item_descriptions=dict(validated["goods"]),
        person_seeds=person_seeds,
        candidate_bundles_by_bidder=None,
        metadata={
            "num_goods": len(items),
            "num_bidders": len(profiles),
            "scenario_seed": instance_index,
            "domain": domain,
            "environment_instance_index": instance_index,
            "benchmark": "pv_calibration_generated",
            "valuation_model": "structured_substitutes_complements",
            "seed_style": "brief_qualitative",
            "profiles": profile_metadata,
        },
    )


def environment_file_name(domain: str, instance_index: int = 0) -> str:
    """Return a stable path name, preserving legacy instance-zero names."""
    if instance_index < 0:
        raise ValueError("instance_index must be non-negative")
    suffix = "" if instance_index == 0 else f"_instance{instance_index}"
    return f"pv_calibration_environment_{domain}{suffix}.json"
