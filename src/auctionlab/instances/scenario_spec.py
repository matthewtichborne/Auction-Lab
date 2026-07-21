"""Frozen scenario-specification schema for PC-build auction profiles.

A :class:`ScenarioProfileSpec` is the latent, human- or LLM-authored
description of a bidder-preference universe (goods, archetypes, base
values, substitute/complement structure). It is the single frozen
artifact that both:

1. the full valuation table generator
   (:mod:`auctionlab.instances.structured_spec`), and
2. the natural-language person-seed renderer

derive from deterministically. Nothing about experiment execution ever
mutates a spec file once it is written.

Schema version ``pc_build_profile_spec_v1`` is deliberately conservative:
every item reference (base values, substitute/complement group items,
core/secondary/low-interest lists) must resolve to a declared good, and
every numeric field is bounds-checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "pc_build_profile_spec_v1"
DOMAIN = "pc_build"


# ---------------------------------------------------------------------------
# Leaf specs
# ---------------------------------------------------------------------------

class GoodSpec(BaseModel):
    id: str = Field(min_length=1)
    description: str = ""
    category: str | None = None
    compatibility_notes: str | None = None


class SubstituteGroupSpec(BaseModel):
    items: list[str] = Field(min_length=1)
    backup_factor: float = Field(ge=0.0, le=1.0)
    description: str = ""


class ComplementGroupSpec(BaseModel):
    items: list[str] = Field(min_length=1)
    bonus: float = Field(ge=0.0)
    description: str = ""


# ---------------------------------------------------------------------------
# Bidder profile spec
# ---------------------------------------------------------------------------

class BidderProfileSpec(BaseModel):
    bidder_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    budget_range: tuple[float, float]
    base_values: dict[str, float]
    substitute_groups: list[SubstituteGroupSpec] = Field(default_factory=list)
    complement_groups: list[ComplementGroupSpec] = Field(default_factory=list)
    budget_cap: float | None = None
    saturation_start: int | None = None
    saturation_penalty: float = Field(default=0.0, ge=0.0)
    notes: str = ""
    core_items: list[str] = Field(default_factory=list)
    secondary_items: list[str] = Field(default_factory=list)
    low_interest_items: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_bidder_invariants(self) -> "BidderProfileSpec":
        errors: list[str] = []

        lo, hi = self.budget_range
        if lo > hi:
            errors.append(
                f"bidder {self.bidder_id!r}: budget_range low ({lo}) > high ({hi})"
            )

        negative = [item for item, v in self.base_values.items() if v < 0]
        if negative:
            errors.append(
                f"bidder {self.bidder_id!r}: negative base_values for {sorted(negative)}"
            )

        if not any(v > 0 for v in self.base_values.values()):
            errors.append(
                f"bidder {self.bidder_id!r}: must have at least one positive base value"
            )

        if errors:
            raise ValueError("; ".join(errors))
        return self


# ---------------------------------------------------------------------------
# Top-level scenario profile spec
# ---------------------------------------------------------------------------

class ScenarioProfileSpec(BaseModel):
    schema_version: str
    domain: str
    description: str = ""
    goods: list[GoodSpec]
    bidder_profiles: list[BidderProfileSpec]
    generation: dict[str, Any] | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check_scenario_invariants(self) -> "ScenarioProfileSpec":
        errors: list[str] = []

        if self.schema_version != SCHEMA_VERSION:
            errors.append(
                f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        if self.domain != DOMAIN:
            errors.append(f"domain must be {DOMAIN!r}, got {self.domain!r}")

        good_ids = [g.id for g in self.goods]
        good_id_set = set(good_ids)
        if len(good_ids) != len(good_id_set):
            dupes = sorted({g for g in good_ids if good_ids.count(g) > 1})
            errors.append(f"duplicate good ids: {dupes}")

        bidder_ids = [b.bidder_id for b in self.bidder_profiles]
        if len(bidder_ids) != len(set(bidder_ids)):
            dupes = sorted({b for b in bidder_ids if bidder_ids.count(b) > 1})
            errors.append(f"duplicate bidder ids: {dupes}")

        for bp in self.bidder_profiles:
            unknown_base = sorted(set(bp.base_values) - good_id_set)
            if unknown_base:
                errors.append(
                    f"bidder {bp.bidder_id!r}: base_values reference unknown goods {unknown_base}"
                )

            for sg in bp.substitute_groups:
                unknown = sorted(set(sg.items) - good_id_set)
                if unknown:
                    errors.append(
                        f"bidder {bp.bidder_id!r}: substitute group references unknown goods {unknown}"
                    )

            for cg in bp.complement_groups:
                unknown = sorted(set(cg.items) - good_id_set)
                if unknown:
                    errors.append(
                        f"bidder {bp.bidder_id!r}: complement group references unknown goods {unknown}"
                    )

            for label, item_list in (
                ("core_items", bp.core_items),
                ("secondary_items", bp.secondary_items),
                ("low_interest_items", bp.low_interest_items),
            ):
                unknown = sorted(set(item_list) - good_id_set)
                if unknown:
                    errors.append(
                        f"bidder {bp.bidder_id!r}: {label} reference unknown goods {unknown}"
                    )

        if errors:
            raise ValueError("; ".join(errors))
        return self


# ---------------------------------------------------------------------------
# Validation result (non-raising)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpecValidationResult:
    """Outcome of validating a raw dict against the scenario-spec schema."""

    valid: bool
    errors: list[str]
    spec: ScenarioProfileSpec | None


def validate_spec_dict(data: dict) -> SpecValidationResult:
    """Validate a raw JSON-decoded dict against the schema, without raising."""
    try:
        spec = ScenarioProfileSpec.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError
        errors: list[str] = []
        for err in getattr(exc, "errors", lambda: [])():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            msg = err.get("msg", str(err))
            errors.append(f"{loc}: {msg}" if loc else msg)
        if not errors:
            errors = [str(exc)]
        return SpecValidationResult(valid=False, errors=errors, spec=None)
    return SpecValidationResult(valid=True, errors=[], spec=spec)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def scenario_profile_spec_to_dict(spec: ScenarioProfileSpec) -> dict:
    """Convert a spec to a plain JSON-serializable dict with stable key order."""
    return spec.model_dump(mode="json")


def scenario_profile_spec_from_dict(data: dict) -> ScenarioProfileSpec:
    """Parse and validate a raw dict into a :class:`ScenarioProfileSpec`."""
    return ScenarioProfileSpec.model_validate(data)


def load_scenario_profile_spec(path: str | Path) -> ScenarioProfileSpec:
    """Load and validate a frozen scenario-spec JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return scenario_profile_spec_from_dict(data)


def write_scenario_profile_spec(spec: ScenarioProfileSpec, path: str | Path) -> None:
    """Write a spec to JSON with stable key order and 2-space indentation."""
    data = scenario_profile_spec_to_dict(spec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
