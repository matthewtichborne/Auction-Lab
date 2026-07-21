#!/usr/bin/env python3
"""Export the current hard-coded PC-build archetype profiles as a frozen
``ScenarioProfileSpec`` JSON file.

This produces ``scenarios/pc_build_v1/pc_build_profiles_v0_manual.json``: a
manual baseline spec used to validate the scenario-spec pipeline before any
LLM-generated spec exists. It is built directly from the archetype builder
functions in :mod:`auctionlab.instances.structured` using a fixed,
documented export seed (default 0) for jitter reproducibility — it is not
LLM-generated.

Usage::

    ./venv/bin/python scripts/export_current_pc_build_profiles.py
    ./venv/bin/python scripts/export_current_pc_build_profiles.py --output path/to/out.json --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from auctionlab.instances.scenario_spec import (
    BidderProfileSpec,
    ComplementGroupSpec,
    GoodSpec,
    ScenarioProfileSpec,
    SubstituteGroupSpec,
    write_scenario_profile_spec,
)
from auctionlab.instances.structured import (
    PC_GOOD_CATALOG,
    PC_ITEM_DESCRIPTIONS,
    _ARCHETYPE_BUILDERS,
    _ARCHETYPE_ORDER,
)

EXPORT_SEED = 0
DEFAULT_OUTPUT = Path("scenarios/pc_build_v1/pc_build_profiles_v0_manual.json")


def build_manual_spec(seed: int = EXPORT_SEED) -> ScenarioProfileSpec:
    """Build the v0 manual ``ScenarioProfileSpec`` from the hard-coded builders.

    Uses the full 10-good catalog and all 10 archetypes so downstream
    consumers can select any ``num_goods``/``num_bidders`` subset from it.
    """
    items = list(PC_GOOD_CATALOG)
    items_set = set(items)
    rng = random.Random(seed)

    goods = [
        GoodSpec(id=item, description=PC_ITEM_DESCRIPTIONS[item])
        for item in items
    ]

    bidder_profiles: list[BidderProfileSpec] = []
    for bidder_id in _ARCHETYPE_ORDER:
        profile = _ARCHETYPE_BUILDERS[bidder_id](items_set, rng)  # type: ignore[operator]
        bidder_profiles.append(
            BidderProfileSpec(
                bidder_id=profile.bidder_id,
                role=profile.role,
                budget_range=profile.budget_range,
                base_values=dict(profile.base_values),
                substitute_groups=[
                    SubstituteGroupSpec(
                        items=sorted(sg.items),
                        backup_factor=sg.backup_factor,
                        description=sg.description,
                    )
                    for sg in profile.substitute_groups
                ],
                complement_groups=[
                    ComplementGroupSpec(
                        items=sorted(cg.items),
                        bonus=cg.bonus,
                        description=cg.description,
                    )
                    for cg in profile.complement_groups
                ],
                budget_cap=profile.budget_cap,
                saturation_start=profile.saturation_start,
                saturation_penalty=profile.saturation_penalty,
                notes=profile.notes,
                core_items=sorted(profile.core_items),
                secondary_items=sorted(profile.secondary_items),
                low_interest_items=sorted(profile.low_interest_items),
            )
        )

    return ScenarioProfileSpec(
        schema_version="pc_build_profile_spec_v1",
        domain="pc_build",
        description=(
            "Manual baseline PC-build profile universe exported from the "
            "hard-coded archetype builders in auctionlab.instances.structured "
            f"(export_seed={seed}). Not LLM-generated; used to validate the "
            "frozen scenario-spec pipeline before any generated spec exists."
        ),
        goods=goods,
        bidder_profiles=bidder_profiles,
        generation={
            "source": "scripts/export_current_pc_build_profiles.py",
            "export_seed": seed,
            "method": "hard_coded_archetype_builders",
        },
        notes=(
            "This is the v0 manual spec, not the target format for future "
            "LLM-generated profiles. It exists to validate the schema, loader, "
            "spec-based factory, and validation report against known-good data."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=EXPORT_SEED)
    args = parser.parse_args()

    spec = build_manual_spec(seed=args.seed)
    write_scenario_profile_spec(spec, args.output)
    print(
        f"Wrote {len(spec.goods)} goods and {len(spec.bidder_profiles)} bidder "
        f"profiles to {args.output}"
    )


if __name__ == "__main__":
    main()
