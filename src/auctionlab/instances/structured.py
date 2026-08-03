"""Structured preference and valuation primitives.

Provides a modular pipeline for generating auction scenarios from latent
bidder preference profiles:

1. A hidden full valuation table (over all non-empty bundles) for evaluation
   and deterministic value/demand-query answers.
2. A brief qualitative disclosure for the simulated person's opening answer.

The generated population is loaded through
``auctionlab.instances.structured_spec``. This module deliberately contains
no hard-coded bidder population or alternative experimental environment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal, Optional, Sequence

# ---------------------------------------------------------------------------
# PC goods catalogue
# ---------------------------------------------------------------------------

PC_GOOD_CATALOG: list[str] = [
    "CPU_HIGH",
    "GPU_HIGH",
    "MOTHERBOARD",
    "RAM_32GB",
    "CPU_BUDGET",
    "GPU_MID",
    "SSD_2TB",
    "PSU",
    "RAM_64GB",
    "SSD_1TB",
]

PC_ITEM_DESCRIPTIONS: dict[str, str] = {
    "CPU_HIGH": (
        "High-end desktop processor (e.g., Intel Core i9 or AMD Ryzen 9). "
        "Offers maximum single-threaded and multi-threaded performance. "
        "Compatible with the MOTHERBOARD. Required for high-end gaming, "
        "video editing, and compute-intensive workloads."
    ),
    "GPU_HIGH": (
        "High-end discrete graphics card (e.g., NVIDIA RTX 4080 or AMD RX 7900 XTX). "
        "Delivers maximum gaming frame rates and GPU compute throughput for AI/ML workloads. "
        "Requires a compatible PSU (650W+). Compatible with any PCIe motherboard."
    ),
    "MOTHERBOARD": (
        "ATX desktop motherboard. Compatible with both CPU_HIGH and CPU_BUDGET (LGA/AM5 socket). "
        "Supports RAM_32GB and RAM_64GB (DDR5 slots). "
        "Includes PCIe x16 slot for GPU_HIGH or GPU_MID, "
        "and M.2 NVMe slots for SSD_2TB and SSD_1TB."
    ),
    "RAM_32GB": (
        "32 GB DDR5 RAM kit (2x16 GB). Compatible with the MOTHERBOARD. "
        "Sufficient for gaming, general productivity, and light content creation. "
        "Can be used as the sole RAM kit or alongside RAM_64GB if needed."
    ),
    "CPU_BUDGET": (
        "Mid-range desktop processor (e.g., Intel Core i5 or AMD Ryzen 5). "
        "Offers good performance per dollar for gaming and everyday tasks. "
        "Uses the same socket as CPU_HIGH and is compatible with the MOTHERBOARD. "
        "Only one CPU can be installed at a time."
    ),
    "GPU_MID": (
        "Mid-range discrete graphics card (e.g., NVIDIA RTX 4060 or AMD RX 7700 XT). "
        "Handles 1080p and 1440p gaming and light GPU compute tasks. "
        "Lower power draw than GPU_HIGH; compatible PSU (500W+) required. "
        "Only one GPU can be installed at a time."
    ),
    "SSD_2TB": (
        "2 TB M.2 NVMe SSD. Installs in an M.2 slot on the MOTHERBOARD. "
        "Provides fast storage for OS, applications, game libraries, and large media files. "
        "Can be used alongside SSD_1TB for additional capacity."
    ),
    "PSU": (
        "750W 80+ Gold modular power supply unit. Powers all PC components. "
        "Sufficient for GPU_HIGH or GPU_MID, CPU_HIGH or CPU_BUDGET, "
        "and storage drives. Required for any complete system build."
    ),
    "RAM_64GB": (
        "64 GB DDR5 RAM kit (2x32 GB). Compatible with the MOTHERBOARD. "
        "Recommended for professional video editing, AI/ML inference, "
        "and large dataset workloads. Only one RAM kit installed at a time "
        "unless the motherboard supports mixed-capacity configurations."
    ),
    "SSD_1TB": (
        "1 TB M.2 NVMe SSD. Installs in an M.2 slot on the MOTHERBOARD. "
        "Provides fast storage for OS and primary applications. "
        "Can be paired with SSD_2TB for larger total capacity."
    ),
}


# ---------------------------------------------------------------------------
# Preference profile dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubstituteGroup:
    """A set of mutually substitutable items.

    The bidder receives full value for the highest-valued item in the group
    and only ``backup_factor * base_value`` for each additional item.
    A high ``backup_factor`` (≥ 0.7) signals near-independent value
    (e.g., a reseller who can sell each unit separately). A low factor
    signals strong substitution (e.g., a gamer who can only install one GPU).
    """

    items: frozenset[str]
    backup_factor: float
    acquisition_mode: Literal["choose_one", "can_use_multiple"] = "choose_one"
    description: str = ""


@dataclass(frozen=True)
class ComplementGroup:
    """A set of items that together earn a super-additive bonus."""

    items: frozenset[str]
    bonus: float
    description: str = ""


@dataclass
class BidderPreferenceProfile:
    """Latent preference structure for one bidder.

    This is the single source of truth from which both a full valuation
    table and a natural-language person seed are derived.
    """

    bidder_id: str
    role: str
    budget_range: tuple[float, float]
    base_values: dict[str, float]
    substitute_groups: list[SubstituteGroup]
    complement_groups: list[ComplementGroup]
    budget_cap: Optional[float] = None
    saturation_start: Optional[int] = None
    saturation_penalty: float = 0.0
    notes: str = ""
    core_items: frozenset[str] = field(default_factory=frozenset)
    secondary_items: frozenset[str] = field(default_factory=frozenset)
    low_interest_items: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Valuation machinery
# ---------------------------------------------------------------------------

def all_nonempty_bundles(items: Sequence[str]) -> list[frozenset[str]]:
    """Return all 2^|items| − 1 non-empty subsets of items."""
    result: list[frozenset[str]] = []
    for size in range(1, len(items) + 1):
        for combo in combinations(items, size):
            result.append(frozenset(combo))
    return result


def value_bundle(profile: BidderPreferenceProfile, bundle: frozenset[str]) -> float:
    """Compute the valuation of a bundle from a preference profile.

    Formula (applied in order):

    1. Substitute groups — full value for the best item, ``backup_factor``
       times base value for each additional item in the same group.
    2. Non-substitute items — full base value added directly.
    3. Complement bonuses — added when the entire complement group is in the bundle.
    4. Saturation penalty — subtracted for bundles larger than ``saturation_start``.
    5. Budget cap — value is clamped to ``budget_cap`` from above.
    6. Non-negative clamp.
    """
    if not bundle:
        return 0.0

    value = 0.0

    # Items covered by at least one substitute group
    all_sub_items: set[str] = set()
    for sg in profile.substitute_groups:
        all_sub_items |= set(sg.items)

    # Substitute group contributions
    for sg in profile.substitute_groups:
        group_in_bundle = bundle & sg.items
        if not group_in_bundle:
            continue
        best = max(group_in_bundle, key=lambda i: profile.base_values.get(i, 0.0))
        best_val = profile.base_values.get(best, 0.0)
        others_val = sum(
            profile.base_values.get(item, 0.0)
            for item in group_in_bundle
            if item != best
        )
        value += best_val + sg.backup_factor * others_val

    # Non-substitute item contributions
    for item in bundle:
        if item not in all_sub_items:
            value += profile.base_values.get(item, 0.0)

    # Complement bonuses
    for cg in profile.complement_groups:
        if cg.items.issubset(bundle):
            value += cg.bonus

    # Saturation penalty
    if profile.saturation_start is not None and len(bundle) > profile.saturation_start:
        excess = len(bundle) - profile.saturation_start
        value -= profile.saturation_penalty * (excess ** 2)

    # Budget cap
    if profile.budget_cap is not None:
        value = min(value, profile.budget_cap)

    return max(0.0, value)


def generate_valuation_table(
    items: Sequence[str],
    profile: BidderPreferenceProfile,
) -> dict[frozenset[str], float]:
    """Generate and monotonicity-repair a full valuation table."""
    bundles = all_nonempty_bundles(items)
    table: dict[frozenset[str], float] = {b: value_bundle(profile, b) for b in bundles}
    return enforce_monotonicity(items, table)


def generate_full_valuations(
    items: Sequence[str],
    profiles: Sequence[BidderPreferenceProfile],
) -> dict[str, dict[frozenset[str], float]]:
    """Generate full valuation tables for all bidders."""
    return {p.bidder_id: generate_valuation_table(items, p) for p in profiles}


def enforce_monotonicity(
    items: Sequence[str],
    table: dict[frozenset[str], float],
) -> dict[frozenset[str], float]:
    """Repair free-disposal monotonicity: if A ⊆ B then v(A) ≤ v(B).

    Processes bundles in ascending size order. Each bundle is set to at
    least the maximum value of its direct sub-bundles (size − 1 subsets).
    By induction this guarantees the constraint for all subset pairs.
    """
    result = dict(table)
    for size in range(2, len(items) + 1):
        for bundle in (b for b in table if len(b) == size):
            for item in bundle:
                sub = bundle - {item}
                if sub in result and result[bundle] < result[sub]:
                    result[bundle] = result[sub]
    return result


# ---------------------------------------------------------------------------
# Brief qualitative disclosure rendering
# ---------------------------------------------------------------------------

def _join_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


_ITEM_DISCLOSURE_LABELS: dict[str, str] = {
    # Current 16-good population.
    "CPU_HI": "high-performance CPU",
    "CPU_MID": "mid-range CPU",
    "CPU_LO": "entry-level CPU",
    "GPU_AI": "professional AI accelerator",
    "GPU_GAM_HI": "high-end gaming graphics card",
    "GPU_VALUE": "value-oriented graphics card",
    "RAM_128": "128GB memory",
    "RAM_64": "64GB memory",
    "RAM_32": "32GB memory",
    "MB_PRO": "professional motherboard",
    "MB_STD": "mainstream motherboard",
    "SSD_4TB": "4TB solid-state storage",
    "SSD_2TB": "2TB solid-state storage",
    "PSU_1000": "1000W power supply",
    "COOL_AIO": "all-in-one liquid cooler",
    "CASE_ATX": "ATX case",
    # Legacy hard-coded catalogue.
    "CPU_HIGH": "high-performance CPU",
    "CPU_BUDGET": "mid-range CPU",
    "GPU_HIGH": "high-end graphics card",
    "GPU_MID": "mid-range graphics card",
    "MOTHERBOARD": "motherboard",
    "RAM_32GB": "32GB memory",
    "RAM_64GB": "64GB memory",
    "SSD_1TB": "1TB solid-state storage",
    "PSU": "power supply",
}


def _disclosure_label(item: str) -> str:
    return _ITEM_DISCLOSURE_LABELS.get(item, item.replace("_", " ").lower())


def render_brief_qualitative_person_seed(
    profile: BidderPreferenceProfile,
    *,
    identity_text: str | None = None,
    available_goods: Sequence[str] | None = None,
) -> str:
    """Render the only person-facing seed used by structured scenarios.

    The disclosure deliberately contains no singleton values, numerical
    substitute factors, complement bonuses, saturation equations, or full
    valuation recipe. It exposes a short, profile-aligned qualitative
    description plus exactly one monetary figure: the maximum total
    willingness to pay over the selected auction's goods.

    Exact bundle values remain private in the scenario valuation table and
    are answered through deterministic value/demand-query lookups.
    """
    identity = (identity_text or profile.role).strip()
    if identity and identity[-1] not in ".!?":
        identity += "."

    if available_goods is None:
        referenced = (
            set(profile.base_values)
            | set(profile.core_items)
            | set(profile.secondary_items)
            | set(profile.low_interest_items)
        )
        available_ids = sorted(referenced)
    else:
        available_ids = list(available_goods)
    available_set = set(available_ids)
    positive = {
        item
        for item in available_ids
        if profile.base_values.get(item, 0.0) > 0
    }

    parts = [identity] if identity else []

    core = sorted(profile.core_items & available_set & positive)
    secondary = sorted(profile.secondary_items & available_set & positive)
    low = sorted(profile.low_interest_items & available_set & positive)
    if core:
        parts.append(
            "They are mainly interested in "
            + _join_list([_disclosure_label(item) for item in core])
            + "."
        )
    if secondary:
        parts.append(
            "They would also consider "
            + _join_list([_disclosure_label(item) for item in secondary])
            + " as lower-priority options."
        )
    if low:
        parts.append(
            _join_list([_disclosure_label(item) for item in low]).capitalize()
            + (" are less important to them." if len(low) > 1 else " is less important to them.")
        )
    classified = set(core) | set(secondary) | set(low)
    other_positive = sorted(positive - classified)
    if other_positive:
        parts.append(
            "They may also be interested in "
            + _join_list(
                [_disclosure_label(item) for item in other_positive]
            )
            + "."
        )

    for sg in profile.substitute_groups:
        selected = [item for item in available_ids if item in sg.items & positive]
        if len(selected) < 2:
            continue
        best = max(selected, key=lambda item: profile.base_values.get(item, 0.0))
        alternatives = [item for item in selected if item != best]
        labels = _join_list(
            [_disclosure_label(item) for item in alternatives]
        )
        if sg.acquisition_mode == "choose_one":
            parts.append(
                f"They are choosing at most one from this set: they prefer "
                f"{_disclosure_label(best)}, with {labels} as fallbacks. "
                "Obtaining more than one would provide no meaningful "
                "additional benefit."
            )
        else:
            parts.append(
                f"They prefer {_disclosure_label(best)}, while {labels} are "
                "related alternatives. They can use, deploy, or resell more "
                "than one, so additional items retain positive value."
            )

    for cg in profile.complement_groups:
        selected = [item for item in available_ids if item in cg.items]
        if len(selected) < 2:
            continue
        parts.append(
            "They especially value obtaining "
            + _join_list([_disclosure_label(item) for item in selected])
            + " together."
        )

    excluded = sorted(available_set - positive)
    if excluded:
        parts.append(
            "They are not interested in "
            + _join_list([_disclosure_label(item) for item in excluded])
            + "."
        )

    selected_table = generate_valuation_table(available_ids, profile)
    selected_ceiling = max(selected_table.values(), default=0.0)
    parts.append(
        "Their maximum total willingness to pay in this auction is "
        f"approximately ${selected_ceiling:,.0f}."
    )

    return " ".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# Jitter and conditional group helpers
# ---------------------------------------------------------------------------

def _jitter(value: float, rng: random.Random) -> float:
    """Apply ±5% multiplicative jitter, rounded to nearest $10."""
    return max(10.0, round(value * rng.uniform(0.95, 1.05) / 10) * 10)


def _sub_if_available(
    items: set[str],
    group_items: frozenset[str],
    backup_factor: float,
    description: str = "",
    acquisition_mode: Literal[
        "choose_one", "can_use_multiple"
    ] = "choose_one",
) -> list[SubstituteGroup]:
    available = group_items & items
    if len(available) >= 2:
        return [SubstituteGroup(
            items=available,
            backup_factor=backup_factor,
            acquisition_mode=acquisition_mode,
            description=description,
        )]
    return []


def _comp_if_available(
    items: set[str],
    group_items: frozenset[str],
    bonus: float,
    description: str = "",
) -> list[ComplementGroup]:
    if group_items.issubset(items):
        return [ComplementGroup(items=group_items, bonus=bonus, description=description)]
    return []
