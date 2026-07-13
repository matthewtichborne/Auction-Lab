"""Structured scenario factory for PC-build combinatorial auctions.

Provides a modular pipeline for generating auction scenarios from latent
bidder preference profiles:

1. A hidden full valuation table (over all non-empty bundles) for evaluation.
2. A budget-calibrated natural-language person seed for the LLM proxy.

Usage::

    from auctionlab.instances.structured import make_pc_build_scenario
    scenario = make_pc_build_scenario(num_goods=6, num_bidders=6, seed=0)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence

from auctionlab.instances.base import AuctionInstance
from auctionlab.instances.nl_types import NaturalLanguageAuctionScenario


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
# Seed rendering
# ---------------------------------------------------------------------------

def _price_range(value: float) -> str:
    lo = max(10, round(value * 0.85 / 10) * 10)
    hi = round(value * 1.15 / 10) * 10
    return f"${lo:,.0f}–${hi:,.0f}"


def _join_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def render_budget_calibrated_seed(profile: BidderPreferenceProfile) -> str:
    """Render a natural-language person seed from a BidderPreferenceProfile.

    The seed includes role, budget range, approximate per-item price ranges
    for core items, complement and substitute relationships, and
    secondary/low-interest items.

    Deliberately omits:
    - Exact bundle values or complete valuation tables.
    - List-formatted bundle enumerations (no "Single Item Bundles:" sections).
    - Complement bonus magnitudes.
    """
    parts: list[str] = []

    # Role and budget
    lo, hi = profile.budget_range
    parts.append(
        f"{profile.role} "
        f"Their overall budget for items in this auction is roughly "
        f"${lo:,.0f}–${hi:,.0f}."
    )

    if profile.notes:
        parts.append(profile.notes)

    # Core items with approximate price ranges
    core = sorted(profile.core_items & set(profile.base_values))
    if core:
        item_strs = [
            f"{item} (they would consider spending "
            f"{_price_range(profile.base_values[item])})"
            for item in core
        ]
        parts.append(
            "Their highest-priority items are " + _join_list(item_strs) + "."
        )

    # Complement relationships — prose only, no bonus values
    for cg in profile.complement_groups:
        available = [
            i for i in PC_GOOD_CATALOG
            if i in cg.items and i in profile.base_values
        ]
        if len(available) >= 2:
            desc = cg.description or (
                "owning the complete set is worth meaningfully more than the items separately"
            )
            last = available[-1]
            rest = available[:-1]
            parts.append(
                f"The items {', '.join(rest)} and {last} work together as a "
                f"complementary set — {desc}."
            )

    # Substitute relationships — framing depends on backup_factor
    for sg in profile.substitute_groups:
        available = [
            i for i in PC_GOOD_CATALOG
            if i in sg.items and i in profile.base_values
        ]
        if len(available) < 2:
            continue

        if sg.backup_factor >= 0.7:
            # Near-independent value (reseller pattern): each item still valuable
            desc = sg.description or "each item retains significant independent value"
            parts.append(f"For {_join_list(available)}: {desc}.")
        else:
            # True substitute: one preferred, others are fallbacks
            best = max(available, key=lambda i: profile.base_values[i])
            others = [i for i in available if i != best]
            desc = sg.description or "owning both adds limited extra value"
            parts.append(
                f"Regarding {_join_list(available)}: {best} is the preferred "
                f"option but {_join_list(others)} can serve as a fallback — "
                f"{desc}."
            )

    # Secondary items
    secondary = sorted(profile.secondary_items & set(profile.base_values))
    if secondary:
        sec_strs = [
            f"{item} ({_price_range(profile.base_values[item])})"
            for item in secondary
        ]
        parts.append(
            "They also have secondary interest in " + _join_list(sec_strs) + "."
        )

    # Low-interest items
    low = sorted(profile.low_interest_items & set(profile.base_values))
    if low:
        parts.append(
            "Items like " + _join_list(low) + " have limited value for their use case."
        )

    return "\n\n".join(parts)


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
) -> list[SubstituteGroup]:
    available = group_items & items
    if len(available) >= 2:
        return [SubstituteGroup(
            items=available,
            backup_factor=backup_factor,
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


# ---------------------------------------------------------------------------
# Archetype profile builders
# ---------------------------------------------------------------------------

def make_competitive_gamer_profile(
    items: set[str], rng: random.Random
) -> BidderPreferenceProfile:
    refs: dict[str, float] = {
        "CPU_HIGH": 900, "GPU_HIGH": 950, "MOTHERBOARD": 380, "RAM_32GB": 340,
        "CPU_BUDGET": 260, "GPU_MID": 450, "SSD_2TB": 170, "PSU": 190,
        "RAM_64GB": 130, "SSD_1TB": 140,
    }
    bv = {item: _jitter(refs[item], rng) for item in items if item in refs}

    sub_groups = (
        _sub_if_available(items, frozenset({"CPU_HIGH", "CPU_BUDGET"}), 0.08,
                          "only one CPU can be installed at a time; the budget option is a last resort")
        + _sub_if_available(items, frozenset({"GPU_HIGH", "GPU_MID"}), 0.08,
                            "only one GPU can be installed at a time; the mid-range is a fallback")
    )
    comp_groups = _comp_if_available(
        items, frozenset({"CPU_HIGH", "GPU_HIGH", "MOTHERBOARD", "RAM_32GB"}),
        _jitter(360, rng),
        "a complete high-end gaming platform worth considerably more than its parts",
    )

    return BidderPreferenceProfile(
        bidder_id="competitive_gamer",
        role=(
            "Alex is a competitive PC gamer building a high-end gaming rig for "
            "1440p and 4K gaming at maximum frame rates."
        ),
        budget_range=(1800, 2800),
        base_values=bv,
        substitute_groups=sub_groups,
        complement_groups=comp_groups,
        saturation_start=6,
        saturation_penalty=25.0,
        core_items=frozenset({"CPU_HIGH", "GPU_HIGH", "MOTHERBOARD", "RAM_32GB"}) & items,
        secondary_items=frozenset({"SSD_2TB", "SSD_1TB", "PSU"}) & items,
        low_interest_items=frozenset({"CPU_BUDGET", "GPU_MID", "RAM_64GB"}) & items,
    )


def make_budget_gamer_profile(
    items: set[str], rng: random.Random
) -> BidderPreferenceProfile:
    refs: dict[str, float] = {
        "CPU_BUDGET": 650, "GPU_MID": 590, "MOTHERBOARD": 320, "RAM_32GB": 290,
        "CPU_HIGH": 300, "GPU_HIGH": 320, "SSD_1TB": 150, "PSU": 160,
        "RAM_64GB": 110, "SSD_2TB": 160,
    }
    bv = {item: _jitter(refs[item], rng) for item in items if item in refs}

    sub_groups = (
        _sub_if_available(items, frozenset({"CPU_BUDGET", "CPU_HIGH"}), 0.08,
                          "the budget CPU is the preferred choice; the high-end one is overkill for 1080p")
        + _sub_if_available(items, frozenset({"GPU_MID", "GPU_HIGH"}), 0.08,
                            "the mid-range GPU is the preferred choice for budget gaming builds")
    )
    core_compl = frozenset({"CPU_BUDGET", "GPU_MID", "MOTHERBOARD", "RAM_32GB"})
    comp_groups = _comp_if_available(
        items, core_compl, _jitter(190, rng),
        "a compatible budget gaming platform",
    )
    if "PSU" in items and core_compl.issubset(items):
        comp_groups += _comp_if_available(
            items, core_compl | {"PSU"}, _jitter(80, rng),
            "a complete and immediately runnable budget build",
        )

    return BidderPreferenceProfile(
        bidder_id="budget_gamer",
        role=(
            "Jordan is a budget-conscious gamer who wants the best 1080p performance "
            "for the money and avoids overspending on premium parts."
        ),
        budget_range=(800, 1500),
        base_values=bv,
        substitute_groups=sub_groups,
        complement_groups=comp_groups,
        saturation_start=4,
        saturation_penalty=80.0,
        core_items=frozenset({"CPU_BUDGET", "GPU_MID", "MOTHERBOARD", "RAM_32GB"}) & items,
        secondary_items=frozenset({"PSU", "SSD_1TB"}) & items,
        low_interest_items=frozenset({"RAM_64GB", "SSD_2TB"}) & items,
    )


def make_video_editor_profile(
    items: set[str], rng: random.Random
) -> BidderPreferenceProfile:
    refs: dict[str, float] = {
        "CPU_HIGH": 780, "RAM_64GB": 450, "RAM_32GB": 350, "SSD_2TB": 420,
        "SSD_1TB": 290, "MOTHERBOARD": 360, "PSU": 175, "GPU_HIGH": 280,
        "GPU_MID": 180, "CPU_BUDGET": 190,
    }
    bv = {item: _jitter(refs[item], rng) for item in items if item in refs}

    sub_groups = (
        _sub_if_available(items, frozenset({"RAM_64GB", "RAM_32GB"}), 0.25,
                          "RAM_64GB is preferred for 4K editing; RAM_32GB is a lighter-duty fallback")
        + _sub_if_available(items, frozenset({"SSD_2TB", "SSD_1TB"}), 0.3,
                            "SSD_2TB is preferred for large video libraries; SSD_1TB is a storage fallback")
    )
    ram_item = "RAM_64GB" if "RAM_64GB" in items else "RAM_32GB"
    comp_core = frozenset({"CPU_HIGH", ram_item, "MOTHERBOARD"})
    comp_groups = _comp_if_available(
        items, comp_core, _jitter(250, rng),
        "a high-throughput editing platform where CPU, RAM, and motherboard reinforce each other",
    )

    return BidderPreferenceProfile(
        bidder_id="video_editor",
        role=(
            "Sam is a freelance video editor working on 4K and 8K footage who needs "
            "maximum CPU throughput and ample fast storage."
        ),
        budget_range=(1200, 2200),
        base_values=bv,
        substitute_groups=sub_groups,
        complement_groups=comp_groups,
        core_items=frozenset({"CPU_HIGH", "RAM_64GB", "RAM_32GB", "SSD_2TB", "MOTHERBOARD"}) & items,
        secondary_items=frozenset({"GPU_HIGH", "PSU", "SSD_1TB"}) & items,
        low_interest_items=frozenset({"CPU_BUDGET", "GPU_MID"}) & items,
    )


def make_ai_hobbyist_profile(
    items: set[str], rng: random.Random
) -> BidderPreferenceProfile:
    refs: dict[str, float] = {
        "GPU_HIGH": 1100, "GPU_MID": 680, "PSU": 200, "CPU_HIGH": 340,
        "MOTHERBOARD": 270, "RAM_32GB": 200, "RAM_64GB": 240, "SSD_1TB": 180,
        "SSD_2TB": 220, "CPU_BUDGET": 190,
    }
    bv = {item: _jitter(refs[item], rng) for item in items if item in refs}

    sub_groups = _sub_if_available(
        items, frozenset({"GPU_HIGH", "GPU_MID"}), 0.08,
        "GPU_HIGH is strongly preferred for LLM inference throughput; GPU_MID is a fallback",
    )
    comp_groups = _comp_if_available(
        items, frozenset({"GPU_HIGH", "PSU"}), _jitter(140, rng),
        "the PSU is required to safely run the high-end GPU at full power",
    )

    return BidderPreferenceProfile(
        bidder_id="ai_hobbyist",
        role=(
            "Riley is an AI/ML hobbyist who runs local language models and trains "
            "small neural networks at home and needs maximum GPU compute."
        ),
        budget_range=(1000, 1800),
        base_values=bv,
        substitute_groups=sub_groups,
        complement_groups=comp_groups,
        core_items=frozenset({"GPU_HIGH"}) & items,
        secondary_items=frozenset({"PSU", "GPU_MID", "CPU_HIGH", "MOTHERBOARD",
                                   "RAM_64GB", "RAM_32GB"}) & items,
        low_interest_items=frozenset({"CPU_BUDGET", "SSD_2TB", "SSD_1TB"}) & items,
    )


def make_office_upgrader_profile(
    items: set[str], rng: random.Random
) -> BidderPreferenceProfile:
    refs: dict[str, float] = {
        "SSD_2TB": 350, "SSD_1TB": 280, "RAM_32GB": 220, "RAM_64GB": 240,
        "MOTHERBOARD": 150, "CPU_BUDGET": 110, "PSU": 80, "CPU_HIGH": 70,
        "GPU_MID": 55, "GPU_HIGH": 40,
    }
    bv = {item: _jitter(refs[item], rng) for item in items if item in refs}

    sub_groups = (
        _sub_if_available(items, frozenset({"SSD_2TB", "SSD_1TB"}), 0.20,
                          "two SSDs add storage redundancy but limited extra office productivity")
        + _sub_if_available(items, frozenset({"RAM_64GB", "RAM_32GB"}), 0.20,
                            "32 GB is sufficient for office tasks; 64 GB is a marginal improvement")
    )
    comp_groups = _comp_if_available(
        items, frozenset({"SSD_1TB", "RAM_32GB"}), _jitter(60, rng),
        "upgrading both storage and RAM together transforms the responsiveness of an aging machine",
    )

    return BidderPreferenceProfile(
        bidder_id="office_upgrader",
        role=(
            "Morgan works in a small office and is upgrading an existing desktop PC for "
            "faster document handling and video calls — not gaming or heavy compute."
        ),
        budget_range=(300, 700),
        base_values=bv,
        substitute_groups=sub_groups,
        complement_groups=comp_groups,
        saturation_start=3,
        saturation_penalty=70.0,
        core_items=frozenset({"SSD_1TB", "SSD_2TB", "RAM_32GB"}) & items,
        secondary_items=frozenset({"RAM_64GB", "MOTHERBOARD"}) & items,
        low_interest_items=frozenset({"GPU_HIGH", "GPU_MID", "CPU_HIGH"}) & items,
    )


def make_overclocker_profile(
    items: set[str], rng: random.Random
) -> BidderPreferenceProfile:
    refs: dict[str, float] = {
        "CPU_HIGH": 950, "MOTHERBOARD": 420, "PSU": 280, "RAM_32GB": 330,
        "RAM_64GB": 290, "GPU_HIGH": 370, "GPU_MID": 240, "SSD_2TB": 200,
        "SSD_1TB": 170, "CPU_BUDGET": 175,
    }
    bv = {item: _jitter(refs[item], rng) for item in items if item in refs}

    sub_groups = _sub_if_available(
        items, frozenset({"CPU_HIGH", "CPU_BUDGET"}), 0.08,
        "only one CPU can be installed; the budget CPU has minimal overclocking headroom",
    )
    comp_with_psu = frozenset({"CPU_HIGH", "MOTHERBOARD", "RAM_32GB", "PSU"})
    comp_core = frozenset({"CPU_HIGH", "MOTHERBOARD", "RAM_32GB"})
    comp_groups = (
        _comp_if_available(items, comp_with_psu, _jitter(360, rng),
                           "a complete overclocking platform: CPU, board, RAM, and robust power delivery")
        or _comp_if_available(items, comp_core, _jitter(240, rng),
                              "the core overclocking trio: CPU, motherboard, and high-speed RAM")
    )

    return BidderPreferenceProfile(
        bidder_id="overclocker",
        role=(
            "Chris is an enthusiast overclocker who pushes CPU frequencies to the limit "
            "and needs a high-quality board, fast RAM, and robust power delivery."
        ),
        budget_range=(1000, 2000),
        base_values=bv,
        substitute_groups=sub_groups,
        complement_groups=comp_groups,
        core_items=frozenset({"CPU_HIGH", "MOTHERBOARD", "PSU"}) & items,
        secondary_items=frozenset({"RAM_32GB", "RAM_64GB"}) & items,
        low_interest_items=frozenset({"GPU_HIGH", "GPU_MID", "SSD_2TB", "SSD_1TB"}) & items,
    )


def make_pc_reseller_profile(
    items: set[str], rng: random.Random
) -> BidderPreferenceProfile:
    refs: dict[str, float] = {
        "CPU_HIGH": 560, "GPU_HIGH": 630, "MOTHERBOARD": 260, "RAM_32GB": 220,
        "CPU_BUDGET": 210, "GPU_MID": 310, "SSD_2TB": 170, "PSU": 130,
        "RAM_64GB": 195, "SSD_1TB": 120,
    }
    bv = {item: _jitter(refs[item], rng) for item in items if item in refs}

    # High backup factors: reseller can sell each unit separately
    sub_groups = (
        _sub_if_available(items, frozenset({"CPU_HIGH", "CPU_BUDGET"}), 0.88,
                          "both CPUs can be listed separately; the duplicate retains nearly full resale value")
        + _sub_if_available(items, frozenset({"GPU_HIGH", "GPU_MID"}), 0.88,
                            "both GPUs can be listed and sold separately")
        + _sub_if_available(items, frozenset({"RAM_64GB", "RAM_32GB"}), 0.85,
                            "both RAM kits can be sold to different buyers")
        + _sub_if_available(items, frozenset({"SSD_2TB", "SSD_1TB"}), 0.85,
                            "both SSDs can be sold separately to different buyers")
    )
    comp_groups: list[ComplementGroup] = []
    if len(items) >= 6:
        comp_groups = _comp_if_available(
            items, frozenset(items), _jitter(200, rng),
            "a near-complete lot sells faster and at a modest premium over loose parts",
        )

    return BidderPreferenceProfile(
        bidder_id="pc_reseller",
        role=(
            "Dana buys and resells PC components and pre-assembled bundles. She has "
            "positive resale value for every catalog item and benefits from complete lots."
        ),
        budget_range=(1500, 3500),
        base_values=bv,
        substitute_groups=sub_groups,
        complement_groups=comp_groups,
        notes=(
            "Unlike most bidders, Dana values duplicate components at nearly their "
            "full individual resale prices, since she can list each item separately. "
            "Her bundling premium is modest — roughly 5–15% above the component sum — "
            "because pre-assembled lots sell faster, not because she personally needs "
            "the items together. She never discounts any item to zero."
        ),
    )


def make_student_builder_profile(
    items: set[str], rng: random.Random
) -> BidderPreferenceProfile:
    refs: dict[str, float] = {
        "CPU_BUDGET": 540, "GPU_MID": 470, "MOTHERBOARD": 250, "RAM_32GB": 230,
        "CPU_HIGH": 260, "GPU_HIGH": 280, "SSD_1TB": 160, "PSU": 130,
        "RAM_64GB": 130, "SSD_2TB": 140,
    }
    bv = {item: _jitter(refs[item], rng) for item in items if item in refs}

    sub_groups = (
        _sub_if_available(items, frozenset({"CPU_BUDGET", "CPU_HIGH"}), 0.08,
                          "prefers the budget CPU to keep total cost low; only one CPU can be installed")
        + _sub_if_available(items, frozenset({"GPU_MID", "GPU_HIGH"}), 0.08,
                            "prefers the mid-range GPU for cost efficiency; only one GPU can be installed")
    )
    comp_groups = _comp_if_available(
        items, frozenset({"CPU_BUDGET", "MOTHERBOARD", "RAM_32GB"}), _jitter(130, rng),
        "the minimal core of a working student machine",
    )

    return BidderPreferenceProfile(
        bidder_id="student_builder",
        role=(
            "Jamie is a university student building their first PC on a tight budget, "
            "prioritising a usable machine over high-end performance."
        ),
        budget_range=(600, 1200),
        base_values=bv,
        substitute_groups=sub_groups,
        complement_groups=comp_groups,
        saturation_start=4,
        saturation_penalty=70.0,
        core_items=frozenset({"CPU_BUDGET", "MOTHERBOARD", "RAM_32GB"}) & items,
        secondary_items=frozenset({"GPU_MID", "PSU", "SSD_1TB"}) & items,
        low_interest_items=frozenset({"RAM_64GB", "SSD_2TB"}) & items,
    )


def make_small_business_buyer_profile(
    items: set[str], rng: random.Random
) -> BidderPreferenceProfile:
    refs: dict[str, float] = {
        "CPU_HIGH": 720, "MOTHERBOARD": 340, "RAM_32GB": 310, "RAM_64GB": 360,
        "SSD_2TB": 300, "SSD_1TB": 240, "PSU": 200, "GPU_MID": 175,
        "GPU_HIGH": 210, "CPU_BUDGET": 290,
    }
    bv = {item: _jitter(refs[item], rng) for item in items if item in refs}

    sub_groups = (
        _sub_if_available(items, frozenset({"CPU_HIGH", "CPU_BUDGET"}), 0.10,
                          "CPU_HIGH is preferred for reliability; CPU_BUDGET is an acceptable fallback")
        + _sub_if_available(items, frozenset({"RAM_64GB", "RAM_32GB"}), 0.25,
                            "RAM_64GB is preferred for multitasking; 32 GB is workstation-adequate")
        + _sub_if_available(items, frozenset({"SSD_2TB", "SSD_1TB"}), 0.3,
                            "SSD_2TB is preferred for document and data storage needs")
    )
    comp_groups = _comp_if_available(
        items, frozenset({"CPU_HIGH", "MOTHERBOARD", "RAM_32GB"}), _jitter(200, rng),
        "a reliable workstation platform for office fleet deployment",
    )

    return BidderPreferenceProfile(
        bidder_id="small_business_buyer",
        role=(
            "Taylor manages IT procurement for a small business and is buying "
            "reliable workstation components for office deployments."
        ),
        budget_range=(800, 1800),
        base_values=bv,
        substitute_groups=sub_groups,
        complement_groups=comp_groups,
        core_items=frozenset({"CPU_HIGH", "MOTHERBOARD", "RAM_32GB", "SSD_1TB"}) & items,
        secondary_items=frozenset({"SSD_2TB", "RAM_64GB", "PSU"}) & items,
        low_interest_items=frozenset({"GPU_HIGH", "GPU_MID"}) & items,
    )


def make_data_science_student_profile(
    items: set[str], rng: random.Random
) -> BidderPreferenceProfile:
    refs: dict[str, float] = {
        "GPU_HIGH": 890, "RAM_64GB": 420, "SSD_2TB": 360, "CPU_HIGH": 490,
        "MOTHERBOARD": 275, "GPU_MID": 610, "RAM_32GB": 310, "SSD_1TB": 260,
        "PSU": 200, "CPU_BUDGET": 270,
    }
    bv = {item: _jitter(refs[item], rng) for item in items if item in refs}

    sub_groups = (
        _sub_if_available(items, frozenset({"GPU_HIGH", "GPU_MID"}), 0.10,
                          "GPU_HIGH is strongly preferred for ML training throughput; GPU_MID is a fallback")
        + _sub_if_available(items, frozenset({"RAM_64GB", "RAM_32GB"}), 0.25,
                            "RAM_64GB is preferred for large dataset operations; 32 GB is a lighter fallback")
        + _sub_if_available(items, frozenset({"SSD_2TB", "SSD_1TB"}), 0.3,
                            "SSD_2TB is preferred for storing datasets and model checkpoints")
    )
    best_gpu = "GPU_HIGH" if "GPU_HIGH" in items else ("GPU_MID" if "GPU_MID" in items else None)
    best_ram = "RAM_64GB" if "RAM_64GB" in items else ("RAM_32GB" if "RAM_32GB" in items else None)
    best_ssd = "SSD_2TB" if "SSD_2TB" in items else ("SSD_1TB" if "SSD_1TB" in items else None)
    comp_items: set[str] = {
        i for i in [best_gpu, best_ram, best_ssd, "CPU_HIGH"] if i is not None
    }
    comp_groups = (
        _comp_if_available(
            items, frozenset(comp_items), _jitter(280, rng),
            "a complete ML workstation: GPU compute, ample RAM, fast storage, and a capable CPU",
        )
        if len(comp_items) >= 3 else []
    )

    return BidderPreferenceProfile(
        bidder_id="data_science_student",
        role=(
            "Casey is a data science student who needs to run GPU-accelerated ML "
            "experiments, store large datasets, and keep training environments in RAM."
        ),
        budget_range=(1200, 2200),
        base_values=bv,
        substitute_groups=sub_groups,
        complement_groups=comp_groups,
        core_items=frozenset({"GPU_HIGH", "RAM_64GB", "SSD_2TB"}) & items,
        secondary_items=frozenset({"CPU_HIGH", "MOTHERBOARD", "GPU_MID"}) & items,
        low_interest_items=frozenset({"CPU_BUDGET", "PSU"}) & items,
    )


# ---------------------------------------------------------------------------
# Archetype registry
# ---------------------------------------------------------------------------

_ARCHETYPE_ORDER: list[str] = [
    "competitive_gamer",
    "budget_gamer",
    "video_editor",
    "ai_hobbyist",
    "office_upgrader",
    "overclocker",
    "pc_reseller",
    "student_builder",
    "small_business_buyer",
    "data_science_student",
]

_ARCHETYPE_BUILDERS: dict[str, object] = {
    "competitive_gamer": make_competitive_gamer_profile,
    "budget_gamer": make_budget_gamer_profile,
    "video_editor": make_video_editor_profile,
    "ai_hobbyist": make_ai_hobbyist_profile,
    "office_upgrader": make_office_upgrader_profile,
    "overclocker": make_overclocker_profile,
    "pc_reseller": make_pc_reseller_profile,
    "student_builder": make_student_builder_profile,
    "small_business_buyer": make_small_business_buyer_profile,
    "data_science_student": make_data_science_student_profile,
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_pc_build_scenario(
    num_goods: int,
    num_bidders: int,
    seed: int = 0,
    name: Optional[str] = None,
) -> NaturalLanguageAuctionScenario:
    """Generate a structured PC-build auction scenario.

    Parameters
    ----------
    num_goods:
        Number of items from :data:`PC_GOOD_CATALOG` (4–10 inclusive).
    num_bidders:
        Number of bidder archetypes (3–10 inclusive).
    seed:
        Random seed for jitter reproducibility.
    name:
        Scenario name override; defaults to
        ``f"pc_build_{num_goods}x{num_bidders}_calibrated"``.

    Returns
    -------
    NaturalLanguageAuctionScenario
        Full valuation tables over all 2^num_goods − 1 non-empty bundles,
        budget-calibrated NL person seeds, and
        ``candidate_bundles_by_bidder=None`` (proxies derive candidates from
        the interest map or generic enumeration).
    """
    if not (4 <= num_goods <= 10):
        raise ValueError(
            f"num_goods must be between 4 and 10 inclusive, got {num_goods}"
        )
    if not (3 <= num_bidders <= 10):
        raise ValueError(
            f"num_bidders must be between 3 and 10 inclusive, got {num_bidders}"
        )

    if name is None:
        name = f"pc_build_{num_goods}x{num_bidders}_calibrated"

    items = PC_GOOD_CATALOG[:num_goods]
    bidder_ids = _ARCHETYPE_ORDER[:num_bidders]
    items_set: set[str] = set(items)
    item_descriptions = {item: PC_ITEM_DESCRIPTIONS[item] for item in items}

    scenario_description = (
        f"A PC component auction with {num_goods} items and {num_bidders} bidders. "
        f"The items available are: {', '.join(items)}. "
        "Bidders have diverse preferences driven by their professional and personal "
        "computing needs. All components are new and compatible with one another "
        "as described in the item descriptions."
    )

    rng = random.Random(seed)
    profiles: list[BidderPreferenceProfile] = [
        _ARCHETYPE_BUILDERS[bidder_id](items_set, rng)  # type: ignore[operator]
        for bidder_id in bidder_ids
    ]

    valuations = generate_full_valuations(items, profiles)
    person_seeds = {
        p.bidder_id: render_budget_calibrated_seed(p)
        for p in profiles
    }

    profile_metadata = {
        p.bidder_id: {
            "budget_cap": p.budget_cap,
            "saturation_start": p.saturation_start,
            "saturation_penalty": p.saturation_penalty,
            "core_items": sorted(p.core_items),
            "secondary_items": sorted(p.secondary_items),
            "low_interest_items": sorted(p.low_interest_items),
            "substitute_groups": [
                {"items": sorted(sg.items), "backup_factor": sg.backup_factor}
                for sg in p.substitute_groups
            ],
            "complement_groups": [
                {"items": sorted(cg.items), "bonus": cg.bonus}
                for cg in p.complement_groups
            ],
        }
        for p in profiles
    }

    return NaturalLanguageAuctionScenario(
        name=name,
        seed_type="structured",
        instance=AuctionInstance(
            items=items,
            bidder_ids=bidder_ids,
            valuations=valuations,
        ),
        scenario_description=scenario_description,
        item_descriptions=item_descriptions,
        person_seeds=person_seeds,
        candidate_bundles_by_bidder=None,
        metadata={
            "num_goods": num_goods,
            "num_bidders": num_bidders,
            "scenario_seed": seed,
            "full_valuation_table_size": 2 ** num_goods - 1,
            "domain": "pc_build",
            "valuation_model": "structured_substitutes_complements",
            "seed_style": "budget_calibrated",
            "profiles": profile_metadata,
        },
    )
