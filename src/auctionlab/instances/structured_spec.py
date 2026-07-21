"""Spec-based factory for PC-build scenarios.

Builds :class:`~auctionlab.instances.nl_types.NaturalLanguageAuctionScenario`
instances from a frozen :class:`~auctionlab.instances.scenario_spec.ScenarioProfileSpec`
instead of the hard-coded archetype builders in
:mod:`auctionlab.instances.structured`. Item/bidder filtering reuses the
exact same substitute/complement availability rules as the hard-coded
builders (:func:`auctionlab.instances.structured._sub_if_available` and
:func:`auctionlab.instances.structured._comp_if_available`), and valuation
generation reuses the same deterministic pipeline
(:func:`generate_full_valuations`, :func:`enforce_monotonicity`,
:func:`render_budget_calibrated_seed`).

Usage::

    from auctionlab.instances.structured_spec import make_pc_build_scenario_from_spec
    scenario = make_pc_build_scenario_from_spec(
        "scenarios/pc_build_v1/pc_build_profiles_v0_manual.json",
        num_goods=6, num_bidders=6,
    )
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Callable, Optional

from auctionlab.instances.base import AuctionInstance
from auctionlab.instances.nl_types import NaturalLanguageAuctionScenario
from auctionlab.instances.scenario_spec import (
    BidderProfileSpec,
    GoodSpec,
    ScenarioProfileSpec,
    load_scenario_profile_spec,
)
from auctionlab.instances.structured import (
    BidderPreferenceProfile,
    ComplementGroup,
    SubstituteGroup,
    _comp_if_available,
    _sub_if_available,
    generate_full_valuations,
    render_budget_calibrated_seed,
)

SELECTION_POLICIES = ("prefix", "seeded_sample", "stratified")


# ---------------------------------------------------------------------------
# Category inference for stratified selection
# ---------------------------------------------------------------------------
#
# Neither GoodSpec nor BidderProfileSpec carries an explicit "archetype
# category" field (GoodSpec.category is a free-text, LLM-populated, often
# absent hint). Stratified selection instead infers a coarse category from
# whatever text is available -- GoodSpec.category/id/description, or
# BidderProfileSpec.bidder_id/role -- via keyword matching. This is a
# best-effort heuristic, not a schema guarantee.

GOOD_CATEGORY_ORDER = ["cpu", "gpu", "motherboard", "ram", "storage", "power_cooling"]

_GOOD_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cpu": ("cpu", "processor"),
    "gpu": ("gpu", "graphics"),
    "motherboard": ("motherboard", "mainboard", "mobo", "mb"),
    "ram": ("ram",),
    "storage": ("ssd", "hdd", "nvme", "storage"),
    "power_cooling": ("psu", "power", "cool"),
}

BIDDER_CATEGORY_ORDER = [
    "gaming_performance",
    "budget_office",
    "professional_ai",
    "reseller_procurement",
]

_BIDDER_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "gaming_performance": ("gamer", "gaming", "enthusiast", "overclock", "performance"),
    "budget_office": (
        "budget", "office", "student", "casual", "productivity", "minimalist",
        "home server", "tinkerer", "hobbyist",
    ),
    "professional_ai": (
        "ai", "ml", "machine learning", "creative", "professional", "workstation",
        "data science", "developer", "practitioner", "video", "editing",
    ),
    "reseller_procurement": ("reseller", "procurement"),
}


def _normalize_for_matching(text: str) -> str:
    """Lowercase and split id-style underscores into word boundaries.

    ``"MB_PRO"`` -> ``"mb pro"`` so a bare ``"mb"`` keyword matches it, while
    leaving prose untouched.
    """
    return text.lower().replace("_", " ")


def _matches_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Whole-word-prefix match: a keyword must start at a word boundary.

    Prevents e.g. the ``"ram"`` keyword (for the RAM category) from matching
    inside "VRAM" (a GPU's video memory) -- there's no word boundary right
    before "ram" in "vram". Only a *leading* boundary is required (not a
    trailing one), so "cool" still matches "cooling"/"cooler" and "cpu"
    still matches "cpus".
    """
    return any(re.search(rf"\b{re.escape(kw)}", text) for kw in keywords)


def categorize_good(good: GoodSpec) -> list[str]:
    """Return every category bucket (of :data:`GOOD_CATEGORY_ORDER`) matched
    by a good's id/category/description text."""
    text = _normalize_for_matching(f"{good.id} {good.category or ''} {good.description}")
    return [cat for cat in GOOD_CATEGORY_ORDER if _matches_any_keyword(text, _GOOD_CATEGORY_KEYWORDS[cat])]


def categorize_bidder(bidder: BidderProfileSpec) -> list[str]:
    """Return every category bucket (of :data:`BIDDER_CATEGORY_ORDER`)
    matched by a bidder's id/role text."""
    text = _normalize_for_matching(f"{bidder.bidder_id} {bidder.role}")
    return [
        cat for cat in BIDDER_CATEGORY_ORDER if _matches_any_keyword(text, _BIDDER_CATEGORY_KEYWORDS[cat])
    ]


def _stratified_select(
    all_ids: list[str],
    categorize: Callable[[str], list[str]],
    category_order: list[str],
    count: int,
    seed: int,
    salt: str,
) -> list[str]:
    """Deterministically pick ``count`` ids, favoring one representative per
    category in ``category_order`` (when available) before filling the rest.

    Categories are satisfied in the order given, so if ``count`` is smaller
    than the number of populated categories, the earliest categories in
    ``category_order`` win. Remaining slots are filled by a seeded shuffle
    of whatever ids are left, so results are reproducible for a fixed seed.
    """
    membership: dict[str, list[str]] = {cat: [] for cat in category_order}
    for item_id in all_ids:
        for cat in categorize(item_id):
            membership[cat].append(item_id)

    rng = random.Random(f"{salt}_stratified:{seed}")
    chosen: list[str] = []
    for cat in category_order:
        candidates = [c for c in membership[cat] if c not in chosen]
        if candidates and len(chosen) < count:
            chosen.append(rng.choice(candidates))

    remaining_pool = [item_id for item_id in all_ids if item_id not in chosen]
    rng.shuffle(remaining_pool)
    for item_id in remaining_pool:
        if len(chosen) >= count:
            break
        chosen.append(item_id)

    chosen_set = set(chosen[:count])
    return [item_id for item_id in all_ids if item_id in chosen_set]


def _select_ids(
    all_ids: list[str],
    count: int,
    seed: int,
    selection_policy: str,
    *,
    label: str,
    salt: str,
    categorize: Callable[[str], list[str]] | None = None,
    category_order: list[str] | None = None,
) -> list[str]:
    if not (1 <= count <= len(all_ids)):
        raise ValueError(
            f"num_{label} must be between 1 and {len(all_ids)} (spec has {len(all_ids)}), "
            f"got {count}"
        )
    if selection_policy == "prefix":
        return all_ids[:count]
    if selection_policy == "seeded_sample":
        rng = random.Random(f"{salt}:{seed}")
        chosen = set(rng.sample(all_ids, count))
        return [item_id for item_id in all_ids if item_id in chosen]
    if selection_policy == "stratified":
        assert categorize is not None and category_order is not None
        return _stratified_select(all_ids, categorize, category_order, count, seed, salt)
    raise ValueError(
        f"Unknown selection_policy {selection_policy!r}; expected one of {SELECTION_POLICIES}"
    )


def profile_from_spec(
    bidder_spec: BidderProfileSpec, items: set[str]
) -> BidderPreferenceProfile:
    """Build a :class:`BidderPreferenceProfile` restricted to ``items``.

    Filtering semantics exactly mirror the hard-coded archetype builders:
    substitute groups need >= 2 of their items present, complement groups
    need all of their items present, and base values / core / secondary /
    low-interest lists are simply intersected with ``items``.
    """
    base_values = {item: v for item, v in bidder_spec.base_values.items() if item in items}

    substitute_groups: list[SubstituteGroup] = []
    for sg in bidder_spec.substitute_groups:
        substitute_groups.extend(
            _sub_if_available(items, frozenset(sg.items), sg.backup_factor, sg.description)
        )

    complement_groups: list[ComplementGroup] = []
    for cg in bidder_spec.complement_groups:
        complement_groups.extend(
            _comp_if_available(items, frozenset(cg.items), cg.bonus, cg.description)
        )

    return BidderPreferenceProfile(
        bidder_id=bidder_spec.bidder_id,
        role=bidder_spec.role,
        budget_range=bidder_spec.budget_range,
        base_values=base_values,
        substitute_groups=substitute_groups,
        complement_groups=complement_groups,
        budget_cap=bidder_spec.budget_cap,
        saturation_start=bidder_spec.saturation_start,
        saturation_penalty=bidder_spec.saturation_penalty,
        notes=bidder_spec.notes,
        core_items=frozenset(bidder_spec.core_items) & items,
        secondary_items=frozenset(bidder_spec.secondary_items) & items,
        low_interest_items=frozenset(bidder_spec.low_interest_items) & items,
    )


def make_pc_build_scenario_from_spec(
    spec_path: str | Path | ScenarioProfileSpec,
    num_goods: int,
    num_bidders: int,
    seed: int = 0,
    name: Optional[str] = None,
    selection_policy: str = "prefix",
) -> NaturalLanguageAuctionScenario:
    """Generate a PC-build auction scenario from a frozen profile spec.

    Parameters
    ----------
    spec_path:
        Path to a frozen ``ScenarioProfileSpec`` JSON file, or an
        already-loaded ``ScenarioProfileSpec``.
    num_goods, num_bidders:
        How many goods/bidder profiles to select from the spec.
    seed:
        Consulted when ``selection_policy`` is ``"seeded_sample"`` or
        ``"stratified"`` (it seeds which goods/bidders are sampled). All
        numeric values in the spec are already frozen, so unlike
        :func:`auctionlab.instances.structured.make_pc_build_scenario`
        there is no further jitter to reseed — the same spec always
        produces the same valuations and person seeds for a given
        selection, regardless of ``seed``, under ``selection_policy="prefix"``.
    selection_policy:
        ``"prefix"`` (default) selects the first ``num_goods``/``num_bidders``
        entries in spec declaration order, matching the behaviour of the
        hard-coded catalog-order factory. ``"seeded_sample"`` deterministically
        samples a subset using ``seed``. ``"stratified"`` deterministically
        picks one representative good per hardware category (CPU, GPU,
        motherboard, RAM, storage, power/cooling) and one bidder per broad
        archetype bucket (gaming/performance, budget/office, professional/AI,
        reseller/procurement) when available, then fills remaining slots via
        a seeded shuffle -- useful for small ``num_goods``/``num_bidders``
        subsets of a larger generated spec, where a plain prefix can miss a
        complete "platform" of complementary goods.
    """
    if isinstance(spec_path, ScenarioProfileSpec):
        spec = spec_path
    else:
        spec = load_scenario_profile_spec(spec_path)

    all_good_ids = [g.id for g in spec.goods]
    all_bidder_ids = [b.bidder_id for b in spec.bidder_profiles]

    goods_by_id = {g.id: g for g in spec.goods}
    bidders_by_id = {b.bidder_id: b for b in spec.bidder_profiles}

    items = _select_ids(
        all_good_ids,
        num_goods,
        seed,
        selection_policy,
        label="goods",
        salt="goods",
        categorize=lambda good_id: categorize_good(goods_by_id[good_id]),
        category_order=GOOD_CATEGORY_ORDER,
    )
    bidder_ids = _select_ids(
        all_bidder_ids,
        num_bidders,
        seed,
        selection_policy,
        label="bidders",
        salt="bidders",
        categorize=lambda bidder_id: categorize_bidder(bidders_by_id[bidder_id]),
        category_order=BIDDER_CATEGORY_ORDER,
    )
    items_set = set(items)

    if name is None:
        name = f"pc_build_{num_goods}x{num_bidders}_from_spec"

    item_descriptions = {item: goods_by_id[item].description for item in items}

    scenario_description = (
        f"A PC component auction with {num_goods} items and {num_bidders} bidders. "
        f"The items available are: {', '.join(items)}. "
        "Bidders have diverse preferences driven by their professional and personal "
        "computing needs. All components are new and compatible with one another "
        "as described in the item descriptions."
    )

    profiles: list[BidderPreferenceProfile] = [
        profile_from_spec(bidders_by_id[bidder_id], items_set) for bidder_id in bidder_ids
    ]

    valuations = generate_full_valuations(items, profiles)
    person_seeds = {p.bidder_id: render_budget_calibrated_seed(p) for p in profiles}

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
            "spec_schema_version": spec.schema_version,
            "selection_policy": selection_policy,
        },
    )
