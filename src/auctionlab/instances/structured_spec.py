"""Spec-based factory for PC-build scenarios.

Builds :class:`~auctionlab.instances.nl_types.NaturalLanguageAuctionScenario`
instances from a frozen :class:`~auctionlab.instances.scenario_spec.ScenarioProfileSpec`
instead of the hard-coded archetype builders in
:mod:`auctionlab.instances.structured`. Item/bidder filtering reuses the
exact same substitute/complement availability rules as the hard-coded
builders (:func:`auctionlab.instances.structured._sub_if_available` and
:func:`auctionlab.instances.structured._comp_if_available`), and valuation
generation reuses the same deterministic valuation pipeline. The simulated
person receives only a brief qualitative disclosure; exact bundle values stay
private in the generated valuation table.

Usage::

    from auctionlab.instances.structured_spec import make_pc_build_scenario_from_spec
    scenario = make_pc_build_scenario_from_spec(
        "scenarios/pc_build_v3/pc_build_population_16x16.json",
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
from auctionlab.instances.population_design import (
    NestedPopulationOrder,
    coverage_aware_nested_order,
)
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
    render_brief_qualitative_person_seed,
)

SELECTION_POLICIES = (
    "prefix",
    "seeded_sample",
    "stratified",
    "coverage_stratified",
)


# ---------------------------------------------------------------------------
# Category inference for stratified selection
# ---------------------------------------------------------------------------
#
# GoodSpec.category is a free-text hint and older bidder specs predate
# BidderProfileSpec.archetype_category. The legacy stratified selector
# therefore retains keyword inference as a fallback. New master populations
# supply explicit bidder categories, which the coverage-aware selector uses.

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
    if bidder.archetype_category:
        return [bidder.archetype_category]
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
        # Build one seed-specific permutation and take prefixes so size
        # sweeps are nested: the selected 4-item instance is a subset of the
        # 5-item instance for the same seed, and so on.
        ordered = list(all_ids)
        rng.shuffle(ordered)
        chosen = set(ordered[:count])
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
            _sub_if_available(
                items,
                frozenset(sg.items),
                sg.backup_factor,
                sg.description,
                sg.acquisition_mode,
            )
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
        ``"coverage_stratified"`` searches deterministically for nested goods
        and bidder orders whose 4--10 scalability cells satisfy the structural
        constraints embedded in the spec.
    """
    if isinstance(spec_path, ScenarioProfileSpec):
        spec = spec_path
    else:
        spec = load_scenario_profile_spec(spec_path)

    all_good_ids = [g.id for g in spec.goods]
    all_bidder_ids = [b.bidder_id for b in spec.bidder_profiles]
    if not 1 <= num_goods <= len(all_good_ids):
        raise ValueError(
            f"num_goods must be between 1 and {len(all_good_ids)}, "
            f"got {num_goods}"
        )
    if not 1 <= num_bidders <= len(all_bidder_ids):
        raise ValueError(
            f"num_bidders must be between 1 and {len(all_bidder_ids)}, "
            f"got {num_bidders}"
        )

    goods_by_id = {g.id: g for g in spec.goods}
    bidders_by_id = {b.bidder_id: b for b in spec.bidder_profiles}

    nested_order = None
    if selection_policy == "coverage_stratified":
        if num_goods < 4 or num_bidders < 4:
            raise ValueError(
                "coverage_stratified requires at least 4 goods and 4 bidders"
            )
        sample_constraints = (
            spec.generation.get("sample_constraints")
            if isinstance(spec.generation, dict)
            else None
        )
        frozen_orders = (
            spec.generation.get("coverage_orders", {})
            if isinstance(spec.generation, dict)
            else {}
        )
        frozen = frozen_orders.get(str(seed))
        can_use_frozen = (
            isinstance(frozen, dict)
            and num_goods <= int(frozen.get("validated_max_goods", 0))
            and num_bidders <= int(frozen.get("validated_max_bidders", 0))
            and set(frozen.get("goods", [])) == set(all_good_ids)
            and set(frozen.get("bidders", [])) == set(all_bidder_ids)
            and len(frozen.get("goods", [])) == len(all_good_ids)
            and len(frozen.get("bidders", [])) == len(all_bidder_ids)
        )
        if can_use_frozen:
            nested_order = NestedPopulationOrder(
                goods=tuple(frozen["goods"]),
                bidders=tuple(frozen["bidders"]),
                attempts=int(frozen.get("attempts", 0)),
            )
        else:
            nested_order = coverage_aware_nested_order(
                spec,
                seed=seed,
                constraints=sample_constraints,
            )
        selected_goods = set(nested_order.goods[:num_goods])
        selected_bidders = set(nested_order.bidders[:num_bidders])
        items = [item_id for item_id in all_good_ids if item_id in selected_goods]
        bidder_ids = [
            bidder_id for bidder_id in all_bidder_ids
            if bidder_id in selected_bidders
        ]
    else:
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

    person_seeds: dict[str, str] = {}
    person_seed_sources: dict[str, str] = {}
    identity_sources: dict[str, str] = {}
    for p in profiles:
        bidder_spec = bidders_by_id[p.bidder_id]
        identity = bidder_spec.identity_text or bidder_spec.role
        person_seeds[p.bidder_id] = render_brief_qualitative_person_seed(
            p,
            identity_text=identity,
            available_goods=items,
        )
        identity_sources[p.bidder_id] = (
            "identity_text" if bidder_spec.identity_text else "role"
        )
        person_seed_sources[p.bidder_id] = "brief_qualitative_disclosure"

    environment_generation_provider = None
    environment_generation_model = None
    if isinstance(spec.generation, dict):
        environment_generation_provider = spec.generation.get("provider")
        environment_generation_model = spec.generation.get("model")

    profile_metadata = {
        p.bidder_id: {
            "role": p.role,
            "budget_range": list(p.budget_range),
            "budget_cap": p.budget_cap,
            "disclosed_budget_hint": max(
                valuations[p.bidder_id].values(), default=0.0
            ),
            "saturation_start": p.saturation_start,
            "saturation_penalty": p.saturation_penalty,
            "disclosed_positive_items": sorted(
                item
                for item in items
                if p.base_values.get(item, 0.0) > 0
            ),
            "core_items": sorted(p.core_items),
            "secondary_items": sorted(p.secondary_items),
            "low_interest_items": sorted(p.low_interest_items),
            "substitute_groups": [
                {
                    "items": sorted(sg.items),
                    "backup_factor": sg.backup_factor,
                    "acquisition_mode": sg.acquisition_mode,
                }
                for sg in p.substitute_groups
            ],
            "complement_groups": [
                {"items": sorted(cg.items), "bonus": cg.bonus}
                for cg in p.complement_groups
            ],
            "person_seed_source": person_seed_sources[p.bidder_id],
            "person_seed_identity_source": identity_sources[p.bidder_id],
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
            "seed_style": "brief_qualitative",
            "profiles": profile_metadata,
            "spec_schema_version": spec.schema_version,
            "environment_generation_provider": environment_generation_provider,
            "environment_generation_model": environment_generation_model,
            "selection_policy": selection_policy,
            "selection_order_attempts": (
                nested_order.attempts if nested_order is not None else None
            ),
            "selected_goods_order": (
                list(nested_order.goods[:num_goods])
                if nested_order is not None else list(items)
            ),
            "selected_bidders_order": (
                list(nested_order.bidders[:num_bidders])
                if nested_order is not None else list(bidder_ids)
            ),
        },
    )
