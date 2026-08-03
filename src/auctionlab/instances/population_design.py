"""Coverage diagnostics and nested sampling for master scenario populations."""

from __future__ import annotations

import random
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Iterable, Sequence

from auctionlab.instances.scenario_spec import ScenarioProfileSpec
from auctionlab.instances.scenario_spec import BidderProfileSpec


DEFAULT_SAMPLE_CONSTRAINTS: dict[str, Any] = {
    "min_positive_bidders_per_good": 2,
    "min_positive_goods_per_bidder": 1,
    "min_substitute_groups_by_goods": {"4": 1, "6": 2, "8": 3},
    "min_complement_groups_by_goods": {"4": 1, "6": 2, "8": 3},
    "min_distinct_substitute_groups_by_goods": {"4": 1, "6": 2, "8": 3},
    "min_distinct_complement_groups_by_goods": {"4": 1, "6": 2, "8": 3},
    "min_bidders_with_substitute_groups_by_goods": {"4": 2, "6": 3, "8": 4},
    "min_bidders_with_complement_groups_by_goods": {"4": 1, "6": 2, "8": 3},
    "min_good_categories_by_goods": {},
    "min_bidder_strata_by_bidders": {},
    "max_mean_interest_density_by_goods": {},
    "min_fraction_bidders_with_exclusions": 0.0,
}


def _stepped_threshold(
    mapping: dict[str, int | float],
    size: int,
) -> int | float:
    eligible = [
        (int(lower_bound), value)
        for lower_bound, value in mapping.items()
        if int(lower_bound) <= size
    ]
    return max(eligible, default=(0, 0), key=lambda pair: pair[0])[1]


def sample_structure_report(
    spec: ScenarioProfileSpec,
    selected_goods: Iterable[str],
    selected_bidders: Iterable[str],
    *,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure structural interest and relationship coverage in one sample."""
    rules = {**DEFAULT_SAMPLE_CONSTRAINTS, **(constraints or {})}
    goods = list(selected_goods)
    bidders = list(selected_bidders)
    goods_set = set(goods)
    profiles = {
        bidder.bidder_id: bidder
        for bidder in spec.bidder_profiles
        if bidder.bidder_id in set(bidders)
    }
    goods_by_id = {good.id: good for good in spec.goods}
    selected_good_categories = sorted(
        {
            goods_by_id[good].category or "uncategorized"
            for good in goods
            if good in goods_by_id
        }
    )
    selected_bidder_strata = sorted(
        {
            profile.archetype_category or "unclassified"
            for profile in profiles.values()
        }
    )
    positive_bidders_by_good = {
        good: sorted(
            bidder_id
            for bidder_id, profile in profiles.items()
            if profile.base_values.get(good, 0.0) > 0
        )
        for good in goods
    }
    positive_goods_by_bidder = {
        bidder_id: sorted(
            good
            for good in goods
            if profile.base_values.get(good, 0.0) > 0
        )
        for bidder_id, profile in profiles.items()
    }
    positive_pair_count = sum(
        len(interested) for interested in positive_bidders_by_good.values()
    )
    possible_pair_count = len(goods) * len(profiles)
    mean_interest_density = (
        positive_pair_count / possible_pair_count
        if possible_pair_count
        else 0.0
    )
    bidders_with_exclusions = sorted(
        bidder_id
        for bidder_id, interested in positive_goods_by_bidder.items()
        if len(interested) < len(goods)
    )
    surviving_substitutes: list[dict[str, Any]] = []
    surviving_complements: list[dict[str, Any]] = []
    for bidder_id, profile in profiles.items():
        for group in profile.substitute_groups:
            available = sorted(goods_set & set(group.items))
            if len(available) >= 2:
                surviving_substitutes.append(
                    {"bidder_id": bidder_id, "items": available}
                )
        for group in profile.complement_groups:
            if set(group.items) <= goods_set:
                surviving_complements.append(
                    {"bidder_id": bidder_id, "items": sorted(group.items)}
                )

    min_good_interest = int(rules["min_positive_bidders_per_good"])
    min_bidder_interest = int(rules["min_positive_goods_per_bidder"])
    min_substitutes = _stepped_threshold(
        rules["min_substitute_groups_by_goods"], len(goods)
    )
    min_complements = _stepped_threshold(
        rules["min_complement_groups_by_goods"], len(goods)
    )
    distinct_substitutes = {
        tuple(group["items"]) for group in surviving_substitutes
    }
    distinct_complements = {
        tuple(group["items"]) for group in surviving_complements
    }
    bidders_with_substitutes = {
        group["bidder_id"] for group in surviving_substitutes
    }
    bidders_with_complements = {
        group["bidder_id"] for group in surviving_complements
    }
    min_distinct_substitutes = _stepped_threshold(
        rules["min_distinct_substitute_groups_by_goods"], len(goods)
    )
    min_distinct_complements = _stepped_threshold(
        rules["min_distinct_complement_groups_by_goods"], len(goods)
    )
    min_substitute_bidders = _stepped_threshold(
        rules["min_bidders_with_substitute_groups_by_goods"], len(goods)
    )
    min_complement_bidders = _stepped_threshold(
        rules["min_bidders_with_complement_groups_by_goods"], len(goods)
    )
    min_good_categories = int(
        _stepped_threshold(
            rules.get("min_good_categories_by_goods", {}),
            len(goods),
        )
    )
    min_bidder_strata = int(
        _stepped_threshold(
            rules.get("min_bidder_strata_by_bidders", {}),
            len(bidders),
        )
    )
    density_thresholds = rules.get(
        "max_mean_interest_density_by_goods", {}
    )
    max_mean_interest_density = (
        float(_stepped_threshold(density_thresholds, len(goods)))
        if density_thresholds
        else 1.0
    )
    min_exclusion_fraction = float(
        rules.get("min_fraction_bidders_with_exclusions", 0.0)
    )
    min_bidders_with_exclusions = math.ceil(
        len(profiles) * min_exclusion_fraction
    )
    violations: list[str] = []
    for good, interested in positive_bidders_by_good.items():
        if len(interested) < min_good_interest:
            violations.append(
                f"good {good} has {len(interested)} interested bidders "
                f"(minimum {min_good_interest})"
            )
    for bidder_id, interested in positive_goods_by_bidder.items():
        if len(interested) < min_bidder_interest:
            violations.append(
                f"bidder {bidder_id} values {len(interested)} selected goods "
                f"(minimum {min_bidder_interest})"
            )
    if len(surviving_substitutes) < min_substitutes:
        violations.append(
            f"{len(surviving_substitutes)} substitute groups survive "
            f"(minimum {min_substitutes})"
        )
    if len(surviving_complements) < min_complements:
        violations.append(
            f"{len(surviving_complements)} complement groups survive "
            f"(minimum {min_complements})"
        )
    if len(distinct_substitutes) < min_distinct_substitutes:
        violations.append(
            f"{len(distinct_substitutes)} distinct substitute groups survive "
            f"(minimum {min_distinct_substitutes})"
        )
    if len(distinct_complements) < min_distinct_complements:
        violations.append(
            f"{len(distinct_complements)} distinct complement groups survive "
            f"(minimum {min_distinct_complements})"
        )
    if len(bidders_with_substitutes) < min_substitute_bidders:
        violations.append(
            f"{len(bidders_with_substitutes)} bidders retain substitutes "
            f"(minimum {min_substitute_bidders})"
        )
    if len(bidders_with_complements) < min_complement_bidders:
        violations.append(
            f"{len(bidders_with_complements)} bidders retain complements "
            f"(minimum {min_complement_bidders})"
        )
    if len(selected_good_categories) < min_good_categories:
        violations.append(
            f"{len(selected_good_categories)} good categories represented "
            f"(minimum {min_good_categories})"
        )
    if len(selected_bidder_strata) < min_bidder_strata:
        violations.append(
            f"{len(selected_bidder_strata)} bidder strata represented "
            f"(minimum {min_bidder_strata})"
        )
    if mean_interest_density > max_mean_interest_density:
        violations.append(
            f"mean bidder-good interest density is "
            f"{mean_interest_density:.3f} "
            f"(maximum {max_mean_interest_density:.3f})"
        )
    if len(bidders_with_exclusions) < min_bidders_with_exclusions:
        violations.append(
            f"{len(bidders_with_exclusions)} bidders exclude at least one "
            f"selected good (minimum {min_bidders_with_exclusions})"
        )

    return {
        "num_goods": len(goods),
        "num_bidders": len(bidders),
        "positive_bidders_by_good": positive_bidders_by_good,
        "positive_goods_by_bidder": positive_goods_by_bidder,
        "surviving_substitute_groups": surviving_substitutes,
        "surviving_complement_groups": surviving_complements,
        "distinct_substitute_groups": [
            list(group) for group in sorted(distinct_substitutes)
        ],
        "distinct_complement_groups": [
            list(group) for group in sorted(distinct_complements)
        ],
        "bidders_with_substitute_groups": sorted(bidders_with_substitutes),
        "bidders_with_complement_groups": sorted(bidders_with_complements),
        "selected_good_categories": selected_good_categories,
        "selected_bidder_strata": selected_bidder_strata,
        "mean_interest_density": mean_interest_density,
        "max_mean_interest_density": max_mean_interest_density,
        "bidders_with_exclusions": bidders_with_exclusions,
        "min_bidders_with_exclusions": min_bidders_with_exclusions,
        "violations": violations,
        "passed": not violations,
    }


def sample_economic_report(
    spec: ScenarioProfileSpec,
    selected_goods: Sequence[str],
    selected_bidders: Sequence[str],
    *,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve one sampled full-information auction and check non-triviality."""
    # Late imports avoid a module cycle: structured_spec uses the structural
    # ordering functions above while constructing scenarios.
    from auctionlab.instances.base import AuctionInstance
    from auctionlab.instances.structured import generate_full_valuations
    from auctionlab.instances.structured_spec import profile_from_spec
    from auctionlab.solvers.wdp_ilp import solve_wdp_xor_ilp

    rules = {**DEFAULT_SAMPLE_CONSTRAINTS, **(constraints or {})}
    goods = list(selected_goods)
    goods_set = set(goods)
    profiles_by_id = {
        bidder.bidder_id: bidder for bidder in spec.bidder_profiles
    }
    profiles = [
        profile_from_spec(profiles_by_id[bidder_id], goods_set)
        for bidder_id in selected_bidders
    ]
    valuations = generate_full_valuations(goods, profiles)
    instance = AuctionInstance(
        items=goods,
        bidder_ids=list(selected_bidders),
        valuations=valuations,
    )
    result = solve_wdp_xor_ilp(instance.items, instance.to_xor_bids())
    winner_values = {
        bidder_id: valuations[bidder_id][bundle]
        for bidder_id, bundle in result.allocation.items()
        if bundle
    }
    num_winners = len(winner_values)
    largest_share = (
        max(winner_values.values()) / result.welfare
        if winner_values and result.welfare > 0
        else 0.0
    )
    min_winners = int(rules.get("min_full_information_winners", 1))
    share_thresholds = rules.get(
        "max_largest_winner_welfare_share_by_goods"
    )
    max_largest_share = (
        float(_stepped_threshold(share_thresholds, len(goods)))
        if share_thresholds
        else float(rules.get("max_largest_winner_welfare_share", 1.0))
    )
    violations: list[str] = []
    if num_winners < min_winners:
        violations.append(
            f"full-information allocation has {num_winners} winners "
            f"(minimum {min_winners})"
        )
    if largest_share > max_largest_share:
        violations.append(
            f"largest winner welfare share is {largest_share:.3f} "
            f"(maximum {max_largest_share:.3f})"
        )
    return {
        "full_information_welfare": result.welfare,
        "num_winners": num_winners,
        "largest_winner_welfare_share": largest_share,
        "max_largest_winner_welfare_share": max_largest_share,
        "allocation": {
            bidder_id: sorted(bundle)
            for bidder_id, bundle in result.allocation.items()
            if bundle
        },
        "violations": violations,
        "passed": not violations,
    }


def sample_acceptance_report(
    spec: ScenarioProfileSpec,
    selected_goods: Sequence[str],
    selected_bidders: Sequence[str],
    *,
    constraints: dict[str, Any] | None = None,
    include_economic: bool = True,
) -> dict[str, Any]:
    """Combine structural coverage and full-information allocation checks."""
    structural = sample_structure_report(
        spec,
        selected_goods,
        selected_bidders,
        constraints=constraints,
    )
    economic = (
        sample_economic_report(
            spec,
            selected_goods,
            selected_bidders,
            constraints=constraints,
        )
        if include_economic
        else None
    )
    violations = list(structural["violations"])
    if economic is not None:
        violations.extend(economic["violations"])
    return {
        "selected_goods": list(selected_goods),
        "selected_bidders": list(selected_bidders),
        "structural": structural,
        "economic": economic,
        "violations": violations,
        "passed": not violations,
    }


def bidder_monetary_semantic_violations(
    profile: BidderProfileSpec,
    *,
    num_goods: int,
) -> list[str]:
    """Return unit/scale inconsistencies in one generated preference profile."""
    violations: list[str] = []
    low_budget, high_budget = profile.budget_range
    if low_budget < 0 or high_budget <= 0:
        violations.append("budget_range must contain non-negative dollar values")
    if profile.budget_cap is not None and not (
        low_budget <= profile.budget_cap <= high_budget
    ):
        violations.append(
            f"budget_cap {profile.budget_cap:g} must lie within budget_range "
            f"[{low_budget:g}, {high_budget:g}]"
        )
    if profile.saturation_start is not None:
        if profile.saturation_start < 1:
            violations.append("saturation_start must be at least 1")
        if profile.saturation_start < num_goods:
            positive_values = sorted(
                value for value in profile.base_values.values() if value > 0
            )
            midpoint = len(positive_values) // 2
            median_value = (
                positive_values[midpoint]
                if len(positive_values) % 2
                else (
                    positive_values[midpoint - 1]
                    + positive_values[midpoint]
                ) / 2
            )
            minimum_meaningful_penalty = max(1.0, median_value * 0.01)
            if profile.saturation_penalty < minimum_meaningful_penalty:
                violations.append(
                    f"saturation_penalty {profile.saturation_penalty:g} is "
                    "not a meaningful absolute-dollar penalty; expected at "
                    f"least {minimum_meaningful_penalty:g} for this value scale"
                )
    return violations


def bidder_interest_density_violations(
    profile: BidderProfileSpec,
    *,
    num_goods: int,
    constraints: dict[str, Any] | None = None,
) -> list[str]:
    """Check bidder-level sparsity while allowing explicit role exceptions."""
    rules = constraints or {}
    by_stratum = rules.get(
        "max_positive_goods_per_bidder_by_stratum", {}
    )
    overrides = rules.get("max_positive_goods_per_bidder_overrides", {})
    limit = int(
        overrides.get(
            profile.bidder_id,
            by_stratum.get(
                profile.archetype_category or "unclassified",
                num_goods,
            ),
        )
    )
    positive_count = sum(value > 0 for value in profile.base_values.values())
    if positive_count <= limit:
        return []
    return [
        f"has positive value for {positive_count}/{num_goods} goods "
        f"(maximum {limit} for this bidder/stratum)"
    ]


def population_coverage_report(
    spec: ScenarioProfileSpec,
    *,
    bidder_strata: dict[str, str] | None = None,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate master-population interest redundancy and group diversity."""
    rules = constraints or {}
    strata = bidder_strata or {
        bidder.bidder_id: bidder.archetype_category or "unclassified"
        for bidder in spec.bidder_profiles
    }
    per_good: dict[str, dict[str, Any]] = {}
    violations: list[str] = []
    min_positive = int(rules.get("min_positive_bidders_per_good", 1))
    max_positive = int(
        rules.get("max_positive_bidders_per_good", len(spec.bidder_profiles))
    )
    min_core = int(rules.get("min_core_bidders_per_good", 0))
    min_strata = int(rules.get("min_strata_with_positive_value_per_good", 1))
    monetary_semantic_violations: dict[str, list[str]] = {}
    interest_density_violations: dict[str, list[str]] = {}
    if rules.get("enforce_monetary_semantics", False):
        for bidder in spec.bidder_profiles:
            bidder_violations = bidder_monetary_semantic_violations(
                bidder,
                num_goods=len(spec.goods),
            )
            if bidder_violations:
                monetary_semantic_violations[bidder.bidder_id] = (
                    bidder_violations
                )
                violations.extend(
                    f"bidder {bidder.bidder_id}: {item}"
                    for item in bidder_violations
                )
    for bidder in spec.bidder_profiles:
        bidder_violations = bidder_interest_density_violations(
            bidder,
            num_goods=len(spec.goods),
            constraints=rules,
        )
        if bidder_violations:
            interest_density_violations[bidder.bidder_id] = bidder_violations
            violations.extend(
                f"bidder {bidder.bidder_id}: {item}"
                for item in bidder_violations
            )
    population_interest_density = (
        sum(
            value > 0
            for bidder in spec.bidder_profiles
            for value in bidder.base_values.values()
        )
        / (len(spec.goods) * len(spec.bidder_profiles))
        if spec.goods and spec.bidder_profiles
        else 0.0
    )
    max_population_interest_density = float(
        rules.get("max_population_interest_density", 1.0)
    )
    if population_interest_density > max_population_interest_density:
        violations.append(
            f"population bidder-good interest density is "
            f"{population_interest_density:.3f} "
            f"(maximum {max_population_interest_density:.3f})"
        )

    for good in (item.id for item in spec.goods):
        positive = [
            bidder.bidder_id
            for bidder in spec.bidder_profiles
            if bidder.base_values.get(good, 0.0) > 0
        ]
        core = [
            bidder.bidder_id
            for bidder in spec.bidder_profiles
            if good in bidder.core_items
        ]
        positive_strata = sorted({strata.get(bidder, "unclassified") for bidder in positive})
        per_good[good] = {
            "positive_bidders": positive,
            "core_bidders": core,
            "positive_strata": positive_strata,
        }
        if len(positive) < min_positive:
            violations.append(
                f"good {good} has {len(positive)} positive bidders "
                f"(minimum {min_positive})"
            )
        if len(positive) > max_positive:
            violations.append(
                f"good {good} has {len(positive)} positive bidders "
                f"(maximum {max_positive})"
            )
        if len(core) < min_core:
            violations.append(
                f"good {good} is core for {len(core)} bidders (minimum {min_core})"
            )
        if len(positive_strata) < min_strata:
            violations.append(
                f"good {good} spans {len(positive_strata)} strata "
                f"(minimum {min_strata})"
            )

    profiles_with_substitutes = sum(
        bool(bidder.substitute_groups) for bidder in spec.bidder_profiles
    )
    profiles_with_complements = sum(
        bool(bidder.complement_groups) for bidder in spec.bidder_profiles
    )
    nonpositive_substitute_items: dict[str, list[str]] = {}
    for bidder in spec.bidder_profiles:
        invalid = sorted(
            {
                good_id
                for group in bidder.substitute_groups
                for good_id in group.items
                if bidder.base_values.get(good_id, 0.0) <= 0
            }
        )
        if invalid:
            nonpositive_substitute_items[bidder.bidder_id] = invalid
            violations.append(
                f"bidder {bidder.bidder_id} has non-positive substitute "
                f"goods {invalid}"
            )
    min_sub_profiles = int(rules.get("min_profiles_with_substitute_groups", 0))
    min_comp_profiles = int(rules.get("min_profiles_with_complement_groups", 0))
    if profiles_with_substitutes < min_sub_profiles:
        violations.append(
            f"{profiles_with_substitutes} profiles have substitutes "
            f"(minimum {min_sub_profiles})"
        )
    if profiles_with_complements < min_comp_profiles:
        violations.append(
            f"{profiles_with_complements} profiles have complements "
            f"(minimum {min_comp_profiles})"
        )

    goods_by_id = {good.id: good for good in spec.goods}
    multiway_categories = set()
    for bidder in spec.bidder_profiles:
        for group in bidder.substitute_groups:
            if len(group.items) < 3:
                continue
            categories = {
                goods_by_id[item].category
                for item in group.items
                if item in goods_by_id
            }
            if len(categories) == 1:
                multiway_categories.update(category for category in categories if category)
    required_multiway = set(
        rules.get("required_multiway_substitute_categories", [])
    )
    missing_multiway = sorted(required_multiway - multiway_categories)
    if missing_multiway:
        violations.append(
            f"missing multiway substitute coverage for categories {missing_multiway}"
        )

    choose_one_backup_max = rules.get("choose_one_backup_factor_max", 0.0)
    multiple_backup_min = rules.get(
        "can_use_multiple_backup_factor_min", 0.0
    )
    high_backup_min = rules.get("reseller_substitute_backup_factor_min")
    high_backup_ids = set(rules.get("high_backup_bidder_ids", []))
    backup_factors_by_bidder = {
        bidder.bidder_id: [
            group.backup_factor for group in bidder.substitute_groups
        ]
        for bidder in spec.bidder_profiles
    }
    acquisition_modes_by_bidder = {
        bidder.bidder_id: [
            group.acquisition_mode for group in bidder.substitute_groups
        ]
        for bidder in spec.bidder_profiles
    }
    for bidder in spec.bidder_profiles:
        for group in bidder.substitute_groups:
            if (
                group.acquisition_mode == "choose_one"
                and group.backup_factor > choose_one_backup_max
            ):
                violations.append(
                    f"bidder {bidder.bidder_id} choose_one group "
                    f"{group.items} has backup_factor "
                    f"{group.backup_factor} above {choose_one_backup_max}"
                )
            if (
                group.acquisition_mode == "can_use_multiple"
                and group.backup_factor <= multiple_backup_min
            ):
                violations.append(
                    f"bidder {bidder.bidder_id} can_use_multiple group "
                    f"{group.items} has backup_factor "
                    f"{group.backup_factor} at/below {multiple_backup_min}"
                )
    if high_backup_min is not None:
        profiles_by_id = {
            bidder.bidder_id: bidder for bidder in spec.bidder_profiles
        }
        for bidder_id in sorted(high_backup_ids):
            if bidder_id not in profiles_by_id:
                violations.append(
                    f"high-backup bidder {bidder_id} is absent"
                )
                continue
            groups = profiles_by_id[bidder_id].substitute_groups
            if not groups:
                violations.append(
                    f"high-backup bidder {bidder_id} has no substitute groups"
                )
            elif any(
                group.acquisition_mode != "can_use_multiple"
                or group.backup_factor < high_backup_min
                for group in groups
            ):
                violations.append(
                    f"high-backup bidder {bidder_id} must use only "
                    f"can_use_multiple groups with factors at least "
                    f"{high_backup_min}"
                )

    return {
        "num_goods": len(spec.goods),
        "num_bidders": len(spec.bidder_profiles),
        "per_good": per_good,
        "profiles_with_substitute_groups": profiles_with_substitutes,
        "profiles_with_complement_groups": profiles_with_complements,
        "nonpositive_substitute_items": nonpositive_substitute_items,
        "multiway_substitute_categories": sorted(multiway_categories),
        "backup_factors_by_bidder": backup_factors_by_bidder,
        "acquisition_modes_by_bidder": acquisition_modes_by_bidder,
        "monetary_semantic_violations": monetary_semantic_violations,
        "interest_density_violations": interest_density_violations,
        "population_interest_density": population_interest_density,
        "max_population_interest_density": max_population_interest_density,
        "violations": violations,
        "passed": not violations,
    }


def _round_robin_order(
    ids: Sequence[str],
    category: Callable[[str], str],
    rng: random.Random,
) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for item_id in ids:
        groups[category(item_id)].append(item_id)
    for members in groups.values():
        rng.shuffle(members)
    category_order = list(groups)
    rng.shuffle(category_order)
    result: list[str] = []
    while any(groups.values()):
        for label in category_order:
            if groups[label]:
                result.append(groups[label].pop())
    return result


def _candidate_goods_order(
    spec: ScenarioProfileSpec,
    rng: random.Random,
) -> list[str]:
    goods_by_id = {good.id: good for good in spec.goods}
    groups: dict[str, list[str]] = defaultdict(list)
    for good in spec.goods:
        groups[good.category or "uncategorized"].append(good.id)
    multi = [category for category, members in groups.items() if len(members) >= 2]
    if not multi:
        result = [good.id for good in spec.goods]
        rng.shuffle(result)
        return result

    focus = rng.choice(multi)
    rng.shuffle(groups[focus])
    result = groups[focus][:2]
    remaining = [
        good_id
        for good_id in _round_robin_order(
            [good.id for good in spec.goods if good.id not in result],
            lambda item_id: goods_by_id[item_id].category or "uncategorized",
            rng,
        )
        if good_id not in result
    ]
    return result + remaining


def _candidate_bidders_order(
    spec: ScenarioProfileSpec,
    rng: random.Random,
) -> list[str]:
    bidders_by_id = {bidder.bidder_id: bidder for bidder in spec.bidder_profiles}
    return _round_robin_order(
        list(bidders_by_id),
        lambda bidder_id: (
            bidders_by_id[bidder_id].archetype_category or "unclassified"
        ),
        rng,
    )


@dataclass(frozen=True)
class NestedPopulationOrder:
    goods: tuple[str, ...]
    bidders: tuple[str, ...]
    attempts: int


def freeze_validated_nested_orders(
    spec: ScenarioProfileSpec,
    validation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Convert accepted validation prefixes into reusable full orders."""
    if not validation.get("passed"):
        raise ValueError("cannot freeze nested orders from failed validation")
    cases_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for case in validation.get("cases", []):
        cases_by_seed[int(case["seed"])].append(case)
    all_goods = [good.id for good in spec.goods]
    all_bidders = [bidder.bidder_id for bidder in spec.bidder_profiles]
    frozen: dict[str, dict[str, Any]] = {}
    for seed, cases in sorted(cases_by_seed.items()):
        longest_goods = max(
            (case["selected_goods"] for case in cases),
            key=len,
        )
        longest_bidders = max(
            (case["selected_bidders"] for case in cases),
            key=len,
        )
        goods_order = list(longest_goods) + [
            good for good in all_goods if good not in set(longest_goods)
        ]
        bidders_order = list(longest_bidders) + [
            bidder
            for bidder in all_bidders
            if bidder not in set(longest_bidders)
        ]
        frozen[str(seed)] = {
            "goods": goods_order,
            "bidders": bidders_order,
            "attempts": int(cases[0]["order_attempts"]),
            "validated_max_goods": len(longest_goods),
            "validated_max_bidders": len(longest_bidders),
        }
    return frozen


def _feasible_smallest_cores(
    spec: ScenarioProfileSpec,
    *,
    size: int,
    constraints: dict[str, Any] | None,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Enumerate structural cores at the smallest scalability size.

    Starting with a valid core makes the nested search substantially more
    efficient and avoids imposing category/stratum patterns that can be
    incompatible with the actual substitute and complement hypergraph.
    """
    goods = [good.id for good in spec.goods]
    bidders = [bidder.bidder_id for bidder in spec.bidder_profiles]
    if size > len(goods) or size > len(bidders):
        return []

    # When checking a goods core against all bidders, do not require every
    # population member to value that core: only the subsequently selected
    # bidder subset needs to satisfy per-bidder interest.
    goods_prefilter_constraints = {
        **(constraints or {}),
        "min_positive_goods_per_bidder": 0,
    }
    # Density is an acceptance property of the completed nested order. Using
    # it to alter the enumerated core list remaps the rejection-sampling
    # sequence for a seed and can make known acceptable orders practically
    # unreachable. Keep core enumeration stable, then enforce density in the
    # full-cell checks performed by coverage_aware_nested_order.
    goods_prefilter_constraints.pop(
        "max_mean_interest_density_by_goods", None
    )
    goods_prefilter_constraints.pop(
        "min_fraction_bidders_with_exclusions", None
    )
    core_constraints = dict(constraints or {})
    core_constraints.pop("max_mean_interest_density_by_goods", None)
    core_constraints.pop("min_fraction_bidders_with_exclusions", None)
    feasible: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for goods_core in combinations(goods, size):
        if not sample_structure_report(
            spec,
            goods_core,
            bidders,
            constraints=goods_prefilter_constraints,
        )["passed"]:
            continue
        for bidders_core in combinations(bidders, size):
            if sample_structure_report(
                spec,
                goods_core,
                bidders_core,
                constraints=core_constraints,
            )["passed"]:
                feasible.append((goods_core, bidders_core))
    return feasible


def coverage_aware_nested_order(
    spec: ScenarioProfileSpec,
    *,
    seed: int,
    sizes: Sequence[int] = tuple(range(4, 11)),
    fixed_size: int = 8,
    constraints: dict[str, Any] | None = None,
    include_economic: bool | None = None,
    max_attempts: int = 10000,
    _smallest_cores: (
        Sequence[tuple[tuple[str, ...], tuple[str, ...]]] | None
    ) = None,
) -> NestedPopulationOrder:
    """Find deterministic nested orders passing every scalability cell.

    Economic checks are enabled automatically when the supplied constraints
    contain a winner-count or welfare-concentration rule. This keeps runtime
    selection identical to generation preflight instead of selecting a
    structurally valid order and rejecting it only afterwards.
    """
    sizes = tuple(
        size
        for size in sorted(set(sizes))
        if size <= len(spec.goods) and size <= len(spec.bidder_profiles)
    )
    if not sizes:
        raise ValueError("population is too small for the requested size grid")
    if fixed_size > len(spec.goods) or fixed_size > len(spec.bidder_profiles):
        raise ValueError("population is smaller than fixed_size")

    cells = sorted({
        (size, fixed_size) for size in sizes
    } | {
        (fixed_size, size) for size in sizes
    } | {
        (size, size) for size in sizes
    })
    rules = constraints or {}
    if include_economic is None:
        include_economic = any(
            key in rules
            for key in (
                "min_full_information_winners",
                "max_largest_winner_welfare_share",
                "max_largest_winner_welfare_share_by_goods",
            )
        )
    smallest_size = min(sizes)
    smallest_cores = (
        list(_smallest_cores)
        if _smallest_cores is not None
        else _feasible_smallest_cores(
            spec,
            size=smallest_size,
            constraints=constraints,
        )
    )
    if not smallest_cores:
        raise ValueError(
            f"no feasible {smallest_size}x{smallest_size} structural core "
            "exists under the sample constraints"
        )
    all_goods = [good.id for good in spec.goods]
    all_bidders = [bidder.bidder_id for bidder in spec.bidder_profiles]
    for attempt in range(max_attempts):
        rng = random.Random(f"coverage_stratified:{seed}:{attempt}")
        goods_core, bidders_core = rng.choice(smallest_cores)
        remaining_goods = [
            good_id for good_id in all_goods if good_id not in goods_core
        ]
        remaining_bidders = [
            bidder_id
            for bidder_id in all_bidders
            if bidder_id not in bidders_core
        ]
        goods_by_id = {good.id: good for good in spec.goods}
        bidders_by_id = {
            bidder.bidder_id: bidder for bidder in spec.bidder_profiles
        }
        remaining_goods = _round_robin_order(
            remaining_goods,
            lambda good_id: (
                goods_by_id[good_id].category or "uncategorized"
            ),
            rng,
        )
        remaining_bidders = _round_robin_order(
            remaining_bidders,
            lambda bidder_id: (
                bidders_by_id[bidder_id].archetype_category
                or "unclassified"
            ),
            rng,
        )
        goods_order = list(goods_core) + remaining_goods
        bidder_order = list(bidders_core) + remaining_bidders
        structural_passed = all(
            sample_structure_report(
                spec,
                goods_order[:num_goods],
                bidder_order[:num_bidders],
                constraints=constraints,
            )["passed"]
            for num_goods, num_bidders in cells
        )
        if not structural_passed:
            continue
        if include_economic and not all(
            sample_economic_report(
                spec,
                goods_order[:num_goods],
                bidder_order[:num_bidders],
                constraints=constraints,
            )["passed"]
            for num_goods, num_bidders in cells
        ):
            continue
        if structural_passed:
            return NestedPopulationOrder(
                goods=tuple(goods_order),
                bidders=tuple(bidder_order),
                attempts=attempt + 1,
            )
    raise ValueError(
        "could not construct a coverage-aware nested ordering that satisfies "
        f"all {len(cells)} scalability cells after {max_attempts} attempts"
    )


def validate_nested_scalability_samples(
    spec: ScenarioProfileSpec,
    *,
    seeds: Sequence[int],
    sizes: Sequence[int] = tuple(range(4, 11)),
    fixed_size: int = 8,
    constraints: dict[str, Any] | None = None,
    include_economic: bool = True,
) -> dict[str, Any]:
    """Validate every nested scalability cell and cross-seed composition."""
    normalized_sizes = sorted(set(sizes))
    smallest_cores = _feasible_smallest_cores(
        spec,
        size=min(normalized_sizes),
        constraints=constraints,
    )
    cases: list[dict[str, Any]] = []
    signatures: dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]] = (
        defaultdict(list)
    )
    violations: list[str] = []
    for seed in seeds:
        try:
            order = coverage_aware_nested_order(
                spec,
                seed=seed,
                sizes=normalized_sizes,
                fixed_size=fixed_size,
                constraints=constraints,
                include_economic=include_economic,
                _smallest_cores=smallest_cores,
            )
        except ValueError as exc:
            violations.append(f"seed {seed}: {exc}")
            continue
        run_shapes: list[tuple[str, int, int]] = []
        for size in normalized_sizes:
            if size == fixed_size:
                run_shapes.append(("anchor", fixed_size, fixed_size))
            else:
                run_shapes.extend(
                    [
                        ("goods", size, fixed_size),
                        ("bidders", fixed_size, size),
                        ("joint", size, size),
                    ]
                )
        for series, num_goods, num_bidders in run_shapes:
            case_name = (
                f"anchor_{fixed_size}x{fixed_size}"
                if series == "anchor"
                else f"{series}_{num_goods}x{num_bidders}"
            )
            selected_goods = order.goods[:num_goods]
            selected_bidders = order.bidders[:num_bidders]
            report = sample_acceptance_report(
                spec,
                selected_goods,
                selected_bidders,
                constraints=constraints,
                include_economic=include_economic,
            )
            cases.append(
                {
                    "seed": seed,
                    "case": case_name,
                    "series": series,
                    "order_attempts": order.attempts,
                    **report,
                }
            )
            signatures[case_name].append(
                (tuple(sorted(selected_goods)), tuple(sorted(selected_bidders)))
            )
            violations.extend(
                f"{case_name} seed={seed}: {item}"
                for item in report["violations"]
            )

    if len(seeds) > 1:
        for case_name, values in signatures.items():
            if len(set(values)) != len(values):
                violations.append(
                    f"{case_name}: duplicate composition across validation seeds"
                )
    return {
        "seeds": list(seeds),
        "sizes": normalized_sizes,
        "fixed_size": fixed_size,
        "include_economic": include_economic,
        "cases": cases,
        "violations": violations,
        "passed": not violations and len(cases) > 0,
    }
