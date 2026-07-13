from __future__ import annotations

import random
from itertools import combinations
from typing import List

from auctionlab.auction_types import Bundle, Item
from auctionlab.instances.base import AuctionInstance


def all_nonempty_bundles(
    items: List[Item],
    *,
    max_bundle_size: int,
) -> List[Bundle]:
    """
    Generate all non-empty bundles up to max_bundle_size.
    """
    if max_bundle_size < 1:
        raise ValueError("max_bundle_size must be at least 1")

    if max_bundle_size > len(items):
        raise ValueError("max_bundle_size cannot exceed number of items")

    bundles: List[Bundle] = []

    for size in range(1, max_bundle_size + 1):
        for combo in combinations(items, size):
            bundles.append(frozenset(combo))

    return bundles


def make_random_xor_instance(
    *,
    n_items: int,
    n_bidders: int,
    atoms_per_bidder: int,
    max_bundle_size: int,
    min_value: float,
    max_value: float,
    seed: int,
) -> AuctionInstance:
    """
    Generate a random XOR combinatorial auction instance.

    For each bidder:
      - sample `atoms_per_bidder` distinct non-empty bundles
      - assign each sampled bundle a uniform random value in [min_value, max_value]

    Notes:
      - This is a simple baseline generator.
      - It does not yet model substitutes, complements, budgets, or bidder types.
    """
    if n_items < 1:
        raise ValueError("n_items must be at least 1")

    if n_bidders < 1:
        raise ValueError("n_bidders must be at least 1")

    if atoms_per_bidder < 1:
        raise ValueError("atoms_per_bidder must be at least 1")

    if min_value < 0:
        raise ValueError("min_value must be non-negative")

    if max_value < min_value:
        raise ValueError("max_value must be at least min_value")

    items: List[Item] = [f"item_{idx}" for idx in range(n_items)]
    possible_bundles = all_nonempty_bundles(
        items,
        max_bundle_size=max_bundle_size,
    )

    if atoms_per_bidder > len(possible_bundles):
        raise ValueError(
            "atoms_per_bidder cannot exceed number of possible bundles"
        )

    rng = random.Random(seed)

    valuations: dict[str, dict[Bundle, float]] = {}

    for bidder_idx in range(n_bidders):
        bidder_id = f"bidder_{bidder_idx}"

        sampled_bundles = rng.sample(possible_bundles, atoms_per_bidder)

        bidder_values: dict[Bundle, float] = {}

        for bundle in sampled_bundles:
            value = rng.uniform(min_value, max_value)
            bidder_values[bundle] = round(value, 6)

        valuations[bidder_id] = bidder_values

    return AuctionInstance(
        items=items,
        bidder_ids=list(valuations.keys()),
        valuations=valuations,
    )