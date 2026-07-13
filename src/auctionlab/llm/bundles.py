from __future__ import annotations

from itertools import combinations

from auctionlab.auction_types import Bundle, Item


def bundle_sort_key(bundle: Bundle) -> tuple[int, tuple[str, ...]]:
    return len(bundle), tuple(sorted(bundle))


def generate_candidate_bundles(
    items: list[Item],
    *,
    max_bundle_size: int,
    include_empty: bool = False,
) -> list[Bundle]:
    if max_bundle_size < 1:
        raise ValueError("max_bundle_size must be at least 1")
    if max_bundle_size > len(items):
        raise ValueError("max_bundle_size cannot exceed the number of items")
    if len(set(items)) != len(items):
        raise ValueError("items must be unique")

    sorted_items = sorted(items)
    bundles: list[Bundle] = []

    if include_empty:
        bundles.append(frozenset())

    for bundle_size in range(1, max_bundle_size + 1):
        bundles.extend(
            frozenset(bundle)
            for bundle in combinations(sorted_items, bundle_size)
        )

    return bundles
