"""Small, deterministic selectors used by mechanism elicitation policies.

The selectors deliberately operate only on a proxy's existing XOR support.
They never enlarge the interest-map candidate set and never call an LLM.
"""

from __future__ import annotations

from collections.abc import Iterable

from auctionlab.auction_types import Bundle, Item
from auctionlab.bids.xor import XorBid
from auctionlab.proxies.base import RefinementRecord


def correction_fraction(record: RefinementRecord) -> float:
    """Return the symmetric relative size of one exact-value correction."""
    old = float(record.old_value or 0.0)
    new = float(record.new_value)
    return abs(new - old) / max(abs(old), abs(new), 1.0)


def best_neighbour_bundle(
    bid: XorBid,
    source: Bundle,
    *,
    excluded: Iterable[Bundle] = (),
) -> Bundle | None:
    """Pick one high-value, structurally local alternative to ``source``."""
    blocked = set(excluded)
    candidates = [
        atom
        for atom in bid.atoms
        if atom.bundle
        and atom.value > 0.0
        and atom.bundle != source
        and atom.bundle not in blocked
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda atom: (
            len(atom.bundle.symmetric_difference(source)),
            -atom.value,
            len(atom.bundle),
            tuple(sorted(atom.bundle)),
        )
    )
    return candidates[0].bundle


def best_scarcity_avoiding_bundle(
    bid: XorBid,
    incumbent: Bundle | None,
    contested_goods: set[Item],
    *,
    excluded: Iterable[Bundle] = (),
    min_value_ratio: float = 0.6,
) -> Bundle | None:
    """Pick an alternative using fewer currently contested goods.

    The alternative must strictly reduce contested-good exposure relative to
    the incumbent while retaining at least ``min_value_ratio`` of its reported
    value. Among those bundles, reported value is the primary tie-break after
    exposure.
    """
    if not contested_goods:
        return None
    incumbent_bundle = incumbent or frozenset()
    incumbent_exposure = len(incumbent_bundle & contested_goods)
    if incumbent_exposure == 0:
        return None
    incumbent_value = next(
        (
            atom.value
            for atom in bid.atoms
            if atom.bundle == incumbent_bundle
        ),
        0.0,
    )
    blocked = set(excluded)
    candidates = [
        atom
        for atom in bid.atoms
        if atom.bundle
        and atom.value > 0.0
        and atom.bundle not in blocked
        and len(atom.bundle & contested_goods) < incumbent_exposure
        and (
            incumbent_value <= 0.0
            or atom.value >= min_value_ratio * incumbent_value
        )
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda atom: (
            len(atom.bundle & contested_goods),
            -atom.value,
            len(atom.bundle),
            tuple(sorted(atom.bundle)),
        )
    )
    return candidates[0].bundle


def contested_goods_from_bundles(
    bundles: Iterable[Bundle | None],
) -> set[Item]:
    """Return goods appearing in two or more non-empty bundles."""
    counts: dict[Item, int] = {}
    for bundle in bundles:
        for item in bundle or frozenset():
            counts[item] = counts.get(item, 0) + 1
    return {item for item, count in counts.items() if count >= 2}
