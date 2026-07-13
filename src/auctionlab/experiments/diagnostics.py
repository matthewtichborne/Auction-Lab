from __future__ import annotations

from typing import Dict, List

from auctionlab.auction_types import Bundle
from auctionlab.bids.xor import XorAtomicBid
from auctionlab.instances.base import AuctionInstance


def count_full_atoms(instance: AuctionInstance) -> int:
    """
    Count all atomic valuation entries in the full instance.
    """
    return sum(
        len(bundle_values)
        for bundle_values in instance.valuations.values()
    )


def count_supplementary_atoms(
    supplementary_atoms: Dict[str, List[XorAtomicBid]],
) -> int:
    """
    Count all supplementary atomic bids collected by the clock.
    """
    return sum(len(atoms) for atoms in supplementary_atoms.values())


def supplementary_coverage_ratio(
    instance: AuctionInstance,
    supplementary_atoms: Dict[str, List[XorAtomicBid]],
) -> float:
    """
    Fraction of full atomic valuations observed as supplementary atoms.

    This treats coverage as exact (bidder_id, bundle) coverage.
    """
    full_atoms = count_full_atoms(instance)

    if full_atoms == 0:
        return 1.0

    observed = 0

    for bidder_id, bundle_values in instance.valuations.items():
        observed_bundles = {
            atom.bundle
            for atom in supplementary_atoms.get(bidder_id, [])
        }

        for bundle in bundle_values:
            if bundle in observed_bundles:
                observed += 1

    return observed / full_atoms


def sealed_winning_bundles_observed(
    sealed_allocation: Dict[str, Bundle],
    supplementary_atoms: Dict[str, List[XorAtomicBid]],
) -> bool:
    """
    Return True if every non-empty sealed-winning bundle appears in the
    corresponding bidder's supplementary atoms.
    """
    for bidder_id, winning_bundle in sealed_allocation.items():
        if not winning_bundle:
            continue

        observed_bundles = {
            atom.bundle
            for atom in supplementary_atoms.get(bidder_id, [])
        }

        if winning_bundle not in observed_bundles:
            return False

    return True


def missing_sealed_winning_bundles(
    sealed_allocation: Dict[str, Bundle],
    supplementary_atoms: Dict[str, List[XorAtomicBid]],
) -> Dict[str, Bundle]:
    """
    Return bidder_id -> sealed-winning bundle for non-empty sealed-winning
    bundles that were not collected during the clock.
    """
    missing: Dict[str, Bundle] = {}

    for bidder_id, winning_bundle in sealed_allocation.items():
        if not winning_bundle:
            continue

        observed_bundles = {
            atom.bundle
            for atom in supplementary_atoms.get(bidder_id, [])
        }

        if winning_bundle not in observed_bundles:
            missing[bidder_id] = winning_bundle

    return missing