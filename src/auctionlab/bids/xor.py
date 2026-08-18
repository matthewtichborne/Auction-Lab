"""Compact XOR bid representation.

An XOR bid is a set of atomic bundle-value pairs, and a bidder wins at most
one of them. Because the solver interprets atoms under free disposal, the
value of an assigned bundle is the largest value among atoms contained in it,
which is what allows a sparse bid to price bundles it never names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from auctionlab.auction_types import Bundle

@dataclass(frozen=True)
class XorAtomicBid:
    bundle: Bundle
    value: float


@dataclass
class XorBid:
    """
    XOR bid: max over its atomic bids for any allocated bundle.
    Interpretation: bidder derives value v if they get exactly one of the atomic bundles,
    and 0 otherwise. (Standard XOR semantics.)
    """
    bidder_id: str
    atoms: List[XorAtomicBid]

    def value_of(self, assigned: Bundle) -> float:
        """
        Reported value of assigned bundle under XOR semantics.
        """

        best = 0
        for a in self.atoms:
            if a.bundle.issubset(assigned):
                best = max(a.value, best)
        return best