"""A small hand-written instance used by demos and unit tests.

Deliberately tiny and fully enumerable, so expected allocations and payments
can be reasoned about by hand rather than trusted from a solver.
"""

from __future__ import annotations

from auctionlab.instances.base import AuctionInstance


def make_toy_instance() -> AuctionInstance:
    """
    Three-item, three-bidder toy instance used in demos and tests.
    """
    return AuctionInstance(
        items=["A", "B", "C"],
        bidder_ids=["i1", "i2", "i3"],
        valuations={
            "i1": {
                frozenset({"A", "B"}): 15.0,
                frozenset({"A"}): 10.0,
            },
            "i2": {
                frozenset({"B"}): 9.0,
                frozenset({"C"}): 7.0,
            },
            "i3": {
                frozenset({"A", "C"}): 14.0,
            },
        },
    )