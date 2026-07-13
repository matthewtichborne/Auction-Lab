from __future__ import annotations

import pytest

from auctionlab.instances.random import (
    all_nonempty_bundles,
    make_random_xor_instance,
)


def test_all_nonempty_bundles_generates_expected_bundles():
    items = ["A", "B", "C"]

    bundles = all_nonempty_bundles(items, max_bundle_size=2)

    assert set(bundles) == {
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"A", "B"}),
        frozenset({"A", "C"}),
        frozenset({"B", "C"}),
    }


def test_random_instance_has_expected_shape():
    instance = make_random_xor_instance(
        n_items=4,
        n_bidders=3,
        atoms_per_bidder=5,
        max_bundle_size=2,
        min_value=1.0,
        max_value=10.0,
        seed=123,
    )

    assert instance.items == ["item_0", "item_1", "item_2", "item_3"]
    assert instance.bidder_ids == ["bidder_0", "bidder_1", "bidder_2"]

    for bidder_id in instance.bidder_ids:
        bidder_values = instance.valuations[bidder_id]

        assert len(bidder_values) == 5

        for bundle, value in bidder_values.items():
            assert 1 <= len(bundle) <= 2
            assert bundle.issubset(set(instance.items))
            assert 1.0 <= value <= 10.0


def test_random_instance_is_reproducible_with_same_seed():
    instance_a = make_random_xor_instance(
        n_items=4,
        n_bidders=3,
        atoms_per_bidder=5,
        max_bundle_size=2,
        min_value=1.0,
        max_value=10.0,
        seed=123,
    )

    instance_b = make_random_xor_instance(
        n_items=4,
        n_bidders=3,
        atoms_per_bidder=5,
        max_bundle_size=2,
        min_value=1.0,
        max_value=10.0,
        seed=123,
    )

    assert instance_a == instance_b


def test_random_instance_rejects_too_many_atoms():
    with pytest.raises(ValueError):
        make_random_xor_instance(
            n_items=2,
            n_bidders=2,
            atoms_per_bidder=4,
            max_bundle_size=1,
            min_value=1.0,
            max_value=10.0,
            seed=123,
        )