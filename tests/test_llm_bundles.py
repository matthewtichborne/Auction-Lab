from __future__ import annotations

import pytest

from auctionlab.llm.bundles import bundle_sort_key, generate_candidate_bundles


def test_generate_candidate_bundles_has_deterministic_order():
    assert generate_candidate_bundles(
        ["A", "B", "C"],
        max_bundle_size=2,
    ) == [
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"A", "B"}),
        frozenset({"A", "C"}),
        frozenset({"B", "C"}),
    ]


def test_generate_candidate_bundles_can_include_empty_first():
    bundles = generate_candidate_bundles(
        ["B", "A"],
        max_bundle_size=1,
        include_empty=True,
    )

    assert bundles == [
        frozenset(),
        frozenset({"A"}),
        frozenset({"B"}),
    ]


@pytest.mark.parametrize("max_bundle_size", [0, -1])
def test_generate_candidate_bundles_rejects_non_positive_size(max_bundle_size):
    with pytest.raises(ValueError, match="at least 1"):
        generate_candidate_bundles(
            ["A", "B"],
            max_bundle_size=max_bundle_size,
        )


def test_generate_candidate_bundles_rejects_size_larger_than_items():
    with pytest.raises(ValueError, match="cannot exceed"):
        generate_candidate_bundles(["A"], max_bundle_size=2)


def test_generate_candidate_bundles_rejects_duplicate_items():
    with pytest.raises(ValueError, match="unique"):
        generate_candidate_bundles(["A", "A"], max_bundle_size=1)


def test_bundle_sort_key_uses_size_then_lexicographic_items():
    bundles = [
        frozenset({"B", "C"}),
        frozenset({"B"}),
        frozenset({"A", "C"}),
    ]

    assert sorted(bundles, key=bundle_sort_key) == [
        frozenset({"B"}),
        frozenset({"A", "C"}),
        frozenset({"B", "C"}),
    ]
