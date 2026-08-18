"""Shared fixtures for the test suite.

Provides the toy instance and its expected allocation, so tests that exercise
mechanisms can assert against a result worked out by hand rather than one
produced by the solver under test. Real sleeps are suppressed suite-wide
because several tests deliberately trigger retried failures.
"""

from __future__ import annotations

import time

import pytest

from auctionlab.auction_types import Bundle
from auctionlab.instances.toy import make_toy_instance


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Skip real delays from retry/backoff logic (e.g. api_retry) in tests.

    Several tests deliberately trigger retried failures against a
    deterministic fake client, so the exponential backoff would otherwise
    burn real wall-clock time for no benefit.
    """
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)


@pytest.fixture
def toy_instance():
    return make_toy_instance()


@pytest.fixture
def toy_items(toy_instance):
    return toy_instance.items


@pytest.fixture
def toy_bids(toy_instance):
    return toy_instance.to_xor_bids()


@pytest.fixture
def expected_toy_allocation() -> dict[str, Bundle]:
    return {
        "i1": frozenset(),
        "i2": frozenset({"B"}),
        "i3": frozenset({"A", "C"}),
    }