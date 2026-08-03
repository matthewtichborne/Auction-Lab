"""Proxy elicitation event API.

This module defines a *separate* event layer from the mechanism-refinement
:class:`~auctionlab.proxies.base.ElicitationEvent` (which is a bundle-keyed
signal emitted by running mechanisms).  The classes here are *proxy
lifecycle* events: initialisation phases (NL question, interest map,
provisional valuations) and mechanism dispatch (submit bid, clock demand)
expressed as data objects rather than direct method calls.

Usage (example)::

    from auctionlab.proxies.events import ProxyElicitationEvent, SUBMIT_BID
    response = adapter.handle_event(
        ProxyElicitationEvent(event_type=SUBMIT_BID, bidder_id="b1")
    )
    bid = response.payload["bid"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from auctionlab.proxies.base import RefinementRecord

# ---------------------------------------------------------------------------
# Event-type string constants
# ---------------------------------------------------------------------------

# Initialisation / proxy-lifecycle events.
INITIAL_PREFERENCE_QUESTION = "initial_preference_question"
INFER_INTEREST_MAP = "infer_interest_map"
GENERATE_CANDIDATE_BUNDLES = "generate_candidate_bundles"
INFER_PROVISIONAL_VALUES = "infer_provisional_values"

# Mechanism-facing events.
SUBMIT_BID = "submit_bid"
SEALED_FEEDBACK = "sealed_feedback"
CLOCK_PRICES = "clock_prices"
CLOCK_NEAR_TIE = "clock_near_tie"
CLOCK_NEAR_ZERO_SURPLUS = "clock_near_zero_surplus"
CLOCK_DEMAND_CHANGED = "clock_demand_changed"

# Late-stage reflective natural-language elicitation. Unlike the other
# mechanism-facing events above (which target a specific bundle/value to
# refine), this event asks the proxy to use accumulated auction context to
# pose one targeted NL clarification near termination, then optionally
# issue a small number of VQ/DQ follow-ups. See
# auctionlab.llm.late_reflection for the orchestration logic; this module
# only registers the event-type constant so it participates in the same
# known-event-type bookkeeping as every other proxy event.
LATE_REFLECTION = "late_reflection"

# Set of all known event types — used by handle_event() to detect unknowns.
_KNOWN_EVENT_TYPES: frozenset[str] = frozenset({
    INITIAL_PREFERENCE_QUESTION,
    INFER_INTEREST_MAP,
    GENERATE_CANDIDATE_BUNDLES,
    INFER_PROVISIONAL_VALUES,
    SUBMIT_BID,
    SEALED_FEEDBACK,
    CLOCK_PRICES,
    CLOCK_NEAR_TIE,
    CLOCK_NEAR_ZERO_SURPLUS,
    CLOCK_DEMAND_CHANGED,
    LATE_REFLECTION,
})

# Initialisation events handled by the inner LlmInferredXorProxy.
_INIT_EVENT_TYPES: frozenset[str] = frozenset({
    INITIAL_PREFERENCE_QUESTION,
    INFER_INTEREST_MAP,
    GENERATE_CANDIDATE_BUNDLES,
    INFER_PROVISIONAL_VALUES,
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProxyElicitationEvent:
    """A proxy lifecycle or mechanism-dispatch event.

    Distinct from :class:`~auctionlab.proxies.base.ElicitationEvent`, which
    is a *mechanism-refinement* signal (clock prices, near-tie, etc.) with a
    bundle-keyed payload.  This class is a *proxy dispatch* event: it covers
    both initialisation phases (NL question, interest map, PV) and mechanism
    interactions (submit bid, clock demand, CECA step).

    ``payload`` carries event-specific parameters:

    * ``initial_preference_question``: optional ``client`` key.
    * ``infer_interest_map``: optional ``client`` key.
    * ``generate_candidate_bundles``: required ``all_items`` key;
      optional ``max_candidate_bundles``.
    * ``infer_provisional_values``: required ``candidate_bundles`` (or uses
      the adapter's stored list); optional ``client``, ``interest_map``,
      ``discount_inferred``.
    * ``submit_bid``: no payload required.
    * ``sealed_feedback``: ``elicitation_event`` (an
      :class:`~auctionlab.proxies.base.ElicitationEvent`).
    * ``clock_prices``: required ``prices`` (``dict[Item, float]``);
      optional ``top_k`` (default 1).
    """

    event_type: str
    bidder_id: str
    mechanism: str | None = None
    round_idx: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProxyResponse:
    """A typed response from :meth:`handle_event`.

    ``response_type`` mirrors the event type that produced it
    (e.g. ``"natural_language_answer"``, ``"xor_bid"``, ``"clock_demand"``).

    ``payload`` carries the primary result object(s):

    * ``natural_language_answer``: ``{"question": str, "answer": str}``
    * ``interest_map``: ``{"interest_map": LlmInterestMap}``
    * ``candidate_bundles``: ``{"candidate_bundles": list[Bundle]}``
    * ``provisional_values``: ``{"raw_values": dict[Bundle, float]}``
    * ``xor_bid``: ``{"bid": XorBid}``
    * ``refinement``: ``{}`` (check ``records`` instead)
    * ``clock_demand``: ``{"demand_response": DemandResponse}``

    ``records`` mirrors any :class:`~auctionlab.proxies.base.RefinementRecord`
    objects generated during the call. ``state_delta`` is a lightweight
    summary of what changed in the proxy's internal state.
    """

    response_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    records: list[RefinementRecord] = field(default_factory=list)
    state_delta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProxyEventLogEntry:
    """Compact log entry written by :meth:`handle_event` implementations."""

    event_type: str
    bidder_id: str
    mechanism: str | None
    round_idx: int | None
    response_type: str
    records_count: int
    state_delta: dict[str, Any] = field(default_factory=dict)
