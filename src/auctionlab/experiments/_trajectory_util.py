"""Small helpers shared by the sealed and clock trajectory recorders."""

from __future__ import annotations

from typing import Any

from auctionlab.auction_types import Bundle
from auctionlab.instances.base import AuctionInstance


def true_welfare_for_allocation(
    instance: AuctionInstance,
    allocation: dict[str, Bundle],
) -> float:
    """Ground-truth welfare of an allocation, evaluated against ``instance``'s
    true valuations -- independent of whatever bids/reports produced
    ``allocation`` in the first place. Never conflate this with a mechanism
    result's ``.welfare`` field, which (for any proxy-mediated runner) is
    *reported* welfare -- the WDP objective over reported/elicited bids,
    which can overstate true welfare when reports are miscalibrated.
    """
    return float(
        sum(
            instance.value_of(
                bidder_id,
                allocation.get(bidder_id, frozenset()),
            )
            for bidder_id in instance.bidder_ids
        )
    )

# Mirrors auctionlab.experiments.run_config.collect_arm_stats's bucketing --
# that function is what the CLI itself uses to compute the "value queries"
# / "demand queries" numbers shown in the final per-arm summary rows and
# results table, so it is the canonical source of truth these buckets must
# match. In particular this is *not* the same thing as
# `auctionlab.proxies.base.ProxyStats.value_queries`: that field only counts
# a proxy's initial lazy bid construction (see
# `LlmAuctionProxyAdapter.current_bid`, whose stats increment is guarded by
# `if not had_cached_bid`) and is deliberately excluded from later
# refinement-triggered lookups -- the CLI's own arm summary prints it under
# the label "value_queries (initial)" for exactly this reason. Refinement-
# and ground-truth-triggered value/demand lookups only show up as
# "value_query"/"value_query_gt"/"demand_query"/"demand_query_gt" call-log
# entries, which is why the logger (not ProxyStats) is the right source for
# a trajectory's aggregate query counts whenever a logger is available.
_VQ_PROMPT_TYPES = frozenset({"value_query", "value_query_gt"})
_DQ_PROMPT_TYPES = frozenset({"demand_query", "demand_query_gt"})
_NL_PROMPT_TYPES = frozenset({"nl_question"})


def logger_total_tokens(logger: Any | None) -> tuple[int, int]:
    """Cumulative (input, output) tokens logged so far, ignoring ``mark()``.

    ``logger`` is duck-typed (any object exposing ``.total_tokens()``, e.g.
    :class:`~auctionlab.llm.logging.LlmCallLogger`) so trajectory recorders
    can be nested inside a caller's own ``mark()`` window without disturbing
    it. ``None`` (no logger, e.g. toy/scripted proxies in tests) returns
    ``(0, 0)``.
    """
    if logger is None:
        return 0, 0
    return logger.total_tokens()


def logger_query_counts(logger: Any | None) -> tuple[int, int, int]:
    """Cumulative (value_query, demand_query, nl_query) *call* counts so far.

    Unlike :func:`logger_total_tokens`, this reads via
    ``logger.stats_since_mark()`` rather than ``logger.total_stats()``: query
    counts should match "since the caller's last ``mark()``" (the same
    watermark the CLI itself reports against), not "since logger creation".
    This function never calls ``mark()`` itself, so repeated calls are safe
    to make at every round boundary without disturbing anyone else's
    watermark. ``None`` (no logger) returns ``(0, 0, 0)``.
    """
    if logger is None:
        return 0, 0, 0
    stats = logger.stats_since_mark()
    vq = sum(s.calls for prompt_type, s in stats.items() if prompt_type in _VQ_PROMPT_TYPES)
    dq = sum(s.calls for prompt_type, s in stats.items() if prompt_type in _DQ_PROMPT_TYPES)
    nl = sum(s.calls for prompt_type, s in stats.items() if prompt_type in _NL_PROMPT_TYPES)
    return vq, dq, nl


def aggregate_query_counts(
    logger: Any | None,
    stats_now: dict[str, dict[str, int]],
) -> tuple[int, int, int]:
    """Cumulative (value_query, demand_query, nl_query) counts so far.

    Prefers :func:`logger_query_counts` (the canonical, refinement- and
    ground-truth-inclusive source) whenever a logger is given. Falls back to
    summing ``ProxyStats`` fields across bidders -- ``stats_now`` is a
    ``bidder_id -> {"value_queries": int, "demand_queries": int,
    "nl_queries": int, ...}`` snapshot, as produced by each trajectory
    module's own per-round ``ProxyStats`` snapshot helper -- when no logger
    is available (e.g. toy/scripted proxies in tests, which have no LLM call
    log to read from).
    """
    if logger is not None:
        return logger_query_counts(logger)
    vq = sum(s["value_queries"] for s in stats_now.values())
    dq = sum(s["demand_queries"] for s in stats_now.values())
    nl = sum(s["nl_queries"] for s in stats_now.values())
    return vq, dq, nl


def refinement_cap_fields(
    refinement_query_count_by_bidder: dict[str, int],
    *,
    max_refinements_per_bidder: int = 0,
    max_total_refinements: int = 0,
) -> dict[str, Any]:
    """Aggregate refinement-cap bookkeeping shared by every CSV row builder.

    Refinement count is meant to be an *outcome* of the elicitation events
    and mechanism, not a tuned quantity -- these caps only exist as safety
    limits against runaway query volume (see
    ``docs/parameter_tuning_methodology.md``). ``0`` means "no cap" for
    either argument. Returns:

    - ``total_refinement_queries``: summed across all bidders.
    - ``per_bidder_refinement_queries``: the input dict, unchanged (kept as
      its own key so callers/aggregators don't have to know the upstream
      dict's original key name).
    - ``cap_binding_indicator``: whether any bidder's count is at/above
      ``max_refinements_per_bidder`` (always ``False`` if that cap is 0).
    - ``safety_cap_hit``: whether the summed total is at/above
      ``max_total_refinements`` (always ``False`` if that cap is 0).
    """
    total = sum(refinement_query_count_by_bidder.values())
    per_bidder_cap_hit = (
        max_refinements_per_bidder > 0
        and any(
            count >= max_refinements_per_bidder
            for count in refinement_query_count_by_bidder.values()
        )
    )
    safety_cap_hit = (
        max_total_refinements > 0 and total >= max_total_refinements
    )
    return {
        "total_refinement_queries": total,
        "per_bidder_refinement_queries": dict(refinement_query_count_by_bidder),
        "cap_binding_indicator": per_bidder_cap_hit,
        "safety_cap_hit": safety_cap_hit,
    }
