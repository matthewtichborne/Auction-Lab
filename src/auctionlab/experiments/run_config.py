"""Run configuration helpers: presets, validation warnings, header formatting,
and refinement-record CSV conversion.

Extracted from ``examples/run_live_llm_curated_batch.py`` so the logic is
importable and testable without running the full experiment runner.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from auctionlab.llm.logging import CallTypeStats as _CallTypeStats

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

_PRESET_STRUCTURED_6X6_DIAGNOSTIC: dict[str, Any] = {
    "scenario": ["pc_build"],
    "num_goods": 6,
    "num_bidders": 6,
    "scenario_seed": 0,
    "seed_type": "structured",
    "proxy_type": "llm",
    "ask_initial_question": True,
    "use_interest_map": True,
    "use_provisional_valuations": True,
    "top_k": [1],
    "max_bundle_size": 3,
    "sealed_elicitation_rounds": 3,
    # all_valued_bundles queries every positive-value atom (highest first),
    # which surfaces large complement-group bundles that 'competitive' misses
    # when PV underestimates synergy and displacement cost turns negative.
    "sealed_feedback_rule": "all_valued_bundles",
    "elicited_clock": True,
    "max_refinement_queries_per_bidder": 3,
    "max_rounds": 30,      # extra rounds help catch near-tie routing decisions
    "clock_tie_threshold": 50.0,  # tighter tie detection for bundle routing
}

PRESETS: dict[str, dict[str, Any]] = {
    "structured-6x6-diagnostic": _PRESET_STRUCTURED_6X6_DIAGNOSTIC,
}


def explicitly_set_args() -> set[str]:
    """Return argparse dest-names that were explicitly passed in sys.argv.

    Works only with ``--flag`` and ``--flag=value`` style arguments (no short
    flags), which is all this project's CLI uses.
    """
    result: set[str] = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            name = token.split("=")[0].lstrip("-").replace("-", "_")
            result.add(name)
    return result


def apply_preset(args: Any, explicitly_set: set[str]) -> list[str]:
    """Mutate ``args`` in-place with preset defaults for flags not in ``explicitly_set``.

    Returns the list of keys that were applied (for logging purposes).
    """
    preset_name: str | None = getattr(args, "preset", None)
    if not preset_name:
        return []
    preset = PRESETS.get(preset_name, {})
    applied: list[str] = []
    for key, value in preset.items():
        if key not in explicitly_set:
            setattr(args, key, value)
            applied.append(key)
    return applied


# ---------------------------------------------------------------------------
# Config validation warnings
# ---------------------------------------------------------------------------

def config_warnings(args: Any) -> list[str]:
    """Return a list of human-readable warning/note strings for the given config.

    Returns an empty list if the configuration looks self-consistent.
    """
    warnings: list[str] = []

    sealed_rounds = getattr(args, "sealed_elicitation_rounds", 0)
    feedback_rule = getattr(args, "sealed_feedback_rule", "none")
    if sealed_rounds > 0 and feedback_rule == "none":
        warnings.append(
            "WARNING: sealed elicitation rounds are enabled but "
            "sealed_feedback_rule is 'none'; sealed proxy will not receive "
            "informative refinement events — elicitation rounds are a no-op."
        )

    elicited_clock = getattr(args, "elicited_clock", False)
    max_ref = getattr(args, "max_refinement_queries_per_bidder", 0)
    if elicited_clock and max_ref == 0:
        warnings.append(
            "NOTE: proxy clock elicitation is enabled with "
            "max_refinement_queries_per_bidder=0 (unlimited). "
            "Refinement budget is bounded only by elicitation-event logic."
        )

    return warnings


# ---------------------------------------------------------------------------
# Full run-configuration header
# ---------------------------------------------------------------------------

_WIDE = "━" * 70


def format_run_config(args: Any, scenarios: list) -> list[str]:
    """Return a list of lines describing the effective run configuration.

    Suitable for printing at the start of a run with ``print("\\n".join(lines))``.
    """
    lines: list[str] = [_WIDE, "  auctionlab  ·  run configuration", _WIDE]

    preset_name: str | None = getattr(args, "preset", None)
    if preset_name:
        lines.append(f"  preset                    {preset_name}")

    provider = getattr(args, "provider", "?")
    model = getattr(args, "model", "?")
    lines.append(f"  provider / model          {provider} / {model}")

    # Scenario summary
    if len(scenarios) == 1:
        s = scenarios[0]
        md = getattr(s, "metadata", {})
        lines.append(f"  scenario                  {s.name}")
        lines.append(f"  num_goods                 {md.get('num_goods', len(s.instance.items))}")
        lines.append(f"  num_bidders               {md.get('num_bidders', len(s.instance.bidder_ids))}")
        lines.append(
            f"  scenario_seed             "
            f"{md.get('scenario_seed', getattr(args, 'scenario_seed', 0))}"
        )
    else:
        seed_type = getattr(args, "seed_type", "all")
        lines.append(f"  scenarios                 {len(scenarios)}  seed_type={seed_type}")

    lines.append(f"  seed_type                 {getattr(args, 'seed_type', 'all')}")
    lines.append("")

    # Proxy construction
    proxy_type = getattr(args, "proxy_type", "llm")
    lines.append(f"  proxy_type                {proxy_type}")
    ask_iq = getattr(args, "ask_initial_question", False)
    use_im = getattr(args, "use_interest_map", False)
    use_pv = getattr(args, "use_provisional_valuations", False)
    max_cb = getattr(args, "max_candidate_bundles", None)
    pv_tok = getattr(args, "pv_max_tokens", 1500)
    mbs = getattr(args, "max_bundle_size", 2)

    lines.append(f"  ask_initial_question      {ask_iq}")
    lines.append(f"  use_interest_map          {use_im}")
    if max_cb:
        lines.append(f"  max_candidate_bundles     {max_cb}")
    lines.append(f"  use_provisional_vals      {use_pv}")
    if use_pv:
        lines.append(f"  pv_max_tokens             {pv_tok}")
    if not use_im:
        lines.append(f"  max_bundle_size           {mbs}")
    lines.append(f"  top_k                     {getattr(args, 'top_k', [1])}")
    lines.append(f"  ground_truth_queries      {getattr(args, 'ground_truth_queries', False)}")
    lines.append("")

    # Arms
    lines.append("  arms:")
    skip = getattr(args, "skip_baselines", False)
    max_rounds = getattr(args, "max_rounds", 20)
    if not skip:
        lines.append(f"    sealed baseline         enabled")
        lines.append(f"    clock baseline          enabled  max_rounds={max_rounds}")
    else:
        lines.append("    sealed baseline         disabled")
        lines.append("    clock baseline          disabled")

    sealed_rounds = getattr(args, "sealed_elicitation_rounds", 0)
    feedback_rule = getattr(args, "sealed_feedback_rule", "none")
    max_ref = getattr(args, "max_refinement_queries_per_bidder", 0)
    ref_str = str(max_ref) if max_ref > 0 else "unlimited"
    max_total_ref = getattr(args, "max_total_refinement_queries", 0)
    if max_total_ref > 0:
        ref_str = f"{ref_str} (total cap {max_total_ref})"
    if sealed_rounds > 0:
        trajectory = getattr(args, "sealed_trajectory", True)
        lines.append(
            f"    proxy sealed            enabled"
            f"  rounds={sealed_rounds}"
            f"  feedback_rule={feedback_rule}"
            f"  max_ref={ref_str}"
            f"  trajectory={'on' if trajectory else 'off'}"
        )
    else:
        lines.append("    proxy sealed            disabled")

    elicited_clock = getattr(args, "elicited_clock", False)
    clock_tie = getattr(args, "clock_tie_threshold", 100.0)
    if elicited_clock:
        top_k = getattr(args, "top_k", [1])
        lines.append(
            f"    proxy clock             enabled"
            f"  top_k={top_k}"
            f"  max_ref={ref_str}"
            f"  max_rounds={max_rounds}"
            f"  tie_threshold={clock_tie}"
        )
    else:
        lines.append("    proxy clock             disabled")

    lines.append("")
    lines.append(f"  output                    {getattr(args, 'log_dir', '—')}")
    lines.append(_WIDE)
    return lines


# ---------------------------------------------------------------------------
# Refinement-record CSV helpers
# ---------------------------------------------------------------------------

def refinement_records_to_rows(
    scenario_name: str,
    arm: str,
    records_by_bidder: dict[str, list],
) -> list[dict[str, Any]]:
    """Convert ``refinement_records_by_bidder`` dicts to flat CSV rows.

    Each ``RefinementRecord`` yields one row with:
    scenario, arm, bidder_id, event_type, round_idx, bundle,
    old_value, new_value, value_delta, reason, query_text, response_summary.
    """
    rows: list[dict[str, Any]] = []
    for bidder_id in sorted(records_by_bidder):
        for rec in records_by_bidder[bidder_id]:
            bundle = getattr(rec, "bundle", frozenset()) or frozenset()
            bundle_str = "{" + ",".join(sorted(bundle)) + "}" if bundle else "∅"
            old = getattr(rec, "old_value", None)
            new = getattr(rec, "new_value", 0.0)
            delta = (new - old) if old is not None else None
            ri = getattr(rec, "round_idx", None)
            rows.append({
                "scenario": scenario_name,
                "arm": arm,
                "bidder_id": bidder_id,
                "mechanism": getattr(rec, "mechanism", ""),
                "event_type": getattr(rec, "event_type", "") or "",
                "round_idx": "" if ri is None else ri,
                "bundle": bundle_str,
                "old_value": "" if old is None else f"{old:.2f}",
                "new_value": f"{new:.2f}",
                "value_delta": "" if delta is None else f"{delta:.2f}",
                "reason": getattr(rec, "reason", "") or "",
                "query_text": getattr(rec, "query_text", "") or "",
                "response_summary": getattr(rec, "response_summary", "") or "",
            })
    return rows


# ---------------------------------------------------------------------------
# Token/query accounting helpers (also used in run_live_llm_curated_batch.py)
# ---------------------------------------------------------------------------

def collect_arm_stats(stats: dict) -> dict:
    """Aggregate logger stats_since_mark into per-arm token/query counters.

    ``stats`` is a ``dict[str, CallTypeStats]`` keyed by prompt type, as
    returned by :meth:`LlmCallLogger.stats_since_mark`.

    Returns:
        vq / dq: total query counts including ground-truth (GT) calls.
        gt_vq / gt_dq: GT-only query counts (zero LLM tokens, useful for
            estimating cost if queries were replaced with real LLM calls).
        tok_in / tok_out: actual LLM token totals (GT calls contribute 0).
    """
    from auctionlab.llm.logging import CallTypeStats

    vq_llm = stats.get("value_query", CallTypeStats()).calls
    vq_gt = stats.get("value_query_gt", CallTypeStats()).calls
    dq_llm = stats.get("demand_query", CallTypeStats()).calls
    dq_gt = stats.get("demand_query_gt", CallTypeStats()).calls
    nl = stats.get("nl_question", CallTypeStats()).calls
    # GT entries log 0 tokens, so this sum is already correct.
    tok_in = sum(s.input_tokens for s in stats.values())
    tok_out = sum(s.output_tokens for s in stats.values())
    return {
        "vq": vq_llm + vq_gt,
        "dq": dq_llm + dq_gt,
        "gt_vq": vq_gt,
        "gt_dq": dq_gt,
        "nl": nl,
        "tok_in": tok_in,
        "tok_out": tok_out,
    }


def collect_initial_stats(stats: dict) -> dict:
    """Aggregate stats for the shared initial elicitation phase (proxy-side calls).

    Proxy-side calls use different prompt-type keys (``proxy_nl_gen``,
    ``proxy_interest_map``, ``proxy_provisional_valuations``) compared to the
    person-side ``nl_question`` counted by :func:`collect_arm_stats`.
    """
    from auctionlab.llm.logging import CallTypeStats

    nlq = stats.get("proxy_nl_gen", CallTypeStats()).calls
    im = stats.get("proxy_interest_map", CallTypeStats()).calls
    pv = stats.get("proxy_provisional_valuations", CallTypeStats()).calls
    tok_in = sum(s.input_tokens for s in stats.values())
    tok_out = sum(s.output_tokens for s in stats.values())
    return {
        "vq": 0,
        "dq": 0,
        "nl": nlq + im + pv,
        "tok_in": tok_in,
        "tok_out": tok_out,
        "token_accounting_note": f"nl_q={nlq}  im={im}  pv={pv}",
    }
