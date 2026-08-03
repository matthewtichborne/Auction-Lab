"""Run configuration helpers: policies, validation warnings, header formatting,
and refinement-record CSV conversion.

Extracted from ``examples/run_live_llm_curated_batch.py`` so the logic is
importable and testable without running the full experiment runner.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

from auctionlab.llm.value_calibration import NO_CALIBRATION, ValueCalibration
from auctionlab.payments.vcg import vcg_witness_count

if TYPE_CHECKING:
    from auctionlab.llm.logging import CallTypeStats as _CallTypeStats

EVENT_POLICIES = ("custom", "recommended", "final-v3")

# Frozen after the five-seed 8x8 ablation.  Keep this mapping explicit rather
# than reconstructing it from CLI defaults: it is part of the experimental
# treatment, not merely a convenience configuration.
SEALED_EVENT_POLICY_V1: dict[str, Any] = {
    "feedback_rule": "competitive",
    "loser_challenger_policy": "off",
    "incumbent_verification": True,
    "pivotal_challengers": False,
    "scarcity_fallbacks": True,
    "large_correction_followup": True,
    "terminal_regret_audit": False,
}

# Frozen after the final five-seed clock ablation. Price discovery runs to
# completion without exact queries; a single terminal pass verifies only VCG
# witness bundles observed in the top-k clock demand trajectory.
CLOCK_EVENT_POLICY_FRONTIER_V2: dict[str, Any] = {
    "framework": "frontier_v1",
    "supplementary_support_policy": "all_atoms",
    "incumbent_verification": False,
    "allocation_change_audit": False,
    "allocation_counterfactual_frontier": False,
    "terminal_stability_audit": False,
    "top_k_frontier_policy": "off",
    "scarcity_fallbacks": False,
    "large_correction_followup": False,
    "additional_pivotal_challengers": False,
    "gate_near_zero_surplus": False,
    "terminal_regret_audit": False,
    "demand_switch_verification": False,
    "contested_bundle_refinement": False,
    "terminal_winner_verification": False,
    "terminal_vcg_witness_verification": False,
    "terminal_best_losing_challenger": False,
    "native_near_zero_surplus": False,
    "native_demand_changed": False,
    "native_near_tie": False,
    "frontier_winner_verification": False,
    "frontier_pivotal_challengers": False,
    "frontier_winner_closure": False,
    "frontier_vcg_witness_verification": False,
    "frontier_vcg_single_pass": True,
    "frontier_vcg_revealed_only": True,
    "frontier_staged_revealed_vcg_closure": False,
}

# Selected after the focused five-seed 8x8 closure ablation.  The clock first
# verifies bidder-removal witnesses seen on its demand path, then closes newly
# winning allocations, and finally alternates revealed witnesses and winner
# closure until no unqueried bidder/bundle pair remains.
CLOCK_EVENT_POLICY_SANDWICH_V3: dict[str, Any] = {
    **CLOCK_EVENT_POLICY_FRONTIER_V2,
    "frontier_winner_closure": True,
    "frontier_vcg_single_pass": True,
    "frontier_vcg_revealed_only": True,
    "frontier_staged_revealed_vcg_closure": True,
}

_RECOMMENDED_POLICY_CONFLICTS = {
    "event_incumbent_verification",
    "event_pivotal_challengers",
    "event_scarcity_fallbacks",
    "event_large_correction_followup",
    "sealed_event_large_correction_followup",
    "clock_event_large_correction_followup",
    "event_gate_near_zero_surplus",
    "event_terminal_regret_audit",
    "sealed_feedback_rule",
    "sealed_loser_challenger_policy",
    "clock_top_k_frontier_policy",
    "clock_refine_top_k_frontier",
    "clock_allocation_counterfactual_frontier",
    "clock_event_framework",
    "clock_supplementary_support_policy",
    "clock_event_demand_switch_verification",
    "clock_event_contested_bundle_refinement",
    "clock_event_terminal_winner_verification",
    "clock_event_terminal_vcg_witness_verification",
    "clock_event_terminal_best_losing_challenger",
    "clock_event_terminal_stability_audit",
    "clock_native_near_zero_surplus",
    "clock_native_demand_changed",
    "clock_native_near_tie",
    "clock_frontier_winner_verification",
    "clock_frontier_pivotal_challengers",
    "clock_frontier_winner_closure",
    "clock_frontier_vcg_witness_verification",
    "clock_frontier_vcg_single_pass",
    "clock_frontier_vcg_revealed_only",
    "clock_frontier_staged_revealed_vcg_closure",
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
            if name.startswith("no_"):
                name = name[3:]
            result.add(name)
    return result


def resolve_event_policy(
    args: Any,
    explicitly_set: set[str] | None = None,
) -> dict[str, Any]:
    """Resolve the mechanism-specific elicitation-event configuration.

    ``custom`` preserves the historical granular CLI flags. ``recommended``
    is the fixed primary specification selected for evaluation after the 8x8
    diagnostic: incumbent and winner-removal verification, scarcity fallbacks
    in both mechanisms, sealed-only large-correction follow-up, and the
    existing clock terminal stability audit. It deliberately excludes the
    additional pivotal, near-zero gating, and terminal-regret treatments.
    """
    explicitly_set = explicitly_set or set()
    policy = getattr(args, "event_policy", "custom")
    if policy not in EVENT_POLICIES:
        raise ValueError(
            f"event_policy must be one of {EVENT_POLICIES}, got {policy!r}"
        )

    if policy in {"recommended", "final-v3"}:
        conflicts = sorted(explicitly_set & _RECOMMENDED_POLICY_CONFLICTS)
        if conflicts:
            rendered = ", ".join(
                f"--{name.replace('_', '-')}" for name in conflicts
            )
            raise ValueError(
                f"--event-policy {policy} is a fixed specification and "
                f"cannot be combined with granular policy flags: {rendered}. "
                "Use --event-policy custom for an ablation or override."
            )
        args.sealed_feedback_rule = SEALED_EVENT_POLICY_V1["feedback_rule"]
        args.sealed_loser_challenger_policy = SEALED_EVENT_POLICY_V1[
            "loser_challenger_policy"
        ]
        args.event_incumbent_verification = True
        args.event_pivotal_challengers = False
        args.event_large_correction_followup = False
        args.event_gate_near_zero_surplus = False
        args.event_terminal_regret_audit = False
        args.sealed_event_large_correction_followup = True
        args.clock_event_large_correction_followup = False
        if policy == "recommended":
            args.clock_event_framework = "legacy"
            args.clock_top_k_frontier_policy = "allocation_pivotal"
            args.clock_allocation_counterfactual_frontier = True
            args.event_scarcity_fallbacks = True
            args.clock_event_demand_switch_verification = False
            args.clock_event_contested_bundle_refinement = False
            args.clock_event_terminal_winner_verification = True
            args.clock_event_terminal_vcg_witness_verification = True
            args.clock_event_terminal_best_losing_challenger = False
        else:
            clock = CLOCK_EVENT_POLICY_SANDWICH_V3
            args.clock_event_framework = clock["framework"]
            args.clock_supplementary_support_policy = clock[
                "supplementary_support_policy"
            ]
            args.clock_top_k_frontier_policy = clock[
                "top_k_frontier_policy"
            ]
            args.clock_allocation_counterfactual_frontier = False
            # Shared switches remain enabled where required by the sealed arm;
            # mechanism-specific resolved clock values below override them.
            args.event_scarcity_fallbacks = True
            args.clock_event_demand_switch_verification = False
            args.clock_event_contested_bundle_refinement = False
            args.clock_event_terminal_winner_verification = False
            args.clock_event_terminal_vcg_witness_verification = False
            args.clock_event_terminal_best_losing_challenger = False
            args.clock_event_terminal_stability_audit = False
            args.clock_native_near_zero_surplus = False
            args.clock_native_demand_changed = False
            args.clock_native_near_tie = False
            args.clock_frontier_winner_verification = False
            args.clock_frontier_pivotal_challengers = False
            args.clock_frontier_winner_closure = clock[
                "frontier_winner_closure"
            ]
            args.clock_frontier_vcg_witness_verification = False
            args.clock_frontier_vcg_single_pass = True
            args.clock_frontier_vcg_revealed_only = True
            args.clock_frontier_staged_revealed_vcg_closure = clock[
                "frontier_staged_revealed_vcg_closure"
            ]
    else:
        shared_large_correction = bool(
            getattr(args, "event_large_correction_followup", False)
        )
        sealed_override = getattr(
            args, "sealed_event_large_correction_followup", None
        )
        clock_override = getattr(
            args, "clock_event_large_correction_followup", None
        )
        args.sealed_event_large_correction_followup = (
            shared_large_correction
            if sealed_override is None
            else bool(sealed_override)
        )
        args.clock_event_large_correction_followup = (
            shared_large_correction
            if clock_override is None
            else bool(clock_override)
        )

    clock_framework = getattr(args, "clock_event_framework", "legacy")
    targeted_clock = clock_framework == "targeted_v1"

    resolved = {
        "name": policy,
        "sealed": {
            "feedback_rule": getattr(args, "sealed_feedback_rule", "none"),
            "loser_challenger_policy": getattr(
                args, "sealed_loser_challenger_policy", "off"
            ),
            "incumbent_verification": bool(
                getattr(args, "event_incumbent_verification", True)
            ),
            "winner_removal_counterfactuals": (
                getattr(args, "sealed_feedback_rule", "none") == "competitive"
            ),
            "scarcity_fallbacks": bool(
                getattr(args, "event_scarcity_fallbacks", False)
            ),
            "large_correction_followup": bool(
                args.sealed_event_large_correction_followup
            ),
            "correction_followup_threshold": float(
                getattr(args, "event_correction_threshold", 0.25)
            ),
            "additional_pivotal_challengers": bool(
                getattr(args, "event_pivotal_challengers", False)
            ),
            "terminal_regret_audit": bool(
                getattr(args, "event_terminal_regret_audit", False)
            ),
        },
        "clock": {
            "framework": clock_framework,
            "supplementary_support_policy": getattr(
                args, "clock_supplementary_support_policy", "all_atoms"
            ),
            "native_near_zero_surplus": bool(
                getattr(args, "clock_native_near_zero_surplus", True)
            ),
            "native_demand_changed": bool(
                getattr(args, "clock_native_demand_changed", True)
            ),
            "native_near_tie": bool(
                getattr(args, "clock_native_near_tie", True)
            ),
            "incumbent_verification": bool(
                getattr(args, "event_incumbent_verification", True)
            ),
            "allocation_change_audit": bool(
                getattr(args, "event_incumbent_verification", True)
            ),
            "allocation_counterfactual_frontier": bool(
                getattr(args, "clock_allocation_counterfactual_frontier", False)
            ),
            "terminal_stability_audit": bool(
                getattr(args, "clock_event_terminal_stability_audit", None)
                if getattr(args, "clock_event_terminal_stability_audit", None)
                is not None
                else getattr(args, "event_incumbent_verification", True)
            ),
            "top_k_frontier_policy": getattr(
                args, "clock_top_k_frontier_policy", "off"
            ),
            "scarcity_fallbacks": bool(
                getattr(args, "event_scarcity_fallbacks", False)
            ),
            "large_correction_followup": bool(
                args.clock_event_large_correction_followup
            ),
            "additional_pivotal_challengers": bool(
                getattr(args, "event_pivotal_challengers", False)
            ),
            "gate_near_zero_surplus": bool(
                getattr(args, "event_gate_near_zero_surplus", False)
            ),
            "terminal_regret_audit": bool(
                getattr(args, "event_terminal_regret_audit", False)
            ),
            "demand_switch_verification": bool(
                getattr(
                    args,
                    "clock_event_demand_switch_verification",
                    targeted_clock,
                )
            ),
            "contested_bundle_refinement": bool(
                getattr(
                    args,
                    "clock_event_contested_bundle_refinement",
                    targeted_clock,
                )
            ),
            "terminal_winner_verification": bool(
                getattr(
                    args,
                    "clock_event_terminal_winner_verification",
                    True,
                )
            ),
            "terminal_vcg_witness_verification": bool(
                getattr(
                    args,
                    "clock_event_terminal_vcg_witness_verification",
                    True,
                )
            ),
            "terminal_best_losing_challenger": bool(
                getattr(
                    args,
                    "clock_event_terminal_best_losing_challenger",
                    False,
                )
            ),
            "frontier_winner_verification": bool(
                getattr(args, "clock_frontier_winner_verification", False)
            ),
            "frontier_pivotal_challengers": bool(
                getattr(args, "clock_frontier_pivotal_challengers", False)
            ),
            "frontier_winner_closure": bool(
                getattr(args, "clock_frontier_winner_closure", False)
            ),
            "frontier_vcg_witness_verification": bool(
                getattr(
                    args,
                    "clock_frontier_vcg_witness_verification",
                    False,
                )
            ),
            "frontier_vcg_single_pass": bool(
                getattr(args, "clock_frontier_vcg_single_pass", False)
            ),
            "frontier_vcg_revealed_only": bool(
                getattr(args, "clock_frontier_vcg_revealed_only", False)
            ),
            "frontier_staged_revealed_vcg_closure": bool(
                getattr(
                    args,
                    "clock_frontier_staged_revealed_vcg_closure",
                    False,
                )
            ),
        },
    }
    if policy == "final-v3":
        resolved["clock"].update(CLOCK_EVENT_POLICY_SANDWICH_V3)
    args.resolved_event_policy = resolved
    return resolved


def event_policy_summary_fields(args: Any) -> dict[str, Any]:
    """Flat resolved-policy fields for run-summary CSV rows."""
    resolved = getattr(args, "resolved_event_policy", None)
    if resolved is None:
        resolved = resolve_event_policy(args)
    sealed = resolved["sealed"]
    clock = resolved["clock"]
    return {
        "event_policy": resolved["name"],
        "sealed_event_incumbent_verification": sealed[
            "incumbent_verification"
        ],
        "sealed_event_winner_removal_counterfactuals": sealed[
            "winner_removal_counterfactuals"
        ],
        "sealed_event_scarcity_fallbacks": sealed["scarcity_fallbacks"],
        "sealed_event_large_correction_followup": sealed[
            "large_correction_followup"
        ],
        "clock_event_incumbent_verification": clock[
            "incumbent_verification"
        ],
        "clock_event_framework": clock["framework"],
        "clock_supplementary_support_policy": clock[
            "supplementary_support_policy"
        ],
        "clock_native_near_zero_surplus": clock[
            "native_near_zero_surplus"
        ],
        "clock_native_demand_changed": clock["native_demand_changed"],
        "clock_native_near_tie": clock["native_near_tie"],
        "clock_event_demand_switch_verification": clock[
            "demand_switch_verification"
        ],
        "clock_event_contested_bundle_refinement": clock[
            "contested_bundle_refinement"
        ],
        "clock_event_terminal_winner_verification": clock[
            "terminal_winner_verification"
        ],
        "clock_event_terminal_vcg_witness_verification": clock[
            "terminal_vcg_witness_verification"
        ],
        "clock_event_terminal_best_losing_challenger": clock[
            "terminal_best_losing_challenger"
        ],
        "clock_frontier_winner_verification": clock[
            "frontier_winner_verification"
        ],
        "clock_frontier_pivotal_challengers": clock[
            "frontier_pivotal_challengers"
        ],
        "clock_frontier_winner_closure": clock[
            "frontier_winner_closure"
        ],
        "clock_frontier_vcg_witness_verification": clock[
            "frontier_vcg_witness_verification"
        ],
        "clock_frontier_vcg_single_pass": clock[
            "frontier_vcg_single_pass"
        ],
        "clock_frontier_vcg_revealed_only": clock[
            "frontier_vcg_revealed_only"
        ],
        "clock_frontier_staged_revealed_vcg_closure": clock[
            "frontier_staged_revealed_vcg_closure"
        ],
        "clock_event_allocation_counterfactual_frontier": clock[
            "allocation_counterfactual_frontier"
        ],
        "clock_event_terminal_stability_audit": clock[
            "terminal_stability_audit"
        ],
        "clock_event_scarcity_fallbacks": clock["scarcity_fallbacks"],
        "clock_event_large_correction_followup": clock[
            "large_correction_followup"
        ],
        "event_additional_pivotal_challengers": clock[
            "additional_pivotal_challengers"
        ],
        "event_gate_near_zero_surplus": clock["gate_near_zero_surplus"],
        "event_terminal_regret_audit": clock["terminal_regret_audit"],
    }


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
            "per-bidder refinement cap: none. "
            "Refinement budget is bounded only by elicitation-event logic."
        )

    return warnings


def refinement_cap_display(args: Any) -> tuple[str, str]:
    """Render the per-bidder/global refinement caps for display.

    Both caps use the CLI's legacy 0-means-unlimited convention (mirroring
    ``per_bidder_refinement_query_limit``/``global_refinement_query_safety_limit``
    being ``null`` for "no cap" at the config-schema layer) -- ``0`` displays
    as ``"none"``, a positive value displays as itself.
    """
    max_ref = getattr(args, "max_refinement_queries_per_bidder", 0)
    max_total_ref = getattr(args, "max_total_refinement_queries", 0)
    per_bidder_str = str(max_ref) if max_ref > 0 else "none"
    global_str = str(max_total_ref) if max_total_ref > 0 else "none"
    return per_bidder_str, global_str


# ---------------------------------------------------------------------------
# Full run-configuration header
# ---------------------------------------------------------------------------

_WIDE = "━" * 70


def format_run_config(
    args: Any,
    scenarios: list,
    *,
    calibration: ValueCalibration | None = None,
) -> list[str]:
    """Return a list of lines describing the effective run configuration.

    Suitable for printing at the start of a run with ``print("\\n".join(lines))``.

    ``calibration`` is the *resolved* provisional-value calibration (not the
    raw flags). It is always reported, including the identity case, so a
    reader can tell an uncalibrated baseline from a calibrated treatment
    without reconstructing it from the command line. Defaults to
    :data:`~auctionlab.llm.value_calibration.NO_CALIBRATION`.
    """
    calibration = calibration or NO_CALIBRATION
    event_policy = getattr(args, "resolved_event_policy", None)
    if event_policy is None:
        event_policy = resolve_event_policy(args)
    lines: list[str] = [_WIDE, "  auctionlab  ·  run configuration", _WIDE]

    provider = getattr(args, "provider", "?")
    model = getattr(args, "model", "?")
    person_provider = getattr(args, "person_provider", None) or provider
    person_model = getattr(args, "person_model", None) or model
    proxy_provider = getattr(args, "proxy_provider", None) or provider
    proxy_model = getattr(args, "proxy_model", None) or model
    lines.append(
        f"  person provider / model   {person_provider} / {person_model}"
    )
    lines.append(
        f"  proxy provider / model    {proxy_provider} / {proxy_model}"
    )
    elicitation_pack = getattr(args, "elicitation_pack", None)
    write_elicitation_pack = getattr(args, "write_elicitation_pack", None)
    if elicitation_pack is not None:
        lines.append(f"  initial elicitation       frozen replay: {elicitation_pack}")
    elif write_elicitation_pack is not None:
        lines.append(
            f"  initial elicitation       live preparation: "
            f"{write_elicitation_pack}"
        )

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
        lines.append(
            f"  selection_policy          "
            f"{md.get('selection_policy', getattr(args, 'selection_policy', 'prefix'))}"
        )
    else:
        seed_type = getattr(args, "seed_type", "all")
        lines.append(f"  scenarios                 {len(scenarios)}  seed_type={seed_type}")

    lines.append(f"  seed_type                 {getattr(args, 'seed_type', 'all')}")
    lines.append("")

    # Proxy construction
    ask_iq = getattr(args, "ask_initial_question", False)
    use_im = getattr(args, "use_interest_map", False)
    use_pv = getattr(args, "use_provisional_valuations", False)
    max_cb = getattr(args, "max_candidate_bundles", None)
    pv_tok = getattr(args, "pv_max_tokens", 1500)
    mbs = getattr(args, "max_bundle_size", 2)

    lines.append(f"  ask_initial_question      {ask_iq}")
    opening_question = getattr(args, "opening_question", None)
    opening_policy = (
        "explicit"
        if opening_question is not None
        else getattr(args, "opening_question_policy", "canonical")
    )
    lines.append(f"  opening_question_policy   {opening_policy}")
    lines.append(
        f"  person_nl_max_tokens      "
        f"{getattr(args, 'person_nl_max_tokens', 1500)}"
    )
    lines.append(f"  use_interest_map          {use_im}")
    if use_im:
        lines.append(
            f"  interest_map_max_tokens   "
            f"{getattr(args, 'interest_map_max_tokens', 1500)}"
        )
    lines.append(
        "  interest_map_failure_policy  "
        f"{getattr(args, 'interest_map_failure_policy', 'raise')}"
    )
    if max_cb:
        lines.append(f"  max_candidate_bundles     {max_cb}")
    lines.append(f"  use_provisional_vals      {use_pv}")
    if use_pv:
        lines.append(f"  pv_max_tokens             {pv_tok}")
        pv_chunk_size = getattr(args, "pv_chunk_size", 0)
        if pv_chunk_size:
            lines.append(f"  pv_chunk_size             {pv_chunk_size}")
        lines.append(
            f"  pv_failure_policy         {getattr(args, 'pv_failure_policy', 'raise')}"
        )
    if not use_im:
        lines.append(f"  max_bundle_size           {mbs}")
    lines.extend(calibration.header_lines())
    lines.append(f"  top_k                     {getattr(args, 'top_k', [1])}")
    query_mode = getattr(args, "person_query_mode", None)
    if query_mode is None:
        query_mode = (
            "deterministic"
            if getattr(args, "ground_truth_queries", False)
            else "llm"
        )
    lines.append(f"  person query mode         {query_mode}")
    lines.append("")

    lines.append(f"  event_policy             {event_policy['name']}")
    sealed_events = event_policy["sealed"]
    clock_events = event_policy["clock"]
    lines.append(
        "  sealed events            "
        f"incumbent={sealed_events['incumbent_verification']}  "
        f"winner_removal={sealed_events['winner_removal_counterfactuals']}  "
        f"scarcity={sealed_events['scarcity_fallbacks']}  "
        f"large_correction={sealed_events['large_correction_followup']}"
    )
    lines.append(
        "  clock events             "
        f"incumbent={clock_events['incumbent_verification']}  "
        f"counterfactual={clock_events['allocation_counterfactual_frontier']}  "
        f"scarcity={clock_events['scarcity_fallbacks']}  "
        f"large_correction={clock_events['large_correction_followup']}  "
        f"terminal_stability={clock_events['terminal_stability_audit']}"
    )
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
    stopping_rule = getattr(args, "sealed_stopping_rule", "fixed_rounds")
    loser_challenger_policy = getattr(
        args, "sealed_loser_challenger_policy", "off"
    )
    per_bidder_cap, global_cap = refinement_cap_display(args)
    ref_str = (
        f"per-bidder refinement cap: {per_bidder_cap}"
        f"  global refinement safety cap: {global_cap}"
    )
    if sealed_rounds > 0:
        trajectory = getattr(args, "sealed_trajectory", True)
        lines.append(
            f"    proxy sealed            enabled"
            f"  {'rounds' if stopping_rule == 'fixed_rounds' else 'max_rounds'}"
            f"={sealed_rounds}"
            f"  feedback_rule={feedback_rule}"
            f"  loser_challenger_policy={loser_challenger_policy}"
            f"  stopping_rule={stopping_rule}"
            f"  {ref_str}"
            f"  trajectory={'on' if trajectory else 'off'}"
        )
    else:
        lines.append("    proxy sealed            disabled")

    elicited_clock = getattr(args, "elicited_clock", False)
    clock_tie = getattr(args, "clock_tie_threshold", 100.0)
    refine_top_k_frontier = getattr(
        args, "clock_refine_top_k_frontier", False
    )
    frontier_policy = getattr(args, "clock_top_k_frontier_policy", "off")
    if refine_top_k_frontier and frontier_policy == "off":
        frontier_policy = "all"
    allocation_counterfactual_frontier = getattr(
        args, "clock_allocation_counterfactual_frontier", False
    )
    if elicited_clock:
        top_k = getattr(args, "top_k", [1])
        lines.append(
            f"    proxy clock             enabled"
            f"  top_k={top_k}"
            f"  {ref_str}"
            f"  max_rounds={max_rounds}"
            f"  tie_threshold={clock_tie}"
            f"  top_k_frontier_policy={frontier_policy}"
            f"  allocation_counterfactual_frontier="
            f"{'on' if allocation_counterfactual_frontier else 'off'}"
            f"  allocation_change_audit="
            f"{'on' if clock_events['allocation_change_audit'] else 'off'}"
            f"  terminal_stability_audit="
            f"{'on' if clock_events['terminal_stability_audit'] else 'off'}"
        )
    else:
        lines.append("    proxy clock             disabled")

    lines.append("")
    lines.append(f"  output                    {getattr(args, 'log_dir', '—')}")
    lines.append(_WIDE)
    return lines


# ---------------------------------------------------------------------------
# Calibration auditability
# ---------------------------------------------------------------------------

#: Column names added to every result CSV that reports auction outcomes.
CALIBRATION_CSV_FIELDS: tuple[str, ...] = tuple(
    NO_CALIBRATION.summary_fields()
)


def calibration_summary_fields(
    calibration: ValueCalibration | None,
) -> dict[str, Any]:
    """Flat calibration columns for run-summary and detailed result rows."""
    return (calibration or NO_CALIBRATION).summary_fields()


def add_calibration_fields(
    rows: list[dict[str, Any]],
    calibration: ValueCalibration | None,
) -> list[dict[str, Any]]:
    """Stamp the effective calibration onto result rows, in place.

    Applied to every outcome CSV rather than only the run summary: detailed
    sealed/clock rows are routinely read on their own, and a welfare number
    whose calibration is recorded in a different file is a number nobody can
    safely reuse.
    """
    fields = calibration_summary_fields(calibration)
    for row in rows:
        row.update(fields)
    return rows


def build_run_config_document(
    args: Any,
    *,
    calibration: ValueCalibration | None,
    scenarios: list | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the machine-readable ``run_config.json`` payload.

    Records the fully-resolved effective calibration -- not the flags the user
    typed -- so replaying from this document reproduces the run exactly even
    when the defaults later change.
    """
    calibration = calibration or NO_CALIBRATION
    event_policy = getattr(args, "resolved_event_policy", None)
    if event_policy is None:
        event_policy = resolve_event_policy(args)
    document: dict[str, Any] = {
        "format": "auctionlab.run_config",
        "version": 1,
        "pv_calibration": calibration.to_dict(),
        "pv_calibration_config_path": calibration.source_path,
        "pv_calibration_config_hash": calibration.config_hash(),
        "models": {
            "person_provider": getattr(args, "person_provider", None),
            "person_model": getattr(args, "person_model", None),
            "person_temperature": getattr(args, "person_temperature", None),
            "proxy_provider": getattr(args, "proxy_provider", None),
            "proxy_model": getattr(args, "proxy_model", None),
            "proxy_temperature": getattr(args, "proxy_temperature", None),
        },
        "elicitation": {
            "ask_initial_question": getattr(args, "ask_initial_question", None),
            "use_interest_map": getattr(args, "use_interest_map", None),
            "use_provisional_valuations": getattr(
                args, "use_provisional_valuations", None
            ),
            "person_query_mode": getattr(args, "person_query_mode", None),
            "elicitation_pack": _as_str(getattr(args, "elicitation_pack", None)),
            "write_elicitation_pack": _as_str(
                getattr(args, "write_elicitation_pack", None)
            ),
        },
        "event_policy": event_policy,
        "log_dir": _as_str(getattr(args, "log_dir", None)),
    }
    if scenarios is not None:
        document["scenarios"] = [
            getattr(scenario, "name", str(scenario)) for scenario in scenarios
        ]
    if extra:
        document.update(extra)
    return document


def write_run_config_json(
    path: str | Path,
    document: dict[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Refinement-record CSV helpers
# ---------------------------------------------------------------------------

def refinement_records_to_rows(
    scenario_name: str,
    arm: str,
    records_by_bidder: dict[str, list],
    *,
    final_allocation: dict | None = None,
    reported_vcg_counterfactuals: dict | None = None,
    full_info_allocation: dict | None = None,
    full_info_vcg_counterfactuals: dict | None = None,
) -> list[dict[str, Any]]:
    """Convert ``refinement_records_by_bidder`` dicts to flat CSV rows.

    Each ``RefinementRecord`` yields one row with:
    scenario, arm, bidder_id, event_type, round_idx, bundle,
    old_value, new_value, value_delta, reason, query_text, response_summary,
    and (when final outcomes are supplied) allocation and VCG-witness hits.
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
            final_hit = (
                ""
                if final_allocation is None
                else bool(bundle)
                and final_allocation.get(bidder_id, frozenset()) == bundle
            )
            reported_count = (
                ""
                if reported_vcg_counterfactuals is None
                else vcg_witness_count(
                    reported_vcg_counterfactuals,
                    bidder_id,
                    bundle,
                )
            )
            full_info_hit = (
                ""
                if full_info_allocation is None
                else bool(bundle)
                and full_info_allocation.get(bidder_id, frozenset()) == bundle
            )
            full_info_count = (
                ""
                if full_info_vcg_counterfactuals is None
                else vcg_witness_count(
                    full_info_vcg_counterfactuals,
                    bidder_id,
                    bundle,
                )
            )
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
                "appears_in_final_allocation": final_hit,
                "reported_vcg_counterfactual_count": reported_count,
                "appears_in_reported_vcg_counterfactual": (
                    "" if reported_count == "" else reported_count > 0
                ),
                "appears_in_any_reported_vcg_witness": (
                    ""
                    if final_hit == "" or reported_count == ""
                    else bool(final_hit) or reported_count > 0
                ),
                "appears_in_full_info_allocation": full_info_hit,
                "full_info_vcg_counterfactual_count": full_info_count,
                "appears_in_full_info_vcg_counterfactual": (
                    "" if full_info_count == "" else full_info_count > 0
                ),
                "appears_in_any_full_info_vcg_witness": (
                    ""
                    if full_info_hit == "" or full_info_count == ""
                    else bool(full_info_hit) or full_info_count > 0
                ),
            })
    return rows


# ---------------------------------------------------------------------------
# Late-reflection record CSV helpers
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
    complement_audit = stats.get(
        "proxy_interest_map_complement_entailment", CallTypeStats()
    ).calls
    pv = stats.get("proxy_provisional_valuations", CallTypeStats()).calls
    verifier = stats.get(
        "person_answer_semantic_extraction", CallTypeStats()
    )
    person = stats.get("nl_question", CallTypeStats())
    proxy_keys = {
        "proxy_nl_gen",
        "proxy_interest_map",
        "proxy_interest_map_complement_entailment",
        "proxy_provisional_valuations",
    }
    proxy_tok_in = sum(
        s.input_tokens for key, s in stats.items() if key in proxy_keys
    )
    proxy_tok_out = sum(
        s.output_tokens for key, s in stats.items() if key in proxy_keys
    )
    tok_in = sum(
        s.input_tokens
        for key, s in stats.items()
        if key != "person_answer_semantic_extraction"
    )
    tok_out = sum(
        s.output_tokens
        for key, s in stats.items()
        if key != "person_answer_semantic_extraction"
    )
    return {
        "vq": 0,
        "dq": 0,
        "nl": nlq + im + complement_audit + pv,
        "tok_in": tok_in,
        "tok_out": tok_out,
        "person_tok_in": person.input_tokens,
        "person_tok_out": person.output_tokens,
        "proxy_tok_in": proxy_tok_in,
        "proxy_tok_out": proxy_tok_out,
        "verification_calls": verifier.calls,
        "verification_tok_in": verifier.input_tokens,
        "verification_tok_out": verifier.output_tokens,
        "token_accounting_note": (
            f"nl_q={nlq}  im={im}  im_complement_audit={complement_audit}  "
            f"pv={pv}  "
            f"offline_verify={verifier.calls}"
        ),
    }
