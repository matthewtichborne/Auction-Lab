"""Run the canonical live LLM experiment over curated auction scenarios."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from auctionlab.auctions.clock import ClockConfig
from auctionlab.experiments.export import write_csv, write_csv_variable_rows
from auctionlab.experiments.llm_comparison import (
    clock_llm_comparison_to_row,
    proxy_clock_result_to_row,
    proxy_sealed_result_to_row,
    proxy_sealed_trajectory_to_rows,
    reported_bids_to_str,
    run_clock_llm_comparison,
    run_sealed_llm_comparison,
    sealed_llm_comparison_to_row,
)
from auctionlab.experiments.proxy_clock_runner import (
    ProxyClockConfig,
    run_proxy_clock_experiment,
)
from auctionlab.experiments.proxy_clock_trajectory import run_proxy_clock_trajectory
from auctionlab.experiments.proxy_sealed_runner import (
    ProxySealedConfig,
    run_proxy_sealed_vcg_experiment,
    run_proxy_sealed_vcg_trajectory,
)
from auctionlab.experiments.runner import run_sealed_vcg_experiment
from auctionlab.experiments.run_config import (
    EVENT_POLICIES,
    add_calibration_fields,
    build_run_config_document,
    calibration_summary_fields,
    collect_arm_stats,
    collect_initial_stats,
    config_warnings,
    explicitly_set_args,
    event_policy_summary_fields,
    format_run_config,
    refinement_records_to_rows,
    resolve_event_policy,
    write_run_config_json,
)
from auctionlab.instances.nl_types import NaturalLanguageAuctionScenario
from auctionlab.llm.bundles import generate_candidate_bundles
from auctionlab.llm.cache import (
    DEFAULT_CACHE_PATH,
    CacheStats,
    CachingLlmClient,
    LlmResponseCache,
)
from auctionlab.llm.clients import MockLlmClient, OpenAICompatibleLlmClient
from auctionlab.llm.frozen_elicitation import (
    BidderElicitationData,
    ModelProvenance,
    build_frozen_elicitation_pack,
    load_frozen_elicitation_pack,
    validate_pack_for_scenario,
    write_frozen_elicitation_pack,
)
from auctionlab.llm.interest_map import (
    interest_map_accuracy,
    interest_map_candidate_counts,
    interest_map_quality_flags,
)
from auctionlab.llm.logging import (
    CallTypeStats,
    LlmCallLogger,
    call_stats_from_records,
)
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.prompts import canonical_opening_question
from auctionlab.llm.provisional_valuations import PvCandidateBundleStats, PvChunkStats
from auctionlab.llm.proxies import LlmAuctionProxyAdapter, LlmInferredXorProxy
from auctionlab.llm.value_calibration import (
    NO_CALIBRATION,
    CalibrationConfigError,
    ValueCalibration,
    resolve_cli_calibration,
)
from auctionlab.proxies.base import RefinementRecord
from auctionlab.proxies.events import (
    GENERATE_CANDIDATE_BUNDLES,
    INFER_INTEREST_MAP,
    INFER_PROVISIONAL_VALUES,
    INITIAL_PREFERENCE_QUESTION,
    ProxyElicitationEvent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live LLM proxy mechanisms on curated scenarios."
    )
    parser.add_argument(
        "--provider",
        choices=[
            "ollama",
            "groq",
            "gemini",
            "openai",
            "anthropic",
            "openai-compatible",
        ],
        default="ollama",
        help=(
            "'openai' uses OPENAI_API_KEY; 'anthropic' uses "
            "ANTHROPIC_API_KEY through Anthropic's OpenAI-compatible "
            "endpoint; 'groq' uses GROQ_API_KEY; 'gemini' uses "
            "GEMINI_API_KEY."
        ),
    )
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Explicit API key. First-class providers otherwise use their "
            "standard environment variable."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--person-provider",
        choices=[
            "ollama",
            "groq",
            "gemini",
            "openai",
            "anthropic",
            "openai-compatible",
        ],
        default=None,
        help=(
            "Provider for the simulated person (NL answers and value/demand "
            "query answers). Defaults to --provider."
        ),
    )
    parser.add_argument(
        "--person-model",
        default=None,
        help="Model for the simulated person. Defaults to --model.",
    )
    parser.add_argument(
        "--person-base-url",
        default=None,
        help="Person-provider base URL. Defaults to --base-url.",
    )
    parser.add_argument(
        "--person-api-key",
        default=None,
        help=(
            "Person-provider API key. Defaults to --api-key; provider "
            "environment-variable fallback still applies when omitted."
        ),
    )
    parser.add_argument(
        "--person-temperature",
        type=float,
        default=None,
        help="Person-model temperature. Defaults to --temperature.",
    )
    parser.add_argument(
        "--proxy-provider",
        choices=[
            "ollama",
            "groq",
            "gemini",
            "openai",
            "anthropic",
            "openai-compatible",
        ],
        default=None,
        help=(
            "Provider for proxy-side interest-map "
            "extraction, provisional valuations, and late reflection. "
            "Also generates the one shared opening question only when "
            "--opening-question-policy proxy_generated is selected. "
            "Defaults to --provider."
        ),
    )
    parser.add_argument(
        "--proxy-model",
        default=None,
        help="Model for proxy-side inference. Defaults to --model.",
    )
    parser.add_argument(
        "--proxy-base-url",
        default=None,
        help="Proxy-provider base URL. Defaults to --base-url.",
    )
    parser.add_argument(
        "--proxy-api-key",
        default=None,
        help=(
            "Proxy-provider API key. Defaults to --api-key; provider "
            "environment-variable fallback still applies when omitted."
        ),
    )
    parser.add_argument(
        "--proxy-temperature",
        type=float,
        default=None,
        help="Proxy-model temperature. Defaults to --temperature.",
    )
    parser.add_argument(
        "--verifier-provider",
        choices=[
            "ollama",
            "groq",
            "gemini",
            "openai",
            "anthropic",
            "openai-compatible",
        ],
        default=None,
        help=(
            "Preparation-time person-answer verifier provider. Defaults to "
            "the environment-generation provider recorded by the scenario, "
            "then to --proxy-provider when unavailable."
        ),
    )
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--verifier-base-url", default=None)
    parser.add_argument("--verifier-api-key", default=None)
    parser.add_argument(
        "--verifier-temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--verifier-max-tokens",
        type=int,
        default=2000,
    )
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument(
        "--person-nl-max-tokens",
        type=int,
        default=1500,
        help=(
            "Output-token budget for simulated-person answers to the opening "
            "natural-language preference question. Kept separate from "
            "--max-tokens so long NL answers do not truncate while ordinary "
            "value/demand queries retain a compact budget."
        ),
    )
    parser.add_argument(
        "--interest-map-max-tokens",
        type=int,
        default=1500,
        help=(
            "Output-token budget for proxy interest-map extraction. Kept "
            "separate from --max-tokens and --pv-max-tokens."
        ),
    )
    parser.add_argument(
        "--person-query-mode",
        choices=["deterministic", "llm"],
        default="deterministic",
        help=(
            "How the simulated person answers value/demand queries. "
            "'deterministic' (default) uses the scenario's private valuation "
            "table and makes no person-LLM call; 'llm' is a noisy robustness "
            "treatment conditioned only on the brief qualitative disclosure."
        ),
    )
    parser.add_argument(
        "--ground-truth-queries",
        action="store_true",
        help=(
            "Deprecated compatibility alias for "
            "--person-query-mode deterministic. Deterministic queries are "
            "already the default."
        ),
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-parse-retries", type=int, default=1)
    parser.add_argument(
        "--interest-map-failure-policy",
        choices=["raise", "all_items"],
        default="raise",
        help=(
            "Behaviour after all interest-map parse attempts fail. 'raise' "
            "(default) aborts the run; 'all_items' is a degraded debugging-only fallback."
        ),
    )
    parser.add_argument(
        "--llm-cache-mode",
        choices=["off", "read-write", "read-only", "refresh"],
        default="off",
        help=(
            "LLM response cache mode (see README.md, 'Frozen elicitation and "
            "caching'). 'off': never "
            "read/write the cache. 'read-write': reuse a cached response if "
            "present, otherwise call the provider and cache it. 'read-only': "
            "require a cached response, raising CacheMissError on a miss "
            "instead of calling the provider. 'refresh': always call the "
            "provider and overwrite the cache entry."
        ),
    )
    parser.add_argument(
        "--llm-cache-path",
        type=str,
        default=DEFAULT_CACHE_PATH,
        help="Path to the SQLite LLM response cache file.",
    )
    parser.add_argument(
        "--log-dir",
        default="outputs/llm_runs/curated_batch",
    )
    parser.add_argument(
        "--elicitation-pack",
        type=Path,
        default=None,
        help=(
            "Replay a previously frozen initial-elicitation pack. The "
            "opening question/answer, interest map, candidate bundles, and "
            "raw PV values are loaded without initial LLM calls."
        ),
    )
    parser.add_argument(
        "--disclosure-pack",
        type=Path,
        default=None,
        help=(
            "Reuse only the opening question/person answers from a frozen "
            "pack, then regenerate interest maps and raw PVs with the current "
            "--proxy-provider/--proxy-model. Intended for model-portability "
            "validation; combine with --write-elicitation-pack."
        ),
    )
    parser.add_argument(
        "--write-elicitation-pack",
        type=Path,
        default=None,
        help=(
            "After live initial elicitation, validate and write a frozen "
            "pack containing raw, uncalibrated PV values and generation "
            "provenance."
        ),
    )
    parser.add_argument(
        "--prepare-elicitation-only",
        action="store_true",
        help=(
            "Generate/validate --write-elicitation-pack and skip auction "
            "mechanisms. Requires --write-elicitation-pack."
        ),
    )
    parser.add_argument(
        "--pv-calibration-config",
        type=Path,
        default=None,
        help=(
            "Preferred way to calibrate raw LLM-inferred provisional values. "
            "Path to a calibration JSON "
            '{"schema_version":"1","family":"none|uniform|exponential",'
            '"scale":1.0,"size_gamma":1.0,"size_threshold":3,'
            '"budget_cap":true}. calibrated = scale * raw * size_gamma ** '
            "max(0, |B| - size_threshold), then capped at the bidder's "
            "disclosed budget when budget_cap is set. scale may exceed 1. "
            "Omitted (the default) means raw, uncalibrated provisional "
            "values. Produce a fitted config with "
            "scripts/fit_pv_calibration.py. Cannot be combined with the "
            "deprecated flags below."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help=(
            "DEPRECATED (use --pv-calibration-config): multiplicative "
            "discount in (0, 1] applied to inferred values. Requires "
            "--discount-inferred; supplying it alone is now an error rather "
            "than a silent no-op."
        ),
    )
    parser.add_argument(
        "--discount-inferred",
        action="store_true",
        help=(
            "DEPRECATED (use --pv-calibration-config): enable the legacy "
            "epsilon/size-discount calibration."
        ),
    )
    parser.add_argument(
        "--size-discount-family",
        choices=["exponential"],
        default=None,
        help=(
            "DEPRECATED (use --pv-calibration-config): bundle-size-dependent "
            "discount applied on top of --epsilon (requires "
            "--discount-inferred). Only 'exponential' is supported: "
            "adjusted = raw_value * epsilon * size_discount_gamma ** "
            "max(0, bundle_size - size_discount_k0)."
        ),
    )
    parser.add_argument(
        "--size-discount-k0",
        type=int,
        default=3,
        help=(
            "DEPRECATED (use --pv-calibration-config): bundle size below/at "
            "which the exponential size discount leaves values unchanged."
        ),
    )
    parser.add_argument(
        "--size-discount-gamma",
        type=float,
        default=1.0,
        help=(
            "DEPRECATED (use --pv-calibration-config): exponential "
            "size-discount factor (must be > 0)."
        ),
    )
    parser.add_argument("--disable-anchor-values", action="store_true")
    parser.add_argument("--max-bundle-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1])
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--price-step", type=float, default=50.0)
    parser.add_argument("--reserve", type=float, default=0.0)
    parser.add_argument("--elicited-clock", action="store_true")
    parser.add_argument(
        "--clock-margin-threshold",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--clock-tie-threshold",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--clock-refine-top-k-frontier",
        action="store_true",
        help=(
            "For the elicited proxy clock, issue one deduplicated deterministic "
            "value query whenever a bundle newly enters a bidder's positive-"
            "surplus top-k demand frontier. Compatibility alias for "
            "--clock-top-k-frontier-policy all."
        ),
    )
    parser.add_argument(
        "--clock-top-k-frontier-policy",
        choices=["off", "all", "allocation_pivotal"],
        default="off",
        help=(
            "Clock treatment for newly seen positive-surplus top-k bundles. "
            "'allocation_pivotal' queries only bundles whose reported forced-"
            "allocation welfare gap is within --clock-tie-threshold."
        ),
    )
    parser.add_argument(
        "--clock-allocation-counterfactual-frontier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "At the initial supplementary WDP allocation and after allocation "
            "changes, query a bounded frontier exposed by removing incumbent "
            "winning atoms."
        ),
    )
    parser.add_argument(
        "--event-policy",
        choices=EVENT_POLICIES,
        default="custom",
        help=(
            "Resolved elicitation-event specification. 'custom' preserves "
            "the granular --event-* and mechanism flags. 'recommended' "
            "enables incumbent/counterfactual verification and scarcity "
            "fallbacks for the sealed-policy ablation, plus sealed-only "
            "large-correction follow-up. 'final-v3' preserves that sealed policy "
            "and uses revealed-witness/winner sandwich closure."
        ),
    )
    parser.add_argument(
        "--clock-event-framework",
        choices=["legacy", "targeted_v1", "native_v1", "frontier_v1"],
        default="legacy",
        help="Clock event-generation architecture used by custom policies.",
    )
    parser.add_argument(
        "--clock-native-near-zero-surplus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the original near-dropout surplus value-query event.",
    )
    parser.add_argument(
        "--clock-native-demand-changed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the original abandoned-demand value-query event.",
    )
    parser.add_argument(
        "--clock-native-near-tie",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the original close runner-up value-query event.",
    )
    parser.add_argument(
        "--clock-frontier-winner-verification",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verify bundles in the terminal clock allocation.",
    )
    parser.add_argument(
        "--clock-frontier-pivotal-challengers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Verify one close, overlapping clock-revealed losing demand per "
            "terminal winning bundle."
        ),
    )
    parser.add_argument(
        "--clock-frontier-winner-closure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verify newly allocated bundles until all final winners are exact.",
    )
    parser.add_argument(
        "--clock-frontier-vcg-witness-verification",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Additionally close the terminal bidder-removal VCG frontier.",
    )
    parser.add_argument(
        "--clock-frontier-vcg-single-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Query only the bidder-removal witnesses frozen before any "
            "terminal correction, then recompute once without closure."
        ),
    )
    parser.add_argument(
        "--clock-frontier-vcg-revealed-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In single-pass mode, retain only witness bundles observed on "
            "the top-k clock demand path."
        ),
    )
    parser.add_argument(
        "--clock-frontier-staged-revealed-vcg-closure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After terminal winner closure, iteratively verify clock-revealed "
            "VCG witnesses and close any newly exposed winners."
        ),
    )
    parser.add_argument(
        "--clock-supplementary-support-policy",
        choices=["all_atoms", "demand_revealed"],
        default="all_atoms",
        help=(
            "Bundles admitted to supplementary WDP/VCG support: the full "
            "proxy candidate bid (historical behavior), or only top-k "
            "positive-surplus bundles revealed along the clock path."
        ),
    )
    parser.add_argument(
        "--clock-event-demand-switch-verification",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Query newly entered and abandoned primary demand bundles.",
    )
    parser.add_argument(
        "--clock-event-contested-bundle-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Query at most one positive-surplus contested alternative per "
            "bidder during the clock."
        ),
    )
    parser.add_argument(
        "--clock-event-terminal-winner-verification",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--clock-event-terminal-vcg-witness-verification",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--clock-event-terminal-best-losing-challenger",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--clock-event-terminal-stability-audit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Independently enable the legacy terminal supplementary-WDP "
            "stability audit. If omitted, it follows incumbent verification."
        ),
    )
    parser.add_argument(
        "--event-incumbent-verification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Verify bundles selected by the current reported allocation. "
            "Enabled by default; use --no-event-incumbent-verification only "
            "for an event-policy ablation."
        ),
    )
    parser.add_argument(
        "--event-pivotal-challengers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Query a bounded exact forced-allocation challenger frontier."
        ),
    )
    parser.add_argument(
        "--event-pivotal-gap-threshold",
        type=float,
        default=100.0,
        help=(
            "Maximum absolute reported-welfare gap for a pivotal challenger."
        ),
    )
    parser.add_argument(
        "--event-scarcity-fallbacks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Query high-ranked alternatives that use fewer contested goods."
        ),
    )
    parser.add_argument(
        "--event-large-correction-followup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After a large exact correction, query one nearby candidate "
            "bundle; the follow-up is non-recursive."
        ),
    )
    parser.add_argument(
        "--sealed-event-large-correction-followup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Custom-policy override for sealed only. Omit to inherit "
            "--event-large-correction-followup. Used by focused ablations."
        ),
    )
    parser.add_argument(
        "--clock-event-large-correction-followup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Custom-policy override for clock only. Omit to inherit "
            "--event-large-correction-followup. Used by focused ablations."
        ),
    )
    parser.add_argument(
        "--event-correction-threshold",
        type=float,
        default=0.25,
        help=(
            "Symmetric relative correction required to trigger the one-step "
            "large-correction follow-up."
        ),
    )
    parser.add_argument(
        "--event-gate-near-zero-surplus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Clock only: retain near-zero-surplus events only when their "
            "bundle touches a good contested in the preceding round."
        ),
    )
    parser.add_argument(
        "--event-terminal-regret-audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Before finalization, query the closest remaining reported "
            "forced-allocation challenger."
        ),
    )
    parser.add_argument(
        "--max-refinement-queries-per-bidder",
        type=int,
        default=0,
        help=(
            "Safety cap on refinement queries per bidder for the proxy "
            "sealed/clock elicited arms (--sealed-elicitation-rounds, "
            "--elicited-clock). 0 means unlimited. Refinement count is "
            "meant to be an outcome of the elicitation events and mechanism, "
            "not a tuning target -- this and --max-total-refinement-queries "
            "should normally be left high enough not to bind. Value/demand queries are "
            "already deduplicated per bidder/bundle regardless of this cap."
        ),
    )
    parser.add_argument(
        "--max-total-refinement-queries",
        type=int,
        default=0,
        help=(
            "Safety cap on refinement queries summed across all bidders, "
            "for the same proxy sealed/clock elicited arms. 0 means "
            "unlimited. A global backstop against runaway query volume, "
            "independent of --max-refinement-queries-per-bidder."
        ),
    )
    parser.add_argument(
        "--sealed-elicitation-rounds",
        type=int,
        default=0,
        help=(
            "Number of sealed refinement rounds. With "
            "--sealed-stopping-rule fixed_rounds this is exact; with "
            "no_new_refinements it is the maximum-round safety cap."
        ),
    )
    parser.add_argument(
        "--sealed-stopping-rule",
        choices=["fixed_rounds", "no_new_refinements"],
        default="fixed_rounds",
        help=(
            "fixed_rounds (default) always runs the requested number of "
            "sealed rounds. no_new_refinements stops after the first "
            "completed round that produces zero new refinement queries; "
            "--sealed-elicitation-rounds remains the maximum."
        ),
    )
    parser.add_argument(
        "--sealed-feedback-rule",
        choices=[
            "none",
            "allocated_bundle",
            "lost_interested_bundle",
            "all_provisional",
            "competitive",
            "all_valued_bundles",
        ],
        default="none",
    )
    parser.add_argument(
        "--sealed-loser-challenger-policy",
        choices=["off", "shadow_price"],
        default="off",
        help=(
            "For competitive sealed feedback, optionally add one independent "
            "shadow-price challenger for each reported loser. 'off' keeps "
            "only allocated and winner-removal counterfactual bundles."
        ),
    )
    parser.add_argument(
        "--no-sealed-trajectory",
        dest="sealed_trajectory",
        action="store_false",
        default=True,
        help=(
            "Disable per-round trajectory recording for the proxy sealed "
            "arm. By default, whenever --sealed-elicitation-rounds > 0, "
            "results are recorded after every round (0..R) using the same "
            "proxy state, and written to "
            "curated_proxy_sealed_trajectory.csv. The final round is "
            "always used as the proxy sealed result/CSV row, so this flag "
            "only affects whether the per-round trajectory is captured."
        ),
    )
    parser.add_argument(
        "--no-clock-trajectory",
        dest="clock_trajectory",
        action="store_false",
        default=True,
        help=(
            "Disable per-round diagnostic trajectory recording for the "
            "proxy clock arm. By default, whenever --elicited-clock is set, "
            "every clock round (for each --top-k value) is recorded to "
            "curated_proxy_clock_rounds_top_{k}.csv, "
            "curated_proxy_clock_bidder_rounds_top_{k}.csv, "
            "curated_proxy_clock_coverage_top_{k}.csv, and "
            "curated_proxy_clock_event_usefulness_top_{k}.csv. The final "
            "round is always used as the proxy clock result/CSV row, so "
            "this flag only affects whether the per-round diagnostics are "
            "captured."
        ),
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help=(
            "Skip the sealed and clock LLM comparison baselines. Useful when "
            "only the proxy-mediated arms (--elicited-clock, "
            "--sealed-elicitation-rounds) are needed."
        ),
    )
    parser.add_argument("--scenario", nargs="*", default=None)
    parser.add_argument(
        "--seed-type",
        choices=["explicit", "implicit", "structured", "all"],
        default="all",
    )
    parser.add_argument(
        "--num-goods",
        type=int,
        default=8,
        help=(
            "Number of PC goods for --scenario pc_build (at least 4 and no "
            "more than the selected scenario spec contains). "
            "Ignored when --scenario names a specific scenario."
        ),
    )
    parser.add_argument(
        "--num-bidders",
        type=int,
        default=8,
        help=(
            "Number of bidder archetypes for --scenario pc_build (at least "
            "3 and no more than the selected scenario spec contains). "
            "Ignored when --scenario names a specific scenario."
        ),
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=0,
        help=(
            "Scenario selection seed. With --scenario-spec it is used by "
            "--selection-policy seeded_sample/stratified/coverage_stratified; "
            "prefix ignores it."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=[
            "prefix",
            "seeded_sample",
            "stratified",
            "coverage_stratified",
        ],
        default="prefix",
        help=(
            "How a spec selects goods and bidders. prefix preserves legacy "
            "declaration order; seeded_sample and stratified produce nested, "
            "seed-dependent subsets; coverage_stratified additionally rejects "
            "nested orders that fail sample-level interest/group constraints."
        ),
    )
    parser.add_argument(
        "--scenario-spec",
        type=str,
        default=None,
        help=(
            "Path to a ScenarioProfileSpec JSON file, normally the current "
            "validated population under scenarios/pc_build_v2/. "
            "When set together with --scenario pc_build, builds the scenario via "
            "make_pc_build_scenario_from_spec instead of the hard-coded archetype "
            "builders."
        ),
    )
    parser.add_argument(
        "--ask-initial-question",
        action="store_true",
        help=(
            "ωnvd: deliver the scenario-level opening question to each bidder "
            "and fold the answer into all later value inference."
        ),
    )
    parser.add_argument(
        "--opening-question-policy",
        choices=["canonical", "proxy_generated"],
        default="canonical",
        help=(
            "canonical (default) reuses one versioned scenario-level question "
            "without an LLM generation call. proxy_generated asks the proxy "
            "model to generate one question for the selected catalogue and "
            "reuses it across all bidders."
        ),
    )
    parser.add_argument(
        "--opening-question",
        default=None,
        help=(
            "Explicit scenario-level opening question. Overrides "
            "--opening-question-policy and is reused across all bidders."
        ),
    )
    parser.add_argument(
        "--refinement-strategy",
        choices=["value_query", "demand_query"],
        default="value_query",
        help=(
            "ωvd1/ωvd2: how the LLM proxy refines a candidate bundle's "
            "value -- a direct value query, or a demand query ('are you "
            "satisfied with this bundle at these prices?') first."
        ),
    )
    parser.add_argument(
        "--use-interest-map",
        action="store_true",
        help=(
            "Derive per-bidder candidate bundles from an NL interest map "
            "rather than enumerating up to --max-bundle-size. Implies "
            "--ask-initial-question for the elicited proxy arm. Implied "
            "automatically by --use-provisional-valuations."
        ),
    )
    parser.add_argument(
        "--max-candidate-bundles",
        type=int,
        default=None,
        help=(
            "Cap the number of candidate bundles per bidder, applied both after "
            "interest-map filtering and again (redundantly, if it's already <= "
            "this cap) before the provisional-valuation LLM call (priority order: "
            "complementary groups first, then singletons, then remaining bundles "
            "by ascending size). Omit for a full, uncapped run at both stages -- "
            "there is no automatic token-budget-derived truncation; a large "
            "candidate count only produces an informational warning suggesting "
            "--pv-max-tokens or this flag, never a silent reduction. Ignored "
            "unless --use-interest-map is set."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print every value/demand query as it fires. By default only "
            "errors, retries, section headers, and per-arm results are shown."
        ),
    )
    parser.add_argument(
        "--use-provisional-valuations",
        action="store_true",
        help=(
            "After asking the initial NL question, call the LLM once to "
            "generate provisional values for all candidate bundles. Implies "
            "--ask-initial-question and --use-interest-map (candidate "
            "bundles are derived from the NL interest map, not enumerated "
            "or hand-curated). Skips per-bundle value queries on first "
            "bid; refinement queries still fire normally."
        ),
    )
    parser.add_argument(
        "--pv-max-tokens",
        type=int,
        default=1500,
        help=(
            "Token budget for the provisional-valuation LLM call. Higher than "
            "--max-tokens because the response contains one entry per bundle. "
            "Default: 1500."
        ),
    )
    parser.add_argument(
        "--pv-chunk-size",
        type=int,
        default=0,
        help=(
            "Split a bidder's candidate bundles into deterministic chunks of "
            "at most this size, calling the provisional-valuation LLM once "
            "per chunk and merging the results into one PV table -- see "
            "README.md, 'Provisional valuations'. 0 (default) disables "
            "chunking: one bulk PV call over every candidate bundle, exactly "
            "as before this flag existed. PV chunks are counted as PV/"
            "shared-initial elicitation calls, never as value queries -- "
            "chunking never changes VQ/refinement counts. Recommended for "
            "8x8+ live runs with a large interest-map candidate set: try 50, "
            "alongside a higher --pv-max-tokens (e.g. 12000)."
        ),
    )
    parser.add_argument(
        "--pv-failure-policy",
        choices=["raise", "zero"],
        default="raise",
        help=(
            "What happens when provisional-valuation generation fails for a "
            "bidder (after chunk/parse retries are exhausted). 'raise' "
            "(default, recommended for live experiments): abort the run "
            "immediately with a diagnostic error naming the bidder, "
            "candidate count, --pv-chunk-size, --pv-max-tokens, and the "
            "original exception -- PV failure must never silently degrade "
            "into a mass direct-value-query fallback, which would swap the "
            "elicitation regime from bulk provisional valuation to mass "
            "direct querying and contaminate VQ counts / mechanism "
            "comparison. 'zero': previous debugging behaviour -- initialise "
            "that bidder's candidate bundles at 0.0 and continue, marking "
            "the bidder as degraded in logs/summary. Even under 'zero', PV "
            "failure never triggers direct value queries over the full "
            "candidate support -- mechanism-triggered refinement caps "
            "(--max-refinement-queries-per-bidder etc.) remain the only "
            "source of post-initialisation value queries."
        ),
    )
    parser.add_argument(
        "--est-tok-per-vq",
        type=int,
        default=0,
        help=(
            "Estimated input tokens per ground-truth value query (for the "
            "tok-in 'est-total' column when --ground-truth-queries is on). "
            "0 disables the estimate. Typical value: 150."
        ),
    )
    parser.add_argument(
        "--est-tok-per-dq",
        type=int,
        default=0,
        help=(
            "Estimated input tokens per ground-truth demand query. "
            "0 disables the estimate. Typical value: 200."
        ),
    )
    return parser.parse_args()


def _build_raw_live_client(
    *,
    model: str,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> OpenAICompatibleLlmClient:
    if provider == "ollama":
        return OpenAICompatibleLlmClient.for_ollama(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "groq":
        return OpenAICompatibleLlmClient.for_groq(
            model=model,
            base_url=base_url or "https://api.groq.com/openai/v1",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "gemini":
        return OpenAICompatibleLlmClient.for_gemini(
            model=model,
            base_url=(
                base_url
                or "https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "openai":
        return OpenAICompatibleLlmClient.for_openai(
            model=model,
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            # GPT-5 reasoning models accept only their default sampling
            # temperature. Omit the parameter for the whole GPT-5 family,
            # including current Sol/Terra/Luna variants.
            temperature=effective_model_temperature(
                provider, model, temperature
            ),
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "anthropic":
        return OpenAICompatibleLlmClient.for_anthropic(
            model=model,
            base_url=base_url or "https://api.anthropic.com/v1/",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    return OpenAICompatibleLlmClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def effective_model_temperature(
    provider: str | None,
    model: str | None,
    requested_temperature: float | None,
) -> float | None:
    """Return the sampling temperature actually sent to the provider."""
    if (
        provider == "openai"
        and model is not None
        and model.startswith("gpt-5")
    ):
        return None
    if (
        provider == "gemini"
        and model is not None
        and model.startswith(("gemini-3.5-", "gemini-3.6-"))
    ):
        return None
    return requested_temperature


def resolve_llm_role_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve person/proxy overrides against the legacy global defaults."""
    for role in ("person", "proxy"):
        for field in (
            "provider",
            "model",
            "base_url",
            "api_key",
            "temperature",
        ):
            role_field = f"{role}_{field}"
            if getattr(args, role_field, None) is None:
                setattr(args, role_field, getattr(args, field))
    return args


def resolve_person_query_mode(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve the deprecated GT flag into the effective person-query mode."""
    if getattr(args, "ground_truth_queries", False):
        args.person_query_mode = "deterministic"
    args.ground_truth_queries = args.person_query_mode == "deterministic"
    return args


def resolve_initial_elicitation_flags(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Apply the documented implications before validation/config output."""
    if args.use_provisional_valuations:
        args.use_interest_map = True
    if args.use_interest_map:
        args.ask_initial_question = True
    return args


def make_live_client(
    *,
    model: str,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    timeout: float,
    cache: LlmResponseCache | None = None,
    cache_mode: str = "off",
    cache_stats: CacheStats | None = None,
    llm_role: str | None = None,
) -> OpenAICompatibleLlmClient | CachingLlmClient:
    """Build the live provider client, optionally wrapped with LLM caching.

    When ``cache`` is given and ``cache_mode != "off"``, the returned client
    is a :class:`~auctionlab.llm.cache.CachingLlmClient` -- callers that pass
    per-call cache context (see :func:`~auctionlab.llm.cache.call_client`,
    used by ``LlmPersonSimulator``/``derive_interest_map``/
    ``generate_provisional_valuations``) get full caching; any other caller
    still gets correct (if less richly indexed) caching keyed on prompt
    text, provider, and model alone.
    """
    raw_client = _build_raw_live_client(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if cache is None or cache_mode == "off":
        client = raw_client
    else:
        client = CachingLlmClient(
            inner=raw_client,
            cache=cache,
            mode=cache_mode,
            provider=provider,
            model=model,
            temperature=raw_client.temperature,
            max_tokens=max_tokens,
            reasoning_effort=raw_client.reasoning_effort,
            stats=cache_stats if cache_stats is not None else CacheStats(),
        )
    # Uniform provenance attributes for raw and caching clients. Logging
    # call-sites use these rather than assuming every client class exposes a
    # provider or role field.
    setattr(client, "_auctionlab_provider", provider)
    setattr(client, "_auctionlab_model", model)
    setattr(client, "_auctionlab_llm_role", llm_role)
    return client


def make_live_persons_for_scenario(
    scenario: NaturalLanguageAuctionScenario,
    *,
    model: str,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    person_nl_max_tokens: int | None = None,
    timeout: float,
    logger: LlmCallLogger,
    max_parse_retries: int,
    use_ground_truth: bool = False,
    verbose: bool = False,
    cache: LlmResponseCache | None = None,
    cache_mode: str = "off",
    cache_stats: CacheStats | None = None,
    verifier_provider: str | None = None,
    verifier_model: str | None = None,
    verifier_base_url: str | None = None,
    verifier_api_key: str | None = None,
    verifier_temperature: float = 0.0,
    verifier_max_tokens: int = 2000,
) -> dict[str, LlmPersonSimulator]:
    persons: dict[str, LlmPersonSimulator] = {}

    for bidder_id in scenario.instance.bidder_ids:
        profile_metadata = (
            (getattr(scenario, "metadata", {}) or {})
            .get("profiles", {})
            .get(bidder_id, {})
        )
        available_items = set(scenario.instance.items)
        if "disclosed_positive_items" in profile_metadata:
            disclosed_positive_items = (
                set(profile_metadata["disclosed_positive_items"])
                & available_items
            )
        else:
            # Compatibility for scenarios created before the explicit
            # disclosure field existed. Singleton positivity is the same
            # criterion used by the qualitative seed renderer; classification
            # lists may also contain zero-valued goods.
            bidder_valuations = scenario.instance.valuations.get(
                bidder_id, {}
            )
            disclosed_positive_items = {
                item
                for item in available_items
                if bidder_valuations.get(frozenset({item}), 0.0) > 0
            }
        expected_interested_items = (
            disclosed_positive_items if profile_metadata else None
        )
        persons[bidder_id] = LlmPersonSimulator(
            bidder_id=bidder_id,
            scenario_description=scenario.scenario_description,
            person_seed=scenario.person_seeds[bidder_id],
            item_descriptions=scenario.item_descriptions,
            client=make_live_client(
                model=model,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                cache=cache,
                cache_mode=cache_mode,
                cache_stats=cache_stats,
                llm_role="person",
            ),
            nl_client=(
                make_live_client(
                    model=model,
                    provider=provider,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=person_nl_max_tokens,
                    timeout=timeout,
                    cache=cache,
                    cache_mode=cache_mode,
                    cache_stats=cache_stats,
                    llm_role="person",
                )
                if person_nl_max_tokens is not None
                else None
            ),
            verifier_client=(
                make_live_client(
                    model=verifier_model,
                    provider=verifier_provider,
                    base_url=verifier_base_url,
                    api_key=verifier_api_key,
                    temperature=verifier_temperature,
                    max_tokens=verifier_max_tokens,
                    timeout=timeout,
                    cache=cache,
                    cache_mode=cache_mode,
                    cache_stats=cache_stats,
                    llm_role="verifier",
                )
                if verifier_provider is not None
                and verifier_model is not None
                else None
            ),
            logger=logger,
            model_name=model,
            provider_name=provider,
            verifier_model_name=verifier_model,
            verifier_provider_name=verifier_provider,
            max_parse_retries=max_parse_retries,
            ground_truth_valuations=(
                scenario.instance.valuations[bidder_id] if use_ground_truth else None
            ),
            verbose=verbose,
            scenario_id=scenario.name,
            expected_interested_items=expected_interested_items,
            expected_excluded_items=(
                set(scenario.instance.items) - expected_interested_items
                if expected_interested_items is not None
                else None
            ),
            expected_substitute_groups=profile_metadata.get(
                "substitute_groups", []
            ),
            expected_complement_groups=profile_metadata.get(
                "complement_groups", []
            ),
            expected_budget_hint=profile_metadata.get(
                "disclosed_budget_hint",
                max(
                    scenario.instance.valuations.get(bidder_id, {}).values(),
                    default=0.0,
                ),
            ),
        )

    return persons


class PvGenerationFailedError(RuntimeError):
    """Raised when ``--pv-failure-policy raise`` (the default) and
    provisional-valuation generation fails for a bidder, after chunk/parse
    retries are exhausted.

    Carries enough context (``bidder_id``, ``candidate_count``,
    ``pv_chunk_size``, ``pv_max_tokens``, and the original exception) to
    diagnose a live PV failure without reproducing it -- see README.md,
    "Provisional valuations". Raising
    here (rather than falling back to zero-bid initialisation, which used to
    let :class:`~auctionlab.llm.proxies.LlmAuctionProxyAdapter` fall through
    to a mass direct-value-query pass over every candidate bundle) is what
    keeps a PV failure from silently switching the elicitation regime from
    bulk provisional valuation to mass direct querying.
    """

    def __init__(
        self,
        *,
        bidder_id: str,
        candidate_count: int,
        pv_chunk_size: int | None,
        pv_max_tokens: int | None,
        original_exception: BaseException,
    ) -> None:
        self.bidder_id = bidder_id
        self.candidate_count = candidate_count
        self.pv_chunk_size = pv_chunk_size
        self.pv_max_tokens = pv_max_tokens
        self.original_exception = original_exception
        message = (
            "Provisional valuation generation failed "
            f"(pv_failure_policy=raise): bidder_id={bidder_id!r}, "
            f"candidate_count={candidate_count}, pv_chunk_size={pv_chunk_size!r}, "
            f"pv_max_tokens={pv_max_tokens!r}, original_exception="
            f"{type(original_exception).__name__}: {original_exception}. "
            "Suggestion: increase --pv-max-tokens, or set --pv-chunk-size to "
            "split this bidder's candidate bundles into smaller PV calls. "
            "(Pass --pv-failure-policy zero to opt into degraded zero-bid "
            "initialisation instead of aborting -- this still never falls "
            "back to mass direct value queries.)"
        )
        super().__init__(message)


def provisional_value_scale_diagnostics(
    *,
    scenario: NaturalLanguageAuctionScenario,
    bidder_id: str,
    raw_pv_values: dict | None,
    ratio_threshold: float = 3.0,
) -> dict[str, object]:
    """Compare inferred PV scale with hidden truth for diagnostics only.

    Ground truth is never passed to the proxy or used to modify bids. This
    check only makes severe scale drift visible in experiment logs.
    """
    if not raw_pv_values:
        return {"quality_flags": ()}
    valuations = getattr(scenario.instance, "valuations", {})
    ground_truth = valuations.get(bidder_id) if isinstance(valuations, dict) else None
    if not ground_truth:
        return {"quality_flags": ()}

    inferred_max = max(float(value) for value in raw_pv_values.values())
    ground_truth_max = max(
        (float(value) for value in ground_truth.values()),
        default=0.0,
    )
    inferred_singletons = [
        float(value)
        for bundle, value in raw_pv_values.items()
        if len(bundle) == 1
    ]
    ground_truth_singletons = [
        float(value)
        for bundle, value in ground_truth.items()
        if len(bundle) == 1
    ]
    inferred_max_singleton = max(inferred_singletons, default=0.0)
    ground_truth_max_singleton = max(ground_truth_singletons, default=0.0)

    def _ratio(inferred: float, truth: float) -> float | None:
        if truth > 0:
            return inferred / truth
        return None

    max_ratio = _ratio(inferred_max, ground_truth_max)
    singleton_ratio = _ratio(
        inferred_max_singleton,
        ground_truth_max_singleton,
    )
    flags: list[str] = []
    if (
        (max_ratio is not None and max_ratio >= ratio_threshold)
        or (ground_truth_max == 0 and inferred_max > 0)
    ):
        flags.append("pv_max_value_scale_exceeds_ground_truth")
    if (
        (singleton_ratio is not None and singleton_ratio >= ratio_threshold)
        or (
            ground_truth_max_singleton == 0
            and inferred_max_singleton > 0
        )
    ):
        flags.append("pv_singleton_scale_exceeds_ground_truth")
    return {
        "quality_flags": tuple(flags),
        "inferred_max_value": inferred_max,
        "ground_truth_max_value": ground_truth_max,
        "max_value_ratio": max_ratio,
        "inferred_max_singleton": inferred_max_singleton,
        "ground_truth_max_singleton": ground_truth_max_singleton,
        "max_singleton_ratio": singleton_ratio,
    }


def compute_elicitation_cache(
    *,
    scenario: NaturalLanguageAuctionScenario,
    persons: dict[str, LlmPersonSimulator],
    use_provisional_valuations: bool,
    max_candidate_bundles: int | None,
    pv_client,
    question_client=None,
    opening_question: str | None = None,
    interest_map_client=None,
    pv_chunk_size: int | None = None,
    pv_failure_policy: str = "raise",
    interest_map_failure_policy: str = "raise",
    pv_max_tokens: int | None = None,
    max_parse_retries: int = 0,
    fixed_disclosures: dict[str, BidderElicitationData] | None = None,
) -> dict[str, BidderElicitationData]:
    """Run the NL-question + interest-map (+ optional PV) phase once.

    This is the expensive, arm-independent part of elicited proxy
    construction. Callers replay the result into one fresh proxy per arm via
    :meth:`LlmInferredXorProxy.replay_elicitation`.
    """
    print("  NL elicitation  [event-driven]", flush=True)
    cache: dict[str, BidderElicitationData] = {}
    shared_question = opening_question

    for bidder_id, person in persons.items():
        proxy = LlmInferredXorProxy(bidder_id=bidder_id, person=person)

        # Emit initialisation events through the proxy event API.
        fixed_entry = (
            None
            if fixed_disclosures is None
            else fixed_disclosures.get(bidder_id)
        )
        if fixed_disclosures is not None and fixed_entry is None:
            raise ValueError(
                f"fixed disclosure pack has no bidder {bidder_id!r}"
            )
        if fixed_entry is not None:
            proxy.nl_transcript.append(
                (fixed_entry.nl_question, fixed_entry.nl_answer)
            )
            if shared_question is None:
                shared_question = fixed_entry.nl_question
            elif fixed_entry.nl_question != shared_question:
                raise ValueError(
                    "fixed disclosure pack must use one shared opening question"
                )
            print(
                f"  {bidder_id:<12}  nl  →  frozen disclosure",
                flush=True,
            )
        else:
            proxy.handle_event(ProxyElicitationEvent(
                event_type=INITIAL_PREFERENCE_QUESTION,
                bidder_id=bidder_id,
                mechanism="init",
                payload={
                    "client": question_client,
                    "question": shared_question,
                },
            ))
            if shared_question is None:
                # Proxy-generated robustness mode generates once for the selected
                # catalogue, then reuses the same question for every bidder.
                shared_question = proxy.nl_transcript[-1][0]

        im_response = proxy.handle_event(ProxyElicitationEvent(
            event_type=INFER_INTEREST_MAP,
            bidder_id=bidder_id,
            mechanism="init",
            payload={
                "failure_policy": interest_map_failure_policy,
                "client": interest_map_client,
            },
        ))
        im = im_response.payload["interest_map"]

        cb_response = proxy.handle_event(ProxyElicitationEvent(
            event_type=GENERATE_CANDIDATE_BUNDLES,
            bidder_id=bidder_id,
            mechanism="init",
            payload={
                "all_items": list(scenario.instance.items),
                "max_candidate_bundles": max_candidate_bundles,
            },
        ))
        cb = cb_response.payload["candidate_bundles"]
        candidate_count_before_filter, candidate_count_after_filter = (
            interest_map_candidate_counts(im, list(scenario.instance.items))
        )
        im_fallback_used = im.reasoning.startswith(
            "Fallback: all interest-map parse attempts failed."
        )
        im_quality_flags = tuple(interest_map_quality_flags(
            im,
            list(scenario.instance.items),
            fallback_used=im_fallback_used,
            candidate_count_after_filter=candidate_count_after_filter,
        ))

        if not cb:
            print(
                f"  {bidder_id:<12}  WARNING: interest map produced no bundles; "
                "the bidder will have empty initial candidate support",
                flush=True,
            )
        im_detail = f"interested={sorted(im.interested_items)}"
        if im.complementary_groups:
            im_detail += f"  compl={[sorted(g) for g in im.complementary_groups]}"
        if im.substitute_groups:
            im_detail += (
                "  subst="
                + str([
                    {
                        "items": sorted(group.items),
                        "mode": group.acquisition_mode,
                    }
                    for group in im.substitute_groups
                ])
            )
        valuation_tables = getattr(scenario.instance, "valuations", None)
        profile_metadata = (
            (getattr(scenario, "metadata", {}) or {})
            .get("profiles", {})
            .get(bidder_id, {})
        )
        im_accuracy = None
        if (
            isinstance(valuation_tables, dict)
            and bidder_id in valuation_tables
        ):
            true_interested = {
                item
                for item in scenario.instance.items
                if valuation_tables[bidder_id].get(
                    frozenset({item}), 0.0
                ) > 0
            }
            im_accuracy = interest_map_accuracy(
                im,
                true_interested_items=true_interested,
                true_substitute_groups=profile_metadata.get(
                    "substitute_groups", []
                ),
                true_complement_groups=profile_metadata.get(
                    "complement_groups", []
                ),
                available_items=set(scenario.instance.items),
                nl_answer=proxy.nl_transcript[-1][1],
                singleton_values={
                    item: valuation_tables[bidder_id].get(
                        frozenset({item}), 0.0
                    )
                    for item in scenario.instance.items
                },
            )
        accuracy_detail = ""
        if im_accuracy is not None:
            accuracy_detail = (
                f"  item_precision={im_accuracy['item_precision']:.2f}"
                f"  item_recall={im_accuracy['item_recall']:.2f}"
                f"  item_f1={im_accuracy['item_f1']:.2f}"
                "  mode_acc="
                f"{im_accuracy['mode_accuracy_on_matched_groups']}"
            )
        print(
            f"  {bidder_id:<12}  im  {im_detail}  →  {len(cb)} bundles"
            f"  flags={list(im_quality_flags)}"
            f"{accuracy_detail}",
            flush=True,
        )

        raw_pv_values = None
        pv_degraded = False
        if use_provisional_valuations:
            try:
                pv_response = proxy.handle_event(ProxyElicitationEvent(
                    event_type=INFER_PROVISIONAL_VALUES,
                    bidder_id=bidder_id,
                    mechanism="init",
                    payload={
                        "candidate_bundles": cb,
                        "client": pv_client,
                        "interest_map": im,
                        "max_candidate_bundles": max_candidate_bundles,
                        "pv_chunk_size": pv_chunk_size,
                        "max_parse_retries": max_parse_retries,
                    },
                ))
                raw_pv_values = pv_response.payload["raw_values"]
            except Exception as exc:
                if pv_failure_policy == "raise":
                    raise PvGenerationFailedError(
                        bidder_id=bidder_id,
                        candidate_count=len(cb),
                        pv_chunk_size=pv_chunk_size,
                        pv_max_tokens=pv_max_tokens,
                        original_exception=exc,
                    ) from exc
                # pv_failure_policy == "zero": degraded debugging fallback.
                # Zero-initialise every candidate bundle (never leave
                # raw_pv_values as None) so the cached XOR bid is populated
                # before replay -- this is what stops
                # LlmAuctionProxyAdapter.current_bid() from falling through
                # to a mass direct-value-query pass over every candidate
                # bundle when no cached bid exists. Mechanism-triggered
                # refinement caps still control any VQs issued afterwards.
                print(
                    f"  {bidder_id:<12}  WARNING: PV failed "
                    f"({type(exc).__name__}: {exc}); DEGRADED: zero-"
                    "initialising this bidder's candidate bundles "
                    "(pv_failure_policy=zero). This does NOT fall back to "
                    "direct value queries.",
                    flush=True,
                )
                raw_pv_values = {bundle: 0.0 for bundle in cb if bundle}
                pv_degraded = True

        pv_scale = provisional_value_scale_diagnostics(
            scenario=scenario,
            bidder_id=bidder_id,
            raw_pv_values=raw_pv_values,
        )
        pv_scale_flags = tuple(pv_scale.get("quality_flags", ()))
        if pv_scale_flags:
            print(
                f"  {bidder_id:<12}  WARNING: PV value scale mismatch "
                f"flags={list(pv_scale_flags)}  "
                f"max_ratio={pv_scale.get('max_value_ratio')}  "
                f"singleton_ratio={pv_scale.get('max_singleton_ratio')}",
                flush=True,
            )

        nl_question, nl_answer = proxy.nl_transcript[-1]
        cache[bidder_id] = BidderElicitationData(
            nl_question=nl_question,
            nl_answer=nl_answer,
            interest_map=im,
            candidate_bundles=cb,
            raw_pv_values=raw_pv_values,
            pv_candidate_stats=proxy.last_pv_candidate_stats,
            pv_chunk_stats=proxy.last_pv_chunk_stats,
            pv_degraded=pv_degraded,
            interest_map_fallback_used=im_fallback_used,
            interest_map_quality_flags=im_quality_flags,
            interest_map_candidate_count_before_filter=candidate_count_before_filter,
            interest_map_candidate_count_after_filter=candidate_count_after_filter,
            interest_map_accuracy=im_accuracy,
            person_answer_verification=(
                fixed_entry.person_answer_verification
                if fixed_entry is not None
                else person.last_answer_verification
            ),
            person_answer_verification_history=(
                fixed_entry.person_answer_verification_history
                if fixed_entry is not None
                else person.answer_verification_history
            ),
            person_answer_attempt_count=(
                fixed_entry.person_answer_attempt_count
                if fixed_entry is not None
                else person.answer_attempt_count
            ),
            person_first_answer_word_count=(
                fixed_entry.person_first_answer_word_count
                if fixed_entry is not None
                else person.first_answer_word_count
            ),
            person_final_answer_word_count=(
                fixed_entry.person_final_answer_word_count
                if fixed_entry is not None
                else person.final_answer_word_count
            ),
            pv_scale_quality_flags=pv_scale_flags,
            pv_inferred_max_value=pv_scale.get("inferred_max_value"),
            pv_ground_truth_max_value=pv_scale.get("ground_truth_max_value"),
            pv_max_value_ratio=pv_scale.get("max_value_ratio"),
            pv_inferred_max_singleton=pv_scale.get("inferred_max_singleton"),
            pv_ground_truth_max_singleton=pv_scale.get(
                "ground_truth_max_singleton"
            ),
            pv_max_singleton_ratio=pv_scale.get("max_singleton_ratio"),
        )

    return cache


def make_live_proxies_for_scenario(
    scenario: NaturalLanguageAuctionScenario,
    *,
    model: str,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    person_nl_max_tokens: int | None = None,
    timeout: float,
    logger: LlmCallLogger,
    max_parse_retries: int,
    calibration: ValueCalibration | None = None,
    ask_initial_question: bool = False,
    opening_question: str | None = None,
    use_ground_truth: bool = False,
    verbose: bool = False,
    cache: LlmResponseCache | None = None,
    cache_mode: str = "off",
    cache_stats: CacheStats | None = None,
    verifier_provider: str | None = None,
    verifier_model: str | None = None,
    verifier_base_url: str | None = None,
    verifier_api_key: str | None = None,
    verifier_temperature: float = 0.0,
    verifier_max_tokens: int = 2000,
) -> dict[str, LlmInferredXorProxy]:
    persons = make_live_persons_for_scenario(
        scenario,
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        person_nl_max_tokens=person_nl_max_tokens,
        timeout=timeout,
        logger=logger,
        max_parse_retries=max_parse_retries,
        use_ground_truth=use_ground_truth,
        verbose=verbose,
        cache=cache,
        cache_mode=cache_mode,
        cache_stats=cache_stats,
        verifier_provider=verifier_provider,
        verifier_model=verifier_model,
        verifier_base_url=verifier_base_url,
        verifier_api_key=verifier_api_key,
        verifier_temperature=verifier_temperature,
        verifier_max_tokens=verifier_max_tokens,
    )

    proxies: dict[str, LlmInferredXorProxy] = {}
    shared_question = opening_question
    for bidder_id, person in persons.items():
        proxy = LlmInferredXorProxy(
            bidder_id=bidder_id,
            person=person,
            calibration=calibration,
            # The legacy epsilon field is inert once `calibration` is set, but
            # llm_comparison still echoes it into `epsilon_by_bidder`; pin it
            # to the neutral 1.0 rather than leaving the 0.75 dataclass
            # default to be misread as an applied discount.
            epsilon=1.0,
        )
        if ask_initial_question:
            proxy.ask_initial_question(question=shared_question)
            if shared_question is None:
                shared_question = proxy.nl_transcript[-1][0]
        proxies[bidder_id] = proxy

    return proxies


def person_disclosure_rows_for_scenario(
    scenario: NaturalLanguageAuctionScenario,
) -> list[dict]:
    """One ``curated_person_disclosures.csv`` row per bidder.

    Sources the exact brief qualitative disclosure used to condition each bidder's
    :class:`~auctionlab.llm.person_simulator.LlmPersonSimulator`
    (``scenario.person_seeds``, always present) plus, when available,
    per-bidder provenance from ``scenario.metadata["profiles"]`` (populated
    by :func:`auctionlab.instances.structured_spec.make_pc_build_scenario_from_spec`
    -- ``person_seed_source`` identifies the brief qualitative renderer
    and ``person_seed_identity_source`` records whether its identity came
    from ``identity_text`` or ``role``). Scenarios without that metadata (e.g. the
    hard-coded curated NL scenarios) still get a row per bidder, with
    ``person_seed_source="unknown"`` and the remaining provenance fields
    blank -- this export is about auditing exactly what text conditioned
    each person, which is always available, not just spec-derived scenarios.
    """
    metadata = getattr(scenario, "metadata", {}) or {}
    profiles_meta = metadata.get("profiles", {}) or {}

    rows: list[dict] = []
    for bidder_id in scenario.instance.bidder_ids:
        profile_meta = profiles_meta.get(bidder_id, {})
        rows.append({
            "scenario": scenario.name,
            "bidder_id": bidder_id,
            "person_disclosure_text": scenario.person_seeds.get(bidder_id, ""),
            "person_disclosure_style": "brief_qualitative",
            "person_disclosure_source": profile_meta.get(
                "person_seed_source", "unknown"
            ),
            "person_disclosure_identity_source": profile_meta.get(
                "person_seed_identity_source", "unknown"
            ),
            "profile_role": profile_meta.get("role", ""),
        })
    return rows


def select_scenarios(
    names: list[str] | None,
    seed_type: str = "all",
    *,
    num_goods: int = 8,
    num_bidders: int = 8,
    scenario_seed: int = 0,
    scenario_spec: str | None = None,
    selection_policy: str = "prefix",
) -> list[NaturalLanguageAuctionScenario]:
    """Build the generated PC scenario used by the final implementation."""
    if names != ["pc_build"]:
        raise ValueError("The retained runner requires --scenario pc_build")
    if scenario_spec is None:
        raise ValueError("The retained runner requires --scenario-spec")

    from auctionlab.instances.structured_spec import make_pc_build_scenario_from_spec

    scenarios = [
        make_pc_build_scenario_from_spec(
            scenario_spec,
            num_goods,
            num_bidders,
            seed=scenario_seed,
            selection_policy=selection_policy,
        )
    ]

    if seed_type != "all":
        scenarios = [
            scenario
            for scenario in scenarios
            if scenario.seed_type == seed_type
        ]

    if not scenarios:
        raise ValueError(
            "No scenarios matched the requested name and seed-type filters"
        )
    return scenarios


_BAR = "━" * 54


def _section(label: str) -> None:
    fill = max(0, 50 - len(label))
    print(f"\n── {label} {'─' * fill}", flush=True)


def _pct(value: float) -> str:
    if value != value or math.isinf(value):  # nan / inf guard
        return "—"
    return f"{value * 100:.1f}%"


def _arm_result(
    label: str,
    efficiency: float,
    true_welfare: float,
    full_info_welfare: float,
    extra: str = "",
    reported_welfare: float | None = None,
) -> None:
    if reported_welfare is not None:
        welfare_str = (
            f"reported {reported_welfare:.0f}  "
            f"true {true_welfare:.0f}/{full_info_welfare:.0f}"
        )
    else:
        welfare_str = f"welfare {true_welfare:.0f}/{full_info_welfare:.0f}"
    print(
        f"  ✓ {label}  {_pct(efficiency)}  {welfare_str}"
        + (f"  {extra}" if extra else ""),
        flush=True,
    )


def print_refinement_records(
    records_by_bidder: dict[str, list[RefinementRecord]],
) -> None:
    records = [
        record
        for bidder_id in sorted(records_by_bidder)
        for record in records_by_bidder[bidder_id]
    ]
    if not records:
        return
    print("  refinements:", flush=True)
    for record in records:
        bundle_str = "{" + ",".join(sorted(record.bundle)) + "}"
        old = f"{record.old_value:.0f}" if record.old_value is not None else "0"
        reason = record.reason or record.event_type or ""
        print(
            f"    {record.bidder_id:<12}  {bundle_str:<28}  "
            f"{old}→{record.new_value:.0f}  {reason}",
            flush=True,
        )


def _print_sealed_proxy_summary(
    result,
    *,
    feedback_rule: str,
    elicitation_rounds: int,
) -> None:
    """Print a post-run diagnostic summary for the proxy sealed arm."""
    recs_by_bidder = result.metadata.get("refinement_records_by_bidder", {})
    rq_by_bidder = result.metadata.get("refinement_query_count_by_bidder", {})
    total_recs = sum(len(recs) for recs in recs_by_bidder.values())
    total_queries = sum(rq_by_bidder.values())
    actual_rounds = result.metadata.get(
        "elicitation_rounds", elicitation_rounds
    )
    stopping_rule = result.metadata.get("stopping_rule", "fixed_rounds")
    termination_reason = result.metadata.get("termination_reason", "")

    fill = max(0, 38)
    print(f"\n  ── proxy sealed post-run summary {'─' * fill}", flush=True)
    print(f"    requested_max_rounds:  {elicitation_rounds}", flush=True)
    print(f"    actual_rounds:         {actual_rounds}", flush=True)
    print(f"    stopping_rule:         {stopping_rule}", flush=True)
    print(f"    termination_reason:    {termination_reason}", flush=True)
    print(f"    feedback_rule:         {feedback_rule}", flush=True)
    print(f"    refinement_records:    {total_recs}", flush=True)
    print(f"    total_queries_issued:  {total_queries}", flush=True)
    for bidder_id, count in sorted(rq_by_bidder.items()):
        print(f"    {bidder_id:<22}  queries={count}", flush=True)

    if total_recs == 0 and elicitation_rounds > 0:
        if feedback_rule == "none":
            print(
                "    (no records: feedback_rule='none' generates no events)",
                flush=True,
            )
        elif total_queries == 0:
            print(
                "    (no queries issued: max_refinements_per_bidder cap reached "
                "or no competitive bundles found)",
                flush=True,
            )


def print_sealed_proxy_trajectory(rows: list[dict]) -> None:
    """Print a compact per-round trajectory table for the proxy sealed arm."""
    if not rows:
        return
    print(f"\n  ── proxy sealed trajectory {'─' * 33}", flush=True)
    print(
        f"  {'round':>5}  {'true welfare':>12}  {'eff':>7}"
        f"  {'new VQ':>6}  {'cum VQ':>6}  {'alloc Δ':>7}",
        flush=True,
    )
    for row in rows:
        eff = row["global_efficiency"]
        eff_str = _pct(eff) if isinstance(eff, (int, float)) else "—"
        changed = "yes" if row["allocation_changed_from_previous_round"] else "no"
        print(
            f"  {row['round']:>5}  {row['true_welfare']:>12.0f}  {eff_str:>7}"
            f"  {row['new_value_queries']:>6}  {row['cumulative_value_queries']:>6}"
            f"  {changed:>7}",
            flush=True,
        )


def print_proxy_clock_trajectory(top_k: int, round_rows: list[dict]) -> None:
    """Print a compact per-round trajectory table for one proxy clock k run.

    ``true welfare`` is always ground-truth welfare (shown as
    true/full-info-optimum) of the allocation the clock would finalize if it
    stopped at this round -- never *reported* welfare (the WDP objective
    over the clock's accumulated supplementary/reported atoms, available
    separately as each row's ``finalise_reported_welfare``). ``supp atoms``
    is the count of bundles in that accumulated supplementary-atom pool
    (what the finalise-at-round WDP actually runs over), not a proxy's
    static candidate-bundle universe (``candidate_atoms_total``, a
    different, unrelated count).
    """
    if not round_rows:
        return
    print(f"\n  ── proxy clock k={top_k} trajectory {'─' * 27}", flush=True)
    print(
        f"  {'round':>5}  {'true welfare':>13}  {'eff':>7}  {'new_vq':>6}"
        f"  {'cum_vq':>6}  {'supp atoms':>10}  {'events':>6}  {'alloc_changed':>13}",
        flush=True,
    )
    for row in round_rows:
        eff = row["finalise_global_efficiency"]
        eff_str = _pct(eff) if isinstance(eff, (int, float)) else "—"
        welfare_str = (
            f"{row['finalise_true_welfare']:.0f}/{row['full_info_welfare']:.0f}"
        )
        changed = "true" if row["allocation_changed_from_previous_round"] else "false"
        print(
            f"  {row['round']:>5}  {welfare_str:>13}  {eff_str:>7}"
            f"  {row['new_value_queries']:>6}  {row['cumulative_value_queries']:>6}"
            f"  {row['supplementary_atoms_total']:>10}"
            f"  {row['num_events_this_round']:>6}  {changed:>13}",
            flush=True,
        )


def print_event_usefulness_summary(top_k: int, event_rows: list[dict]) -> None:
    """Print per-event-type usefulness stats for one proxy clock k run."""
    if not event_rows:
        return
    by_type: dict[str, list[dict]] = {}
    for row in event_rows:
        by_type.setdefault(row["event_type"], []).append(row)

    print(f"\n  ── proxy clock k={top_k} event usefulness {'─' * 20}", flush=True)
    print(
        f"  {'event_type':<20}  {'count':>5}  {'refinements':>11}"
        f"  {'avg_abs_correction':>18}  {'final_hits':>10}"
        f"  {'pricing_hits':>12}  {'oracle_price_hits':>17}",
        flush=True,
    )
    for event_type in sorted(by_type):
        rows = by_type[event_type]
        corrections = [r["abs_correction"] for r in rows if r["abs_correction"] != ""]
        avg_correction = sum(corrections) / len(corrections) if corrections else 0.0
        final_hits = sum(1 for r in rows if r["appears_in_final_allocation"])
        pricing_hits = sum(
            1 for r in rows if r["appears_in_reported_vcg_counterfactual"]
        )
        oracle_pricing_hits = sum(
            1 for r in rows if r["appears_in_full_info_vcg_counterfactual"]
        )
        print(
            f"  {event_type:<20}  {len(rows):>5}  {len(rows):>11}"
            f"  {avg_correction:>18.1f}  {final_hits:>10}"
            f"  {pricing_hits:>12}  {oracle_pricing_hits:>17}",
            flush=True,
        )


def print_topk_comparison(rows: list[dict]) -> None:
    """Print the end-of-scenario top-k comparison summary.

    ``welfare`` here is always *true* welfare (ground-truth value of the
    final allocation) over the global full-info optimum -- never reported
    (proxy self-declared) welfare, which can overstate true welfare and
    would silently mismatch the efficiency column otherwise. Reported
    welfare is shown separately so the two are never conflated. ``supp
    atoms`` is the clock's final accumulated supplementary/reported atom
    count (what the finalizing WDP ran over), not a proxy's static
    candidate-bundle universe.
    """
    if not rows:
        return
    print(f"\n  ── top-k comparison {'─' * 35}", flush=True)
    print(
        f"  {'k':>3}  {'final true welfare':>19}  {'best true welfare':>19}"
        f"  {'final eff':>9}  {'best eff':>8}  {'best rnd':>8}  {'term':>18}"
        f"  {'reported':>9}  {'cum_vq':>7}  {'supp atoms':>10}"
        f"  {'fi_coverage':>12}  {'failure':>28}",
        flush=True,
    )
    for row in rows:
        welfare_str = f"{row['true_welfare']:.0f}/{row['full_info_welfare']:.0f}"
        eff_str = _pct(row["efficiency"])
        coverage_str = f"{row['fi_winner_coverage_hits']}/{row['fi_winner_coverage_total']}"

        best_welfare = row.get("best_true_welfare")
        best_eff = row.get("best_true_efficiency")
        best_round = row.get("best_round")
        termination = row.get("termination_reason")

        best_welfare_str = (
            f"{best_welfare:.0f}/{row['full_info_welfare']:.0f}"
            if best_welfare is not None
            else "—"
        )
        best_eff_str = _pct(best_eff) if best_eff is not None else "—"
        best_round_str = str(best_round) if best_round is not None else "—"
        term_str = termination if termination else "—"

        print(
            f"  {row['top_k']:>3}  {welfare_str:>19}  {best_welfare_str:>19}"
            f"  {eff_str:>9}  {best_eff_str:>8}  {best_round_str:>8}  {term_str:>18}"
            f"  {row['reported_welfare']:>9.0f}  {row['cumulative_value_queries']:>7}"
            f"  {row['supplementary_atoms_total']:>10}  {coverage_str:>12}"
            f"  {row['failure_classification']:>28}",
            flush=True,
        )


_PERSON_PROMPT_TYPES = {"value_query", "demand_query", "nl_question"}
_VERIFIER_PROMPT_TYPES = {"person_answer_semantic_extraction"}
_PROXY_PROMPT_TYPES = {
    "proxy_nl_gen", "proxy_interest_map", "proxy_provisional_valuations",
    "proxy_interest_map_complement_entailment",
}


def print_nl_sample(proxies: dict) -> None:
    """Print one bidder's NL question/answer and any interest map or PV data."""
    sample_proxy: LlmInferredXorProxy | None = None
    sample_id: str = ""
    for bidder_id, proxy_obj in sorted(proxies.items()):
        inner = getattr(proxy_obj, "proxy", proxy_obj)
        if isinstance(inner, LlmInferredXorProxy) and inner.nl_transcript:
            sample_proxy = inner
            sample_id = bidder_id
            break

    if sample_proxy is None:
        return

    question, answer = sample_proxy.nl_transcript[0]
    print(f"\n  ── NL sample ({sample_id}) {'─' * 36}", flush=True)

    def _wrap(text: str, indent: str = "       ") -> str:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > 72:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return f"\n{indent}".join(lines)

    print(f"    Q  \"{_wrap(question, '       ')}\"", flush=True)
    print(f"    A  \"{_wrap(answer,   '       ')}\"", flush=True)

    if sample_proxy.interest_map is not None:
        im = sample_proxy.interest_map
        print(f"\n    interest map", flush=True)
        print(f"      interested:  {sorted(im.interested_items)}", flush=True)
        if im.complementary_groups:
            print(
                f"      complements: {[sorted(g) for g in im.complementary_groups]}",
                flush=True,
            )
        if im.substitute_groups:
            print(
                "      substitutes: "
                + str([
                    {
                        "items": sorted(group.items),
                        "mode": group.acquisition_mode,
                    }
                    for group in im.substitute_groups
                ]),
                flush=True,
            )
        if im.budget_hint is not None:
            print(f"      budget:      {im.budget_hint}", flush=True)

    if sample_proxy._provisional_raw_values is not None:
        pv = sample_proxy._provisional_raw_values
        print(f"\n    provisional valuations", flush=True)
        for bundle in sorted(pv, key=lambda b: (len(b), sorted(b))):
            bundle_str = "{" + ", ".join(sorted(bundle)) + "}"
            print(f"      {bundle_str:<28}  {pv[bundle]:.1f}", flush=True)

    print(flush=True)


def print_arm_summary(
    arm_label: str,
    proxies: dict,
    token_stats: dict[str, CallTypeStats],
    n_bidders: int,
) -> None:
    """Print query counts and token usage for one elicited arm."""
    total_vq = total_dq = total_rq = 0
    for proxy_obj in proxies.values():
        s = proxy_obj.stats()
        total_vq += s.value_queries
        total_dq += s.demand_queries
        total_rq += s.refinement_queries

    _PROXY_LABEL = {
        "nl_gen": "nl-gen",
        "interest_map": "im",
        "provisional_valuations": "pv",
    }
    proxy_calls: dict[str, int] = {}
    proxy_in: dict[str, int] = {}
    proxy_out: dict[str, int] = {}
    for pt, stats in token_stats.items():
        if pt in _PROXY_PROMPT_TYPES:
            short = pt.replace("proxy_", "")
            proxy_calls[short] = stats.calls
            proxy_in[short] = stats.input_tokens
            proxy_out[short] = stats.output_tokens

    person_in = sum(s.input_tokens for pt, s in token_stats.items() if pt in _PERSON_PROMPT_TYPES)
    person_out = sum(s.output_tokens for pt, s in token_stats.items() if pt in _PERSON_PROMPT_TYPES)
    prx_in = sum(proxy_in.values())
    prx_out = sum(proxy_out.values())
    verifier_calls = sum(
        s.calls for pt, s in token_stats.items() if pt in _VERIFIER_PROMPT_TYPES
    )
    verifier_in = sum(
        s.input_tokens
        for pt, s in token_stats.items()
        if pt in _VERIFIER_PROMPT_TYPES
    )
    verifier_out = sum(
        s.output_tokens
        for pt, s in token_stats.items()
        if pt in _VERIFIER_PROMPT_TYPES
    )
    total_in = person_in + prx_in
    total_out = person_out + prx_out

    fill = max(0, 44 - len(arm_label))
    print(f"\n  ── {arm_label} summary {'─' * fill}", flush=True)
    print(
        f"  {'queries':<30}  {'calls':>5}  {'avg/bidder':>10}",
        flush=True,
    )

    def _qrow(label: str, count: int) -> None:
        avg = f"{count / n_bidders:.1f}" if n_bidders else "—"
        print(f"    {label:<28}  {count:>5}  {avg:>10}", flush=True)

    _qrow("value_queries (initial)", total_vq)
    if total_dq:
        _qrow("demand_queries", total_dq)
    if total_rq:
        _qrow("refinement_queries", total_rq)
    for short, cnt in sorted(proxy_calls.items()):
        _qrow(f"proxy  {_PROXY_LABEL.get(short, short)}", cnt)

    have_tokens = total_in or total_out or verifier_in or verifier_out
    if have_tokens:
        print(
            f"\n  {'tokens':<30}  {'in':>8}  {'out':>8}",
            flush=True,
        )

        def _trow(label: str, tok_in: int, tok_out: int, *, avg: bool = False) -> None:
            if avg and n_bidders:
                label = f"{label} (avg/bidder)"
                tok_in = round(tok_in / n_bidders)
                tok_out = round(tok_out / n_bidders)
            print(f"    {label:<28}  {tok_in:>8}  {tok_out:>8}", flush=True)

        if person_in or person_out:
            _trow("person (vq/dq/nl)", person_in, person_out)
        if prx_in or prx_out:
            _trow("proxy  (nl-gen/im/pv)", prx_in, prx_out)
        if verifier_calls:
            _trow(
                f"offline verifier ({verifier_calls} calls)",
                verifier_in,
                verifier_out,
            )
        _trow("total", total_in, total_out)
        if n_bidders > 1:
            _trow("total", total_in, total_out, avg=True)


def _collect_arm_stats(stats: dict) -> dict:
    return collect_arm_stats(stats)


def _collect_initial_stats(stats: dict) -> dict:
    return collect_initial_stats(stats)


def _print_results_table(summary: list[dict]) -> None:
    """Print a formatted results table comparing all mechanism arms."""
    if not summary:
        return
    scen_w = max(len(r["scenario"]) for r in summary)
    arm_w  = max(len(r["arm"])      for r in summary)

    # Decide whether to show the est-in column (amortized shared or GT est).
    has_est = any(
        r.get("shared_tok_in_amort", 0) > 0 or r.get("est_gt_tok_in", 0) > 0
        for r in summary
    )
    extra_cols = 12 if has_est else 0
    wide = "━" * (scen_w + arm_w + 88 + extra_cols)

    print()
    print(wide)
    hdr = (
        f"  {'scenario':<{scen_w}}  {'arm':<{arm_w}}"
        f"  {'eff':>7}  {'true welfare':>13}"
        f"  {'revenue':>8}  {'surplus':>8}"
        f"  {'tok-in':>8}  {'tok-out':>8}"
    )
    if has_est:
        hdr += f"  {'est-in':>8}"
    hdr += f"  {'vq':>5}  {'dq':>5}  {'nl':>4}"
    print(hdr)
    print(wide)
    for r in summary:
        true_w = r.get("true_welfare", float("nan"))
        full_w = r.get("full_info_welfare", float("nan"))
        nan_w = true_w != true_w or full_w != full_w or not full_w
        welfare_str = (
            "—"
            if nan_w
            else f"{true_w:.0f}/{full_w:.0f}"
        )

        rev = r.get("revenue", float("nan"))
        sur = r.get("surplus", float("nan"))
        rev_str = "—" if rev != rev else f"{rev:.0f}"
        sur_str = "—" if sur != sur else f"{sur:.0f}"

        note = r.get("token_accounting_note", "")
        # Annotate GT query counts in the note when present.
        gt_vq = r.get("gt_vq", 0)
        gt_dq = r.get("gt_dq", 0)
        if gt_vq or gt_dq:
            gt_note = (
                f"gt: vq={gt_vq}" if not gt_dq
                else f"gt: vq={gt_vq} dq={gt_dq}" if gt_vq
                else f"gt: dq={gt_dq}"
            )
            note = f"{note}  {gt_note}" if note else gt_note
        note_suffix = f"  [{note}]" if note else ""

        tok_in = r.get("tok_in", 0)
        tok_out = r.get("tok_out", 0)
        est_in = (
            tok_in
            + r.get("shared_tok_in_amort", 0)
            + r.get("est_gt_tok_in", 0)
        )

        row = (
            f"  {r['scenario']:<{scen_w}}  {r['arm']:<{arm_w}}"
            f"  {_pct(r['efficiency']):>7}"
            f"  {welfare_str:>13}"
            f"  {rev_str:>8}  {sur_str:>8}"
            f"  {tok_in:>8,}  {tok_out:>8,}"
        )
        if has_est:
            row += f"  {est_in:>8,}"
        row += (
            f"  {r.get('vq', 0):>5}  {r.get('dq', 0):>5}  {r.get('nl', 0):>4}"
            + note_suffix
        )
        print(row)
    print(wide)
    print(
        "  NOTE: 'true welfare' is ground-truth welfare (shown as true/full-info-optimum); "
        "'revenue' is VCG revenue computed from each arm's REPORTED bids (not true values); "
        "'surplus' is true surplus = true welfare − revenue, and can be negative when an "
        "arm's reported bids overstate true value.",
        flush=True,
    )
    # Token accounting legend
    has_shared = any(r.get("arm", "") == "shared initial (nl+im+pv)" for r in summary)
    if has_shared:
        n_elicited = sum(
            1
            for r in summary
            if r.get("arm", "") != "shared initial (nl+im+pv)"
            and "shared_tok_in_amort" in r
        )
        print(
            f"  NOTE: 'shared initial' tokens are the one-time NL/IM/PV "
            f"elicitation cost, amortised across {n_elicited} elicited arms "
            f"in the 'est-in' column.",
            flush=True,
        )
    if has_est:
        print(
            "  est-in = actual arm tok-in  +  amortised shared  +  estimated GT query cost",
            flush=True,
        )


def print_ollama_help(model: str) -> None:
    print(
        "Ensure Ollama is running:\n"
        "  ollama serve\n"
        "Ensure model is installed:\n"
        f"  ollama pull {model}",
        file=sys.stderr,
    )


def llm_role_summary_fields(logger: LlmCallLogger) -> dict[str, int]:
    """Aggregate successful logged calls and tokens by person/proxy role."""
    totals = logger.total_stats()
    person_prompt_types = {
        "nl_question",
        "value_query",
        "demand_query",
    }
    person_stats = [
        stats
        for prompt_type, stats in totals.items()
        if prompt_type in person_prompt_types
    ]
    proxy_stats = [
        stats
        for prompt_type, stats in totals.items()
        if prompt_type.startswith("proxy_")
    ]
    return {
        "person_llm_calls": sum(stats.calls for stats in person_stats),
        "person_tokens_in": sum(stats.input_tokens for stats in person_stats),
        "person_tokens_out": sum(stats.output_tokens for stats in person_stats),
        "proxy_llm_calls": sum(stats.calls for stats in proxy_stats),
        "proxy_tokens_in": sum(stats.input_tokens for stats in proxy_stats),
        "proxy_tokens_out": sum(stats.output_tokens for stats in proxy_stats),
    }


def main() -> None:
    # Track which flags were explicitly set before parse_args mutates sys.argv context
    _explicitly_set = explicitly_set_args()

    args = parse_args()

    resolve_llm_role_args(args)
    resolve_person_query_mode(args)
    resolve_initial_elicitation_flags(args)
    try:
        resolve_event_policy(args, _explicitly_set)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    _deprecations: list[str] = []
    try:
        calibration = resolve_cli_calibration(
            args,
            _explicitly_set,
            warn=_deprecations.append,
        )
    except CalibrationConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    try:
        scenarios = select_scenarios(
            args.scenario,
            args.seed_type,
            num_goods=args.num_goods,
            num_bidders=args.num_bidders,
            scenario_seed=args.scenario_seed,
            scenario_spec=args.scenario_spec,
            selection_policy=args.selection_policy,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.prepare_elicitation_only and args.write_elicitation_pack is None:
        print(
            "Error: --prepare-elicitation-only requires "
            "--write-elicitation-pack",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.write_elicitation_pack is not None and not (
        args.use_interest_map and args.use_provisional_valuations
    ):
        print(
            "Error: frozen elicitation format v1 requires both "
            "--use-interest-map and --use-provisional-valuations",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.elicitation_pack is not None and args.write_elicitation_pack is not None:
        print(
            "Error: --elicitation-pack and --write-elicitation-pack are "
            "mutually exclusive",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.elicitation_pack is not None and args.disclosure_pack is not None:
        print(
            "Error: --elicitation-pack and --disclosure-pack are mutually "
            "exclusive",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if (
        args.disclosure_pack is not None
        and args.write_elicitation_pack is None
    ):
        print(
            "Error: --disclosure-pack requires --write-elicitation-pack",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if (
        args.elicitation_pack is not None
        or args.disclosure_pack is not None
        or args.write_elicitation_pack is not None
    ) and len(scenarios) != 1:
        print(
            "Error: frozen elicitation currently requires exactly one "
            "materialised scenario per invocation",
            file=sys.stderr,
        )
        raise SystemExit(2)

    frozen_pack = None
    disclosure_pack = None
    if args.elicitation_pack is not None:
        try:
            frozen_pack = load_frozen_elicitation_pack(
                args.elicitation_pack
            )
            validate_pack_for_scenario(
                frozen_pack,
                scenarios[0],
                scenario_spec_path=args.scenario_spec,
            )
        except (OSError, ValueError) as exc:
            print(f"Error loading elicitation pack: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        settings = frozen_pack.generation_settings
        args.ask_initial_question = True
        args.use_interest_map = bool(
            settings.get("use_interest_map", True)
        )
        args.use_provisional_valuations = bool(
            settings.get("use_provisional_valuations", True)
        )
        # Initial-role provenance comes from the immutable pack, not from
        # whichever provider/model defaults happen to be on this replay CLI.
        args.person_provider = frozen_pack.person_model.provider
        args.person_model = frozen_pack.person_model.model
        args.person_temperature = frozen_pack.person_model.temperature or 0.0
        args.proxy_provider = frozen_pack.proxy_model.provider
        args.proxy_model = frozen_pack.proxy_model.model
        args.proxy_temperature = frozen_pack.proxy_model.temperature or 0.0
    elif args.disclosure_pack is not None:
        try:
            disclosure_pack = load_frozen_elicitation_pack(
                args.disclosure_pack
            )
            validate_pack_for_scenario(
                disclosure_pack,
                scenarios[0],
                scenario_spec_path=args.scenario_spec,
            )
        except (OSError, ValueError) as exc:
            print(f"Error loading disclosure pack: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        args.ask_initial_question = True
        args.use_interest_map = True
        args.use_provisional_valuations = True
        # The person role is immutable in this treatment; only the proxy role
        # is regenerated with the current provider/model.
        args.person_provider = disclosure_pack.person_model.provider
        args.person_model = disclosure_pack.person_model.model
        args.person_temperature = (
            disclosure_pack.person_model.temperature or 0.0
        )

    # Full config header
    for line in format_run_config(args, scenarios, calibration=calibration):
        print(line, flush=True)

    for _deprecation in _deprecations:
        print(f"\n  DEPRECATION: {_deprecation}", flush=True)

    # Config warnings / notes
    for warning in config_warnings(args):
        print(f"\n  {warning}", flush=True)

    log_dir = Path(args.log_dir)
    log_path = log_dir / "calls.jsonl"
    sealed_path = log_dir / "curated_sealed_llm_comparison.csv"
    clock_paths = {
        top_k: log_dir / f"curated_clock_llm_comparison_top_{top_k}.csv"
        for top_k in args.top_k
    }
    sealed_proxy_path = log_dir / "curated_sealed_proxy_elicited.csv"
    sealed_proxy_trajectory_path = log_dir / "curated_proxy_sealed_trajectory.csv"
    clock_proxy_paths = {
        top_k: log_dir / f"curated_clock_proxy_elicited_top_{top_k}.csv"
        for top_k in args.top_k
    }
    clock_round_paths = {
        top_k: log_dir / f"curated_proxy_clock_rounds_top_{top_k}.csv"
        for top_k in args.top_k
    }
    clock_bidder_round_paths = {
        top_k: log_dir / f"curated_proxy_clock_bidder_rounds_top_{top_k}.csv"
        for top_k in args.top_k
    }
    clock_coverage_paths = {
        top_k: log_dir / f"curated_proxy_clock_coverage_top_{top_k}.csv"
        for top_k in args.top_k
    }
    clock_event_paths = {
        top_k: log_dir / f"curated_proxy_clock_event_usefulness_top_{top_k}.csv"
        for top_k in args.top_k
    }
    person_disclosures_path = log_dir / "curated_person_disclosures.csv"
    refinement_path = log_dir / "curated_refinement_records.csv"
    pv_candidate_bundle_stats_path = log_dir / "curated_pv_candidate_bundle_stats.csv"
    run_summary_path = log_dir / "curated_run_summary.csv"
    run_config_path = log_dir / "run_config.json"

    # Written before any mechanism runs so the effective calibration is on
    # disk even if the run later fails.
    write_run_config_json(
        run_config_path,
        build_run_config_document(
            args,
            calibration=calibration,
            scenarios=scenarios,
            extra={
                "deprecation_warnings": list(_deprecations),
                "top_k": list(args.top_k),
            },
        ),
    )
    _calibration_fields = calibration_summary_fields(calibration)

    # Each invocation owns its calls.jsonl. Reusing a log directory must not
    # silently mix this run's accounting with an earlier run.
    logger = LlmCallLogger(log_path, append=False)
    llm_cache = (
        LlmResponseCache(args.llm_cache_path)
        if args.llm_cache_mode != "off"
        else None
    )
    llm_cache_stats = CacheStats()
    print(
        f"  llm_cache_mode            {args.llm_cache_mode}"
        + (f"  path={args.llm_cache_path}" if llm_cache is not None else ""),
        flush=True,
    )
    cfg = ClockConfig(
        max_rounds=args.max_rounds,
        price_step=args.price_step,
        reserve=args.reserve,
    )
    sealed_rows = []
    clock_rows_by_top_k = {top_k: [] for top_k in args.top_k}
    sealed_proxy_rows = []
    clock_proxy_rows_by_top_k = {top_k: [] for top_k in args.top_k}
    clock_round_rows_by_top_k = {top_k: [] for top_k in args.top_k}
    clock_bidder_round_rows_by_top_k = {top_k: [] for top_k in args.top_k}
    clock_coverage_rows_by_top_k = {top_k: [] for top_k in args.top_k}
    clock_event_rows_by_top_k = {top_k: [] for top_k in args.top_k}
    _summary: list[dict] = []
    all_refinement_rows: list[dict] = []
    all_trajectory_rows: list[dict] = []
    all_pv_candidate_bundle_stats_rows: list[dict] = []
    all_person_disclosure_rows: list[dict] = []

    for scenario in scenarios:
        max_bundle_size = min(
            args.max_bundle_size,
            len(scenario.instance.items),
        )
        candidate_bundles = generate_candidate_bundles(
            scenario.instance.items,
            max_bundle_size=max_bundle_size,
        )
        candidate_bundles_by_bidder = None

        print(f"\n▶  {scenario.name}", flush=True)

        _scenario_md = getattr(scenario, "metadata", {}) or {}
        _scenario_seed = _scenario_md.get("scenario_seed", args.scenario_seed)
        _scenario_num_goods = _scenario_md.get(
            "num_goods", len(scenario.instance.items)
        )
        _scenario_num_bidders = _scenario_md.get(
            "num_bidders", len(scenario.instance.bidder_ids)
        )
        _scenario_opening_question = (
            args.opening_question.strip()
            if args.opening_question is not None
            else (
                canonical_opening_question(
                    domain=_scenario_md.get("domain")
                )
                if args.opening_question_policy == "canonical"
                else None
            )
        )
        if _scenario_opening_question == "":
            raise ValueError("--opening-question must be non-empty")
        _scenario_verifier_provider = (
            args.verifier_provider
            or _scenario_md.get("environment_generation_provider")
            or args.proxy_provider
        )
        _scenario_verifier_model = (
            args.verifier_model
            or _scenario_md.get("environment_generation_model")
            or args.proxy_model
        )

        all_person_disclosure_rows.extend(
            person_disclosure_rows_for_scenario(scenario)
        )

        bundles_by_bidder = {
            bidder_id: candidate_bundles
            for bidder_id in scenario.instance.bidder_ids
        }

        _nl_sample_shown = [False]
        _n_bidders = len(scenario.instance.bidder_ids)
        _topk_comparison = []

        _persons_cache: list[dict[str, LlmPersonSimulator] | None] = [None]
        _elicitation_cache: list[
            dict[str, BidderElicitationData] | None
        ] = [None]

        def _get_persons() -> dict[str, LlmPersonSimulator]:
            if _persons_cache[0] is None:
                if (
                    (frozen_pack is not None or disclosure_pack is not None)
                    and args.ground_truth_queries
                ):
                    # Frozen replay plus deterministic refinements requires no
                    # network client.  The placeholder would fail loudly if a
                    # future code path accidentally attempted a live call.
                    _persons_cache[0] = {
                        bidder_id: LlmPersonSimulator(
                            bidder_id=bidder_id,
                            scenario_description=scenario.scenario_description,
                            person_seed=scenario.person_seeds[bidder_id],
                            item_descriptions=scenario.item_descriptions,
                            client=MockLlmClient(responses=[]),
                            logger=logger,
                            model_name=args.person_model,
                            provider_name=args.person_provider,
                            max_parse_retries=args.max_parse_retries,
                            ground_truth_valuations=(
                                scenario.instance.valuations[bidder_id]
                            ),
                            verbose=args.verbose,
                            scenario_id=scenario.name,
                        )
                        for bidder_id in scenario.instance.bidder_ids
                    }
                else:
                    _persons_cache[0] = make_live_persons_for_scenario(
                        scenario,
                        model=args.person_model,
                        provider=args.person_provider,
                        base_url=args.person_base_url,
                        api_key=args.person_api_key,
                        temperature=args.person_temperature,
                        max_tokens=args.max_tokens,
                        person_nl_max_tokens=args.person_nl_max_tokens,
                        timeout=args.timeout,
                        logger=logger,
                        max_parse_retries=args.max_parse_retries,
                        use_ground_truth=args.ground_truth_queries,
                        verbose=args.verbose,
                        cache=llm_cache,
                        cache_mode=args.llm_cache_mode,
                        cache_stats=llm_cache_stats,
                        verifier_provider=_scenario_verifier_provider,
                        verifier_model=_scenario_verifier_model,
                        verifier_base_url=args.verifier_base_url,
                        verifier_api_key=args.verifier_api_key,
                        verifier_temperature=args.verifier_temperature,
                        verifier_max_tokens=args.verifier_max_tokens,
                    )
            return _persons_cache[0]

        def _get_elicitation_cache() -> dict[str, BidderElicitationData]:
            if _elicitation_cache[0] is None:
                if frozen_pack is not None:
                    _elicitation_cache[0] = dict(frozen_pack.bidders)
                    print(
                        f"  frozen elicitation replay → "
                        f"{args.elicitation_pack}",
                        flush=True,
                    )
                else:
                    pv_client = None
                    question_client = (
                        make_live_client(
                            model=args.proxy_model,
                            provider=args.proxy_provider,
                            base_url=args.proxy_base_url,
                            api_key=args.proxy_api_key,
                            temperature=args.proxy_temperature,
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                            cache=llm_cache,
                            cache_mode=args.llm_cache_mode,
                            cache_stats=llm_cache_stats,
                            llm_role="proxy",
                        )
                        if _scenario_opening_question is None
                        else None
                    )
                    interest_map_client = make_live_client(
                        model=args.proxy_model,
                        provider=args.proxy_provider,
                        base_url=args.proxy_base_url,
                        api_key=args.proxy_api_key,
                        temperature=args.proxy_temperature,
                        max_tokens=args.interest_map_max_tokens,
                        timeout=args.timeout,
                        cache=llm_cache,
                        cache_mode=args.llm_cache_mode,
                        cache_stats=llm_cache_stats,
                        llm_role="proxy",
                    )
                    if args.use_provisional_valuations:
                        pv_client = make_live_client(
                            model=args.proxy_model,
                            provider=args.proxy_provider,
                            base_url=args.proxy_base_url,
                            api_key=args.proxy_api_key,
                            temperature=args.proxy_temperature,
                            max_tokens=args.pv_max_tokens,
                            timeout=args.timeout,
                            cache=llm_cache,
                            cache_mode=args.llm_cache_mode,
                            cache_stats=llm_cache_stats,
                            llm_role="proxy",
                        )
                    _elicitation_cache[0] = compute_elicitation_cache(
                        scenario=scenario,
                        persons=_get_persons(),
                        use_provisional_valuations=(
                            args.use_provisional_valuations
                        ),
                        max_candidate_bundles=args.max_candidate_bundles,
                        pv_client=pv_client,
                        question_client=question_client,
                        opening_question=(
                            None
                            if disclosure_pack is not None
                            else _scenario_opening_question
                        ),
                        interest_map_client=interest_map_client,
                        pv_chunk_size=args.pv_chunk_size,
                        pv_failure_policy=args.pv_failure_policy,
                        interest_map_failure_policy=(
                            args.interest_map_failure_policy
                        ),
                        pv_max_tokens=args.pv_max_tokens,
                        max_parse_retries=args.max_parse_retries,
                        fixed_disclosures=(
                            None
                            if disclosure_pack is None
                            else disclosure_pack.bidders
                        ),
                    )
                    if args.write_elicitation_pack is not None:
                        pack = build_frozen_elicitation_pack(
                            scenario=scenario,
                            entries=_elicitation_cache[0],
                            scenario_spec_path=args.scenario_spec,
                            selection_policy=args.selection_policy,
                            person_model=(
                                disclosure_pack.person_model
                                if disclosure_pack is not None
                                else ModelProvenance(
                                    provider=args.person_provider,
                                    model=args.person_model,
                                    temperature=effective_model_temperature(
                                        args.person_provider,
                                        args.person_model,
                                        args.person_temperature,
                                    ),
                                )
                            ),
                            proxy_model=ModelProvenance(
                                provider=args.proxy_provider,
                                model=args.proxy_model,
                                temperature=effective_model_temperature(
                                    args.proxy_provider,
                                    args.proxy_model,
                                    args.proxy_temperature,
                                ),
                            ),
                            generation_settings={
                                "use_interest_map": args.use_interest_map,
                                "use_provisional_valuations": (
                                    args.use_provisional_valuations
                                ),
                                "max_candidate_bundles": (
                                    args.max_candidate_bundles
                                ),
                                "pv_chunk_size": args.pv_chunk_size,
                                "pv_failure_policy": args.pv_failure_policy,
                                "interest_map_failure_policy": (
                                    args.interest_map_failure_policy
                                ),
                                "max_parse_retries": args.max_parse_retries,
                                "opening_question_policy": (
                                    "explicit"
                                    if args.opening_question is not None
                                    else args.opening_question_policy
                                ),
                                "fixed_disclosure_source": (
                                    None
                                    if args.disclosure_pack is None
                                    else str(args.disclosure_pack)
                                ),
                                "opening_question": next(iter(
                                    _elicitation_cache[0].values()
                                )).nl_question,
                                "verifier_provider": (
                                    _scenario_verifier_provider
                                ),
                                "verifier_model": _scenario_verifier_model,
                                "closed_world_interest_map": True,
                                "deterministic_refinements_unscaled": True,
                            },
                            generation_calls=logger.records(),
                        )
                        write_frozen_elicitation_pack(
                            pack, args.write_elicitation_pack
                        )
                        print(
                            f"  frozen elicitation pack  → "
                            f"{args.write_elicitation_pack}",
                            flush=True,
                        )
                for bidder_id, entry in _elicitation_cache[0].items():
                    stats = entry.pv_candidate_stats
                    if stats is None:
                        continue
                    chunk_stats = entry.pv_chunk_stats or PvChunkStats(
                        pv_chunk_size=args.pv_chunk_size,
                        pv_chunks=0,
                        candidate_count=stats.candidate_bundles_sent_to_pv,
                        per_chunk_bundle_counts=(),
                        chunking_used=False,
                    )
                    all_pv_candidate_bundle_stats_rows.append({
                        "scenario": scenario.name,
                        "bidder_id": bidder_id,
                        "person_provider": args.person_provider,
                        "person_model": args.person_model,
                        "proxy_provider": args.proxy_provider,
                        "proxy_model": args.proxy_model,
                        **stats.as_dict(),
                        **chunk_stats.as_dict(),
                        "pv_degraded": entry.pv_degraded,
                        "interest_map_interested_count": len(entry.interest_map.interested_items),
                        "interest_map_excluded_count": len(entry.interest_map.excluded_items),
                        "interest_map_substitute_group_count": len(entry.interest_map.substitute_groups),
                        "interest_map_choose_one_group_count": sum(
                            group.acquisition_mode == "choose_one"
                            for group in entry.interest_map.substitute_groups
                        ),
                        "interest_map_can_use_multiple_group_count": sum(
                            group.acquisition_mode == "can_use_multiple"
                            for group in entry.interest_map.substitute_groups
                        ),
                        "interest_map_unclear_group_count": sum(
                            group.acquisition_mode == "unclear"
                            for group in entry.interest_map.substitute_groups
                        ),
                        "interest_map_complement_group_count": len(entry.interest_map.complementary_groups),
                        "interest_map_candidate_count_before_filter": entry.interest_map_candidate_count_before_filter,
                        "interest_map_candidate_count_after_filter": entry.interest_map_candidate_count_after_filter,
                        "interest_map_fallback_used": entry.interest_map_fallback_used,
                        "interest_map_quality_flags": "|".join(entry.interest_map_quality_flags),
                        **{
                            f"interest_map_accuracy_{key}": value
                            for key, value in (
                                entry.interest_map_accuracy or {}
                            ).items()
                            if not isinstance(value, (list, dict))
                        },
                        "interest_map_accuracy_details": json.dumps(
                            entry.interest_map_accuracy or {},
                            sort_keys=True,
                        ),
                        "pv_scale_quality_flags": "|".join(entry.pv_scale_quality_flags),
                        "pv_inferred_max_value": entry.pv_inferred_max_value,
                        "pv_ground_truth_max_value": entry.pv_ground_truth_max_value,
                        "pv_max_value_ratio": entry.pv_max_value_ratio,
                        "pv_inferred_max_singleton": entry.pv_inferred_max_singleton,
                        "pv_ground_truth_max_singleton": entry.pv_ground_truth_max_singleton,
                        "pv_max_singleton_ratio": entry.pv_max_singleton_ratio,
                    })
            return _elicitation_cache[0]

        _shared_direct_question = [_scenario_opening_question]

        def make_elicited_proxies(*, use_pv: bool = True):
            persons = _get_persons()

            if args.use_interest_map or args.use_provisional_valuations:
                cache = _get_elicitation_cache()

                def get_llm_adapter(bidder_id: str) -> LlmAuctionProxyAdapter:
                    entry = cache[bidder_id]
                    proxy = LlmInferredXorProxy(
                        bidder_id=bidder_id,
                        person=persons[bidder_id],
                        calibration=calibration,
                        # Inert once `calibration` is set; pinned to the
                        # neutral 1.0 so the legacy field never reads as an
                        # applied discount in downstream metadata.
                        epsilon=1.0,
                    )
                    proxy.replay_elicitation(
                        nl_question=entry.nl_question,
                        nl_answer=entry.nl_answer,
                        interest_map=entry.interest_map,
                        provisional_raw_values=entry.raw_pv_values if use_pv else None,
                        discount_inferred=args.discount_inferred,
                    )
                    return LlmAuctionProxyAdapter(
                        bidder_id=bidder_id,
                        proxy=proxy,
                        candidate_bundles=entry.candidate_bundles,
                        discount_inferred=args.discount_inferred,
                        use_anchor_values=not args.disable_anchor_values,
                        refinement_strategy=args.refinement_strategy,
                    )
            else:
                def get_llm_adapter(bidder_id: str) -> LlmAuctionProxyAdapter:  # type: ignore[no-redef]
                    person = persons[bidder_id]
                    proxy = LlmInferredXorProxy(
                        bidder_id=bidder_id,
                        person=person,
                        calibration=calibration,
                        # Inert once `calibration` is set; pinned to the
                        # neutral 1.0 so the legacy field never reads as an
                        # applied discount in downstream metadata.
                        epsilon=1.0,
                    )
                    if args.ask_initial_question:
                        proxy.ask_initial_question(
                            question=_shared_direct_question[0]
                        )
                        if _shared_direct_question[0] is None:
                            _shared_direct_question[0] = (
                                proxy.nl_transcript[-1][0]
                            )
                    return LlmAuctionProxyAdapter(
                        bidder_id=bidder_id,
                        proxy=proxy,
                        candidate_bundles=bundles_by_bidder[bidder_id],
                        discount_inferred=args.discount_inferred,
                        use_anchor_values=not args.disable_anchor_values,
                        refinement_strategy=args.refinement_strategy,
                    )

            result_proxies = {
                bidder_id: get_llm_adapter(bidder_id)
                for bidder_id in persons
            }

            if not _nl_sample_shown[0]:
                _nl_sample_shown[0] = True
                print_nl_sample(result_proxies)

            return result_proxies

        # ----------------------------------------------------------------
        # Pre-capture shared initial elicitation cost (runs once per
        # scenario; all subsequent make_elicited_proxies() calls are free
        # because the cache is already populated).
        # ----------------------------------------------------------------
        _needs_elicited = (
            args.sealed_elicitation_rounds > 0
            or args.elicited_clock
            or args.prepare_elicitation_only
        ) and (args.use_interest_map or args.use_provisional_valuations)

        # Number of elicited arms that amortize the shared initial cost.
        _n_elicited_arms: int = (
            (1 if args.sealed_elicitation_rounds > 0 else 0)
            + (len(args.top_k) if args.elicited_clock else 0)
        )
        _shared_tok_in: int = 0
        _shared_tok_out: int = 0

        if _needs_elicited:
            logger.mark()
            _elicitation_cache_result = _get_elicitation_cache()
            _shared_initial_stats = (
                call_stats_from_records(
                    frozen_pack.generation_calls,
                    logical_cached_tokens=True,
                )
                if frozen_pack is not None
                else logger.stats_since_mark()
            )
            _shared_initial_row = _collect_initial_stats(_shared_initial_stats)
            _shared_tok_in = _shared_initial_row["tok_in"]
            _shared_tok_out = _shared_initial_row["tok_out"]
            # pv_bidders/pv_chunks are PV/shared-initial accounting, never
            # value or refinement queries: pv_bidders counts bidders with a
            # (possibly degraded/zero) PV table, pv_chunks counts the total
            # number of PV LLM calls actually made (1 per bidder unless
            # --pv-chunk-size split some bidders into multiple chunk calls).
            _pv_bidders = sum(
                1 for e in _elicitation_cache_result.values()
                if e.raw_pv_values is not None
            )
            _pv_chunks_total = sum(
                e.pv_chunk_stats.pv_chunks
                for e in _elicitation_cache_result.values()
                if e.pv_chunk_stats is not None
            )
            _pv_degraded_bidders = sum(
                1 for e in _elicitation_cache_result.values() if e.pv_degraded
            )
            if args.use_provisional_valuations:
                _shared_initial_row["pv_bidders"] = _pv_bidders
                _shared_initial_row["pv_chunks"] = _pv_chunks_total
                _shared_initial_row["pv_degraded_bidders"] = _pv_degraded_bidders
                _shared_initial_row["token_accounting_note"] = (
                    f"{_shared_initial_row['token_accounting_note']}  "
                    f"pv_bidders={_pv_bidders}  pv_chunks={_pv_chunks_total}"
                    + (
                        f"  pv_degraded={_pv_degraded_bidders}"
                        if _pv_degraded_bidders
                        else ""
                    )
                )
            _summary.append({
                "scenario": scenario.name,
                "arm": "shared initial (nl+im+pv)",
                "efficiency": float("nan"),
                "true_welfare": float("nan"),
                "full_info_welfare": float("nan"),
                **_shared_initial_row,
            })

        if args.prepare_elicitation_only:
            continue

        # Reused for allocation/pricing-witness annotations on every
        # refinement record. This is deterministic and independent of
        # whether the ordinary full-information baseline arm is skipped.
        _full_info_witness_result = run_sealed_vcg_experiment(
            scenario.instance
        )

        def _amortized_shared() -> dict:
            """Per-arm share of the shared initial elicitation cost."""
            if _n_elicited_arms <= 0:
                return {"shared_tok_in_amort": 0, "shared_tok_out_amort": 0}
            return {
                "shared_tok_in_amort": _shared_tok_in // _n_elicited_arms,
                "shared_tok_out_amort": _shared_tok_out // _n_elicited_arms,
            }

        def _est_gt_tok(arm_stats: dict) -> dict:
            """Estimated token cost for ground-truth queries in this arm."""
            est_per_vq = getattr(args, "est_tok_per_vq", 0)
            est_per_dq = getattr(args, "est_tok_per_dq", 0)
            gt_vq = arm_stats.get("gt_vq", 0)
            gt_dq = arm_stats.get("gt_dq", 0)
            return {
                "est_gt_tok_in": gt_vq * est_per_vq + gt_dq * est_per_dq,
                "est_gt_tok_out": 0,
            }

        mechanism = "sealed"
        _sealed_comparison_true_welfare: float | None = None
        try:
            if not args.skip_baselines:
                _section("sealed comparison")
                logger.mark()
                sealed = run_sealed_llm_comparison(
                    instance=scenario.instance,
                    instance_name=scenario.name,
                    proxies=make_live_proxies_for_scenario(
                        scenario,
                        model=args.model,
                        provider=args.provider,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        person_nl_max_tokens=args.person_nl_max_tokens,
                        timeout=args.timeout,
                        logger=logger,
                        max_parse_retries=args.max_parse_retries,
                        calibration=calibration,
                        ask_initial_question=args.ask_initial_question,
                        opening_question=_scenario_opening_question,
                        use_ground_truth=args.ground_truth_queries,
                        verbose=args.verbose,
                        cache=llm_cache,
                        cache_mode=args.llm_cache_mode,
                        cache_stats=llm_cache_stats,
                        verifier_provider=_scenario_verifier_provider,
                        verifier_model=_scenario_verifier_model,
                        verifier_base_url=args.verifier_base_url,
                        verifier_api_key=args.verifier_api_key,
                        verifier_temperature=args.verifier_temperature,
                        verifier_max_tokens=args.verifier_max_tokens,
                    ),
                    candidate_bundles=candidate_bundles,
                    candidate_bundles_by_bidder=candidate_bundles_by_bidder,
                    discount_inferred=args.discount_inferred,
                    use_anchor_values=not args.disable_anchor_values,
                )
                sealed_row = sealed_llm_comparison_to_row(sealed)
                sealed_rows.append(sealed_row)
                _sealed_comparison_true_welfare = sealed_row["llm_proxy_true_welfare"]
                _arm_result(
                    "sealed comparison",
                    sealed_row["efficiency"],
                    sealed_row["llm_proxy_true_welfare"],
                    sealed_row["full_info_welfare"],
                    reported_welfare=sealed_row["llm_proxy_reported_welfare"],
                )
                _summary.append({
                    "scenario": scenario.name,
                    "arm": "sealed",
                    "efficiency": sealed_row["efficiency"],
                    "true_welfare": sealed_row["llm_proxy_true_welfare"],
                    "full_info_welfare": sealed_row["full_info_welfare"],
                    "revenue": sealed_row["llm_proxy_revenue"],
                    "surplus": sealed_row["llm_proxy_true_welfare"] - sealed_row["llm_proxy_revenue"],
                    **_collect_arm_stats(logger.stats_since_mark()),
                })

            if args.sealed_elicitation_rounds > 0:
                mechanism = "sealed proxy elicitation"
                _section("sealed proxy elicitation")
                logger.mark()
                _elicited = make_elicited_proxies()
                _sealed_config = ProxySealedConfig(
                    elicitation_rounds=args.sealed_elicitation_rounds,
                    feedback_rule=args.sealed_feedback_rule,
                    stopping_rule=args.sealed_stopping_rule,
                    loser_challenger_policy=(
                        args.sealed_loser_challenger_policy
                    ),
                    max_refinements_per_bidder=(
                        args.max_refinement_queries_per_bidder
                    ),
                    max_total_refinements=args.max_total_refinement_queries,
                    incumbent_verification=(
                        args.event_incumbent_verification
                    ),
                    pivotal_challengers=args.event_pivotal_challengers,
                    pivotal_gap_threshold=(
                        args.event_pivotal_gap_threshold
                    ),
                    scarcity_fallbacks=args.event_scarcity_fallbacks,
                    large_correction_followup=(
                        args.sealed_event_large_correction_followup
                    ),
                    correction_followup_threshold=(
                        args.event_correction_threshold
                    ),
                    terminal_regret_audit=(
                        args.event_terminal_regret_audit
                    ),
                )
                if args.sealed_trajectory:
                    _proxy_sealed_trajectory = run_proxy_sealed_vcg_trajectory(
                        instance=scenario.instance,
                        proxies=list(_elicited.values()),
                        config=_sealed_config,
                        logger=logger,
                        scenario_name=scenario.name,
                    )
                    proxy_sealed_result = _proxy_sealed_trajectory[-1]

                    _trajectory_rows = proxy_sealed_trajectory_to_rows(
                        scenario_name=scenario.name,
                        scenario_seed=_scenario_seed,
                        num_goods=_scenario_num_goods,
                        num_bidders=_scenario_num_bidders,
                        instance=scenario.instance,
                        trajectory=_proxy_sealed_trajectory,
                        comparison_welfare=_sealed_comparison_true_welfare,
                    )
                    all_trajectory_rows.extend(_trajectory_rows)
                    print_sealed_proxy_trajectory(_trajectory_rows)
                else:
                    proxy_sealed_result = run_proxy_sealed_vcg_experiment(
                        instance=scenario.instance,
                        proxies=list(_elicited.values()),
                        config=_sealed_config,
                        scenario_name=scenario.name,
                    )
                proxy_sealed_row = proxy_sealed_result_to_row(
                    instance_name=scenario.name,
                    instance=scenario.instance,
                    result=proxy_sealed_result,
                )
                sealed_proxy_rows.append(proxy_sealed_row)
                _arm_result(
                    "proxy sealed",
                    proxy_sealed_row["efficiency"],
                    proxy_sealed_row["proxy_true_welfare"],
                    proxy_sealed_row["full_info_welfare"],
                    extra=f"queries {proxy_sealed_row['refinement_query_count_by_bidder']}",
                    reported_welfare=proxy_sealed_row["proxy_reported_welfare"],
                )
                _proxy_sealed_stats = logger.stats_since_mark()
                _ps_arm = _collect_arm_stats(_proxy_sealed_stats)
                _summary.append({
                    "scenario": scenario.name,
                    "arm": "proxy sealed",
                    "efficiency": proxy_sealed_row["efficiency"],
                    "true_welfare": proxy_sealed_row["proxy_true_welfare"],
                    "full_info_welfare": proxy_sealed_row["full_info_welfare"],
                    "revenue": proxy_sealed_row["proxy_revenue"],
                    "surplus": proxy_sealed_row["proxy_true_welfare"] - proxy_sealed_row["proxy_revenue"],
                    "requested_elicitation_rounds": proxy_sealed_row[
                        "requested_elicitation_rounds"
                    ],
                    "actual_elicitation_rounds": proxy_sealed_row[
                        "elicitation_rounds"
                    ],
                    "sealed_stopping_rule": proxy_sealed_row[
                        "stopping_rule"
                    ],
                    "sealed_termination_reason": proxy_sealed_row[
                        "termination_reason"
                    ],
                    **_ps_arm,
                    **_amortized_shared(),
                    **_est_gt_tok(_ps_arm),
                })
                print_refinement_records(
                    proxy_sealed_result.metadata["refinement_records_by_bidder"],
                )
                _print_sealed_proxy_summary(
                    proxy_sealed_result,
                    feedback_rule=args.sealed_feedback_rule,
                    elicitation_rounds=args.sealed_elicitation_rounds,
                )
                all_refinement_rows.extend(
                    refinement_records_to_rows(
                        scenario.name,
                        f"proxy_sealed_{args.sealed_feedback_rule}",
                        proxy_sealed_result.metadata["refinement_records_by_bidder"],
                        final_allocation=proxy_sealed_result.allocation,
                        reported_vcg_counterfactuals=proxy_sealed_result.metadata[
                            "vcg_counterfactuals"
                        ],
                        full_info_allocation=(
                            _full_info_witness_result.allocation
                        ),
                        full_info_vcg_counterfactuals=(
                            _full_info_witness_result.metadata[
                                "vcg_counterfactuals"
                            ]
                        ),
                    )
                )
                if args.verbose:
                    print_arm_summary(
                        "proxy sealed",
                        _elicited,
                        _proxy_sealed_stats,
                        _n_bidders,
                    )

            for top_k in args.top_k:
                if not args.skip_baselines:
                    mechanism = f"clock top_k={top_k}"
                    _section(f"clock comparison  top_k={top_k}")
                    logger.mark()
                    clock = run_clock_llm_comparison(
                        instance=scenario.instance,
                        instance_name=scenario.name,
                        proxies=make_live_proxies_for_scenario(
                            scenario,
                            model=args.model,
                            provider=args.provider,
                            base_url=args.base_url,
                            api_key=args.api_key,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            person_nl_max_tokens=args.person_nl_max_tokens,
                            timeout=args.timeout,
                            logger=logger,
                            max_parse_retries=args.max_parse_retries,
                            calibration=calibration,
                            ask_initial_question=args.ask_initial_question,
                            opening_question=_scenario_opening_question,
                            use_ground_truth=args.ground_truth_queries,
                            verbose=args.verbose,
                            cache=llm_cache,
                            cache_mode=args.llm_cache_mode,
                            cache_stats=llm_cache_stats,
                            verifier_provider=_scenario_verifier_provider,
                            verifier_model=_scenario_verifier_model,
                            verifier_base_url=args.verifier_base_url,
                            verifier_api_key=args.verifier_api_key,
                            verifier_temperature=args.verifier_temperature,
                            verifier_max_tokens=args.verifier_max_tokens,
                        ),
                        candidate_bundles=candidate_bundles,
                        candidate_bundles_by_bidder=(
                            candidate_bundles_by_bidder
                        ),
                        cfg=cfg,
                        top_k=top_k,
                        discount_inferred=args.discount_inferred,
                        use_anchor_values=not args.disable_anchor_values,
                        elicited=args.elicited_clock,
                        margin_threshold=args.clock_margin_threshold,
                        tie_threshold=args.clock_tie_threshold,
                        max_refinement_queries_per_bidder=(
                            args.max_refinement_queries_per_bidder
                        ),
                        refinement_strategy=args.refinement_strategy,
                    )
                    clock_row = clock_llm_comparison_to_row(clock)
                    clock_rows_by_top_k[top_k].append(clock_row)
                    _arm_result(
                        f"clock k={top_k}",
                        clock_row["efficiency"],
                        clock_row["clock_llm_true_welfare"],
                        clock_row["full_info_welfare"],
                        extra=f"rounds {clock_row['clock_rounds']}",
                        reported_welfare=clock_row["clock_llm_reported_welfare"],
                    )
                    _summary.append({
                        "scenario": scenario.name,
                        "arm": f"clock k={top_k}",
                        "efficiency": clock_row["efficiency"],
                        "true_welfare": clock_row["clock_llm_true_welfare"],
                        "full_info_welfare": clock_row["full_info_welfare"],
                        "revenue": clock_row["clock_llm_revenue"],
                        "surplus": clock_row["clock_llm_true_welfare"] - clock_row["clock_llm_revenue"],
                        **_collect_arm_stats(logger.stats_since_mark()),
                    })

                if args.elicited_clock:
                    mechanism = f"proxy clock top_k={top_k}"
                    _section(f"proxy clock  top_k={top_k}")
                    logger.mark()
                    _elicited = make_elicited_proxies()
                    _resolved_clock_policy = args.resolved_event_policy[
                        "clock"
                    ]
                    _clock_proxy_config = ProxyClockConfig(
                        top_k=top_k,
                        elicited=True,
                        margin_threshold=args.clock_margin_threshold,
                        tie_threshold=args.clock_tie_threshold,
                        refine_top_k_frontier=(
                            args.clock_refine_top_k_frontier
                        ),
                        top_k_frontier_policy=(
                            args.clock_top_k_frontier_policy
                        ),
                        allocation_counterfactual_frontier=(
                            args.clock_allocation_counterfactual_frontier
                        ),
                        max_refinements_per_bidder=(
                            args.max_refinement_queries_per_bidder
                        ),
                        max_total_refinements=args.max_total_refinement_queries,
                        incumbent_verification=(
                            _resolved_clock_policy["incumbent_verification"]
                        ),
                        pivotal_challengers=(
                            _resolved_clock_policy[
                                "additional_pivotal_challengers"
                            ]
                        ),
                        scarcity_fallbacks=_resolved_clock_policy[
                            "scarcity_fallbacks"
                        ],
                        large_correction_followup=(
                            args.clock_event_large_correction_followup
                        ),
                        correction_followup_threshold=(
                            args.event_correction_threshold
                        ),
                        gate_near_zero_surplus=(
                            args.event_gate_near_zero_surplus
                        ),
                        terminal_regret_audit=(
                            args.event_terminal_regret_audit
                        ),
                        event_framework=args.clock_event_framework,
                        supplementary_support_policy=(
                            args.clock_supplementary_support_policy
                        ),
                        native_near_zero_surplus=(
                            args.clock_native_near_zero_surplus
                        ),
                        native_demand_changed=(
                            args.clock_native_demand_changed
                        ),
                        native_near_tie=args.clock_native_near_tie,
                        frontier_winner_verification=(
                            _resolved_clock_policy[
                                "frontier_winner_verification"
                            ]
                        ),
                        frontier_pivotal_challengers=(
                            _resolved_clock_policy[
                                "frontier_pivotal_challengers"
                            ]
                        ),
                        frontier_winner_closure=(
                            _resolved_clock_policy[
                                "frontier_winner_closure"
                            ]
                        ),
                        frontier_vcg_witness_verification=(
                            _resolved_clock_policy[
                                "frontier_vcg_witness_verification"
                            ]
                        ),
                        frontier_vcg_single_pass=(
                            _resolved_clock_policy[
                                "frontier_vcg_single_pass"
                            ]
                        ),
                        frontier_vcg_revealed_only=(
                            _resolved_clock_policy[
                                "frontier_vcg_revealed_only"
                            ]
                        ),
                        frontier_staged_revealed_vcg_closure=(
                            _resolved_clock_policy[
                                "frontier_staged_revealed_vcg_closure"
                            ]
                        ),
                        demand_switch_verification=(
                            args.clock_event_demand_switch_verification
                        ),
                        contested_bundle_refinement=(
                            args.clock_event_contested_bundle_refinement
                        ),
                        terminal_winner_verification=(
                            args.clock_event_terminal_winner_verification
                        ),
                        terminal_vcg_witness_verification=(
                            args.clock_event_terminal_vcg_witness_verification
                        ),
                        terminal_best_losing_challenger=(
                            args.clock_event_terminal_best_losing_challenger
                        ),
                        allocation_change_audit=(
                            _resolved_clock_policy[
                                "allocation_change_audit"
                            ]
                        ),
                        terminal_stability_audit=(
                            args.clock_event_terminal_stability_audit
                            if args.clock_event_terminal_stability_audit
                            is not None
                            else args.event_incumbent_verification
                        ),
                    )
                    if args.clock_trajectory:
                        _clock_trajectory = run_proxy_clock_trajectory(
                            instance=scenario.instance,
                            proxies=list(_elicited.values()),
                            clock_config=cfg,
                            proxy_config=_clock_proxy_config,
                            scenario_name=scenario.name,
                            scenario_seed=_scenario_seed,
                            num_goods=_scenario_num_goods,
                            num_bidders=_scenario_num_bidders,
                            logger=logger,
                        )
                        proxy_clock_result = _clock_trajectory.final_result
                        clock_round_rows_by_top_k[top_k].extend(
                            _clock_trajectory.round_rows
                        )
                        clock_bidder_round_rows_by_top_k[top_k].extend(
                            _clock_trajectory.bidder_round_rows
                        )
                        clock_coverage_rows_by_top_k[top_k].extend(
                            _clock_trajectory.coverage_rows
                        )
                        clock_event_rows_by_top_k[top_k].extend(
                            _clock_trajectory.event_rows
                        )
                        print_proxy_clock_trajectory(
                            top_k, _clock_trajectory.round_rows
                        )
                        print_event_usefulness_summary(
                            top_k, _clock_trajectory.event_rows
                        )
                        print(
                            f"  failure_classification: "
                            f"{_clock_trajectory.failure_classification}",
                            flush=True,
                        )
                        _fi_nonempty = [
                            r for r in _clock_trajectory.coverage_rows
                            if r["full_info_winning_bundle_nonempty"]
                        ]
                        _fi_hits = sum(
                            1 for r in _fi_nonempty if r["seen_in_top_k_by_final"]
                        )
                        _last_round = (
                            _clock_trajectory.round_rows[-1]
                            if _clock_trajectory.round_rows
                            else {}
                        )
                        _best_final = _clock_trajectory.best_final
                        _topk_comparison.append({
                            "top_k": top_k,
                            # true_welfare (ground-truth) is the correct
                            # numerator for efficiency -- proxy_clock_result
                            # .welfare is the proxy's *reported* welfare,
                            # which can exceed full_info_welfare and must
                            # never be used as the efficiency numerator.
                            "true_welfare": _last_round.get(
                                "finalise_true_welfare", float("nan")
                            ),
                            "reported_welfare": proxy_clock_result.welfare,
                            "full_info_welfare": _last_round.get(
                                "full_info_welfare", float("nan")
                            ),
                            "efficiency": _last_round.get(
                                "finalise_global_efficiency", float("nan")
                            ),
                            "cumulative_value_queries": _last_round.get(
                                "cumulative_value_queries", 0
                            ),
                            "cumulative_tokens_in": _last_round.get(
                                "cumulative_tokens_in", 0
                            ),
                            "cumulative_tokens_out": _last_round.get(
                                "cumulative_tokens_out", 0
                            ),
                            "supplementary_atoms_total": _last_round.get(
                                "supplementary_atoms_total", 0
                            ),
                            "fi_winner_coverage_hits": _fi_hits,
                            "fi_winner_coverage_total": len(_fi_nonempty),
                            "failure_classification": (
                                _clock_trajectory.failure_classification
                            ),
                            "best_true_welfare": _best_final.get("best_true_welfare"),
                            "best_true_efficiency": _best_final.get(
                                "best_true_efficiency"
                            ),
                            "best_round": _best_final.get("best_round"),
                            "termination_reason": _best_final.get(
                                "termination_reason"
                            ),
                        })
                        _final_eff = _best_final.get("final_true_efficiency")
                        _best_eff = _best_final.get("best_true_efficiency")
                        if (
                            _final_eff is not None
                            and _best_eff is not None
                            and _best_eff - _final_eff > 0.05
                        ):
                            print(
                                f"  WARNING: top_k={top_k}  final clock allocation "
                                f"is worse than best observed allocation "
                                f"(best round {_best_final.get('best_round')} "
                                f"eff={_pct(_best_eff)}, final round "
                                f"{_best_final.get('final_round')} "
                                f"eff={_pct(_final_eff)}); late reported-value "
                                f"corrections may have changed the WDP.",
                                flush=True,
                            )
                    else:
                        proxy_clock_result = run_proxy_clock_experiment(
                            instance=scenario.instance,
                            proxies=list(_elicited.values()),
                            clock_config=cfg,
                            proxy_config=_clock_proxy_config,
                            scenario_name=scenario.name,
                        )
                    proxy_clock_row = proxy_clock_result_to_row(
                        instance_name=scenario.name,
                        instance=scenario.instance,
                        result=proxy_clock_result,
                    )
                    clock_proxy_rows_by_top_k[top_k].append(proxy_clock_row)
                    _arm_result(
                        f"proxy clock k={top_k}",
                        proxy_clock_row["efficiency"],
                        proxy_clock_row["proxy_true_welfare"],
                        proxy_clock_row["full_info_welfare"],
                        extra=(
                            f"rounds {proxy_clock_result.rounds}  "
                            f"queries {proxy_clock_row['refinement_query_count_by_bidder']}"
                        ),
                        reported_welfare=proxy_clock_row["proxy_reported_welfare"],
                    )
                    _proxy_clock_stats = logger.stats_since_mark()
                    _pc_arm = _collect_arm_stats(_proxy_clock_stats)
                    _summary.append({
                        "scenario": scenario.name,
                        "arm": f"proxy clock k={top_k}",
                        "efficiency": proxy_clock_row["efficiency"],
                        "true_welfare": proxy_clock_row["proxy_true_welfare"],
                        "full_info_welfare": proxy_clock_row["full_info_welfare"],
                        "revenue": proxy_clock_row["proxy_revenue"],
                        "surplus": proxy_clock_row["proxy_true_welfare"] - proxy_clock_row["proxy_revenue"],
                        **_pc_arm,
                        **_amortized_shared(),
                        **_est_gt_tok(_pc_arm),
                    })
                    print_refinement_records(
                        proxy_clock_result.metadata["refinement_records_by_bidder"],
                    )
                    all_refinement_rows.extend(
                        refinement_records_to_rows(
                            scenario.name,
                            f"proxy_clock_k{top_k}",
                            proxy_clock_result.metadata["refinement_records_by_bidder"],
                            final_allocation=proxy_clock_result.allocation,
                            reported_vcg_counterfactuals=proxy_clock_result.metadata[
                                "vcg_counterfactuals"
                            ],
                            full_info_allocation=(
                                _full_info_witness_result.allocation
                            ),
                            full_info_vcg_counterfactuals=(
                                _full_info_witness_result.metadata[
                                    "vcg_counterfactuals"
                                ]
                            ),
                        )
                    )
                    if args.verbose:
                        print_arm_summary(
                            f"proxy clock k={top_k}",
                            _elicited,
                            _proxy_clock_stats,
                            _n_bidders,
                        )

            if args.elicited_clock and args.clock_trajectory:
                print_topk_comparison(_topk_comparison)

        except Exception as exc:
            print(
                f"Scenario {scenario.name}, mechanism {mechanism} failed: "
                f"{exc}",
                file=sys.stderr,
            )
            ollama_models = {
                role_model
                for role_provider, role_model in (
                    (args.person_provider, args.person_model),
                    (args.proxy_provider, args.proxy_model),
                )
                if role_provider == "ollama"
            }
            for ollama_model in sorted(ollama_models):
                print_ollama_help(ollama_model)
            print(f"Logs were written to: {log_path}", file=sys.stderr)
            raise SystemExit(1) from exc

    # Summary table
    if _summary:
        _cache_summary_fields = {
            "person_disclosure_style": "brief_qualitative",
            "person_query_mode": args.person_query_mode,
            "person_provider": args.person_provider,
            "person_model": args.person_model,
            "person_temperature": args.person_temperature,
            "proxy_provider": args.proxy_provider,
            "proxy_model": args.proxy_model,
            "proxy_temperature": args.proxy_temperature,
            "llm_cache_mode": args.llm_cache_mode,
            "llm_cache_path": args.llm_cache_path if llm_cache is not None else "",
            **_calibration_fields,
            **event_policy_summary_fields(args),
            **llm_cache_stats.as_dict(),
            **llm_role_summary_fields(logger),
        }
        if scenarios:
            _environment_md = getattr(scenarios[0], "metadata", {}) or {}
            _cache_summary_fields.update({
                "environment_generation_provider": _environment_md.get(
                    "environment_generation_provider"
                ),
                "environment_generation_model": _environment_md.get(
                    "environment_generation_model"
                ),
            })
        for _row in _summary:
            _row.update(_cache_summary_fields)
        _print_results_table(_summary)
        write_csv_variable_rows(_summary, run_summary_path)
        print(f"  run summary CSV         →  {run_summary_path}")
        if llm_cache is not None:
            print(
                f"  llm cache               →  hits={llm_cache_stats.hits}  "
                f"misses={llm_cache_stats.misses}  writes={llm_cache_stats.writes}  "
                f"read_only_misses={llm_cache_stats.read_only_misses}"
            )

    write_csv_variable_rows(
        all_person_disclosure_rows,
        person_disclosures_path,
    )
    print(f"  person disclosures CSV  →  {person_disclosures_path}")

    if llm_cache is not None:
        llm_cache.close()

    # Every outcome CSV carries the effective calibration, so a detailed
    # sealed/clock file read on its own still says how its values were
    # produced.
    for _rows in (
        sealed_rows,
        sealed_proxy_rows,
        all_trajectory_rows,
        *clock_rows_by_top_k.values(),
        *clock_proxy_rows_by_top_k.values(),
    ):
        add_calibration_fields(_rows, calibration)

    # Write CSVs
    if sealed_rows:
        write_csv(sealed_rows, sealed_path)
        print(f"  sealed CSV              →  {sealed_path}")
    for top_k, rows in clock_rows_by_top_k.items():
        if rows:
            write_csv(rows, clock_paths[top_k])
            print(f"  clock k={top_k} CSV          →  {clock_paths[top_k]}")
    if args.sealed_elicitation_rounds > 0:
        write_csv(sealed_proxy_rows, sealed_proxy_path)
        print(f"  proxy sealed CSV        →  {sealed_proxy_path}")
    if all_trajectory_rows:
        write_csv(all_trajectory_rows, sealed_proxy_trajectory_path)
        print(f"  proxy sealed trajectory →  {sealed_proxy_trajectory_path}")
    if all_pv_candidate_bundle_stats_rows:
        write_csv(all_pv_candidate_bundle_stats_rows, pv_candidate_bundle_stats_path)
        print(f"  pv candidate bundle stats →  {pv_candidate_bundle_stats_path}")
    if args.elicited_clock:
        for top_k, rows in clock_proxy_rows_by_top_k.items():
            write_csv(rows, clock_proxy_paths[top_k])
            print(f"  proxy clock k={top_k} CSV    →  {clock_proxy_paths[top_k]}")
    if args.elicited_clock and args.clock_trajectory:
        for top_k, rows in clock_round_rows_by_top_k.items():
            if rows:
                write_csv(rows, clock_round_paths[top_k])
                print(f"  clock rounds k={top_k} CSV   →  {clock_round_paths[top_k]}")
        for top_k, rows in clock_bidder_round_rows_by_top_k.items():
            if rows:
                write_csv(rows, clock_bidder_round_paths[top_k])
                print(
                    f"  clock bidder-rounds k={top_k} CSV →  "
                    f"{clock_bidder_round_paths[top_k]}"
                )
        for top_k, rows in clock_coverage_rows_by_top_k.items():
            if rows:
                write_csv(rows, clock_coverage_paths[top_k])
                print(f"  clock coverage k={top_k} CSV →  {clock_coverage_paths[top_k]}")
        for top_k, rows in clock_event_rows_by_top_k.items():
            if rows:
                write_csv(rows, clock_event_paths[top_k])
                print(f"  clock events k={top_k} CSV   →  {clock_event_paths[top_k]}")
    if all_refinement_rows:
        write_csv(all_refinement_rows, refinement_path)
        print(f"  refinement records CSV  →  {refinement_path}")
    print(f"  run config JSON         →  {run_config_path}")
    print(f"  logs                    →  {log_path}")


if __name__ == "__main__":
    main()
