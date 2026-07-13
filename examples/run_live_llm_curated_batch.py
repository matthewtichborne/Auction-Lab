"""Run the canonical live LLM experiment over curated auction scenarios."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import sys

from auctionlab.auctions.ceca import CecaConfig
from auctionlab.auctions.clock import ClockConfig
from auctionlab.experiments.export import write_csv
from auctionlab.experiments.llm_comparison import (
    ceca_result_to_row,
    ceca_winner_diagnostics_rows,
    clock_llm_comparison_to_row,
    proxy_clock_result_to_row,
    proxy_sealed_result_to_row,
    reported_bids_to_str,
    run_clock_llm_comparison,
    run_sealed_llm_comparison,
    sealed_llm_comparison_to_row,
)
from auctionlab.experiments.proxy_ceca_runner import (
    ProxyCecaConfig,
    ProxyCecaSharedResult,
    ceca_satisfaction_diagnostic_rows,
    finalize_proxy_ceca_result,
    run_proxy_ceca_elicitation,
    run_proxy_ceca_experiment,
)
from auctionlab.experiments.proxy_clock_runner import (
    ProxyClockConfig,
    run_proxy_clock_experiment,
)
from auctionlab.experiments.proxy_sealed_runner import (
    ProxySealedConfig,
    run_proxy_sealed_vcg_experiment,
)
from auctionlab.experiments.run_config import (
    PRESETS,
    apply_preset,
    collect_arm_stats,
    collect_initial_stats,
    config_warnings,
    explicitly_set_args,
    format_run_config,
    refinement_records_to_rows,
)
from auctionlab.instances.nl_scenarios import (
    NaturalLanguageAuctionScenario,
    curated_natural_language_scenarios,
)
from auctionlab.llm.bundles import generate_candidate_bundles
from auctionlab.llm.clients import OpenAICompatibleLlmClient
from auctionlab.llm.logging import CallTypeStats, LlmCallLogger  # CallTypeStats used in print_arm_summary
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmAuctionProxyAdapter, LlmInferredXorProxy
from auctionlab.llm.schemas import LlmInterestMap
from auctionlab.proxies.base import RefinementRecord
from auctionlab.proxies.baselines.dnf_learning import DnfLearningProxy
from auctionlab.proxies.events import (
    GENERATE_CANDIDATE_BUNDLES,
    INFER_INTEREST_MAP,
    INFER_PROVISIONAL_VALUES,
    INITIAL_PREFERENCE_QUESTION,
    ProxyElicitationEvent,
)
from auctionlab.proxies.baselines.hybrid import HybridProxy
from auctionlab.proxies.baselines.llm_ceca import (
    NvdCecaProxy,
    SizeLimitedScope,
    Vd1CecaProxy,
    Vd2CecaProxy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live LLM proxy mechanisms on curated scenarios."
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        default=None,
        help=(
            "Apply a recommended default configuration. Individual flags "
            "that appear on the command line override preset values. "
            "Available: " + ", ".join(PRESETS)
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "groq", "gemini", "openai-compatible"],
        default="ollama",
        help=(
            "'groq' uses the Groq API (set GROQ_API_KEY, e.g. --model "
            "llama-3.3-70b-versatile); 'gemini' uses the Gemini API "
            "(set GEMINI_API_KEY, e.g. --model gemini-2.0-flash)."
        ),
    )
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for --provider groq/openai-compatible. For groq, "
        "falls back to the GROQ_API_KEY env var.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument(
        "--ground-truth-queries",
        action="store_true",
        help=(
            "Replace all LLM value/demand queries with ground-truth lookups. "
            "Query counts still accumulate normally. Useful for verifying "
            "mechanism logic without spending tokens."
        ),
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-parse-retries", type=int, default=1)
    parser.add_argument(
        "--log-dir",
        default="outputs/llm_runs/curated_batch",
    )
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--discount-inferred", action="store_true")
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
        "--max-refinement-queries-per-bidder",
        type=int,
        default=0,
        help="Cap on refinement queries per bidder. 0 means unlimited.",
    )
    parser.add_argument(
        "--elicited-ceca",
        action="store_true",
        help=(
            "Run the CECA (Competitive Equilibrium Combinatorial Auction) "
            "proxy-mediated arm: iterative Lindahl-style personalized "
            "bundle pricing until every bidder is simultaneously satisfied."
        ),
    )
    parser.add_argument(
        "--ceca-max-rounds",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--ceca-proxy-type",
        choices=["llm", "dnf", "vd1", "vd2", "nvd"],
        default="llm",
        help=(
            "Proxy implementation for the --elicited-ceca arm: "
            "'llm' (default) uses the modular LLM proxy (LlmAuctionProxyAdapter / "
            "LlmInferredXorProxy), the primary cross-mechanism architecture -- "
            "controlled by --proxy-type, --use-interest-map, and "
            "--use-provisional-valuations; "
            "'dnf' uses the non-LLM proper-learning baseline (ωxor); "
            "'vd1', 'vd2', 'nvd' are legacy CECA-specific literature baselines "
            "(ωvd1/ωvd2/ωnvd) kept for comparison -- they require --provider and "
            "--model for a separate proxy LLM client."
        ),
    )
    parser.add_argument(
        "--gamma-refresh-every",
        type=int,
        default=1,
        help=(
            "ωvd2/ωnvd: refresh γ estimates every N CECA rounds (1 = every round, "
            "0 = disabled after the initial seed). Ignored for vd1/dnf."
        ),
    )
    parser.add_argument(
        "--nvd-num-questions",
        type=int,
        default=1,
        help="ωnvd: number of NL preference questions to ask before the CECA loop.",
    )
    parser.add_argument(
        "--ceca-payment-rule",
        choices=["pay_as_bid", "vcg", "both"],
        default="pay_as_bid",
        help=(
            "'pay_as_bid' is the faithful CECA rule (pay your own reported "
            "value for what you win); 'vcg' computes Clarke-pivot payments "
            "over the same final allocation for comparison; 'both' runs the "
            "round loop once and finalizes both ways. CECA internal "
            "elicitation always uses Lindahl-style prices regardless of "
            "this setting."
        ),
    )
    parser.add_argument(
        "--sealed-elicitation-rounds",
        type=int,
        default=0,
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
            "Number of PC goods for --scenario pc_build (4–10). "
            "Ignored when --scenario names a specific scenario."
        ),
    )
    parser.add_argument(
        "--num-bidders",
        type=int,
        default=8,
        help=(
            "Number of bidder archetypes for --scenario pc_build (3–10). "
            "Ignored when --scenario names a specific scenario."
        ),
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=0,
        help="Random seed for structured scenario jitter (--scenario pc_build).",
    )
    parser.add_argument(
        "--ask-initial-question",
        action="store_true",
        help=(
            "ωnvd: ask each bidder one open-ended NL question up front and "
            "fold the answer into all later value inference."
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
        "--proxy-type",
        choices=["llm", "dnf", "hybrid"],
        default="llm",
        help=(
            "Proxy implementation used for elicited proxy-mediated runs "
            "(--elicited-clock / --sealed-elicitation-rounds): 'llm' is the "
            "NL-and-inference proxy (ωnvd/ωvd), 'dnf' is the non-LLM "
            "proper-learning baseline (ωxor), 'hybrid' is ωh (ωnvd/ωvd for "
            "the first --hybrid-alpha refinements, then ωxor). --elicited-ceca "
            "requires 'llm' -- 'dnf'/'hybrid' proxies don't implement "
            "ceca_step."
        ),
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=int,
        default=10,
        help="ωh: number of early refinements handled by the LLM proxy.",
    )
    parser.add_argument(
        "--hybrid-delta",
        type=float,
        default=0.95,
        help=(
            "ωh: per-refinement decay factor applied to the LLM proxy's "
            "remaining inferred values after the switch to ωxor."
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
            "Cap the number of candidate bundles per bidder after interest-map "
            "filtering (priority order: complementary groups first, then "
            "singletons, then remaining bundles by ascending size). Pass None "
            "for a full uncapped run. Ignored unless --use-interest-map is set."
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
        "--ceca-no-pv",
        action="store_true",
        help=(
            "Build the CECA proxy arm from the shared NL+interest-map "
            "elicitation but WITHOUT replaying provisional valuations. This "
            "prevents CECA from converging in round 1 due to PV-initialized "
            "bids that already form a competitive equilibrium."
        ),
    )
    parser.add_argument(
        "--ceca-initial-bid-mode",
        nargs="+",
        choices=["full_proxy", "singletons", "empty"],
        default=["full_proxy"],
        help=(
            "How to seed the CECA manifest from the proxy's current XOR bid. "
            "Accepts one or more values to run multiple CECA arms in one pass. "
            "'full_proxy' (labelled 'prior'): seed from the full bid — tests the "
            "proxy prior quality; tends to converge in one round. "
            "'singletons': seed only singleton atoms; complement bundles must be "
            "discovered through CECA's demand/satisfaction loop — isolates CECA's "
            "iterative value from the prior. "
            "'empty': seed with no atoms — stress-test mode. "
            "Example: --ceca-initial-bid-mode full_proxy singletons"
        ),
    )
    parser.add_argument(
        "--ceca-atomic-trimming",
        action="store_true",
        default=True,
        dest="ceca_atomic_trimming",
        help="Enable CECA atomic trimming (default: on).",
    )
    parser.add_argument(
        "--no-ceca-atomic-trimming",
        action="store_false",
        dest="ceca_atomic_trimming",
        help="Disable CECA atomic trimming.",
    )
    parser.add_argument(
        "--ceca-trim-value-tolerance",
        type=float,
        default=0.0,
        help=(
            "Tolerance for atomic trimming: remove item if "
            "abs(reduced_value - demanded_value) <= tol. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--ceca-stop-on-no-new-information",
        action="store_true",
        default=False,
        dest="ceca_stop_on_no_new_information",
        help=(
            "Stop CECA early if K consecutive rounds produce no new manifest atoms."
        ),
    )
    parser.add_argument(
        "--ceca-stall-patience",
        type=int,
        default=1,
        dest="ceca_stall_patience",
        help=(
            "Number of consecutive stall rounds before stopping "
            "(requires --ceca-stop-on-no-new-information). Default: 1."
        ),
    )
    parser.add_argument(
        "--ceca-stop-on-round-with-no-useful-counterexamples",
        action="store_true",
        default=False,
        dest="ceca_stop_on_round_no_useful_counterexamples",
        help=(
            "Stop CECA immediately after the first round in which every unsatisfied "
            "bidder's demand trims to an atom already in the manifest (no new info). "
            "Stronger than --ceca-stop-on-no-new-information with patience=1."
        ),
    )
    parser.add_argument(
        "--ceca-exhaust-repeated-bidders",
        action="store_true",
        default=False,
        dest="ceca_exhaust_repeated_bidders",
        help=(
            "Skip demand queries for bidders that return the same trimmed atom "
            "--ceca-bidder-stall-patience consecutive times while their allocation is unchanged."
        ),
    )
    parser.add_argument(
        "--ceca-bidder-stall-patience",
        type=int,
        default=3,
        dest="ceca_bidder_stall_patience",
        help="Consecutive same-trimmed-atom count before skipping a bidder (default: 3).",
    )
    parser.add_argument(
        "--ceca-demand-universe",
        type=str,
        default="all_items",
        dest="ceca_demand_universe",
        choices=["all_items", "interested_items", "candidate_bundles", "manifest_plus_candidates"],
        help=(
            "Constrain CECA demand queries to this bundle universe. "
            "'all_items': current behaviour (no constraint). "
            "'interested_items': subsets of each bidder's interest-map items. "
            "'candidate_bundles': bidder's pre-CECA candidate atoms only. "
            "'manifest_plus_candidates': union of candidates and current manifest atoms. "
            "(default: all_items)"
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


def make_live_client(
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

    return OpenAICompatibleLlmClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def make_live_persons_for_scenario(
    scenario: NaturalLanguageAuctionScenario,
    *,
    model: str,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    timeout: float,
    logger: LlmCallLogger,
    max_parse_retries: int,
    use_ground_truth: bool = False,
    verbose: bool = False,
) -> dict[str, LlmPersonSimulator]:
    persons: dict[str, LlmPersonSimulator] = {}

    for bidder_id in scenario.instance.bidder_ids:
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
            ),
            logger=logger,
            model_name=model,
            max_parse_retries=max_parse_retries,
            ground_truth_valuations=(
                scenario.instance.valuations[bidder_id] if use_ground_truth else None
            ),
            verbose=verbose,
        )

    return persons


@dataclass
class _BidderElicitationCache:
    """One bidder's NL question/answer, interest map, and PV results.

    Computed once per scenario and replayed (no LLM calls) into a fresh
    :class:`LlmInferredXorProxy` for each mechanism arm, so the sealed and
    clock arms start from an identical informational baseline instead of
    each re-asking the same NL question and re-running the same bulk PV
    call independently.
    """

    nl_question: str
    nl_answer: str
    interest_map: LlmInterestMap | None
    candidate_bundles: list
    raw_pv_values: dict | None


def compute_elicitation_cache(
    *,
    scenario: NaturalLanguageAuctionScenario,
    persons: dict[str, LlmPersonSimulator],
    use_provisional_valuations: bool,
    max_candidate_bundles: int | None,
    pv_client,
) -> dict[str, _BidderElicitationCache]:
    """Run the NL-question + interest-map (+ optional PV) phase once.

    This is the expensive, arm-independent part of elicited proxy
    construction. Callers replay the result into one fresh proxy per arm via
    :meth:`LlmInferredXorProxy.replay_elicitation`.
    """
    print("  NL elicitation  [event-driven]", flush=True)
    cache: dict[str, _BidderElicitationCache] = {}

    for bidder_id, person in persons.items():
        proxy = LlmInferredXorProxy(bidder_id=bidder_id, person=person)

        # Emit initialisation events through the proxy event API.
        proxy.handle_event(ProxyElicitationEvent(
            event_type=INITIAL_PREFERENCE_QUESTION,
            bidder_id=bidder_id,
            mechanism="init",
        ))

        im_response = proxy.handle_event(ProxyElicitationEvent(
            event_type=INFER_INTEREST_MAP,
            bidder_id=bidder_id,
            mechanism="init",
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

        if not cb:
            print(
                f"  {bidder_id:<12}  WARNING: interest map produced no bundles; "
                "falling back to all items",
                flush=True,
            )
            cb = generate_candidate_bundles(
                scenario.instance.items,
                max_bundle_size=len(scenario.instance.items),
            )
        im_detail = f"interested={sorted(im.interested_items)}"
        if im.complementary_groups:
            im_detail += f"  compl={[sorted(g) for g in im.complementary_groups]}"
        if im.substitute_groups:
            im_detail += f"  subst={[sorted(g) for g in im.substitute_groups]}"
        print(
            f"  {bidder_id:<12}  im  {im_detail}  →  {len(cb)} bundles",
            flush=True,
        )

        raw_pv_values = None
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
                    },
                ))
                raw_pv_values = pv_response.payload["raw_values"]
            except Exception as exc:
                print(
                    f"  {bidder_id:<12}  WARNING: PV failed "
                    f"({type(exc).__name__}: {exc}); "
                    "using zero-bid initialisation for this bidder.",
                    flush=True,
                )

        nl_question, nl_answer = proxy.nl_transcript[-1]
        cache[bidder_id] = _BidderElicitationCache(
            nl_question=nl_question,
            nl_answer=nl_answer,
            interest_map=im,
            candidate_bundles=cb,
            raw_pv_values=raw_pv_values,
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
    timeout: float,
    logger: LlmCallLogger,
    max_parse_retries: int,
    epsilon: float,
    ask_initial_question: bool = False,
    use_ground_truth: bool = False,
    verbose: bool = False,
) -> dict[str, LlmInferredXorProxy]:
    persons = make_live_persons_for_scenario(
        scenario,
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        logger=logger,
        max_parse_retries=max_parse_retries,
        use_ground_truth=use_ground_truth,
        verbose=verbose,
    )

    proxies: dict[str, LlmInferredXorProxy] = {}
    for bidder_id, person in persons.items():
        proxy = LlmInferredXorProxy(
            bidder_id=bidder_id,
            person=person,
            epsilon=epsilon,
        )
        if ask_initial_question:
            proxy.ask_initial_question()
        proxies[bidder_id] = proxy

    return proxies


def make_ceca_llm_proxies(
    scenario: NaturalLanguageAuctionScenario,
    *,
    persons: dict[str, LlmPersonSimulator],
    proxy_client,
    ceca_proxy_type: str,
    max_bundle_size: int,
    gamma_refresh_every: int,
    nvd_num_questions: int,
    max_parse_retries: int,
    logger,
) -> dict:
    """Build ωvd1/ωvd2/ωnvd CECA proxies for one scenario."""
    scope = SizeLimitedScope(max_size=max_bundle_size)
    result = {}
    for bidder_id, person in persons.items():
        common = dict(
            bidder_id=bidder_id,
            person=person,
            items=list(scenario.instance.items),
            proxy_client=proxy_client,
            scenario_description=scenario.scenario_description,
            item_descriptions=scenario.item_descriptions,
            max_parse_retries=max_parse_retries,
            logger=logger,
        )
        if ceca_proxy_type == "vd1":
            result[bidder_id] = Vd1CecaProxy(**common)
        elif ceca_proxy_type == "vd2":
            result[bidder_id] = Vd2CecaProxy(
                **common,
                bundle_scope=scope,
                gamma_refresh_every=gamma_refresh_every,
            )
        else:  # nvd
            result[bidder_id] = NvdCecaProxy(
                **common,
                bundle_scope=scope,
                gamma_refresh_every=gamma_refresh_every,
                num_nl_questions=nvd_num_questions,
            )
    return result


def select_scenarios(
    names: list[str] | None,
    seed_type: str = "all",
    *,
    num_goods: int = 8,
    num_bidders: int = 8,
    scenario_seed: int = 0,
) -> list[NaturalLanguageAuctionScenario]:
    """Select scenarios by name/seed-type, or generate a single pc_build scenario."""
    if names and len(names) == 1 and names[0] == "pc_build":
        from auctionlab.instances.structured import make_pc_build_scenario
        scenarios = [make_pc_build_scenario(num_goods, num_bidders, scenario_seed)]
    else:
        scenarios = curated_natural_language_scenarios()
        if names:
            by_name = {scenario.name: scenario for scenario in scenarios}
            unknown = sorted(set(names) - set(by_name))
            if unknown:
                raise ValueError(
                    f"Unknown scenario names: {unknown}. "
                    f"Available: {sorted(by_name)}"
                )
            scenarios = [by_name[name] for name in names]

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

    fill = max(0, 38)
    print(f"\n  ── proxy sealed post-run summary {'─' * fill}", flush=True)
    print(f"    elicitation_rounds:    {elicitation_rounds}", flush=True)
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


_PERSON_PROMPT_TYPES = {"value_query", "demand_query", "nl_question"}
_PROXY_PROMPT_TYPES = {"proxy_nl_gen", "proxy_interest_map", "proxy_provisional_valuations"}


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
                f"      substitutes: {[sorted(g) for g in im.substitute_groups]}",
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

    have_tokens = total_in or total_out
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
        f"  {'eff':>7}  {'welfare':>13}"
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
    # Token accounting legend
    has_shared = any(r.get("arm", "") == "shared initial (nl+im+pv)" for r in summary)
    if has_shared:
        n_amort = sum(1 for r in summary if r.get("shared_tok_in_amort", 0) > 0)
        n_elicited = n_amort or "?"
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


def _print_ceca_payment_table(
    instance,
    shared: "ProxyCecaSharedResult",
    results: dict,
) -> None:
    """Print combined allocation + payment table for one CECA mode.

    ``results`` is keyed by payment_rule; all entries share the same
    allocation. When only one payment rule is present the table has a
    single payment/surplus column pair; when two are present both appear
    side-by-side for direct comparison.
    """
    state = shared.ceca_state
    rounds = len(state.history)
    stopped_reason = getattr(shared, 'stopped_reason', None)
    if stopped_reason == "converged":
        conv_str = "converged"
    elif stopped_reason == "no_new_information":
        conv_str = "stopped: no new information"
    elif stopped_reason == "max_rounds":
        conv_str = f"NOT converged ({rounds} rounds)"
    else:
        conv_str = "converged" if state.converged else f"NOT converged ({rounds} rounds)"
    mode = shared.ceca_initial_bid_mode
    print(
        f"\n  CECA final allocation  ({mode}  {rounds} rounds  {conv_str})",
        flush=True,
    )

    rules = sorted(results)
    if not rules:
        return

    # Take allocation from the first result (all share the same allocation).
    allocation = next(iter(results.values())).allocation
    manifest_bids = state.manifest_bids

    winners = [
        (bidder_id, bundle)
        for bidder_id, bundle in sorted(allocation.items())
        if bundle
    ]
    if not winners:
        print("    (no winners)", flush=True)
        return

    # Column headers depend on number of payment rules.
    if len(rules) == 1:
        rule = rules[0]
        short = "PAB" if rule == "pay_as_bid" else "VCG"
        print(
            f"    {'bidder':<20}  {'bundle':<24}  {'reported':>9}"
            f"  {'true':>9}  {short+' pay':>9}  {'surplus':>9}",
            flush=True,
        )
        for bidder_id, bundle in winners:
            rep = (
                manifest_bids[bidder_id].value_of(bundle)
                if bidder_id in manifest_bids
                else float("nan")
            )
            true_v = instance.value_of(bidder_id, bundle)
            pay = results[rule].payments.get(bidder_id, 0.0)
            bstr = "{" + ",".join(sorted(bundle)) + "}"
            print(
                f"    {bidder_id:<20}  {bstr:<24}  {rep:>9.0f}"
                f"  {true_v:>9.0f}  {pay:>9.0f}  {true_v - pay:>9.0f}",
                flush=True,
            )
    else:
        # Both payment rules: side-by-side.
        print(
            f"    {'bidder':<20}  {'bundle':<24}  {'reported':>9}  {'true':>9}"
            f"  {'PAB pay':>8}  {'VCG pay':>8}  {'surp-PAB':>9}  {'surp-VCG':>9}",
            flush=True,
        )
        pab_result = results.get("pay_as_bid")
        vcg_result = results.get("vcg")
        for bidder_id, bundle in winners:
            rep = (
                manifest_bids[bidder_id].value_of(bundle)
                if bidder_id in manifest_bids
                else float("nan")
            )
            true_v = instance.value_of(bidder_id, bundle)
            pab_pay = pab_result.payments.get(bidder_id, 0.0) if pab_result else float("nan")
            vcg_pay = vcg_result.payments.get(bidder_id, 0.0) if vcg_result else float("nan")
            bstr = "{" + ",".join(sorted(bundle)) + "}"
            print(
                f"    {bidder_id:<20}  {bstr:<24}  {rep:>9.0f}  {true_v:>9.0f}"
                f"  {pab_pay:>8.0f}  {vcg_pay:>8.0f}"
                f"  {true_v - pab_pay:>9.0f}  {true_v - vcg_pay:>9.0f}",
                flush=True,
            )
    print(flush=True)


def main() -> None:
    # Track which flags were explicitly set before parse_args mutates sys.argv context
    _explicitly_set = explicitly_set_args()

    args = parse_args()

    # Apply preset defaults for any flags not explicitly set on the command line
    _preset_applied = apply_preset(args, _explicitly_set)

    try:
        scenarios = select_scenarios(
            args.scenario,
            args.seed_type,
            num_goods=args.num_goods,
            num_bidders=args.num_bidders,
            scenario_seed=args.scenario_seed,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    # Full config header
    for line in format_run_config(args, scenarios):
        print(line, flush=True)

    if _preset_applied:
        print(
            f"  (preset '{args.preset}' applied defaults for: "
            + ", ".join(_preset_applied) + ")",
            flush=True,
        )

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
    clock_proxy_paths = {
        top_k: log_dir / f"curated_clock_proxy_elicited_top_{top_k}.csv"
        for top_k in args.top_k
    }
    ceca_payment_rules = (
        ["pay_as_bid", "vcg"]
        if args.ceca_payment_rule == "both"
        else [args.ceca_payment_rule]
    )
    # nargs="+" means this is always a list; normalise for safety.
    _raw_modes = getattr(args, "ceca_initial_bid_mode", ["full_proxy"])
    ceca_initial_bid_modes: list[str] = (
        _raw_modes if isinstance(_raw_modes, list) else [_raw_modes]
    )
    # Keyed by (mode, payment_rule) tuple.
    ceca_proxy_paths = {
        (mode, rule): log_dir / f"curated_ceca_proxy_elicited_{mode}_{rule}.csv"
        for mode in ceca_initial_bid_modes
        for rule in ceca_payment_rules
    }
    ceca_diagnostics_paths = {
        mode: log_dir / f"curated_ceca_value_payment_diagnostics_{mode}.csv"
        for mode in ceca_initial_bid_modes
    }
    refinement_path = log_dir / "curated_refinement_records.csv"

    logger = LlmCallLogger(log_path)
    cfg = ClockConfig(
        max_rounds=args.max_rounds,
        price_step=args.price_step,
        reserve=args.reserve,
    )
    ceca_cfg = CecaConfig(max_rounds=args.ceca_max_rounds)

    sealed_rows = []
    clock_rows_by_top_k = {top_k: [] for top_k in args.top_k}
    sealed_proxy_rows = []
    clock_proxy_rows_by_top_k = {top_k: [] for top_k in args.top_k}
    # Keyed by (mode, payment_rule) tuple, parallel to ceca_proxy_paths.
    ceca_proxy_rows: dict[tuple[str, str], list] = {
        (mode, rule): []
        for mode in ceca_initial_bid_modes
        for rule in ceca_payment_rules
    }
    ceca_winner_diagnostics: list[dict] = []
    ceca_satisfaction_diag_rows: list[dict] = []
    _summary: list[dict] = []
    all_refinement_rows: list[dict] = []

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

        bundles_by_bidder = {
            bidder_id: candidate_bundles
            for bidder_id in scenario.instance.bidder_ids
        }

        _nl_sample_shown = [False]
        _n_bidders = len(scenario.instance.bidder_ids)

        _persons_cache: list[dict[str, LlmPersonSimulator] | None] = [None]
        _elicitation_cache: list[dict[str, _BidderElicitationCache] | None] = [None]

        def _get_persons() -> dict[str, LlmPersonSimulator]:
            if _persons_cache[0] is None:
                _persons_cache[0] = make_live_persons_for_scenario(
                    scenario,
                    model=args.model,
                    provider=args.provider,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    logger=logger,
                    max_parse_retries=args.max_parse_retries,
                    use_ground_truth=args.ground_truth_queries,
                    verbose=args.verbose,
                )
            return _persons_cache[0]

        def _get_elicitation_cache() -> dict[str, _BidderElicitationCache]:
            if _elicitation_cache[0] is None:
                pv_client = None
                if args.use_provisional_valuations:
                    pv_client = make_live_client(
                        model=args.model,
                        provider=args.provider,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        temperature=args.temperature,
                        max_tokens=args.pv_max_tokens,
                        timeout=args.timeout,
                    )
                _elicitation_cache[0] = compute_elicitation_cache(
                    scenario=scenario,
                    persons=_get_persons(),
                    use_provisional_valuations=args.use_provisional_valuations,
                    max_candidate_bundles=args.max_candidate_bundles,
                    pv_client=pv_client,
                )
            return _elicitation_cache[0]

        def make_elicited_proxies(*, use_pv: bool = True):
            persons = _get_persons()

            if args.use_interest_map or args.use_provisional_valuations:
                cache = _get_elicitation_cache()

                def get_llm_adapter(bidder_id: str) -> LlmAuctionProxyAdapter:
                    entry = cache[bidder_id]
                    proxy = LlmInferredXorProxy(
                        bidder_id=bidder_id,
                        person=persons[bidder_id],
                        epsilon=args.epsilon,
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
                        epsilon=args.epsilon,
                    )
                    if args.ask_initial_question:
                        proxy.ask_initial_question()
                    return LlmAuctionProxyAdapter(
                        bidder_id=bidder_id,
                        proxy=proxy,
                        candidate_bundles=bundles_by_bidder[bidder_id],
                        discount_inferred=args.discount_inferred,
                        use_anchor_values=not args.disable_anchor_values,
                        refinement_strategy=args.refinement_strategy,
                    )

            if args.proxy_type == "llm":
                result_proxies = {
                    bidder_id: get_llm_adapter(bidder_id)
                    for bidder_id in persons
                }
            elif args.proxy_type == "dnf":
                result_proxies = {
                    bidder_id: DnfLearningProxy(
                        bidder_id=bidder_id,
                        person=person,
                        items=list(scenario.instance.items),
                    )
                    for bidder_id, person in persons.items()
                }
            else:
                result_proxies = {
                    bidder_id: HybridProxy(
                        bidder_id=bidder_id,
                        llm_proxy=get_llm_adapter(bidder_id),
                        dnf_proxy=DnfLearningProxy(
                            bidder_id=bidder_id,
                            person=person,
                            items=list(scenario.instance.items),
                        ),
                        alpha=args.hybrid_alpha,
                        delta=args.hybrid_delta,
                    )
                    for bidder_id, person in persons.items()
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
            or args.elicited_ceca
        ) and (args.use_interest_map or args.use_provisional_valuations)

        # Number of elicited arms that amortize the shared initial cost.
        _n_elicited_arms: int = (
            (1 if args.sealed_elicitation_rounds > 0 else 0)
            + (len(args.top_k) if args.elicited_clock else 0)
            + (len(ceca_initial_bid_modes) * len(ceca_payment_rules) if args.elicited_ceca else 0)
        )
        _shared_tok_in: int = 0
        _shared_tok_out: int = 0

        if _needs_elicited:
            logger.mark()
            _get_elicitation_cache()
            _shared_initial_stats = logger.stats_since_mark()
            _shared_initial_row = _collect_initial_stats(_shared_initial_stats)
            _shared_tok_in = _shared_initial_row["tok_in"]
            _shared_tok_out = _shared_initial_row["tok_out"]
            _summary.append({
                "scenario": scenario.name,
                "arm": "shared initial (nl+im+pv)",
                "efficiency": float("nan"),
                "true_welfare": float("nan"),
                "full_info_welfare": float("nan"),
                **_shared_initial_row,
            })

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
                        timeout=args.timeout,
                        logger=logger,
                        max_parse_retries=args.max_parse_retries,
                        epsilon=args.epsilon,
                        ask_initial_question=args.ask_initial_question,
                        use_ground_truth=args.ground_truth_queries,
                        verbose=args.verbose,
                    ),
                    candidate_bundles=candidate_bundles,
                    candidate_bundles_by_bidder=candidate_bundles_by_bidder,
                    discount_inferred=args.discount_inferred,
                    use_anchor_values=not args.disable_anchor_values,
                )
                sealed_row = sealed_llm_comparison_to_row(sealed)
                sealed_rows.append(sealed_row)
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
                proxy_sealed_result = run_proxy_sealed_vcg_experiment(
                    instance=scenario.instance,
                    proxies=list(_elicited.values()),
                    config=ProxySealedConfig(
                        elicitation_rounds=args.sealed_elicitation_rounds,
                        feedback_rule=args.sealed_feedback_rule,
                        max_refinements_per_bidder=(
                            args.max_refinement_queries_per_bidder
                        ),
                    ),
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
                            timeout=args.timeout,
                            logger=logger,
                            max_parse_retries=args.max_parse_retries,
                            epsilon=args.epsilon,
                            ask_initial_question=args.ask_initial_question,
                            use_ground_truth=args.ground_truth_queries,
                            verbose=args.verbose,
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
                    proxy_clock_result = run_proxy_clock_experiment(
                        instance=scenario.instance,
                        proxies=list(_elicited.values()),
                        clock_config=cfg,
                        proxy_config=ProxyClockConfig(
                            top_k=top_k,
                            elicited=True,
                            margin_threshold=args.clock_margin_threshold,
                            tie_threshold=args.clock_tie_threshold,
                            max_refinements_per_bidder=(
                                args.max_refinement_queries_per_bidder
                            ),
                        ),
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
                        )
                    )
                    if args.verbose:
                        print_arm_summary(
                            f"proxy clock k={top_k}",
                            _elicited,
                            _proxy_clock_stats,
                            _n_bidders,
                        )

            if args.elicited_ceca:
                mechanism = "proxy ceca"
                _section("proxy ceca")
                logger.mark()
                if args.ceca_proxy_type in ("vd1", "vd2", "nvd"):
                    _elicited = make_ceca_llm_proxies(
                        scenario,
                        persons=_get_persons(),
                        proxy_client=make_live_client(
                            model=args.model,
                            provider=args.provider,
                            base_url=args.base_url,
                            api_key=args.api_key,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                        ),
                        ceca_proxy_type=args.ceca_proxy_type,
                        max_bundle_size=args.max_bundle_size,
                        gamma_refresh_every=args.gamma_refresh_every,
                        nvd_num_questions=args.nvd_num_questions,
                        max_parse_retries=args.max_parse_retries,
                        logger=logger,
                    )
                elif args.ceca_proxy_type == "dnf":
                    persons = _get_persons()
                    _elicited = {
                        bidder_id: DnfLearningProxy(
                            bidder_id=bidder_id,
                            person=person,
                            items=list(scenario.instance.items),
                        )
                        for bidder_id, person in persons.items()
                    }
                else:
                    _elicited = make_elicited_proxies(
                        use_pv=not getattr(args, "ceca_no_pv", False)
                    )

                _mode_labels = {"full_proxy": "prior", "singletons": "singletons", "empty": "empty"}
                for _ceca_mode in ceca_initial_bid_modes:
                    _variant_label = _mode_labels.get(_ceca_mode, _ceca_mode)
                    _section(f"proxy ceca {_variant_label}  (elicitation)")
                    logger.mark()
                    # Run CECA once per mode — payment rules share the same run.
                    _shared = run_proxy_ceca_elicitation(
                        instance=scenario.instance,
                        proxies=list(_elicited.values()),
                        ceca_config=ceca_cfg,
                        proxy_config=ProxyCecaConfig(
                            payment_rule=ceca_payment_rules[0],
                            initial_bid_mode=_ceca_mode,
                            atomic_trimming=getattr(args, 'ceca_atomic_trimming', True),
                            trim_value_tolerance=getattr(args, 'ceca_trim_value_tolerance', 0.0),
                            stop_on_no_new_information=getattr(args, 'ceca_stop_on_no_new_information', False),
                            stall_patience=getattr(args, 'ceca_stall_patience', 1),
                            stop_on_round_no_useful_counterexamples=getattr(args, 'ceca_stop_on_round_no_useful_counterexamples', False),
                            exhaust_repeated_bidders=getattr(args, 'ceca_exhaust_repeated_bidders', False),
                            bidder_stall_patience=getattr(args, 'ceca_bidder_stall_patience', 3),
                            ceca_demand_universe=getattr(args, 'ceca_demand_universe', 'all_items'),
                            max_bundle_size=getattr(args, 'max_bundle_size', None),
                        ),
                    )
                    _proxy_ceca_stats = logger.stats_since_mark()
                    _ceca_arm = _collect_arm_stats(_proxy_ceca_stats)

                    # Per-round satisfaction diagnostic (Task 2).
                    for _diag_row in ceca_satisfaction_diagnostic_rows(_shared):
                        ceca_satisfaction_diag_rows.append({
                            "scenario": scenario.name,
                            "mode": _ceca_mode,
                            **_diag_row,
                        })

                    # Finalize with each payment rule — pure arithmetic, no LLM calls.
                    _ceca_results = {
                        rule: finalize_proxy_ceca_result(scenario.instance, _shared, rule)
                        for rule in ceca_payment_rules
                    }
                    _print_ceca_payment_table(scenario.instance, _shared, _ceca_results)

                    for payment_rule, proxy_ceca_result in _ceca_results.items():
                        proxy_ceca_row = ceca_result_to_row(
                            instance_name=scenario.name,
                            instance=scenario.instance,
                            result=proxy_ceca_result,
                        )
                        ceca_proxy_rows[(_ceca_mode, payment_rule)].append(proxy_ceca_row)
                        ceca_winner_diagnostics.extend(
                            ceca_winner_diagnostics_rows(
                                instance_name=scenario.name,
                                instance=scenario.instance,
                                result=proxy_ceca_result,
                            )
                        )
                        arm_label = f"proxy ceca {_variant_label} {payment_rule}"
                        _arm_result(
                            arm_label,
                            proxy_ceca_row["efficiency"],
                            proxy_ceca_row["proxy_true_welfare"],
                            proxy_ceca_row["full_info_welfare"],
                            extra=(
                                f"rounds {proxy_ceca_result.rounds}  "
                                f"converged={proxy_ceca_row['converged']}  "
                                f"init_atoms={proxy_ceca_row['initial_manifest_total_atoms']}  "
                                f"growth={proxy_ceca_row['manifest_growth_total']}  "
                                f"queries {proxy_ceca_row['demand_query_count_by_bidder']}"
                                + ("  [same CECA run]" if len(ceca_payment_rules) > 1 else "")
                            ),
                            reported_welfare=proxy_ceca_row["proxy_reported_welfare"],
                        )
                        _summary.append({
                            "scenario": scenario.name,
                            "arm": arm_label,
                            "efficiency": proxy_ceca_row["efficiency"],
                            "true_welfare": proxy_ceca_row["proxy_true_welfare"],
                            "full_info_welfare": proxy_ceca_row["full_info_welfare"],
                            "revenue": proxy_ceca_row["proxy_revenue"],
                            "surplus": proxy_ceca_row["proxy_true_welfare"] - proxy_ceca_row["proxy_revenue"],
                            **_ceca_arm,
                            **_amortized_shared(),
                            **_est_gt_tok(_ceca_arm),
                        })
                if args.verbose:
                    print_arm_summary(
                        "proxy ceca",
                        _elicited,
                        logger.stats_since_mark(),
                        _n_bidders,
                    )
        except Exception as exc:
            print(
                f"Scenario {scenario.name}, mechanism {mechanism} failed: "
                f"{exc}",
                file=sys.stderr,
            )
            if args.provider == "ollama":
                print_ollama_help(args.model)
            print(f"Logs were written to: {log_path}", file=sys.stderr)
            raise SystemExit(1) from exc

    # Summary table
    if _summary:
        _print_results_table(_summary)

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
    if args.elicited_clock:
        for top_k, rows in clock_proxy_rows_by_top_k.items():
            write_csv(rows, clock_proxy_paths[top_k])
            print(f"  proxy clock k={top_k} CSV    →  {clock_proxy_paths[top_k]}")
    if args.elicited_ceca:
        _mode_labels_out = {"full_proxy": "prior", "singletons": "singletons", "empty": "empty"}
        for (mode, rule), rows in ceca_proxy_rows.items():
            if rows:
                path = ceca_proxy_paths[(mode, rule)]
                label = _mode_labels_out.get(mode, mode)
                write_csv(rows, path)
                print(f"  proxy ceca {label} {rule} CSV  →  {path}")
        for mode in ceca_initial_bid_modes:
            mode_diag = [r for r in ceca_winner_diagnostics if r.get("ceca_initial_bid_mode") == mode]
            if mode_diag:
                path = ceca_diagnostics_paths[mode]
                write_csv(mode_diag, path)
                print(f"  ceca value/payment diagnostics ({mode}) CSV  →  {path}")
    if ceca_satisfaction_diag_rows:
        sat_diag_path = log_dir / "curated_ceca_satisfaction_diagnostics.csv"
        write_csv(ceca_satisfaction_diag_rows, sat_diag_path)
        print(f"  ceca satisfaction diagnostics CSV  →  {sat_diag_path}")
    if all_refinement_rows:
        write_csv(all_refinement_rows, refinement_path)
        print(f"  refinement records CSV  →  {refinement_path}")
    print(f"  logs                    →  {log_path}")


if __name__ == "__main__":
    main()
