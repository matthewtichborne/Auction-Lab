#!/usr/bin/env python3
"""Run and audit opening elicitation for one bidder in a ten-good instance.

This deliberately stops before either auction mechanism begins.  It writes
the hidden deterministic valuation table beside the LLM question, person
answer, inferred interest map, candidate bundles, and inference-accuracy
diagnostics so the substitute-mode redesign can be inspected directly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from auctionlab.instances.base import AuctionInstance
from auctionlab.instances.nl_types import NaturalLanguageAuctionScenario
from auctionlab.instances.structured_spec import make_pc_build_scenario_from_spec
from auctionlab.llm.cache import (
    CacheStats,
    CachingLlmClient,
    LlmResponseCache,
)
from auctionlab.llm.clients import OpenAICompatibleLlmClient
from auctionlab.llm.interest_map import (
    generate_candidate_bundles_from_interest_map,
    interest_map_accuracy,
    interest_map_candidate_counts,
)
from auctionlab.llm.logging import LlmCallLogger
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.prompts import canonical_opening_question
from auctionlab.llm.proxies import LlmInferredXorProxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate person/proxy opening elicitation against explicit hidden "
            "valuations for one bidder and ten goods."
        )
    )
    parser.add_argument(
        "--scenario-spec",
        default="scenarios/pc_build_v3/pc_build_population_16x16.json",
    )
    parser.add_argument("--bidder-id", default="enthusiast_gamer")
    parser.add_argument("--num-goods", type=int, default=10)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument(
        "--selection-policy",
        choices=("prefix", "seeded_sample", "stratified", "coverage_stratified"),
        default="prefix",
    )
    parser.add_argument("--person-provider", default="gemini")
    parser.add_argument("--person-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--proxy-provider", default="gemini")
    parser.add_argument("--proxy-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--verifier-provider")
    parser.add_argument("--verifier-model")
    parser.add_argument("--verifier-base-url")
    parser.add_argument("--verifier-api-key")
    parser.add_argument("--verifier-temperature", type=float, default=0.0)
    parser.add_argument("--verifier-max-tokens", type=int, default=2000)
    parser.add_argument(
        "--opening-question-policy",
        choices=("canonical", "proxy_generated"),
        default="canonical",
    )
    parser.add_argument("--opening-question")
    parser.add_argument("--person-base-url")
    parser.add_argument("--proxy-base-url")
    parser.add_argument("--person-api-key")
    parser.add_argument("--proxy-api-key")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-parse-retries", type=int, default=1)
    parser.add_argument(
        "--generate-provisional-valuations",
        action="store_true",
        help=(
            "Also run the proxy's bulk provisional-valuation stage and "
            "compare it with hidden deterministic values."
        ),
    )
    parser.add_argument(
        "--pv-chunk-size",
        type=int,
        default=0,
        help="Bundles per PV call; 0 keeps the complete candidate set in one call.",
    )
    parser.add_argument(
        "--pv-top-k",
        type=int,
        default=10,
        help="Top-k bundle overlap reported by optional PV validation.",
    )
    parser.add_argument(
        "--interest-map-failure-policy",
        choices=("raise", "all_items"),
        default="raise",
        help=(
            "Use all_items only to demonstrate the conservative parser-failure "
            "fallback; raise is recommended for validation."
        ),
    )
    parser.add_argument(
        "--llm-cache-mode",
        choices=("off", "read-write", "read-only", "refresh"),
        default="read-write",
    )
    parser.add_argument(
        "--llm-cache-path",
        default="cache/single_bidder_validation.sqlite",
    )
    parser.add_argument(
        "--output",
        default="outputs/validation/single_bidder_10_goods.json",
    )
    parser.add_argument(
        "--calls-log",
        default="outputs/validation/single_bidder_10_goods_calls.jsonl",
    )
    parser.add_argument(
        "--print-all-valuations",
        action="store_true",
        help="Also print all 2^n-1 hidden bundle values to stdout.",
    )
    return parser.parse_args()


def _raw_client(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> OpenAICompatibleLlmClient:
    if provider == "gemini":
        return OpenAICompatibleLlmClient.for_gemini(
            model=model,
            base_url=base_url
            or "https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
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
    if provider == "ollama":
        return OpenAICompatibleLlmClient.for_ollama(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    if provider == "openai":
        return OpenAICompatibleLlmClient.for_openai(
            model=model,
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            temperature=(
                None if model.startswith("gpt-5") else temperature
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


def _client(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    args: argparse.Namespace,
    cache: LlmResponseCache | None,
    cache_stats: CacheStats,
    role: str,
) -> Any:
    client: Any = _raw_client(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    if cache is not None:
        client = CachingLlmClient(
            inner=client,
            cache=cache,
            mode=args.llm_cache_mode,
            provider=provider,
            model=model,
            temperature=client.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=client.reasoning_effort,
            stats=cache_stats,
        )
    setattr(client, "_auctionlab_provider", provider)
    setattr(client, "_auctionlab_model", model)
    setattr(client, "_auctionlab_llm_role", role)
    return client


def _one_bidder_scenario(args: argparse.Namespace) -> NaturalLanguageAuctionScenario:
    # Select goods against the full bidder population, then narrow to the
    # requested bidder. This permits any named bidder without allowing bidder
    # selection to alter the ten-good catalogue under test.
    full = make_pc_build_scenario_from_spec(
        args.scenario_spec,
        num_goods=args.num_goods,
        num_bidders=16,
        seed=args.scenario_seed,
        selection_policy=args.selection_policy,
        name=f"pc_build_{args.num_goods}x1_{args.bidder_id}_validation",
    )
    if args.bidder_id not in full.instance.bidder_ids:
        raise ValueError(
            f"unknown bidder {args.bidder_id!r}; available={full.instance.bidder_ids}"
        )
    bidder_id = args.bidder_id
    metadata = dict(full.metadata)
    metadata["num_bidders"] = 1
    metadata["profiles"] = {
        bidder_id: full.metadata["profiles"][bidder_id]
    }
    return NaturalLanguageAuctionScenario(
        name=full.name,
        seed_type=full.seed_type,
        instance=AuctionInstance(
            items=list(full.instance.items),
            bidder_ids=[bidder_id],
            valuations={
                bidder_id: full.instance.valuations[bidder_id]
            },
        ),
        scenario_description=(
            f"{full.scenario_description} This validation run contains one bidder."
        ),
        item_descriptions=dict(full.item_descriptions),
        person_seeds={bidder_id: full.person_seeds[bidder_id]},
        candidate_bundles_by_bidder=None,
        metadata=metadata,
    )


def _bundle_rows(values: dict[frozenset[str], float]) -> list[dict[str, Any]]:
    return [
        {"bundle": sorted(bundle), "value": float(value)}
        for bundle, value in sorted(
            values.items(),
            key=lambda pair: (len(pair[0]), tuple(sorted(pair[0]))),
        )
    ]


def _average_ranks(values: list[float]) -> list[float]:
    """Return one-based average ranks, with smaller values ranked first."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def provisional_value_accuracy(
    predicted: dict[frozenset[str], float],
    truth: dict[frozenset[str], float],
    *,
    top_k: int,
) -> dict[str, Any]:
    bundles = sorted(
        predicted,
        key=lambda bundle: (len(bundle), tuple(sorted(bundle))),
    )
    errors = [
        float(predicted[bundle]) - float(truth.get(bundle, 0.0))
        for bundle in bundles
    ]
    predicted_values = [float(predicted[bundle]) for bundle in bundles]
    true_values = [float(truth.get(bundle, 0.0)) for bundle in bundles]
    predicted_order = sorted(
        bundles,
        key=lambda bundle: (
            -float(predicted[bundle]),
            len(bundle),
            tuple(sorted(bundle)),
        ),
    )
    candidate_true_order = sorted(
        bundles,
        key=lambda bundle: (
            -float(truth.get(bundle, 0.0)),
            len(bundle),
            tuple(sorted(bundle)),
        ),
    )
    effective_k = min(max(top_k, 1), len(bundles))
    predicted_top = set(predicted_order[:effective_k])
    candidate_true_top = set(candidate_true_order[:effective_k])
    top_k_boundary_value = (
        float(truth.get(candidate_true_order[effective_k - 1], 0.0))
        if effective_k
        else None
    )
    candidate_true_top_tie_set = {
        bundle
        for bundle in bundles
        if (
            top_k_boundary_value is not None
            and float(truth.get(bundle, 0.0)) >= top_k_boundary_value
        )
    }
    chosen = predicted_order[0] if predicted_order else frozenset()
    global_best_value = max(truth.values(), default=0.0)
    chosen_true_value = float(truth.get(chosen, 0.0))
    return {
        "bundle_count": len(bundles),
        "mae": mean(abs(error) for error in errors) if errors else None,
        "rmse": (
            math.sqrt(mean(error * error for error in errors))
            if errors else None
        ),
        "mean_error": mean(errors) if errors else None,
        "spearman_rank_correlation": _pearson(
            _average_ranks(predicted_values),
            _average_ranks(true_values),
        ),
        "top_k": effective_k,
        "candidate_true_top_k_recall": (
            len(predicted_top & candidate_true_top) / effective_k
            if effective_k else None
        ),
        "candidate_top_k_overlap_count": len(
            predicted_top & candidate_true_top
        ),
        "candidate_top_k_tie_aware_hit_rate": (
            len(predicted_top & candidate_true_top_tie_set) / effective_k
            if effective_k else None
        ),
        "candidate_true_top_tie_set_size": len(
            candidate_true_top_tie_set
        ),
        "candidate_true_top_k_boundary_value": top_k_boundary_value,
        "predicted_best_bundle": sorted(chosen),
        "predicted_best_reported_value": (
            float(predicted[chosen]) if chosen else None
        ),
        "predicted_best_true_value": chosen_true_value,
        "global_true_best_value": float(global_best_value),
        "single_bidder_welfare_efficiency": (
            chosen_true_value / global_best_value
            if global_best_value > 0
            else 1.0
        ),
        "global_true_optimum_in_candidate_support": any(
            bundle in predicted
            for bundle, value in truth.items()
            if value == global_best_value
        ),
        "predicted_max_to_true_max_ratio": (
            max(predicted_values) / global_best_value
            if predicted_values and global_best_value > 0
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    if args.num_goods != 10:
        raise ValueError("this focused validator requires --num-goods 10")

    scenario = _one_bidder_scenario(args)
    bidder_id = args.bidder_id
    output_path = Path(args.output)
    logger = LlmCallLogger(args.calls_log)
    cache_stats = CacheStats()
    cache = (
        None
        if args.llm_cache_mode == "off"
        else LlmResponseCache(args.llm_cache_path)
    )

    try:
        person_client = _client(
            provider=args.person_provider,
            model=args.person_model,
            base_url=args.person_base_url,
            api_key=args.person_api_key,
            args=args,
            cache=cache,
            cache_stats=cache_stats,
            role="person",
        )
        proxy_client = _client(
            provider=args.proxy_provider,
            model=args.proxy_model,
            base_url=args.proxy_base_url,
            api_key=args.proxy_api_key,
            args=args,
            cache=cache,
            cache_stats=cache_stats,
            role="proxy",
        )
        verifier_provider = (
            args.verifier_provider
            or scenario.metadata.get("environment_generation_provider")
            or args.proxy_provider
        )
        verifier_model = (
            args.verifier_model
            or scenario.metadata.get("environment_generation_model")
            or args.proxy_model
        )
        verifier_values = vars(args).copy()
        verifier_values.update(
            max_tokens=args.verifier_max_tokens,
            temperature=args.verifier_temperature,
        )
        verifier_args = argparse.Namespace(**verifier_values)
        verifier_client = _client(
            provider=verifier_provider,
            model=verifier_model,
            base_url=args.verifier_base_url,
            api_key=args.verifier_api_key,
            args=verifier_args,
            cache=cache,
            cache_stats=cache_stats,
            role="verifier",
        )
        truth_values = scenario.instance.valuations[bidder_id]
        true_interested = {
            item
            for item in scenario.instance.items
            if truth_values.get(frozenset({item}), 0.0) > 0
        }
        profile_metadata = scenario.metadata["profiles"][bidder_id]
        true_groups = profile_metadata["substitute_groups"]
        true_complements = profile_metadata["complement_groups"]
        person = LlmPersonSimulator(
            bidder_id=bidder_id,
            scenario_description=scenario.scenario_description,
            person_seed=scenario.person_seeds[bidder_id],
            item_descriptions=scenario.item_descriptions,
            client=person_client,
            verifier_client=verifier_client,
            logger=logger,
            model_name=args.person_model,
            provider_name=args.person_provider,
            verifier_model_name=verifier_model,
            verifier_provider_name=verifier_provider,
            max_parse_retries=args.max_parse_retries,
            ground_truth_valuations=None,
            verbose=True,
            scenario_id=scenario.name,
            expected_interested_items=true_interested,
            expected_excluded_items=(
                set(scenario.instance.items) - true_interested
            ),
            expected_substitute_groups=true_groups,
            expected_complement_groups=true_complements,
            expected_budget_hint=profile_metadata.get(
                "disclosed_budget_hint",
                max(truth_values.values(), default=0.0),
            ),
        )
        proxy = LlmInferredXorProxy(bidder_id=bidder_id, person=person)
        opening_question = (
            args.opening_question.strip()
            if args.opening_question is not None
            else (
                canonical_opening_question(
                    domain=scenario.metadata.get("domain")
                )
                if args.opening_question_policy == "canonical"
                else None
            )
        )
        if opening_question == "":
            raise ValueError("--opening-question must be non-empty")
        proxy.ask_initial_question(
            question_client=(
                proxy_client if opening_question is None else None
            ),
            question=opening_question,
        )
        interest_map = proxy.build_interest_map(
            client=proxy_client,
            failure_policy=args.interest_map_failure_policy,
        )
        candidates = generate_candidate_bundles_from_interest_map(
            interest_map,
            list(scenario.instance.items),
        )
        provisional_values: dict[frozenset[str], float] | None = None
        provisional_accuracy: dict[str, Any] | None = None
        if args.generate_provisional_valuations:
            provisional_values = proxy.build_provisional_valuations(
                candidates,
                client=proxy_client,
                interest_map=interest_map,
                discount_inferred=False,
                pv_chunk_size=args.pv_chunk_size or None,
                max_parse_retries=args.max_parse_retries,
            )
            provisional_accuracy = provisional_value_accuracy(
                provisional_values,
                truth_values,
                top_k=args.pv_top_k,
            )

        question, answer = proxy.nl_transcript[-1]
        accuracy = interest_map_accuracy(
            interest_map,
            true_interested_items=true_interested,
            true_substitute_groups=true_groups,
            true_complement_groups=true_complements,
            available_items=set(scenario.instance.items),
            nl_answer=answer,
            singleton_values={
                item: truth_values.get(frozenset({item}), 0.0)
                for item in scenario.instance.items
            },
        )
        before_count, after_count = interest_map_candidate_counts(
            interest_map, list(scenario.instance.items)
        )
        payload = {
            "scenario": {
                "name": scenario.name,
                "items": scenario.instance.items,
                "item_descriptions": scenario.item_descriptions,
                "bidder_id": bidder_id,
                "person_seed": scenario.person_seeds[bidder_id],
            },
            "ground_truth": {
                "singleton_values": {
                    item: truth_values.get(frozenset({item}), 0.0)
                    for item in scenario.instance.items
                },
                "substitute_groups": true_groups,
                "complement_groups": true_complements,
                "all_bundle_valuations": _bundle_rows(truth_values),
            },
            "elicitation": {
                "proxy_question": question,
                "person_answer": answer,
                "inferred_interest_map": interest_map.model_dump(),
                "candidate_count_before_choose_one_filter": before_count,
                "candidate_count_after_choose_one_filter": after_count,
                "candidate_bundles": [
                    sorted(bundle) for bundle in candidates
                ],
                "accuracy": accuracy,
                "provisional_valuations": (
                    _bundle_rows(provisional_values)
                    if provisional_values is not None
                    else None
                ),
                "provisional_valuation_accuracy": provisional_accuracy,
            },
            "models": {
                "person": {
                    "provider": args.person_provider,
                    "model": args.person_model,
                },
                "proxy": {
                    "provider": args.proxy_provider,
                    "model": args.proxy_model,
                },
            },
            "opening_question": {
                "policy": (
                    "explicit"
                    if args.opening_question is not None
                    else args.opening_question_policy
                ),
                "text": question,
            },
            "person_answer_diagnostics": {
                "verification": person.last_answer_verification,
                "verification_history": person.answer_verification_history,
                "attempt_count": person.answer_attempt_count,
                "first_word_count": person.first_answer_word_count,
                "final_word_count": person.final_answer_word_count,
                "verifier_provider": verifier_provider,
                "verifier_model": verifier_model,
            },
            "cache": {
                "mode": args.llm_cache_mode,
                "path": args.llm_cache_path,
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "writes": cache_stats.writes,
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        print("\nValidation result")
        print(f"  bidder: {bidder_id}")
        print(f"  items: {scenario.instance.items}")
        print(f"  person seed: {scenario.person_seeds[bidder_id]}")
        print(f"  question: {question}")
        print(f"  answer: {answer}")
        print(f"  true substitute groups: {true_groups}")
        print(
            "  inferred substitute groups: "
            f"{interest_map.model_dump()['substitute_groups']}"
        )
        print(f"  accuracy: {json.dumps(accuracy, sort_keys=True)}")
        print(f"  candidates: {before_count} -> {after_count}")
        if provisional_accuracy is not None:
            print(
                "  provisional valuation accuracy: "
                f"{json.dumps(provisional_accuracy, sort_keys=True)}"
            )
        print(f"  full audit: {output_path}")
        print(f"  call log: {args.calls_log}")
        if args.print_all_valuations:
            print(
                json.dumps(
                    payload["ground_truth"]["all_bundle_valuations"],
                    indent=2,
                )
            )
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    main()
