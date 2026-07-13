"""Minimal live smoke test for an LLM provider, client, parser, and logger."""

from __future__ import annotations

import argparse
import sys

from auctionlab.llm.clients import OpenAICompatibleLlmClient
from auctionlab.llm.logging import LlmCallLogger
from auctionlab.llm.person_simulator import LlmPersonSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one live LLM-backed bundle value query."
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai-compatible"],
        default="ollama",
    )
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument(
        "--log-path",
        default="outputs/llm_runs/value_query_smoke/calls.jsonl",
    )
    parser.add_argument("--max-parse-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.provider == "ollama":
        client = OpenAICompatibleLlmClient.for_ollama(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    else:
        client = OpenAICompatibleLlmClient(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )

    scenario_description = (
        "A small auction of electronics for digital art."
    )
    item_descriptions = {
        "IPAD": (
            "Apple iPad tablet suitable for drawing and note-taking."
        ),
        "PENCIL": "Apple Pencil stylus for digital art.",
        "AIRPODS": "Wireless earbuds for listening to music.",
    }
    person_seed = (
        "Cecilia is mainly interested in digital art. She values the Apple "
        "Pencil at about $120. She values an iPad together with an Apple "
        "Pencil highly because the pair supports her drawing workflow. "
        "AirPods are mostly irrelevant."
    )
    bundle = frozenset({"IPAD", "PENCIL"})
    simulator = LlmPersonSimulator(
        bidder_id="cecilia",
        scenario_description=scenario_description,
        person_seed=person_seed,
        item_descriptions=item_descriptions,
        client=client,
        logger=LlmCallLogger(args.log_path),
        model_name=args.model,
        max_parse_retries=args.max_parse_retries,
    )

    print(f"Logging calls to: {args.log_path}", flush=True)
    print(
        "Running one live value query. Ensure Ollama is running if using "
        "--provider ollama.",
        flush=True,
    )

    try:
        value = simulator.value_query(bundle)
    except Exception as exc:
        print(f"Live value query failed: {exc}", file=sys.stderr)
        if args.provider == "ollama":
            print(
                "Ensure Ollama is running:\n"
                "  ollama serve\n"
                "Ensure model is installed:\n"
                f"  ollama pull {args.model}",
                file=sys.stderr,
            )
        print(f"Logs were written to: {args.log_path}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        "Estimated value for bundle "
        f"{sorted(bundle)}: {value}"
    )
    print(f"Logged call(s) to: {args.log_path}")


if __name__ == "__main__":
    main()
