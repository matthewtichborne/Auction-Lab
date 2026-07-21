#!/usr/bin/env python3
"""One-time, manually-triggered LLM generation of a PC-build profile spec.

This script is never invoked automatically by any test, experiment runner,
or CI step — it makes a live network call to an LLM provider and costs
real API credits. It is intended to be run by hand, once, to produce a
trial spec such as ``pc_build_profiles_v1_gemini_trial.json``.

It:

1. Reads ``scenarios/pc_build_v1/prompt_generate_profiles.md``.
2. Sends it to the requested provider/model.
3. Saves the raw completion text verbatim (``--raw-output``).
4. Extracts a JSON object from the completion (tolerating stray markdown
   fences around the JSON).
5. Validates it against the ``pc_build_profile_spec_v1`` schema.
6. On success, writes the frozen spec (``--output``) and a generation
   manifest (``--manifest``) recording provider/model/params/file
   provenance. On schema failure, the raw output is still saved for
   debugging, but no frozen spec or manifest is written.

Example (run manually — DO NOT run this from an agent or test)::

    ./venv/bin/python scripts/generate_scenario_spec_with_llm.py \\
        --provider gemini \\
        --model gemini-3.1-flash-lite \\
        --api-key "$GEMINI_API_KEY" \\
        --prompt scenarios/pc_build_v1/prompt_generate_profiles.md \\
        --raw-output scenarios/pc_build_v1/raw_gemini_flash_lite_response.json \\
        --output scenarios/pc_build_v1/pc_build_profiles_v1_gemini_trial.json \\
        --manifest scenarios/pc_build_v1/generation_manifest_v1_gemini_trial.yaml \\
        --temperature 0.2 \\
        --max-tokens 12000
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from auctionlab.instances.scenario_spec import validate_spec_dict, write_scenario_profile_spec
from auctionlab.llm.clients import OpenAICompatibleLlmClient


def build_client(
    provider: str,
    model: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int | None,
) -> OpenAICompatibleLlmClient:
    if provider == "gemini":
        return OpenAICompatibleLlmClient.for_gemini(
            model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens
        )
    if provider == "groq":
        return OpenAICompatibleLlmClient.for_groq(
            model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens
        )
    if provider == "ollama":
        return OpenAICompatibleLlmClient.for_ollama(
            model=model, temperature=temperature, max_tokens=max_tokens
        )
    raise ValueError(f"Unsupported provider: {provider!r} (expected gemini/groq/ollama)")


def extract_json_object(raw_text: str) -> dict:
    """Parse a JSON object out of raw model output, tolerating markdown fences."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate a JSON object in the model output")
    return json.loads(text[start : end + 1])


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def write_manifest(
    path: Path,
    *,
    provider: str,
    model: str,
    temperature: float,
    top_p: float | None,
    max_tokens: int | None,
    prompt_file: Path,
    raw_output_file: Path,
    frozen_spec_file: Path | None,
    schema_valid: bool,
    validation_report_file: Path | None,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "generator:",
        f"  provider: {provider}",
        f"  model: {model}",
        f"  generated_at: {generated_at!r}",
        f"  temperature: {temperature}",
        f"  top_p: {top_p if top_p is not None else 'null'}",
        f"  max_tokens: {max_tokens if max_tokens is not None else 'null'}",
        f"  prompt_file: {prompt_file}",
        f"  raw_output_file: {raw_output_file}",
        f"  frozen_spec_file: {frozen_spec_file if frozen_spec_file is not None else 'null'}",
        "  human_edits: none",
        f"  code_commit: {_git_commit()}",
        "",
        "validation:",
        f"  schema_valid: {'true' if schema_valid else 'false'}",
        "  economic_valid: null  # fill in after manually reviewing the validation report",
        f"  validation_report_file: {validation_report_file if validation_report_file is not None else 'null'}",
        "",
        "notes:",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["gemini", "groq", "ollama"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=12000)
    args = parser.parse_args()

    prompt_text = args.prompt.read_text(encoding="utf-8")
    client = build_client(
        args.provider, args.model, args.api_key, args.temperature, args.max_tokens
    )

    raw_text = client.complete(prompt_text)

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(raw_text, encoding="utf-8")
    print(f"Wrote raw output to {args.raw_output}")

    try:
        data = extract_json_object(raw_text)
    except ValueError as exc:
        print(f"Failed to extract JSON from model output: {exc}")
        raise SystemExit(1) from exc

    result = validate_spec_dict(data)
    if not result.valid:
        print("Generated spec FAILED schema validation:")
        for err in result.errors:
            print(f"  - {err}")
        write_manifest(
            args.manifest,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            prompt_file=args.prompt,
            raw_output_file=args.raw_output,
            frozen_spec_file=None,
            schema_valid=False,
            validation_report_file=None,
        )
        raise SystemExit(1)

    assert result.spec is not None
    write_scenario_profile_spec(result.spec, args.output)
    print(f"Wrote frozen spec to {args.output}")

    write_manifest(
        args.manifest,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        prompt_file=args.prompt,
        raw_output_file=args.raw_output,
        frozen_spec_file=args.output,
        schema_valid=True,
        validation_report_file=None,
    )
    print(f"Wrote generation manifest to {args.manifest}")
    print(
        "Next step: run scripts/validate_scenario_spec.py on the frozen spec, "
        "then fill in economic_valid and validation_report_file in the manifest."
    )


if __name__ == "__main__":
    main()
