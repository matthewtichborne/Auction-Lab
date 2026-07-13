"""Bulk provisional valuation from a single LLM call over the NL Q/A pair."""

from __future__ import annotations

import time

from auctionlab.auction_types import Bundle, Item
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.logging import LlmCallLogger, LlmCallRecord, current_timestamp
from auctionlab.llm.parsing import parse_provisional_valuations_response
from auctionlab.llm.prompts import build_provisional_valuation_prompt
from auctionlab.llm.schemas import LlmInterestMap


_TOKENS_PER_BUNDLE_ESTIMATE = 50
"""Estimated output tokens consumed per bundle entry in the PV JSON response.

Used to auto-cap the bundle list to what can fit in the model's token budget.
Each entry serialises to roughly ``["ITEM_ID", ...]`` + a number + overhead.
"""


def _max_bundles_for_token_budget(max_tokens: int) -> int:
    """How many bundles can fit in ``max_tokens`` output tokens (conservatively)."""
    # Reserve ~200 tokens for the reasoning field and JSON envelope.
    usable = max(max_tokens - 200, 1)
    return max(usable // _TOKENS_PER_BUNDLE_ESTIMATE, 1)


def generate_provisional_valuations(
    *,
    client: LlmClient,
    scenario_description: str,
    item_descriptions: dict[Item, str],
    nl_question: str,
    nl_answer: str,
    candidate_bundles: list[Bundle],
    interest_map: LlmInterestMap | None = None,
    logger: LlmCallLogger | None = None,
    bidder_id: str | None = None,
    model_name: str | None = None,
    max_bundles: int | None = None,
) -> dict[Bundle, float]:
    """Call the LLM once to estimate values for all ``candidate_bundles``.

    Returns a mapping from every non-empty bundle in ``candidate_bundles``
    to a non-negative estimated value. Bundles absent from the LLM response
    default to 0.0. The optional ``interest_map`` enriches the prompt with
    complement/substitute structure and a budget hint; the function is fully
    usable without it. Deliberately does not take the person's preference
    seed -- estimates are grounded only in what ``nl_question``/``nl_answer``
    actually revealed, not private knowledge of the person's true values.

    ``max_bundles``, when set, silently truncates ``candidate_bundles`` to the
    first ``max_bundles`` entries before building the prompt.  Callers
    should ensure the highest-priority bundles (complement groups, singletons)
    appear first, which :meth:`LlmInferredXorProxy.candidate_bundles_from_interest_map`
    already guarantees.  When ``None`` (default), a conservative cap is derived
    automatically from the client's ``max_tokens`` setting.
    """
    if not candidate_bundles:
        return {}

    # Auto-cap based on the client's token budget if no explicit limit set.
    client_max = getattr(client, "max_tokens", None)
    effective_max = max_bundles if max_bundles is not None else (
        _max_bundles_for_token_budget(client_max) if client_max else None
    )
    if effective_max is not None and len(candidate_bundles) > effective_max:
        import warnings
        label = bidder_id or "?"
        warnings.warn(
            f"PV for {label}: {len(candidate_bundles)} candidate bundles exceed "
            f"the estimated token budget (max_bundles={effective_max}). "
            f"Truncating to the first {effective_max} bundles (complement-group "
            f"bundles are prioritised by candidate_bundles_from_interest_map). "
            f"Pass --max-candidate-bundles to control this.",
            UserWarning,
            stacklevel=2,
        )
        candidate_bundles = candidate_bundles[:effective_max]

    prompt = build_provisional_valuation_prompt(
        scenario_description=scenario_description,
        item_descriptions=item_descriptions,
        nl_question=nl_question,
        nl_answer=nl_answer,
        candidate_bundles=candidate_bundles,
        interest_map=interest_map,
    )
    started = time.perf_counter()
    raw = client.complete(prompt)
    latency = time.perf_counter() - started

    result = parse_provisional_valuations_response(raw, candidate_bundles)

    if logger is not None:
        logger.log(
            LlmCallRecord(
                timestamp=current_timestamp(),
                bidder_id=bidder_id,
                prompt_type="proxy_provisional_valuations",
                prompt=prompt,
                raw_response=raw,
                parsed_response={
                    "n_bundles": len(result),
                    "valuations": {
                        str(sorted(b)): v for b, v in result.items()
                    },
                },
                success=True,
                error=None,
                latency_seconds=latency,
                model=model_name,
                attempt=1,
                input_tokens=getattr(client, "_last_input_tokens", None),
                output_tokens=getattr(client, "_last_output_tokens", None),
                total_tokens=getattr(client, "_last_total_tokens", None),
            )
        )

    return result
