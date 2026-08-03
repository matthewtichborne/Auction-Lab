"""Run sealed and clock mechanisms using XOR bids inferred by LLM proxies."""

from __future__ import annotations

from typing import Callable

from auctionlab.auction_types import Bundle, validate_bidder_keys
from auctionlab.auctions.clock import (
    ClockConfig,
    finalize_from_supplementary_vcg,
    run_ascending_clock_with_supplementary,
)
from auctionlab.auctions.sealed_vcg import run_sealed_xor_vcg
from auctionlab.bids.xor import XorBid
from auctionlab.experiments.runner import MechanismResult
from auctionlab.instances.base import AuctionInstance
from auctionlab.llm.clients import LlmClient
from auctionlab.llm.knowledge import KnowledgeBase
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import LlmAuctionProxyAdapter, LlmInferredXorProxy


def _resolve_candidate_bundles(
    *,
    instance: AuctionInstance,
    candidate_bundles: list[Bundle] | None,
    candidate_bundles_by_bidder: dict[str, list[Bundle]] | None,
) -> tuple[dict[str, list[Bundle]], dict[str, int], bool]:
    """Normalize shared or bidder-specific candidate bundle sets."""
    targeted = candidate_bundles_by_bidder is not None
    if targeted:
        validate_bidder_keys(
            bidder_ids=instance.bidder_ids,
            values=candidate_bundles_by_bidder,
            label="candidate_bundles_by_bidder",
        )
        bundles_by_bidder = candidate_bundles_by_bidder
    elif candidate_bundles is not None:
        bundles_by_bidder = {
            bidder_id: candidate_bundles
            for bidder_id in instance.bidder_ids
        }
    else:
        raise ValueError(
            "candidate_bundles or candidate_bundles_by_bidder is required"
        )

    valid_items = set(instance.items)
    for bidder_id, bundles in bundles_by_bidder.items():
        for bundle in bundles:
            unknown_items = sorted(set(bundle) - valid_items)
            if unknown_items:
                raise ValueError(
                    f"Candidate bundle for {bidder_id} contains unknown "
                    f"items: {unknown_items}"
                )

    counts = {
        bidder_id: sum(1 for bundle in bundles_by_bidder[bidder_id] if bundle)
        for bidder_id in instance.bidder_ids
    }
    return bundles_by_bidder, counts, targeted


def run_sealed_llm_proxy_experiment(
    instance: AuctionInstance,
    proxies: dict[str, LlmInferredXorProxy],
    candidate_bundles: list[Bundle] | None = None,
    *,
    candidate_bundles_by_bidder: dict[str, list[Bundle]] | None = None,
    discount_inferred: bool = True,
    use_anchor_values: bool = True,
) -> MechanismResult:
    """Infer one XOR bid per bidder, then run sealed-bid VCG."""
    validate_bidder_keys(
        bidder_ids=instance.bidder_ids,
        values=proxies,
        label="proxies",
    )
    bundles_by_bidder, candidate_counts, targeted = (
        _resolve_candidate_bundles(
            instance=instance,
            candidate_bundles=candidate_bundles,
            candidate_bundles_by_bidder=candidate_bundles_by_bidder,
        )
    )

    reported_bids: dict[str, XorBid] = {}
    for bidder_id in instance.bidder_ids:
        reported_bids[bidder_id] = proxies[bidder_id].infer_xor_bid(
            bundles_by_bidder[bidder_id],
            discount_inferred=discount_inferred,
            use_anchor_values=use_anchor_values,
        )

    outcome = run_sealed_xor_vcg(
        items=instance.items,
        bids=[
            reported_bids[bidder_id]
            for bidder_id in instance.bidder_ids
        ],
    )
    candidate_bundle_count = sum(candidate_counts.values())

    return MechanismResult(
        mechanism="sealed_llm_proxy_vcg",
        allocation=outcome.allocation,
        welfare=outcome.welfare,
        payments=outcome.payments,
        revenue=sum(outcome.payments.values()),
        rounds=None,
        query_count=candidate_bundle_count,
        metadata={
            "candidate_bundle_count": candidate_bundle_count,
            "candidate_bundle_count_by_bidder": candidate_counts,
            "targeted_candidate_bundles": targeted,
            "discount_inferred": discount_inferred,
            "use_anchor_values": use_anchor_values,
            "epsilon_by_bidder": {
                bidder_id: proxies[bidder_id].epsilon
                for bidder_id in instance.bidder_ids
            },
            "reported_bids": reported_bids,
            "vcg_counterfactuals": outcome.vcg_counterfactuals,
        },
    )


def run_clock_llm_proxy_experiment(
    instance: AuctionInstance,
    proxies: dict[str, LlmInferredXorProxy],
    candidate_bundles: list[Bundle] | None = None,
    cfg: ClockConfig | None = None,
    *,
    candidate_bundles_by_bidder: dict[str, list[Bundle]] | None = None,
    top_k: int = 1,
    discount_inferred: bool = True,
    use_anchor_values: bool = True,
    elicited: bool = False,
    margin_threshold: float = 100.0,
    tie_threshold: float = 100.0,
    max_refinement_queries_per_bidder: int = 0,
    refinement_strategy: str = "value_query",
) -> MechanismResult:
    """Infer bids once, then answer every clock round from the cached bids.

    Query counts distinguish expensive value inference from deterministic
    clock-demand lookups while preserving their combined historical metric.
    """
    validate_bidder_keys(
        bidder_ids=instance.bidder_ids,
        values=proxies,
        label="proxies",
    )
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if margin_threshold < 0.0:
        raise ValueError("margin_threshold must be non-negative")
    if tie_threshold < 0.0:
        raise ValueError("tie_threshold must be non-negative")
    if max_refinement_queries_per_bidder < 0:
        raise ValueError(
            "max_refinement_queries_per_bidder must be non-negative"
        )
    if refinement_strategy not in ("value_query", "demand_query"):
        raise ValueError(
            "refinement_strategy must be 'value_query' or 'demand_query', "
            f"got {refinement_strategy!r}"
        )
    if cfg is None:
        cfg = ClockConfig()
    bundles_by_bidder, candidate_counts, targeted = (
        _resolve_candidate_bundles(
            instance=instance,
            candidate_bundles=candidate_bundles,
            candidate_bundles_by_bidder=candidate_bundles_by_bidder,
        )
    )

    reported_bids: dict[str, XorBid] = {}
    for bidder_id in instance.bidder_ids:
        reported_bids[bidder_id] = proxies[bidder_id].infer_cached_xor_bid(
            bundles_by_bidder[bidder_id],
            discount_inferred=discount_inferred,
            use_anchor_values=use_anchor_values,
        )

    previous_primary_by_bidder: dict[str, Bundle | None] = {
        bidder_id: None
        for bidder_id in instance.bidder_ids
    }
    round_idx_by_bidder = {
        bidder_id: 0
        for bidder_id in instance.bidder_ids
    }

    if elicited:
        for proxy in proxies.values():
            proxy.reset_refinement_state()

        def demand_oracle(bidder_id, prices):
            round_idx = round_idx_by_bidder[bidder_id]
            response = proxies[bidder_id].clock_demand_with_refinement(
                prices=prices,
                top_k=top_k,
                round_idx=round_idx,
                previous_primary_bundle=previous_primary_by_bidder[bidder_id],
                margin_threshold=margin_threshold,
                tie_threshold=tie_threshold,
                max_refinement_queries=(
                    max_refinement_queries_per_bidder
                ),
                use_anchor_values=use_anchor_values,
                refinement_strategy=refinement_strategy,
            )
            previous_primary_by_bidder[bidder_id] = response.primary_bundle
            round_idx_by_bidder[bidder_id] += 1
            return response
    else:
        def demand_oracle(bidder_id, prices):
            return proxies[bidder_id].clock_demand_from_cached_bid(
                prices,
                top_k=top_k,
            )

    state, _provisional = run_ascending_clock_with_supplementary(
        items=instance.items,
        bidder_ids=instance.bidder_ids,
        demand_oracle=demand_oracle,
        cfg=cfg,
    )
    outcome = finalize_from_supplementary_vcg(
        items=instance.items,
        supplementary_atoms=state.supplementary,
    )

    candidate_bundle_count = sum(candidate_counts.values())
    value_query_count = candidate_bundle_count
    clock_demand_query_count = len(state.history) * len(instance.bidder_ids)
    refinement_query_count_by_bidder = (
        {
            bidder_id: proxies[bidder_id].refinement_query_count
            for bidder_id in instance.bidder_ids
        }
        if elicited
        else {
            bidder_id: 0
            for bidder_id in instance.bidder_ids
        }
    )
    refinement_query_count = sum(
        refinement_query_count_by_bidder.values()
    )
    mechanism_prefix = (
        "elicited_clock_llm_proxy_vcg"
        if elicited
        else "clock_llm_proxy_vcg"
    )

    return MechanismResult(
        mechanism=f"{mechanism_prefix}_top_{top_k}",
        allocation=outcome.allocation,
        welfare=outcome.welfare,
        payments=outcome.payments,
        revenue=sum(outcome.payments.values()),
        rounds=len(state.history),
        query_count=value_query_count + clock_demand_query_count + refinement_query_count,
        metadata={
            "candidate_bundle_count": candidate_bundle_count,
            "candidate_bundle_count_by_bidder": candidate_counts,
            "targeted_candidate_bundles": targeted,
            "top_k": top_k,
            "discount_inferred": discount_inferred,
            "use_anchor_values": use_anchor_values,
            "vcg_counterfactuals": outcome.vcg_counterfactuals,
            "elicited": elicited,
            "margin_threshold": margin_threshold,
            "tie_threshold": tie_threshold,
            "max_refinement_queries_per_bidder": (
                max_refinement_queries_per_bidder
            ),
            "refinement_strategy": refinement_strategy,
            "refinement_query_count_by_bidder": (
                refinement_query_count_by_bidder
            ),
            "epsilon_by_bidder": {
                bidder_id: proxies[bidder_id].epsilon
                for bidder_id in instance.bidder_ids
            },
            "reported_bids": reported_bids,
            "supplementary_atoms": sum(
                len(atoms)
                for atoms in state.supplementary.values()
            ),
            "final_prices": state.prices,
            "clock_demand_query_count": clock_demand_query_count,
        },
    )


def make_llm_proxies_for_instance(
    *,
    instance: AuctionInstance,
    scenario_description: str,
    person_seeds: dict[str, str],
    item_descriptions: dict[str, str],
    clients: dict[str, LlmClient],
    epsilon: float = 0.75,
    ask_initial_question: bool = False,
    knowledge_base_factory: Callable[[str], KnowledgeBase] | None = None,
) -> dict[str, LlmInferredXorProxy]:
    """Construct one consistently configured proxy per instance bidder.

    ``knowledge_base_factory``, if provided, is called with each ``bidder_id``
    and must return a fresh :class:`~auctionlab.llm.knowledge.KnowledgeBase`
    for that bidder.  Pass a factory that returns a
    :class:`~auctionlab.llm.knowledge.SummaryKnowledgeBase` to use rolling
    LLM-generated summaries instead of the default raw Q/A transcript.
    """
    validate_bidder_keys(
        bidder_ids=instance.bidder_ids,
        values=person_seeds,
        label="person_seeds",
    )
    validate_bidder_keys(
        bidder_ids=instance.bidder_ids,
        values=clients,
        label="clients",
    )

    proxies: dict[str, LlmInferredXorProxy] = {}
    for bidder_id in instance.bidder_ids:
        person = LlmPersonSimulator(
            bidder_id=bidder_id,
            scenario_description=scenario_description,
            person_seed=person_seeds[bidder_id],
            item_descriptions=item_descriptions,
            client=clients[bidder_id],
        )
        proxy = LlmInferredXorProxy(
            bidder_id=bidder_id,
            person=person,
            epsilon=epsilon,
            knowledge_base=(
                knowledge_base_factory(bidder_id)
                if knowledge_base_factory is not None
                else None
            ),
        )
        if ask_initial_question:
            proxy.ask_initial_question()
        proxies[bidder_id] = proxy

    return proxies
