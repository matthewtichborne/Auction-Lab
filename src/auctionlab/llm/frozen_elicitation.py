"""Versioned, auditable frozen initial-elicitation artefacts.

A frozen pack contains the scenario-level opening question, simulated-person
answer, parsed interest map, deterministic candidate support, and *raw*
provisional LLM valuations for one materialised auction instance.  Mechanism
runners can replay this state without making initial LLM calls; calibration
is deliberately applied later, at replay time.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from auctionlab.auction_types import Bundle
from auctionlab.llm.provisional_valuations import (
    PvCandidateBundleStats,
    PvChunkStats,
)
from auctionlab.llm.schemas import LlmInterestMap


FROZEN_ELICITATION_FORMAT = "auctionlab.frozen_elicitation"
FROZEN_ELICITATION_VERSION = 1


def _model_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return _sha256_bytes(resolved.read_bytes())


def _bundle_list(bundle: Bundle) -> list[str]:
    return sorted(str(item) for item in bundle)


def _valuations_fingerprint_payload(scenario: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for bidder_id in scenario.instance.bidder_ids:
        rows = [
            {
                "bundle": _bundle_list(bundle),
                "value": float(value),
            }
            for bundle, value in scenario.instance.valuations[bidder_id].items()
        ]
        rows.sort(key=lambda row: (len(row["bundle"]), row["bundle"]))
        payload[bidder_id] = rows
    return payload


def scenario_fingerprint(scenario: Any) -> str:
    """Hash all instance-defining content without storing hidden values."""
    payload = {
        "name": scenario.name,
        "items": list(scenario.instance.items),
        "bidder_ids": list(scenario.instance.bidder_ids),
        "scenario_description": scenario.scenario_description,
        "item_descriptions": scenario.item_descriptions,
        "person_seed_hashes": {
            bidder_id: _sha256_bytes(
                scenario.person_seeds[bidder_id].encode("utf-8")
            )
            for bidder_id in scenario.instance.bidder_ids
        },
        "valuations": _valuations_fingerprint_payload(scenario),
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


@dataclass(frozen=True)
class ModelProvenance:
    provider: str | None
    model: str | None
    temperature: float | None = None


@dataclass
class BidderElicitationData:
    """Runtime-neutral initial elicitation state for one bidder."""

    nl_question: str
    nl_answer: str
    interest_map: LlmInterestMap | None
    candidate_bundles: list[Bundle]
    raw_pv_values: dict[Bundle, float] | None
    pv_candidate_stats: PvCandidateBundleStats | None = None
    pv_chunk_stats: PvChunkStats | None = None
    pv_degraded: bool = False
    interest_map_fallback_used: bool = False
    interest_map_quality_flags: tuple[str, ...] = ()
    interest_map_candidate_count_before_filter: int = 0
    interest_map_candidate_count_after_filter: int = 0
    interest_map_accuracy: dict[str, Any] | None = None
    person_answer_verification: dict[str, Any] | None = None
    person_answer_verification_history: list[dict[str, Any]] | None = None
    person_answer_attempt_count: int = 1
    person_first_answer_word_count: int | None = None
    person_final_answer_word_count: int | None = None
    pv_scale_quality_flags: tuple[str, ...] = ()
    pv_inferred_max_value: float | None = None
    pv_ground_truth_max_value: float | None = None
    pv_max_value_ratio: float | None = None
    pv_inferred_max_singleton: float | None = None
    pv_ground_truth_max_singleton: float | None = None
    pv_max_singleton_ratio: float | None = None


@dataclass(frozen=True)
class FrozenElicitationPack:
    format: str
    version: int
    scenario_name: str
    scenario_fingerprint: str
    scenario_spec_path: str | None
    scenario_spec_sha256: str | None
    scenario_seed: int | str | None
    selection_policy: str | None
    items: tuple[str, ...]
    bidder_ids: tuple[str, ...]
    environment_model: ModelProvenance
    person_model: ModelProvenance
    proxy_model: ModelProvenance
    generation_settings: dict[str, Any]
    bidders: dict[str, BidderElicitationData]
    generation_calls: tuple[dict[str, Any], ...]


def _entry_to_dict(entry: BidderElicitationData) -> dict[str, Any]:
    raw_pv = (
        None
        if entry.raw_pv_values is None
        else [
            {"bundle": _bundle_list(bundle), "value": float(value)}
            for bundle, value in sorted(
                entry.raw_pv_values.items(),
                key=lambda pair: (len(pair[0]), sorted(pair[0])),
            )
        ]
    )
    return {
        "nl_question": entry.nl_question,
        "nl_answer": entry.nl_answer,
        "interest_map": (
            None
            if entry.interest_map is None
            else _model_dict(entry.interest_map)
        ),
        "candidate_bundles": [
            _bundle_list(bundle) for bundle in entry.candidate_bundles
        ],
        "raw_provisional_values": raw_pv,
        "pv_candidate_stats": (
            None
            if entry.pv_candidate_stats is None
            else asdict(entry.pv_candidate_stats)
        ),
        "pv_chunk_stats": (
            None
            if entry.pv_chunk_stats is None
            else {
                **asdict(entry.pv_chunk_stats),
                "per_chunk_bundle_counts": list(
                    entry.pv_chunk_stats.per_chunk_bundle_counts
                ),
            }
        ),
        "diagnostics": {
            "pv_degraded": entry.pv_degraded,
            "interest_map_fallback_used": entry.interest_map_fallback_used,
            "interest_map_quality_flags": list(
                entry.interest_map_quality_flags
            ),
            "interest_map_candidate_count_before_filter": (
                entry.interest_map_candidate_count_before_filter
            ),
            "interest_map_candidate_count_after_filter": (
                entry.interest_map_candidate_count_after_filter
            ),
            "interest_map_accuracy": entry.interest_map_accuracy,
            "person_answer_verification": (
                entry.person_answer_verification
            ),
            "person_answer_verification_history": (
                entry.person_answer_verification_history
            ),
            "person_answer_attempt_count": (
                entry.person_answer_attempt_count
            ),
            "person_first_answer_word_count": (
                entry.person_first_answer_word_count
            ),
            "person_final_answer_word_count": (
                entry.person_final_answer_word_count
            ),
            "pv_scale_quality_flags": list(entry.pv_scale_quality_flags),
            "pv_inferred_max_value": entry.pv_inferred_max_value,
            "pv_ground_truth_max_value": entry.pv_ground_truth_max_value,
            "pv_max_value_ratio": entry.pv_max_value_ratio,
            "pv_inferred_max_singleton": entry.pv_inferred_max_singleton,
            "pv_ground_truth_max_singleton": (
                entry.pv_ground_truth_max_singleton
            ),
            "pv_max_singleton_ratio": entry.pv_max_singleton_ratio,
        },
    }


def pack_to_dict(pack: FrozenElicitationPack) -> dict[str, Any]:
    return {
        "format": pack.format,
        "version": pack.version,
        "scenario": {
            "name": pack.scenario_name,
            "fingerprint": pack.scenario_fingerprint,
            "scenario_spec_path": pack.scenario_spec_path,
            "scenario_spec_sha256": pack.scenario_spec_sha256,
            "scenario_seed": pack.scenario_seed,
            "selection_policy": pack.selection_policy,
            "items": list(pack.items),
            "bidder_ids": list(pack.bidder_ids),
        },
        "models": {
            "environment": asdict(pack.environment_model),
            "person": asdict(pack.person_model),
            "proxy": asdict(pack.proxy_model),
        },
        "generation_settings": pack.generation_settings,
        "bidders": {
            bidder_id: _entry_to_dict(entry)
            for bidder_id, entry in pack.bidders.items()
        },
        "generation_calls": list(pack.generation_calls),
    }


def _entry_from_dict(data: Mapping[str, Any]) -> BidderElicitationData:
    interest_map_data = data.get("interest_map")
    interest_map = (
        None
        if interest_map_data is None
        else LlmInterestMap(**interest_map_data)
    )
    pv_stats_data = data.get("pv_candidate_stats")
    chunk_stats_data = data.get("pv_chunk_stats")
    diagnostics = dict(data.get("diagnostics") or {})
    return BidderElicitationData(
        nl_question=str(data["nl_question"]),
        nl_answer=str(data["nl_answer"]),
        interest_map=interest_map,
        candidate_bundles=[
            frozenset(bundle) for bundle in data.get("candidate_bundles", [])
        ],
        raw_pv_values=(
            None
            if data.get("raw_provisional_values") is None
            else {
                frozenset(row["bundle"]): float(row["value"])
                for row in data["raw_provisional_values"]
            }
        ),
        pv_candidate_stats=(
            None
            if pv_stats_data is None
            else PvCandidateBundleStats(**pv_stats_data)
        ),
        pv_chunk_stats=(
            None
            if chunk_stats_data is None
            else PvChunkStats(
                **{
                    **chunk_stats_data,
                    "per_chunk_bundle_counts": tuple(
                        chunk_stats_data["per_chunk_bundle_counts"]
                    ),
                }
            )
        ),
        pv_degraded=bool(diagnostics.get("pv_degraded", False)),
        interest_map_fallback_used=bool(
            diagnostics.get("interest_map_fallback_used", False)
        ),
        interest_map_quality_flags=tuple(
            diagnostics.get("interest_map_quality_flags", [])
        ),
        interest_map_candidate_count_before_filter=int(
            diagnostics.get(
                "interest_map_candidate_count_before_filter", 0
            )
        ),
        interest_map_candidate_count_after_filter=int(
            diagnostics.get(
                "interest_map_candidate_count_after_filter", 0
            )
        ),
        interest_map_accuracy=diagnostics.get("interest_map_accuracy"),
        person_answer_verification=diagnostics.get(
            "person_answer_verification"
        ),
        person_answer_verification_history=diagnostics.get(
            "person_answer_verification_history"
        ),
        person_answer_attempt_count=int(
            diagnostics.get("person_answer_attempt_count", 1)
        ),
        person_first_answer_word_count=diagnostics.get(
            "person_first_answer_word_count"
        ),
        person_final_answer_word_count=diagnostics.get(
            "person_final_answer_word_count"
        ),
        pv_scale_quality_flags=tuple(
            diagnostics.get("pv_scale_quality_flags", [])
        ),
        pv_inferred_max_value=diagnostics.get("pv_inferred_max_value"),
        pv_ground_truth_max_value=diagnostics.get(
            "pv_ground_truth_max_value"
        ),
        pv_max_value_ratio=diagnostics.get("pv_max_value_ratio"),
        pv_inferred_max_singleton=diagnostics.get(
            "pv_inferred_max_singleton"
        ),
        pv_ground_truth_max_singleton=diagnostics.get(
            "pv_ground_truth_max_singleton"
        ),
        pv_max_singleton_ratio=diagnostics.get("pv_max_singleton_ratio"),
    )


def pack_from_dict(data: Mapping[str, Any]) -> FrozenElicitationPack:
    if data.get("format") != FROZEN_ELICITATION_FORMAT:
        raise ValueError(
            "not an auctionlab frozen elicitation pack: "
            f"format={data.get('format')!r}"
        )
    if data.get("version") != FROZEN_ELICITATION_VERSION:
        raise ValueError(
            "unsupported frozen elicitation version: "
            f"{data.get('version')!r}; expected "
            f"{FROZEN_ELICITATION_VERSION}"
        )
    scenario = data["scenario"]
    models = data["models"]
    bidders = {
        str(bidder_id): _entry_from_dict(entry)
        for bidder_id, entry in data["bidders"].items()
    }
    pack = FrozenElicitationPack(
        format=str(data["format"]),
        version=int(data["version"]),
        scenario_name=str(scenario["name"]),
        scenario_fingerprint=str(scenario["fingerprint"]),
        scenario_spec_path=scenario.get("scenario_spec_path"),
        scenario_spec_sha256=scenario.get("scenario_spec_sha256"),
        scenario_seed=scenario.get("scenario_seed"),
        selection_policy=scenario.get("selection_policy"),
        items=tuple(scenario["items"]),
        bidder_ids=tuple(scenario["bidder_ids"]),
        environment_model=ModelProvenance(**models["environment"]),
        person_model=ModelProvenance(**models["person"]),
        proxy_model=ModelProvenance(**models["proxy"]),
        generation_settings=dict(data.get("generation_settings") or {}),
        bidders=bidders,
        generation_calls=tuple(data.get("generation_calls") or ()),
    )
    _validate_pack_structure(pack)
    return pack


def _validate_pack_structure(pack: FrozenElicitationPack) -> None:
    item_set = set(pack.items)
    if len(item_set) != len(pack.items):
        raise ValueError("frozen pack contains duplicate item IDs")
    if set(pack.bidders) != set(pack.bidder_ids):
        raise ValueError(
            "frozen pack bidder records do not match scenario bidder_ids"
        )
    for bidder_id, entry in pack.bidders.items():
        if not entry.nl_question.strip() or not entry.nl_answer.strip():
            raise ValueError(
                f"{bidder_id}: opening question/answer must be non-empty"
            )
        candidates = entry.candidate_bundles
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"{bidder_id}: duplicate candidate bundles")
        unknown = set().union(*candidates) - item_set if candidates else set()
        if unknown:
            raise ValueError(
                f"{bidder_id}: candidate bundles contain unknown items "
                f"{sorted(unknown)}"
            )
        if entry.interest_map is not None:
            interested = set(entry.interest_map.interested_items)
            excluded = set(entry.interest_map.excluded_items)
            if interested & excluded:
                raise ValueError(
                    f"{bidder_id}: interest map overlaps interested/excluded"
                )
            classified = interested | excluded
            if classified != item_set:
                raise ValueError(
                    f"{bidder_id}: closed-world interest map must classify "
                    f"every item; missing={sorted(item_set-classified)}, "
                    f"unknown={sorted(classified-item_set)}"
                )
            for group in entry.interest_map.complementary_groups:
                if not set(group) <= interested:
                    raise ValueError(
                        f"{bidder_id}: group members must be interested"
                    )
            for group in entry.interest_map.substitute_groups:
                if not set(group.items) <= interested:
                    raise ValueError(
                        f"{bidder_id}: group members must be interested"
                    )
        if entry.raw_pv_values is not None:
            unknown_pv = set(entry.raw_pv_values) - set(candidates)
            if unknown_pv:
                raise ValueError(
                    f"{bidder_id}: PV contains non-candidate bundles"
                )
            if any(value < 0 for value in entry.raw_pv_values.values()):
                raise ValueError(f"{bidder_id}: PV values must be non-negative")


def validate_pack_for_scenario(
    pack: FrozenElicitationPack,
    scenario: Any,
    *,
    scenario_spec_path: str | Path | None = None,
) -> None:
    """Reject stale/mismatched packs before any mechanism is run."""
    if pack.scenario_name != scenario.name:
        raise ValueError(
            f"elicitation pack scenario {pack.scenario_name!r} does not "
            f"match selected scenario {scenario.name!r}"
        )
    if tuple(scenario.instance.items) != pack.items:
        raise ValueError("elicitation pack goods/order do not match scenario")
    if tuple(scenario.instance.bidder_ids) != pack.bidder_ids:
        raise ValueError(
            "elicitation pack bidders/order do not match scenario"
        )
    actual_fingerprint = scenario_fingerprint(scenario)
    if actual_fingerprint != pack.scenario_fingerprint:
        raise ValueError(
            "elicitation pack scenario fingerprint mismatch; regenerate the "
            "pack for this exact sampled auction"
        )
    actual_spec_hash = file_sha256(scenario_spec_path)
    if (
        pack.scenario_spec_sha256 is not None
        and actual_spec_hash is not None
        and pack.scenario_spec_sha256 != actual_spec_hash
    ):
        raise ValueError(
            "elicitation pack scenario-spec hash mismatch"
        )


def build_frozen_elicitation_pack(
    *,
    scenario: Any,
    entries: Mapping[str, BidderElicitationData],
    scenario_spec_path: str | Path | None,
    selection_policy: str | None,
    person_model: ModelProvenance,
    proxy_model: ModelProvenance,
    generation_settings: Mapping[str, Any],
    generation_calls: list[dict[str, Any]],
) -> FrozenElicitationPack:
    metadata = getattr(scenario, "metadata", {}) or {}
    pack = FrozenElicitationPack(
        format=FROZEN_ELICITATION_FORMAT,
        version=FROZEN_ELICITATION_VERSION,
        scenario_name=scenario.name,
        scenario_fingerprint=scenario_fingerprint(scenario),
        scenario_spec_path=(
            None if scenario_spec_path is None else str(scenario_spec_path)
        ),
        scenario_spec_sha256=file_sha256(scenario_spec_path),
        scenario_seed=metadata.get("scenario_seed"),
        selection_policy=selection_policy,
        items=tuple(scenario.instance.items),
        bidder_ids=tuple(scenario.instance.bidder_ids),
        environment_model=ModelProvenance(
            provider=metadata.get("environment_generation_provider"),
            model=metadata.get("environment_generation_model"),
            temperature=None,
        ),
        person_model=person_model,
        proxy_model=proxy_model,
        generation_settings=dict(generation_settings),
        bidders=dict(entries),
        generation_calls=tuple(generation_calls),
    )
    _validate_pack_structure(pack)
    validate_pack_for_scenario(
        pack,
        scenario,
        scenario_spec_path=scenario_spec_path,
    )
    return pack


def write_frozen_elicitation_pack(
    pack: FrozenElicitationPack,
    path: str | Path,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(pack_to_dict(pack), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_frozen_elicitation_pack(
    path: str | Path,
) -> FrozenElicitationPack:
    return pack_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def frozen_pack_sha256(pack: FrozenElicitationPack) -> str:
    """Return a stable content hash for an in-memory frozen pack."""
    return _sha256_bytes(
        _canonical_json(pack_to_dict(pack)).encode("utf-8")
    )


def project_frozen_pack_to_bidders(
    parent: FrozenElicitationPack,
    target_scenario: Any,
    *,
    scenario_spec_path: str | Path | None = None,
) -> FrozenElicitationPack:
    """Project a frozen pack to an ordered bidder subset with identical goods.

    Person disclosures and proxy inference depend on the catalogue, so goods
    projection is intentionally forbidden. Bidder-count projection is safe:
    bidders do not observe their rivals during initial elicitation.
    """
    target_items = tuple(target_scenario.instance.items)
    target_bidders = tuple(target_scenario.instance.bidder_ids)
    if target_items != parent.items:
        raise ValueError(
            "frozen-pack projection may only change bidders; target goods/order "
            "must exactly match the parent pack"
        )
    if not target_bidders:
        raise ValueError("projected frozen pack must contain at least one bidder")

    parent_positions = {
        bidder_id: idx for idx, bidder_id in enumerate(parent.bidder_ids)
    }
    unknown = [
        bidder_id
        for bidder_id in target_bidders
        if bidder_id not in parent_positions
    ]
    if unknown:
        raise ValueError(
            "target bidders are not contained in parent pack: "
            f"{unknown}"
        )
    positions = [parent_positions[bidder_id] for bidder_id in target_bidders]
    if positions != sorted(positions):
        raise ValueError(
            "target bidders must preserve their parent-pack order"
        )

    metadata = getattr(target_scenario, "metadata", {}) or {}
    target_seed = metadata.get("scenario_seed")
    if (
        parent.scenario_seed is not None
        and target_seed is not None
        and target_seed != parent.scenario_seed
    ):
        raise ValueError(
            "target scenario seed does not match parent frozen pack"
        )

    selected = set(target_bidders)
    projected_calls = tuple(
        deepcopy(call)
        for call in parent.generation_calls
        if call.get("bidder_id") in selected
        or call.get("bidder_id") in (None, "")
    )
    parent_hash = frozen_pack_sha256(parent)
    settings = deepcopy(parent.generation_settings)
    settings["projection"] = {
        "kind": "ordered_bidder_subset",
        "parent_pack_sha256": parent_hash,
        "parent_scenario_name": parent.scenario_name,
        "parent_scenario_fingerprint": parent.scenario_fingerprint,
        "parent_bidder_ids": list(parent.bidder_ids),
        "target_bidder_ids": list(target_bidders),
    }
    effective_spec_path = (
        parent.scenario_spec_path
        if scenario_spec_path is None
        else str(scenario_spec_path)
    )
    projected = FrozenElicitationPack(
        format=parent.format,
        version=parent.version,
        scenario_name=target_scenario.name,
        scenario_fingerprint=scenario_fingerprint(target_scenario),
        scenario_spec_path=effective_spec_path,
        scenario_spec_sha256=(
            parent.scenario_spec_sha256
            if scenario_spec_path is None
            else file_sha256(scenario_spec_path)
        ),
        scenario_seed=target_seed,
        selection_policy=parent.selection_policy,
        items=target_items,
        bidder_ids=target_bidders,
        environment_model=parent.environment_model,
        person_model=parent.person_model,
        proxy_model=parent.proxy_model,
        generation_settings=settings,
        bidders={
            bidder_id: deepcopy(parent.bidders[bidder_id])
            for bidder_id in target_bidders
        },
        generation_calls=projected_calls,
    )
    _validate_pack_structure(projected)
    validate_pack_for_scenario(
        projected,
        target_scenario,
        scenario_spec_path=effective_spec_path,
    )
    return projected
