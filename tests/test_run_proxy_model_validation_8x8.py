from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_proxy_model_validation_8x8 import (
    auction_command,
    preparation_command,
)


def _spec():
    return {
        "scenario": {
            "path": "scenario.json",
            "selection_policy": "coverage_stratified",
        },
        "sealed": {"max_rounds": 40, "stopping_rule": "no_new_refinements"},
        "clock": {
            "max_rounds": 50,
            "top_k": 3,
            "price_step": 50.0,
            "tie_threshold": 100.0,
        },
        "event_policy": {"correction_threshold": 0.25},
    }


def test_preparation_reuses_disclosure_and_changes_only_proxy(tmp_path):
    source = tmp_path / "seed_2" / "anchor_8x8" / "frozen_elicitation.json"
    args = argparse.Namespace(
        interest_map_max_tokens=2000,
        max_tokens=12000,
        pv_chunk_size=64,
        max_parse_retries=2,
        timeout=240,
        llm_cache_mode="read-write",
        llm_cache_path="cache.sqlite",
    )
    command = preparation_command(
        spec=_spec(),
        source_pack=source,
        output_pack=tmp_path / "alt.json",
        log_dir=tmp_path / "logs",
        provider="anthropic",
        model="claude-test",
        args=args,
    )
    joined = " ".join(command)
    assert "--scenario-seed 2" in joined
    assert f"--disclosure-pack {source}" in joined
    assert "--proxy-provider anthropic --proxy-model claude-test" in joined
    assert "--person-query-mode deterministic" in joined


def test_auction_validation_is_raw_and_uses_frozen_policy(tmp_path):
    command = auction_command(
        spec=_spec(),
        pack=tmp_path / "alt.json",
        seed=1,
        run_dir=tmp_path / "auction",
    )
    joined = " ".join(command)
    assert "--event-policy recommended" in joined
    assert "--price-step 50.0" in joined
    assert "--pv-calibration-config" not in command
    assert "--llm-cache-mode off" in joined
