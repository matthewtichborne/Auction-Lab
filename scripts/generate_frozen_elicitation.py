#!/usr/bin/env python3
"""Generate one validated frozen initial-elicitation pack.

This is a preparation-only wrapper around the canonical live runner. It
accepts the same scenario, role-model, parsing, cache, and PV arguments, plus
``--output`` as a concise alias for ``--write-elicitation-pack``. Auction
mechanisms are always skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from examples.run_live_llm_curated_batch import main


def _normalise_args(argv: list[str]) -> list[str]:
    args = list(argv)
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 >= len(args):
            raise SystemExit("--output requires a path")
        if "--write-elicitation-pack" in args:
            raise SystemExit(
                "use either --output or --write-elicitation-pack, not both"
            )
        args[idx] = "--write-elicitation-pack"
    if "--write-elicitation-pack" not in args and "--help" not in args:
        raise SystemExit(
            "generate_frozen_elicitation.py requires --output PATH"
        )
    if "--prepare-elicitation-only" not in args:
        args.append("--prepare-elicitation-only")
    if "--skip-baselines" not in args:
        args.append("--skip-baselines")
    return args


if __name__ == "__main__":
    sys.argv[1:] = _normalise_args(sys.argv[1:])
    main()
