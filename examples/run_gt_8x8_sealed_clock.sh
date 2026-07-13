#!/usr/bin/env zsh
# Ground-truth 8x8 comparison: sealed (2-round and 3-round) + clock (k=1, k=2).
#
# PV uses the LLM (8 calls per run via gemini-3.1-flash-lite).
# All value/demand queries use ground-truth lookups — no further LLM calls.
#
# Usage:
#   chmod +x examples/run_gt_8x8_sealed_clock.sh
#   ./examples/run_gt_8x8_sealed_clock.sh

set -euo pipefail

BASE=(
  python examples/run_live_llm_curated_batch.py
  --provider gemini
  --model gemini-3.1-flash-lite
  --scenario pc_build
  --num-goods 8
  --num-bidders 8
  --scenario-seed 0
  --seed-type structured
  --ground-truth-queries
  --proxy-type llm
  --use-provisional-valuations
  --max-bundle-size 4
  --max-rounds 30
  --sealed-feedback-rule all_valued_bundles
  --elicited-clock
  --top-k 1 2
)

STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Run 1/2: 2-round sealed + clock k=1 + clock k=2 ==="
"${BASE[@]}" \
  --sealed-elicitation-rounds 2 \
  --log-dir "logs/gt_8x8_sealed2_${STAMP}"

echo ""
echo "=== Run 2/2: 3-round sealed + clock k=1 + clock k=2 ==="
"${BASE[@]}" \
  --sealed-elicitation-rounds 3 \
  --log-dir "logs/gt_8x8_sealed3_${STAMP}"

echo ""
echo "Done. Results in:"
echo "  logs/gt_8x8_sealed2_${STAMP}/"
echo "  logs/gt_8x8_sealed3_${STAMP}/"
