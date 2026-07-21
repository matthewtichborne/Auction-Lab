# Proxy architecture: the paper's NL proxy family on AuctionLab's mechanisms

This document describes the current implementation of the proxy-mediated
elicitation architecture, including the natural-language proxy family
(ωnvd, ωvd1/ωvd2, ωxor, ωh) adapted from the paper's Section 4 proxy designs.

**Out of scope:** the paper's CECA auction loop is not implemented.
AuctionLab's own mechanisms — sealed XOR VCG (`auctions/sealed_vcg.py`) and
ascending clock with supplementary VCG (`auctions/clock.py`) — keep their
original allocation/payment logic (winner determination, VCG payments).
Mostly the *proxy* layer (what stands between a simulated bidder and the
mechanism) was extended, but the clock mechanism itself also received
correctness fixes to how it accumulates and finalizes proxy-reported
information — see "Bid integrity guarantees" below.

## Overview

```
LlmPersonSimulator  <-- ground truth preferences, answers Q_V / Q_D / Q_N
        |
        v
   one or more "ω" proxies (this document)
        |
        v
ClockAuctionProxy / SealedAuctionProxy protocols (proxies/base.py)
        |
        v
proxy_clock_runner.py / proxy_sealed_runner.py
        |
        v
auctions/clock.py / auctions/sealed_vcg.py  (allocation/payment logic unchanged)
```

A proxy maintains a candidate XOR bid (`bids/xor.py`) on behalf of one
bidder and answers `current_bid()` / `submit_bid()` / `demand_at_prices(...)`.
When a mechanism has a useful signal (a near-zero-surplus clock round, a
changed primary demand, a near-tie runner-up, or a provisional sealed
allocation), it sends the proxy an `ElicitationEvent`
(`proxies/base.py`), and the proxy's `refine(event)` decides whether and how
to update its bid.

All proxy classes implement the `@runtime_checkable` protocols in
`proxies/base.py`:

- `AuctionProxy` — `current_bid()`, `refine(event)`, `stats()`,
  `refinement_records()`.
- `ClockAuctionProxy` — adds `demand_at_prices(prices, round_idx, top_k)`.
- `SealedAuctionProxy` — adds `submit_bid()`,
  `receive_provisional_feedback(event)`.

Because the runners (`experiments/proxy_clock_runner.py`,
`experiments/proxy_sealed_runner.py`) only depend on these protocols, every
proxy below is a drop-in replacement — no mechanism code changes.

## The `LlmPersonSimulator` (ground truth)

`llm/person_simulator.py` simulates a bidder by answering questions about a
fixed (hidden) valuation, via an LLM. It supports three query types:

- `value_query(bundle, anchor_values=None, transcript_context=None) -> float`
  — "what is bundle `B` worth to you?" (Q_V). `anchor_values` lets the prompt
  cite already-known singleton values as calibration anchors.
  `transcript_context` (from `llm/prompts.format_transcript_context`) lets a
  prior NL Q&A exchange be folded into the prompt.
- `demand_query(bundle, prices) -> LlmDemandQueryResponse` — "are you
  satisfied with bundle `B` at prices `φ`? if not, what would you prefer?"
  (Q_D). Returns `{satisfied: bool, preferred_bundle: list[str] | None}`.
- `answer_question(question: str) -> str` — answers an arbitrary open-ended
  natural-language question (Q_N) about the bidder's preferences.

All three are logged via `_log_attempt` (prompt type
`"value_query"` / `"demand_query"` / `"nl_question"`) and retried up to
`max_parse_retries` on malformed JSON.

## ωxor — `DnfLearningProxy` (`proxies/dnf_learning.py`)

The non-LLM, "proper learning" baseline (Algorithms 1–2 in the paper). It
starts with **no information** about the bidder's preferences and learns an
exact XOR bid purely from `demand_query` (Q_D) and `value_query` (Q_V) calls.

- **Initial bid**: two atoms, `{∅: 0}` and `{grand bundle: 0}`. The grand-
  bundle atom is an exploration probe (price-zero initial demand from
  Algorithm 1) — it gives the clock "near-tie" trigger and the sealed
  "allocated bundle" feedback something non-empty to challenge on round 1.
- **`demand_at_prices(prices, top_k)`**: ranks the current bid's atoms by
  surplus (`demand_rank_key`, same as the LLM proxy's clock demand) and
  returns the top positive-surplus atom plus `top_k` supplementary atoms.
- **`refine(event)`** (Algorithm 1's body, one event per call):
  1. If `event.prices` is set (a clock event), issue `demand_query(bundle,
     prices)`. If the person is satisfied, no-op.
  2. Otherwise (not satisfied, or a sealed event with no prices), run
     `_learn_atomic_bundle(bundle)` (Algorithm 2): `value_query(bundle)` for
     `ν`, then for each item `i` (ascending), `value_query(bundle \ {i})`;
     if unchanged, permanently drop `i`. The resulting minimal `(b', ν)` is
     upserted into the candidate XOR bid.
- Every `value_query`/`demand_query` increments `ProxyStats` counters and
  appends a `RefinementRecord`.

Factory: `experiments/llm_runner.make_dnf_learning_proxies_for_instance(...)`
— one `DnfLearningProxy` per bidder, sharing the same signature shape as
`make_llm_proxies_for_instance` (minus `epsilon`/`ask_initial_question`,
which don't apply to a non-LLM proxy).

## ωnvd — initial NL question (`LlmInferredXorProxy.ask_initial_question`)

`llm/proxies.py`'s `LlmInferredXorProxy` gained:

- `nl_transcript: list[tuple[str, str]]` — accumulated (question, answer)
  pairs.
- `ask_initial_question(question_client=None)`: builds a prompt via
  `build_initial_proxy_question_prompt`, asks the LLM (or `question_client`)
  what open-ended question to pose, then calls
  `person.answer_question(question)` and records the pair. **No-op if
  called more than once** — matches the paper's "very first time it is run"
  semantics. `build_initial_proxy_question_prompt` deliberately does not take
  the person's preference seed: the proxy crafts its question from the
  scenario and item catalog alone, not from private knowledge of the
  person's actual values. (See "Bid integrity guarantees" below.)
- Every subsequent `value_query` (in `_infer_reported_values` and
  `_revalue_and_upsert_atom`) passes `transcript_context =
  format_transcript_context(self.nl_transcript)`, so the initial Q&A informs
  all later value inference, including refinements.

Enabled via `make_llm_proxies_for_instance(..., ask_initial_question=True)`,
which calls `proxy.ask_initial_question()` immediately after construction
for each bidder.

## Interest maps + provisional valuation (PV) — bulk bundle estimation

The default elicitation path above issues one `value_query` per candidate
bundle. For scenarios with many candidate bundles this is expensive, so
`LlmInferredXorProxy` supports a faster bulk path: derive a structured
interest map from the initial NL answer, generate a candidate set from that
map, then estimate every candidate bundle's value in a single LLM call.

- `build_interest_map(client=None)` (`llm/interest_map.py`'s
  `derive_interest_map`): parses the most recent NL question/answer pair
  into an `LlmInterestMap` — `interested_items`, `excluded_items`,
  `complementary_groups`, `substitute_groups`, an optional `budget_hint`.
  Requires `ask_initial_question()` to have run first. Grounded only in the
  NL exchange — never sees the person's preference seed.
- `candidate_bundles_from_interest_map(all_items, max_candidate_bundles=None)`:
  generates a priority-ordered candidate bundle list from the interest map
  (complementary groups first, then singletons, then remaining bundles by
  ascending size). `max_candidate_bundles` caps the list; `None` is
  uncapped. A bidder whose interest map covers most/all items (e.g. a
  reseller persona with no narrow preference) can produce a combinatorially
  large candidate set — capping trades completeness for per-bundle estimate
  precision in one bulk call.
- `build_provisional_valuations(candidate_bundles, client=None,
  interest_map=None, discount_inferred=True)`
  (`llm/provisional_valuations.py`'s `generate_provisional_valuations`):
  one LLM call estimating every candidate bundle's value at once, then
  `set_provisional_bid(...)` pre-populates the cached XOR bid from the
  result (no further `value_query` needed until a refinement event fires).
  Like the interest map, this prompt is grounded only in the NL
  question/answer (+ the interest map's structure) — never the seed.
- `replay_elicitation(nl_question, nl_answer, interest_map=None,
  provisional_raw_values=None, discount_inferred=True)`: replays a
  previously-computed NL Q&A / interest map / PV result into a *fresh*
  proxy instance with **zero LLM calls**. `examples/run_live_llm_curated_batch.py`'s
  `compute_elicitation_cache(...)` runs the NL+interest-map(+PV) phase once
  per scenario, then each mechanism arm (sealed, each clock `top_k`) gets
  its own fresh proxy seeded via `replay_elicitation` instead of repeating
  the NL/interest-map/PV calls per arm — roughly halving elicitation cost
  when comparing multiple mechanisms in one invocation, while keeping each
  arm's mutable refinement state independent.

Enabled via `--use-provisional-valuations` (implies `--use-interest-map` and
`--ask-initial-question`) or `--use-interest-map` alone (candidate bundles
from the interest map, but still one `value_query` per bundle rather than a
single bulk PV call).

## ωvd1/ωvd2 — demand-query refinement (`refine_via_demand_query`)

`LlmInferredXorProxy.refine_via_demand_query(bundle, prices, reason,
use_anchor_values=True) -> (Bundle | None, float | None)`:

1. `demand_query(bundle, prices)`.
2. If satisfied → no-op, returns `(None, None)`.
3. If not satisfied and the person names a different `preferred_bundle` →
   value-query and upsert *that* bundle's atom (ωvd1's "action 3" — refine
   the bundle the person actually wants). Returns `(preferred_bundle,
   value)`.
4. If not satisfied with no useful alternative → falls back to
   `refine_bundle_value(bundle, ...)` (plain value query on `bundle`).
   Returns `(bundle, value)`.

This is selected via `refinement_strategy: Literal["value_query",
"demand_query"]`, threaded through:

- `LlmInferredXorProxy.clock_demand_with_refinement(...,
  refinement_strategy=...)` — the static (non-proxy-runner) clock path.
- `LlmAuctionProxyAdapter.refine(event)` — the proxy-runner path; uses
  demand-query refinement only when `event.prices` is present (clock events
  carry prices; sealed events don't, so they always fall back to
  `refine_bundle_value`).
- `experiments/llm_comparison.run_clock_llm_comparison(...,
  refinement_strategy=...)`.

Default remains `"value_query"`, preserving prior behavior exactly.

## `LlmAuctionProxyAdapter` (`llm/proxies.py`)

Wraps `LlmInferredXorProxy` to satisfy `AuctionProxy` /
`ClockAuctionProxy` / `SealedAuctionProxy` without changing its inference
behavior. Fields: `candidate_bundles`, `discount_inferred`,
`use_anchor_values`, `refinement_strategy`. `refine(event)` dispatches to
`refine_via_demand_query` or `refine_bundle_value` per the strategy above,
records `ProxyStats`/`RefinementRecord`s.

## ωh — `HybridProxy` (`proxies/hybrid.py`)

Combines an `LlmAuctionProxyAdapter` (ωnvd/ωvd-style inference) and a
`DnfLearningProxy` (ωxor-style exact learning), switching between them based
on a refinement-call counter:

- **Fields**: `bidder_id`, `llm_proxy: LlmAuctionProxyAdapter`, `dnf_proxy:
  DnfLearningProxy`, `alpha: int` (must be ≥ 1), `delta: float` (must be in
  `(0, 1)`).
- **`current_bid()`**: the DNF proxy's atoms (exact, always included) plus
  any LLM-proxy atoms for bundles *not yet* in the DNF bid, each scaled by
  an internal `_gamma_factor`.
- **`refine(event)`**:
  - While `_calls < alpha`: delegate to `llm_proxy.refine(event)` (ωnvd/ωvd
    path — early refinements rely on NL inference).
  - From the `alpha`-th call onward: delegate to `dnf_proxy.refine(event)`
    (ωxor exact learning). On the first post-switch call, `_gamma_factor`
    resets to `1.0`; on every call after that, `_gamma_factor *= delta`
    (`γ' = δγ`, decaying trust in the still-unconfirmed LLM-inferred atoms as
    exact learning proceeds).
  - `_calls` increments either way.
- **`demand_at_prices`/`submit_bid`**: derived from `current_bid()` via the
  same surplus-ranking (`demand_rank_key`) as `DnfLearningProxy`.
- **`stats()`/`refinement_records()`**: sum/concatenate both delegates'.

Factory: `experiments/llm_runner.make_hybrid_proxies_for_instance(...,
alpha=10, delta=0.95)` — builds one shared `LlmPersonSimulator` per bidder,
used by both an `LlmAuctionProxyAdapter`-wrapped `LlmInferredXorProxy` and a
`DnfLearningProxy`, combined into a `HybridProxy`.

## Bid integrity guarantees

These hold for every proxy-mediated arm and were each fixed after an
experiment-output anomaly revealed the system was not honestly testing what
it appeared to be testing.

**The proxy never sees the person's preference seed.** Only
`LlmPersonSimulator`'s own methods (`value_query`, `demand_query`,
`answer_question`) read `person_seed` — that's the simulated person
answering from their own knowledge, which is legitimate. Nothing on the
proxy side does: `build_initial_proxy_question_prompt` (the opening
question) and `build_provisional_valuation_prompt` (the bulk PV call) both
explicitly exclude it. Before this was enforced, the PV call quoted seed
text containing literal dollar figures back as its "estimate" for explicit
scenarios, and the opening question was generated with full knowledge of
the answer it was nominally trying to discover. Regression tests
(`test_llm_prompts.py`) assert `TypeError` if `person_seed` is passed to
either builder, so this can't silently regress.

**Bundle values are kept internally consistent.**
`enforce_atom_monotonicity(atoms)` (`llm/proxies.py`) clamps any bundle's
value down to at most the value of any of its cached supersets for the same
bidder, run after every write to a proxy's cached bid (`set_provisional_bid`,
both refinement-write paths, and the legacy direct-query `infer_xor_bid`).
PV and independently-asked refinement queries can otherwise disagree with
each other — e.g. a 3-item bundle refined to a different (higher) value than
an already-confirmed 4-item superset containing it — which a WDP solver can
trivially exploit by "shrinking" a bundle to gain reported value. The clamp
only ever lowers an over-claiming subset, never raises a value, so it can't
inflate a bidder's reported surplus.

**The clock's final WDP+VCG resolution sees everything elicited, not just
this round's top choice.** `DemandResponse.supplementary_atoms` (every
`demand_at_prices`/`clock_demand_from_cached_bid` implementation) always
returns the proxy's entire currently-known bid, not capped by `top_k`.
Before this, `top_k` capped supplementary atoms to the `top_k` best-surplus
bundles *that round*, so a bundle that was correctly valued but never
happened to be the single best choice at any round's prices was silently
dropped from the final resolution even though the proxy knew the right
answer the whole time.

**Supplementary bids keep the latest observation, not the historical
maximum.** `record_supplementary_bids` (`auctions/clock.py`) overwrites a
bundle's stored value with each round's latest observation. It used to keep
whichever value had been highest across all rounds — reasonable when
information only ever firms up, wrong once PV-seeded bids can be
overestimates that refinement later corrects downward: a stale, inflated
pre-refinement value would silently survive in the welfare-determining
bid even after being corrected.

**`top_k` is a real auction-dynamics lever again, not just a result label.**
`DemandResponse.primary_bundles` (new field, alongside the existing
single-bundle `primary_bundle`) carries the bidder's `top_k` best-surplus
bundles; `run_ascending_clock_with_supplementary` uses it (falling back to
`[primary_bundle]` for an oracle that doesn't set it) to compute excess
demand. Previously `top_k` only ever sliced `supplementary_atoms` — once
that was uncapped for the integrity fix above, `top_k` had no remaining
behavioral effect at all (a `--top-k 1 2 3` sweep produced identical
results under every value). Now it controls how many packages a bidder
reports as primary demand each round, which is a genuine convergence-speed/
cost lever — though, with the supplementary-completeness fix above, no
longer the determinant of final welfare correctness.

## Wiring summary (`experiments/llm_runner.py`)

| Factory | Proxy type | ω |
|---|---|---|
| `make_llm_proxies_for_instance(..., ask_initial_question=...)` | `LlmInferredXorProxy` (wrap in `LlmAuctionProxyAdapter` for elicited runs) | ωnvd / ωvd1 / ωvd2 |
| `make_dnf_learning_proxies_for_instance(...)` | `DnfLearningProxy` | ωxor |
| `make_hybrid_proxies_for_instance(..., alpha=, delta=)` | `HybridProxy` | ωh |

`experiments/llm_comparison.py`'s `proxy_sealed_result_to_row` /
`proxy_clock_result_to_row` are generic over any `MechanismResult` produced
by `run_proxy_sealed_vcg_experiment` / `run_proxy_clock_experiment`, so all
three proxy types serialize to CSV rows without additional code.

## CLI: `examples/run_live_llm_curated_batch.py`

Flags exposed for the above, only effective for elicited proxy-mediated runs
(`--elicited-clock` and/or `--sealed-elicitation-rounds N`).
`--max-refinement-queries-per-bidder` caps per-bidder refinement spend;
it defaults to `0`, which means **unlimited**, not zero allowed — there is
no minimum-budget requirement to enable elicitation.
`--max-total-refinement-queries` is the same idea but summed across all
bidders (also `0`/unlimited by default). Both are safety backstops against
runaway query volume, not tuning targets — refinement count is meant to be
an outcome of the elicitation events and mechanism; see
`docs/parameter_tuning_methodology.md`. Value/demand queries are already
deduplicated per bidder/bundle (a bundle is refined at most once) regardless
of either cap.

- `--ask-initial-question` — ωnvd.
- `--use-interest-map` — derive candidate bundles from the NL interest map
  instead of enumerating up to `--max-bundle-size`; implies
  `--ask-initial-question`.
- `--use-provisional-valuations` — bulk PV call instead of one `value_query`
  per candidate bundle; implies `--use-interest-map`.
- `--pv-max-tokens` — token budget for the bulk PV call (higher than
  `--max-tokens` since the response contains one entry per bundle).
- `--max-candidate-bundles` — caps candidate bundles after interest-map
  filtering; `None`/omitted is uncapped.
- `--refinement-strategy {value_query,demand_query}` — ωvd1/ωvd2.
- `--proxy-type {llm,dnf,hybrid}` — selects `LlmAuctionProxyAdapter` (default),
  `DnfLearningProxy` (ωxor), or `HybridProxy` (ωh).
- `--hybrid-alpha`, `--hybrid-delta` — ωh's `alpha`/`delta` (only used when
  `--proxy-type hybrid`).
- `--skip-baselines` — skip the non-proxy-mediated direct-vq baseline arms
  (every candidate bundle queried individually, no NL inference at all) so
  only the proxy-mediated arms run.

`--proxy-type dnf` ignores `--ask-initial-question` and
`--refinement-strategy` (the DNF proxy never does NL inference or value-query
refinement — only demand/value queries per Algorithm 1–2). `--proxy-type
hybrid` applies both flags to its LLM delegate.

Example (ωh, demand-query refinement):

```bash
python examples/run_live_llm_curated_batch.py --provider ollama --model llama3.1:8b \
  --elicited-clock --max-refinement-queries-per-bidder 1 \
  --proxy-type hybrid --hybrid-alpha 10 --hybrid-delta 0.95 \
  --ask-initial-question --refinement-strategy demand_query
```

See the README for the recommended PV + interest-map example.

## Tests

- `tests/test_dnf_learning_proxy.py` — ωxor: initial bid shape, Algorithm 2
  shrinking, protocol conformance via `run_proxy_clock_experiment` /
  `run_proxy_sealed_vcg_experiment`.
- `tests/test_hybrid_proxy.py` — ωh: invalid `alpha`/`delta`, `current_bid`
  combination, early-vs-late refinement routing, γ decay sequence.
- `tests/test_llm_proxy.py`, `tests/test_llm_proxy_adapter.py`,
  `tests/test_llm_clock_runner.py` — ωnvd transcript wiring, ωvd1/ωvd2
  demand-query refinement branches, monotonicity clamping
  (`test_set_provisional_bid_clamps_subset_above_superset` and friends).
- `tests/test_llm_runner.py` — factory wiring
  (`make_dnf_learning_proxies_for_instance`,
  `make_hybrid_proxies_for_instance`).
- `tests/test_llm_prompts.py` — regression guards that
  `build_initial_proxy_question_prompt` / `build_provisional_valuation_prompt`
  reject a `person_seed` argument outright.
- `tests/test_clock.py` — `record_supplementary_bids` latest-wins semantics,
  `top_k` multi-package primary demand (`primary_bundles`).

Full suite: `pytest tests/` — all using `MockLlmClient`, no live LLM
required.
