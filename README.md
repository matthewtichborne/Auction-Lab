# AuctionLab project and codebase guide

This is the authoritative documentation for AuctionLab. It is written for both
researchers and coding agents: read it before changing the implementation,
running live model calls, or interpreting results.

AuctionLab studies whether a language-mediated proxy can make combinatorial
auction preference elicitation more scalable. A structured private environment
is converted into one short natural-language disclosure; a bidder proxy infers
an interest map and provisional bundle values; sealed and clock mechanisms then
select exact value queries through a shared elicitation-event interface.

The main experiment freezes all model-generated initial information before an
auction starts. Mechanism runs then replay the same opening answer, interest
map, candidate support, and raw provisional values, while subsequent value
queries are deterministic lookups against the private valuation table. This
keeps mechanism comparisons reproducible and requires no live LLM calls during
auction replay.

## Read this first

The current research specification is:

- one validated 16-good, 16-bidder PC-component master population;
- coverage-aware nested samples with 4 through 10 goods and/or bidders;
- five scenario seeds (`0` through `4`);
- a canonical catalogue-level opening question;
- a brief qualitative simulated-person answer with one total budget figure;
- an inferred, closed-world interest map;
- candidate bundles generated from that map;
- raw LLM provisional valuations over those candidates;
- frozen replay in iterative sealed XOR and ascending clock mechanisms;
- deterministic exact value queries selected by mechanism events;
- final VCG payments computed from each mechanism's reported XOR bids.

The principal model provenance in the prepared experiment is:

- environment generation: OpenAI `gpt-5.6-sol`;
- person answers and answer verification: OpenAI `gpt-5.6-sol`;
- proxy interest maps and provisional valuations: Gemini
  `gemini-3.6-flash`.

These are logical roles, not a requirement that the providers differ. The
environment and person roles may use the same model while retaining a strict
information boundary. Model identities and generation settings are recorded in
the environment, frozen packs, call logs, and run summaries.

Before modifying code, preserve these scientific invariants:

1. **The bidder proxy must not see private preference parameters or the full
   valuation table.** It sees the catalogue, opening question and answer, and
   its own inferred state.
2. **The environment is deterministic after generation.** LLM output specifies
   structured preferences; Python expands them into exact bundle values.
3. **Frozen packs store raw provisional values.** Calibration is applied only
   at replay time, so treatments share identical LLM output.
4. **Exact refinements are not calibrated.** A deterministic value query
   replaces an estimate with private ground truth for that bundle.
5. **Candidate filtering is bidder-specific.** Functional similarity alone is
   not enough to declare goods mutually exclusive.
6. **Failures should be visible.** Final experiments use fail-closed interest
   map and PV policies rather than silently switching elicitation regimes.
7. **Mechanism comparisons must share the same frozen pack.** Do not regenerate
   initial answers or PVs separately for sealed and clock arms.
8. **Reported-bid VCG revenue is not oracle revenue.** Even a 100%-efficient
   allocation can have different payments if allocation counterfactuals remain
   provisional.

The LaTeX paper, `revised_current_implementation_paper.tex`, is the research
write-up. It is not a substitute for this operational guide and may contain
provisional results that need rebuilding as experiments change.

## Research questions and contributions

The project asks:

> Can a structured language-to-bid pipeline reduce combinatorial
> communication while preserving allocation-relevant preferences, and how
> effectively can sealed and clock feedback refine the resulting provisional
> bids?

The implementation supports four closely related contributions:

1. **Structured environment generation.** A constrained LLM generation and
   repair pipeline produces heterogeneous bidder preference specifications.
   Deterministic validators enforce schema, population coverage, interest
   density, structural diversity, and economically non-trivial samples.
2. **Interest maps as a scalability device.** The proxy reconstructs relevant
   goods, exclusions, complements, and person-specific acquisition modes from
   the opening conversation. This reduces the bundle support that needs to be
   provisionally scored.
3. **Provisional valuations as an initial ranking method.** A proxy model
   assigns a value to each candidate bundle in bulk, creating an initial XOR
   bid before any exact bundle queries are made.
4. **A mechanism-feedback and modular-proxy framework.** Sealed and clock
   mechanisms emit typed events to a common proxy interface. Event policies
   decide which candidate bundle to verify exactly, making elicitation paths
   measurable and ablatable.

This is an experimental framework, not a strategy-proof production auction.
It does not implement the CECA loop from the motivating LLM-proxy paper.

## End-to-end architecture

```text
population design (fixed goods, roles, validation constraints)
        |
        v
environment-model structured bidder profiles
        |
        +--> deterministic complete private valuation tables
        |
        +--> brief qualitative disclosure contracts
                         |
                         v
               one canonical opening question
                         |
                         v
                  person-model answer
                         |
                   blind semantic verifier
                         |
                         v
proxy model: interest map --> candidate support --> raw provisional values
                         |
                         v
                 frozen elicitation pack
                    /             \
                   v               v
        iterative sealed XOR   ascending clock
                   \               /
                    mechanism events
                         |
                         v
          deterministic exact bundle-value lookups
                         |
                         v
       final reported allocation and reported-bid VCG payments
```

The environment model is not called during auction execution. The person,
proxy, and verifier models are called only while preparing frozen elicitation.
With `--person-query-mode deterministic`, replay and every refinement after
freezing are offline.

## Preference environment

### What is fixed and what is generated

`scenarios/pc_build_v2/population_design_16x16.json` fixes the public design:
16 PC-component goods, functional categories, 16 bidder roles across four
strata, generation constraints, and validation thresholds. The environment
model generates the private structured bidder profiles: base item values,
priority classifications, person-specific substitute groups and acquisition
modes, complement groups, budgets, saturation behaviour, and qualitative
identity text.

The published environment is
`scenarios/pc_build_v3/pc_build_population_16x16.json`. Its generation
manifest and validation report live beside it. Treat this file as the canonical
master population; do not regenerate it merely to run another seed.

`src/auctionlab/instances/structured.py` deterministically converts each
accepted profile into a complete valuation over all available bundles.
`src/auctionlab/instances/structured_spec.py` selects goods and bidders from
the master population and materialises an auction instance.

### Selection and seeds

The primary selection policy is `coverage_stratified`. For a fixed seed it
constructs nested orders, so (for example) the five-good sample contains the
four-good sample. Seeds alter which goods and bidders are selected from the
16x16 master population; they do not create new LLM-generated profiles.

The scalability grid contains 19 unique cells:

- goods path: 8 bidders and 4, 5, ..., 10 goods;
- bidders path: 8 goods and 4, 5, ..., 10 bidders;
- joint path: 4x4, 5x5, ..., 10x10;
- the shared 8x8 point is stored once as the anchor.

The grid is validated before generation or execution. Checks include positive
interest coverage, strata/category diversity, complements and substitutes,
interest-density bounds, multiple efficient winners, and winner-welfare-share
limits. Different seeds are also checked for distinct selected compositions.

### Simulated-person disclosure

The person does not receive a rich numeric seed. Python renders a short
qualitative disclosure from the private structured profile: identity,
priorities, fallbacks, alternatives, complements, exclusions, and exactly one
maximum-total-willingness-to-pay figure. It omits item prices, base values,
complement bonuses, substitute factors, and the valuation equation.

All bidders normally answer the same versioned, catalogue-specific opening
question. `--opening-question-policy proxy_generated` is available as a
robustness treatment, but the canonical question is the main specification.
A blind verifier extracts economic claims from the answer without receiving
the hidden answer key; deterministic code compares that extraction with the
private qualitative contract and requests repairs when necessary.

## Interest maps and candidate bundles

`src/auctionlab/llm/interest_map.py` derives an `LlmInterestMap` from the
catalogue and opening conversation. It records:

- `interested_items` and `excluded_items` as a closed-world partition;
- complementary groups backed by explicit extra joint-use evidence;
- substitute/alternative groups backed by quoted evidence;
- an acquisition mode for each alternative group:
  `choose_one`, `can_use_multiple`, or `unclear`;
- an optional disclosed budget hint.

The modes are bidder-specific. Two GPUs may be `choose_one` for a single-PC
buyer but `can_use_multiple` for a reseller or system integrator. Rankings,
fallback wording, or shared function alone must not produce exclusive
filtering. A dangerous false exclusivity metric records cases where an inferred
`choose_one` group would remove bundles that the hidden profile can use jointly.

`src/auctionlab/llm/bundles.py` generates candidate support from the inferred
map. Conceptually it:

1. forms non-empty subsets of the inferred interested goods;
2. removes bundles containing more than one member of an explicitly inferred
   `choose_one` group;
3. retains joint bundles for `can_use_multiple` and `unclear` groups;
4. prioritises complement groups, then singletons, then remaining bundles by
   size when a cap or ordered PV request is needed.

Candidate bundles are the proxy's hypothesis/support set, not value queries or
demand queries. The uncapped full-support baseline for an auction with `m`
goods and `n` bidders is:

```text
n * (2**m - 1)
```

Interest-map analysis separates reduction due to excluding uninterested goods
from the additional reduction due to `choose_one` structure. An omitted
candidate cannot normally be recovered by later mechanism feedback, so
candidate recall and false exclusivity must be reported alongside reduction.

## Provisional valuations

`src/auctionlab/llm/provisional_valuations.py` asks the proxy model to value
every candidate bundle using the catalogue, opening answer, and inferred
interest map. The compact response contains one numeric value per bundle in
the exact requested order. It never sees the hidden preference table.

The prompt contains no numerical scaling or discount. It asks for internally
coherent, conservative estimates, treats the disclosed budget as a cap rather
than a target, and accounts for inferred complement and acquisition structure.
Raw PVs initialise a cached XOR bid. Exact refinement queries later overwrite
individual atoms.

Large supports can be split with `--pv-chunk-size`. Chunking changes the
number of initial proxy calls, not the candidate support or the number of
mechanism refinement queries. Use `--pv-failure-policy raise` for experiments.
The `zero` policy is debugging-only and marks the bidder as degraded; it never
falls back to querying the full candidate support one bundle at a time.

### Optional calibration

Frozen packs always retain raw PVs. `src/auctionlab/llm/value_calibration.py`
can apply a replay-time sensitivity treatment:

```text
calibrated = scale * raw * size_gamma**max(0, bundle_size - size_threshold)
calibrated = min(disclosed_budget, calibrated)  # if budget_cap is true
```

Supported families are `none`, `uniform`, and `exponential`. Calibration never
touches deterministic exact-query answers. The main experimental specification
is uncalibrated unless a run explicitly supplies `--pv-calibration-config`.
The old `--epsilon`, `--discount-inferred`, and `--size-discount-*` flags are
deprecated compatibility options and should not be used in new experiments.

The out-of-domain calibration benchmark uses three small six-good,
three-bidder domains (travel packages, camera/video kits, and kitchen
appliances), with three independently generated environments per domain. It
selects one uniform PV scale by leave-one-instance-index-out VCG payment error
and validates downstream sealed/clock efficiency. A two-PC-environment
diagnostic also exists, but it is explicitly in-domain evidence and does not
publish an accepted runtime calibration.

## Frozen elicitation and caching

`src/auctionlab/llm/frozen_elicitation.py` defines a versioned frozen pack. A
pack contains the scenario fingerprint, selected goods and bidders, opening
question and answers, verifier audit, inferred interest maps, candidate
supports, raw PVs, chunk statistics, model provenance, settings, and call
audit records. It does **not** serialize the private valuation table.

Replay recomputes and checks the scenario fingerprint and selected composition,
so a pack cannot silently be used with a different environment. Calibration is
resolved after loading the raw values.

`src/auctionlab/llm/cache.py` provides a SQLite response cache. Cache keys
include provider, model, temperature, token limit, call type, parser/prompt
versions, identifiers, and a hash of the complete rendered prompt.

- `off`: never read or write;
- `read-write`: reuse a hit, otherwise call and save;
- `read-only`: require a hit and never contact a provider;
- `refresh`: always call and overwrite the matching entry.

For live preparation, use `read-write`. Each successful response is committed
to SQLite as it completes, so an interrupted master can reuse prior calls even
if its final pack was not yet written. The preparation runner also skips
already validated master packs and projects them into case packs without LLM
calls. For frozen replay, use `off`; no endpoint is needed in the first place.

Do not put API keys in committed commands, logs, or documentation. Use
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, or an
explicit shell variable. If a key has appeared in terminal output shared with
another party, revoke and replace it.

## Mechanisms and elicitation events

### Shared bid and solver layer

Both mechanisms use XOR bids (`src/auctionlab/bids/xor.py`), the OR-Tools WDP
solver (`src/auctionlab/solvers/wdp_ilp.py`), and VCG computation
(`src/auctionlab/payments/vcg.py`). Proxy protocols and typed events live in
`src/auctionlab/proxies/`.

The main proxy is `LlmInferredXorProxy`, adapted to the common mechanism
protocol by `LlmAuctionProxyAdapter` in `src/auctionlab/llm/proxies.py`.
Non-primary DNF-learning and hybrid proxies remain available for robustness
experiments.

Every exact bidder/bundle refinement is deduplicated. Query caps are safety
backstops (`0` means unlimited), not parameters to tune results.

### Iterative sealed XOR auction

`src/auctionlab/experiments/proxy_sealed_runner.py` repeatedly solves the WDP
over reported XOR bids and emits feedback. The useful primary rule is
`competitive`: verify provisional incumbent winners and a bounded frontier
obtained from winner-removal counterfactual WDPs. An optional shadow-price
loser challenger is available.

`--sealed-stopping-rule no_new_refinements` stops after a completed round
produces zero new exact queries; the configured round count remains a safety
maximum. Do not allow a round-zero allocation with no elicitation to count as
convergence.

### Ascending clock with supplementary XOR resolution

`src/auctionlab/experiments/proxy_clock_runner.py` posts item prices, obtains
top-k positive-surplus package demand, raises prices on contested goods, and
retains the proxy's complete known XOR support for final supplementary WDP and
VCG resolution. It can emit demand-change, near-tie, near-zero-surplus,
top-k-frontier, allocation-change, allocation-counterfactual, and terminal
audit events.

`top_k` affects clock demand and price dynamics, while the complete known bid
still reaches final resolution. Supplementary observations use the latest
reported value, so a corrected downward estimate is not masked by an older
overestimate.

### Event-policy factors

The control policy verifies current reported incumbents. Independent ablation
factors add or remove:

- incumbent verification;
- forced-allocation pivotal challengers;
- scarcity-avoiding fallback bundles;
- one non-recursive neighbour query after a large value correction;
- gating near-zero-surplus events to recently contested goods;
- a terminal closest-challenger regret audit.

`src/auctionlab/experiments/event_policy.py` contains deterministic selectors.
They operate only on existing candidate support, never inspect hidden values,
and never call an LLM. `scripts/run_event_ablation_8x8.py` evaluates each
factor and the combined treatment on the same five frozen 8x8 environments.

### Payments and VCG witnesses

Final payments are VCG payments over the mechanism's **reported** XOR bids.
The full-information arm computes a separate oracle VCG benchmark. True
welfare evaluates a reported allocation using hidden values; true surplus is
true welfare minus reported-bid revenue and can be negative if reported bids
overstate truth.

The code logs bidder-removal WDP witnesses used in VCG pricing. Refinement
records identify whether a queried bundle appears in the final allocation, a
reported VCG witness, or a full-information VCG witness. This distinction is
essential: allocation efficiency alone does not establish payment accuracy.

## Repository map

```text
src/auctionlab/
  auctions/       sealed XOR and ascending clock execution
  bids/           XOR bid representation
  experiments/    runners, event policies, diagnostics, CSV aggregation
  instances/      structured preferences, sampling, scenario materialisation
  llm/            clients, prompts, parsers, cache, person/proxy inference
  payments/       VCG payments and counterfactual witnesses
  proxies/        mechanism-independent proxy protocols and events
  solvers/        OR-Tools XOR winner determination

examples/
  run_live_llm_curated_batch.py    central single-case CLI (live or replay)

scripts/
  generate_pc_build_population.py          generate/repair master environment
  validate_pc_build_population.py          offline environment validation
  validate_single_bidder_elicitation.py    focused live path diagnostic
  generate_frozen_elicitation.py           thin preparation entry point
  prepare_scalability_elicitation_packs.py make/project catalogue masters
  run_scalability_experiment.py            execute the 19-cell grid
  plot_scalability_results.py              aggregate/plot mechanism results
  analyze_interest_maps.py                 offline support/accuracy analysis
  run_event_ablation_8x8.py                offline five-seed 8x8 ablation
  run_event_ablation_scalability.py        offline full-grid event ablation
  generate_pv_calibration_environments.py  create three calibration domains
  prepare_pv_calibration_benchmark.py       freeze calibration predictions
  fit_pv_calibration.py                     detailed offline fitting
  validate_pv_calibration.py                cross-validated mechanism checks
  diagnose_pc_pv_calibration.py             in-domain PC diagnostic only
  build_provisional_paper_results.py         build offline paper figures/tables

scenarios/
  pc_build_v2/population_design_16x16.json  fixed public population design
  pc_build_v3/pc_build_population_16x16.json canonical validated environment

tests/                                      unit and integration tests
outputs/                                    generated experiments (gitignored)
cache/                                      generated SQLite caches
revised_current_implementation_paper.tex    research manuscript
```

## Setup and tests

Python 3.11 or newer is recommended. The current local environment may use a
newer version.

```bash
python3 -m venv venv
./venv/bin/pip install -e .
./venv/bin/python -m pytest
```

Runtime dependencies are declared in `pyproject.toml`: Matplotlib, the OpenAI
Python client, OR-Tools, and Pydantic. Provider adapters use OpenAI-compatible
interfaces where appropriate.

Tests use mock LLM clients and should not make live API calls. Run focused
tests while iterating, then the full suite before handoff. Add regression tests
for prompt information boundaries, parsing/failure behavior, frozen-pack
compatibility, query deduplication, and output schema whenever those areas
change.

## Canonical workflows

Commands below assume the repository root and `./venv/bin/python`. Run a
command with `--help` before changing its options; the CLI is the final source
of truth for flags and defaults.

### 1. Validate the existing environment offline

```bash
./venv/bin/python scripts/validate_pc_build_population.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --design scenarios/pc_build_v2/population_design_16x16.json \
  --report outputs/validation/pc_build_population_16x16.validation.json \
  --sizes 4 5 6 7 8 9 10 \
  --fixed-size 8 \
  --validation-seeds 0 1 2 3 4
```

This makes no model calls. The canonical environment is already generated;
normal experimental work starts at frozen-pack preparation or replay.

### 2. Regenerate the environment only when intentionally redesigning it

```bash
./venv/bin/python scripts/generate_pc_build_population.py \
  --design scenarios/pc_build_v2/population_design_16x16.json \
  --output scenarios/pc_build_v3/pc_build_population_16x16.json \
  --raw-output-dir scenarios/pc_build_v3/raw_generation \
  --manifest scenarios/pc_build_v3/generation_manifest.json \
  --provider openai \
  --model gpt-5.6-sol \
  --api-key "$OPENAI_API_KEY" \
  --reasoning-effort medium \
  --max-tokens 24000 \
  --timeout 240 \
  --profiles-per-call 4 \
  --max-batch-repair-retries 3 \
  --max-estimated-cost-usd 9 \
  --validation-seeds 0 1 2 3 4 \
  --resume
```

The generator writes prompts and raw responses, validates each resumable
batch, and publishes the requested output only after all population/sample
checks pass. On failure, inspect the `.candidate.json` and `.validation.json`
files. Use `--regenerate-batches N ...` only for batches implicated by the
report.

### 3. Validate one live person/proxy path before a large preparation

```bash
./venv/bin/python scripts/validate_single_bidder_elicitation.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --bidder-id enthusiast_gamer \
  --num-goods 10 \
  --scenario-seed 0 \
  --selection-policy coverage_stratified \
  --person-provider openai \
  --person-model gpt-5.6-sol \
  --person-api-key "$OPENAI_API_KEY" \
  --proxy-provider gemini \
  --proxy-model gemini-3.6-flash \
  --proxy-api-key "$GEMINI_API_KEY" \
  --verifier-provider openai \
  --verifier-model gpt-5.6-sol \
  --verifier-api-key "$OPENAI_API_KEY" \
  --opening-question-policy canonical \
  --max-tokens 12000 \
  --verifier-max-tokens 2000 \
  --timeout 240 \
  --max-parse-retries 2 \
  --interest-map-failure-policy raise \
  --generate-provisional-valuations \
  --pv-chunk-size 64 \
  --pv-top-k 10 \
  --llm-cache-mode read-write \
  --llm-cache-path cache/single_bidder_validation.sqlite \
  --output outputs/validation/single_bidder_10_goods.json \
  --calls-log outputs/validation/single_bidder_10_goods_calls.jsonl
```

This is a diagnostic, not part of the final five-seed aggregate.

### 4. Prepare all frozen scalability packs

Initial elicitation depends on the goods catalogue but not the number of
rivals. The runner therefore generates seven master packs per seed (one for
each goods count) and projects nested bidder subsets into all 19 cases.

```bash
./venv/bin/python scripts/prepare_scalability_elicitation_packs.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --output-dir outputs/elicitation_packs/scalability \
  --sizes 4 5 6 7 8 9 10 \
  --fixed-size 8 \
  --seeds 0 1 2 3 4 \
  --selection-policy coverage_stratified \
  -- \
  --provider openai \
  --model gpt-5.6-sol \
  --person-provider openai \
  --person-model gpt-5.6-sol \
  --proxy-provider gemini \
  --proxy-model gemini-3.6-flash \
  --verifier-provider openai \
  --verifier-model gpt-5.6-sol \
  --pv-chunk-size 64
```

The preparation runner supplies conservative defaults for PV tokens, parsing
retries, timeout, and its shared read-write SQLite cache. Repeating the command
reuses validated masters and cached calls. `--project-only` rebuilds projected
case packs from completed masters without API calls. Use `--dry-run` to inspect
the exact plan.

### 5. Replay one frozen 8x8 pack through both mechanisms

```bash
./venv/bin/python examples/run_live_llm_curated_batch.py \
  --scenario pc_build \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --num-goods 8 \
  --num-bidders 8 \
  --scenario-seed 0 \
  --selection-policy coverage_stratified \
  --elicitation-pack outputs/elicitation_packs/scalability/seed_0/anchor_8x8/frozen_elicitation.json \
  --person-query-mode deterministic \
  --skip-baselines \
  --event-policy recommended \
  --sealed-elicitation-rounds 20 \
  --sealed-stopping-rule no_new_refinements \
  --elicited-clock \
  --top-k 3 \
  --max-rounds 50 \
  --price-step 50 \
  --llm-cache-mode off \
  --log-dir outputs/replay/seed_0/anchor_8x8
```

This command makes no LLM calls. The fixed `recommended` policy resolves to
incumbent and winner-removal verification, scarcity fallbacks for both
mechanisms, sealed-only large-correction follow-up, and the existing clock
counterfactual and terminal-stability audits. Use `--event-policy custom` for
granular ablations; it cannot be mixed with the fixed recommended policy.

### 6. Run the complete scalability suite

```bash
./venv/bin/python scripts/run_scalability_experiment.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --output-dir outputs/scalability/main \
  --sizes 4 5 6 7 8 9 10 \
  --fixed-size 8 \
  --seeds 0 1 2 3 4 \
  --selection-policy coverage_stratified \
  --elicitation-pack-dir outputs/elicitation_packs/scalability \
  -- \
  --person-query-mode deterministic \
  --skip-baselines \
  --event-policy recommended \
  --sealed-elicitation-rounds 20 \
  --sealed-stopping-rule no_new_refinements \
  --elicited-clock \
  --top-k 3 \
  --max-rounds 50 \
  --price-step 50 \
  --llm-cache-mode off
```

The runner writes a preflight report, skips cases that already contain a
complete summary, continues past failures unless `--fail-fast` is supplied,
and records `scalability_runs.csv`. Use `--dry-run` to inspect all commands and
`--rerun-complete` only when deliberately replacing completed results.

### 7. Plot mechanism scaling and analyse interest maps

When a suite contains both mechanisms, select each arm explicitly:

```bash
./venv/bin/python scripts/plot_scalability_results.py \
  --input-dir outputs/scalability/main \
  --output-dir outputs/scalability/main/analysis/sealed \
  --arm "proxy sealed"

./venv/bin/python scripts/plot_scalability_results.py \
  --input-dir outputs/scalability/main \
  --output-dir outputs/scalability/main/analysis/clock \
  --arm "proxy clock"

./venv/bin/python scripts/analyze_interest_maps.py \
  --input-dir outputs/elicitation_packs/scalability \
  --output-dir outputs/interest_map_analysis
```

All three commands are offline. Scalability plotting writes aggregate CSVs and
figures under the selected mechanism directories. Interest-map analysis writes bidder-,
case-, and across-seed CSVs plus plots. Master packs are excluded by default
to avoid double-counting bidders also present in projected cases.

### 8. Run the five-seed 8x8 event ablation

```bash
./venv/bin/python scripts/run_event_ablation_8x8.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --elicitation-pack-dir outputs/elicitation_packs/scalability \
  --output-dir outputs/event_ablation_8x8 \
  --seeds 0 1 2 3 4 \
  --sealed-rounds 20 \
  --clock-rounds 50 \
  --clock-top-k 3 \
  --price-step 50
```

The ablation collates all listed seeds into treatment-level summaries and
event-level outputs. It is offline. Completed treatment directories are
resumable; by default detailed auction stdout is captured in each treatment's
`ablation_runner.log`, with periodic heartbeat messages in the parent process.
Use `--verbose-runs` only when full streaming is useful.

To test whether those event effects generalise beyond 8x8, run the paired
ablation over all 95 frozen datasets (19 scalability cells × five seeds):

```bash
./venv/bin/python scripts/run_event_ablation_scalability.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --elicitation-pack-dir outputs/elicitation_packs/scalability \
  --output-dir outputs/event_ablation_scalability \
  --sizes 4 5 6 7 8 9 10 \
  --fixed-size 8 \
  --seeds 0 1 2 3 4 \
  --sealed-rounds 10 \
  --clock-rounds 20 \
  --clock-top-k 3 \
  --price-step 50 \
  --jobs 2
```

This is 760 deterministic treatment runs, each producing both mechanism arms.
It makes no LLM calls but is CPU-intensive; start with `--jobs 2` and increase
only if the machine remains responsive. The runner resumes completed
seed/case/treatment directories, writes each subprocess's detail to its own
`ablation_runner.log`, and produces pooled, series/size-specific, and paired
treatment-minus-control tables. `--aggregate-only` rebuilds those tables and
plots from whatever runs are complete without executing auctions.

### 9. Build provisional paper figures

First ensure each mechanism input root has been plotted and the combined
interest-map case table exists. Then run:

```bash
./venv/bin/python scripts/build_provisional_paper_results.py \
  --mechanism-input \
    outputs/scalability/seed0_results \
    outputs/scalability/seed1_results \
  --interest-map-cases outputs/interest_map_analysis/interest_map_case_metrics.csv \
  --output-dir outputs/paper/provisional
```

Replace the example mechanism roots with the actual per-seed completed roots.
This script is offline and expects one seed per mechanism input root.

## PV calibration workflow

Calibration is optional and should not be fitted on the PC results used for
the main claims.

Generate nine small structured environments. If the original three are
already present, they are validated and skipped, so exactly six new files are
generated:

```bash
./venv/bin/python scripts/generate_pv_calibration_environments.py \
  --output-dir outputs/pv_calibration/environments \
  --provider openai \
  --model gpt-5.6-sol \
  --api-key "$OPENAI_API_KEY" \
  --reasoning-effort medium \
  --max-tokens 12000 \
  --max-repair-retries 2 \
  --instances-per-domain 3
```

Prepare 27 frozen bidder paths (three bidders in each environment):

```bash
./venv/bin/python scripts/prepare_pv_calibration_benchmark.py \
  --domains all \
  --seeds 0 1 2 \
  --environment-dir outputs/pv_calibration/environments \
  --output-dir outputs/pv_calibration/benchmark \
  --person-provider openai \
  --person-model gpt-5.6-sol \
  --person-api-key "$OPENAI_API_KEY" \
  --proxy-provider gemini \
  --proxy-model gemini-3.6-flash \
  --proxy-api-key "$GEMINI_API_KEY" \
  --verifier-provider openai \
  --verifier-model gpt-5.6-sol \
  --verifier-api-key "$OPENAI_API_KEY" \
  --pv-max-tokens 12000 \
  --pv-chunk-size 0 \
  --max-parse-retries 2 \
  --llm-cache-mode read-write \
  --llm-cache-path cache/pv_calibration.sqlite
```

Select a uniform scale by held-out bidder-level VCG payment error and validate
downstream efficiency offline:

```bash
./venv/bin/python scripts/validate_pv_calibration.py \
  --benchmark-dir outputs/pv_calibration/benchmark \
  --output-dir outputs/pv_calibration/validation \
  --objective payment_error_over_optimum_welfare
```

The validator leaves one environment index out across all three domains and
selects the uniform scale on the other six environments. Acceptance requires
lower mean held-out payment error than scale 1, improvement in a majority of
held-out environments, and no greater than a two-percentage-point reduction
in mean initial, sealed, or clock efficiency. Worst-case changes remain in the
report but do not independently privilege the uncalibrated scale. It writes
`pv_calibration.json` only when these criteria pass.

## Final results pipeline

After accepting the calibration, justify the sealed policy on the five 8x8
anchors. The `primary` treatment set contains the staged construction and the
leave-one-component-out comparisons; it makes no LLM calls. These results
freeze `sealed-v1` (competitive counterfactuals, incumbent verification,
scarcity fallback, and sealed-only large-correction follow-up):

```bash
./venv/bin/python scripts/run_event_ablation_8x8.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --elicitation-pack-dir outputs/elicitation_packs/scalability \
  --output-dir outputs/final/event_policy_8x8 \
  --seeds 0 1 2 3 4 \
  --treatment-set primary \
  --pv-calibration-config outputs/pv_calibration/validation/pv_calibration.json \
  --sealed-rounds 40 \
  --clock-rounds 50 \
  --clock-top-k 3 \
  --price-step 50
```

The legacy clock columns in that study are diagnostic only. Select the new
clock-specific event framework on the nine out-of-domain calibration
environments. The four treatments add contested refinement and terminal VCG
witness auditing to a demand-switch/allocation-change core:

```bash
./venv/bin/python scripts/run_clock_event_framework_calibration.py \
  --benchmark-dir outputs/pv_calibration/benchmark \
  --pv-calibration-config outputs/pv_calibration/validation/pv_calibration.json \
  --output-dir outputs/final/clock_event_framework_calibration \
  --max-rounds 50 \
  --price-step 50 \
  --top-k 3 \
  --tie-threshold 100
```

For the query-efficient PC clock ablation, run the provisional-only control,
allocation-only audit, terminal-winner audit, terminal settlement audit, and
their lean combination.  This treatment set disables sealed elicitation and
does not impose a value-query budget; exact bidder/bundle queries are
deduplicated and arise only from enabled mechanism events. It also uses
`demand_revealed` supplementary support: only top-k positive-surplus bundles
observed along the clock price path enter the supplementary WDP and VCG
settlement, while the proxy's remaining candidate bid stays private:

```bash
./venv/bin/python scripts/run_event_ablation_8x8.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --elicitation-pack-dir outputs/elicitation_packs/scalability \
  --output-dir outputs/final/clock_lean_ablation_pc_8x8 \
  --seeds 0 1 2 3 4 \
  --treatment-set clock-lean \
  --pv-calibration-config outputs/pv_calibration/validation/pv_calibration.json \
  --clock-rounds 200 \
  --clock-top-k 3 \
  --price-step 50 \
  --fail-fast
```

To reproduce the original, pre-shared-policy clock elicitation framework,
use the `clock-native` factorial. It runs all eight combinations of the
clock-price events `near_zero_surplus`, `demand_changed`, and `near_tie`, with
historical full supplementary support but no allocation, counterfactual,
terminal, or sealed-derived elicitation events:

```bash
./venv/bin/python scripts/run_event_ablation_8x8.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --elicitation-pack-dir outputs/elicitation_packs/scalability \
  --output-dir outputs/final/clock_native_ablation_pc_8x8 \
  --seeds 0 1 2 3 4 \
  --treatment-set clock-native \
  --pv-calibration-config outputs/pv_calibration/validation/pv_calibration.json \
  --sealed-rounds 0 \
  --clock-rounds 200 \
  --clock-top-k 3 \
  --price-step 50 \
  --pivotal-gap-threshold 100 \
  --fail-fast
```

Before the definitive four-treatment ablation, tune dimensionless clock
parameters on those same design environments. Price increments and pivotal
gaps are fractions of median disclosed budget per available good, rather than
fixed currency amounts that would not transfer across auction domains:

```bash
./venv/bin/python scripts/tune_clock_parameters_calibration.py \
  --benchmark-dir outputs/pv_calibration/benchmark \
  --pv-calibration-config outputs/pv_calibration/validation/pv_calibration.json \
  --output-dir outputs/final/clock_parameter_tuning_calibration \
  --price-step-fractions 0.1 0.2 0.4 \
  --top-k-values 1 3 5 \
  --tie-threshold-fractions 0.2 0.4 0.8 \
  --max-rounds 50 \
  --efficiency-band 0.005
```

Then rerun `run_clock_event_framework_calibration.py` with the selected
`--top-k`, `--price-step-fraction`, and `--tie-threshold-fraction`. The runner
performs the environment-specific conversion using the same rule. The current
fixed-dollar invocation above is retained only as the preliminary sanity
check.

Then run the offline clock-parameter robustness grid. The deterministic
recommendation first retains configurations within 0.5 percentage points of
the best mean efficiency, then minimises welfare-normalised payment error,
exact value queries, and rounds:

```bash
./venv/bin/python scripts/run_clock_parameter_robustness_8x8.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --elicitation-pack-dir outputs/elicitation_packs/scalability \
  --output-dir outputs/final/clock_parameters_8x8 \
  --pv-calibration-config outputs/pv_calibration/validation/pv_calibration.json \
  --seeds 0 1 2 3 4 \
  --price-steps 25 50 100 \
  --top-k-values 1 3 5 \
  --tie-thresholds 50 100 200 \
  --max-rounds 50 \
  --event-policy final-v1
```

Insert the selected clock values when freezing the final specification. This
hashes the population, calibration, and all 95 frozen packs and refuses an
unaccepted calibration:

```bash
./venv/bin/python scripts/freeze_final_experiment.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --elicitation-pack-dir outputs/elicitation_packs/scalability \
  --pv-calibration-config outputs/pv_calibration/validation/pv_calibration.json \
  --output configs/final_experiment.json \
  --sizes 4 5 6 7 8 9 10 \
  --fixed-size 8 \
  --seeds 0 1 2 3 4 \
  --sealed-max-rounds 40 \
  --clock-max-rounds 50 \
  --clock-price-step SELECTED_PRICE_STEP \
  --clock-top-k SELECTED_TOP_K \
  --clock-tie-threshold SELECTED_TIE_THRESHOLD \
  --robustness-model openai:gpt-5-mini \
  --robustness-model anthropic:claude-haiku-4-5
```

The paired 95-case suite can subsequently be launched only from that frozen
configuration:

```bash
./venv/bin/python scripts/run_frozen_final_experiment.py \
  --spec configs/final_experiment.json \
  --output-dir outputs/final/scalability \
  --fail-fast
```

Build paired mechanism metrics and the four rule-selected example trajectories:

```bash
./venv/bin/python scripts/analyze_final_experiment.py \
  --spec configs/final_experiment.json \
  --input-dir outputs/final/scalability \
  --output-dir outputs/final/analysis
```

Finally, validate alternative proxy models on the same five opening-disclosure
sets. This stage regenerates only interest maps and raw PVs, then evaluates the
frozen mechanisms without model-specific calibration:

```bash
./venv/bin/python scripts/run_proxy_model_validation_8x8.py \
  --spec configs/final_experiment.json \
  --output-dir outputs/final/model_validation \
  --model openai:gpt-5-mini \
  --model anthropic:claude-haiku-4-5 \
  --llm-cache-mode read-write \
  --llm-cache-path cache/final_model_validation.sqlite
```

## Outputs and interpretation

The central single-case runner writes, as applicable:

- `curated_run_summary.csv`: one row per arm, including allocation, true
  welfare, efficiency, reported-bid revenue, true surplus, queries, tokens,
  model provenance, and calibration identity;
- `curated_person_disclosures.csv`: accepted opening answers and verifier
  audit;
- `curated_pv_candidate_bundle_stats.csv`: support and PV chunk statistics;
- `curated_sealed_proxy_elicited.csv` and
  `curated_proxy_sealed_trajectory.csv`: sealed final and round-level results;
- `curated_clock_proxy_elicited_top_K.csv`, clock rounds, bidder-rounds,
  coverage, and event-usefulness CSVs;
- `curated_refinement_records.csv`: every exact refinement with event reason,
  correction, allocation hit, and VCG-witness hits;
- `run_config.json`: resolved experimental configuration;
- `calls.jsonl`: LLM call audit when calls occur.

Interpret the metrics carefully:

- **True/full-information welfare** is computed from the hidden deterministic
  valuations.
- **Efficiency** is achieved true welfare divided by the full-information
  optimum.
- **Revenue** is VCG revenue from the arm's reported bids, not hidden values.
- **True surplus** is true welfare minus reported revenue and can be negative.
- **Value/demand queries** are post-initialisation person queries selected by
  mechanism events. In the main specification they are deterministic lookups,
  not model calls.
- **PV calls/chunks** and opening/interest-map calls are shared initial LLM
  work. They are not refinement VQs.
- **Token counts** distinguish actual initial LLM usage from optional estimated
  input cost assigned to deterministic person queries. Projected packs retain
  logical per-auction token attribution even though a catalogue master was paid
  for only once.
- **Event hit rate** may be zero for an allocation but non-zero for pricing;
  use VCG-witness columns before declaring an event economically useless.

## Development guide for coding agents

### Where to make a change

- Valuation semantics or disclosure rendering: `instances/structured.py` and
  `instances/scenario_spec.py`.
- Sampling/coverage: `instances/population_design.py` and
  `instances/structured_spec.py`.
- Prompts or parsing: `llm/prompts.py`, `llm/parsing.py`, and `llm/schemas.py`.
- Interest-map semantics/candidates: `llm/interest_map.py` and `llm/bundles.py`.
- PV generation/chunking: `llm/provisional_valuations.py`.
- Frozen schema/replay: `llm/frozen_elicitation.py`.
- Proxy state and event handling: `llm/proxies.py`, `proxies/base.py`, and
  `proxies/events.py`.
- Event selection: the sealed/clock experiment runners and
  `experiments/event_policy.py`.
- Allocation/payment semantics: `solvers/wdp_ilp.py`, `auctions/`, and
  `payments/vcg.py`.
- CSV schemas and plots: `experiments/llm_comparison.py`, `run_config.py`, and
  the analysis scripts.

### Change discipline

1. Inspect the current implementation and relevant tests before editing; the
   working tree may contain intentional research changes.
2. Keep hidden truth out of proxy prompts, event selectors, and runtime state.
3. Version prompt/parser behavior or frozen formats when a semantic change
   invalidates old cache entries or artifacts.
4. Preserve backward artifact reading only when it does not obscure the
   current experiment; this repository does not aim to maintain unused legacy
   mechanisms.
5. Make failure-policy and degraded-mode behavior explicit in output.
6. When adding a metric, thread it through detailed records, aggregate CSVs,
   and tests rather than calculating it only in a plot.
7. Validate offline first, then run one bidder, one 8x8 replay, and only then a
   full live preparation or scalability suite.
8. Never infer success from console completion alone: inspect manifests,
   per-case summaries, failure rows, and seed counts.

## Troubleshooting

### A live preparation appears stuck

PV generation can be slow for broad-interest bidders. Check the current
master's log and `calls.jsonl`; the preparation cache is written per successful
call. Interrupting with Ctrl-C is safe. Rerun the identical command to reuse
the cache and validated masters. Reduce `--pv-chunk-size` only if the provider
repeatedly truncates or times out; it increases call count but reduces each
response size.

### Interest-map or PV parsing fails repeatedly

Inspect the raw call in the case log. Confirm model-specific output limits,
`--interest-map-max-tokens`, `--pv-max-tokens`, and chunk size. Keep
`--interest-map-failure-policy raise` and `--pv-failure-policy raise` for final
runs. Parser retries evict a bad cached response before trying again.

### OpenAI reports model not found or organization verification required

API billing credit and model access are separate. Verify the organization in
the OpenAI platform, wait for access propagation, and test a model available to
that project. A 404 with an explicit verification message is not caused by the
AuctionLab parser or cache.

### A provider says the API key is invalid

Check that the correct role flag and environment variable are paired. For
example, `--verifier-provider gemini` uses `GEMINI_API_KEY`, while
`--person-provider anthropic` uses `ANTHROPIC_API_KEY`. Print only whether a
variable is set, never its full value. Revoke any key accidentally disclosed in
logs or chat.

### Allocation reaches 100% efficiency but revenue differs from oracle

This is possible and expected under partial elicitation. The chosen allocation
may be correct while bidder-removal counterfactual allocations used by VCG
still depend on inaccurate reported atoms. Inspect reported and full-information
VCG witness fields in the detailed CSVs.

### Plots omit cases or seeds

Inspect `scalability_runs.csv` and confirm every case contains
`curated_run_summary.csv`. Plotters aggregate what exists; they cannot infer a
missing or failed run. The paper-results builder additionally expects one seed
per mechanism input root.

## Current limitations

- The domain is one structured PC-component master population; seeds resample
  it rather than creating five independently generated populations.
- Interest-map reconstruction is aided by a validated, exhaustive disclosure
  contract. Treat its accuracy as reconstruction from controlled language, not
  arbitrary human conversation.
- Candidate filtering can permanently omit a useful bundle.
- Provisional values are point estimates and may be systematically biased.
- Deterministic value queries improve experimental control but abstract away
  human response noise and effort; optional token estimates are not actual API
  usage.
- VCG prices can be much less accurate than allocations because payments
  depend on counterfactual reported welfare.
- The completed diagnostic ablation is 8x8; the full-grid ablation runner is
  available but substantially more computationally expensive.
- Late reflection and LLM-based value/demand queries are robustness extensions,
  not part of the primary frozen deterministic design.
- The code evaluates proxy-mediated sealed and clock mechanisms; it does not
  claim strategic incentive guarantees for the iterative elicitation process.

When this guide and the code disagree, verify the CLI and tests, fix the guide
in the same change, and record any experiment-breaking semantic change in the
paper and output provenance.
