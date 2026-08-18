# AuctionLab

AuctionLab is the reference implementation for a modular LLM-proxy framework
for combinatorial auctions. It studies how a bidder can provide one concise
natural-language disclosure while a proxy converts that disclosure into a
structured bid and responds to mechanism-generated elicitation events.

The retained empirical implementation covers two mechanisms:

- an iterative sealed XOR auction with reported-space VCG payments; and
- an ascending item-price clock with supplementary XOR bids and terminal
  revealed-witness verification.

Across 95 matched cases, the sealed and ascending-clock mechanisms recovered 94.8% and 92.8% of full-information welfare while requiring an average of 21.8 and 12.3 selective value queries per auction. The repository includes case-level results, seed-clustered uncertainty intervals, ablation studies and an 880-test offline suite.

The repository is intentionally scoped to the final PC-build experiment. 

## Final architecture

The information path is:

```text
generated hidden environment
        ↓
one validated person disclosure
        ↓
inferred interest map
        ↓
candidate bundle support
        ↓
provisional XOR values
        ↓
sealed or clock mechanism
        ↓
deterministic exact-value responses to elicitation events
```

Three model roles are separated during preparation:

- **Environment model:** generates hidden bidder preference specifications.
- **Person model:** produces the opening disclosure under a disclosure
  contract derived from the hidden specification.
- **Proxy model:** reconstructs the interest map and provisional valuations.

After the elicitation packs are frozen, auction execution requires no LLM
calls. Exact value queries are deterministic lookups into the hidden valuation
table. The empirical proxy policy is deterministic; the interface supports a
richer stateful proxy, but this project does not claim agentic behaviour.

## Repository map

Every module carries a docstring stating what it is for and any non-obvious
commitment it makes, so the fastest way to orient is to read the docstrings of
the directory you land in.

```text
src/auctionlab/
  auctions/        sealed and clock mechanism logic
  bids/            XOR bid representation
  experiments/     final policies, runners, metrics, and calibration
  instances/       generated-population schema and valuation primitives
  llm/             providers, caching, disclosures, interest maps, and PVs
  payments/        VCG payment calculation and witness diagnostics
  proxies/         common proxy protocol and elicitation events
  solvers/         XOR winner determination

tests/             880 offline tests; no credentials or network required
scenarios/         the generated 16x16 population and its provenance
outputs/           experiment artefacts (git-ignored apart from the
                   analytic package and the final manifest)
cache/             persistent LLM response cache used during preparation

scripts/
  generate_pc_build_population.py
  validate_pc_build_population.py
  generate_pv_calibration_environments.py
  prepare_pv_calibration_benchmark.py
  fit_pv_calibration.py
  validate_pv_calibration.py
  tune_clock_parameters_calibration.py
  generate_frozen_elicitation.py
  prepare_scalability_elicitation_packs.py
  run_event_ablation_8x8.py
  run_scalability_experiment.py
  freeze_final_experiment.py
  run_frozen_final_experiment.py
  analyze_interest_maps.py
  build_final_analytic_package.py
  build_paper_figures.py
  analyze_generated_environment.py

examples/
  run_live_llm_curated_batch.py   shared single-case execution engine
```

## Where to change what

| Concern | File |
|---|---|
| Bundle valuation formula (substitutes, complements, saturation, caps) | `src/auctionlab/instances/structured.py` |
| Population validity and non-triviality checks | `src/auctionlab/instances/population_design.py` |
| Interest-map inference and grounding rules | `src/auctionlab/llm/interest_map.py` |
| Candidate support filtering | `src/auctionlab/llm/interest_map.py` (`generate_candidate_bundles_from_interest_map`) |
| Provisional values and chunking | `src/auctionlab/llm/provisional_valuations.py` |
| Calibration rule and its fitting | `src/auctionlab/llm/value_calibration.py`, `src/auctionlab/experiments/pv_calibration.py` |
| Sealed elicitation events | `src/auctionlab/experiments/proxy_sealed_runner.py` |
| Clock elicitation events and terminal closure | `src/auctionlab/experiments/proxy_clock_runner.py` |
| Event on/off switches used by the ablations | `src/auctionlab/experiments/event_policy.py` |
| Winner determination and tie-breaking | `src/auctionlab/solvers/wdp_ilp.py` |
| VCG payments and bidder-removal witnesses | `src/auctionlab/payments/vcg.py` |
| Prompt text and output contracts | `src/auctionlab/llm/prompts.py`, `schemas.py` |
| Response parsing and failure policy | `src/auctionlab/llm/parsing.py` |
| Frozen specification and hashing | `src/auctionlab/experiments/final_pipeline.py` |
| Result tables and figures | `scripts/build_final_analytic_package.py`, `scripts/build_paper_figures.py` |

Anything under `experiments/` that changes an event policy alters what the
mechanisms ask, so a change there invalidates the frozen specification and
requires a new version rather than an in-place edit.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv venv
./venv/bin/pip install -e '.[dev]'
```

Provider keys are read from environment variables. Never commit them:

```bash
export OPENAI_API_KEY='...'
export GEMINI_API_KEY='...'
```

## Frozen experiment

The final specification is:

```text
outputs/final/final_experiment_v3.json
```

It content-addresses:

- `scenarios/pc_build_v3/pc_build_population_16x16.json`;
- all 95 frozen scalability elicitation packs;
- `outputs/pv_calibration/validation/pv_calibration.json`;
- the selected sealed and clock event policies; and
- the mechanism parameters and seed/size grid.

Verify all hashes without running an auction:

```bash
./venv/bin/python scripts/run_frozen_final_experiment.py \
  --spec outputs/final/final_experiment_v3.json \
  --output-dir outputs/reproduction \
  --dry-run
```

Run the paired final suite:

```bash
./venv/bin/python scripts/run_frozen_final_experiment.py \
  --spec outputs/final/final_experiment_v3.json \
  --output-dir outputs/reproduction \
  --mechanisms both \
  --fail-fast
```

Completed cases are skipped unless `--rerun-complete` is supplied.

## Environment generation

The public design fixes the goods, bidder roles, validation constraints, and
sampling requirements. The model supplies the hidden numeric and qualitative
preference details.

```bash
./venv/bin/python scripts/generate_pc_build_population.py \
  --design scenarios/pc_build_v2/population_design_16x16.json \
  --output scenarios/pc_build_v3/pc_build_population_16x16.json \
  --raw-output-dir scenarios/pc_build_v3/raw_generation \
  --manifest scenarios/pc_build_v3/generation_manifest.json \
  --provider openai \
  --model gpt-5.6-sol \
  --reasoning-effort medium \
  --max-tokens 24000 \
  --timeout 240 \
  --profiles-per-call 4 \
  --max-batch-repair-retries 3 \
  --validation-seeds 0 1 2 3 4 \
  --resume
```

Offline validation is also available:

```bash
./venv/bin/python scripts/validate_pc_build_population.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --report outputs/reproduction/population_validation.json
```

## Frozen elicitation packs

The final grid contains three paths for sizes 4 through 10 and five
composition seeds: goods varying with eight bidders, bidders varying with
eight goods, and both varying together. The shared 8x8 anchor is generated
once per seed, yielding 95 cases.

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

The preparation runner is resumable. Completed master packs and cases are
reused, and API calls use a persistent read-write cache during preparation.

## Calibration and policy selection

Provisional-value calibration is performed on generated non-PC domains. The
retained workflow is:

```bash
./venv/bin/python scripts/generate_pv_calibration_environments.py --help
./venv/bin/python scripts/prepare_pv_calibration_benchmark.py --help
./venv/bin/python scripts/fit_pv_calibration.py --help
./venv/bin/python scripts/validate_pv_calibration.py --help
```

Clock parameter calibration remains available separately:

```bash
./venv/bin/python scripts/tune_clock_parameters_calibration.py --help
```

The final event-policy comparison uses the five frozen 8x8 packs:

```bash
./venv/bin/python scripts/run_event_ablation_8x8.py --help
```

The final sealed policy verifies incumbents and relevant bidder-removal
counterfactuals, with scarcity and large-correction follow-ups. The final
clock policy lets prices discover revealed demand first, then performs a
single revealed-witness pass followed by staged winner/witness closure.

## Final analysis and paper

First reconstruct interest-map support tables from the frozen packs:

```bash
./venv/bin/python scripts/analyze_interest_maps.py \
  --input-dir outputs/elicitation_packs/scalability \
  --output-dir outputs/interest_map_analysis
```

Then rebuild the complete analytic package without LLM calls:

```bash
./venv/bin/python scripts/build_final_analytic_package.py \
  --spec outputs/final/final_experiment_v3.json \
  --sealed-dir outputs/final/scalability_v2 \
  --clock-dir outputs/final/clock_sandwich_scalability_v3 \
  --interest-map-dir outputs/interest_map_analysis \
  --sealed-ablation outputs/final/event_policy_8x8 \
  --clock-ablation \
    outputs/final/clock_focused_closure_ablation_pc_8x8_calibrated \
  --output-dir outputs/final/analytic_package_v3
```

Rebuild generated-environment tables and the environment figure:

```bash
./venv/bin/python scripts/analyze_generated_environment.py \
  --scenario-spec scenarios/pc_build_v3/pc_build_population_16x16.json \
  --validation-report \
    scenarios/pc_build_v3/pc_build_population_16x16.validation.json \
  --output-dir outputs/final/analytic_package_v3
```

Two report figures are derived from the tracked tables and written outside
the frozen package:

```bash
./venv/bin/python scripts/build_paper_figures.py
```

The report source is held separately from this repository. Its reproducibility
appendix lists the commands above as the way to regenerate every table and
figure it contains.

## Testing

```bash
./venv/bin/python -m pytest -q
```

The tests cover winner determination, VCG counterfactuals, both mechanisms,
interest-map filtering, provisional values, frozen-pack integrity, population
sampling, event policies, calibration, caching/parsing, and final-pipeline
hash verification.

## Generated data policy

Bulk auction results and frozen elicitation packs are excluded from Git. The repository tracks the compact final analytic package, final experiment manifest, generated population and generation provenance. Large frozen inputs should be distributed through a versioned research-data archive.
