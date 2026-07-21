# Parameter-tuning methodology

This document explains how AuctionLab's proxy-mediated experiments separate
five kinds of parameter, why the separation matters, and the recommended
pipeline for running clean (non-overfit) experiments. It complements
`docs/proxy_architecture.md` (what each parameter *does*) with a
methodological framing (what each parameter is *for*, and where it should be
selected).

## The five categories

1. **Valuation calibration** -- parameters that control how a proxy converts
   a bidder's natural-language answers into estimated bundle values:
   `epsilon`, `discount_inferred`, anchor-value usage
   (`disable_anchor_values`), and (if enabled in future) any other PV-scale
   knob. These are properties of the *estimator*, not of any one auction
   environment.

2. **Implementation budgets** -- parameters that exist only so a model call
   doesn't get truncated or fail to parse: `pv_max_tokens` chiefly, but also
   generic client `max_tokens`/`timeout`/`max_parse_retries`. These have no
   economic meaning. A budget that's too low produces a parsing failure or a
   silently incomplete response, not a "worse" experimental condition -- so
   it should be set generously, logged, and checked for
   truncation/parse-retry symptoms, never swept as if it were a treatment.
   The example configs under `configs/` set `pv_max_tokens: 6000` in every
   development/final arm -- generous enough to avoid truncation with
   uncapped candidate bundles, and deliberately identical across arms so it
   can never be mistaken for a swept variable.

3. **Mechanism treatment variables** -- parameters that change what
   information the mechanism reveals or how much a bidder is asked to
   articulate: sealed `sealed_feedback_rule` and `sealed_elicitation_rounds`,
   and clock `top_k`. These are the actual experimental questions this
   codebase exists to answer ("does more elicitation depth help?", "does a
   richer top_k demand signal help?") -- they belong in the *results*
   section of a write-up, not in a tuning loop that searches for the value
   that maximizes welfare.

4. **Development-tuned thresholds** -- clock `clock_tie_threshold`,
   `clock_margin_threshold`, and `price_step`. Unlike category 3, these
   don't change what information is revealed in principle; they only change
   how eagerly the mechanism reacts to noisy signals and how fast prices
   converge. It's legitimate to select these on a development environment
   (a scenario/seed not used for the final reported numbers) and then freeze
   them before final evaluation, the same way a learning-rate or
   convergence tolerance would be tuned in an ML experiment.

5. **Final evaluation variables** -- everything that's held fixed once
   development tuning and valuation calibration are done: the frozen
   valuation-calibration settings, the frozen development-tuned thresholds,
   and the held-out environment(s) the final sealed/clock sweep runs on.

A parameter that is nominally in category 1, 2, or 4 but gets swept directly
against final auction welfare has effectively been promoted to category 3
without anyone deciding that on purpose -- that's the overfitting failure
mode this document exists to prevent.

## Why valuation calibration needs its own out-of-sample benchmark

If `epsilon`/`discount_inferred`/anchor-value settings are chosen by trying
several values and keeping whichever maximizes welfare on the PC-build
auction scenario, the reported final numbers are no longer an honest test of
the mechanism -- they're partly a test of how well that one scenario's noise
was fit. `scripts/run_value_calibration_benchmark.py` exists to break that
loop: it generates a synthetic bundle-pricing benchmark in a domain the final
auction experiments never touch (home office, travel package, camera/video
kit, kitchen appliances, gaming peripherals -- see `DOMAIN_CATALOGS` in that
script), with hidden ground-truth valuations from the same deterministic
substitute/complement/saturation model used elsewhere in the codebase
(`auctionlab.instances.structured`). Valuation-calibration parameters are
selected by minimizing PV-vs-ground-truth error *there*, then frozen before
any PC-build auction is run. See "Recommended pipeline" below.

## Why hidden candidate caps are removed from main experiments

`max_candidate_bundles` (and, if set, `max_bundle_size`) determine how many
bundles a proxy is even allowed to *have an opinion about*. A capped
candidate set can silently improve reported welfare (by discarding
low-information bundles a real PV call would have overestimated) or silently
hurt it (by discarding a bundle that would have won). Either way, changing
the cap is changing what the proxy is allowed to know, not how well it
reasons -- so it isn't a fair axis to tune against final welfare. The fix is
structural: main/final experiment configs leave `max_candidate_bundles` and
`max_bundle_size` unset (`None`), and any explicit cap is confined to an arm
labelled `"robustness"` in the grid config (`ExperimentArm.label`) so it's
never mistaken for a tuned default. `grid_methodology_warnings()` in
`auctionlab.experiments.grid_config` flags a cap set outside a
robustness-labelled arm.

## Why sealed feedback rule and sealed elicitation rounds are experimental treatments

`sealed_feedback_rule` (`none`, `allocated_bundle`, `lost_interested_bundle`,
`all_provisional`, `competitive`, `all_valued_bundles`) and
`sealed_elicitation_rounds` change *what the mechanism tells a losing or
winning bidder to refine*, and *how many chances* it gets to refine. That is
exactly the research question ("which feedback signal helps a
miscalibrated proxy converge to something closer to true value, and how
much elicitation depth is needed"). Treating them as tuning knobs to
maximize one scenario's welfare would just find whichever rule happens to
flatter that scenario's PV errors -- the final grids in
`configs/final_experiment_grid_example*.json` instead vary them as labelled
treatment arms and report the comparison. The example final grids sweep
seven sealed treatment arms (`final_sealed_none_r0`,
`final_sealed_allocated_bundle_r1`, `final_sealed_allocated_bundle_r3`,
`final_sealed_competitive_r3`, `final_sealed_all_provisional_r3`,
`final_sealed_all_provisional_r5`, `final_sealed_all_valued_bundles_r3`) so
both the feedback rule *and* the elicitation-depth axis (1 vs. 3 vs. 5
rounds) are compared, not just one rule at a fixed round count.

## Why clock top_k is an experimental treatment

`docs/proxy_architecture.md` documents the mechanics: `top_k` controls how
many packages a bidder reports as primary demand each round, which changes
the price path, which bundles get flagged for supplementary inclusion, and
when the clock terminates. It is a genuine information-structure lever, not
a free performance dial -- comparing `top_k = 1, 2, 3` is itself one of the
things these experiments are meant to measure. The final grid configs
sweep it explicitly (`final_clock_topk1/2/3` arms in
`configs/final_experiment_grid_example*.json`); the grid aggregator's
"Effect of clock top_k" report section exists specifically to surface this
comparison (mean efficiency/rounds/value-queries per `top_k`), not to pick a
"winning" value and discard the rest.

## Why clock tie/margin thresholds and price step may be development-tuned

Unlike `top_k`, these three don't change *what* the clock can express --
they change how sensitive its near-tie/near-zero-surplus detection is and
how fast prices move. A `clock_tie_threshold` that's too loose fires
refinement events on noise; one that's too tight misses genuine near-ties.
This is closer to a numerical-stability parameter than an economic one, so
it's acceptable to select on a development environment
(`configs/auction_development_grid_example.json`, `grid_type:
"development"`) and then freeze the winning combination for the final
sealed/clock sweep. The development grid's arms are labelled `"development"`
precisely so a later reviewer can tell these were selected, not treatment
conditions being compared. Tuning thresholds requires refinements to
actually happen, so the development grid's arms also set a generous global
refinement budget (see below) rather than 0 -- a cap of 0 is *unlimited* by
this schema's convention, but that's exactly the ambiguity a tuning-focused
grid should avoid leaving implicit.

## Refinement-query safety caps are not tuning targets either

The grid config schema (`auctionlab.experiments.grid_config.SafetyLimits`)
exposes two fields, both **`null`/omitted by default, meaning "no limit"**:

- `per_bidder_refinement_query_limit` -- caps refinement value queries for
  a single bidder. Refinement *count* is meant to be an outcome of the
  feedback rule / elicitation rounds / `top_k` treatment described above,
  not a cap that quietly decides how much refinement is "allowed" before a
  comparison even starts. In **main experiments** (development and final
  grids) this stays `null` -- no arm should impose a per-bidder ceiling. It
  is only meaningful as an explicit, positive value inside an arm labelled
  `"robustness"`, testing a fixed query budget (e.g. `3`, as in
  `robustness_query_budget_per_bidder_3` in
  `configs/auction_development_grid_example.json`) --
  `grid_methodology_warnings()` flags a non-null value outside a
  robustness-labelled arm.
- `global_refinement_query_safety_limit` -- a backstop against runaway
  query volume, summed across *all* bidders combined. It's a guardrail, not
  an economic tuning parameter, so it's set to a high, explicit value (the
  example grids use `200`, several orders of magnitude above anything a
  real run should hit) rather than left `null` -- the point of the example
  configs is to *show* a concrete guardrail is in place, not to leave it
  implicit.

**`0` is deliberately not a valid value for either field** -- it reads too
easily as "zero refinements allowed" rather than "unlimited," which is
exactly the ambiguity that used to show up as
`--max-refinement-queries-per-bidder 0` in every dry-run command regardless
of whether a cap was ever intended. Passing a positive value flattens to
the pre-existing CLI flags of the same name for backwards compatibility
(`--max-refinement-queries-per-bidder N` / `--max-total-refinement-queries
N`) via `ExperimentArm.to_cli_args()`; passing `null`/omitting the field
means neither flag is emitted at all, so
`ProxySealedConfig`/`ProxyClockConfig` (which still use their own internal
`0`-means-unlimited convention at the runtime layer) simply see their
own defaults. The deprecated field names
(`max_refinement_queries_per_bidder` / `max_total_refinement_queries`) are
still accepted as aliases on input for backwards compatibility with older
configs -- a legacy `0` under either old name is translated to `null` with
a `DeprecationWarning`; a legacy positive value carries over unchanged.
Value/demand queries are already deduplicated per bidder/bundle (a bundle
is refined at most once), so neither cap is a source of legitimate
experimental variation regardless of which spelling is used.

## Recommended pipeline

1. **Generate/evaluate the value-calibration benchmark**
   (`scripts/run_value_calibration_benchmark.py generate --domains all`, or
   equivalently `--config configs/value_calibration_example.json` to
   reproducibly generate all five registered domains in one call; then
   `evaluate` against a cached PV output) to compare
   `epsilon`/`discount_inferred`/anchor-value settings against hidden ground
   truth on non-PC-build domains.
2. **Select valuation-calibration settings** that minimize PV error there
   (mean/median absolute error, MAPE, rank correlation, top-k recall, and
   large-bundle overvaluation bias are all reported).
3. **Select clock thresholds on development environments**
   (`scripts/run_proxy_parameter_grid.py` against a `grid_type:
   "development"` config) -- tie/margin thresholds and price step only.
4. **Freeze** both sets of settings.
5. **Run final sealed feedback/round and clock top_k sweeps** on held-out
   environments (`grid_type: "final"`), with candidate caps unset and the
   per-bidder refinement cap left at its unlimited default (only a high,
   explicit global safety backstop is set); any cap ablation is a
   separately labelled `grid_type: "robustness"` run. To cover multiple
   environment sizes, run the size-specific example grids
   (`configs/final_experiment_grid_example.json` for 6x6,
   `configs/final_experiment_grid_example_8x8.json`,
   `configs/final_experiment_grid_example_10x10.json`) and aggregate them
   together -- `scripts/run_proxy_parameter_grid.py`'s summary-by-environment
   output is grouped by `(scenario_spec, num_goods, num_bidders,
   scenario_seed)`, so results from all three sizes compare cleanly
   side by side without needing a single mega-config.

## Dissertation-methods note

> Valuation calibration was performed out-of-sample on a synthetic
> bundle-pricing benchmark. Auction experiments then fixed
> valuation-calibration parameters and investigated mechanism-specific
> feedback signals as experimental treatments.

## Tooling reference

| Category | Where it's selected | Tooling |
|---|---|---|
| Valuation calibration | Out-of-sample benchmark | `scripts/run_value_calibration_benchmark.py` |
| Implementation budgets | Set generously, logged | `--pv-max-tokens`, `PvCandidateBundleStats` truncation diagnostics |
| Mechanism treatments | Final grid, compared not tuned | `sealed_feedback_rule`, `sealed_elicitation_rounds`, clock `top_k` |
| Development-tuned thresholds | Development grid, then frozen | `clock_tie_threshold`, `clock_margin_threshold`, `price_step` |
| Safety limits | `null` (unlimited) by default | `per_bidder_refinement_query_limit`, `global_refinement_query_safety_limit` |

All five categories are represented as distinct sections of the
`ExperimentArm`/`ExperimentGrid` schema in
`auctionlab.experiments.grid_config`, so a grid config file's shape mirrors
this document's structure directly. `scripts/run_proxy_parameter_grid.py`
runs or aggregates grids built from that schema; see its module docstring
for `--dry-run` / `--aggregate-only` / normal run-mode usage.
