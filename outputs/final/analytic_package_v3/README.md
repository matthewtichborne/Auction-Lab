# Final analytic package

Offline analysis of 95 matched auction cases across five environment seeds. Model robustness was deliberately deferred.

## Inputs

- Sealed: `outputs/final/scalability_v2`
- Clock: `outputs/final/clock_sandwich_scalability_v3`
- Interest maps: `outputs/interest_map_analysis`
- Sealed ablation: `outputs/final/event_policy_8x8`
- Clock ablation: `outputs/final/clock_focused_closure_ablation_pc_8x8_calibrated`
- PV calibration: `outputs/pv_calibration/validation`
- Environment validation: `scenarios/pc_build_v3/pc_build_population_16x16.validation.json`
- Frozen specification: `outputs/final/final_experiment_v3.json`

## Contents

- `FINAL_ANALYSIS.md`: concise interpretation and paper-ready headline claims.
- `final_summary.json`: machine-readable headline results and provenance.
- `tables/`: case-level data, seed-clustered summaries, event diagnostics, ablations, calibration evidence and selected trajectories.
- `figures/`: paper-ready PNG and PDF figures.

Uncertainty intervals use the five seed-level means as the independent units. Auction-size cells within a seed are treated as repeated observations rather than independent replications.

## Rebuild

```bash
./venv/bin/python scripts/build_final_analytic_package.py \
  --spec outputs/final/final_experiment_v3.json \
  --sealed-dir outputs/final/scalability_v2 \
  --clock-dir outputs/final/clock_sandwich_scalability_v3 \
  --interest-map-dir outputs/interest_map_analysis \
  --sealed-ablation outputs/final/event_policy_8x8 \
  --clock-ablation outputs/final/clock_focused_closure_ablation_pc_8x8_calibrated \
  --output-dir outputs/final/analytic_package_v3
```
