# Final analysis

## Scope

The primary analysis contains 95 matched cases: 19 scalability cells for each of five independently sampled environment seeds. The sealed and clock mechanisms use the same frozen elicitation pack within each case. Model robustness is deferred and is not part of the final claims below.

## Main results

- Initial provisional valuations allocate **86.4%** of full-information welfare on average.
- Sealed elicitation raises mean efficiency to **94.8%**, a paired gain of **8.4 percentage points** (seed-clustered 95% CI **3.9 to 12.9**). It improves 78/95 cases and uses 21.8 value queries on average.
- Clock elicitation raises mean efficiency to **92.8%**, a paired gain of **6.5 percentage points** (95% CI **1.3 to 11.6**). It improves 68/95 cases and uses 12.3 value queries on average.
- Sealed is **1.9 percentage points** more efficient than clock on average, but the seed-clustered CI crosses zero. Clock saves **9.6 queries per case** relative to sealed (95% CI **7.4 to 11.7 fewer**).
- Payment reconstruction remains materially harder than allocation reconstruction. Mean payment error normalised by optimum welfare is **32.1%** for sealed and **36.5%** for clock. Their paired difference is not resolved by five seeds because its CI crosses zero.

## Interest-map scalability

Across all 95 cases, inferred candidate support is **55.5% smaller** than the full non-empty powerset baseline. At 10 goods with 8 bidders, the mean auction-level support falls from **8184** full-powerset valuations to **2619.2** candidates, a **68.0% reduction**. These figures measure reconstruction from validated disclosures; the perfect qualitative-map reconstruction is therefore a pipeline check, not a claim about unrestricted natural-language inference.

## Policy evidence and elicitation events

The sealed 8x8 ablation supports adaptive competitive-frontier refinement at **99.1%** mean efficiency with **26.2** queries. The clock 8x8 ablation supports the revealed-winner sandwich at **95.7%** with **13.4** queries, compared with 88.1% for PV-only clock allocation.

The most targeted clock event is the single-pass revealed-VCG event: 57.5% of its queries appear in the final reported VCG witness set. In sealed elicitation, allocated-bundle and competitive-counterfactual events have similar reported-witness hit rates (about 30–31%), while scarcity fallbacks are less often pricing witnesses but can still alter allocation-relevant alternatives.

## Calibration and clearance checks

The frozen uniform PV scale is **1.4758**. Across nine held-out synthetic environments, it reduces mean normalised payment error from **0.289** to **0.233**—a **19.5% relative reduction**—and improves 7/9 held-out cases.

All 95 clock cases clear naturally through no excess demand. The longest takes 84 rounds; 17 cases exceed 50 rounds, confirming that the 500-round setting acts only as a safeguard.

## Interpretation limits

- The five seeds, rather than the 95 correlated size cells, are the independent replication units used for uncertainty intervals.
- The environment domain is PC components; mechanism conclusions should be framed as evidence within this structured domain.
- VCG revenue is computed from reported bids. High welfare efficiency does not imply oracle-equivalent payments because counterfactual witness bundles may remain misreported or unqueried.
- Policy ablations are 8x8 selection evidence; the 95-case scalability suite evaluates the frozen selected policies.
