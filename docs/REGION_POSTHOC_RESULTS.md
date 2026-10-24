# Post-hoc region analyses

**2026-10-24. Exploratory, no new model inference.** Same inspected PolypGen
cohort:21 promptable sequences,2,012 targets. The original frozen hypothesis
decision is unchanged. These analyses do not establish full-propagation gains.

## Boundary equivalence

Sequence-level TOST of **control-adjusted excess Dice loss**, post-hoc margin
±0.02,90% t intervals; Holm correction within the six-definition family.
Population: informative present-GT targets. Equivalence concerns the cohort
mean, not every frame/sequence or semantic content.

| Definition | Mean excess loss | 90% CI | Holm TOST p | Within ±0.02 |
|---|---:|---|---:|---|
| Mixed cells | −0.00051 | [−0.00485,0.00384] | 0.00000059 | Yes |
| Transfiner incoherence | −0.00086 | [−0.01080,0.00908] | 0.00596 | Yes |
| Trimap k15 | +0.00162 | [−0.00528,0.00852] | 0.00044 | Yes |
| Trimap k21 | +0.00235 | [−0.00666,0.01136] | 0.00596 | Yes |
| Trimap k29 | +0.00157 | [−0.01015,0.01329] | 0.01342 | Yes |
| Inner band0.02 | +0.01301 | [−0.00344,0.02946] | 0.23606 | No |

**5/6** pass for informative present targets; only **3/6** pass across all valid
targets. Do not generalize the former result to absence or every definition.
Raw native-minus-excluded loss tests are retained in the detailed report.

## GT-empty targets and false alarms

323 GT-empty targets in17 sequences; native predictions are nonempty on
**163/323 (50.46%)**. These are annotation-defined empty masks, not independently
verified clinical absence. Below are one-step exclusions, not video propagation.

| Primary exclusion | Residual false alarms / all GT-empty | New false alarms / baseline-clear | Cleared false alarms / baseline-FP |
|---|---|---|---|
| Background | 312/323 (96.59%) | 149/160 (93.13%) | 0/163 (0%) |
| Foreground interior | 134/323 (41.49%) | 0/160 (0%) | 29/163 (17.79%) |
| Boundary | 159/323 (49.23%) | 0/160 (0%) | 4/163 (2.45%) |

The supplied **85.00% background-induced false alarms** is the mean of12
eligible sequence rates, not the pooled93.13%. **42.26% foreground clearance**
is the mean of13 eligible sequence rates, not the pooled17.79%. Conditional
denominators exclude sequences with no baseline-clear or baseline-FP frames.
Both summaries are valid but answer different questions; never interchange them.

Interpretation: foreground memory can support persistent false alarms on some
GT-empty targets. This motivates a causal propagation test, not a conclusion
that permanently deleting foreground improves segmentation. The background
control also removes object positions; see [control note](REGION_POSTHOC_NOTES.md).

The subsequent previous-score gate **does not directly test persistent-FP
clearance**. Its pre-inference option-A amendment tests relapse prevention only;
see [feasibility and scope](FULL_PROPAGATION_FALSE_ALARM.md).

## Attention by query region and layer

Primary mixed cells. Entries are boundary-memory enrichment; parentheses give
mean boundary share of spatial-memory attention, excluding pointers. Enrichment
is exp(median sequence log pooled enrichment). Query labels are offline GT
references; stored-memory labels are predicted masks. Layer indices are0–3.

| Memory-attention layer | Background query | Foreground query | Boundary query |
|---|---:|---:|---:|
| 0 | 4.8566 (8.67%) | 8.1996 (15.52%) | 22.0003 (37.76%) |
| 1 | 1.0379 (2.00%) | 3.8004 (7.14%) | 23.6546 (36.81%) |
| 2 | 11.4095 (16.74%) | 9.4618 (16.96%) | 10.2764 (18.46%) |
| 3 | 16.8121 (27.31%) | 19.5257 (30.00%) | 36.5005 (54.10%) |

Late-layer boundary enrichment is high, but not monotonic across layers, and
only boundary queries assign a majority of spatial mass to boundary memory in
layer3. “All queries converge on edge tokens” would overstate these results.

## Polyp size

Frozen per-frame positive-GT area quartiles:2.034683%,5.384635%,11.534966%.
Entries are mean excess Dice loss. Equal-sequence means within each bin;
different bins contain different sequence mixes. Descriptive, not adjusted tests.

| Size bin | Sequences | Targets | Background | Foreground interior | Boundary |
|---|---:|---:|---:|---:|---:|
| Q1,smallest | 14 | 423 | +0.1675 | −0.0069 | +0.0046 |
| Q2 | 19 | 422 | +0.3197 | +0.0375 | +0.0041 |
| Q3 | 18 | 423 | +0.2907 | +0.0413 | −0.0036 |
| Q4,largest | 16 | 421 | +0.1846 | +0.1571 | −0.0065 |

Foreground exclusion has greater excess harm in larger-size bins, but this
does not isolate size causally. Native Dice is not monotonic across these bins.

## Attention-sink test and sources

**Not computable from the saved current-cohort artifacts:** source16 saved
attention, mask logits, region labels and feature hashes, not raw memory vectors.
Do not use mask-logit norms instead. Even a future norm association would not
prove a sink; projected Q/K, positions and softmax require consideration.

Reproduction: `scripts/analyze_posthoc_regions.py`; generated tables/JSON and
artifact provenance are in `outputs/experiment/18_posthoc_region_analysis_v1/`.
17 cloud-only case archives were downloaded through the W&B API into that Drive
folder; artifact MD5 and ZIP CRC checks passed. Original source files were not
overwritten. Detailed outputs include all definitions,36 query/layer/region
cells, exact denominators and per-sequence values.
