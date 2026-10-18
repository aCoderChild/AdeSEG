# Region-token experiments: results

Updated **19 October 2026 (UTC+7)**. Region-focused study; superseded latent-utility experiments excluded.

## 1. Completion

| Component | Verified status |
|---|---|
| Chronological SAM2.1 B+ baseline | 21 promptable sequences; 2,012 post-prompt frames |
| Regional segmentation replays | 13 development sequences, 25 targets; 3200 rows (2940 complete, 260 infeasible/skipped) |
| New attention measurements | seq11–14: 32 target-condition observations × 4 layers; seq3 repeat used only for validation |
| Seq15 attention | Rejected by unchanged reconstruction threshold; excluded |
| Seq15 segmentation-only retry | Complete; native/replay exactness checks retained |
| Persistent region ablations | seq13/14, 18 rollouts; 4,410 frame records |
| Kaggle mirror results | Quarantined: sampled seq13 frames are nonchronological; Version 2 cancelled |
| Learned v2 method / cross-domain study | Incomplete; no validated trained-method gain |
| New disjoint geometry + reweighting headroom | Complete: 13/13 development sequences; 1,500 geometry rows, 60 skipped; training gate not met |
| Efficient-SAM2 MPS port | Execution smokes complete; native sparse reads and overlays recorded. Repeat/no-op validation failed; no accepted Dice-gain claim or full run |

**Protocol:** SAM2.1 Hiera-B+, 1024; one first-visible labeled mask, no corrections; native pointers retained. Replay regions use fixed oracle GT supports, for analysis only. Altered target predictions are not written back.

**Statistics:** average seeds and targets within sequence; sequence is the independent unit. Paired two-sided Wilcoxon, 10,000 whole-sequence bootstrap resamples. CIs are unadjusted. Differences use 0–1 units; 0.01 = one percentage point. Analyses are post-interim/exploratory.

Sections 2–9 report local chronological runs. Sections 10–12 cover saved Kaggle
outputs, additional local diagnostics and MPS portability attempts. Failed runs
are listed for coverage, not included in performance averages. The full registered
study is **not complete**; no live job status is inferred from these files.

## 2. Regional removal effects

H = baseline − region-removed score; positive means removal hurts. E = H(region) − mean H(random).

Controls match the **boundary reference's** frame/distance strata and budget; overlap is allowed. Only the boundary comparison is distance-matched. Other regions share its budget, **not** its distance distribution. Unequal feasible supports prohibit a direct regional ranking.

### Natural predicted memory — recent bank

| Endpoint | Region | Sequences / targets | H | E [95% CI] | Holm p |
|---|---|---:|---:|---|---:|
| Dice | Boundary | 13 / 24 | +0.00014 | −0.00046 [−0.00148, +0.00042] | 1.0000 |
| Dice | Foreground interior | 7 / 9 | −0.00032 | +0.00019 [−0.00016, +0.00072] | 1.0000 |
| Dice | Near background | 13 / 24 | −0.00037 | −0.00097 [−0.00316, +0.00131] | 1.0000 |
| Dice | Far background | 13 / 24 | +0.00006 | −0.00054 [−0.00209, +0.00090] | 1.0000 |
| Boundary F | Boundary | 13 / 24 | −0.00306 | −0.00289 [−0.00710, +0.00098] | 1.0000 |
| Boundary F | Foreground interior | 7 / 9 | −0.00260 | −0.00037 [−0.00208, +0.00142] | 1.0000 |
| Boundary F | Near background | 13 / 24 | −0.00053 | −0.00035 [−0.00827, +0.00881] | 1.0000 |
| Boundary F | Far background | 13 / 24 | +0.00026 | +0.00044 [−0.00475, +0.00541] | 1.0000 |

### Clean GT spatial memory (c0) — recent bank

| Endpoint | Region | Sequences / targets | H | E [95% CI] | Holm p |
|---|---|---:|---:|---|---:|
| Dice | Boundary | 13 / 24 | +0.00581 | +0.00594 [+0.00119, +0.01197] | 0.1880 |
| Dice | Foreground interior | 7 / 9 | −0.00733 | −0.00193 [−0.00506, −0.00017] | 0.3125 |
| Dice | Near background | 13 / 24 | −0.01098 | −0.01085 [−0.02420, −0.00139] | 0.0273 |
| Dice | Far background | 13 / 24 | +0.00009 | +0.00023 [−0.00660, +0.00932] | 0.2549 |
| Boundary F | Boundary | 13 / 24 | +0.02541 | +0.02187 [+0.00828, +0.03676] | 0.0156 |
| Boundary F | Foreground interior | 7 / 9 | −0.01317 | −0.00483 [−0.01321, −0.00025] | 0.7500 |
| Boundary F | Near background | 13 / 24 | −0.02708 | −0.03062 [−0.06679, −0.00286] | 0.1914 |
| Boundary F | Far background | 13 / 24 | −0.00023 | −0.00377 [−0.01844, +0.00938] | 0.8008 |

Holm family: eight region × condition comparisons **per endpoint**, across both tables. Conditioning-bank and per-sequence results remain in the detailed sources.

**Analysis:** no natural-memory contrast survives correction. Under c0, boundary removal has excess boundary-F harm (+0.02187, Holm p = 0.0156). Near-background removal has negative excess Dice harm (−0.01085, p = 0.0273), but that comparison does not control its own distance distribution. Neither establishes uniform regional superiority or a deployable gain.

## 3. Regional attention

seq11–14 only, n = 4; average four layers and sampled targets, then equally over sequences. All native queries/heads; nonempty regional supports only. Enrichment = mass / region's fraction of **all keys**, including pointers. Recent spatial bank shown; saliency is not causal importance.

| Region | c0 mass | c0 enrichment | Natural mass | Natural enrichment |
|---|---:|---:|---:|---:|
| Boundary | 16.40% | 3.604× | 14.57% | 3.326× |
| Foreground interior | 7.51% | 1.713× | 8.21% | 1.826× |
| Near background | 7.50% | 1.205× | 7.66% | 1.220× |
| Far background | 28.58% | 0.407× | 31.81% | 0.448× |

Pointer mass, separately: c0 **2.04%**, natural **1.94%**. Conditioning spatial keys are not included in the region table.

**Analysis:** boundary tokens receive the most attention per token among these regions. Natural-memory removal does not establish greater causal benefit. Enrichment alone cannot separate identity from proximity.

## 4. Boundary-relative condition contrasts

Δ = E(condition) − E(c0), n = 13 paired sequences. These are target-bank perturbations, not accumulated-corruption rollouts or method gains.

| Endpoint | Condition vs c0 | Mean Δ [95% CI] | Holm p |
|---|---|---|---:|
| Dice | c1 | −0.01540 [−0.02876, −0.00516] | 0.0146 |
| Dice | c2 | +0.00819 [−0.00769, +0.03286] | 0.5186 |
| Dice | natural | −0.00640 [−0.01233, −0.00170] | 0.0146 |
| Boundary F | c1 | −0.03272 [−0.06006, −0.00548] | 0.1543 |
| Boundary F | c2 | −0.00053 [−0.02915, +0.02925] | 0.9097 |
| Boundary F | natural | −0.02476 [−0.04129, −0.01015] | 0.0029 |

Holm family: three contrasts per endpoint, separate from Section 2. The c2 pattern does not establish monotonic degradation.

## 5. Persistent regional masking

| Region removed | seq13 Dice (Δ native) | Infeasible / reads | seq14 Dice (Δ native) | Infeasible / reads |
|---|---:|---:|---:|---:|
| Boundary | 0.30024 (−0.28186) | 194 / 241 | 0.73919 (+0.00061) | 68 / 247 |
| Foreground interior | 0.29180 (−0.29031) | 212 / 241 | 0.73941 (+0.00083) | 144 / 247 |
| Near background | 0.28937 (−0.29274) | 207 / 241 | 0.74308 (+0.00450) | 93 / 247 |
| Far background | 0.58215 (+0.00004) | 8 / 241 | 0.73864 (+0.00006) | 65 / 247 |
| Random, five-seed mean | 0.28980 (−0.29230) | 194 / 241 | 0.73860 (+0.00002) | 65 / 247 |

Native Dice: seq13 **0.58210**, seq14 **0.73858**. Two failure-selected sequences; descriptive only. Realized intervention doses differ as each policy changes subsequent predictions. Seq13 degrades similarly under regional and random masking; no unique regional error attribution.

## 6. Chronological baseline

Equal-sequence mean Dice **0.71833**, descriptive 95% bootstrap CI **[0.60744, 0.81494]**. Seq1/7 are unpromptable. Failure episode: Dice < 0.5 for ≥3 consecutive evaluated frames.

| Sequence | Post-prompt frames | Dice | Failure episodes |
|---|---:|---:|---:|
| seq2 | 59 | 0.89468 | 0 |
| seq3 | 14 | 0.86275 | 0 |
| seq4 | 47 | 0.78244 | 1 |
| seq5 | 249 | 0.90277 | 1 |
| seq6 | 66 | 0.79368 | 1 |
| seq8 | 72 | 0.36862 | 3 |
| seq9 | 47 | 0.93187 | 0 |
| seq10 | 7 | 0.65298 | 0 |
| seq11 | 225 | 0.18034 | 1 |
| seq12 | 249 | 0.94488 | 0 |
| seq13 | 242 | 0.58210 | 3 |
| seq14 | 248 | 0.73858 | 4 |
| seq15 | 115 | 0.93398 | 0 |
| seq16 | 33 | 0.94692 | 0 |
| seq17 | 46 | 0.89256 | 0 |
| seq18 | 62 | 0.45605 | 4 |
| seq19 | 55 | 0.84881 | 2 |
| seq20 | 35 | 0.14286 | 1 |
| seq21 | 47 | 0.79382 | 0 |
| seq22 | 45 | 0.90463 | 0 |
| seq23 | 49 | 0.52950 | 2 |

## 7. Earlier detector-assisted comparisons

MedSAM2-Tiny + repeated YOLO boxes; **not directly comparable to single-prompt B+**. The original held-out eight sequences have now been inspected.

| Method | All-23 Dice, descriptive | Original held-out-eight Dice |
|---|---:|---:|
| Frame-wise YOLO–MedSAM2 | 0.77000 | 0.77803 |
| Native temporal memory | 0.72460 | 0.66160 |
| Fixed reliability-only bias | 0.72618 | 0.68357 |
| Fixed region + reliability bias | 0.72646 | 0.68037 |

| Region + reliability minus comparator (n = 8) | ΔDice [95% CI] | Holm p |
|---|---|---:|
| Frame-wise | −0.09766 [−0.26337, +0.06582] | 0.445 |
| Native memory | +0.01877 [−0.00350, +0.04447] | 0.766 |
| Reliability-only | −0.00319 [−0.00775, +0.00012] | 0.844 |

Cutie refresh, seq13/14: mean Dice **0.83738** versus frame-wise **0.83383** (Δ +0.00355); **370 later mask corrections / 499 frames**, plus two initialization masks. Not autonomous region-aware improvement. Earlier learned readout used only two updates; no trained-method conclusion.

## 8. Seq15 diagnosis and verification

| Check | Verified result |
|---|---|
| Observed vs native logits | Bit-identical |
| Attention mass, four layers | 1.000000 each |
| Layer-0 reconstruction | 0.000101566 > unchanged 0.000100000 limit: rejected |
| Saved native vs CPU float64 | Max 0.000101605; RMS 0.000006977 |
| MPS explicit reconstruction vs CPU float64 | Max 0.000000370 |
| CPU SDPA vs CPU float64 | Max 0.000093642 |
| Segmentation-only seq15 retry | 256 rows; eight target-condition blocks; no attention claim |
| Project tests | 123 passed after removing custom learned heads and adding upstream/MPS adapter tests; one upstream Torch deprecation warning |
| Delivery | Successful follow-ups and numerical audit finished; W&B result artifacts COMMITTED |

Consistent with SDPA/reconstruction numerics; exact internal kernel cause unproven. Seq15 attention remains excluded. **No acceptance threshold was relaxed.**

## 9. Disjoint geometry and regional-bias headroom

13 development sequences. Each region has its own disjoint frame/distance-matched
control; selected subsets are capped at 64 keys. No Dice or boundary-F geometry
comparison survives Holm correction. These results concern sampled subsets, not
whole-region removal.

| Endpoint | Memory | Oracle gain | Equal-search null gain | Excess [95% CI] | Holm p |
|---|---|---:|---:|---|---:|
| Dice | Natural | +0.005535 | +0.000347 | +0.005188 [+0.001220, +0.011926] | 0.00098 |
| Dice | Clean c0 | +0.004535 | +0.000913 | +0.003621 [+0.000958, +0.007254] | 0.00098 |
| Boundary F | Natural | +0.012481 | +0.002038 | +0.010444 [+0.005250, +0.016705] | 0.00391 |
| Boundary F | Clean c0 | +0.009673 | +0.002375 | +0.007299 [+0.001034, +0.015311] | 0.03711 |

GT selects among eight regional bias candidates and a shared baseline; three
equal-search null seeds. Sequence-level statistics; two-condition Holm family
per endpoint. The oracle exceeds its null, but mean Dice gain is below the
prespecified 0.01 gate. **Training was not launched.** This is not a deployable gain.

### All geometry-controlled subset effects

13 sequences / 24 targets per displayed contrast; E is regional harm minus its
own disjoint control. Holm family: ten region × condition comparisons per endpoint.

| Memory | Region | E Dice [95% CI] | Holm p | E boundary F [95% CI] | Holm p |
|---|---|---|---:|---|---:|
| natural | boundary | -0.000113 [-0.000212, -0.000033] | 0.18896 | -0.000256 [-0.001097, +0.000372] | 1.00000 |
| natural | foreground | +0.000068 [+0.000010, +0.000140] | 0.29736 | -0.000163 [-0.001164, +0.000658] | 1.00000 |
| natural | near_background | -0.000061 [-0.000119, -0.000010] | 0.27344 | +0.000019 [-0.000420, +0.000469] | 1.00000 |
| natural | far_background | -0.000019 [-0.000060, +0.000030] | 1.00000 | -0.000034 [-0.000516, +0.000472] | 1.00000 |
| natural | mixed_side | -0.000075 [-0.000251, +0.000066] | 1.00000 | +0.000077 [-0.000465, +0.000796] | 1.00000 |
| c0 | boundary | +0.000095 [-0.000154, +0.000384] | 1.00000 | +0.000306 [-0.000177, +0.000871] | 1.00000 |
| c0 | foreground | +0.000111 [-0.000292, +0.000609] | 1.00000 | +0.000794 [-0.000839, +0.003419] | 1.00000 |
| c0 | near_background | +0.000431 [-0.000063, +0.001360] | 1.00000 | +0.000332 [-0.000162, +0.000985] | 1.00000 |
| c0 | far_background | +0.000053 [-0.000031, +0.000200] | 1.00000 | +0.000112 [-0.000132, +0.000465] | 1.00000 |
| c0 | mixed_side | +0.000649 [+0.000098, +0.001310] | 0.16113 | +0.000837 [-0.000358, +0.002588] | 1.00000 |

No comparison survives correction; unsupported strata were not imputed.


## 10. Kaggle notebook results — separate, mirror-derived evidence

These are **saved-output results, not a live job-status check**. Model-free tracking
and operator exactness checks remain useful; temporal segmentation claims from
this mirror are quarantined following the source-order audit. Do not pool these
numbers with Sections 2–9 or treat their stored p-values as confirmed evidence.

### Coverage and technical checks

| Stage / saved output | Persisted coverage | Outcome |
|---|---|---|
| CPU tracking preflight v6 → v7 | Credential-service failure → successful editor submission | W&B connectivity established; not a model experiment |
| E0 (`region_tokens_execution_e0_v2`) | seq2, target 7 | No-op, restoration, value/memory identity and full-token equivalence passed; reconstruction error 0 |
| E1 (`region_tokens_v1`) | 10 sequences; 2,463 complete / 281 skipped rows | Stored gate: proceed to E2; no hypothesis marked confirmed |
| E2 calibration / mean-value smoke | 10 sequences; one target × two banks = 20 paired rows | Operator smoke complete, not full E2 |
| E2 primary first attempt / retry2 | Download failure / later worker failure | Not complete; retry2 contains partial rows, including a superseded seq5 fragment |
| E2 primary combined analysis | 10-sequence saved analysis; up to 992 paired targets per comparison | Uses completed seq2–4 units from retry2 and seven retry3 units; not a 21-sequence test |
| E2 frame/distance batch01 | 10 sequences, 992 targets; 25,064 complete / 2,712 skipped rows | Saved batch complete; chronology quarantined |
| E2 frame/distance batch02 | seq13 child: 242 targets; 6,230 complete / 546 skipped rows | seq13 child complete; parent job cancelled; no persisted seq14 result accepted |
| Review: contour-union / frame-distance | seq3, eight targets each; 208/16 and 216/8 complete/skipped | Sensitivity smokes only |
| E3 c0/c1/c2/natural | seq3, eight targets per condition; each 216 complete / 8 skipped | Four diagnostic shards complete; full multi-sequence E3 not complete |
| E4 condition extraction | 21 sequences, 2,141 rows; 2,120 nonmissing motion values | Measurements complete; not a method-performance result |
| E5 primary analysis | Global and query-specific exports: 90 rows each | Stored correlations available; quarantine applies |
| E6 / cross-domain method evaluation | No completed full registered matrix verified | No accepted method gain |

The seq5 frame137 restore diagnosis passed its updated memory-read checks:
mutation changed the memory read even though native absence gating kept decoder
logits constant at −1024. A constant final prediction was not taken as evidence
that the intervention did nothing.

### E1 pilot: stored regional harm

Recent-bank boundary removal, n=10; H is baseline minus intervened score.

| Endpoint | H [95% CI] | Excess over largest random-seed mean | Stored gate |
|---|---|---:|---|
| Dice | +0.03215 [+0.00435, +0.06953] | +0.03074 | Pass |
| Boundary F | +0.04127 [+0.01668, +0.07270] | +0.03773 | Pass |

This historical pilot pass does not override the later chronology finding.

### E2: boundary versus controls, primary c0 analysis

Here the contrast is **H(boundary) − H(control)**, unlike Section 2's
region-minus-random estimator. Controls are equal-budget; supports differ by
comparison. Values below are the saved sequence-level analysis, with its stored
Holm corrections across controls/banks within each endpoint/analysis family.

| Bank | Control | Sequences / paired targets | ΔH Dice [95% CI] | Holm p | ΔH boundary F [95% CI] | Holm p |
|---|---|---:|---|---:|---|---:|
| conditioning | foreground | 6 / 611 | -0.00120 [-0.00736, +0.00578] | 1.0000 | -0.00089 [-0.00774, +0.00691] | 1.0000 |
| conditioning | near_background | 7 / 851 | +0.00221 [-0.00305, +0.00906] | 1.0000 | +0.00549 [-0.00613, +0.02117] | 1.0000 |
| conditioning | far_background | 10 / 992 | +0.00041 [-0.00387, +0.00558] | 1.0000 | +0.00327 [-0.00499, +0.01420] | 1.0000 |
| conditioning | random | 10 / 992 | +0.00019 [-0.00422, +0.00540] | 1.0000 | +0.00295 [-0.00478, +0.01342] | 1.0000 |
| conditioning | spatial_random | 10 / 992 | -0.00043 [-0.00279, +0.00167] | 1.0000 | -0.00213 [-0.00683, +0.00185] | 1.0000 |
| recent | foreground | 7 / 208 | +0.00103 [-0.01359, +0.01538] | 1.0000 | +0.01763 [+0.00048, +0.04010] | 0.9375 |
| recent | near_background | 9 / 810 | +0.01377 [-0.00277, +0.03661] | 1.0000 | +0.02274 [+0.00240, +0.04885] | 0.7812 |
| recent | far_background | 10 / 890 | +0.01764 [+0.00065, +0.04260] | 0.7422 | +0.02687 [+0.00718, +0.05425] | 0.1953 |
| recent | random | 10 / 890 | +0.01675 [+0.00030, +0.04087] | 0.7422 | +0.02522 [+0.00618, +0.05178] | 0.1953 |
| recent | spatial_random | 10 / 890 | -0.00138 [-0.01207, +0.00692] | 1.0000 | +0.00681 [-0.00223, +0.01514] | 0.7812 |

No displayed primary contrast survives correction. In recent memory, the
boundary–spatial-random Dice contrast is negative, not evidence of superior
boundary utility. Chronology quarantine prevents confirmatory interpretation.

### E2 mean-value substitution smoke

Ten sequences; bank-level changes averaged within sequence. Δ = intervention − baseline.

| Intervention | Mean ΔDice [95% CI] | Improved sequences |
|---|---|---:|
| Boundary attention masking | −0.00406 [−0.01388, +0.00604] | 3/10 |
| Boundary mean-value substitution | −0.00577 [−0.01254, +0.00059] | 3/10 |

Neither smoke establishes an improvement; cross-sequence swap and the full
registered operator matrix are not completed by these checks.

### E3: single-sequence corruption diagnostics

seq3 only; eight target frames and five spatial-random seeds. H is raw regional
harm; E subtracts the mean spatial-random harm at each target, then averages
targets. These are descriptive values without sequence-population CIs or p-values.

| Condition | Bank | Boundary H Dice | E Dice | Boundary H boundary F | E boundary F |
|---|---|---:|---:|---:|---:|
| c0 | conditioning | +0.02026 | +0.00584 | +0.04494 | +0.00457 |
| c0 | recent | +0.01572 | +0.00208 | +0.03356 | +0.00795 |
| c1 | conditioning | +0.05591 | +0.02090 | +0.05865 | +0.04316 |
| c1 | recent | -0.21455 | -0.00678 | -0.26854 | -0.04550 |
| c2 | conditioning | +0.03722 | +0.02729 | +0.04236 | +0.05154 |
| c2 | recent | -0.13726 | -0.01866 | -0.20297 | -0.01863 |
| natural | conditioning | +0.00182 | +0.00412 | +0.00567 | +0.00691 |
| natural | recent | +0.00004 | +0.00236 | -0.00165 | +0.00067 |

Large negative c1/c2 recent-bank harm means removal improved these particular
corrupted predictions; it is not a demonstrated general corruption mechanism.
Conditioning-bank effects can change because the rest of the attended bank changes.

### E4: saved region × imaging-condition interaction tests

Model outcome is per-token boundary-F harm. These mixed-model fits all report
singular/boundary random-effect warnings; one additionally reports a non-positive-
definite Hessian. Stored significance is not accepted as a robust finding.

| Bank | Condition | Sequences / rows | Interaction p | Holm p |
|---|---|---:|---:|---:|
| conditioning | contrast | 9 / 2549 | 0.709017 | 1 |
| conditioning | motion | 10 / 3446 | 0.293545 | 1 |
| conditioning | specular_fraction | 10 / 3446 | 0.784617 | 1 |
| conditioning | coverage_tokens | 10 / 3446 | 0.803025 | 1 |
| recent | contrast | 9 / 2341 | 3.80449e-06 | 3.04359e-05 |
| recent | motion | 10 / 2798 | 0.00764185 | 0.0534929 |
| recent | specular_fraction | 10 / 2798 | 0.607923 | 1 |
| recent | coverage_tokens | 10 / 2798 | 0.214587 | 1 |

The recent-bank contrast interaction is nominally significant in the saved model,
but chronology and model warnings preclude confirmation. Corruption fits lack
independent condition variation; class-group fits lack cross-class variation.
Cluster-robust sensitivities are separate, few-cluster analyses in the source JSON.

### E5: foreground-query attention–effect agreement

c0; layer-mean attention mass. Correlations are computed per frame and summarized
at sequence level. Endpoints are not interchangeable.

| Endpoint | Bank | Sequences / frames | Spearman ρ [95% CI] |
|---|---|---:|---|
| Whole-object boundary-F harm/token | recent | 6 / 121 | +0.10431 [+0.04108, +0.17127] |
| Whole-object boundary-F harm/token | conditioning | 3 / 281 | +0.10107 [-0.26793, +0.62182] |
| Same-query-region signed Dice harm | recent | 6 / 80 | -0.00137 [-0.25581, +0.19444] |
| Same-query-region signed Dice harm | conditioning | 3 / 74 | +0.09391 [-0.22012, +0.46078] |

These are weak/uncertain stored associations, not evidence that attention measures regional causal importance.

Sources: [E1 decision](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/region_tokens_v1/E1/decision_verified.json>) · [primary analysis](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/region_tokens_primary10_analysis_v2/analysis.json>) · [all exported tables](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/region_tokens_primary10_analysis_v2/tables>) · [mean-substitution smoke](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/region_tokens_e2_mean_smoke_v1/E2_mean_smoke/summary.md>) · [condition measurements](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/region_tokens_e4_polypgen_conditions_v1/study/conditions.jsonl>).


## 11. Additional local results and comparator archives

### Spatial-memory suppression and feasibility

| Diagnostic | Result | Scope |
|---|---|---|
| Distance-only full-budget disjoint matching | Boundary 0/2,141; mixed coverage 1,644/2,141 feasible frames | Includes 431 empty-mask frames |
| Same-side full-budget matching | Boundary 0/2,141; mixed coverage 1,347/2,141 | Not the later capped-subset estimator |
| All spatial keys suppressed, seq13 | Dice 0.18182 versus native 0.58210 | Pointers retained |
| All spatial keys suppressed, seq14 | Dice 0.19758 versus native 0.73858 | Pointers retained |

Infeasibility is not evidence that boundary tokens are useless. Spatial suppression
can destroy target identity and is not full memory-off.

### Archived MedSAM2 baselines

These five saved evaluations include all 23 sequences, including absent sequences.
Prompt schedules differ; this is an archive table, not a paired method ranking.
The target **0.77000** is equal-sequence Dice; pooled-frame Dice is **0.83000**.

| Saved evaluation | Sequences | Equal-sequence Dice | Pooled-frame Dice |
|---|---:|---:|---:|
| MedSAM2_anchor_masks_naive_memory | 23 | 0.70010 | 0.56398 |
| MedSAM2_anchor_masks_top_iou_memory | 23 | 0.68641 | 0.56350 |
| MedSAM2_default_data_boxes_naive_memory | 23 | 0.69548 | 0.55508 |
| MedSAM2_frame_manual_boxes | 23 | 0.92318 | 0.90960 |
| MedSAM2_frame_yolov8_boxes | 23 | 0.77000 | 0.83000 |

Sources: [default evaluations](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/MedSAM2_default>) · [frame-wise evaluations](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/MedSAM2_frame>). Per-frame manual boxes supply more supervision than a single-prompt video protocol.

### seq13/14 strategy diagnostics

All-sequence means are not substituted for these failure-selected pair results.

| Arm | seq13 Dice | seq14 Dice | Interpretation |
|---|---:|---:|---|
| Frame-wise YOLO–MedSAM2 | 0.84367 | 0.82398 | Detector-assisted reference |
| Video path without memory | 0.84440 | 0.81447 | Not a temporal-memory method |
| Native video memory | 0.43709 | 0.38104 | 242 / 248 memory calls |
| Fixed region + reliability, video path | 0.32762 | 0.34104 | No improvement over native memory |
| MedSAM2 detector refresh | 0.67976 | 0.67906 | Repeated detector prompting; not autonomous |
| Former learned region head | 0.43652 | 0.38163 | Two-update technical training only; implementation removed |
| Cutie, initial mask only | 0.20776 | 0.53268 | No later mask corrections |
| Cutie, detector-refresh masks | 0.86231 | 0.81244 | 177 / 193 later corrections, plus one initialization per sequence |

Cutie corrections use frozen frame-wise predicted masks; they are not new GT masks.
The refresh mean of 0.83738 therefore does not establish a region-aware memory gain.
The two-update learned head is not evidence of adequately trained performance.

### Small technical checks, excluded from population inference

| Saved check | Verified outcome |
|---|---|
| MedSAM2 zero-scale smoke, seq2/3 | All temporal zero-bias arms match native-arm scores; historical masks exact: seq2 12/12, seq3 10/12 (mean Dice drift 1.75e−7) |
| Nonzero fixed-bias smoke, seq2 | Native 0.90336; region+reliability 0.90438; region-only 0.90351 |
| Nonzero fixed-bias smoke, seq3 | Native 0.87928; region+reliability 0.87928; region-only 0.87990 |
| CPU zero-readout smoke, seq3 | Native = zero-bias Dice 0.87928345; 10/12 historical masks exact, mean Dice drift 1.75e−7 |
| CPU gradient smoke | Two steps; two nonzero-gradient steps; zero-initialized logits exact; not a completed training experiment |
| Cutie author smoke, seq3 | Three-frame prefix; initial-only and refresh Dice 0.93195; no later correction |

Sources: [region_boxes_video_path_diagnosis_v1/summary.json](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/region_boxes_video_path_diagnosis_v1/summary.json>) · [region_readout_seq13_14_verification_v1/summary.json](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/region_readout_seq13_14_verification_v1/summary.json>) · [medsam2_detector_refresh_seq13_14_v1/summary.json](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/medsam2_detector_refresh_seq13_14_v1/summary.json>) · [cutie_seq13_14_development_v1/summary.json](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/cutie_seq13_14_development_v1/summary.json>).


## 12. Efficient-SAM2 MPS port — technical results only

seq3 12-frame prefixes; 11 post-prompt evaluated frames for completed runs.
Dense and SMR columns are **raw means, not an accepted treatment comparison**.
v3/v4 use different dense/SMR attention kernels; v5 onward use 0%-pruning SMR
as the matched dense control. Repeat variation remains unresolved.

| Version | Dense Dice | Dense repeat Dice | SMR Dice | Validation outcome |
|---|---:|---:|---:|---|
| v1 | — | — | — | Failed before propagation: adapter reinitialization |
| v2 | — | — | — | Failed after six post-prompt records: missing native logging list; partial scores excluded |
| v3 | 0.87989 | — | 0.89118 | Execution only; repeatability not established |
| v4 | 0.88083 | — | 0.88695 | Five actual sparse reads/layer verified; kernel/repeat confounds |
| v5 | 0.88078 | 0.88189 | 0.88684 | MPS-resident state; repeat gate failed |
| v6 | 0.88064 | 0.88131 | 0.88618 | Deterministic algorithms enabled; repeat gate failed |
| v7 | 0.88226 | 0.88167 | — | Explicit Metal preference; repeat gate failed |

| Repeat condition | Identical post-prompt masks | Maximum frame Dice difference |
|---|---:|---:|
| v5 default backend | 0/11 | 0.00809352 |
| v6 deterministic algorithms | 0/11 | 0.00571291 |
| v7 Metal preference | 1/11 | 0.00517172 |

Native retained-mask observations: 36 layer/frame records per completed SMR smoke,
each retaining 204/4,096 positions; 12 overlays per smoke. The final frame's new
selection is unavailable after upstream cache clearing. These visualizations
show where selected tokens lie; they do not prove exclusive semantic encoding.

**Port status:** execution works; strict repeat/no-op gate fails. No full
Efficient-SAM2 run or accepted Dice-gain claim. Passing 123 project tests does
not override this runtime failure.

Sources: [repeat audit](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/efficient_sam2_mps_port_validation/validation.json>) · [native selections](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/efficient_sam2_mps_smr_validation_v6/retained_tokens.jsonl>) · [overlays](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/efficient_sam2_mps_smr_validation_v6/visuals>) · [W&B audit](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/lc5bjgno).


## 13. Assessment

| Question | Conclusion |
|---|---|
| Do regions differ in attention allocation? | Yes descriptively, n = 4; boundary enrichment highest |
| Does boundary help more than its control? | Clean-memory boundary-F signal; not established under natural memory |
| Is identity separated from proximity? | New disjoint subset comparisons control distance; no corrected significant regional effect. Earlier whole-region tables retain their confounds |
| Is uniform boundary boosting justified? | No |
| Continue the region-focused study? | Report completed diagnostics; current headroom does not meet the training gate. Defer method claims |
| Beat frame-wise YOLO–MedSAM2? | Not demonstrated |

## Evidence

- Completed follow-up: [detailed geometry/headroom tables](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/region_v2_local_2026_10_18_v1/geometry_headroom_2026_10_19_v1/analysis/results.md>) · [W&B analysis](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/8edo5p71). Section 9 uses this new estimator; earlier tables are unchanged.
- [Detailed tables](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/region_v2_local_2026_10_18_v1/regional_results_review_2026_10_19_v1/results.md>) · [statistics JSON](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/region_v2_local_2026_10_18_v1/regional_results_review_2026_10_19_v1/results.json>) · [W&B summary](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/xbfggk2s).
- [Seq15 numerical audit](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026/region_v2_local_2026_10_18_v1/attention_numerical_audit_v1/diagnosis.json>) · [audit tracking](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/5wgcleee) · [segmentation retry](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/ln308dck).
- Earlier comparators: [detector-assisted comparison](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/yhv3w20t) · [Cutie refresh](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/y4i19en4).
- Raw completed shards: counterfactual_seq3_retry2; counterfactual_seven; attention_seq11_v1–attention_seq14_v1; segmentation_seq15_v3. Failed attention runs are excluded; repeated seq3 intervention rows are not pooled.
- [Study methods and validation](REGION_TOKENS.md) summarize the implementation and validity limits. Kaggle outputs are preserved but excluded from chronological inference. Drive-synced paths still use local disk; filesystem presence does not verify cloud synchronization.
