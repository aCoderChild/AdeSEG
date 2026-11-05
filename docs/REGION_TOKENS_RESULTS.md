# Region-level memory: results

Updated **2026-11-05**. Detailed frame records and receipts are stored in Drive under `outputs/experiment/`. [Method and definitions](REGION_TOKENS.md).

## Summary

- No completed regional method has shown a reliable, meaningful Dice gain over its matched native SAM2 baseline.
- Among the newer trainable CVs, the largest native gain was **+0.008575 Dice** (Run 119); it did not beat the global-role control and added false alarms. Earlier sparse-history C2 gained +0.01935 over native, but its interval crossed zero and Holm p=1.000.
- Region labels are reproducible positional groups of the model's own predicted mask. Boundary tokens receive high attention per token, but exclusion studies do not show boundary indispensability.
- Run 155's interim hard-ToMe mean is +0.03073 Dice across 16/21 matched sequences, but the median paired gain is +0.00017; excluding seq8, the mean gain is +0.00121. It has 8 wins and 8 losses against native, with more pooled false alarms; this is not reliable meaningful-improvement evidence.
- Run 150, using the official SAM2.1 Hiera-Tiny checkpoint with all original SAM2 weights frozen, is still running. Its completed Fold 1 is negative for the regional arm; Fold 2 regional training is at update 350/560 (epoch 24, seq23) as of 2026-11-05 12:43 +07.

PolypGen has informed repeated development, so these are exploratory results. Sequences 1 and 7 have no eligible first-frame prompt and are excluded. Patient-to-sequence independence is unverified.

## Cohort and metric

The official native run covers **21 sequences, 2,012 post-prompt targets**: **1,689 GT-present** and **323 GT-empty**. Present-frame Dice is averaged within sequence, then sequences receive equal weight. False alarms count nonempty predictions on GT-empty targets. CV uncertainty is reported as a paired sequence-bootstrap 95% interval; planned paired tests use Wilcoxon with Holm correction.

### Native references

These checkpoints are different and their scores must not be pooled.

| Reference | Scope | Dice | IoU | Boundary F | False alarms / empty targets |
|---|---|---:|---:|---:|---:|
| Official SAM2.1 Hiera-Tiny, CPU FP32 | 21 sequences | 0.710224 | 0.651707 | 0.616994 | 194/323 |
| MedSAM2 Hiera-Tiny | Native reference for Runs 119, 127, 134 and 139 | 0.536615 | 0.478123 | 0.422339 | 164/323 |

Source: Drive `143_official_native_feature_audit_v1/full/native_results.json` and each CV run's `results.json`.

## Completed trainable regional CV

All four runs use the same MedSAM2 Hiera-Tiny native reference and 21 sequence-grouped validation sequences. The global arm is a matched control that removes token-level regional assignments.

| Run / method | Native Dice | Global control | Regional | Δ vs native | Δ vs global | False alarms N/G/R |
|---|---:|---:|---:|---:|---:|---:|
| 119 · SimVOS regional prototypes, seed 2 | 0.536615 | 0.546151 | 0.545190 | +0.008575 | −0.000961 | 164/169/188 |
| 134 · Region-conditioned low-rank adapter | 0.536615 | 0.539666 | 0.539822 | +0.003207 | +0.000156 | 164/174/177 |
| 139 · Same adapter with native-history training | 0.536615 | 0.539967 | 0.540171 | +0.003556 | +0.000204 | 164/179/183 |
| 127 · Prediction-conditioned regional affinity | 0.536615 | 0.536785 | 0.539585 | +0.002970 | +0.002801 | 164/164/165 |

| Run | 95% CI, regional − native | Holm p, native | 95% CI, regional − global | Holm p, global | Wins/losses/ties vs native | Screen |
|---|---|---:|---|---:|---:|---|
| 119 | [−0.001714, 0.020102] | 0.012982 | [−0.013184, 0.009043] | 0.083556 | 15/3/3 | Fail |
| 134 | [−0.000129, 0.009452] | 0.248466 | [0.000015, 0.000326] | 0.035241 | 10/8/3 | Fail |
| 139 | [0.000024, 0.009932] | 0.244191 | [−0.000058, 0.000511] | 0.244191 | 13/5/3 | Fail |
| 127 | [−0.000032, 0.008956] | 0.944798 | [−0.000534, 0.008957] | 0.944798 | 9/9/3 | Fail |

The Holm values use the two planned regional contrasts per run. The positive but tiny Run 134 regional-versus-global result is **0.000156 Dice**; its gain over native is uncertain. Run 139's native bootstrap interval barely excludes zero, but its corrected paired test does not. Every regional arm added false alarms over native: +24, +13, +19 and +1, respectively.

All four frozen meaningful-improvement screens failed. Gains are small or concentrated in a few sequences, and none shows a reliable advantage over the matched global control. Saved results: Drive `119_simvos_region_prototypes_polypgen_cv_seed2_v1/`, `127_prediction_conditioned_affinity_corrected_polypgen_cv_v1/`, `134_regional_low_rank_polypgen_cv_v1/`, and `139_regional_low_rank_native_history_polypgen_cv_v1/`.

## Current official-SAM2 comparison — Run 150

Run 150 uses the official SAM2.1 Hiera-Tiny checkpoint, keeps all pretrained SAM2 weights frozen, and learns three regional TokenLearners (29,670 added parameters). Native dense memory and frame selection remain unchanged. The prespecified control globalizes region roles while matching training, parameters and token counts. [Frozen protocol](../protocols/regional_memory_prompts_cv_v1.json).

**Fold 1 is complete (7 sequences); full CV is not.** Both arms' training and all 14 validation sequence receipts/frame records passed the read-only audit. Fold 1 is descriptive only; no full-cohort inference is available.

| Fold 1 arm | Mean present Dice | Δ vs native | False alarms / 239 empty targets | Wins vs native |
|---|---:|---:|---:|---:|
| Native | 0.605524 | — | 138 | — |
| Global-role control | 0.673627 | +0.068104 | 146 | 4/7 |
| Regional | 0.583174 | −0.022350 | 143 | 2/7 |

| Sequence | Native | Global | Regional | Regional − native |
|---|---:|---:|---:|---:|
| seq2 | 0.888730 | 0.888721 | 0.893299 | +0.004568 |
| seq5 | 0.541987 | 0.541999 | 0.541884 | −0.000102 |
| seq8 | 0.279446 | 0.756405 | 0.240082 | −0.039363 |
| seq11 | 0.280751 | 0.280862 | 0.280656 | −0.000095 |
| seq14 | 0.839942 | 0.839537 | 0.720632 | −0.119310 |
| seq18 | 0.639817 | 0.639890 | 0.635199 | −0.004618 |
| seq21 | 0.767995 | 0.767978 | 0.770465 | +0.002471 |

The largest losses are seq8 and seq14. This is an interim warning, not the final CV result. As of **2026-11-05 10:22 +07**, the live continuation was at Fold 2 regional training, update **189/560** (epoch 13, seq13). Progress and checkpoints remain in Drive `150_regional_memory_prompts_official_cv_v1/`.

The frozen pass screen requires regional Dice at least +0.01 over native, positive regional-minus-global Dice, at least 14/21 sequence wins over native, positive lower bounds for both paired 95% intervals, Holm-adjusted p<0.05 for both contrasts, and no increase in pooled false alarms. Do not change the method based on these partial results.

## Region assignment and attention diagnostics

Labels come from SAM2's stored predicted mask; they describe token positions, not semantically pure vectors. The audit covers **2,033 stored frames**: stored-mask/PNG checks pass, prompt-frame low-resolution masks match the SAM2 prompt path bit-for-bit, and all 2,033 low-resolution-to-memory-input sign checks have zero disagreement. Fresh-model native replay was separately checked on seq2/seq3: **73/73 targets bit-identical**.

### Attention enrichment

Exponentiated median sequence log pooled enrichment; all four layers and all query regions, **N=21**. Values above 1 mean greater attention per token than the region's spatial-token prevalence; they do not measure total mass or causal importance.

| Region definition | Background | Interior | Boundary |
|---|---:|---:|---:|
| Mixed cells | 0.702 | 3.559 | 9.339 |
| Mask Transfiner incoherence | 0.661 | 2.565 | 7.281 |
| MatteFormer trimap k15 | 0.690 | 2.948 | 7.630 |
| MatteFormer trimap k21 | 0.666 | 2.581 | 6.908 |
| MatteFormer trimap k29 | 0.652 | 2.354 | 6.057 |
| Boundary IoU inner band | 0.702 | 2.049 | 7.074 |

Boundary tokens are enriched under all six definitions. This is descriptive allocation, not proof that boundary vectors encode only boundary content or are necessary for accurate segmentation.

### One-step exclusion diagnostic

Matched-control excess Dice loss = control Dice minus Dice after excluding a region. These interventions do not write altered predictions back into memory.

| Excluded region | Excess Dice loss | 95% CI | Holm p |
|---|---:|---:|---:|
| Background | +0.2979 | [0.2134, 0.3771] | 0.000172 |
| Interior | +0.0445 | [0.0261, 0.0643] | 0.007189 |
| Boundary | −0.0005 | [−0.0051, 0.0046] | 1.000000 |

Across six region definitions, no region passed the frozen all-definition claim rule. Boundary exclusion's near-zero immediate effect does not establish equivalence across every frame or long-term propagation.

### Boundary geometry

| Definition | Contour coverage | Boundary-footprint precision |
|---|---:|---:|
| Mixed cells | 83.72% | 29.61% |
| Mask Transfiner incoherence | 96.81% | 16.49% |
| Boundary IoU inner band | 94.10% | 15.37% |

Mixed cells provide a more compact contour footprint under this audit, but these measures do not prove semantic purity or identify an optimal boundary definition.

## Earlier region-sparse memory comparison

This is a separate earlier protocol and baseline; do not combine its scores with the newer MedSAM2 or official SAM2 CVs.

| Method | Present Dice | Paired result |
|---|---:|---|
| Native SAM2 | 0.71904 | Reference |
| Dense longer history (C1) | 0.69114 | Δ−0.02789; 95% CI [−0.11006, 0.05235] |
| Region-sparse longer history (C2) | 0.73838 | Δ+0.01935 vs native; CI [−0.02309, 0.07282]; Holm p=1.000 |
| Matched random-token control (C3 mean) | 0.70908 | C2−C3 +0.02930; CI [−0.01270, 0.08372]; Holm p=1.000 |
| SAMURAI | 0.76652 | Reference |
| SAMURAI + regional sparse history | 0.75258 | Δ−0.01394; CI [−0.04442, 0.01697]; p=0.109 |
| SAM2Long | 0.74253 | Reference |
| SAM2Long + regional sparse history | 0.73640 | Δ−0.00613; CI [−0.04715, 0.03548]; p=0.191 |
| Lean-SAM2 author run | 0.73446 | Separate author benchmark |

C2 won/lost/tied versus native on 8/12/1 sequences. Its mean gain was not statistically established; adding regional sparse history did not improve SAMURAI or SAM2Long. Drive records: `23_long_history_E3_screen_v1/`, `24_region_sparse_C2_C3_polypgen_v1/`, and `25_author_region_sparse_polypgen_cpu_v1/`.

## Region-constrained ToMe: engineering gate and full run

This exploratory method reuses ToMe's bipartite token-merging operation, constrained to local 4×4 windows and the model-predicted foreground, boundary, or background role. It merges positioned memory keys/values at read time; SAM2 weights, memory writes, frame selection, and pointer tokens remain unchanged. The generic local-ToMe control uses the same per-window merge budget without using region roles. [Frozen protocol](../protocols/region_constrained_tome_polypgen_v1.json).

The seq2/seq3 engineering gate **passed**: all 73 native replays matched archived outputs bit-for-bit; all four memory-attention layers were intercepted; pointer keys and model parameters were unchanged; attention mass was conserved; and regional and generic arms matched spatial-token budgets across 292 target-layer reads. Mean spatial-key retention was about 50%. This verifies the pipeline, not benefit.

| Gate sequence | Native Dice | Region-constrained ToMe | Generic local ToMe | Native / regional seconds |
|---|---:|---:|---:|---:|
| seq2 | 0.8887 | 0.8898 | 0.8833 | 91.3 / 168.8 |
| seq3 | 0.8746 | 0.8721 | 0.8734 | 19.9 / 35.4 |

The two-sequence Dice differences are mixed and cannot establish efficacy. The modified CPU reads were slower in both gate sequences (about 1.8× native on these cases), a material limitation to report. No code or protocol tuning will be based on these outcomes.

Seq11 is complete in all arms. This is an interim single-sequence result: present Dice was **0.28075 native, 0.28110 region-constrained ToMe, and 0.28039 generic local ToMe**; all three produced false alarms on 90/90 GT-empty targets. Regional ToMe took 695 s versus 240 s for native. The +0.00035 Dice difference versus native is descriptive and does not establish efficacy.

### Interim paired snapshot (16/21 sequences)

Only sequences with complete receipts for all three arms are included. Values are per-sequence present-frame Dice; FP is false alarms / GT-empty targets. Mean Dice is equal-weighted by sequence; FP totals are pooled. The frame-record audit verifies that all three arms use the same native reference hashes, target indices, and GT-presence labels. Summary Dice and false-alarm counts reproduce from the saved frame records.

| Sequence | Post-prompt targets | Native | Region ToMe | Δ vs native | Generic ToMe | Δ vs native | FP native | FP region | FP generic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seq2 | 59 | 0.8887 | 0.8898 | +0.0011 | 0.8833 | −0.0055 | 0/0 | 0/0 | 0/0 |
| seq3 | 14 | 0.8746 | 0.8721 | −0.0025 | 0.8734 | −0.0012 | 0/0 | 0/0 | 0/0 |
| seq4 | 47 | 0.8142 | 0.8161 | +0.0019 | 0.8164 | +0.0021 | 2/2 | 2/2 | 2/2 |
| seq5 | 249 | 0.5420 | 0.5411 | −0.0009 | 0.5412 | −0.0007 | 1/51 | 1/51 | 2/51 |
| seq6 | 66 | 0.8647 | 0.8602 | −0.0045 | 0.8615 | −0.0033 | 5/5 | 5/5 | 5/5 |
| seq8 | 72 | 0.2794 | 0.7530 | +0.4735 | 0.2985 | +0.0191 | 4/19 | 12/19 | 4/19 |
| seq9 | 47 | 0.9067 | 0.9037 | −0.0030 | 0.9033 | −0.0035 | 0/6 | 0/6 | 0/6 |
| seq10 | 7 | 0.9367 | 0.9704 | +0.0337 | 0.9701 | +0.0334 | 1/1 | 1/1 | 1/1 |
| seq11 | 225 | 0.2808 | 0.2811 | +0.0004 | 0.2804 | −0.0004 | 90/90 | 90/90 | 90/90 |
| seq12 | 249 | 0.9444 | 0.9422 | −0.0023 | 0.9410 | −0.0035 | 0/0 | 0/0 | 0/0 |
| seq13 | 242 | 0.7699 | 0.7699 | −0.00002 | 0.7698 | −0.0001 | 42/44 | 42/44 | 42/44 |
| seq14 | 248 | 0.8399 | 0.8382 | −0.0017 | 0.8376 | −0.0023 | 32/49 | 32/49 | 32/49 |
| seq15 | 115 | 0.9046 | 0.9053 | +0.0007 | 0.9059 | +0.0013 | 0/0 | 0/0 | 0/0 |
| seq16 | 33 | 0.9757 | 0.9761 | +0.0004 | 0.9757 | −0.00003 | 1/1 | 1/1 | 1/1 |
| seq17 | 46 | 0.8171 | 0.8257 | +0.0086 | 0.8216 | +0.0045 | 3/3 | 1/3 | 3/3 |
| seq18 | 62 | 0.6398 | 0.6262 | −0.0136 | 0.6361 | −0.0037 | 2/7 | 2/7 | 2/7 |
| **Mean Dice (16 sequences)** | — | **0.7675** | **0.7982** | **+0.0307** | **0.7697** | **+0.0023** | **183/278** | **189/278** | **184/278** |

The interim mean remains highly sensitive to seq8: region ToMe's mean gain over native is +0.03073, while its median paired gain is +0.00017 (8 wins, 8 losses); omitting seq8 reduces the mean gain to +0.00121. Its mean advantage over generic ToMe is +0.02846, but falls to +0.00006 without seq8. On seq8, regional ToMe increases false alarms from 4/19 native to 12/19; generic ToMe remains at 4/19. Across all 16 sequences, pooled false alarms are 189/278 for regional ToMe, 184/278 for generic ToMe, and 183/278 for native. Seq18 also declines by 0.0136 Dice under regional ToMe. This is not evidence of a stable, meaningful cohort-wide Dice gain, and the full-cohort screen has not been evaluated. A frame-level seq8 check shows the source of its Dice outlier: on 29 GT-present targets where native Dice is 0, regional ToMe returns nonempty masks (mean Dice 0.757 on those targets), while generic ToMe remains at zero. On GT-empty targets, regional ToMe adds eight false alarms relative to native; five occur in frames 45–50. The seq8 signal therefore suggests a possible memory-persistence/recall effect paired with more false positives, not a general boundary-quality improvement.

Across the 16 completed sequences, the per-read audit covers **7,124 layer reads** (four layers per target). Regional and generic ToMe have identical spatial-key counts at each target and layer, matched memory-frame lists, equal pointer counts, and conserved spatial key mass; they retain **50.25%** of the dense spatial keys in aggregate. The original SAM2 parameter digest is unchanged. Recorded CPU wall times are 1,914.2 s native, 5,455.7 s region ToMe and 5,214.4 s generic ToMe, so this implementation currently incurs substantial runtime overhead despite shortening attention keys. These are interim measurements from the current pipeline, not a speed benchmark.

The full 21-sequence, 2,012-target run is **currently running** on Drive at `155_region_constrained_tome_polypgen_v1/full/`, with native, region-constrained ToMe, and matched generic local-ToMe arms. As of **2026-11-05 12:43 +07**, all three arms have complete matched receipts for seq2–seq6 and seq8–seq18 (16 sequences). Native has completed seq19 and region ToMe is at frame 47/55 for seq19; generic seq19 and seq20–seq23 remain. The results above were recomputed from Drive frame records and checked for matching references, labels, per-frame-to-summary metrics, budgets, pointers, and mass conservation.

## Continuous mask-aware ToMe candidate

The hard three-role ToMe gate passed, but its two-sequence Dice screen was mixed and does not show efficacy. I therefore implemented a separate, prospective soft-occupancy variant that still tests region-level memory-token use: each selected stored mask becomes continuous 64×64 foreground occupancy; ToMe matches keys using an equal blend of post-RoPE appearance and occupancy similarity, inside fixed 4×4 windows. It merges 8 of 16 tokens in every window. The generic appearance-only control gets the identical per-window budget. No hard region labels, thresholds, frame selection, or SAM2 parameter changes are used.

**Pre-gate alignment correction (2026-11-05).** A source audit found that the first draft used SAM2's separate 256×256 `pred_masks` field, while SAM2 forms each memory write from `pred_masks_high_res`. Before the soft-method gate started, the implementation and protocol were amended to use each frame's stored high-resolution logits, apply sigmoid, and area-pool onto that frame's actual `maskmem_features` grid. The gate now checks the source field, mask dimensions, pooling footprint, and key-grid count. Occupancy is raw predicted probability before SAM2's configured memory-mask scale/bias. This correction was made before any soft-method gate or efficacy output was inspected; the similarity weight, window, and merge budget are unchanged.

**Prior-work fit and limitation.** The implementation reuses the pinned [ToMe](https://arxiv.org/abs/2210.09461) matching and weighted-merge functions. ToMe is primarily an efficiency method; it does not justify expecting better Dice than dense SAM2. [ALGM](https://openaccess.thecvf.com/content/CVPR2024/papers/Norouzi_ALGM_Adaptive_Local-then-Global_Token_Merging_for_Efficient_Semantic_Segmentation_with_Plain_Vision_Transformers_CVPR_2024_paper.pdf) is a closer segmentation precedent: it merges locally and then globally in a Segmenter image encoder and reports that some settings improve mIoU while reducing compute. We cloned and inspected its official [Segmenter implementation](https://github.com/tue-mps/algm-segmenter) at `external/ALGM` (`674eaa2`). Its encoder-specific patch and adaptive similarity threshold are not copied: neither matches SAM2's already-positioned temporal memory read or the frozen no-threshold protocol.

[StructSAM](https://arxiv.org/abs/2603.07307) studies boundary-preserving merging in SAM image encoders, using feature-gradient/flatness screening and a merge-unmerge path. This motivates protecting structural distinctions, but its placement and thresholded screening differ from this memory-attention test; the paper does not provide an official SAM2-memory implementation. [MaskMerge](https://arxiv.org/abs/2610.22921) is a recent preprint using coarse predicted masks to constrain merges for image instance segmentation. Its central idea is relevant, but the task differs and we did not locate an official repository to reuse. These works motivate a test of predicted-mask structure; none establishes a Dice gain in frozen SAM2 video memory.

The frozen test asks whether continuous predicted-mask occupancy makes a fixed local merge budget more segmentation-preserving than appearance-only merging and dense native SAM2. It is an exploratory transfer of published merge operations, not a reproduction of ALGM, StructSAM, or MaskMerge. One explicit approximation is that ToMe averages locally different keys after RoPE, so the merged key does not retain one exact positional phase; the fixed 4×4 window limits, but does not eliminate, this effect. A gate pass is implementation evidence only; the precommitted full-cohort screen determines whether the Dice hypothesis survives.

| Check | Status |
|---|---|
| Protocol/code freeze | Local commit `81af0911` records the pre-gate mask-source correction and implementation |
| Upstream ToMe revision and source | Pinned and unchanged (`af95e4b1`; source integrity check passed) |
| Focused candidate tests | 9 passed across the soft-ToMe module and full-run budget checks after the pre-gate alignment correction |
| Gate runner and protocol | Implemented in `protocols/soft_mask_aware_tome_polypgen_v1.json`; settings fixed before gate |
| Soft-method seq2/seq3 gate | Not run yet; queue 156 is active and waiting for Run 155 to exit, to avoid CPU contention (checked 2026-11-05 12:20 +07) |
| Full-cohort runner and queue | Implemented; queued to launch only after a passing soft-method gate |
| Full-cohort efficacy run | Not started; requires this variant's gate to pass |

The gate will compare native SAM2, generic local ToMe, and soft mask-aware ToMe on seq2/seq3. It checks native replay against archived outputs, all four memory-attention layers, stored-mask-to-key-span alignment, matched budgets, pointer preservation, mass conservation, and frozen parameters. Passing establishes engineering integrity only; Dice benefit remains unknown. The gate runner writes to Drive `156_soft_mask_aware_tome_polypgen_v1/gate/`; the full runner is queued for `157_soft_mask_aware_tome_polypgen_v1/full/` only after the gate passes. Queue state is saved in Drive `queue.json` files. The prospective screen is frozen in the protocol and will be applied once; a failed screen will be recorded without retuning on PolypGen.

## External-data status and next step

DAVIS 2017 validation data are prepared (30 videos, 61 objects, 1,999 frames) and the official evaluator self-test passes. Run 151 passes a **three-frame, two-object engineering check** on DAVIS training sequence `bmx-bumps`; this is not a benchmark score. SA-V is not downloaded, and no new-method DAVIS or SA-V scores exist.

Finish Run 150 and independently audit its full results. If it fails, report that result and diagnose the largest errors before freezing a materially different method; do not tune against these already-used PolypGen sequences. The current evidence supports region-dependent attention, but **does not yet support the claim that regional memory utilization meaningfully improves Dice**.
