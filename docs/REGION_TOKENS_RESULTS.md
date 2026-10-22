# Regional memory tokens: results

Updated 2026-10-22. Exploratory PolypGen study: 21 promptable sequences,
2,012 post-prompt targets; seq1/seq7 are N/A. Dice is the equal-sequence
present-frame mean. See [methods](REGION_TOKENS.md) for controls and definitions.

> **Assignment v1.** Every regional number below uses the superseded v1 token
> rule (any band pixel or mixed footprint → boundary). The code now uses v3,
> composed of unchanged SAM2, Efficient-SAM2 and Boundary IoU code. Author
> benchmark Dice in section 1 is unaffected; sections 2–5 need rerunning.

## 1. Author benchmarks — completed

| Method, CPU FP32 | Dice |
|---|---:|
| Native SAM2 | 0.71904 |
| SAMURAI | 0.76652 |
| SAM2Long | 0.74253 |
| Lean-SAM2 | 0.73446 |

None of the recorded author-versus-SAM2 comparisons passes Holm correction.
SAM2Long uses offline final-path selection, not equivalent online information.

## 2. Regional attention — completed

Read-only SAM2/SAMURAI/SAM2Long measurements preserve native predictions.
All full runs and the final comparison have confirmed W&B artifact uploads.

Final memory-attention layer (layer 3), equal-sequence means:

| Method | Background mass | Interior mass | Boundary mass | Pointer mass |
|---|---:|---:|---:|---:|
| SAM2 | 0.25929 | 0.08420 | 0.63417 | 0.02235 |
| SAMURAI | 0.26204 | 0.08789 | 0.62645 | 0.02362 |
| SAM2Long | 0.25704 | 0.08457 | 0.63650 | 0.02189 |

| Method | Background enrichment | Interior enrichment | Boundary enrichment [95% CI] |
|---|---:|---:|---|
| SAM2 | 0.309 | 6.837 | 20.856 [8.767, 37.785] |
| SAMURAI | 0.317 | 6.351 | 8.276 [6.649, 10.208] |
| SAM2Long | 0.306 | 6.625 | 9.812 [6.954, 13.724] |

Foreground enrichment uses 20 sequences for SAM2/SAM2Long and 21 for SAMURAI;
empty groups are excluded. Enrichment is averaged per read, not calculated by
dividing aggregate means. SAM2Long covers all executed branch reads, not only
the final path. Full per-sequence/four-layer statistics remain in source report 12.

Interpretation: late layers favor boundary-associated positions, while layer 1
is background-dominated. Higher mean boundary enrichment does not track higher
Dice across these methods; this is descriptive, not a causal comparison.

## 3. Fixed enrichment weighting — completed

Background/interior/boundary weights: **0.61878693 / 3.42351487 / 10.34936733**.
Inverse uses reciprocals; random uses the same weights with shuffled labels.
Ratios were calibrated and evaluated on the same full eligible cohort.

| Condition | Dice | Delta vs native | 95% paired sequence CI | Holm p |
|---|---:|---:|---|---:|
| Native SAM2 | 0.71904 | — | — | — |
| Proportional | 0.65216 | −0.06688 | [−0.15106, 0.01330] | 0.18586 |
| Inverse | 0.62267 | −0.09637 | [−0.17870, −0.02104] | 0.18397 |
| Proportional random | 0.73277 | +0.01374 | [−0.01005, 0.05620] | 0.18586 |

Proportional − random: −0.08061, CI [−0.16017, −0.00504], Holm p=0.18586.
Holm covers four contrasts. Bootstrap mean CIs and signed-rank tests target
different quantities. No corrected test establishes an improvement; mean
enrichment is not a demonstrated useful weighting policy.

Superseded development-only ratios gave Dice 0.66677 / 0.63401 / 0.71524
for proportional / inverse / random. These are separate experiments (source 09).
The earlier fixed ×1.1 boundary rule is retired.

## 4. Regional exclusion — completed, earlier MPS diagnostic

Fixed-bank, single-target interventions; no memory writeback. Baseline Dice
**0.72032** belongs to the earlier bundled predictor, not the CPU benchmark above.
Random controls match excluded counts within each stored memory frame.

| Excluded region | Dice | Random-control Dice | Excess loss [95% CI] | Holm p |
|---|---:|---:|---|---:|
| Background | 0.58428 | 0.72401 | 0.13973 [0.07100, 0.21381] | 0.000878 |
| Interior | 0.71887 | 0.72087 | 0.00200 [−0.00290, 0.00704] | 0.270714 |
| Boundary | 0.69459 | 0.72047 | 0.02588 [0.00114, 0.05272] | 0.270714 |

Background removal produces the strongest count-matched harm. Boundary harm
does not pass the corrected signed-rank test. Deletion effects do not establish
that boosting a region improves full-rollout Dice; controls do not match contiguity.

## 5. Assignment and sensitivity — completed

Full audit: 23 sequences / 2,225 source frames; 2,033 native memory frames
including prompts; 192 reference-only frames; 12,774 overlays pixel-verified.
At width 0.02, GT/predicted assignment IoU: background **0.92538**, interior
**0.55729**, boundary **0.58845**. These are spatial agreement, not classifier scores.

Earlier native MPS final-layer boundary enrichment:

| Band ratio | Predicted labels | GT-reference labels |
|---|---:|---:|
| 0.01 | 29.050 | 14.753 |
| 0.02 | 20.820 | 10.736 |
| 0.04 | 12.046 | 6.964 |

Boundary preference persists, but magnitude depends on region definition.
GT supplies offline labels only; spatial grouping does not imply semantic purity.

## Status and sources

Last verified regression suite: **92 passed**. Efficient-SAM2 SMR-only executed
on MPS with dense-encoder parity and active pruning, but repeated logits/masks
differed. Full SMR accuracy remains blocked. Author-specific exclusions and
stored-mask-quality analyses remain pending; no region-aware improvement is established.

Detailed tables, masks, overlays and provenance are retained under the
[Drive output root](</Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/16:10:2026>).
Mounted storage is verified; independent cloud synchronization is not.

| Source | Folder under output root | Main evidence |
|---|---|---|
| 03 | `03_memory_token_region_validation_polypgen_v1` | Assignment validation |
| 04 | `04_memory_region_attention_polypgen_v1` | Original attention and overlays |
| 05 | `05_memory_region_attention_sensitivity_polypgen_v1` | Width/GT sensitivity |
| 06 | `06_memory_region_exclusions_polypgen_v1` | Exclusion tables |
| 08 | `08_region_readout_polypgen_v1/serial_release_v1` | Author benchmarks |
| 09 | `09_enrichment_readout_polypgen_v1` | Superseded development-only weighting |
| 10 | `10_efficient_smr_polypgen_mps_v1` | Failed MPS replay diagnostics |
| 11 | `11_all_sequence_enrichment_polypgen_cpu_v1` | Full-cohort weighting comparison |
| 12 | `12_author_region_attention_polypgen_cpu_v1` | Cross-method attention comparison |

[W&B attention summary](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/e6dxrwgm)
· [W&B weighting summary](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/sxurya29).
