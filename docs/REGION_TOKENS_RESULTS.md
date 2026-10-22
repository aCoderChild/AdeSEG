# Regional memory tokens: results

Updated 2026-10-23. Exploratory PolypGen study: 21 promptable sequences,
2,012 post-prompt targets; seq1/seq7 are N/A. Dice is the equal-sequence
present-frame mean. See [methods](REGION_TOKENS.md) for controls and definitions.

> **Do not combine definitions.** Historical regional results in sections 2–5
> use superseded v1 (any band pixel or mixed footprint → boundary). The new audit
> uses native-mask mixed cells and separate comparators. Author benchmark Dice
> in section 1 is unaffected; historical regional results have not been rerun
> with the new primary definition.

## Current native-grid audit — 2026-10-23

**2026-10-23 width amendment:** the original run retains k3/k7/k15. A separate,
exploratory comparison uses **k15/k21/k29**, with k3/k7 retained as sub-token
sampling diagnostics outside the amended robustness rule. Original results and
claim rule are not overwritten. The supplied 89-cell example is illustrative,
not independently reproduced. See the [amendment](../protocols/region_definition_trimap_amendment_2026_10_23.json).
New-width attention has not been measured; saved logits permit geometry-only
relabeling, not reconstruction of regional attention.

Follow-up audit implementation: contour-miss fractions (self and GT, with/without
image border), frozen source-frame size quartiles, missing-group counts and
all-source-frame coverage are now included. Primary attribution is Efficient-SAM2
occupancy code with the mixed-region motivation described in the
[methods](REGION_TOKENS.md), not an exact Mask Transfiner reproduction.
**108 tests pass.** The fresh gate
passed: 73/73 replay logits bit-identical, 75/75 stored-mask checks, 292 attention
reads normalized (maximum mass error 0.00000718), and confirmed W&B delivery.
The full 23-sequence CPU audit has been launched; it is not complete.
The earlier pilot is recorded separately below.

New gate only: pooled one-pixel native256 contour misses, 73 post-prompt frames.
These measure geometry, not semantic vector contents or causal usefulness.

| Definition | Own-prediction contour missed | GT contour missed |
|---|---:|---:|
| Exact mixed cells (primary) | 17.64% | 36.82% |
| Author incoherence adapter | 1.14% | 16.51% |
| MatteFormer kernel 3 | 82.65% | 86.76% |
| MatteFormer kernel 7 | 60.22% | 68.56% |
| MatteFormer kernel 15 | 23.23% | 38.80% |
| Inner band 0.02 | 6.17% | 26.21% |

Thin-trimap nearest-neighbor sampling misses many contour pixels. Mixed cells
also leave contour gaps; do not call them a complete contour detector. GT misses
add prediction error. Full-cohort size/empty-group statistics remain pending;
no cross-definition robustness or segmentation improvement is established.

New outputs: `15_region_definition_contour_audit_cpu_v1/{validation,full}`.
[Passed gate](https://forge.coreweave.com/api/wandb/adenoid-hypertrophy/ade-seg/runs/kmqukgz0).
[Running full audit](https://forge.coreweave.com/api/wandb/adenoid-hypertrophy/ade-seg/runs/oq7ekeyo).

### Verification scope and protocol timing

| Check | Verified scope / result |
|---|---|
| 73/73 exact replays | Seq2: 59 targets (frames 4–62); seq3: 14 targets (frames 1–14), not the full cohort |
| 75 stored frames | Seq2: 60 (frames 3–62); seq3: 15 (frames 0–14), including two conditioning frames |
| Prompt low-res logits | Rechecked bit-identical on seq2 frame 3 and seq3 frame 0 |
| Primary native-versus-PNG labels, mean / maximum token disagreement | Seq2: 0.0836% / 0.3174%; seq3: 0.1500% / 0.3662%; predicted frames only |
| PNG decision | Close, not identical; the running audit already uses native logits, not PNG-derived labels |
| Size bins | Per-frame GT area fraction; positive-frame quartiles 2.034683%, 5.384635%, 11.534966%; saved before new-run inference |
| Freeze timing | Protocol/source SHA snapshots preceded inference; Git commit did not. Numeric bins were in `size_policy.json`, added to a companion protocol record after launch |

These are gate checks, not full-cohort efficacy results. Historical results had
already been seen; do not describe this audit as prospectively preregistered.
The running protocol remains byte-identical to its pre-inference snapshot.
See [protocol clarification](../protocols/region_definition_audit_v1_record.json)
and `receipt_verification/verification.json` under the new output root.

### Earlier definition pilot — output 14

These earlier seq2/seq3 statistics are retained as recorded, not results from
the ongoing full audit. The older v3 PNG comparison includes conditioning frames;
the primary PNG comparison above uses predicted frames only.

Primary candidate: exact mixed-cell occupancy from native stored 256×256 logits,
using the unchanged Efficient-SAM2 window selector on foreground and background.
Comparators: author Mask Transfiner incoherence, MatteFormer kernels 3/7/15,
and direct-native v3 inner bands. No model component or parameter was replaced.

| CPU seq2/seq3 verification | Result |
|---|---|
| Tests | 104 passed; new files lint passed |
| Complete fresh-model no-observer replay | 73/73 post-prompt logits bit-identical |
| Prompt masks / stored-frame masks / actual writer association | Passed on both sequences |
| Attention mass | Sums to one in all 292 layer reads |
| GT image-edge contact, including prompt | Seq2: 2/60 frames; seq3: 1/15 |
| Native-v3 versus PNG-route disagreement, mean / maximum | Seq2: 0.045% / 0.244%; seq3: 0.083% / 0.293% |
| Empty primary interior / boundary, post-prompt | 0/0 on both sequences; not a full-cohort estimate |

Square-input overlays were inspected: labels align with the stored prediction;
GT disagreements remain where segmentation is inaccurate. The absent pre-prompt
example is explicitly reference-only. Mixed cells do not cover contours lying
exactly between pure cells; they are not semantically pure vectors. No full
attention/exclusion rerun or improvement is claimed by this pilot.

Output: `14_region_definition_comparison_cpu_v1` in the Drive output root below.
[Pilot tracking](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/waa5c6i9).

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

Last verified regression suite: **108 passed**. Efficient-SAM2 SMR-only executed
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
| 14 | `14_region_definition_comparison_cpu_v1` | Earlier native-grid definition pilot; seq2/seq3 only |
| 15 | `15_region_definition_contour_audit_cpu_v1` | Passed gate, post-launch verification and ongoing full native-grid audit |

[W&B attention summary](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/e6dxrwgm)
· [W&B weighting summary](https://forge.coreweave.com/wandb/adenoid-hypertrophy/ade-seg/runs/sxurya29).
