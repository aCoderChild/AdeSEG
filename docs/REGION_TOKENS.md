# Regional memory tokens: methods

## Question

How does attention to foreground-interior, boundary-associated and background
memory tokens relate to downstream endoscopic segmentation?
Spatial grouping does not imply semantically pure feature vectors.

## Setup

- PolypGen: 23 sequences; 21 promptable, 2,012 post-prompt targets.
  Seq1/seq7 have no foreground prompt and receive N/A propagation scores.
- SAM2.1 Hiera-B+ checkpoint; one GT mask at the first labeled foreground frame.
  Forward propagation, no corrections or detectors; prompt excluded from scores.
- Matched author benchmarks use CPU FP32, native BF16 memory and no postprocessing.
  Earlier bundled-predictor MPS diagnostics have a separate baseline.
- Primary endpoint: equal-sequence mean present-frame Dice. Absent frames are
  reported separately; a 21-sequence average is not a 23-sequence score.

## Three-region assignment

Each stored prediction labels its own 64×64 memory grid (4,096 vectors,
64 channels). The initial conditioning frame uses the GT prompt.

Assignment `sam2_efficient_sam2_boundary_iou_v3` chains four pinned author
functions; project code only passes outputs between them
([`region_tokens.py`](../modeling/region_tokens.py)).

| Step | Author code (unchanged) | Result |
|---|---|---|
| 1 | SAM2 `SAM2VideoPredictor.add_new_mask` resize block | Mask resized to 1024×1024 (bilinear, antialiased), binarized at 0.5 |
| 2 | SAM2 `SAM2Base._use_mask_as_output` | SAM2's 256×256 low-resolution mask logits, as stored for a mask prompt |
| 3 | Efficient-SAM2 `select_mask_windows` (window = one token) | Object tokens on the 64×64 memory grid: any mask pixel in the cell |
| 4 | [Boundary IoU](https://github.com/bowenc0221/boundary-iou-api) `mask_to_boundary` on the object tokens | Object tokens within d = round(ratio × grid diagonal) of the contour |

| Region | Definition |
|---|---|
| Boundary | Step 4 band, i.e. Boundary IoU's G_d ∩ G, inside the object |
| Foreground interior | Object tokens outside the band |
| Background | Non-object tokens |

At ratio 0.02 the band is 2 tokens (32 px at 1024); sensitivity ratios
0.01/0.04 give 1/4 tokens. Because step 3 counts any covered cell as object,
the outer band ring holds the tokens the true contour crosses. Context just
outside the object is background, as in Boundary IoU. Small objects may have
no interior tokens. Source revisions and clean checkouts are verified before
execution, and tests check exact agreement with SAM2's mask-prompt path.

Superseded v1 (project rule) marked a token boundary if it touched any band
pixel on either side of the contour or mixed classes. **All regional results
in [REGION_TOKENS_RESULTS.md](REGION_TOKENS_RESULTS.md) dated 2026-10-22 or
earlier use v1** and must be rerun before comparison with v3.

GT-reference labels are for offline analysis only. Object pointers remain
separate. No learned classifier or DINOv2 head is used.

## Experiments

| Experiment | Implementation and control |
|---|---|
| Assignment audit | Exact vector reconstruction, overlays and GT/predicted agreement |
| Native attention | Reuse Efficient-SAM2's author attention calculation on native post-RoPE Q/K/V; return original SDPA outputs |
| Sensitivity | Same attention weights grouped by predicted/GT masks at three widths |
| Regional exclusion | Mask one region's keys in all four cross-attention layers; three per-frame count-matched random controls; no memory writeback |
| Author benchmark | Official SAM2, SAMURAI, SAM2Long and Lean-SAM2 predictors in separate processes |
| Enrichment weighting | Fixed proportional, reciprocal and proportional shuffled-label logit biases; native full propagation |

Attention mass averages summed regional weights over all spatial queries/heads.
Enrichment = regional spatial-attention share / regional spatial-token share.
Empty-group enrichment is undefined, not zero. Pointer mass is reported separately.
Enrichment measures allocation, not causal utility.

Weights use four-layer, equal-sequence mean enrichment from all eligible
sequences. Calibration and evaluation share the cohort: dataset-specific,
exploratory adaptation, not held-out validation. Shuffling preserves counts
within each stored frame; pointers get zero bias but their normalized mass can
change. No matched shuffled-inverse control was run.

## Verification and execution

Pinned author sources and the protocol record revisions/settings. Full runs
require complete seq2/seq3 fresh-model replay, frozen parameters, matching
source/checkpoint/runtime identities and confirmed online W&B artifact upload.
Read-only instrumentation must preserve baseline logit hashes exactly;
attention reconstruction uses numerical tolerances, not bit equality.

Use [`scripts/README.md`](../scripts/README.md) for entry points.
`queue_author_region_measurements.py` produces the SAM2/SAMURAI/SAM2Long
comparison; SAM2 calibration is reused. SAM2Long retains its author key scaling
and offline final-path selection; attention summaries cover all executed branch
reads, not final-path-only attribution. SAMURAI trims pre-prompt frames while
preserving source IDs. Efficient-SAM2 SMR-only remains blocked on MPS replay.

Statistics use sequence means, 10,000 whole-sequence bootstrap resamples and
paired Wilcoxon tests with Holm correction within each comparison family.
Patient independence is unverified. Outputs/caches/logs use the requested
mounted Drive directory and W&B; independent Google Drive cloud sync is not verified.

See [results](REGION_TOKENS_RESULTS.md). Author-specific regional exclusions and
stored-mask-quality analyses are not yet completed.
