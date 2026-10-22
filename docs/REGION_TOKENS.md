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

| Region | Token-footprint definition |
|---|---|
| Foreground interior | Pure foreground outside the boundary band |
| Boundary-associated | Intersects the band or contains mixed foreground/background pixels |
| Background | Remaining pure background |

[Boundary IoU](https://github.com/bowenc0221/boundary-iou-api) supplies unchanged
boundary extraction. The symmetric band, area-resampled token mapping and
vector grouping are explicit project adaptations in
[`region_tokens.py`](../modeling/region_tokens.py), not an author token classifier.
Band width: 0.02 × image diagonal; sensitivity: 0.01/0.04. Small objects may
have no interior tokens. GT-reference labels are for offline analysis only.
Object pointers remain separate. No learned classifier or DINOv2 head is used.

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
