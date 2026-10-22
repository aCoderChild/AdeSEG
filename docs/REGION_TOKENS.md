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

**2026-10-23 frozen audit definition:** historical results must not be
silently relabeled. The primary rule for the new audit is exact mixed-cell occupancy,
using Efficient-SAM2's unchanged window selector on foreground and background
of each frame's **native stored 256×256 logits**. Mixed cells are boundary-associated;
pure cells are interior or background. Grid-aligned contours may have no mixed
cells; this is a spatial partition, not a complete anatomical contour detector.
Boundary-associated tokens are cells occupied by both object and background,
following the mixed-region idea examined in [BPR, Appendix B](https://arxiv.org/html/2104.05239)
and [Mask Transfiner, section 3.1](https://arxiv.org/html/2111.13673), computed with
Efficient-SAM2's occupancy code. BPR's mixed-grid rule is an ablation, not its
preferred dense-sampling/NMS method. This is conceptual attribution, not a claim
to reproduce either complete method or Efficient-SAM2 defining a boundary class.
Source functions: Efficient-SAM2 `select_mask_windows`, Mask Transfiner
`get_incoherent_mask`, MatteFormer `GenMask`, and Boundary IoU `mask_to_boundary`.
Author repositories: [Efficient-SAM2](../external/Efficient-SAM2),
[Mask Transfiner](https://github.com/SysCV/transfiner),
[MatteFormer](https://github.com/webtoon/matteformer), and
[Boundary IoU](../external/boundary-iou-api). Commit pins and adaptations are
recorded in the protocol; only the named source functions execute unchanged.

`modeling/region_definitions.py` also loads unchanged Mask Transfiner incoherence
and MatteFormer trimap code from pinned submodules. The former has bilinear
sampling and an author 0.01 cutoff (not an exact mixed-cell rule). Trimap widths
are fixed at kernels 3/7/15 in the author's 640×640 generation space. These
adapters and scale choices are documented, not claimed as unchanged full models.
The older v3 inner band remains a comparator.

The primary has no band-width ratio: its 4×4 native-mask footprint is fixed by
the memory grid (16×16 square-input pixels), not a continuous one-token-wide
band. Current width sensitivity uses only trimap kernels 3/7/15. The current
v3 comparator is 0.02 only; historical 0.01/0.04 measurements are not new-primary
sensitivity results.

Size bins use **per-frame GT area / original image area**, over 1,710 positive
frames: cut-points **2.034683%, 5.384635%, 11.534966%**; ties go to the lower
quartile and absent frames are separate. The procedure and numeric values were
saved before this run's inference, not before historical studies. Numeric bins
are now also in the [companion protocol record](../protocols/region_definition_audit_v1_record.json).
The executed protocol was hash-frozen but **not Git-committed before launch**.
This post-launch clarification does not change it or confer prospective
preregistration; the original protocol bytes remain unchanged.

`scripts/audit_region_definitions.py --method sam2 --device cpu --smoke` plus
the usual checkpoint/manifest/new Drive output arguments runs complete seq2/seq3
verification with online W&B, native-mask provenance, prompt parity, square-grid
overlays, attention checks and fresh-model no-observer replay. A full audit uses
the same command without `--smoke`, adding `--validation-dir` for the matching
successful gate. Source/checkpoint/device identities and artifact delivery must
match. Exact settings: [verification protocol](../protocols/region_definition_verification.json).

Every audited frame records missed one-pixel native256 contours, both including
and excluding the image border. Self-prediction contours measure geometric
coverage; GT contours additionally reflect segmentation error. Counts, empty
groups, GT-region agreement and GT-area quartiles are reported; quartiles are
frozen before inference. Preprompt/unpromptable frames are GT-reference-only,
never fabricated native memory. Claims concern token positions, not pure vector
contents. A cross-definition claim requires consistent effects across all
prespecified definitions; otherwise report definition dependence. Causal claims
still require contiguity-matched CPU FP32 exclusions after the audit.

Each stored prediction labels its own 64×64 memory grid (4,096 vectors,
64 channels). The initial conditioning frame uses the GT prompt.

| Primary candidate region | Exact occupancy of the native 4×4 mask footprint |
|---|---|
| Foreground interior | Foreground only |
| Boundary-associated | Both foreground and background |
| Background | Background only |

Existing experiment runners retain v3 until verification and explicit rerun;
the new candidate is currently used by the separate definition audit only.

### Retained v3 comparator

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

At ratio 0.02 the band is 2 tokens (32 px at 1024); ratios 0.01/0.04 would
give 1/4 tokens, but are not rerun in the current audit. The outer ring
approximates the predicted contour; it is neither guaranteed complete nor a GT
contour detector. Context just outside the object is background, as in Boundary
IoU. Small objects may have
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
| Current definition audit | Native mixed cells, author incoherence, trimap kernels 3/7/15 and v3 ratio 0.02; contour misses, size bins and missing groups |
| Historical sensitivity (v1) | Same attention weights grouped by predicted/GT masks at three widths; not new-primary sensitivity |
| Historical regional exclusion (v1) | Mask regional keys in all four cross-attention layers; three count-matched random controls; no memory writeback. Not contiguity-matched; new controlled rerun pending |
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
