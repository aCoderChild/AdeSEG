# FGPI target-anchored feature-token selection: implementation and evidence

**Status:** source implemented; unit-level CPU geometry tests performed;
full real-YOLO/MedSAM2 video results **not yet executed**. No Dice improvement,
initialization reliability gain, clinical efficacy, or scientific novelty
is established by these commits.

## Why the original one-prototype prompt was weak

The earlier implementation ranked foreground tokens against a mean of central
bounding-box features and sampled one outside point based on background
similarity. Its pseudo-foreground mean can be polluted by a box-localized
background or distractor. A large ring around the detector can also sample
irrelevant distant background. On uniform features, the old strategy silently
used arbitrary tied candidates.

### The crucial difference from Lean-SAM2

**Lean-SAM2 TAMP** constructs diverse foreground anchors from an *already
prompted/identified target object* and uses maximum cosine alignment to protect
target-memory tokens. In FGPI the initial YOLO box is the **only target prior**;
there is no trusted foreground mask. We therefore construct *provisional
box-derived pseudo-anchors*, not ground-truth anchors, and retain central
spatial support to reduce contamination. The method **adapts** target-anchor
principles; it is not identical to, nor a new invention of, TAMP.

## Deterministic method (v1)

Let `B = (x0,y0,x1,y1)` be the frozen YOLO bounding box, `F` an early,
unaltered MedSAM2 FPN feature map, and `z_p = L2normalize(F[p])` a spatial
feature token. The source frame is the detector-selected frame; labels,
ground-truth masks, future frames and teacher logits are forbidden at prompt
selection time.

1. Define a conservative **positive core** `C` by shrinking the box 15% in
   both dimensions. Use its full interior only if the shrunk region contains
   no feature-token centers. Construct a **negative annulus** `N` outside a
   10% enlarged box, but within an 85% enlarged search range. This explicitly
   keeps negatives away from uncertain bounding-box boundaries.
2. Form a center-weighted pseudo-target mean `u` over `C`; score interior
   tokens by `0.80 cos(z_p,u) + 0.20 exp(-1.5 r_p^2)`, where `r_p^2`
   is the squared normalized distance from the box center.
3. Take the strongest 35% of central candidates (subject to a minimum
   anchor budget), seed the highest-supported token, and greedily choose
   up to four anchors by
   `0.70(1-max_a cos(z_p,a)) + 0.30 exp(-1.5 r_p^2)`.
   This encourages feature diversity **within a spatially supported pool**.
4. Compute `t_p = max_a cos(z_p,a)`, `b_p = cos(z_p,mean(z_N))`, then
   `s_p = 0.65(t_p-b_p) + 0.35 AvgPool3x3(t-b)_p`.
5. Choose foreground token centers inside `C` using `s_p + 0.12
   exp(-1.5r_p^2)`; choose background centers in `N` using `-s_p`
   plus a small inverse-distance preference for nearby, safely external
   background. A 22%-of-shorter-box-side greedy spatial separation
   discourages duplicate/clustered points. If spacing is infeasible but
   enough distinct tokens exist, use best remaining distinct locations.
6. Pass the exact original YOLO box and all selected FG/BG points **together**
   to the untouched SAM2 `add_new_points_or_box` prompt encoder.

**No feature reliability gating:** the feature-contrast value is diagnostic
only. Even zero-contrast (uniform) features produce the predeclared
positive/negative point count. Insufficient spatial geometry raises an error,
not a hidden box-only fallback. Input validity is not accuracy reliability.

The reference default is **one positive + one negative** (four anchors
maximum), for comparability with the previous FGPI branch. On DEV only,
pre-register the small ablation grid (1+,1-), (2+,1-), and (2+,2-). Do not
use TEST scores to choose hyperparameters.

## Experiment contract and evidence

- **Primary initialization endpoint:** prompt-frame Dice and IoU, measured
  separately from subsequent video propagation. This exposes whether FGPI
  really helps the initial segmentation rather than simply reporting
  downstream Dice after propagation.
- **Secondary endpoints:** GT-present post-prompt Dice (frame-weighted and
  equal-video), GT-empty false alarms, boundary F, temporal IoU, centroid
  displacement, and per-sequence failure/exclusion rates. Do not omit videos
  lacking YOLO detections when computing detection coverage.
- **Mandatory matched control:** identical first eligible YOLO detection,
  checkpoint, box, frame, preprocessing, decoder, and propagation, changing
  **only** additional feature-derived points.
- **Strong selection-specific controls still needed:** matched box + two
  geometric points, box + spatially random points (fixed seed), naive
  one-prototype FGPI and varied numbers of positive/negative points.
  Comparing only against box-only does **not** establish the novelty or
  usefulness of target-anchor point ranking.
- **Robustness studies:** controlled YOLO box jitter/under-coverage,
  distractor tissue, blur, lighting, lesion scale and occlusion. Report
  distributional results, not just a curated successful example.
- **Statistical unit:** verified patient/source groups, not correlated video
  frames. Bootstrap by these groups; report eligibility and missingness.
- **Data caveat:** PolypGen colonoscopy is an engineering proxy for adenoid
  segmentation, not a validated nasopharyngeal clinical test set.

## Prior methods and scientific claim boundaries

1. Ravi et al., **SAM 2: Segment Anything in Images and Videos**, ICLR
   2025 / arXiv:2408.00714. Establishes the multi-modal box/point prompt
   encoder and propagation; combining prompt types is **not novel**.
   https://arxiv.org/abs/2408.00714
2. **Lean-SAM2: Target-Anchored Memory and Encoder Acceleration for SAM2**,
   arXiv:2607.19811, §IV-B, Eqs. (12)–(15): diverse prompt-derived FG
   anchors and cosine consistency for memory pruning. Our pseudo-anchor
   adaptation operates **before** mask initialization and uses box geometry
   rather than trusted foreground labels. That distinction is important.
   https://arxiv.org/abs/2607.19811
3. **Systematic Evaluation and Guidelines for Segment Anything Model in
   Surgical Video Analysis**, npj Digital Surgery (2025): documents
   multiple points, box-and-point prompts and sensitivity to surgical
   object/background ambiguity. This supports testing, **not** a general
   assumption that more prompts improve segmentation.
   https://www.nature.com/articles/s44484-025-00002-2
4. **AutoPrompt-SAM3D: Integrated Generation and Selection for SAM2-based
   3D Medical Segmentation** (2026): fuses multi-level image features for
   automatic prompting and selects reference frames. This is highly
   relevant prior art; YOLO + feature prompts alone cannot be claimed new.
   https://doi.org/10.1186/s12859-026-06390-7
5. **Slice-based YOLO-SAM Hybrid Prompting Framework for Automatic Medical
   Image Segmentation** (2026): already combines YOLO boxes with inside
   positives/outside negatives. FGPI must demonstrate that its particular
   token selection outperforms this type of geometric hybrid.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC13766228/
6. **Domain-agnostic Weakly Supervised Surgical Instrument Segmentation**
   (2026): uses a saliency-based high-scoring pixel to seed SAM2,
   reinforcing that automated point choice is a populated research area.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC13003140/

**Provisional contribution statement:** a training-free, detector-conditioned,
spatially constrained diverse pseudo-target-anchor strategy for fixed-budget
joint box+point initialization in medical video segmentation. This is a
**testable engineering hypothesis** until matched held-out experiments
demonstrate improvements and distinguish it from prior hybrid prompting.
