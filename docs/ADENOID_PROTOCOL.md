# Adenoid Video Segmentation Protocol

## Research Framing

The intended clinical task is prompt-guided, temporally consistent segmentation
of adenoid tissue and the nasopharyngeal airway in endoscopy video. The project
goal is not only high frame-level Dice, but stable masks that support
quantitative adenoid hypertrophy assessment with limited clinician correction.

The current codebase uses PolypGen as a public endoscopy proxy while the adenoid
dataset is not yet available. PolypGen experiments should be treated as
engineering validation of the pipeline, not as clinical validation for adenoid
hypertrophy.

## Target Labels

Use separate object tracks for:

- `adenoid`: visible adenoid tissue.
- `nasopharynx_airway`: open nasopharyngeal airway, choana, or clinically
  defined reference cavity.

Do not merge these labels into one foreground class. The downstream clinical
measurement depends on the relationship between tissue area and airway/reference
area.

## Annotation Protocol

Document these decisions with ENT annotators before labeling:

- Which anatomical boundary defines the adenoid edge.
- Whether the airway label means dark open lumen, choana opening, full
  nasopharyngeal cavity, or another clinical reference region.
- How to handle mucus, bubbles, blur, specular highlights, instruments, blood,
  smoke, and partial visibility.
- Whether incomplete-exposure frames are excluded, weakly labeled, or annotated
  as visible-only.
- Whether mask boundaries should follow visible surface texture, inferred
  anatomy, or a conservative visible-only rule.
- How disagreements between annotators are adjudicated.

Recommended minimum pilot:

- 30-50 videos.
- Patient-level train/validation/test split.
- Sparse keyframe masks for all videos.
- Dense masks on a smaller subset for temporal evaluation.
- Two independent ENT annotations on a subset for inter-rater agreement.
- A small set of hard cases reserved for qualitative failure review.

## Mapping To Current Project Methods

The current PolypGen config tests the same prompt-effort questions needed for
the adenoid study:

- `YOLO_SAM2_YOLO_BOX_FRAME`: automatic frame-by-frame detector plus segmenter
  baseline.
- `YOLO_SAM2_GT_BOX_FRAME`: oracle frame-by-frame upper bound for box-prompted
  SAM2.
- `MedSAM2_YOLO_BOX_BOX` and `MedSAM2_YOLO_BOX_MASK`: automatic MedSAM2 video
  propagation seeded from YOLO boxes or YOLO-derived mask prompts.
- `MedSAM2_GT_BOX_BOX` and `MedSAM2_GT_BOX_MASK`: oracle-prompt MedSAM2 upper
  bounds that separate propagation/segmentation errors from detector errors.
- `*_STRIDE5`, `*_STRIDE10`, and `*_FIRST`: reduced-prompt settings that test
  how much temporal memory reduces annotation or detection burden.
- `MedSAM3_TEXT_POLYP`: disabled text-prompted baseline that can later be
  adapted to prompts such as `adenoid` or `nasopharyngeal airway` if the
  MedSAM3 environment and weights are available.

For clinical adenoid experiments, duplicate this structure with label-specific
method names and output roots rather than overloading the PolypGen method names.

## Prompting Experiments

Run these conditions on PolypGen first, then repeat on adenoid videos:

- First clear-frame box prompt.
- First clear-frame mask prompt.
- Detector-derived prompt every frame.
- Detector-derived prompt every 5 frames.
- Detector-derived prompt every 10 frames.
- First successful detector prompt only.
- Oracle box prompt every frame.
- Oracle mask prompt every frame.
- Simulated correction prompt when tracking drifts.

For MedSAM2, treat every-frame prompting as an upper-effort condition. The
clinically interesting question is how much performance is retained with sparse
prompts and how often a clinician must correct drift.

## Correction-Burden Evaluation

Measure model quality as a function of human effort:

- Start from one prompted keyframe.
- Propagate forward and backward through the video.
- Mark drift frames by Dice/IoU threshold or reviewer flag.
- Add a correction prompt on drift frames.
- Re-propagate from corrected frames.
- Report Dice, IoU, temporal stability, and clinical-ratio error versus number
  of correction prompts per video.

Recommended reporting units:

- prompts per video
- prompts per 100 frames
- seconds of video between corrections
- reviewer correction time when available

## Clinical Metrics

Frame-level segmentation metrics are necessary but not sufficient. Add:

- Adenoid Dice/IoU.
- Nasopharynx-airway Dice/IoU.
- Boundary F-score or Hausdorff-style boundary error.
- Adenoid-to-nasopharynx ratio error.
- Obstruction percentage error.
- Clinical grade agreement with ENT assessment.
- Cohen's kappa for grade agreement.
- Correlation with AHI or clinician measurement when available.

Define the ratio before analysis. Candidate forms include:

- `adenoid_area / nasopharynx_airway_area`
- `adenoid_area / (adenoid_area + nasopharynx_airway_area)`
- `1 - airway_area / reference_area`

Use one primary ratio and keep any alternatives as secondary analyses.

## Temporal Metrics

Report:

- Predicted area over time.
- Adjacent-frame mask IoU.
- Area-change smoothness.
- Centroid displacement smoothness.
- Failure segment count and duration.
- Recovery after blur, secretion, occlusion, or camera motion.

These correspond to the current project metrics `pred_area_frac`,
`temporal_iou_prev`, `area_change_abs_prev`, and
`centroid_shift_norm_prev` in `scripts/utils/eval_metrics.py`.

## Proxy Dataset Scope

PolypGen is an engineering benchmark, not a clinical surrogate. Use it to debug:

- Video loading and frame ordering.
- Ground-truth box generation from masks.
- Prompt strategy.
- YOLO/SAM2 frame-by-frame inference.
- MedSAM2 video propagation.
- Failure accounting with blank masks.
- Mask, bbox, temporal, and overlay evaluation.

Do not use PolypGen performance to claim adenoid or nasopharyngeal clinical
validity. Clinical claims require adenoid videos, ENT-defined labels, and
patient-level splits.
