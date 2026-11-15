# Segmentation refinement methods

AdeSEG studies a trainable refinement module for Lean-SAM2 video segmentation.
The research question is whether early and deep image features, together with
foreground–background context, can improve a predicted mask before it is
returned and/or used to encode memory for later frames. The detector supplies
the automatic initialization; the refiner uses the model's own mask logits,
not ground-truth labels, as its inference-time foreground signal. These are
research candidates. Their contribution and generalization are not established
by the current exploratory results.

## Candidate 1: foreground-conditioned boundary refiner

The earlier candidate is implemented in
[`ForegroundConditionedRefiner`](../modeling/foreground_conditioned_refiner.py).
It projects and aligns early and deep features, derives soft foreground and
background descriptors from the predicted mask, and combines their cosine
contrast with a channel gate adapted from TCMN's CBAM implementation. The gate
**reweights** feature channels; it does not mix their values. A learned residual
head corrects the native mask logits. The tested `region_boundary` checkpoint
has 48,005 trained parameters and used 1,640 training steps; see the
[boundary experiment audit](../protocols/BOUNDARY_FULL_VIDEO_EXPLORATORY_AUDIT_20261115.md).

The [boundary routing adapter](../modeling/lean_boundary_refinement_routing_v1.py)
tests four settings with the initial conditioning output protected:

| Arm | Returned mask logits | Mask input to memory encoder |
| --- | --- | --- |
| A | Original | Original |
| B | Refined | Original |
| C | Original | Refined |
| D | Refined | Refined |

The memory intervention changes the **mask supplied to the existing memory
encoder**. It does not directly refine the encoder's feature-vector output or
replace Lean's token selection. The predictor retains its native compact mask,
object score and pointer at the current step; later states can change through
feedback.

On 12 previously inspected sequences, arm D achieved patient-macro
present-frame Dice **0.6787**, compared with **0.6593** for matched project
Lean arm A. D improved seven sequences and worsened five; false alarms on
labeled-empty frames were **138/184** versus **137/184** for A. Sequence 8
fell from 0.8525 to 0.7090, sequence 20 from 0.1883 to 0.1777, while
sequence 23 rose from 0.1768 to 0.3075. These are development results from
one selected checkpoint, not evidence that the method avoids regressions.
Sequence 23's inset-excluded viewport was selected retrospectively. The
audited table is stored under the Drive experiment folder at
`contrast_refinement_v1/combined_12_boundary_factorial_v1_attempt01/report.md`.

## Candidate 2: multiscale channel-value refiner

The newer candidate in
[`MultiscaleChannelValueRefinerV1`](../modeling/multiscale_channel_value_refiner_v1.py)
addresses the project's channel-aggregation hypothesis more directly. It
projects early and deep features, computes learned query/key channel
correlations using soft predicted foreground and background, and applies the
resulting channel-attention matrix to feature **values**. A foreground–background
contrast map and a residual head then refine the native logits. The
[`foreground_background_asinh`](../modeling/multiscale_channel_value_refiner_v1.py)
mode transforms the logits supplied to the head for numerical stability; the
original logits remain the residual anchor. This is a project adaptation,
inspired by prior work; it is not the released CMSA-Net or TCMN architecture.

The separate [channel-value routing adapter](../modeling/lean_channel_value_routing_v1.py)
supports the same A/B/C/D output-versus-memory-input comparison. Its earlier
four-sequence direct-initialization pilot gave patient-macro Dice 0.3882 for A,
0.3979 for memory-only C, and 0.3636 for combined D. C worsened sequence 20
and added one labeled-empty false alarm. These preliminary results do not
establish that channel-value aggregation improves masks, nor do they match the
warm initialization planned below. The report is stored at
`contrast_refinement_v1/channel_value_asinh_video_v1_0nzo16w1/scores/report.md`
under the Drive experiment folder.

**The 0.6787 result belongs to Candidate 1, not Candidate 2.** Both are
project-specific refinement candidates; the newer module's claimed value must
be established by matched controls and held-out predictions.

## Evaluation roadmap

The active [three-fold protocol](../protocols/BOUNDED_CROSSFIT_FIXED_READ_3FOLD_V1.md)
uses the completed source-bound warm captures. Each fold tests seven initialized
sequences and trains on the other 14 using the same fixed schedule for all six
arms. Potentially related filename pairs stay together; seq1 and seq7 have no
automatic initialization and receive no invented Dice. There is no separate
validation split or held-out checkpoint selection.

Before reporting fixed-read results, verify the capture registry and MPS
optimizer/checkpoint probe, train the three folds, and freeze and verify all
18 held-out prediction sets before opening held-out labels. Report per-sequence
and patient-macro Dice, frame-weighted Dice, labeled-empty false alarms and
matched-arm changes. Separately test the planned mechanism controls and
warm-matched full-video output-only, memory-only and combined routes; fixed-read
scores alone do not measure memory feedback. Compare native SAM2, original
Lean-SAM2, SAMURAI, SAM2Long and CMSA-Net only with their initialization,
causality, viewport and provenance differences stated.

All sequences or outcomes were previously inspected. Patient identity is
user-reported and no independent external cohort is available. The resulting
three-fold study is exploratory, not a blinded confirmation. Runtime progress
and audit receipts are kept in the Drive experiment folder under
`contrast_refinement_v1/`.
