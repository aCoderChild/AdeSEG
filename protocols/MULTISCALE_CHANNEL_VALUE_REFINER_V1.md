# Multi-scale channel-value refiner: exploratory proposal

Status: fixed-read module and CPU unit checks only. It has not been trained,
integrated into a live video predictor, compared with the prior refiner or
shown to improve any sequence. The architecture was chosen after inspecting
these videos and prior outcomes; grouped cross-validation here is exploratory.

## Mechanism

`modeling/multiscale_channel_value_refiner_v1.py` projects the available early
SAM2 FPN and deep memory-conditioned features to equal channel counts. It
retains their fine-resolution features and pools a copy to a bounded grid.
From that grid it constructs learned Q, K and V. Channel-axis attention is
`softmax(Q K^T)` and its matrix multiplies V, so values from different channels
are actually aggregated. The FG/BG arm subtracts the soft-background channel
correlation from soft-foreground channel correlation before row softmax.
Both Q and K are multiplied by the same sigmoid probability: the unnormalized
spatial contribution is weighted by p² and (1-p)² respectively. Degenerate
empty or full masks yield finite normalized profiles via epsilon, without a
presence gate. A trainable projection and fine skip feed a zero-initialized
residual logit head, so the initial training-mode result is the native logit.

This is a new project adapter inspired by CMSA's multiscale attention and
CFBI's foreground/background distinction; it is not their published module.
The existing TCMN CBAM applies channel weights and does not itself perform
channel-value attention. No initial-prompt descriptor is included: the
current fixed-read capture interface has not yet demonstrated a source-bound
prompt-feature input. It may be investigated only as a separate extension.

The existing logit, including SAM2's -1024 suppression, is the residual base.
A sufficiently large learned residual could still override that suppression,
while the native object-presence score and pointer remain unchanged. This can
create inconsistencies and false alarms. Zero initialization and finite
sentinel gradients do not show that practical recovery is possible: previous
raw-candidate and bounded-anchor pilots did not recover the training hard case
and degraded validation. No redetection, threshold, gating or token-pruning
rule is introduced. Fine-resolution features and native mask coordinates must
be verified on real captures before any video use.

A label-permitted training-read CPU diagnostic found that the suppressed
present seq4/24 read produces a much larger initial gradient than seq2/4 and
the empty seq4/18 read. This is not proof of a failure but could dominate the
optimization. Every arm records peak per-read gradient L2 and its training
identity for every epoch; no inference decision is based on it. A different
loss or bounded input would be a separately matched future experiment.

In a same-initial-state CPU probe of those three training reads, replacing
only the head's logit input with `asinh(logit)` preserved the zero-initialized
output and loss exactly. On seq4/24, initial gradient L2 was 53.445 for the raw
input and 0.356 for asinh; seq2/4 was 0.102 versus 0.099, and empty seq4/18
was 0.0103 versus 0.0036. These are three training examples and do not establish
segmentation improvement or prove the suppressed target can be recovered.
The raw-logit reference and the 2×2 controls remain unchanged. `asinh` changes
the learned head's input representation but preserves the native residual
anchor, and has derivative one at zero.

## Controls and first study

Eight arms have the same input shapes, training budget and 63,809 allocated
parameters: global attention with both feature inputs zeroed (mask-only),
diagonal global attention, a 2×2 comparison of global versus FG/BG attention
matrix and zero versus FG/BG spatial contrast map, foreground-only attention
and its foreground contrast map, and a separate FG/BG arm using `asinh(logit)`
only as the residual head's logit input. Its residual skip remains the original
native logit, so zero initialization is still exactly native. The mask-only and global
arms have identical computation and initial parameters except that the
mask-only mode zeros both feature inputs. A ninth same-data region-conditioned
CBAM reference has 48,005 parameters and is not parameter-matched. The auxiliary
contrast feature is either zero, FG cosine, or FG-minus-BG cosine, so
foreground-only does not secretly consume a background descriptor. The 2×2
controls allow matrix-source and spatial-contrast effects to be compared while
holding the other fixed; they do not by themselves establish an improvement.
The diagonal attention uses a sigmoid on its diagonal rather than dense
row-softmax, and its off-diagonal attention interactions have no effect: it
therefore has fewer effective interactions and changes normalization as well.
The shared 1×1 Q/K/V and output projections can themselves mix channels.
It is an informative diagonal-attention control, not a pure no-channel-mixing
or strictly effective-capacity-matched control.

Use the existing frozen Lean automatic-prompt reads, the corrected CMSA-inspired
weighted BCE plus weighted IoU plus soft-Dice objective used by the prior
boundary-trained reference, fixed training duration, optimizer, sample order and
seeds across arms. Do not train on test-fold masks or use them for sample
selection, stopping or checkpoint selection. Train separate fresh weights for
each mode and fold; existing CBAM checkpoints cannot load as these models.
Record parameter counts, input tensor hashes and real numerical device results.
No prompt descriptors are inferred from labels or substituted by random data.
The fixed-read launcher first runs 36 real CPU/MPS loss/gradient checks on
four training examples (including a verified suppressed-positive seq4/24)
and exact zero-residual identity checks. It trains nine arms,
freezes all 900 baseline/refined validation masks, and separately replays all
of them without validation labels. Only then does it score the 50 reused
validation reads, followed by a separate CPU/SciPy metric and summary audit.
The metric auditor shares the structural verifier; these are not independent
datasets, and a passing engineering check is not a segmentation result.

Use five fixed groups, retaining known shared filename stems together even
though the user reports one patient per sequence:

| Fold | Sequence IDs |
|---|---|
| 1 | 6, 10, 11, 20, 23 |
| 2 | 1, 2, 3, 4, 5 |
| 3 | 7, 8, 9, 15 |
| 4 | 12, 13, 14, 16, 17 |
| 5 | 18, 19, 21, 22 |

Seq1 and seq7 have no saved automatic initialization: report their all-frame
detector outcomes separately and never fabricate segmentation prompts. The other
21 videos can support post-prompt segmentation; enforce the existing prompt,
crop and inclusion rules. Some filename-linked groups or detector exposure may
still be uncertain even after the user-confirmed patient distinction.

Reuse the existing 214 training/validation captures only after binding each
read's input tensor, prompt, crop, runtime and source identity. The other 12
initialized videos need fresh captures: predeclare at most 12 uniformly spaced
post-prompt indices per video (143 reads in total under current frame counts),
without choosing by labels, predictions or outcome. At roughly 41 MiB per read,
the new tensor payload is approximately 5.7 GiB before logs and overhead; check
the existing 22 GiB reserve and account for shared Drive sync and other jobs.
Freeze an index manifest and use fold-specific label guards. Training labels
from a video may be used in the other folds but not in that video's test fold.
All five folds' test predictions must freeze before pooled scoring. Previously
inspected seq11/20/23 remain development examples, not blinded test cases.

Forty trained models and 168 video trajectories (five folds × eight
matched modes, 21 prompted videos × eight modes) are required for a one-seed
full-video comparison, plus any exact-matched original Lean baselines that
cannot be reused. Retraining the smaller regional reference, extra seeds, or
an output/memory routing factorial adds runs. First run a fixed-read feasibility study with the same controls and
baseline; do not imply fixed reads establish propagation quality. A live
adapter must separately pass source-bound feature, prompt, no-op, gradient,
presence/pointer and CPU/MPS engineering checks on real clips, then freeze
all full-video outputs before held-out-fold scoring. The 21-video/cross-fold
results are not an independent external test and cannot by themselves support
a MICCAI generalization claim. Report per-patient and frame-weighted Dice,
precision/recall, zero overlap, empty-frame alarms/area, boundary scores,
pre-prompt misses, compute and matched patient-level differences, including
all failures; no frame-level independence claims.
