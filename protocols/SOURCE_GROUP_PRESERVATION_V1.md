# Training-only foreground preservation and mask-expansion pilot

This is a **post-hoc, exploratory** follow-up after inspecting previous fixed-frame
and priority-video results. It is not preregistered or independent validation.
The previously inspected seq9/15 validation reads are reused; patient/source
independence is unknown. Do not train, choose inputs, or change loss weights
using seq8/11/20/23 or their labels. The inference model and runtime are not
changed by this pilot.

## Rationale and limitations

The audited full-video output-only arm removed much true foreground on seq20,
but the mechanism is not established. A subsequent training-only audit found
147 present / 17 labeled-empty reads in the frozen 164-read training set. No
present read had native recall >= 0.8 and precision < 0.2; the three with
recall >= 0.8 and precision < 0.5 all came from seq13. These reads therefore do
not adequately cover the observed error geometry. Audit evidence is stored at
`contrast_refinement_v1/source_group_training_error_audit_v1_attempt01/` on
the configured Drive.

To test robustness to an artificial form of excess mask area, training can
use a deterministic spatial maximum over the unmodified 256×256 automatic
logits. Widths 1, 33 and 65 correspond to identity and expansions with radii
16 and 32 grid pixels, respectively. Widths are assigned by a fixed hash of
sequence and frame plus training epoch, independent of label and prediction.
All augmented arms receive the same assignment for the same read and epoch.
This cannot establish that the synthetic masks reproduce real seq20 errors;
it changes logits without recomputing backbone or temporal-memory features.
Uniform −1024 logits stay −1024 and this augmentation cannot manufacture a
missed object's spatial evidence. Validation and inference are **unaugmented**.

An audit generated all expanded masks before reading each **training** label:
widths 1/33/65 produced 0/19/58 of 147 high-recall (>=0.8), low-precision
(<0.2) reads across 0/3/7 distinct training sequences. All widths retained
6/147 zero-overlap reads and 7/17 empty-label alarms. These counts are
correlated training examples and are not held-out evidence. Evidence lives at
`contrast_refinement_v1/training_logit_expansion_audit_v1_attempt01/`.

The training-only positive-preservation loss uses the ground-truth positive
pixels and the automatic teacher's sigmoid confidence, but only penalizes
decreases in the refiner's probability. It imposes no penalty on teacher false
positives and cannot constrain empty masks; the baseline supervised boundary
loss is still responsible for suppressing background. Teacher tensors and
training labels are detached. Its fixed weight of 1 is inherited from the
previous training-only teacher study, not optimized here. This one-sided form
differs from the earlier symmetric teacher penalty; it is not known to fix
either missed objects or the native presence/pointer inconsistency.

## Arms and checks

Each of the eight arms uses exactly 48,005 parameters, the same 164 training
reads and 50 validation reads, one seed, Adam settings and sample order, ten
epochs and the same corrected CMSA-inspired boundary loss. Region arms have
both early/deep features and FG/BG descriptor; no-contrast removes the contrast
slot; mask-only zeros early/deep inputs. The eight pre-specified arms are:

1. `region_clean_base`: native logits, boundary loss.
2. `region_clean_preserve`: native logits, boundary + preservation.
3. `region_aug_base`: scheduled logit expansions, boundary loss.
4. `region_aug_preserve`: scheduled expansions, boundary + preservation.
5. `no_contrast_aug_base` / `no_contrast_aug_preserve`.
6. `mask_only_aug_base` / `mask_only_aug_preserve`.

Thus base-vs-preserve at each mechanism holds augmentation fixed, while the
four region arms estimate augmentation and preservation jointly. Mechanism
comparisons with clean no-contrast or clean mask-only are outside this plan.
The protocol deliberately does not tune width, weight or epoch using validation
or priority outcomes. Prior validation and many alternatives were already
inspected, so even the above fixed order does not confer statistical blindness.

The separate launcher must bind the frozen 214-read MPS capture manifest,
all sources, the actual manifest SHA, checkpoint/optimizer/schedule and per-read
expansion widths. Require CPU/MPS parity on all arms for real training-only
present, labeled-empty and fully suppressed reads, with finite gradients and
native-size masks checked; don't silently fall back. Protect validation masks
with the existing CPython guard during planning, training, and replay. Train
each arm and freeze all 50 predictions, then independently replay all 800
baseline/refined masks before the separate scoring process opens labels.
Use the user's Drive for all artifacts and the 22-GiB free-space starting guard.

Report every arm with equal-video and per-frame Dice, precision, recall,
zero-overlap, 5-native-pixel boundary F, labeled-empty alarms and area; report
all counts and data reuse. Labeled empty does not guarantee confirmed absence,
and only three such validation frames are available. A candidate is not
authorized for video evaluation merely because this post-hoc pilot improves
one inspected mean: report matched controls and the unresolved original
promotion criteria, run separate A/B/C/D engineering if later requested, and
seek patient/center-disjoint evaluation before generalization claims.
