# Predicted-region-conditioned channel reweighting

Research prototype; not yet trained, authorized for live video, or shown to
improve segmentation. It adapts the released TCMN CBAM channel gate; it is not
TCMN's whole model or the CMSA-Net CMA module.

For each frame, project the early 32-channel and memory-conditioned deep
256-channel SAM2 features to 32 channels each and align them with the decoder
logits. Let `F` be the resulting 64-channel map and `p = sigmoid(logits)`.
Compute foreground and background descriptors as soft spatial means weighted by
`p` and `1-p`, with denominators clamped to `1e-6`. Pass the two-position
descriptor `[foreground, -background]` through the *author's* average/max
pooling and shared CBAM MLP. The resulting sigmoid weights multiply `F`.
The existing foreground-minus-background cosine-contrast map and original
logits enter the existing zero-initialized residual head alongside the
weighted features. No label, confidence gate, new threshold, detector call or
new pruning rule is used during inference.

The author MLP combines channels to calculate channel weights; the output
operation itself reweights channel values. It is not a full cross-channel
value-mixing attention matrix. In particular, the original project refiner
used global CBAM plus a separate foreground/background contrast map; this
version actually conditions the channel weights on those two region summaries.

Matched-capacity comparisons are implemented with identical parameters and head:
`global` uses the full feature map's author CBAM pooled descriptors;
the `shuffled` code mode applies a **fixed one-channel permutation** to region
descriptors before the same shared MLP. Since a trainable MLP can potentially
learn the inverse permutation, this arm tests fixed-permutation sensitivity;
it is not an information-destruction placebo or a strong negative control.
For a future independent control, define a per-example deterministic
permutation independent of ground truth, preserve a matched training/evaluation
regime and document its seed before inspecting outcomes.
`contrast_head=False` zeros the existing contrast-map input and supports a
separately trained no-contrast-head control. A mask-only input ablation must be
trained independently and checkpointed with that input mode. Compare all
arms on identical captures, training allocation, epochs, optimizer and seeds.
The fixed permutation is an experimental comparison, not a new pruning policy.

The mode and source hashes are bound into a distinct
`foreground_conditioned_refiner_v1` research checkpoint format; previous
multilevel checkpoints cannot be loaded as this candidate. Metadata hashes
are checks against changed sources and do not by themselves verify a live
predictor. The shared video routing guard must stay in force until live model,
runtime and feature-pipeline provenance is independently checked.

Engineering tests cover exact agreement with author gate output, equal
parameter counts, gradient flow, CPU/MPS numerical agreement, missing/old
checkpoint rejection, zero-residual identity and finite forward/backward
behavior for SAM2's suppressed `-1024` logits. They do not show recovery from
a missed object, effectiveness in memory or statistical generalization.
