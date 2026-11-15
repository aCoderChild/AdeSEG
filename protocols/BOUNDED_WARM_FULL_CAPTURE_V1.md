# Separate bounded warm full-capture preparation

This draft is a separate source-bound experiment, not a replacement for any
frozen eager run. Do not execute it until the V2 seven-frame same-MPS loader
gate is independently reconstructed, resource/source review passes, and the
shared GPU lock and monitored disk/RAM headroom are available. The short gate
alone is not evidence of full-video equivalence.

First run a full 91-frame seq6 warm bounded reference and separately launched
bounded observer under the same archived automatic box, points, mask,
checkpoint and token runtime. Compare every available full-video prediction
mask and named selection/position/anchor/memory trace to the certified eager
seq6 reference, and compare captured early/deep decoder observations with the
12 phase-bound eager gate captures, including historical overlap frames 24 and
90. Record numerical differences using specified same-device tolerances and
mask sign-bit differences rather than demanding byte-identical tensors; any
material disagreement must block treating the sources as compatible. Record
actual process RSS, sampled system-available RAM, MPS allocations, free disk,
cache size and eviction count. All outputs, partials and diagnostics go to the
Drive experiment folder; no target labels or new detector/threshold are used.

For this engineering check, floating tensors use `rtol=1e-5, atol=1e-6`;
integer and metadata fields and the named selection/position/anchor/event
traces must agree. Predicted-mask agreement is reported for every frame and
must have Dice at least 0.995 (both-empty is 1) against its comparator. This
is a predeclared compatibility criterion only, not an inference-time gate,
detector threshold, ground-truth Dice or proof of clinical equivalence.

Only after a separately audited seq6 full-sequence compatibility result may a
full seq11 bounded reference and separate bounded observer run. Their output
masks, named traces and captured tensors must be independently checked against
each other, and the no-op result reported. Two earlier eager seq11 attempts
stopped before producing a complete reference; therefore no full seq11
bounded-versus-eager equivalence can be claimed. A separately versioned
aggregate/training preflight must bind the full seq6 proof, seq11 receipts,
loader/source/runtime identity and original 214+143 observation grid before
any bounded data may be mixed into cross-fit training. The frozen original V2
aggregate/preflight cannot silently treat a bounded run as its own capture.

Even a successful seq6 comparison establishes evidence only on that sequence,
not a mathematical guarantee for every video or a segmentation improvement.
No held-out label may be used until the entire out-of-fold prediction set is
frozen and verified; prior examined sequences remain exploratory data.
