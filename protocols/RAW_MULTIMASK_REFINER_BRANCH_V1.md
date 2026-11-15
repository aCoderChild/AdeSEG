# Raw-multimask feature branch: development design

This is a new experimental refiner, not an author Lean/TCMN/CMSA module and not
a video result. Use the existing author decoder exactly as captured: all 49
saved head-input snapshots already have `multimask_output=True`. Replay its
head with an observational hook to save the three raw low-resolution candidate
logits and three model-predicted IoU values. Reproduce the original suppressed
head output, IoU predictions, object score and pointer against each frozen
snapshot on the **same capture device**; do not run any new video inference,
alter a prompt or memory, or change the decoder's selection. Freeze and hash
raw tensors for all 39 training and 10 previously inspected validation reads
before this branch opens any validation labels. Old validation outcomes have
already been inspected, so this is development, not a blinded final test.

The proposed feature branch supplies three smooth-normalized raw candidate
logit maps and their three broadcast author-predicted IoU values to a learned,
zero-initialized 1x1 projection, which adds a 64-channel residual to the
existing early/deep fused features. Foreground/background descriptors and
channel reweighting remain as in the region-conditioned refiner. The existing
head still returns a continuous residual on the native selected output logits;
there is no new confidence threshold, hard presence gate, redetection, token
pruning or GT-derived inference input. The native object score and pointer are
not modified. Nonetheless, a large residual could overcome a native -1024
suppressed mask, so this method can **relax the final mask suppression** even
when the native presence score remains negative. False-positive behavior must
be measured; preserving the score is not preserving the original mask.

Same-capacity controls use identical architecture and training: `raw` uses
all candidate channels; `zero` sets them to zero; `spatial_shift` circularly
shifts only candidate spatial maps by a predetermined, GT-independent offset
while retaining their global IoU channels. The shift tests spatial alignment,
not all candidate semantics. The user-visible threshold remains the baseline
zero-logit convention. Predeclare training loss, dataset, validation scope,
epochs and seeds separately before running this branch.

The training inputs combine old CPU and new MPS captures; their identities
must remain separate in provenance. The existing live-inference guard remains
in force: static snapshots and source hashes cannot authorize a live predictor.
Any later full-video use must capture candidates as part of the existing single
decoder call, verify same-runtime tensors and trajectories, and preserve the
original presence score, pointer and memory policy. Test masked-empty and
no-object cases as well as true visible polyps.
