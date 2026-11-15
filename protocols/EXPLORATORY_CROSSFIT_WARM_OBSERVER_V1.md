# Exploratory cross-fit: common warm-prompt observer, V1

## Purpose and scope

Preserve the 214 existing source-group observations only after their archived
receipts, manifest, source identities and every tensor hash have been checked.
Capture the frozen 143 additional observations with the same automatic
box-to-positive/negative-points-to-mask sequence, on one live predictor state,
used for the 214 observations. There is no new detector call, re-detection,
confidence threshold, new memory pruning or ground-truth initialization. The
23 sequences are subject groups as reported by the user; existing development
and prior inspection prevent a blinded-test claim. Seq1 and seq7 have no
initialized object and receive no invented decoder observations.

The attempted direct-mask observer is a different pipeline: in seq6 the first
shared snapshot matches, but frame90 has different memory-conditioned features
and decoder logits and the warm/direct mask trajectories diverge after frame24.
It must not be mixed with these warm reads, nor relabeled equivalent on the
basis of the matching early feature file. The failed direct-mask audit and its
mismatch diagnostic are retained in the experiment directory.

## Frozen data and automatic prompts

Use independently reviewed exploratory cross-fit preflight V2 plan
`44426e0b883e331de59429629e8c9ddaf8471673b71d2e04a44cf7719d152ab2` for the
existing 214 and 143 fresh frame indices, original patient groups, duplicate
checks and all causal source-frame hashes. It is an index only, not a warm
runtime authorization. Independently bind the old nine source-group reference,
capture and audit receipts, the exact 214 feature/snapshot paths and hashes,
and all nine full identity dictionaries. Matching identity is necessary but
not sufficient for warm pipeline equivalence. Check every old output path and
hash; never open an annotation mask for prompt or capture eligibility.

For each new patient, use the archived automatic prompt JSON and mask bound by
the index preflight. Reproduce the original MPS box prompt followed by the
archived positive/negative points with `clear_old_points=False`, then derive
the generated binary mask and require pixelwise equality to the archived mask
before adding it via `add_new_mask` to the same state. A mismatch fails closed.
For seq23 stage only `[446,0,1280,720]`, use archived `box_viewport_xyxy` and
`points_viewport` as viewport-local coordinates, and compare masks in that
viewport. Restore saved outputs to full-frame coordinates only for scoring.
Never derive coordinates or the conditioning mask from the ground truth.

## Required engineering evidence before study

1. Freeze a separately versioned runner, source closure, config, saved plan and
   completion receipt. Match the actual old observer SHA, predictor checkpoint
   and model fingerprint, MPS adapters, fork revision/patches, token budget
   16,384 and insurance reserve 3,072, anchor implementation/strength, seed,
   deterministic settings, MPS fallback and fast-math flags. Differences in
   wrappers require explicit review and cannot be represented as identical
   runner SHAs.
2. Run a warm seq6 observer using the archived box/points and compare its whole
   saved masks, token-window events, position/selection and anchor events against
   the audited historical warm reference. Compare feature and decoder snapshot
   tensors exactly at both shared indices 24 and 90 against the historical
   seq6 capture, whose audited identity differs in runner SHA and must be
   verified explicitly. This is a compatibility gate, not an efficacy result.
3. Replay the audited seq3 historical warm short clip and compare every
   shared captured tensor and recorded trajectory with its frozen reference,
   so compatibility is also checked on an actual source-group training patient.
4. Check a seq23 warm short reference and observer with the exact local viewport
   box/point coordinates and pixelwise archived-mask equality. Check all saved
   outputs and recorded trajectories for observer no-op. The seq23 observer
   tensors are not assumed repeatable from these checks.
5. Run a separate CPU-only artifact audit for the compatibility and viewport
   checks; bind input bytes, outputs, source code, checkpoint, logs, receipts,
   MPS flags and label-open guard. User waived CPU/MPS numerical parity, so do
   not add CPU-inference comparisons or fresh redundant repeat passes.

Failure of any compatibility gate blocks pooling with the 214 historical
observations. Do not widen numeric tolerances after seeing the outcome. If the
warm implementation cannot reproduce the historical data, use a separately
planned unified recapture rather than silently mixing representations.

## Capture, comparison and storage

For each new initialized video, produce a fresh unobserved warm reference and
then one warm observer pass; separately audit their full saved masks, positions,
selections, anchor and token-window decisions and exact target coverage. Seq6
may reuse its historical warm reference only if the old source/receipt binding
and complete warm trajectories are independently verified. Only successful
phase-bound sequence audits enter an append-only combined index. Rehash captured
features and snapshots before consuming them, and keep each batch resumable.

Training, validation and propagated model testing must use this same warm
prompt schedule. Old nine warm A references may be reused only if their source,
mask, trajectory and audit receipts independently prove the intended policy.
For the 12 new videos, frozen warm A references must be paired with the modified
method under the same prompt and viewport. Historical direct-mask baseline
numbers remain separately labeled and cannot be described as matched
comparisons with warm outputs. For cross-method tables, each method must use a
comparable explicitly documented prompt protocol; a comparison is not matched
if only this method receives warmed state.

Per-sequence capture reserves the full unproduced current-sequence payload,
22 GiB on the volume and 4 GiB swap/staging margin; check repeatedly during
children and before audit. Count only actual in-folder output bytes up to that
sequence's payload. Later sequences are not pre-reserved and may stop safely;
do not claim the whole 143-read study fits. No existing research artifact is
deleted. No full capture or MPS job is authorized by this document alone;
independent code review and GPU coordination are required.
