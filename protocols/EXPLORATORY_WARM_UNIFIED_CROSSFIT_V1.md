# Proposed warm-initialization unified cross-fit study

Status: **design only; no warm-unified captures, trained models or efficacy
results are asserted**. This version is an alternative to mixing the existing
214 warm-initialized reads with direct-mask captures. On seq6, the latter
matched the historical first read but differed later; its observed trajectory
and stored features must not be assigned the historical pipeline identity.

## Population and data

Use the five patient/source groups and the existing frame selections in
`EXPLORATORY_CROSSFIT_CAPTURE_PREFLIGHT_V2.md`. The user states each sequence
represents a different patient; the conservative grouping of potentially
related source names remains. Seq1/7 have no automatic initialization, so
they contribute neither invented training reads nor present-frame Dice. Report
their no-initialization outcome separately. The group splits and all previous
sequences were already explored; this is not blinded external validation.

The saved nine-patient capture manifest contains 214 warm-initialized reads:
seq2/4/5/9/12/13/14/15 each have 25 and seq3 has 14. Its nine capture receipts
share a recorded runtime identity and pass the existing source, trajectory,
image and tensor hash checks. Their manifest SHA-256 is
`827f6be88046f89761313a67e6d0ed84d35b809ec553539f32104b53cb111c24`.
These checks establish historical artifact identity, not compatibility of a
new runner. Bind the manifest, reference/capture receipts, source freeze,
observer and model identities in the new plan; do not modify old artifacts.

## Unified initialization and compatibility

For every new sequence, reconstruct the historical automatic initialization:
load the saved detector box, call the box prompt, add the saved positive and
negative points in the original order with `clear_old_points=False`, then
uniformly commit the saved automatic conditioning mask in the same view. Use
that saved mask for prompt prototypes too. The newly generated point mask is
recorded as a diagnostic, not used to choose the inference path. This policy
matches the historical seq3/6 inputs when their generated and saved masks are
equal, while accommodating seq23's known discrepancy. It uses existing
automatic inputs, not a label, another detector invocation or redetection.

For seq23, stage the main viewport `[446,0,1280,720]` and use the saved
viewport-relative box and point coordinates directly. Verify input shape and
every coordinate, compare the generated view mask to the archived mask cropped
to that same viewport, and restore predictions to full-image coordinates for
full-frame scoring. Report viewport-restricted and full-frame metrics separately.

Before accepting the missing 143 reads, establish the new runner against the
historical warm pipeline with label-free same-device comparisons. Bind the
original checkpoint, model fingerprints, fork/config, token-window policy,
RoPE hooks, anchor, observer, seed, environment, source-image hashes, prompts,
ROI, and target grid. Test the seq6 overlaps at frames 24 and 90 against the
previously audited warm captures and a late historical read from at least one
of the original nine patients. Compare feature/head tensor values, not only
binary masks, and inspect stored state/trace fields that exist in the old
receipts. Independently review this gate before extending the capture. The
seq6 historical grid has ten reads and overlaps the planned twelve only at
24/90; do not silently replace the frozen grid. These checks are targeted MPS
engineering tests, not a CPU-versus-MPS parity requirement.

Each remaining sequence needs a separately audited warm reference and observer
trajectory, or a comparably rigorous source-bound no-op audit. Label-free
observer and reference checks must explicitly list which model states, traces
and tensors are covered. Preserve the full source-frame list and original
label metadata bindings, but open no target label during capture. Fresh warm
A trajectories should be recorded for all 21 initialized patients to support
the later matched video evaluation; the old nine already have historical warm
reference trajectories and may be reused only if their required provenance,
state/trace coverage and evaluation protocol are demonstrably sufficient.
Never use the rejected direct seq6 partial capture as warm training data.

## Fold training and evaluation

Once all captures pass, freeze a separate aggregate of 214 old plus 143 new
reads with the precise component and compatibility evidence; do not call the
historical and new source hashes identical. Derive independent fold-scoped
label authorization from the original five-fold plan. Train each fold using
only its training patients, a fresh seed-controlled model and patient-balanced
updates. Fold test and validation patients remain disjoint from its training
patients. Use masks only as training targets or later evaluation truth, never
as an inference-time input. Hash permitted target files before/after decoding,
and explicitly crop the seq23 training target to the view. Keep the frozen
baseline and mechanism controls; freeze arms, optimizer and schedule only after
review of numerical stability and prior development outcomes.

The training worker can retain up to 1 GiB of validated CPU tensors in RAM;
this bound is on cached tensor bytes, not total process memory. Inputs are
hashed on load, checked for metadata changes on reuse and epoch boundaries,
and rehashed before a checkpoint is accepted. Cached and uncached synthetic
updates must agree; actual hit rate and speed are not known until a real run.

An independently reviewable, separately versioned shared-fold runner is also
available: it reads each scheduled training capture once and updates the six
independent model/optimizer/RNG states in the same per-arm order as separate
workers. It records per-arm checkpoints and a fold receipt binding all of
them, registers each completed arm as its own provenance phase, and preserves
partial work on failure. Its process peak RSS is recorded separately from
cached tensor bytes; neither captures complete device memory. Synthetic CPU
tests compare all six arms' stochastic updates and check non-mutation of shared
inputs with separate workers. Real MPS timing, memory and efficacy are not yet
measured. The original separate-worker runner remains the fallback.
The shared runner's label-free MPS probe invokes its actual six-arm
initialization and update functions using synthetic targets derived from
native logits; it exercises MPS random-state save/restore, optimizer updates,
checkpoint serialization and reload before any fold training. Its probe is
not a cross-device comparison or an efficacy test.

Fixed-read predictions must all be frozen and independently verified before
their held-out targets are opened for scoring. A patient's label can appear in
another fold's training, so this cross-fit result is not a globally untouched
test. One targeted label-free MPS probe checks initialized output identity,
finite gradients, and full/seq23 view geometry; the all-fold verifier checks
phase-bound sources, checkpoints, prediction files, fold ownership and views.
There is no exhaustive repeated prediction inference or CPU/MPS parity test.
The independent metric audit reuses the structural verifier, so its scope is
metric recomputation, not a new independent data cohort. Full-video claims
additionally require per-fold live integration with
that fold's checkpoint and a warm-matched Lean A, output-only, memory-only,
and combined routing controls where feasible. Output-only B must preserve the
recorded native memory-state components and traces; the encoder intervention
does not claim to replace object pointer, object score, or all compact state.
Verify trajectories before opening evaluation masks. Report present Dice,
empty-frame false alarms, patient-macro and frame-weighted values and each
patient's outcome. Dataset-level claims must account for no-init sequences.

Use previously reported direct-initialization SAM2/Lean/SAMURAI and CMSA-Net
only with a precise protocol disclosure. In particular, warm and direct prompt
schedules can produce different memory histories even with the same first
mask; do not rank their scores as matched without appropriate reruns or a
separately validated comparison. No mechanism or improvement claim follows
from the compatibility tests themselves.

Reserve 22 GiB plus a 4 GiB margin and the current sequence's expected payload;
this is not a reservation for all future sequences. Full-size 143-read snapshots
are roughly 5.7 GiB before other artifacts. A compact lossless representation
is a separate version and requires validation of every tensor needed for both
training and replay before any full snapshot is discarded.
