# Proposed patient-grouped cross-fit channel-refiner study, v1

Superseded by `EXPLORATORY_WARM_UNIFIED_CROSSFIT_V1.md` after the observed
warm/direct feature mismatch and the user's request to avoid exhaustive
inference repeats. This file remains a historical proposal.

Status: **design in preparation; no five-fold models or results are claimed**.
Freeze a machine-readable plan and its source/receipt hashes only after the
capture study and current fixed-read stability study finish and are reviewed.
This proposal does not itself authorize or start training.

## Question and comparison

Does a trained early/deep feature adapter that aggregates channel values and
contrasts predicted foreground and background improve a fixed Lean-SAM2 read?
The controlled factorial compares global versus foreground/background channel
matrices with contrast-map absent versus present (`global`, `global_contrast`,
`foreground_background_no_contrast`, `foreground_background`). `mask_only` is
the mask-only control. `foreground_background_asinh` is a separate stability
control for the very negative native logits; do not silently pool its result
with the factorial. Optionally run the smaller `region_reference` as a
non-parameter-matched reference. Report all planned arms, including failures,
and all per-patient results; do not pick a winner from the held-out folds.

The native output logit is the residual anchor; initial adapter output must
equal it. Region summaries use only the model's own soft mask, never the target.
The frozen Lean backbone, automatic prompt, memory algorithm, token window,
and viewport handling are common across arms. No confidence gating,
redetection, or new pruning rule is introduced.

## Scope and split

Use the exact five groups, prompt identities, and source hashes in the V2
preflight plan. Test group *i*, validation group *i+1* cyclically, and train
on the other three groups. Each patient is in exactly one held-out test group.
The user states that each sequence is a different patient; related filename
stems remain grouped conservatively because their source relationship has not
been independently checked. Previous project investigations have inspected
these videos and outcomes. This is exploratory out-of-fold evaluation, not a
previously untouched or externally validated test.

Capture expected: 214 previously audited reads plus 143 fresh planned reads.
Use neither until the fresh aggregate, full-trajectory audits, source/prompt
provenance and the seq6 warm-up-versus-direct observer checks pass. If either
feature pipeline differs, do not merge them or infer equivalence. Seq1 and seq7
have no initialized automatic prompt: no invented samples or Dice; report
their no-initialization outcomes separately. Seq23 trains and predicts in the
verified main viewport `[446, 0, 1280, 720]`, with the full-source target cropped
to that same viewport; restore predictions to full-source coordinates with
background outside the viewport. Report both full-source metrics (including
any outside-viewport positives) and viewport-restricted metrics, and disclose
the evaluated area. Never compare a viewport-only result to a full-frame result
as though their denominators matched. There is no assumption that the source frame
index equals its ordered metadata ordinal.

## Training plan to freeze after stability review

Each fold and arm starts from fresh seeded weights; matched arms share an
initialization and sample order where tensor shapes permit. Tentative sampler:
12 seeded updates per initialized training patient per epoch, shuffled across
patients and carried through each patient's shuffled read stream across epochs.
This gives equal update counts and eventual coverage but is not mathematically
identical to a patient-weighted full-batch Adam gradient. At ten epochs it
would require 7,560 steps per arm across five folds, hence 45,360 for six arms,
before validation, replay, or live videos. This is a resource estimate, not an
approved duration or evidence that ten epochs are optimal.

Freeze optimizer, loss, epoch count, seed, chosen arms and sample order before
running this study. Investigate the current fixed-read study's unusually large
finite gradients first, choose numerical stabilization on development data
only, and record any changes as a new version. Reject non-finite inputs, logits,
losses, gradients, parameters, and masks; separately log gradient magnitudes.
Use MPS with targeted identity, finite, routing, ROI, prompt and source checks;
no CPU-versus-MPS numerical parity workflow is required by this protocol.

## Data access and score release

Each fold runs in a fresh process with a fold-specific CPython mask-open guard:
only that fold's training masks may be decoded. Resolve each allowed historical
label hash only after verifying its archived metadata binding and the fold
authorization; check label bytes before and after decoding. CPython audit hooks
are not a native-code sandbox, so inspect imports and IO paths as well. Do not
open the validation or test masks in a fold's training process. No checkpoint
or choice may use its own validation/test outcomes. Prediction needs no target
labels; freeze all fold models and held-out prediction artifacts, source and
runtime hashes, logs and a separate reproducible replay receipt before scoring
any of the held-out labels. Cross-fitting allows a patient's labels in other
folds' training; it does not create a globally unseen test cohort.

Report present-frame Dice and per-patient Dice alongside empty-frame false
alarms and the number of eligible frames and patients. State macro versus
frame-weighted aggregation explicitly and compare only matched frames, prompts,
viewport and backend. Fixed-read results do not test error feedback through
memory. If warranted, freeze a separate per-fold live-video integration and
run its independent routing/source and feedback checks before full propagation.
Native SAM2, Lean-SAM2, SAMURAI, SAM2Long and CMSA-Net can only be ranked
together after prompt, data, initialization, viewport and metric differences
are reconciled; otherwise present their protocols separately.
