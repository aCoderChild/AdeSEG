# Binding a refiner checkpoint to its actual video runtime

Status: requirements only. No trusted runtime verifier or approved learned
video-refinement run exists yet. A checkpoint's metadata and caller-supplied
hash strings do not establish the identity of a live predictor.

## Pipeline that produced the current pilot reads

The 25 pilot reads came from an automatic box-plus-feature-point prompt, a
project multi-scale anchor, indexed spatial RoPE, and the project raw token
window (`16384` total tokens and `3072` insurance tokens) on top of an amended
Lean checkout. They are not a plain author Lean or the separate recent-window
runtime. The saved channel-feature receipt contains the full pipeline and
source signature; its canonical JSON hash is recorded as the checkpoint's
feature-pipeline hash. The existing preflight records only that hash, not a
binding to a running predictor.

## Requirements for an actual match

1. Build the predictor in a verifier-owned, fresh process from explicit
   checkpoint, Lean bypass weights and repository worktree. Hash every input
   file and verify source revisions and actual diffs, not just a user-provided
   revision string. Record package import paths and class implementations to
   catch loading a different checkout. Verify configuration values and exact
   parameter **and buffer** names, dtypes, shapes and bytes after loading.
   Persist fingerprints and verify them after inference, not only parameters.
2. Capture or recompute preprocessing from the actual execution: image decoder,
   RGB channel order, resizing/interpolation, normalization, viewport rules,
   CPU/GPU/dtype and runtime/library versions. A configured seq23 crop is part
   of that sequence's input policy and must be separately recorded. The
   user-visible mask resize and the memory-encoder resize are different paths.
3. Verify the active prompt inputs and schedule, Lean memory selection and
   bypass configuration, positional encoding/RoPE implementation, project
   anchor, pruning and raw-window settings against the declared runtime. No
   ground truth can select a prompt or enter a live prediction. Policies that
   differ from the captured pipeline must be labeled as a distribution shift,
   not silently declared compatible. Original author Lean and project runtime
   are distinct conditions.
4. Generate an immutable manifest from measured values and hash its canonical
   representation. Compare every measured value to a versioned expected
   manifest and to the checkpoint's training-capture identity. Validate the
   checked checkpoint's model code and author gate hashes; a string copied out
   of that checkpoint is not an independent measurement. A digest detects
   changes but is not, by itself, proof that the manifest is truthful.
5. On fixed saved development frames, compare live early and conditioned deep
   features, original low-resolution logits, object scores, memory source IDs,
   token counts, RoPE positions and memory-encoder inputs with trusted captured
   snapshots. Check exact equality when using the recorded deterministic CPU
   runtime; if another supported device requires tolerances, define them in
   advance and report the measured differences. Capture artifacts must be
   hashed against trustworthy receipts. Existing `parameter_integrity_after`
   files are not currently bound by the 25-sample preflight receipt and cannot
   alone be treated as independent proof of a live model's identity.
6. The verifier must create or own the predictor and bind the approval to that
   *particular* model, checkpoint, pipeline, and execution context; invalidate
   it when any of those changes or hooks are replaced. Do not accept a free
   `verified=True` flag or caller-provided digest as proof. A new verification
   must also cover any refiner checkpoint and its model implementation.
7. Before a full experiment, reproduce a longer no-op trajectory, including
   active memory pruning and nonzero prompt frame, and test nonzero routing on
   real logits for the four arms. Verify independent A/B histories when only
   output changes and intended memory writes for C/D. Synthetic forced-logit
   routing tests establish wiring only.

The current adapter deliberately blocks learned-refiner B/C/D routing until
these conditions have an implemented verifier. It allows explicitly marked
synthetic test stimuli, which cannot establish research-checkpoint suitability.
