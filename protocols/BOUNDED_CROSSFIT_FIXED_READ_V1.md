# Mixed eager/bounded cross-fit fixed-read study

This is a separately versioned exploratory study. It must not change or reuse
the original eager preflight as if bounded captures came from the eager
producer. All experiment artifacts and partial runs belong under the Drive
experiment directory. Execution flags are intentionally disabled until the
actual captures, source set and runtime are independently reviewed.

The input chain first reconstructs the completed `cache_release_crossfit_handoff_v1`,
including the separately audited full-sequence seq6 compatibility result,
the V3 memory diagnostic, four eager audits and seven cache-release V2
reference/observer audits with sidecars and traces. It separately validates the existing 214-read historical manifest,
the exact 357-read grid, source frames, ROIs and five patient-grouped fold
assignments. The mixed preflight and sample registry explicitly do not
authorize training. The original eager loader does not itself check that
flag; currently its source-identity check and separate eager-aggregate schema
prevent it from using the mixed outputs. They are not a substitute for the
bounded adapter's own checks.

The proposed supervised fixed-read sequence is:

1. Reconstruct the registry in a fresh process that blocks all target-label
   opens and writes a source-bound verification receipt.
2. Run a separate label-free same-MPS six-arm probe on seq6 and seq23 reads.
   It checks initialization, finite synthetic-target gradients, independent
   optimizer/RNG updates, checkpoint reload and target-view geometry.
3. Independently audit the 12 probe records and saved checkpoints on CPU. This
   audits provenance and finite/nonzero weights; it does not repeat the MPS
   optimizer or claim CPU/MPS numerical parity.
4. In another label-blocked process, issue technical execution evidence only
   after the registry, probe and audit all validate. A fresh fold worker then
   installs the fold-specific training-label guard before checking evidence
   and loading only that fold's training targets. Six models and optimizers
   reuse the existing shared-input update primitives with isolated RNG states.
5. Save all five folds' six checkpoints and run all 30 held-out fixed-read
   prediction sets with every label blocked. Verify the complete prediction
   matrix, patient/fold ownership, same-arm baseline, source hashes, geometry,
   view restoration and source-bound phase ledger before target release.
6. Score the frozen predictions using existing project metrics and repeat the
   scoring with the independent CPU metric implementation. Report present
   Dice, labeled-empty false alarms, patient/frame denominators, seq23 full
   and viewport views, and the uninitialized seq1/seq7 separately.

The supervisor holds the existing shared study lock and uses a 26 GiB free
disk floor, at least 4 GiB system-available RAM and at most 6 GiB child RSS;
the MPS probe and fold receipts record actual process high-water RSS. Sampled
free RAM/disk and process RSS are safeguards, not guaranteed hard resource
limits. Each MPS/training entry point requires the recorded live supervisor
parent, source-bound manifest, lock and expected phase/output. Partial runs
must remain in Drive when a guard stops a child. Separate resumability and
later full-video evaluation remain to be reviewed.

The worker, probe, predictor, scorer and supervisor source files—including
their disabled/enabled settings—are included in one source identity. After
independent review, freeze the **entire** intended source/flag set before a
real MPS probe; the supervised run archives and rechecks those exact source
bytes. Any later source change, even only an enable flag, invalidates
the probe/checkpoint evidence and requires a new preflight and probe; never
silently carry a receipt across that change.

This study does not evaluate live-video propagation. All available patients
have been inspected, patient identity is user-reported, seq23's crop was
chosen retrospectively, and no independent external patient cohort exists.
The bounded seq6 comparison cannot prove eager equivalence for the seven
cache-release sequences. The earlier `bounded_crossfit_handoff_v1` is a
different format and is not accepted by this trainer.
