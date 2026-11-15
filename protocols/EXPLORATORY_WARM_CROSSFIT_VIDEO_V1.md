# Exploratory full-video follow-up for warm five-fold checkpoints

This is a draft implementation contract, not an experiment result. It must be
frozen before execution. It uses the already specified channel-value
`foreground_background_asinh` arm rather than choosing a model from held-out
scores. All six architecture controls remain in the fixed-read study. The
patient grouping is user reported and all sequences were previously examined;
these experiments are exploratory, not an untouched independent test.

For each initialized seq2–6, seq8–23, load only that patient's out-of-fold
checkpoint, trained on neither that patient's test group nor validation group.
Report seq1 and seq7 as automatic-initialization failures with no invented
mask-based present Dice. Hold every checkpoint and video-route prediction
fixed before opening any target labels for video scoring. No new detection,
redetection, presence gate, or token-pruning strategy is added.

Every branch starts from the same source-bound automatic box, then two stored
points (positive, negative), then the exact archived automatic mask on that
same state. Seq23 uses the previously fixed main viewport [446,0,1280,720];
predictions are restored to the source canvas for primary scoring and the
viewport score is separately reported. No evaluator supplies a target mask at
inference. Reuse of an archived warm Lean A is permissible only after
source/checkpoint/runtime, frame, prompt, conditioning, binary mask and
recorded token/position/anchor trace lineage is verified. The nine historical
references record masks and named trajectories, not all compact memory state;
never label that as complete memory-state parity.
The special seq6 historical reference predates saved conditioning and
label-guard fields. Its conditioning is verified through the later
phase-bound warm gate capture and its matched recorded trajectory. The
historical receipt itself cannot establish a recorded zero-open guard.

The warm A is the project Lean token-window runtime (16,384 total spatial
tokens, 3,072 insurance reserve and a 0.5 multiscale anchor), not unmodified
author Lean. Its scores and the new refiner scores share the warm runtime;
label author Lean and other methods according to their own verified setup.
The budget counts persistent spatial-feature tokens and reserves insurance
capacity; pointers, masks, positional tensors, caches and transient
allocations are outside this count, so it is not a total-memory-byte limit.

The controlled Lean routes use the same fold checkpoint, memory policy,
checkpoint, input frames and warm prompt. A is warm Lean; B changes output only;
C changes only the mask fed to the native memory encoder; D changes output and
encoder mask. Native object score/pointer and no-object handling are not
replaced by the refiner, so a positive corrected mask can coexist with an
absent representation. Verify B native masks and recorded traces against A;
where stored states are not available, explicitly report the limitation.
C and D may legitimately change future trajectories. Test conditioning-frame
protection, exact/no-op routing and geometry, finite values, source and
checkpoint integrity in a label-free MPS smoke before full-video branches.

Use separate fold-aware training, prediction and scoring receipts; enforce
one checkpoint per patient and independently verify all predictions before
opening video labels. Score on the same matched frames and distinguish
initialization frame, all-frame and post-prompt metrics, present-frame Dice,
empty-frame false alarms, patient-macro and frame-weighted summaries. Check
source labels by hash at scoring and recompute metrics independently. Do not
read target-derived outcomes during inference or tune using held-out results.

Previously completed direct-initialization native SAM2, Lean and SAMURAI
runs are contextual: warm and direct initialization can yield different
histories. Their author APIs expose compatible box/point and mask calls, but
that alone does not establish matching internal state. A fair matched
comparison requires source/config/checkpoint-bound warm runs and a labeled
same-input protocol audit. CMSA-Net is a separate architecture and its own
reference-frame/preprocessing requirements must also be disclosed rather than
assumed to match the SAM2-family propagation protocol.
