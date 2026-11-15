# Raw decoder candidate diagnostic: training reads only

This retrospective fixed-read diagnostic is not an inference method, a video
experiment, or independent evaluation. It changes no detector, prompt, object
presence decision, pointer or memory. Training labels were used only after
the decoder candidate tensors, thresholded masks and automatic candidate
choices were frozen. Frame selection used already inspected training outcomes.

The original captured `_forward_sam_heads` inputs specify
`multimask_output=True`; three decoder candidates are part of the original
native SAM2 call. The diagnostic made one offline replay per saved read with
the **same inputs and model** and an observational forward hook on
`sam_mask_decoder`, which captures the raw candidates before the existing
hard no-object suppression. All seven post-suppression outputs matched their
captured tensors, and each resulting native video mask matched the archived
trajectory mask. The decoder's own predicted-IoU argmax is reproduced exactly.
No second candidate-generating decoder mode was introduced.

On training seq4 frame 24, the object score is -0.12245 and SAM2's native
selected output is suppressed to -1024, yielding an empty mask. The original
predicted-IoU values are approximately `[0.000032, 0.010485, 0.011639]`, so
the native ranking chooses candidate 2. Its *unsuppressed*, hypothetical mask
has Dice 0.0115. Candidate 0 overlaps 160,838 of 254,326 GT pixels and has
Dice 0.7690. This label-derived oracle value is not an automatic result: the
native object score still suppresses all candidates, and choosing the best
mask by GT is unavailable at inference.

Two additional automatic, label-free scoring diagnostics were applied to the
frozen raw masks before reading training labels: soft mask-weighted cosine to
the automatic initial prompt's early-FPN foreground prototype, and its
foreground-minus-background counterpart. Both selected candidate 0 on this
particular frame, but their *hypothetical unsuppressed* mean Dice across all
17 labeled-present training reads was 0.7511 and 0.7415, below the raw
predicted-IoU selection's 0.7722. The native post-suppression mean was 0.7716.
The retrospective label oracle's mean was 0.8273. All methods, including the
oracle, produce false-positive masks on both labeled-empty reads (2/2). On
training seq4 frame 21 none of the three raw candidates overlaps the label.
Thus the single successful case does not establish a reliable selection rule,
justify changing presence behavior, or demonstrate improved propagation.

Drive artifacts under `outputs/experiment/contrast_refinement_v1/`:
`train_raw_decoder_diagnostic_attempt02/` and
`train_raw_candidate_selection_audit_attempt01/`. The latter stores frozen
automatic score choices, per-candidate masks, complete scores and source
hashes. Scripts: `scripts/diagnose_lean_raw_decoder_train_reads.py` and
`scripts/audit_lean_raw_candidate_selection.py`; five focused tests pass.
