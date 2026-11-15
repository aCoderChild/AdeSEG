# Observer-only cross-fit capture V2

V2 retains the separately source-bound, independently audited V1 short gate on
seq6 and seq23. It runs one observed MPS pass per full sequence, rather than a
second fresh full reference. Each observed frame is checked directly against
that sequence's previously audited archived automatic-prompt Lean A run. The
comparison covers saved prediction-mask and output-logit hashes, native
`pred_masks`, `maskmem_features`, `obj_ptr` and `object_score_logits` tensor
hashes, plus memory events, positions and selection traces. It also checks
full source-frame and automatic-prompt identity and the live runtime identity.
These comparisons do not establish equality of every predictor internal
(e.g. compact payload or `maskmem_pos_enc`) and do not add a full-video fresh
MPS repeat. The short seq23 observer tensors were not independently repeated.
The historical seq6 observations at frames 24 and 90 must match.
All predictions and reads are label-free; seq23 uses its frozen viewport.

Each full sequence runs as an independently audited, resumable batch. Only
phase-bound prior audits with their complete intact feature/snapshot files
reduce future uncompressed payload reservations. Full 143-read aggregate
publication is permitted only when all twelve sequences pass. Runtime space
checks reserve 22 GiB plus 4 GiB margin and the remaining estimated payload;
no important artifacts are deleted. The user-supplied patient mapping and
all prior development exposure still apply; these captures are not blind test
results. Seq1 and seq7 have no initialization and no invented observations.
