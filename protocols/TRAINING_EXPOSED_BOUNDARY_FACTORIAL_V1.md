# Training/selection-exposed boundary-refiner video extension

This descriptive extension uses the already selected 20261115 region/boundary
checkpoint on seq2/3/4/5/9/12/13/14/15. Seq2/3/4/5/12/13/14 contributed
training reads; seq9/15 contributed checkpoint-selection validation reads.
These results are in-sample or selection-exposed, not held-out evidence.

Use the exact existing archived automatic YOLO prompt, project Lean runtime,
source frames and memory policy. No new detector, redetection, inference gate,
threshold, pruning rule or ground-truth input is introduced. All outputs and
temporary staged frames remain under the requested Drive experiment folder.

Before full videos, run independent short A/A-repeat/B/C/D trajectories on
seq2 and seq15, then a source-bound gate and separate CPU artifact audit. Both
must match the already verified original learned-checkpoint runtime. These are
engineering tests, not efficacy selection. Full runs use A/B/C/D on all nine
videos (36 trajectories); all 36 and their logs must finish before any scorer
opens a label mask. An independent CPU metrics audit follows the primary score;
it shares the structural verifier and says so explicitly. Output and reference
mask hashes must be checked before and after decoding.

Report each arm's post-prompt and all-frame Dice, present zero-overlap, labeled-
empty false alarms and predicted area, plus pre-prompt misses. Initialization is
excluded from post-prompt metrics. Record the user's distinct-patient mapping,
prior selection/exposure, unknown detector and backbone training data, and
possible related source stems. These numbers cannot be pooled with held-out
results and described as unbiased generalization or clinical performance.
