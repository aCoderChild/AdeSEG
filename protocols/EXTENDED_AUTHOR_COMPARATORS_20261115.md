# Additional automatic-prompt author comparators

This exploratory extension evaluates native SAM2.1, author SAMURAI and
SAM2Long on seq2–6, seq8–10, seq12–19, seq21 and seq22 (18 sequences).
An explicitly scoped seq10-only batch is permitted first to compare the user's
prioritized difficult case; it freezes all three author predictions before any
new scoring. Its outcomes are exploratory and were previously inspected.
It uses exact archived automatic masks and source frames. Nine nontraining
sequences already have audited A/B/C/D project Lean outputs; training and
validation sequence A/D video rollouts are a separate pending comparison.
Sequence 23 remains in the earlier
three-video comparison and is not rerun here. Source IDs are user-reported
distinct patients; the videos or baseline results have been inspected.
Seq2/3/4/5/12/13/14 are refiner-training patients and seq9/15 were used for
refiner validation; author comparator masks can be evaluated on them, but
refiner scores for these nine are in-sample or selection-exposed and must not
be represented as independent evaluation. Seq1/7 had no automatic detection
and are separately reported as all-frame detector outcomes without any model
initialization, not as post-prompt segmentation scores.

For each method, source-bound engineering gates use seq6 (1920×1080, prompt
ordinal 23), seq8 (1920×1080, prompt ordinal 0), seq17 (1280×720, prompt
ordinal 16), seq5 (1440×1064), seq12 (1280×1024) and seq15 (1704×1072).
Each requires one CPU and two independent MPS short-clip runs
with identical source frames, prompt and model state. MPS repeat disagreement
may be at most 0.01% of image area per frame; CPU/MPS disagreement is recorded,
not treated as exact equivalence. These are engineering checks, not evidence
that devices or full trajectories are equivalent. The full run checks its own
source-frame identities, exact automatic prompt, model state, numerical
environment, per-frame finite outputs and coverage independently, and uses
the representative gate for its image geometry. Prompt ordinals differ for
other sequences; it does not claim a separate repeat test for every sequence
or an equivalent SAM2Long offline pathway. SAM2Long
selects its path after observing the complete staged clip and is offline; do not
compare its short-clip output to a full-video prefix. SAMURAI retains its author
configuration and thresholds; the MPS Kalman conversion and native SAM2 pointer
transfer are the previously audited scoped compatibility fixes.

The detector is not called. Historical YOLO scanned until its first detection
and the saved automatic mask initializes each model. All full predictions in
the chosen predeclared scope (54 for all 18 videos, or three for seq10-only)
must be frozen before a separate scorer opens any labels. Compare the same
post-initialization frame IDs against the existing project Lean A predictions;
report pre-prompt frames separately. The project Lean baseline is a modified
token-window system, not unmodified author Lean. This is not a blinded or
independent final test; detector training exposure and near-duplicates remain
unresolved. Results and temporary staged frames belong only under the user's
Drive experiment folder. No new inference gates, thresholds or redetection are
introduced by this adapter.
