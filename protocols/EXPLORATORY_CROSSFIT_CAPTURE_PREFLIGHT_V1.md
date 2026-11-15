# Exploratory cross-fit capture index preflight

This is a label-free capture plan, not model training, propagation, validation,
or evidence of generalization. The user reports each sequence is one patient;
independent patient, examination, detector-training and backbone-exposure
records are unavailable. Prior sequence outcomes have influenced development.

Fix five outer test groups: {6,10,11,20,23}, {1,2,3,4,5}, {7,8,9,15},
{12,13,14,16,17}, {18,19,21,22}. Each fold uses the following group cyclically
as validation and the other three as training. Known suspicious common filename
stems remain in the same group; same stems do not prove common identity and
different stems do not prove independence.

Retain 214 existing source-bound fixed reads subject to verified reuse. For
the other twelve initialized sequences select up to twelve uniformly spaced
post-initialization ordinals without labels, errors or predictions; seq10 has
only eleven. Seq1/7 have no automatic initialization in either audited replay
and receive no fabricated prompt or training capture. Keep their outcomes in
the end-to-end cohort accounting and exclude undefined present-frame Dice.

Future training must use equal total loss weight per initialized training
sequence: average its captured-frame losses, then average across sequences.
The different 25/14/12/11 capture counts do not imply unequal patient weight.
Freeze architecture, schedule, training/validation/test grouping, seed and
model initialization before each outer-fold fit. Feature tensors from an
unmodified source-bound baseline can be shared, but any learned rollout or
fitted checkpoint is fold-specific. Outer-fold labels must not enter that
fold's training or validation. Other folds can use those patients for training
under standard out-of-fold cross-fitting, with all prior exposure disclosed.

This preflight must bind the archived prompt, crop, frame bytes, original
backbone, existing captures and source hashes; screen exact and decoded-image
duplicates across the 23 sequences; and verify no CPython annotation-mask open.
Exact matches cannot rule out near duplicates or common exams. Fresh captures
still require an independently reviewed generalized observer, real CPU/MPS
parity and no-op/full-trajectory gates, source-freeze and 22-GiB free-space
reserve. Do not infer authorization for live video or full model training from
this plan. Results and temporary artifacts stay in the requested Drive folder.
