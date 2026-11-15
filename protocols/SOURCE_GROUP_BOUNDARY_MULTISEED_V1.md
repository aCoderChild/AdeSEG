# Source-group boundary-factorial additional seeds

This exploratory replication is specified after the first seed's validation and
video outcomes were inspected. It is not preregistration, independent validation,
or evidence of patient independence. The two additional fixed seeds are 20261125
and 20261135; the prior 20261115 result is retained as a separately sourced run.
All six region/no-contrast/mask-only × BCE/boundary arms share the existing 164
training and 50 validation reads, 48,005 parameters, ten epochs, optimizer,
and per-seed initialization and order. For each epoch the original default_rng(seed + epoch) method selects the same
permutation in every arm. The three seed ranges do not overlap over ten epochs.

The capture manifest and each seed's plan bind current source, seed, sample
identities, initialization, order, checkpoint, environment and MPS settings.
Training uses only training labels. All twelve arms must freeze their validation
predictions, and both six-arm no-label replay verifiers must pass, before either
seed's scorer may open a validation label. Independently recheck both scores and
report every seed and paired boundary-minus-BCE outcome; do not select the best
seed. A label guard covers CPython opens, not arbitrary native-code I/O.

Output is restricted to the user's Drive experiment folder. The launcher guards
the existing 22 GiB reserve before each phase, monitors free space throughout,
and aborts on failures or changed sources; an already running phase is not
interrupted by the monitor. The capture lock does not control unrelated GPU work.
