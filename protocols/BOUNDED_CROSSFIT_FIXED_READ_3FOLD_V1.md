# Exploratory three-fold fixed-read comparison

This protocol replaces the five-fold and 30-prediction schedule in
`BOUNDED_CROSSFIT_FIXED_READ_V1.md` for this separately versioned run. It
otherwise retains that protocol's source-bound capture evidence, label guards,
MPS probe and independent audit, resource limits, freeze-before-score rule and
limitations. The five-fold artifacts remain preserved and must not be used as
three-fold evidence.

The predeclared held-out groups are:

- Fold 1: seq1, seq2, seq3, seq4, seq11, seq15, seq16, seq22.
- Fold 2: seq6, seq7, seq8, seq10, seq12, seq17, seq18, seq20.
- Fold 3: seq5, seq9, seq13, seq14, seq19, seq21, seq23.

Each fold contains seven initialized sequences. Seq1 and seq7 have no automatic
initialization and contribute no invented samples or segmentation Dice. The
source-name pairs identified in the project remain in the same group. The
user reports that sequences represent different patients; this identity has
not been independently verified.

In each fold, the other two groups supply training patients. There is no
validation split or held-out checkpoint selection: all six arms use the same
fixed ten-epoch, patient-balanced update schedule. The 21 initialized patients
are each evaluated in exactly one held-out fold. Every one of the 3 x 6 = 18
held-out prediction sets must be frozen and independently verified before any
held-out labels are opened. The same MPS probe and source/label protections
apply before training; no CPU/MPS bit identity is required.

With the existing 12 updates per training patient per epoch, this schedule
has 5,040 shared-input updates, or 30,240 individual arm updates. It reduces
the fixed-read training work by one third from the earlier five-fold schedule,
not by two fifths, because each three-fold model trains on more patients.
Actual runtime, cache behavior and memory use remain unmeasured.

All videos have been inspected, the seq23 viewport was chosen retrospectively,
and no independent external cohort exists. Three-fold results are exploratory,
not a blinded or confirmatory estimate; this fixed-read study does not measure
live-video feedback. A separate warm-matched full-video evaluation is needed
before making propagation or overall-comparison claims.
