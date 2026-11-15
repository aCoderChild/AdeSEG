# Warm compatibility gate V1: late source-group read

This addendum strengthens the separately frozen warm-observer plan. In addition
to exact historical seq6 overlaps at frames 24 and 90 and the seq23 cropped
short no-op check, replay the full 15-frame historical seq3 source-group video
and compare all 14 captured decoder snapshots and early features and the full
saved masks, anchor, token-window, selection and position trajectories against
its independently audited historical full reference and capture. This includes
frame 14, so agreeing only at the first frame cannot justify reusing old data.
This is a same-MPS compatibility test, not CPU/MPS parity or efficacy evidence.
