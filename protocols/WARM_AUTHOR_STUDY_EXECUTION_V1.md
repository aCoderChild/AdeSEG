# Warm-matched author study execution

This document supplements `WARM_MATCHED_AUTHOR_COMPARATORS_V1.md`; the latter
and its frozen input preflight remain unchanged. Four methods are planned on 21
initialized sequences: native SAM2, author Lean with scoped MPS compatibility
fixes, SAMURAI and offline SAM2Long. A fresh process is used per trajectory.

For each method, two five-frame MPS smokes exercise seq6's nonzero prompt and
the cropped seq23 view. A third five-frame seq6 run omits only the additional
feature-access/prototype callback. The independent control audit compares its
per-frame output logits hashes and masks with the callback-enabled smoke;
differences are reported and do not trigger an equality gate. This is a
same-device short diagnostic, not exhaustive internal-state equivalence or
CPU/MPS parity. All 84 full trajectories keep the callback enabled; its
prototype values are not used to select prompts or alter logits. Its additional
feature access may affect runtime cache/state and must be evaluated.

The launcher freezes its source closure, prompt inputs, model assets and
per-phase receipts, monitors the disk reserve, and keeps partially completed
work on a failed phase. The verifier requires the 12 short trajectories,
four smoke gates, four feature-control audits and all 84 full outputs before
any target mask is read. The scorer excludes initialization from post-prompt
segmentation summaries, records pre-prompt misses, empty-frame false alarms,
predicted area, precision, recall, boundary measures and patient/frame-weighted
Dice. It reports seq23 both restored to full frame and within the fixed viewport.
The independent CPU audit recalculates metrics but shares structural checks.

Wall times include staging, warm prompting, feature telemetry, propagation,
synchronization and mask writes; they exclude model loading. They are not a
real-time throughput benchmark. SAM2Long's final-path output is offline.
These patients and settings have been inspected; results are exploratory, not
an independently blinded or clinical evaluation. No MPS execution or model
quality result is asserted by this document alone.
