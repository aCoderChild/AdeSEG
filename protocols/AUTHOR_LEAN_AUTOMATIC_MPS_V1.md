# Automatic-prompt Lean author-system comparator

This is an exploratory comparison on the 21 initialized PolypGen sequences.
It uses the existing automatically generated masks and initialization frames,
without detector calls, annotation-based initialization, or later redetection.
Sequence 23 uses the previously selected fixed main-view crop; masks are
restored to full-frame coordinates before evaluation. Sequences 1 and 7 had
no automatic initialization and are reported separately, not silently
omitted from detector coverage.

The method builds the pinned clean Lean-SAM2 author predictor, config, trained
bypass and SAM2.1 B+ checkpoint. It does not install the project's token window,
anchor or refiner. The existing indexed spatial-RoPE adapter and scoped blocking
pointer transfer are applied for MPS compatibility. Indexed RoPE changes the
author attention implementation, so this is a **compatibility-corrected Lean
author-system adaptation**, not an unmodified or byte-identical author runtime.
No performance equivalence with CUDA is asserted.
The released Lean encoder configuration itself enables its low-confidence
full-encoder fallback; this inherited behavior is retained and reported as part
of the author system. The adapter adds no confidence rule of its own.

The first MPS smoke uses seq6's nonzero initialization and the second uses
seq23's crop. Each produces five frames including initialization. A separate
CPU verifier reconstructs their input, source, loaded-state and binary-output
identities before any full video runs. These two smokes do not prove all video
geometries and prompt timings will be numerically equivalent. Every full video
independently checks the exact input, finite logits, output geometry, frame
coverage and unchanged model weights. All 21 videos must be frozen and verified
before any annotation mask is opened for scoring; pre-prompt frames are empty,
and the initialization frame is excluded from the post-prompt segmentation
summary. Report labeled-empty false alarms, pre-prompt misses and denominators.
Independent metric recalculation and patient-macro versus frame-weighted scores
are separate from the trajectory checks. These patients were inspected before
this study, and no independent clinical or confirmatory claim is appropriate.
