# Source-group fixed-read mechanism controls, V3

This is a development-only follow-up after inspecting the V2 outcomes on seq9 and seq15. It is not preregistration, independent confirmation, a trained-live rollout or a clinical evaluation. The image and feature data, training/validation grouping, detector prompts, optimizer, seed, order, epochs, loss and fixed-read metrics are the same as the source-group V2 study. Source-group filenames are not verified patient identifiers; detector exposure remains unknown.

V3 uses the exact same 48,005-parameter channel gate, early/deep projections and segmentation head in six separately initialized and trained arms:

- `reference`: regional foreground/background conditioning and learned-feature contrast, rerun in the V3 plan as the matched reference.
- `early_only`: zero deep features before the original trainable projection.
- `deep_only`: zero early features before the original trainable projection.
- `foreground_only`: use the predicted-foreground descriptor alone for channel conditioning and contrast, without background descriptor.
- `rgb_contrast`: replace only the contrast-map input with foreground/background contrast computed from the hash-checked original resized RGB image. SAM early/deep features and their region-conditioned channel gate still enter the rest of the network; this is not an RGB-only model or a capacity-matched RGB encoder.
- `spatial_shuffle`: apply a fixed seeded, bijective spatial permutation to concatenated projected features, shared by channels and samples, before comparing them to the unchanged, automatically predicted spatial mask. This destroys feature-mask spatial correspondence while preserving feature values and parameters; it is not a random permutation per video or causal proof of the semantic role of individual channels.

All runs retain the same 164 deterministic training reads, 50 previously inspected validation reads, initial parameters, ten epochs, sample orders, single seed and 48,005 trainable parameters. The plan, model, inputs, dependencies, capture identities and each source image are hash-bound. CPU/MPS parity must check all six arms on the same three training-only reads before any training. Each arm must freeze all 50 validation predictions under the train-only mask-open guard; the verifier must independently replay all 600 native/refined outputs with all manifest mask reads blocked. Only then may a separate scorer load the validation labels. The CPython guard cannot intercept arbitrary native-code I/O.

This is a mechanism-control study on reused development outcomes, not a comparison with a blind test. It does not add a new selection/pruning rule, confidence gate or redetection and does not justify full-video propagation. Any RGB contrast effect is confounded with retaining the learned SAM gating path; patient and source independence, annotation completeness and detector training exposure remain unresolved. Positive findings require independent videos and multiple seeds. Preserve V2 outputs and report failed runs rather than overwriting them.
