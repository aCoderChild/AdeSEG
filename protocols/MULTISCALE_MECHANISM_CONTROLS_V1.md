# Separately versioned mechanism controls

These controls apply the existing foreground/background channel-value refiner
with the asinh head-logit representation. The original module, six-arm study,
checkpoints, optimizer, loss and prompt policy are not changed. Each control
must be trained separately in the same patient-disjoint folds, with the same
training schedule, captures and initialization, before it is evaluated.
Implementation alone is not an experiment result.

* `early_only`: the deep feature input is zeroed, leaving early features.
* `deep_only`: the early feature input is zeroed, leaving deep features.
* `spatial_shuffle`: a deterministic, fixed, bijective permutation is applied
  separately to each scale's spatial positions, equally for every channel and
  batch item. The unchanged logits remain in their original coordinates.
* `rgb_contrast`: keep the full feature branch and attention; replace only
  the head's contrast channel with RGB foreground-minus-background cosine
  contrast. It uses the same soft predicted probabilities and spatial
  pooling/interpolation as the learned-feature contrast. RGB is decoded as
  sRGB floats in [0,1] from the hash-verified same source frame and fixed view.

All controls allocate the baseline's exact parameter tensors, but zeroed
branches have fewer effective input-dependent degrees of freedom and their
projection biases remain trainable. The shuffle preserves feature values and
each channel's marginal distribution, but disrupts both local spatial layout
and correspondence between scales and with the mask. It is not a pure test of
only one of these properties. No inference-time labels or new prompt decisions
are used. New checkpoints have a distinct format and bind both the control and
baseline source versions. Preflight, fold-specific checkpoint routing, an
MPS probe and real inference still require separate integration and review.

## RGB control feasibility

The frozen index identifies original frame paths and hashes, and seq23 uses
the fixed main-viewport crop. Thus a same-frame RGB contrast comparator is
feasible by re-reading the source image, checking its hash and applying that
same crop and alignment before computing any contrast statistic. Use predicted
probabilities, not labels, for foreground/background summaries; open labels
only for the prespecified training or evaluation phase. Color, illumination
and preprocessing may drive the signal, so it is an independent comparator,
not evidence for learned-feature attention.

RGB is loaded separately from the frozen source frame, not from the saved
feature snapshots. The control adds no learned RGB projection and retains
the baseline's parameter tensors; three RGB channels are not an equivalent
representation to learned high-dimensional features. The head input is
replaced by a temporary, per-call forward hook protected against concurrent
or reentrant calls; it is removed even if an exception occurs. This adapter
requires independent review and MPS probing before training or inference.
Source-image availability and frame/ROI hash checks must be verified in the
final capture and training preflight before use.
