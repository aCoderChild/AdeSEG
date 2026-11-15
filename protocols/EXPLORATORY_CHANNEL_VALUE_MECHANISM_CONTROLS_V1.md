# Separately versioned channel-value mechanism controls: design only

Early-only, deep-only, RGB contrast and spatial-shuffle controls evaluated in
older regional-refiner studies are not controls for the current six-arm
channel-value architecture or its five-fold warm-unified captures. The current
study remains frozen: no result or mechanism claim follows from this note.

A follow-up should pre-register models and training before opening its own
held-out outcomes, reuse the same verified 357 capture reads, frozen patient
folds, training order, per-fold initialization seeds, corrected loss and Adam
schedule, and preserve zero-residual initialization. Compare against the
current six controls and a re-run all-feature model under the same follow-up
architecture. Match initialized prompt, ROI, MPS device, checkpoint, targets,
attention dimensions and output decoder. Every mode should instantiate the
same modules/parameter shapes; report allocated trainable parameters and
which branches actually receive gradients. Equal allocated weights alone do not
mean equal effective capacity.

* Early-only and deep-only: retain both projection branches in each model but
  replace the ablated source tensor by zeros before its branch, leaving biases
  and all remaining layers unchanged. Compare with all-feature and mask-only
  controls. These identify dependence on each source, but an inactive branch
  has less effective information and gradient capacity.
* RGB context: a separately implemented shared RGB input encoder should be
  instantiated across *every* follow-up arm, with unused inputs set to zero,
  and its active parameter count disclosed. Load only source-bound original
  frames, crop seq23 to the identical viewport, use a frozen resize/color
  policy, and compare RGB-only, early/deep-only and combined input controls.
  This requires new computation of aligned image tensors; existing early/deep
  snapshots are not themselves RGB. Preserve labels solely for loss/evaluation.
* Spatial-shuffle negative control: specify a deterministic per-sample
  permutation independently of annotations, apply identically throughout a
  run to the intended spatial feature grid, and retain a clearly defined
  alignment policy with the logits and other feature scales. A shuffle tests
  that particular disruption of spatial correspondence; it is not a clean
  test of every possible semantic or attention mechanism. Include an unshuffled
  model with the same capacity and inputs.

The new implementation, independently reviewed source closure, deterministic
permutation, ROI alignment, matching initial state, checkpoint provenance,
patient separation, finite MPS loss/gradients and output geometry all require
unit tests and a label-free MPS probe before real training. Record empty-frame
false alarms, present Dice, per-patient differences and uncertainty; retain all
folds and outputs before opening evaluation labels. The previously inspected
patients remain development data rather than an untouched external test.
