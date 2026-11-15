# Exploratory CMSA-inspired boundary objective

This is a proposed training-only ablation, not a trained model, finished
experiment or a claim that boundary errors will be fixed. The implementation
is `modeling/cmsa_boundary_training_loss.py`. Its inputs are prediction logits
and training targets at the same native video resolution. It has no inference
path, no test-time labels, no new prompt or redetection, no presence gate and
no new memory-token pruning.

## Rationale and relationship to CMSA

CMSA-Net's released `structure_loss` builds a spatial weight using a
31-pixel local average, then combines weighted BCE and weighted IoU. Its code
uses the legacy keyword `reduce='none'` for BCE; on the current PyTorch this
returns a mean scalar, so that released code does not apply the spatial
weights to individual BCE pixels. We do not know what runtime or loss was
used to train the released checkpoint, and this observation does not alter
its inference. UACANet's similarly named loss computes its IoU term per pixel,
which also differs from aggregated mask IoU.

The proposed function explicitly computes corrected per-pixel weighted BCE,
the CMSA-style aggregated weighted IoU, and the project's existing soft Dice.
It preserves the author's box-average border treatment. Two separable pools
replace the 31x31 pool, with numerical equivalence checked on CPU. A 31-pixel
window on native 1080x1920 frames occupies a smaller fraction of the image
than on a 352x352 model input. No claim of architectural equivalence to CMSA
is made. Suppressed -1024 logits can have finite loss and gradients while
still being difficult to recover; this objective does not by itself address
absence decisions or guarantee recovery.

## Prespecified feasibility study

Use the same frozen nine-sequence source-group manifest as the corrected
V2 fixed-read study: seven training videos, 164 reads; two previously
inspected development videos, 50 reads. No final test, protected target
sequences, or new validation-label selection. Patient, source and detector
pretraining independence remain unverified. Use a matched 3x2 factorial:
region, no-contrast and same-capacity mask-only refiner arms, each with
ordinary BCE+Dice or the corrected boundary loss. All six arms use identical
seed, initialization, order, optimization, ten epochs and examples. Compare
each boundary arm with its ordinary-loss counterpart and compare the three
mechanisms within each objective. The new loss changes both boundary
weighting and the IoU term, so a benefit cannot be attributed uniquely to
either component. Prior V2 runs remain separate historical evidence and the
native output is the unmodified baseline.

Freeze and source-bind code, loss formula, manifest, optimizer and complete
sample order before training. Verify CPU/MPS loss and gradients on these
**training-only** reads: seq2 frame 4 (present), seq4 frame 18 (empty label),
and seq4 frame 24 (present, native logits all -1024). These properties were
checked using training data only before running this objective. Match feature/head capture,
native-to-video resize and per-pixel threshold to V2. Compare deterministic
parameter initialization, loss/output/gradient and native-sized masks across
devices under the documented tolerance; abort on non-finite values or failed
parity. Preserve the MPS capture device, precision and software environment.
Pin the adapted loss source and its parent formula. An output or parameter
hash alone is not a substitute for independently validating the manifest.

Only after all arms freeze all 50 predictions, independently replay the
frozen masks under a guard that blocks the validation labels. Then score
native and fitted masks using a separate process. Report per-video and
equal-video present Dice, precision/recall, empty-frame count and false
alarms/area, plus zero-overlap counts. The prespecified boundary F measure
uses 3x3 inner contours (zero beyond the image) and an OpenCV 11x11 elliptical
kernel with a five-native-pixel radius. Five pixels is approximately 0.26% of
a 1920-pixel frame width and is not a calibrated physical distance. It is
not optimized using the final test. Existing
observed seq11/20/23 results remain exploratory. No video rollout is
authorized solely by a training-loss change; the existing fixed-read
promotion criterion, empty-frame behavior and live-hook identity tests
still apply.

## Implemented verification

CPU tests check exact dense-31x31 equivalence within floating-point tolerance
for values and gradients, empty/full/present targets, invalid shapes/ranges,
and finite loss/gradients with -1024 logits. On the available machine the
synthetic 1080x1920 CPU forward/backward completed in about 0.054 seconds;
this is a single engineering timing, not a training benchmark. No real-data
parity, training, frozen validation output, validation score or propagation
has been run for this objective. The small synthetic CPU/MPS loss/gradient
test passed, but must not be mistaken for actual-read parity.
