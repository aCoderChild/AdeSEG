# Fixed-read multiscale and contrast controls

These controls complete part of the exploratory fixed-read design; they are
not a new video system or an independent test. They reuse the already audited
214 source-group reads, with the same seven training and two previously
inspected validation sequences as the v2 study. Patient, video-source and
detector-training independence remain unknown. No seq8/11/20/23 target labels
or images determine training, selection, features, or model inputs.

All six arms have the same 48,005 parameters, initial state, sample order,
10-epoch seed, optimizer, input geometry and loss as the v2 region arm. They
reuse the TCMN-derived CBAM gate, early/deep projections and the same residual
head. The module computes channel weights using foreground/background
descriptors; it reweights channels rather than mixing output channel values.
The native mask is evaluated alongside all frozen predictions. A no-op at
initialization and an explicit matched `reference` rerun make the comparison
auditable. These controls do not add a redetection, inference-time confidence
threshold, new pruning rule, or target-label input.

| Arm | Prespecified change from `reference` |
|---|---|
| reference | The exact foreground/background-conditioned v2 region mechanism. |
| early_only | Zero the deep input before its existing projection; the projection's bias and capacity are retained. |
| deep_only | Zero the early input before its existing projection; its bias and capacity are retained. |
| foreground_only | Supply a zero background descriptor to the author gate, and use the foreground cosine instead of foreground-minus-background cosine in the contrast slot. This changes both uses of the background prototype together; local feature maps still contain background information. |
| rgb_contrast | Keep SAM2 features and the region gate, but replace the one-channel feature-contrast input to the head with foreground-minus-background RGB cosine contrast. This is an RGB-*contrast* control, not an RGB-only network. |
| spatial_shuffle | Apply one fixed-seed bijective permutation of aligned projected feature pixels, shared by all channels and the early/deep groups, while leaving logits in place. This is an alignment-destroying control, not the earlier invertible channel permutation. |

For RGB, the hash-bound source image is decoded as RGB using the same PIL
conversion and 1024-square resize as the captured SAM2 loader, converted to
float32 [0,1], and bilinearly resampled to the logit grid. Its color contrast
uses the prediction's continuous foreground/background weights and the same
finite epsilon convention as feature contrast. The SAM2 backbone features
still use their captured original preprocessing. Source image geometry and
SHA-256 are checked before any RGB read; no ground-truth mask selects samples
or supplies prototypes. `foreground_only` and RGB cosine maps can have
different numeric distributions from the reference: matching capacity does
not establish matching input statistics.

The spatial permutation uses a fixed CPU generator seed (20261115), generated
from feature-grid dimensions and moved to the feature device. It never depends
on labels, frame identity, logits or outcomes. Applying it to aligned features
retains each channel's marginal distribution and destroys pixel-to-mask
alignment, although spatial positions can be accidentally fixed points. It
does not preserve local image geometry and should not be described as an
anatomically realistic perturbation.

Before any results are joined with validation masks, require source-bound
plans, real training-only CPU/MPS numerical parity for all six arms, matching
initial parameter identities and sample schedules, frozen validation outputs,
and independent label-free replay of every saved mask. Report native and all
fitted arms, per-video and pooled present Dice, the 3 labeled-empty cases,
and all failures. Seq9 has 22 labeled-present and three labeled-empty reads;
seq15 has 25 labeled-present and no labeled-empty reads. Neither sequence is
a patient-independent held-out test. These controls were specified after v2
scores had already been inspected, so all findings remain exploratory.
One seed cannot support uncertainty, clinical claims or seed robustness.
