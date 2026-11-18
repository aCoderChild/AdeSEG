# Lean feature aggregation

**Contribution 2:** a learned six-stage feature refiner on the
**Lean-SAM2 Hiera-B+** backbone. It is different from MedSAM2 Hiera-T.

## What changes?

Early FPN features, deep memory-conditioned features, and native mask logits
are fed through projection, context pooling, region-aware Q/K/V channel
attention, feature fusion, foreground/background contrast, and a residual mask
decoder. The original source-bound parameter registration remains intact.

**Public API:** `adeseg.adapters.multiscale_channel_attention`.
**Training:** `python -m scripts.train_lean --help`.
**Memory routing:** `adeseg.inference.lean_memory_routing`.

Original checkpoint/source contract:
`modeling/multiscale_channel_value_refiner_v1.py`.
Original frozen method definition:
`protocols/MULTISCALE_CHANNEL_VALUE_REFINER_V1.md`.

## Evidence

Small historical pilots had mixed outcomes; **there is no established
21-sequence improvement from this learned module**. The verified
0.742071 patient-macro Dice belongs to a *separate* corrected author
Lean-SAM2 baseline: [see exact run](../reference_baselines/modified_lean_742.md).
