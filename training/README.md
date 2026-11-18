# Training

There are **two different training pipelines**, with different backbones,
feature captures and checkpoints.

## MedSAM2 — high-resolution adapter

```bash
python -m scripts.train_med --help
```

- Backbone: SHA-verified MedSAM2 Hiera-T/512, **frozen**.
- Learnable module: `adeseg.adapters.high_resolution_skip`.
- Losses: `adeseg.losses.segmentation` and training-only
  `adeseg.losses.teacher_kl` (detached native teacher).
- Inputs: frozen TRAIN decoder captures with aligned segmentation targets.
- Status: a **reconstruction** of a lost historical implementation; old
  four-sequence scores are not reproduced by this command.

## Lean-SAM2 — multi-scale feature aggregation

```bash
python -m scripts.train_lean --help
```

- Backbone: Lean-SAM2 / SAM2.1 Hiera-B+.
- Learnable module: `adeseg.adapters.multiscale_channel_attention`.
- Losses: `adeseg.losses.segmentation`; optional positive preservation.
- Inputs: TRAIN-only `early`, `deep`, `logits` and `target` tensors.
- The structured refiner subclasses the original checkpoint-bound class
  and retains its parameter names and strict source-hash loader.
- Status: no verified 21-sequence improvement from learned fusion.

**Leakage warning:** requiring manifest rows marked `split: train` is a
format check, not verification of patient-level data separation. Seal and
audit source-group IDs before using either trainer for a research claim.
Never supply target masks to the test-time forward pass or memory writer.
