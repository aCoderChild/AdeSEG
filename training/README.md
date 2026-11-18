# Training

There are **two different training pipelines**, with different backbones,
feature captures and checkpoints.

## MedSAM2 — high-resolution adapter

```bash
python -m scripts.train_med --split-file split.json --help
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
python -m scripts.train_lean --split-file split.json --help
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

## Required source-group and capture provenance (new public commands)

The public `scripts.train_med` and `scripts.train_lean` commands fail closed
unless a separate `--split-file path/to/groups.json` is supplied. Example:

```json
{
  "train": ["patient_A", "patient_B"],
  "dev": ["patient_C"],
  "test": ["patient_D"]
}
```

Each manifest row additionally requires `"source_group":"patient_A"`
and `"capture_sha256":"<lowercase SHA256 of the .pt file>"`.
For MedSAM2 use `"file":"captures/frame.pt"`; for Lean use
`"tensor_file":"captures/frame.pt"`. The original trainer options are
unchanged, but new public commands verify group membership, group
disjointness, capture-byte checksums and duplicate files before training.
Each completed run writes a `<checkpoint>.split.json` receipt referencing
the checkpoint's SHA256.

This is a declaration and integrity check, not proof that the supplied
group mapping is correct: an independent dataset-owner/patient-ID audit is
still necessary before making held-out efficacy claims.
