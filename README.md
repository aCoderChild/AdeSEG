# AdeSEG

AdeSEG is a research codebase for video-level adenoid hypertrophy assessment from nasopharyngoscopy. The target task is two-region segmentation (`adenoid`, `nasopharynx_airway`) followed by temporal measurement and grading. PolypGen and REFUGE2 are proxy datasets used to verify separate parts of the implementation.

## Project roles

- `datasets/`: dataset adapters for PolypGen, REFUGE2, and the future adenoid dataset.
- `modeling/`: MedSAM2 wrapper, recurrent dynamic memory, and adaptive fusion.
- `inference/`: image and video inference paths.
- `training/`: custom training helpers; the main proposed training path freezes MedSAM2 and trains only adaptive state fusion.
- `evaluation/`: segmentation, temporal, structural-ratio, and REFUGE2 evaluation.
- `adenoid/`: target-task measurement and grading logic.
- `scripts/`: runnable entry points.
- `MedSAM2/`: bundled upstream MedSAM2 implementation.

## Proposed video method

The proposed temporal path replaces MedSAM2's multi-frame spatial memory bank with one recurrent spatial state. The state is updated by a learnable fusion module:

```text
previous state S(t-1)
        +
current candidate C(t)
        +
foreground probability P(t)
        |
        v
AdaptiveStateFusion
        |
        v
new state S(t)
```

`adaptive` is the proposed method. `native` and `fixed_ema` are retained only as research baselines/ablations so the contribution can be tested fairly. `current` has been removed from the public backend choices because it does not add a necessary comparison for the current study.

## Proxy datasets

### PolypGen

Used to test video propagation, recurrent memory, adaptive fusion, and temporal robustness. Complete videos are split rather than individual frames. Empty-GT sequences `seq1` and `seq7` are excluded from train/validation/test splits.

### REFUGE2

Used to test two-region image segmentation and structural measurement. The zero-shot baseline uses GT-derived oracle boxes for optic disc and optic cup and reports disc Dice, cup Dice, and vertical cup-to-disc ratio error. It is not an apples-to-apples comparison with fully automatic REFUGE2 challenge methods.

## Training

The intended custom training setup is:

```text
frozen MedSAM2
    |
video clip + first-frame prompt
    |
DynamicMemoryState
    |
AdaptiveStateFusion   <- trainable
    |
propagated masks
    |
Dice + BCE loss
```

`training.video_trainer.freeze_except_fusion()` freezes the MedSAM2 parameters and leaves only `AdaptiveStateFusion` trainable. `training/fusion_trainer.py` contains the project-specific optimizer/loss entry points. The full clip sampling and backward loop still needs to be connected to the existing MedSAM2 training/data stack.

## Inference

PolypGen video inference supports:

```bash
python scripts/infer.py \
  --memory_backend native \
  -o outputs/native

python scripts/infer.py \
  --memory_backend fixed_ema \
  --fixed_ema_alpha 0.1 \
  -o outputs/fixed_ema

python scripts/infer.py \
  --memory_backend adaptive \
  -o outputs/adaptive
```

For fair method comparison, replay the same `prompt_records.json` across all three runs.

REFUGE2 zero-shot oracle-box inference:

```bash
python scripts/run_refuge2.py \
  --split val \
  --output_dir outputs/refuge2_oracle
```

## Important current limitations

- Adaptive fusion is connected to the recurrent-state predictor, but it is untrained until a fusion checkpoint is produced.
- Dynamic-state video inference currently supports one prompted object and forward propagation only.
- `native` and `fixed_ema` are baselines, not the proposed final method.
- Clinical adenoid grading thresholds and the final obstruction-ratio definition must be validated on the real annotated adenoid dataset.
