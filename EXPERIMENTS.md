# Experiment record

## Current LiVOS-gated RDE architecture

The current recurrent-memory method uses a LiVOS-style channel forget gate and
the RDE-VOS spatio-temporal aggregation path:

```text
g_t = mean(sigmoid(Conv1x1(F_t)), spatial_dims)
retained = g_t * S_(t-1)
S_t = RDE_Fusion([retained, C_t])
```

`F_t` is the current MedSAM2 image feature, `S_(t-1)` is the recurrent spatial
state, and `C_t` is the current MedSAM2 mask-memory candidate.

The gate is channel-wise and acts on the previous state before fusion.
`RDE_Fusion` performs non-local spatio-temporal extraction, residual ASPP3D
enhancement, and a final `2 x 3 x 3` Conv3D squeeze from two temporal slots to
one recurrent state.

There is no post-fusion addition of `S_(t-1)`, so the previous state is used
only once in the update.

## Training protocol

MedSAM2 is frozen during recurrent-memory training. The recurrent state is
updated from predicted masks, and Dice+BCE supervises future-frame predictions.

The learned gate is derived from the current image feature. Decoder-predicted
IoU is diagnostic only and is not an input to the gate.

Training sequences:

```text
seq2-seq6, seq8-seq15
```

Validation sequences:

```text
seq16-seq19
```

Held-out test sequences:

```text
seq20-seq23
```

`seq7` is excluded when the frozen YOLO prompt protocol produces no valid box;
it is not replaced by an oracle prompt.

## Current checkpoint

The current implementation uses checkpoint format:

```text
adseg_livos_rde_v1
```

Only checkpoints produced by the current LiVOS-gated RDE implementation should
be used for training resume or inference.

## Required recurrent-memory evaluation

The required sequence is:

1. train/inference parity check;
2. training on the frozen training split;
3. validation on `seq16-seq19`;
4. fixed-gate and ungated RDE ablations;
5. held-out evaluation on `seq20-seq23`;
6. retained-state storage and FPS measurement.

No numerical result from a superseded recurrent-memory architecture is treated
as evidence for the current implementation.

## REFUGE2 measurement-pipeline preparation

REFUGE2 is a static engineering proxy for the future adenoid two-region
segmentation-to-measurement pipeline. Inference predicts disc and cup from
GT-derived oracle boxes.

The evaluator derives a non-overlapping rim:

```text
rim = disc AND NOT cup
```

and reports Dice/IoU for `disc`, `cup`, and `rim`.

Three structural measurements are evaluated:

```text
vCDR           = vertical_diameter(cup) / vertical_diameter(disc)
cup_rim_ratio  = area(cup) / area(rim)
cup_fraction   = area(cup) / (area(cup) + area(rim))
```

For each ratio, report MAE, RMSE, Pearson, and Spearman against ground truth.
The per-image CSV stores predicted/GT ratios and absolute errors. For the two
area-ratio proxies, the summary also reports Pearson/Spearman association
between cup/rim Dice and absolute measurement error.

These area-ratio experiments verify the software path that will later support
candidate adenoid measurements such as `adenoid / airway` and
`adenoid / (adenoid + airway)`. They do not establish anatomical equivalence or
clinical validity for adenoid hypertrophy.

Run:

```bash
python3 scripts/run_refuge2.py \
  --split val \
  --output_dir outputs/refuge2_val
```

Outputs:

```text
outputs/refuge2_val/metrics_per_image.csv
outputs/refuge2_val/summary.json
```
