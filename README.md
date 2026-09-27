# AdeSEG

**A shared segmentation, temporal, and quantitative-measurement foundation for AdeSEG.**

AdeSEG compares MedSAM2's native spatial memory bank with one recurrent spatial
state. The compact state is initialized by the first YOLO box-prompted frame and
updated after each predicted mask:

```text
state_t = (1 - alpha) * state_(t-1) + alpha * candidate_t
```

`alpha=1` is current-only memory. Small alpha values retain history for longer.
The custom state keeps MedSAM2's native object-pointer tokens when reading memory.

## Pipeline foundations

The active PolypGen experiment validates prompting and temporal propagation.
The shared clinical modules support the future two-region workflows for REFUGE2
and adenoid video without assuming either dataset's file layout or clinical
grading protocol.

```text
dataset frame + named masks
    -> per-frame measurement and validity
    -> valid-frame selection
    -> robust video-level ratio
    -> configurable grade
```

`clinical.compute_ratio(region_a_mask, region_b_mask, mode="fraction_of_total")`
uses `area_a / (area_a + area_b)`. `region_a_over_region_b` is also available.
The clinical protocol must choose the final ratio and grade thresholds; the code
does not provide or infer them.

`datasets.FrameSample` is the common adapter record. The implemented PolypGen
adapter yields a named `polyp` mask. Future REFUGE2 and adenoid adapters should
provide their actual named masks through the same record.

For two-region data, `clinical.measure_frame(..., region_a_name="optic_cup",
region_b_name="optic_disc")` produces the same record shape used later for
`adenoid` and `airway`. `evaluation.evaluate_regions` reports Dice and IoU for
each named mask, and `evaluation.evaluate_ratios` reports ratio MAE, RMSE,
Pearson, and Spearman correlation.

## Methods

| Backend | Update |
|---|---|
| `native` | MedSAM2 native memory bank |
| `current` | `state_t = candidate_t` |
| `fixed_ema` | fixed EMA with `--fixed_ema_alpha` |

## Inference

```bash
DATA=data/PolypGen2021_MultiCenterData_v3/sequenceData/positive

# Create shared YOLO prompts with the main compact-state method.
python infer.py -i "$DATA" -o outputs/fixed_ema_01 \
  --memory_backend fixed_ema --fixed_ema_alpha 0.1 --device auto

# Replay the exact prompts for a baseline.
python infer.py -i "$DATA" -o outputs/native \
  --memory_backend native \
  --prompt_records outputs/fixed_ema_01/prompt_records.json --device auto
```

`--device auto` selects CUDA, then MPS, then CPU. On the development Mac, MPS
produced non-finite SAM2 memory features in this path, so use `--device cpu` if
that recurs.

## Evaluation

```bash
python utils/eval.py \
  --output_mask_dir outputs/fixed_ema_01 \
  --data_root data/PolypGen2021_MultiCenterData_v3/sequenceData/positive \
  --output_eval_dir outputs/fixed_ema_01/evaluation \
  --sequences seq16 seq17 seq18 seq19
```

The evaluator writes per-sequence Dice and IoU plus drift summaries for prompt
distances `1–5`, `6–10`, `11–20`, and `>20` frames.

## PolypGen splits

`MedSAM2/training/assets/polypgen/` stores sequence-level lists:

- train: `seq2–15`, excluding empty `seq1` and `seq7`
- validation: `seq16–19`
- final held-out test: `seq20–23`

Use validation to select a fixed alpha. Run the final test split once after
freezing that choice.
