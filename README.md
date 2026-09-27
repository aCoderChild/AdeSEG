# AdeSEG

**Adenoid hypertrophy grading from nasopharyngoscopy video.**

```text
video -> prompts -> adenoid + airway masks -> temporal propagation
      -> frame validity -> obstruction measurement -> video aggregation -> grade
```

## Repository layout

```text
adenoid/          Target-pipeline components: dataset record, prompts,
                  segmentation-mask I/O, temporal backend names, measurement,
                  validity, selection, aggregation, and grading.
models/           MedSAM2 wrapper and compact recurrent EMA memory.
evaluation/       Per-region segmentation, temporal PolypGen, and ratio metrics.
proxy_datasets/   PolypGen adapter and its frame/prompt helpers.
experiments/      Dataset-specific experiment utilities.
scripts/          Runnable PolypGen, frame, and multi-object adenoid workflows.
external/MedSAM2/ Third-party MedSAM2 source.
```

PolypGen validates prompting and temporal propagation. REFUGE2 will validate
two-region segmentation and structural-ratio accuracy once its dataset adapter
is implemented. The adenoid pipeline combines both for the target task.

## Current components

`adenoid.measurement.measure_frame` creates one common two-region record:

```python
{
    "frame_idx": 42,
    "adenoid_mask": ...,
    "airway_mask": ...,
    "adenoid_area": ...,
    "airway_area": ...,
    "ratio": ...,
    "valid_frame": True,
}
```

`compute_ratio(..., mode="fraction_of_total")` uses
`area_a / (area_a + area_b)`. `region_a_over_region_b` is also available.
Clinical ratio definitions and grading thresholds are supplied by callers; the
repository does not infer them.

The compact state supports `native`, `current`, and `fixed_ema` memory modes:

```text
state_t = (1 - alpha) * state_(t-1) + alpha * candidate_t
```

## PolypGen temporal experiment

```bash
DATA=data/PolypGen2021_MultiCenterData_v3/sequenceData/positive

python scripts/run_polypgen.py -i "$DATA" -o outputs/fixed_ema_01 \
  --memory_backend fixed_ema --fixed_ema_alpha 0.1 --device auto

python scripts/run_polypgen.py -i "$DATA" -o outputs/native \
  --memory_backend native \
  --prompt_records outputs/fixed_ema_01/prompt_records.json --device auto

python -m evaluation.temporal \
  --output_mask_dir outputs/fixed_ema_01 \
  --data_root "$DATA" \
  --output_eval_dir outputs/fixed_ema_01/evaluation \
  --sequences seq16 seq17 seq18 seq19
```

`--device auto` selects CUDA, then MPS, then CPU. On the development Mac, MPS
produced non-finite SAM2 memory features in this path, so use `--device cpu` if
that recurs.

## Validation splits

`external/MedSAM2/training/assets/polypgen/` stores whole-video split lists:

- train: `seq2–15`, excluding empty `seq1` and `seq7`
- validation: `seq16–19`
- held-out test: `seq20–23`

## Tests

```bash
PYTHONPATH="$PWD:$PWD/external:$PWD/external/MedSAM2" \
  python -m unittest discover -s tests -v
```
