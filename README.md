# AdeSEG

**Adenoid hypertrophy grading from nasopharyngoscopy video.**

```text
video -> prompts -> adenoid + airway masks -> temporal propagation
      -> frame validity -> obstruction measurement -> video aggregation -> grade
```

## Repository layout

```text
adenoid/          Target dataset record, pipeline, mask I/O, measurement, and grading.
models/           MedSAM2 wrapper and compact recurrent EMA state.
validation/       Proxy validation code, currently PolypGen only.
evaluation/       Dataset-independent segmentation and ratio metrics.
scripts/          Small command-line entry points.
configs/          One JSON configuration per supported workflow.
external/MedSAM2/ Third-party MedSAM2 source.
```

PolypGen validates prompting and temporal propagation. REFUGE2 will validate
two-region segmentation and structural-ratio accuracy after its label mapping
and evaluation protocol are defined. The adenoid pipeline is the target task.

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

## Commands

```bash
DATA=data/PolypGen2021_MultiCenterData_v3/sequenceData/positive

python scripts/run_polypgen.py --config configs/polypgen.json -i "$DATA" -o outputs/fixed_ema_01 \
  --memory_backend fixed_ema --fixed_ema_alpha 0.1 --device auto

python scripts/run_polypgen.py --config configs/polypgen.json -i "$DATA" -o outputs/native \
  --memory_backend native \
  --prompt_records outputs/fixed_ema_01/prompt_records.json --device auto

python -m validation.polypgen.evaluate \
  --output_mask_dir outputs/fixed_ema_01 \
  --data_root "$DATA" \
  --output_eval_dir outputs/fixed_ema_01/evaluation \
  --sequences seq16 seq17 seq18 seq19

python scripts/run_adenoid.py --config configs/adenoid.json \
  -i path/to/adenoid_video_or_sequences -o outputs/adenoid
```

The adenoid runner performs prompt-driven, multi-object MedSAM2 propagation.
`adenoid.measurement` and `adenoid.grading` provide the frame and video-level
steps once the clinical region definitions and grading thresholds are supplied.

`--device auto` selects CUDA, then MPS, then CPU. Use `--device cpu` if MPS
produces non-finite MedSAM2 memory features.

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
