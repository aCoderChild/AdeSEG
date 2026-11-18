# Experiment settings

Historical files here keep their exact original names (including `_v1`).
They are part of checkpoint and score provenance, **not** recommended names
for current software interfaces.

| Old setting file | Plain-language meaning |
|---|---|
| `MULTISCALE_CHANNEL_VALUE_REFINER_V1.md` | Lean feature aggregation |
| `SOURCE_GROUP_PRESERVATION_V1.md` | Positive teacher preservation |
| `lean_auto_multiscale_early_prompt_v1.json` | Automatic prompt + pooled memory |
| `lean_indexed_spatial_rope_v1.json` | Retained spatial-token positions |
| `lean_cpu_rope_repeat_fix_v1.json` | CPU rotary-frequency ordering |
| `CMSA_NET_POLYPGEN.md` | CMSA-Net comparison |
| `medical_highres_reconstruction_v1.json` | New reconstructed medical adapter |

For current study navigation use the short names under
`experiments/`, `adeseg/`, and `scripts/`. Never overwrite the
original experimental protocol behind a saved score.
