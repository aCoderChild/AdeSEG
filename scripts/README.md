# Commands

Only **core, supported research entry points** are kept in active `main`.
The original exploratory scripts remain in the full Git backup.

| Run | Meaning |
|---|---|
| `python -m scripts.infer_med_video` | Real MedSAM2 oracle-first-mask video propagation, optional verified adapter |
| `python -m scripts.evaluate_med_video` | Strict post-prompt scorer for the above runner |
| `python -m scripts.research` | Validated experiment lifecycle execution |
| `python -m scripts.ablate` | Matched-protocol paired ablation analysis |
| `python -m scripts.train_med` | Train reconstructed MedSAM2 high-resolution adapter |
| `python -m scripts.train_lean` | Train Lean feature aggregation adapter |
| `python -m scripts.compare_models` | Pinned native SAM2, Lean, SAM2Long and SAMURAI |
| `python -m scripts.evaluate_cmsa` | Image-only CMSA-Net comparison with GT-free prediction export |
| `python -m scripts.evaluate_lean` | GT-first-mask Lean comparison (NOT the automatic 0.742 run) |
| `python -m scripts.validate_results` | Independently rescore saved native/Lean masks |
| `python -m scripts.validate_cmsa` | Independently rescore CMSA predictions |

The `run_...` and `audit_...` modules still present here implement
the commands; they have been retained as compatibility dependencies.
Historical experiments using additional memory/region controls and their
original inputs are available in
[the full experiment backup](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118).

Each run requires the appropriate pretrained checkpoint, manifest, isolated
model package and input/validation receipts. No command should inherit the
0.742071 exploratory baseline score without reproducing its exact 21-video
automatic initialization and corrected MPS state.

| `python -m scripts.compare_med_runs` | Check matched native/zero/trained MedSAM2 full-video run receipts |
| `python -m scripts.check_med_gpu_smoke` | Fail-closed preflight for one curated DEV CUDA smoke video |

The manual [GPU smoke workflow](../.github/workflows/gpu-smoke.yml) runs
native inference, zero-adapter parity, and optional trained-adapter comparison.
It requires a configured self-hosted GPU runner and authorized local assets.
See [docs/VERIFICATION.md](../docs/VERIFICATION.md).

Research-grade MedSAM2 video execution requires a sealed `--protocol` created
with `python -m scripts.prepare_med_video_protocol`. Do not mix old
`--manifest`/`--checkpoint` flags with the controlled inference CLI.

| `python -m scripts.prepare_med_video_protocol` | Generate sealed MedSAM2 oracle prompt-plan and checkpoint protocol |
| `python -m scripts.seal_med_video_study` | Verify full native/zero/trained evidence, hashes and Git revision; write final study receipt |

For numerical parity, supply `--record-logits` on both native and identity
inference runs. The evaluator authenticates saved float32 tensors and checks
their mask threshold agrees with the exported PNG. Keep those source tensors
local; only metrics are uploaded by the GPU workflow.
