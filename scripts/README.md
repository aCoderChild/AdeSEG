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
