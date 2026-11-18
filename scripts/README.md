# Commands

Use these **short commands** for new work. The older `run_...` and
`audit_...` files are retained unchanged for historical hashes and
reproduction; do not import them directly in new research code.

| Command | What it does |
|---|---|
| `python -m scripts.train_med` | Train reconstructed MedSAM2 high-res skip adapter |
| `python -m scripts.train_lean` | Train Lean channel-value aggregation |
| `python -m scripts.compare_models` | SAM2, Lean, SAM2Long, SAMURAI comparisons |
| `python -m scripts.evaluate_cmsa` | CMSA-Net image-only benchmark; no GT mask reads in inference |
| `python -m scripts.evaluate_lean` | GT-first-mask Lean benchmark |
| `python -m scripts.evaluate_memory` | Automatic Lean pooled-memory ablation |
| `python -m scripts.evaluate_rope` | Indexed spatial RoPE ablation |
| `python -m scripts.evaluate_cpu_rope` | CPU RoPE correction control |
| `python -m scripts.validate_results` | Independently rescore Lean/native masks |
| `python -m scripts.validate_cmsa` | Independently rescore CMSA results |
| `python -m scripts.validate_memory` | Verify memory-anchor result records |
| `python -m scripts.validate_rope` | Verify indexed-RoPE result records |

**Important:** `evaluate_lean` is the GT-initialized comparison, **not**
the historical automatic-prompt run scored at 0.742071. That run's complete
original source is preserved only in the full experiment backup.
Validation checks results; it cannot turn development data into a sealed test.

**Public training split requirement:** both `train_med` and `train_lean` now require `--split-file` and capture-level SHA256/source-group declarations (see `training/README.md`). The historical runner `run_cmsa_polypgen.py` is unchanged; the new `run_cmsa_image_only.py` intentionally uses a new output-export protocol.
