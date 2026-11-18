# AdeSEG — Multi-stage Feature Aggregation for Lean-SAM2

**Research focus:** learn to aggregate early image features, deep memory-conditioned features, and foreground/background information to refine video-segmentation masks. A teacher-preservation objective is available during **training only**. PolypGen is an engineering proxy for nasopharyngoscopy, not clinical adenoid evaluation.

## Directory layout

| Directory | Responsibility |
|---|---|
| `modeling/` | Feature aggregation, teacher preservation, memory routing and positional corrections |
| `datasets/` | Video/frame manifests, mask I/O and patient/video split records |
| `training/` | Offline training of the learned aggregation adapter |
| `scripts/` | Evaluation, benchmark, ablation and auditing entry points |
| `experiments/` | Experimental questions, protocols and interpreted evidence |
| `evaluation/` | Segmentation and sequence-level metrics |
| `protocols/` | Original source-bound experiment protocols (filenames intentionally unchanged) |
| `tests/` | Unit tests and public-interface checks |
| `external/` | Pinned SAM2, Lean-SAM2, CMSA-Net, SAM2Long and SAMURAI implementations |
| `archive/` | Link to the complete historical experiment tree |

## Primary scientific interfaces

| Current file | Legacy implementation | Purpose |
|---|---|---|
| `modeling/feature_aggregation.py` | `multiscale_channel_value_refiner_v1.py` | Early/deep projections, channel Q/K/V value mixing, foreground/background contrast, residual head |
| `modeling/teacher_preservation.py` | `positive_teacher_preservation_loss.py` and `training_foreground_preservation_v1.py` | Training-only teacher confidence and positive-region preservation |
| `modeling/refinement_routing.py` | `lean_channel_value_routing_v1.py` | Output-only, memory-only or combined mask refinement |
| `modeling/memory_aggregation.py` | `lean_multiscale_anchor.py` | Multi-scale memory-anchor scoring (separate, untrained control) |
| `modeling/spatial_position_correction.py` | `lean_indexed_spatial_rope.py` and `lean_indexed_spatial_rope_mps.py` | Retained-token spatial RoPE correction |
| `modeling/rotary_position_encoding.py` | `lean_rope_cpu_repeat_fix.py` | CPU rotary-frequency repeat-order correction |

Legacy files remain **unchanged** when archival scripts, checkpoint digests and source-hash receipts require their original paths. Descriptive entry points in `scripts/` call their original, pinned implementation. The descriptive `feature_aggregation.py` re-exports the **exact original model class**, so existing runtime type checks and trained-checkpoint source hashes remain compatible. The original implementation's bytes are not renamed or altered.

## Get started

```bash
git clone --recurse-submodules https://github.com/aCoderChild/AdeSEG.git
cd AdeSEG
python -m pip install -r requirements.txt
python -m pytest -q tests/

python -m training.train_feature_aggregator --help
python -m scripts.evaluate_reference_backbones --help
python -m scripts.evaluate_lean_sam2 --help
python -m scripts.evaluate_cmsa_net --help
python -m scripts.evaluate_multiscale_lean_sam2 --help
python -m scripts.evaluate_spatial_position_lean_sam2 --help
```

Offline aggregation training requires **captured TRAIN tensors and training masks**, with checkpoint/source hashes. Video evaluation requires dataset manifests, original checkpoints and experiment-specific protocols. A successful import/test does not reproduce published Dice.

## Historical 21-sequence evidence

- **0.73446** mean present-frame Dice: author Lean-SAM2, GT-mask initial prompt.
- **0.7172** all-present-frame mean Dice: automatic early-feature-prompt + multi-scale-anchor Lean.
- **0.7318** all-present-frame mean Dice (**0.7353** historical matched): indexed-spatial-RoPE Lean, automatic prompt.

The requested **0.742** modified-Lean mean Dice still requires identification of its exact 21-video result artifact. Do not attribute 0.742 to learned aggregation or teacher preservation without matching frames, prompts and score receipts. All historical scripts and research records remain in the [full experiment backup](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118).

## Reproducibility

Original source hashes/checkpoint provenance may reject renamed or refactored modules. **Use the exact archived Git commit and Drive artifacts for historical reproduction**; use the scientifically named modules for new experiments with freshly recorded hashes. PolypGen has already been repeatedly inspected, so these numbers are exploratory rather than independent external validation.
