# AdeSEG — Multi-stage Feature Aggregation for Lean-SAM2

**Research focus:** a six-block, checkpoint-compatible multi-stage feature aggregation model for Lean-SAM2. Teacher preservation is training-only. **MedSAM2-latest is a separate Hiera-T/512 checkpoint**, not the Hiera-B+ backbone used by the verified 0.7421 Lean comparison. PolypGen is a colonoscopy engineering proxy, not clinical adenoid validation.

## Directory layout

| Directory | Responsibility |
|---|---|
| `modeling/` | Six-block aggregation, teacher preservation, MedSAM2 builder, Lean memory routing and positional corrections |
| `datasets/` | Video/frame manifests, mask I/O and patient/video split records |
| `training/` | Offline training of the learned aggregation adapter |
| `scripts/` | Evaluation, benchmark, ablation and auditing entry points |
| `experiments/` | Experimental questions, protocols and interpreted evidence |
| `evaluation/` | Segmentation and sequence-level metrics |
| `protocols/` | Original source-bound experiment protocols (filenames intentionally unchanged) |
| `tests/` | Unit tests and public-interface checks |
| `external/` | Pinned SAM2/Lean-SAM2/CMSA-Net/SAM2Long/SAMURAI submodules **plus vendored MedSAM2** |
| `archive/` | Link to the complete historical experiment tree |

## Primary scientific interfaces

| Current file | Legacy implementation | Purpose |
|---|---|---|
| `modeling/feature_aggregation.py` | `structured_feature_aggregation.py` and `aggregation_blocks/` | **Six explicit computational blocks**, compatible with the original trainable parameters and checkpoint schema |
| `modeling/teacher_preservation.py` | `positive_teacher_preservation_loss.py` and `training_foreground_preservation_v1.py` | Training-only teacher confidence and positive-region preservation |
| `modeling/refinement_routing.py` | `lean_channel_value_routing_v1.py` | Output-only, memory-only or combined mask refinement |
| `modeling/memory_aggregation.py` | `lean_multiscale_anchor.py` | Multi-scale memory-anchor scoring (separate, untrained control) |
| `modeling/spatial_position_correction.py` | `lean_indexed_spatial_rope.py` and `lean_indexed_spatial_rope_mps.py` | Retained-token spatial RoPE correction |
| `modeling/rotary_position_encoding.py` | `lean_rope_cpu_repeat_fix.py` | CPU rotary-frequency repeat-order correction |

Legacy files remain **unchanged** when archival scripts, checkpoint digests and source-hash receipts require their original paths. Descriptive entry points in `scripts/` call their original, pinned implementation. The structured `MultistageFeatureAggregator` **subclasses the unchanged original model**, inherits all registered weights and the strict trained-checkpoint loader, and is accepted by its Lean runtime `isinstance` guard. The block decomposition is required to pass exact output and gradient parity tests before being regarded as a mathematical equivalent. Original source bytes and source-hash receipts remain intact.

**Architecture diagram:** [Six-block model](docs/MODEL_ARCHITECTURE.md). **Checkpoint audit:** [Lean-SAM2 B+ versus MedSAM2 Hiera-T](docs/CHECKPOINT_PROVENANCE.md).

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

**0.742071** patient-macro mean present-frame Dice is now traced to the [21-sequence automatic-prompt author Lean result](https://drive.google.com/file/d/1jjqesyv8RXcUI9JN5gcoVVnvSFn9u7O3/view): 1,683 present frames and 188/336 labeled-empty false alarms, with indexed MPS spatial RoPE, blocking pointer transfer and **SAM2.1 Hiera-B+**, not MedSAM2-latest. Its [archived runner](https://github.com/aCoderChild/AdeSEG/blob/backup/experiment-full-20261118/scripts/run_author_lean_automatic_v1.py) has a larger dependency tree than this focused `main`. The learned aggregation/teacher adapter does **not** have a verified 21-sequence 0.742 result.

## Reproducibility

Original source hashes/checkpoint provenance may reject renamed or refactored modules. **Use the exact archived Git commit and Drive artifacts for historical reproduction**; use the scientifically named modules for new experiments with freshly recorded hashes. PolypGen has already been repeatedly inspected, so these numbers are exploratory rather than independent external validation.
