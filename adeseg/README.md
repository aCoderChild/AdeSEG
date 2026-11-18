# AdeSEG public model API

For new code, import `adeseg.*`. Original source-bound implementations
remain in `modeling/` so model checkpoints retain their expected parameter
names and source SHA256 checks.

| Subpackage | Responsibility |
|---|---|
| `backbones/` | MedSAM2 Hiera-T/512 and Lean-SAM2 Hiera-B+ model loaders |
| `adapters/` | **Added trainable components:** medical decoder skip and Lean channel Q/K/V refiner |
| `losses/` | **Training-only:** teacher KL, foreground preservation, segmentation |
| `memory/` | Optional temporal-anchor and position ablations |
| `inference/` | Reversible MedSAM2 decoder hook and source-bound Lean routing |
| `evaluation/` | Frame/sequence segmentation and temporal metrics |

[**Architecture with exact block flow**](../docs/ARCHITECTURE.md).

The two backbones and their checkpoints are **not interchangeable**. Teacher
KL is an objective, not an inference-time attention module. The 0.742071
result is a separate corrected author Lean baseline, not a score belonging
to either learned adapter.
