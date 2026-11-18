# Evidence by research question

| Study | Modified part | Verified status |
|---|---|---|
| [MedSAM2 high resolution](medsam2_high_resolution/README.md) | High-res skip adapter and train-only native KL | Reconstructed method; mixed four-sequence historical experiments |
| [Lean feature aggregation](lean_feature_aggregation/README.md) | Q/K/V channel mixing of early/deep features | Architecture and training capture flow exist; no verified full-cohort gain |
| [Comparisons](reference_baselines/README.md) | No learned adapter; original/corrected vendor models | Includes the 0.742071 corrected Lean-SAM2 exploratory result |
| [Exploratory ablations](exploratory_ablations/README.md) | Memory, position, region and loss controls | Historical source in backup; not independent evaluation |

Do not compare scores across different pretrained backbones, frame subsets,
prompt budgets or retrospective preprocessing as an architectural ablation.
Retain failed gates, empty-mask false alarms, source revisions and checkpoint
SHA256 values. The current train-split validator checks **declared** group
disjointness only, not actual patient identity.

See [architecture and proposed novelty](../docs/ARCHITECTURE.md).
Original source-specific recipes are preserved in the
[full experiment backup](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118).
