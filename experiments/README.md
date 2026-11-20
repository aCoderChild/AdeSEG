# Experiments by research contribution

| Research line | Controlled intervention | Documentation |
|---|---|---|
| **FGPI** | Same YOLO detector box, with versus without early-feature selected positive/negative points | [Method](../docs/methodology/FGPI.md) · [Protocol](../docs/experiments/PLAN.md) |
| **RMFA** | Same pretrained Lean logits, with versus without learned multistage region-aware residual | [Method](../docs/methodology/RMFA.md) · [Feature aggregation](lean_feature_aggregation/README.md) |
| **TTMR** | Same prompt, backbone and memory budget; alternate token selection/coverage criterion | [Method](../docs/methodology/TTMR.md) |
| **Reference models** | Pinned author SAM2/Lean-SAM2/SAM2Long/SAMURAI and CMSA-Net | [Reference utilities](reference_baselines/README.md) |
| **Other studies** | MedSAM2 high-resolution skip/teacher research and mechanism controls | [MedSAM2](medsam2_high_resolution/README.md) · [Exploratory ablations](exploratory_ablations/README.md) |

## Experimental strategy

1. Fix first-detection eligibility, source images and checkpoint identity for the initialization study.
2. Evaluate **prompt-frame** segmentation before analyzing propagation effects.
3. Train RMFA on correctly partitioned feature captures; compare *output-only* refinement before routing corrected masks back into memory.
4. Investigate TTMR against native Lean target-anchored pruning under a matched memory budget.
5. Apply the same per-video population, mask policy and statistical unit across each controlled comparison.

Do not combine results from the MedSAM2 Hiera-T branch and the Lean Hiera-B+ branch as though they were one matched model. The full design is in [docs/experiments/PLAN.md](../docs/experiments/PLAN.md).
