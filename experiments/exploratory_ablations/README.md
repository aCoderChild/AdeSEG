# Lean-SAM2 memory and positional ablations

- `adeseg.memory.temporal_anchors`: multi-scale anchor scoring; fixed-prompt 8-video audit found a negligible (~0.000006) Dice difference.
- `adeseg.memory.spatial_position_encoding`: original retained-token RoPE positions for pruned Lean memory; 21-video automatic results **0.7318** all-present-frame mean Dice and **0.7353** matched-target mean Dice.
- `adeseg.memory.spatial_position_encoding`: CPU key-frequency repetition correction, distinct from original spatial token indexing.

Command-line entry points: `scripts/evaluate_multiscale_lean_sam2.py`, `scripts/evaluate_spatial_position_lean_sam2.py`, `scripts/evaluate_cpu_rotary_lean_sam2.py`.

The verified **0.742071** result is a separate corrected-Lean automatic-prompt baseline, not this learned aggregation method; see [the original result](../reference_baselines/modified_lean_742.md).

[Indexed spatial RoPE audit](https://drive.google.com/file/d/13gm6M61y2zKyJSBS5Vxxok5AY1lePhpt/view).

## Experiment-specific controls

The old `modeling/seq23_viewport.py` uses a retrospectively inspected crop of seq23. This is not a generally learned feature and must be disclosed for exploratory results. Historical region-bias and CBAM/TCMN controls are not active model blocks; some require vendors missing from the cleaned source tree.
