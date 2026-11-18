# Lean-SAM2 memory and positional ablations

- `modeling/memory_aggregation.py`: multi-scale anchor scoring; fixed-prompt 8-video audit found a negligible (~0.000006) Dice difference.
- `modeling/spatial_position_correction.py`: original retained-token RoPE positions for pruned Lean memory; 21-video automatic results **0.7318** all-present-frame mean Dice and **0.7353** matched-target mean Dice.
- `modeling/rotary_position_encoding.py`: CPU key-frequency repetition correction, distinct from original spatial token indexing.

Command-line entry points: `scripts/evaluate_multiscale_lean_sam2.py`, `scripts/evaluate_spatial_position_lean_sam2.py`, `scripts/evaluate_cpu_rotary_lean_sam2.py`.

The user-reported 0.742 result has **not** yet been verified as a particular 21-sequence Lean experiment; the complete experiment backup protects all candidates.

[Indexed spatial RoPE audit](https://drive.google.com/file/d/13gm6M61y2zKyJSBS5Vxxok5AY1lePhpt/view).
