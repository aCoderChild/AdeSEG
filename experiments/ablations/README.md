# Minimal retained ablations

- Automatic early-feature prompt with pooled anchor: `scripts/run_lean_auto_multiscale.py` and its cohort/audit scripts; saved in Drive.
- Matched fixed-prompt pooled-anchor control: `scripts/run_lean_auto_memory_ablation.py`; only 8 selected videos, negligible Dice change.
- Lean CPU RoPE frequency-repeat correction: `scripts/run_lean_rope_cpu_fix.py` and `protocols/lean_cpu_rope_repeat_fix_v1.json`.
- Original retained spatial positions in Lean memory RoPE: `scripts/run_lean_indexed_spatial_rope.py` and `protocols/lean_indexed_spatial_rope_v1.json`. The indexed change scored 0.7353 on 21 *historically matched* targets and 0.7318 on all present frames, using automatic prompts.

For exact historical reproduction, use [full experiment backup](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118) because some prior evidence contracts pin absolute source locations and hashes.
