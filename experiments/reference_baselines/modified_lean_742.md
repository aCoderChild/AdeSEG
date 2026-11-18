# Verified 0.742071 modified Lean-SAM2 baseline

This is **not** the learned multistage feature aggregator and **not**
the MedSAM2-latest checkpoint.

| Item | Verified saved value |
|---|---|
| Source method | Compatibility-corrected author Lean-SAM2 with indexed spatial RoPE on MPS |
| Pretrained backbone | SAM2.1 Hiera-B+ |
| Backbone SHA-256 | `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5` |
| Prompts | One archived automatic mask per video, no later redetection |
| Eligible sequences | 21; seq1/seq7 not initialized |
| Post-prompt positive frames | 1,683 |
| Patient-macro Dice | **0.7420711278676757** |
| Frame-weighted Dice | 0.77249528554744 |
| Empty-frame false alarms | 188/336 |
| Interpretation | Exploratory, not independent clinical validation |

**Exact evidence:** [Original scored 21-sequence report](https://drive.google.com/file/d/1jjqesyv8RXcUI9JN5gcoVVnvSFn9u7O3/view).

**Full source and dependencies:** [Original experiment backup](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118). The historical runner is
`scripts/run_author_lean_automatic_v1.py`; supporting files include
`modeling/lean_indexed_spatial_rope_mps.py`,
`modeling/lean_mps_blocking_pointer_position.py`, the pinned author
submodule and bypass weights. Do not treat the contemporary
`scripts/evaluate_lean_sam2.py` (GT-initialized benchmark) as this run.

The purpose of preserving this result is reproducibility, not to claim
that an untested six-block feature refiner already achieves Dice 0.742.
