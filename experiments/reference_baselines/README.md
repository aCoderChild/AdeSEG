# Reference models

| Comparison | Source | Relevant caution |
|---|---|---|
| SAM2, Lean-SAM2, SAM2Long, SAMURAI | Pinned Git submodules | Prompt policies and temporal information access differ |
| CMSA-Net | Pinned author submodule | Active comparison exports image-only predictions; historical protocol used GT for image sizing |
| Corrected Lean 0.742071 | [Saved 21-sequence automatic-prompt run](modified_lean_742.md) | **Not** a learned adapter, not MedSAM2 |

The retained `scripts.compare_models` runner uses
`experiments/reference_baselines/checkpoint.json` as its B+ checkpoint
identity setting. This **replaces** the original umbrella research recipe
and means newly generated source-bound receipts differ from historic runs.

The original pinned CMSA-Net test loader provided both image and GT arrays.
The new `scripts.evaluate_cmsa` runner supplies zero labels, avoids opening
GT mask bytes during inference, and sizes output from its RGB source frames.
Ground truth is accessed only **after predictions are written** for scoring.
This is a new comparison method and old Dice values should not be inherited.

**Results:** a historical GT-initialized author Lean baseline scored
0.73446 mean present Dice over 21 eligible videos. The distinct corrected,
automatically initialized author Lean run scored **0.742071** patient-macro
Dice. They are not directly matched experiments.

[Original GT-initialized comparison](https://drive.google.com/drive/folders/1DmkKRrGmEcJQUp0Pb0fp5u5CS0DSo0B5).
