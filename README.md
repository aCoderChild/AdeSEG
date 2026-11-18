# AdeSEG — Lean-SAM2 feature aggregation

A focused video-segmentation research repository. The active contribution is a **multiscale, early/deep feature channel-value aggregation adapter** for Lean-SAM2, with **training-only teacher-preservation**. SAM2, Lean-SAM2, CMSA-Net, SAM2Long and SAMURAI are retained as separately identified comparison methods.

**Experimental status:** This is exploratory research. Do not confuse trained refiner results with the independently audited Lean-SAM2 baseline, and do not claim adenoid-specific clinical validation from PolypGen.

## Repository map

- `modeling/`: the feature aggregator (`multiscale_channel_value_refiner_v1.py`), its output/memory routing, teacher-preservation losses, and the recorded Lean positional/anchor modifications.
- `datasets/`: ordered frame loading, image/mask I/O and manifests.
- `training/`: offline frozen-feature training and experiment tracking.
- `scripts/`: original experiment-branch benchmark, inference, audit and cohort runners (their provenance-dependent paths are retained).
- `experiments/`: short, honest method/result cards and links to complete Drive receipts.
- `evaluation/`: segmentation metrics and independent per-sequence statistics.
- `protocols/`: frozen original protocol files; **do not silently edit a protocol used for a saved result**.
- `external/`: five pinned submodules, initialized using `git submodule update --init --recursive`.
- `tests/`: targeted CPU tests for aggregation, teacher preservation and Lean memory controls.

## Core research components

1. **Multiscale channel-value aggregator**: projects early FPN and deep memory-conditioned features, computes foreground/background-conditioned channel attention, aggregates channel values and predicts a residual logit correction. See `modeling/multiscale_channel_value_refiner_v1.py` and `protocols/MULTISCALE_CHANNEL_VALUE_REFINER_V1.md`.
2. **Teacher preservation**: `modeling/positive_teacher_preservation_loss.py` and `modeling/training_foreground_preservation_v1.py` provide training-only penalties. No evaluation label may enter an inference prompt or memory update.
3. **Lean positional / multi-scale controls**: `modeling/lean_multiscale_anchor.py`, `lean_rope_cpu_repeat_fix.py` and `lean_indexed_spatial_rope.py`. These are separate ablations, not automatically part of the learned aggregation model.

## Reproduced historical references (not an apples-to-apples leaderboard)

| Run | 21-sequence mean Dice | Initialization | Status |
|---|---:|---|---|
| Lean-SAM2 author baseline | 0.73446 | First GT mask | Audited, exploratory |
| Native SAM2.1 Base+ | 0.71904 | First GT mask | Audited, exploratory |
| Automatic Lean + early-feature points + pooled anchor | 0.71721 all present frames / 0.72040 matched targets | YOLO + automatic mask | Audited, exploratory |
| Lean indexed-spatial RoPE | 0.7318 all present frames / 0.7353 matched targets | Automatic; matched to CPU-corrected automatic | Audited, exploratory |

The **0.742** modified-Lean 21-sequence aggregate requested for preservation has **not been independently matched to a specific score artifact**. The entire original experiment history remains in `backup/experiment-full-20261118`; the indexed-RoPE and CPU-RoPE modification code and protocols are also included here. No 0.742 efficacy claim is made for the channel-value aggregator or teacher adapter.

Full experiment tables, artifacts and limitations: see `experiments/README.md`.

## Getting started

```bash
git clone --recurse-submodules https://github.com/aCoderChild/AdeSEG.git
cd AdeSEG
python -m pip install -r requirements.txt
python -m pytest tests/test_positive_teacher_preservation_loss.py tests/test_training_foreground_preservation_v1.py tests/test_lean_multiscale_anchor.py
python scripts/train_aggregation.py --help
```

Training requires **frozen, source-identified feature captures** and training-only labels, not PolypGen test labels. See `training/README.md`. To reproduce a historical run exactly, use its pinned source commit, submodules, checkpoint hashes and original Drive artifacts: cleaning/reorganizing source changes hash-based provenance receipts.

## Provenance and backups

The cleanup resolves branch differences in favor of the `experiment` implementation for retained files, while merging the histories of `main` and `experiment`. Original starting points are preserved in Git branches:

- `backup/main-before-cleanup-20261118`
- `backup/experiment-full-20261118`

Data, checkpoints and result pixels are **not** checked into this repository; no dataset/test-set independence is implied by any split filename. The previously inspected PolypGen cohort is exploratory.
