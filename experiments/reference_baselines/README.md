# Reference models and comparison infrastructure

This directory contains source/checkpoint identification and evaluators for **reference** segmentation methods. It does not define FGPI, RMFA or TTMR.

| Model family | Code provenance |
|---|---|
| SAM2, Lean-SAM2, SAM2Long, SAMURAI | Pinned author submodules under `external/` |
| CMSA-Net | Author implementation under `external/CMSA-Net` |
| SAM2.1 Hiera-B+ checkpoint | `checkpoint.json` identity for retained author-model comparison |

Available research entrypoints include `scripts.compare_models`, `scripts.evaluate_lean`, `scripts.evaluate_cmsa` and `scripts.validate_results`. Prompting and frame-eligibility protocols must be matched before attributing differences to architectural modifications.

The exploratory historical benchmark summary was moved to the [backup branch reports](https://github.com/aCoderChild/AdeSEG/blob/backup/experiment-full-20261118/reports/modified_lean_742.md) rather than presented as a score for the three proposed modules.
