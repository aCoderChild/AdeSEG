# Scripts

Run from the repository root with `python scripts/<name>.py --help`.
Public CLI filenames are retained for pinned Kaggle notebooks and saved receipts.
Shared frame staging lives in `datasets.video`, not the inference CLI.

| Workflow | Entry points |
|---|---|
| Local region diagnostics | `run_region_v2_local.py`, `summarize_region_v2_local.py` |
| E0/E1 validation and pilot | `run_region_token_pilot.py`, `analyze_region_token_pilot.py` |
| Registered E2–E6 execution and analysis | `run_region_experiments.py`, `analyze_region_experiments.py` |
| Upstream baselines | `run_region_baseline.py`; `efficient_sam2_adapter.py` is its Efficient-SAM2 helper |
| Imaging conditions | `measure_region_conditions.py`, `analyze_illumination.py` |
| Chronology and initialization audits | `audit_polypgen_alignment.py`, `audit_region_initialization.py` |
| Manifests | `build_polypgen_manifest.py`, `build_polypgen_kaggle_manifest.py`, `build_cholecseg8k_manifest.py`, `build_davis_manifest.py` |
| Tracking and delivery | `verify_wandb_connection.py`, `collect_region_live.py`, `collect_region_results.py` |
| Historical detector-assisted comparators | `run_local_region_boxes.py`, `run_medsam2_region_boxes.py`, `run_cutie_region_baseline.py` |
| General inference/evaluation utilities | `infer.py`, `evaluate_binary_manifest.py`, `evaluate_davis.py`, `build_mask_manifest.py` |

The registered runner's available stages are not evidence that every stage has
passed runtime validation or completed. Check [results](../docs/REGION_TOKENS_RESULTS.md)
before launching work. Efficient-SAM2 full runs remain blocked by the MPS repeat gate.
The superseded latent-utility label generator and learned-head trainer are removed.
No outputs, weights, credentials or runtime caches belong in this folder.
