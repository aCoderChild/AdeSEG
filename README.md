# AdeSEG: native memory regions

Current goal: attribute SAM2's stored spatial memory vectors to foreground
interior, boundary-related and background regions before analysing their
attention and segmentation contribution.

- [Methods](docs/REGION_TOKENS.md)
- [Retained results](docs/REGION_TOKENS_RESULTS.md)
- [Scripts](scripts/README.md) · [Protocol](protocols/memory_token_regions.json)

## Workflows

| Entry point | Purpose |
|---|---|
| `scripts/audit_memory_token_regions.py` | Reference assignment, vector grouping, coverage and overlays; no training |
| `scripts/measure_region_attention.py` | Verify overlays and observe native regional attention mass/enrichment; no intervention |
| `scripts/run_region_exclusions.py` | Counterfactual regional key exclusions and matched random controls using native inference; no intervention writeback |
| `scripts/run_author_baseline.py` | Official author-method adapters; repeatability checks before signature-gated full inference |
| `scripts/memory_token_io.py` | Shared native propagation, copying, output validation and visualization |

`modeling/` contains the predictor loader, spatial label definitions,
read-only vector grouping and attention observation/exclusion adapters. `datasets/`, `evaluation/` and
`training/tracking.py` provide shared I/O, metrics and W&B tracking.

## Setup

```sh
git submodule update --init --recursive
git -C external/Efficient-SAM2 apply --check ../../patches/efficient-sam2-device.patch
git -C external/Efficient-SAM2 apply ../../patches/efficient-sam2-device.patch
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/audit_memory_token_regions.py --help
```

Apply the device patch once on a fresh checkout; skip it if already applied.
See [patch details](patches/efficient-sam2-device.md). This enables device
allocation only; it does not establish MPS repeatability or SMR accuracy.

Native predictor code is bundled under `external/MedSAM2`; `external/sam2` is retained as the
official reference. Boundary IoU is a pinned author repository; no DINOv2 dependency remains.
`external/Efficient-SAM2` supplies its unchanged `Attention.forward` for
read-only weight reconstruction. Its pruning, encoder routing and predictor
are not enabled. Project code only wires the observer and aggregates regions.
`external/SAM2Long` is pinned for the planned author-default benchmark;
its model execution has not yet been validated. The
[comparison design](docs/REGION_TOKENS.md#next-study-frame-selection-versus-regional-reading)
separates full author methods, native stride controls and region-aware reading.

Outputs belong in the configured Google Drive folder or Kaggle runtime, not
Git. Online tracking requires `WANDB_API_KEY` and uses
`adenoid-hypertrophy/ade-seg` on Forge.

Spatial-assignment outputs:

| Prefix | Output folder | W&B purpose |
|---|---|---|
| 01 | `01_memory_token_region_audit_seq2_v1` | Verified spatial assignment audit |
| 03 | `03_memory_token_region_validation_polypgen_v1` | Full all-frame PolypGen assignment validation |

Historical classifier outputs remain untouched but are not part of this pipeline.
Recorded source snapshots retain their original names for provenance.
Spatial assignment and vector grouping now live together in
`modeling/region_tokens.py`; there is no separate feature-classifier module.

The retired experiments, old notebooks and historical outputs were removed at
the user's request. Current grouping does not prove semantic purity, causal
attention contribution or segmentation improvement.
