# AdeSEG: region-level SAM2 memory experiment

The `experiment` branch tests how boundary, foreground and background memory
tokens affect frozen SAM2 segmentation of endoscopic video. E0 sanity checks
and the E1 pilot gate the later analysis and method experiments.

## Start here

- [`docs/REGION_TOKENS.md`](docs/REGION_TOKENS.md): implementation map,
  execution commands, recording guarantees, and remaining work.
- [`protocols/region_tokens_v1.json`](protocols/region_tokens_v1.json):
  preregistration, pilot sampling, controls and decision rules.
- [`kaggle/region-tokens/`](kaggle/region-tokens/): the active notebook and runner.
- [`scripts/`](scripts/): manifest preparation, utility labels, probes,
  attention baseline, inference, and evaluation.
- [`modeling/utility_memory.py`](modeling/utility_memory.py): frozen SAM2
  memory-attention intervention.
- [`training/utility_trainer.py`](training/utility_trainer.py): matched-budget
  attention baseline training.
- [`evaluation/`](evaluation/): segmentation, temporal, and official DAVIS
  metrics. The DAVIS metric source and license are retained from the upstream
  DAVIS evaluation package previously bundled with RDE-VOS.
- [`external/`](external/): upstream models needed for this study: SAM2,
  SAM2Long, Lean-SAM2, MedSAM2, MA-SAM2, and DINOv2.

The earlier frame-level utility study is documented by the frozen
`protocols/latent_utility_v2.json` and `docs/UTILITY_RETRIEVAL.md`.
Its components supply the intervention, metrics, data and tracking infrastructure.
Historical studies remain recoverable from Git history.

## Verify

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Experiment runs target Kaggle; W&B tracking uses
`adenoid-hypertrophy/ade-seg`. Download completed Kaggle outputs directly to
the dated Google Drive destination with `scripts/collect_region_results.py`.
Do not interpret a missing or gated stage as a positive result.
