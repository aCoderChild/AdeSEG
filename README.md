# AdeSEG: region-level SAM2 memory experiment

The `experiment` branch tests how boundary, foreground and background memory
tokens affect frozen SAM2 segmentation of endoscopic video. E0 sanity checks
and the E1 pilot gate the later analysis and method experiments.

## Start here

- [`docs/REGION_TOKENS.md`](docs/REGION_TOKENS.md): implementation map,
  execution commands, recording guarantees, and remaining work.
- [`protocols/region_tokens_v1_1.json`](protocols/region_tokens_v1_1.json):
  preregistration, pilot sampling, controls and decision rules.
- [`kaggle/region-tokens/`](kaggle/region-tokens/): the active notebook and runner.
- [`scripts/`](scripts/): active region pilot, delivery, manifest preparation,
  reused memory/corruption utilities, inference, and evaluation.
- [`datasets/io.py`](datasets/io.py): shared mask I/O, formerly `adenoid/io.py`.
- [`modeling/utility_memory.py`](modeling/utility_memory.py): frozen SAM2
  memory-attention intervention.
- [`evaluation/`](evaluation/): segmentation, temporal, and official DAVIS
  metrics. The DAVIS metric source and license are retained from the upstream
  DAVIS evaluation package previously bundled with RDE-VOS.
- [`external/`](external/): upstream models needed for this study: SAM2,
  SAM2Long, Lean-SAM2, MedSAM2, MA-SAM2, and DINOv2.

Only the region-token E0–E6 study is active. Shared intervention, corruption,
metrics and tracking helpers retain some historical utility names; these are
dependencies of the region study, not a separate latent-capability experiment.
Frozen protocol ancestors remain solely for split and provenance references.
Historical studies remain recoverable from Git history. The old standalone
latent-study analysis and optional training adapter have been removed.

Clinical grading/ratio code, old Phase A launchers and legacy one-off analysis,
backfill and training entry points have been removed from the active tree.
The three `modeling` modules are all active dependencies. Frozen protocol
ancestors remain because the approved region protocol references them.

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
