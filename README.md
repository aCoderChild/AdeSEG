# AdeSEG: SAM2 memory-utility experiment

This branch tests whether SAM2's frozen memory features encode which stored
memories help the current segmentation, whether attention loses that alignment
after memory errors accumulate, and whether a small readout can restore it.
The hypothesis is **not** established until the preregistered behavioral,
representational, and causal tests all pass.

## Start here

- [`protocols/latent_utility_v2.json`](protocols/latent_utility_v2.json):
  frozen design, splits, evidence bars, and go/no-go rules.
- [`docs/UTILITY_RETRIEVAL.md`](docs/UTILITY_RETRIEVAL.md): implemented
  attention instrumentation and experiment runner; its v1 protocol documents
  the earlier implementation layer, while v2 is the current claim policy.
- [`kaggle/`](kaggle/): Kaggle notebooks and the current staged runner.
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
  SAM2Long, Lean-SAM2, MedSAM2, and DINOv2.

The older Kalman/RDE recurrent-memory study is not part of this design. Its
write-up and historical results are in [`docs/archive/`](docs/archive/);
its removed source remains recoverable from Git history before this refactor.

## Verify

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Experiment runs target Kaggle; W&B tracking uses
`adenoid-hypertrophy/ade-seg`. Download completed Kaggle outputs directly to
the dated Google Drive destination with `scripts/fetch_kaggle_results.sh`.
Do not interpret a missing or gated stage as a positive result.
