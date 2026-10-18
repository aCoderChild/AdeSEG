# AdeSEG: region-token experiments

Study: foreground-interior, boundary, near-background and far-background token
contributions to SAM2 memory attention and endoscopic segmentation.

- [Results and statistics](docs/REGION_TOKENS_RESULTS.md)
- [Study idea, methods and implementation](docs/REGION_TOKENS.md)
- [Frozen protocols and amendments](protocols/README.md)
- [Script entry points](scripts/README.md) · [Test coverage](tests/README.md)

| Folder | Purpose |
|---|---|
| modeling/ | Upstream predictor loader, attention hook, region and mask interventions |
| scripts/ | Data audits, experiment runners, analysis and result collection |
| evaluation/ | Segmentation metrics and sequence-level statistics |
| datasets/ | Manifest and mask I/O |
| training/ | Tracking shared with frozen inference |
| kaggle/ | Pinned notebook bootstraps and runners |
| external/ | Upstream model repositories |
| tests/ | Active project regression tests |

```sh
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pytest -q
```

Models are reused from upstream; project code supplies experimental controls,
not a replacement segmentation architecture. Only the region study is active.
Frozen ancestor protocols remain for referenced splits and provenance.

W&B project: `adenoid-hypertrophy/ade-seg`. Outputs go to the configured Drive
folder, not Git. The Drive desktop folder still uses local disk; filesystem
delivery alone does not prove cloud synchronization. The full study is incomplete.
