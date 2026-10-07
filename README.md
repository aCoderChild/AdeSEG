# AdeSEG

Video-level adenoid hypertrophy assessment from nasopharyngoscopy: two-region
(adenoid / nasopharyngeal airway) segmentation with MedSAM2, an experimental
1-slot recurrent memory replacing MedSAM2's 7-frame bank, and a two-region
ratio evaluator. Adenoid data are not yet available; PolypGen, CholecSeg8k and
REFUGE are used as stand-ins.

- [docs/README.md](docs/README.md): scope, what is implemented, how to run it.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): every result, marked as
  regenerated from saved outputs or archived.
- [docs/report.tex](docs/report.tex): internship report (Vietnamese).

```bash
pip install -r requirements.txt     # plus the MedSAM2 checkpoint in checkpoints/
python3 -m pytest tests
```
