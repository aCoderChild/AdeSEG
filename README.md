# adenoid_segmentation

This repo develops a proxy benchmark for prompt-guided medical video
segmentation. The target application is adenoid and nasopharyngeal airway
segmentation in nasopharyngoscopy/endoscopy videos for quantitative adenoid
hypertrophy assessment.

PolypGen is currently used only as a public endoscopy proxy. It is useful for
debugging temporal propagation, prompt strategies, and endoscopic artifacts, but
it does not establish clinical validity for adenoid hypertrophy.

The current public-data comparison includes frame-by-frame SAM2 baselines,
MedSAM2 video-propagation variants, oracle ground-truth-box prompt variants, and
an optional disabled MedSAM3 text-prompt baseline.

The experiment entrypoint is:

```bash
bash scripts/run_polypgen_experiment.sh
```

Useful variants:

```bash
bash scripts/run_polypgen_experiment.sh --dry_run
bash scripts/run_polypgen_experiment.sh --methods YOLO_SAM2_YOLO_BOX_FRAME --seqs 1 2 3
bash scripts/run_polypgen_experiment.sh --methods MedSAM2_YOLO_BOX_MASK_FIRST
bash scripts/run_polypgen_experiment.sh --methods MedSAM2_YOLO_BOX_MASK_STRIDE5
bash scripts/run_polypgen_experiment.sh --stage eval
```

Edit experiment settings in
`experiments/polypgen_medsam2_yolo_sam2.json`. See `docs/EXPERIMENTS.md` for
the folder layout, method definitions, and stage-by-stage commands.

For the intended adenoid study, see `docs/ADENOID_PROTOCOL.md`. The planned
clinical-data target is multi-object video segmentation:

- adenoid tissue
- nasopharynx / nasopharyngeal airway / choana region

Core evaluation should include Dice/IoU, temporal stability, correction burden,
and downstream adenoid-to-nasopharynx or obstruction-ratio error.
