# Experiment Structure

This project is an experiment harness for prompt-guided endoscopy video
segmentation. Vendor/model implementations live in `external/`; project-owned
orchestration and evaluation live in `scripts/`; experiment settings live in
`experiments/polypgen_medsam2_yolo_sam2.json`.

PolypGen is the current public proxy benchmark. It is used to test prompt
strategies, video propagation, output bookkeeping, and metrics before the
intended adenoid/nasopharynx dataset is available.

## Folder Contract

- `data/polypgen/seq*/images`: input video frames.
- `data/polypgen/seq*/masks`: ground-truth binary masks.
- `data/bbox/seq*/bboxes.csv`: ground-truth boxes generated from masks during
  the `prepare` stage.
- `outputs/<method>/masks/seq*/predicted`: predicted binary masks for SAM2 and
  MedSAM2 methods.
- `outputs/<method>/bbox/seq*.csv`: YOLO-predicted prompt boxes for
  YOLO-prompted methods only.
- `outputs/<method>/logs/seq*.json`: MedSAM2 per-sequence inference logs.
- `outputs/<method>/eval/mask`: mask metrics and mask overlays.
- `outputs/<method>/eval/bbox`: detector-box IoU metrics when bbox evaluation is
  enabled.
- `outputs_MedSAM3/<method>`: MedSAM3 output root from the default config.

Ground-truth-box methods use boxes from `data/bbox` as prompts. Those boxes are
oracle prompts, not detector predictions, so their default evaluation is mask
only.

## Main Commands

Run the enabled methods on the configured sequence range:

```bash
bash scripts/run_polypgen_experiment.sh
```

Run selected methods:

```bash
bash scripts/run_polypgen_experiment.sh --methods YOLO_SAM2_YOLO_BOX_FRAME
bash scripts/run_polypgen_experiment.sh --methods MedSAM2_YOLO_BOX_MASK
bash scripts/run_polypgen_experiment.sh --methods MedSAM2_GT_BOX_MASK_STRIDE5
```

Run selected sequences:

```bash
bash scripts/run_polypgen_experiment.sh --seqs 1 2 3
```

Run stages separately:

```bash
bash scripts/run_polypgen_experiment.sh --stage prepare
bash scripts/run_polypgen_experiment.sh --stage infer
bash scripts/run_polypgen_experiment.sh --stage eval
```

Run one method on one sequence:

```bash
python3 scripts/experiments/run_experiment.py \
  --stage all \
  --methods MedSAM2_YOLO_BOX_MASK_FIRST \
  --seqs 3
```

Run mask-only evaluation:

```bash
python3 scripts/experiments/run_experiment.py \
  --stage eval \
  --methods YOLO_SAM2_GT_BOX_FRAME MedSAM2_GT_BOX_MASK \
  --eval_type mask
```

Preview commands without running models:

```bash
bash scripts/run_polypgen_experiment.sh --dry_run
```

Use another Python interpreter or output root:

```bash
PYTHON_BIN=/path/to/python bash scripts/run_polypgen_experiment.sh --dry_run

python3 scripts/experiments/run_experiment.py \
  --output_root outputs_trial \
  --methods MedSAM2_YOLO_BOX_BOX_STRIDE10
```

## Current Methods

Enabled methods in the default config:

- `YOLO_SAM2_YOLO_BOX_FRAME`: YOLO boxes prompt SAM2 image inference on every
  frame.
- `YOLO_SAM2_GT_BOX_FRAME`: ground-truth boxes prompt SAM2 image inference on
  every frame.
- `SAM2_LARGE_GT_BOX_FRAME`: same oracle-box frame baseline with the large SAM2
  checkpoint.
- `MedSAM2_YOLO_BOX_BOX`: YOLO boxes are passed directly as MedSAM2 video
  prompts every frame.
- `MedSAM2_YOLO_BOX_MASK`: YOLO boxes first produce SAM2/MedSAM2 image masks;
  those masks seed MedSAM2 video propagation every frame.
- `MedSAM2_GT_BOX_BOX`: ground-truth boxes are passed directly as MedSAM2 video
  prompts every frame.
- `MedSAM2_GT_BOX_MASK`: ground-truth boxes are converted to mask prompts, then
  used for MedSAM2 video propagation every frame.
- `*_STRIDE5` and `*_STRIDE10`: same MedSAM2 prompt source and prompt type, but
  re-prompt only every 5 or 10 frames.
- `*_FIRST`: seed MedSAM2 only from the first successful prompt frame
  (`video_prompt_limit=1`).

Disabled method in the default config:

- `MedSAM3_TEXT_POLYP`: text-prompted MedSAM3 inference with prompt `polyp`,
  written under `outputs_MedSAM3`.

Use the `GT_BOX` methods to isolate segmentation and propagation quality from
YOLO detector quality. Use the YOLO methods to measure the full automatic
pipeline. Use stride and first-prompt variants to test how much MedSAM2 temporal
memory helps when prompt frequency is reduced.

## Configuration

The default config is:

```text
experiments/polypgen_medsam2_yolo_sam2.json
```

Change checkpoint paths, confidence thresholds, sequence ranges, prompt
frequency, enabled methods, and output roots there instead of editing model
scripts.

Important config fields:

- `runner`: implementation family. Current runners are `YOLO_SAM2`, `MedSAM2`,
  and `MedSAM3`.
- `output_name`: output folder name under the selected output root.
- `prompt_source`: `yolo` for detector prompts or `gt_bbox` for boxes from
  `data/bbox`.
- `eval_type`: `both`, `mask`, or `bbox`. Ground-truth-box methods should stay
  `mask` unless a specific detector-style check is intended.
- `yolo_checkpoint`, `yolo_conf`, `yolo_imgsz`, `max_yolo_boxes_per_frame`:
  detector settings.
- `sam2_cfg` and `sam2_checkpoint`: SAM2/MedSAM2 model settings.
- `video_prompt_source`: `box` or `mask` for MedSAM2.
- `video_prompt_stride`: prompt every N frames. Larger values rely more on
  temporal propagation and less on repeated prompting.
- `video_prompt_limit`: maximum number of prompt frames. `1` means first
  successful prompt only; `0` means no limit.

## Evaluation

`scripts/experiments/run_experiment.py` groups methods by output root and calls
`scripts/utils/eval_metrics.py`.

Mask evaluation writes per-frame, per-sequence, and aggregate outputs under
`outputs/<method>/eval/mask`, including:

- `seq*/metrics.csv`
- `seq*/metrics_avg.csv`
- `seq*/overlays/*_mask_overlay.png`
- `metrics_per_sequence.csv`
- `metrics_avg.csv`
- `metrics_stats.csv`

Mask metrics:

- Dice
- IoU
- F-measure
- sensitivity
- specificity
- S-measure
- E-measure
- `pred_area_frac`
- `temporal_iou_prev`
- `area_change_abs_prev`
- `centroid_shift_norm_prev`

BBox evaluation scores every ground-truth frame for YOLO-prompted methods.
Frames with no predicted YOLO box receive IoU `0.0`, so missed detections are
included in detector quality.

Do not interpret bbox IoU for `GT_BOX` methods as detector quality. Their prompt
boxes are copied from ground truth.

## Failure Accounting

Methods should write one predicted mask per input frame. If MedSAM2 cannot
generate any prompt for a sequence, the runner writes blank masks so that the
failure appears as low Dice/IoU instead of disappearing from the mean.

MedSAM2 writes JSON logs under `outputs/<method>/logs`. Check these first when a
sequence has missing or blank masks; they record prompt attempts, YOLO detection
counts, seeded frames, propagated frames, saved-mask count, status, and
tracebacks.
