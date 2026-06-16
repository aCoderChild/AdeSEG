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
- `outputs/data_quality_report.csv`: sequence-level raw data quality scores.
- `outputs_MedSAM3/<method>`: MedSAM3 output root from the default config.

Ground-truth-box methods use boxes from `data/bbox` as prompts. Those boxes are
oracle prompts, not detector predictions, so their default evaluation is mask
only.

## Project-Owned Scripts

- `scripts/run_polypgen_experiment.sh`: thin shell wrapper around the Python
  orchestrator. It passes the default config and honors `PYTHON_BIN`.
- `scripts/experiments/run_experiment.py`: reads the JSON config, merges method
  defaults, runs prepare/infer/eval stages, and groups evaluation by output
  root.
- `scripts/utils/ground_truth_bbox_gen.py`: creates `data/bbox/seq*/bboxes.csv`
  from binary masks. Empty masks are skipped; all-empty sequences still get a
  header-only CSV.
- `scripts/utils/eval_metrics.py`: evaluates masks, YOLO boxes, temporal
  stability, overlays, and aggregate CSV summaries.
- `scripts/utils/data_quality.py`: scans raw PolypGen frames for sequence-level
  no-reference quality scores.
- `scripts/utils/failure_analysis.py`: finds detection-segmentation disconnects
  and plots per-frame metric timelines.
- `scripts/utils/model_defaults.py` and `scripts/utils/eval.py`: helper modules
  kept for shared defaults and legacy/simple evaluation entrypoints.

## Required Inputs

The default config expects these local assets:

- PolypGen frames and masks under `data/polypgen`.
- YOLO checkpoint at `checkpoints/polypgen_yolov8n.pt`.
- MedSAM2 checkpoint at `checkpoints/MedSAM2_latest.pt`.
- SAM2 large checkpoint at `checkpoints/sam2_hiera_large.pt` for
  `SAM2_LARGE_GT_BOX_FRAME`.
- Optional MedSAM3 LoRA weights at
  `checkpoints/MedSAM3_v1/best_lora_weights.pt`.
- Optional SAM3 base checkpoint at `checkpoints/facebook_sam3/sam3.pt`.

The repo does not include these large data/checkpoint assets. Keep their paths
in the JSON config rather than hardcoding them inside external model scripts.

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

Run the raw data quality audit:

```bash
python3 scripts/utils/data_quality.py
```

The audit writes `outputs/data_quality_report.csv`. The report has one row per
sequence with no-reference capsule/endoscopy IQA proxy scores, including
contrast, entropy, sharpness, blur, blockiness, natural-scene-statistics
distortion, DCT frequency balance, directional gradient entropy, and noise.

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

Enabled method groups in the default config:

- `YOLO_SAM2_YOLO_BOX_FRAME`: YOLO boxes prompt SAM2 image inference every
  frame. This is the automatic frame-by-frame baseline.
- `YOLO_SAM2_GT_BOX_FRAME`: ground-truth boxes prompt SAM2 image inference every
  frame. This is an oracle prompt baseline.
- `SAM2_LARGE_GT_BOX_FRAME`: large-checkpoint oracle SAM2 frame baseline.
- `MedSAM2_YOLO_BOX_BOX`: YOLO boxes are passed directly as MedSAM2 video
  prompts every frame.
- `MedSAM2_YOLO_BOX_MASK`: YOLO boxes first produce image-predictor masks; those
  masks seed MedSAM2 video propagation every frame.
- `MedSAM2_GT_BOX_BOX`: ground-truth boxes are passed directly as MedSAM2 video
  prompts every frame.
- `MedSAM2_GT_BOX_MASK`: ground-truth boxes are converted to mask prompts, then
  used for MedSAM2 video propagation every frame.
- `MedSAM2_*_STRIDE5`: same source/type as the base MedSAM2 method, but prompt
  every 5 frames.
- `MedSAM2_*_STRIDE10`: same source/type as the base MedSAM2 method, but prompt
  every 10 frames.
- `MedSAM2_*_FIRST`: seed MedSAM2 only from the first successful prompt frame
  by setting `video_prompt_limit=1`.

Disabled method in the default config:

- `MedSAM3_TEXT_POLYP`: text-prompted MedSAM3 inference with prompt `polyp`,
  written under `outputs_MedSAM3`.

Use the `GT_BOX` methods to isolate segmentation and propagation quality from
YOLO detector quality. Use the YOLO methods to measure the full automatic
pipeline. Use stride and first-prompt variants to test how much MedSAM2 temporal
memory helps when prompt frequency is reduced.

## Method Name Grammar

Most method names encode their prompt strategy:

```text
<model>_<prompt_source>_<video_prompt_source>_<prompt_schedule>
```

- `YOLO`: boxes come from the YOLO detector.
- `GT_BOX`: boxes come from `data/bbox`.
- `BOX`: pass boxes directly to the video predictor.
- `MASK`: convert boxes to mask prompts first.
- `STRIDE5` or `STRIDE10`: prompt every N frames.
- `FIRST`: use only the first successful prompt frame.
- No suffix: prompt every frame for MedSAM2 video methods.

Frame-by-frame SAM2 methods use shorter names because they do not have a video
prompt schedule.

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
- `output_root`: root output directory. MedSAM3 defaults to `outputs_MedSAM3`.
- `enabled`: include or exclude a method from the default run.
- `prompt`: text prompt list for MedSAM3.
- `threshold`, `nms_iou`, `resolution`, `max_detections_per_frame`: MedSAM3
  inference settings.

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

## Failure Analysis

Find cases where detector boxes look good but final masks are poor:

```bash
python3 scripts/utils/failure_analysis.py disconnect \
  --methods MedSAM2_YOLO_BOX_MASK YOLO_SAM2_YOLO_BOX_FRAME \
  --bbox_thresh 0.7 \
  --mask_thresh 0.4 \
  --output_dir outputs/failure_analysis
```

Plot per-frame metric timelines:

```bash
python3 scripts/utils/failure_analysis.py timeline \
  --seqs 10 20 \
  --methods MedSAM2_YOLO_BOX_MASK YOLO_SAM2_YOLO_BOX_FRAME \
  --metric iou \
  --output_dir outputs/failure_analysis/timelines
```

Useful timeline metrics include `dice`, `iou`, `pred_area_frac`,
`temporal_iou_prev`, `area_change_abs_prev`, and
`centroid_shift_norm_prev`.

## Failure Accounting

Methods should write one predicted mask per input frame. If MedSAM2 cannot
generate any prompt for a sequence, the runner writes blank masks so that the
failure appears as low Dice/IoU instead of disappearing from the mean.

MedSAM2 writes JSON logs under `outputs/<method>/logs`. Check these first when a
sequence has missing or blank masks; they record prompt attempts, YOLO detection
counts, seeded frames, propagated frames, saved-mask count, status, and
tracebacks.

## Troubleshooting

- Unknown method: check the exact method name in
  `experiments/polypgen_medsam2_yolo_sam2.json`.
- Missing `data/bbox`: run `--stage prepare` before oracle-box methods or
  evaluation.
- Missing predicted masks: inspect `outputs/<method>/logs/seq*.json` for
  MedSAM2 or rerun with a single `--seqs` value for a smaller repro.
- Empty or low mask metrics: confirm the method wrote one mask per input frame
  under `outputs/<method>/masks/seq*/predicted`.
- Bbox metrics missing: only YOLO-prompted methods write predicted prompt boxes.
- MedSAM3 outputs missing: `MedSAM3_TEXT_POLYP` is disabled by default and uses
  the separate `outputs_MedSAM3` root.
- Import or checkpoint errors: verify the active Python environment,
  `requirements.txt`, model-specific dependencies in `external/`, and checkpoint
  paths in the JSON config.

## Interpreting Results

Use results in layers:

- YOLO bbox IoU shows detector quality.
- `YOLO_SAM2_YOLO_BOX_FRAME` shows automatic frame-by-frame segmentation.
- `GT_BOX` SAM2 methods show how well SAM2 can segment with perfect boxes.
- MedSAM2 every-frame methods show dense video prompting behavior.
- MedSAM2 stride and first-prompt methods show temporal propagation and prompt
  burden tradeoffs.
- Failure-analysis timelines show drift, recovery, and unstable sequences that
  aggregate metrics can hide.

PolypGen results are engineering evidence for the pipeline only. Clinical claims
must use adenoid/nasopharynx data, ENT-defined annotation rules, and
patient-level splits.
