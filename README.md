# AdeSEG: Adenoid Video Segmentation Experiments

AdeSEG is a research codebase for prompt-guided medical video segmentation. The
target clinical application is segmentation of adenoid tissue and the
nasopharyngeal airway in nasopharyngoscopy/endoscopy videos for quantitative
adenoid hypertrophy assessment.

The current runnable benchmark uses PolypGen as a public endoscopy proxy. That
proxy is useful for testing data loading, prompt generation, SAM2/MedSAM2 video
propagation, temporal metrics, and failure accounting. It should not be used as
clinical evidence for adenoid or nasopharyngeal airway segmentation.

## What This Repo Contains

- `README.md`: project overview and quick start.
- `docs/EXPERIMENTS.md`: full experiment manual for data layout, methods,
  commands, configuration, outputs, and evaluation.
- `docs/ADENOID_PROTOCOL.md`: clinical protocol for the intended adenoid study.
- `experiments/polypgen_medsam2_yolo_sam2.json`: default experiment config.
- `scripts/run_polypgen_experiment.sh`: main shell entrypoint.
- `scripts/experiments/run_experiment.py`: prepare/infer/evaluate orchestrator.
- `scripts/utils/ground_truth_bbox_gen.py`: converts masks to oracle boxes.
- `scripts/utils/eval_metrics.py`: mask, bbox, temporal, aggregate, and overlay
  evaluation.
- `scripts/utils/failure_analysis.py`: disconnect and timeline analysis tools.
- `external/YOLO_SAM2`: YOLO-prompted SAM2 frame-by-frame inference.
- `external/MedSAM2`: MedSAM2 video propagation and SAM2/MedSAM2 model code.
- `external/MedSAM3`: optional text-prompted MedSAM3/LoRA experiments.
- `requirements.txt`: Python package list for the project-owned pipeline.

## Current Benchmark

The default config compares these families:

- YOLO plus SAM2 frame-by-frame segmentation.
- Ground-truth-box plus SAM2 frame-by-frame oracle baselines.
- YOLO-prompted MedSAM2 video propagation.
- Ground-truth-box-prompted MedSAM2 oracle propagation.
- Reduced-prompt MedSAM2 variants: first prompt only, stride 5, and stride 10.
- Optional disabled MedSAM3 text-prompt inference.

Use the YOLO methods to evaluate the full automatic pipeline. Use the `GT_BOX`
methods to isolate segmentation and propagation quality from detector quality.

## Expected Local Assets

Large assets are not committed here. The default config expects:

- `data/polypgen/seq*/images`: input frames.
- `data/polypgen/seq*/masks`: ground-truth masks.
- `checkpoints/polypgen_yolov8n.pt`: YOLO checkpoint.
- `checkpoints/MedSAM2_latest.pt`: MedSAM2/SAM2 checkpoint.
- `checkpoints/sam2_hiera_large.pt`: large SAM2 checkpoint for
  `SAM2_LARGE_GT_BOX_FRAME`.
- `checkpoints/MedSAM3_v1/best_lora_weights.pt`: optional MedSAM3 LoRA weights.
- `checkpoints/facebook_sam3/sam3.pt`: optional SAM3 base checkpoint.

The prepare stage creates `data/bbox/seq*/bboxes.csv` from masks.

## Setup

Create and activate your Python environment, then install the project
requirements:

```bash
pip install -r requirements.txt
```

Some external model folders may need their own setup or checkpoints depending on
which methods you run. Start with `--dry_run` to confirm paths and commands
before launching model inference.

## Quick Start

Run the enabled methods on the configured PolypGen sequences:

```bash
bash scripts/run_polypgen_experiment.sh
```

Preview commands without running models:

```bash
bash scripts/run_polypgen_experiment.sh --dry_run
```

Run a small smoke test command set:

```bash
bash scripts/run_polypgen_experiment.sh \
  --methods YOLO_SAM2_YOLO_BOX_FRAME \
  --seqs 1 2 3 \
  --dry_run
```

Run stages separately:

```bash
bash scripts/run_polypgen_experiment.sh --stage prepare
bash scripts/run_polypgen_experiment.sh --stage infer
bash scripts/run_polypgen_experiment.sh --stage eval
```

Useful method examples:

```bash
bash scripts/run_polypgen_experiment.sh --methods YOLO_SAM2_YOLO_BOX_FRAME
bash scripts/run_polypgen_experiment.sh --methods YOLO_SAM2_GT_BOX_FRAME
bash scripts/run_polypgen_experiment.sh --methods MedSAM2_YOLO_BOX_MASK_FIRST
bash scripts/run_polypgen_experiment.sh --methods MedSAM2_YOLO_BOX_MASK_STRIDE5
bash scripts/run_polypgen_experiment.sh --methods MedSAM2_GT_BOX_MASK_STRIDE10
```

## Reliability-Gated Output Experiment

`experiments/reliability_gated_memory_experiment.py` is a gated-only inference
and evaluation entrypoint. It does not run an ungated baseline and it does not
modify SAM2's internal feature-memory tensors. MedSAM2 first produces candidate
binary masks; the reliability gate then accepts the current candidate or holds
the most recent non-empty accepted mask.

Reliability combines a constant confidence proxy, adjacent-mask IoU, mask-area
consistency, and frame sharpness. The gate threshold is effective, and a forced
update occurs after `MAX_CONSECUTIVE_REJECTIONS` rejected candidates. Shared
mask helpers live in `scripts/utils/mask_utils.py`; reliability scoring and gate
state live in `scripts/utils/reliability_gate.py`; common Dice, IoU, temporal
IoU, natural sorting, and bbox extraction come from
`scripts/utils/eval_metrics.py`.

The script supports GT-box or YOLO prompts, prompt stride/limit controls,
explicit sequence ranges, CPU/MPS/CUDA candidate inference, threshold
ablations, and a guard that rejects non-Google-Drive output paths. With
GT-box prompting, each eligible stride frame uses its own frame-specific box;
frames without a box rely on video propagation.

Run the full 23-sequence stride experiment set directly into the synced Google
Drive `AdeSEG` folder:

```bash
cd /Users/maianhpham/Documents/AdeSEG
set -euo pipefail

DRIVE_ADESEG="$HOME/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG"

for STRIDE in 1 5 10; do
  RUN_ROOT="$DRIVE_ADESEG/outputs/reliability_gated_stride${STRIDE}"
  mkdir -p "$RUN_ROOT"

  PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  caffeinate -dimsu .venv/bin/python \
    experiments/reliability_gated_memory_experiment.py \
    --output-root "$RUN_ROOT" \
    --require-google-drive-output \
    --sequences 1-23 \
    --prompt-source gt_bbox \
    --video-prompt-stride "$STRIDE" \
    --video-prompt-limit 0 \
    --candidate-device cpu \
    --no-generate-bboxes \
    2>&1 | tee "$RUN_ROOT/run.log"
done
```

Each stride uses an isolated output root containing `_candidate/`, `gated/`,
`metrics.csv`, `summary.csv`, `experiment_notes.json`, and `run.log`. Missing
predictions are evaluated as blank failures. The first-frame temporal metric is
excluded as NaN. The experiment never creates `baseline/` or
`comparison.json` outputs.

### Completed stride results

| Stride | Dice | IoU | Temporal IoU | Prompts | Total time |
|---:|---:|---:|---:|---:|---:|
| 1 | **0.6919** | **0.6499** | 0.6750 | 1,710 | 15m 57s |
| 5 | 0.6419 | 0.5948 | 0.6934 | 349 | 12m 45s |
| 10 | 0.6255 | 0.5769 | **0.7051** | 174 | **9m 53s** |

Stride 1 has the best spatial accuracy; stride 10 is fastest and has the
highest temporal IoU. A direct diagnostic of the saved candidate masks shows
that the current output gate reduces Dice at every stride while increasing
temporal persistence. See
[`RESULTS_ANALYSIS_RELIABILITY_GATED_MEMORY.md`](RESULTS_ANALYSIS_RELIABILITY_GATED_MEMORY.md)
for the complete tables, sequence-level results, calibration audit, diagrams,
and recommended corrective experiments.

## Configuration

Edit experiment settings in:

```text
experiments/polypgen_medsam2_yolo_sam2.json
```

This file controls sequence ranges, enabled methods, checkpoints, model runners,
YOLO thresholds, prompt type, prompt stride, prompt limits, and output roots.
Prefer config edits over changing external model scripts.

## Outputs

Main output roots:

- `outputs/<method>/masks/seq*/predicted`: predicted binary masks.
- `outputs/<method>/bbox/seq*.csv`: YOLO prompt boxes for YOLO-prompted methods.
- `outputs/<method>/logs/seq*.json`: MedSAM2 inference logs.
- `outputs/<method>/eval/mask`: mask metrics and overlays.
- `outputs/<method>/eval/bbox`: detector bbox IoU summaries.
- `outputs/data_quality_report.csv`: raw PolypGen sequence quality scores.
- `outputs_MedSAM3/<method>`: optional MedSAM3 outputs.

Mask evaluation includes Dice, IoU, F-measure, sensitivity, specificity,
S-measure, E-measure, predicted area fraction, adjacent-frame IoU, area-change
smoothness, and centroid-shift smoothness.

## Failure Analysis

Audit raw PolypGen sequence quality before model inference:

```bash
python3 scripts/utils/data_quality.py
```

This writes a sequence-level report with no-reference capsule/endoscopy IQA
proxy scores such as contrast, entropy, sharpness, blur, blockiness,
natural-scene-statistics distortion, DCT frequency balance, and noise.

Find sequences where detector boxes are good but final masks are poor:

```bash
python3 scripts/utils/failure_analysis.py disconnect \
  --methods MedSAM2_YOLO_BOX_MASK YOLO_SAM2_YOLO_BOX_FRAME \
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

## Clinical Target

For the intended adenoid study, the clinical-data target is multi-object video
segmentation:

- `adenoid`: visible adenoid tissue.
- `nasopharynx_airway`: open nasopharyngeal airway, choana, or clinically
  defined reference cavity.

Core evaluation should include label-specific Dice/IoU, temporal stability,
correction burden, clinical grade agreement, and downstream
adenoid-to-nasopharynx or obstruction-ratio error. See
`docs/ADENOID_PROTOCOL.md` before creating clinical labels.

## Reading Order

Read in this order when onboarding:

1. `README.md` for the project map.
2. `docs/EXPERIMENTS.md` for exact runnable commands and output contracts.
3. `experiments/polypgen_medsam2_yolo_sam2.json` for active method settings.
4. `docs/ADENOID_PROTOCOL.md` for the clinical study design.
