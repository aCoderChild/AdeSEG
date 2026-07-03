# Technical Inspection: Reliability-Gated MedSAM2 Experiment

## 1. Current scope and verified interpretation

This report describes the current implementation of
`experiments/reliability_gated_memory_experiment.py` after the June 2026
refactor.

The entrypoint is **gated-only**. It does not contain or run an ungated baseline,
does not create a `baseline/` directory, and does not write a baseline
comparison file. Its execution is:

```text
frames + frame-specific GT/YOLO boxes
  -> MedSAM2 candidate video inference
  -> candidate binary PNGs
  -> reliability scoring
  -> accept current mask or hold previous accepted non-empty mask
  -> gated masks
  -> gated evaluation and summaries
```

The term “memory” must be interpreted carefully. The experiment uses MedSAM2's
ordinary internal temporal memory while producing candidates, but the
reliability gate does **not** edit SAM2 memory features, object pointers, logits,
or inference state. It is an output-mask state gate.

The inspected checkpoint strictly loads into a SAM2.1 Hiera-T 512 model with
38,962,498 instantiated parameters. The checkpoint contains model weights only;
its original optimizer, scheduler, loss weights, training data, epochs, and
model-selection criterion cannot be recovered from the artifact.

## 2. Relevant files and responsibilities

| File | Current responsibility |
|---|---|
| `experiments/reliability_gated_memory_experiment.py` | CLI, sequence selection, bbox availability, candidate subprocess, gated postprocessing, evaluation, and reports. |
| `scripts/utils/reliability_gate.py` | `ReliabilityConfig`, reliability components, blur score, area/centroid temporal statistics, rejection state machine, and NaN-safe means. |
| `scripts/utils/mask_utils.py` | Binary mask loading, JPEG-safe thresholding, nearest-neighbor resizing, path resolution, and mask saving. |
| `scripts/utils/eval_metrics.py` | Shared natural sort key, tight bbox extraction, Dice, IoU, and adjacent-mask IoU. |
| `external/MedSAM2/medsam2_infer_video_with_yolo.py` | GT/YOLO prompt generation, image-predictor mask prompts, SAM2 video propagation, and candidate serialization. |
| `external/MedSAM2/sam2/utils/misc.py` | Video frame loading. It now natural-sorts unpadded names so model frame indices match prompt/output indices. |
| `external/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml` | Active neural architecture. |
| `checkpoints/MedSAM2_latest.pt` | Active learned weights. |

The repository-wide JSON experiment config is not read by this entrypoint. Its
runtime controls are Python defaults plus CLI arguments.

## 3. Architecture and dependency graph

```text
data/test/polypgen/seq*/images/*.jpg
data/test/polypgen/seq*/masks/*
data/test/bbox/seq*/bboxes.csv
                 |
                 v
Candidate subprocess (once per stride experiment)
  external/MedSAM2/medsam2_infer_video_with_yolo.py
                 |
                 +-- prompt_source=gt_bbox
                 |     frame-specific tight GT box when available
                 |
                 `-- prompt_source=yolo
                       top-confidence detection when available
                 |
                 v
  VIDEO_PROMPT_SOURCE=mask
  frame RGB + box -> SAM2 image predictor -> binary mask prompt
                 |
                 v
  SAM2 video predictor
  Hiera/FPN -> memory encoder/attention -> SAM decoder
                 |
                 v
  OUTPUT_ROOT/_candidate/candidate/masks/seq*/predicted/*.png
                 |
                 v
Reliability gate (`scripts/utils/reliability_gate.py`)
  current mask + previous accepted state + frame sharpness
                 |
                 v
  OUTPUT_ROOT/gated/masks/seq*/predicted/*.png
                 |
                 v
Evaluation
  per-frame CSV -> per-sequence summary -> global/project summaries
```

No parent image/video model is built in the experiment process. Candidate
models are constructed only inside one child process that receives all selected
sequence IDs, so model weights are loaded once per stride experiment rather
than once per sequence.

## 4. Prompt behavior and stride semantics

### Default GT-box mode

`PROMPT_SOURCE="gt_bbox"` uses `data/test/bbox/seq*/bboxes.csv`. When bbox
generation is allowed and a CSV is missing, every non-empty GT mask is converted
to its tight half-open box:

```text
(xmin, ymin, xmax + 1, ymax + 1)
```

Empty masks have no box. Header-only bbox files are valid for all-empty
sequences. The recommended Google Drive command uses
`--no-generate-bboxes`, making these local bbox files read-only inputs.

### Prompt schedule

The candidate wrapper loops over naturally ordered frame indices. A frame is
eligible when:

```text
frame_idx % video_prompt_stride == 0
```

`--video-prompt-limit 0` means no limit. A stride-eligible frame is seeded only
if it also has a GT or YOLO box. If index 5 has no box under stride 5, the code
does not move the prompt to index 6; that scheduled prompt is simply absent.

The three requested experiments therefore test:

| Experiment | Eligible indices | Expected behavior |
|---|---|---|
| stride 1 | 0, 1, 2, … | Densest prompting, highest prompt cost, least reliance on propagation. |
| stride 5 | 0, 5, 10, … | Moderate prompting and greater temporal-memory dependence. |
| stride 10 | 0, 10, 20, … | Sparse prompting, lowest prompt cost, greatest drift/recovery stress. |

At most one box is used per frame by default. All prompts are assigned object ID
1, matching the single-polyp task.

### Mask prompt conversion

`VIDEO_PROMPT_SOURCE="mask"` means a box does not go directly into video
tracking. For each prompted frame:

1. Load RGB `[H0,W0,3]` uint8.
2. Resize to `512x512`, convert to `[0,1]`, and apply ImageNet normalization.
3. Transform the original XYXY box into 512-square coordinates.
4. Run the SAM2 image prompt encoder and mask decoder.
5. Threshold original-resolution logits at zero.
6. Inject the resulting bool mask into the SAM2 video predictor.

Unprompted frames are produced by SAM2 video propagation.

## 5. Neural model internals

### Image encoder

The active Hiera-T configuration uses:

- input size 512;
- 7x7 patch convolution, stride 4, padding 3;
- initial embedding dimension 96;
- stages `[1,2,7,2]`;
- global-attention blocks `[5,7,9]`;
- stage windows `[8,4,14,7]` from constructor defaults;
- FPN output dimension 256;
- high-resolution FPN levels at 128x128 and 64x64;
- final image embedding `[B,256,32,32]`.

For one frame:

```text
[B,3,512,512]
 -> [B,96,128,128]
 -> [B,192,64,64]
 -> [B,384,32,32]
 -> [B,768,16,16]
 -> FPN/scalp
 -> [B,256,128,128], [B,256,64,64], [B,256,32,32]
```

### Prompt encoder and mask decoder

Box corners become two sparse 256-D prompt tokens. The decoder uses a two-layer
two-way transformer with 8 heads and 2,048-wide MLPs. High-resolution decoder
skips produce low-resolution mask logits `[B,1,128,128]`, which are bilinearly
resized to original image dimensions.

### Video memory

The active configuration stores up to seven mask-memory positions nominally.
The memory encoder combines `[B,256,32,32]` image features with a mask
representation, applies two ConvNeXt-style fusion blocks, and projects to
`[B,64,32,32]`. Stored memory features are converted to bfloat16.

Four memory-attention layers condition 1,024 current image tokens on spatial
memories and 256-D object pointers. With unlimited conditioning frames,
cross-attention cost grows with the number of prompted frames; increasing prompt
stride reduces this conditioning set and image-prompt work.

## 6. Reliability gate

The configuration defaults are defined by `ReliabilityConfig`:

```text
r_confidence weight = 0.35
r_temporal   weight = 0.30
r_area       weight = 0.25
r_blur       weight = 0.10
```

The current call supplies `mask_confidence=0.5`, so confidence is a constant
proxy rather than learned uncertainty.

When a previous non-empty accepted mask exists:

```text
r_temporal = IoU(current, previous)

r_area = exp(
    -abs(log((current_area + 1e-6) / (previous_area + 1e-6)))
)

r_blur = clip(LaplacianVariance(gray) / 150, 0, 1)
```

Without usable previous state, temporal and area scores are both 0.5. Active
weights are normalized; disabling blur removes and renormalizes its term rather
than granting a perfect blur score.

After the weighted score:

- current empty while previous is non-empty: multiply by 0.25;
- current area fraction below 0.0005 or above 0.80: multiply by 0.50;
- clamp final reliability to `[0,1]`.

### State transition

For threshold `tau`:

```text
if no previous valid state:
    accept current
elif reliability >= tau:
    accept current and reset rejection count
else:
    hold previous and increment rejection count

if the rejection count becomes 4:
    force current and reset rejection count
```

`MAX_CONSECUTIVE_REJECTIONS=3` therefore permits three held outputs and forces
the fourth rejected candidate. Only a non-empty final mask replaces
`previous_valid_mask`; a forced blank output does not erase the stored non-empty
state.

`RELIABILITY_THRESHOLD=0.35` is effective. With
`--threshold-ablation`, the same candidate masks are reused for thresholds 0.3,
0.5, and 0.7, avoiding repeated neural inference inside that run.

## 7. Inputs and outputs

### Inputs

| Input | Type and shape | Processing |
|---|---|---|
| RGB frame | JPEG, uint8 `[H0,W0,3]` | Direct 512-square resize and ImageNet normalization for SAM2. |
| GT mask | image file -> uint8 `[H0,W0]` in `{0,1}` | Native 0/1 masks threshold at 0; ordinary/JPEG masks threshold at 127. |
| GT box | float32 XYXY | Frame-specific and transformed from original coordinates to 512 coordinates. |
| YOLO box | float32 XYXY + confidence | Used only with `--prompt-source yolo`; top box retained by default. |
| Candidate mask | uint8 `[H0,W0]` in `{0,1}` | Loaded from Drive candidate PNGs and resized nearest-neighbor if necessary. |
| Previous state | uint8 `[H0,W0]` or None | Last non-empty final gated mask. |

### Output tree

For one stride run:

```text
OUTPUT_ROOT/
  _candidate/
    candidate/masks/seq*/predicted/*.png
    logs/seq*.json
  gated/
    masks/seq*/predicted/*.png
    logs/seq*.json
    logs/seq*.csv
    eval/seq*.csv
    eval/summary.csv
    eval/global_summary.json
  metrics.csv
  summary.csv
  experiment_notes.json
  run.log                    # created by the documented shell command
```

Threshold ablation additionally creates `gated_thr_0.3`, `gated_thr_0.5`, and
`gated_thr_0.7` directories. No baseline or comparison artifacts are created.

## 8. Evaluation

Evaluation enumerates source frames, not predicted files. If a prediction is
missing, it creates an in-memory blank mask and records
`prediction_missing=True`. Missing outputs therefore lower metrics instead of
silently disappearing.

Per-frame fields include:

- Dice and IoU;
- predicted and GT area fraction;
- adjacent predicted-mask IoU;
- centroid shift in pixels;
- absolute foreground-area change;
- reliability and accepted/rejected decision;
- missing-prediction flag.

The first frame's temporal IoU, centroid shift, and area change are NaN and are
excluded by NaN-aware means. Global metrics are frame-weighted, so longer
sequences contribute more frames. Per-sequence means remain available in
`gated/eval/summary.csv`.

## 9. Training behavior

This entrypoint is inference-only:

- no training dataset or dataloader;
- no augmentation;
- no loss construction;
- no backward pass;
- no optimizer or scheduler;
- no AMP autocast or GradScaler;
- no gradient accumulation or clipping;
- no EMA;
- no early stopping;
- no DDP;
- no checkpoint saving.

The top-level call runs under `torch.inference_mode()`. Model builders set eval
mode and load the checkpoint strictly.

## 10. Current runtime controls

| CLI option / constant | Default | Effect |
|---|---|---|
| `--output-root` | local default | Root for candidates, gated masks, logs, and metrics. Use a unique Drive root per stride. |
| `--require-google-drive-output` | false | Rejects paths outside a macOS `GoogleDrive-*` CloudStorage namespace. |
| `--sequences` | scan all `seq*` | Accepts `1-23`, comma lists, or individual IDs. |
| `--candidate-device` | CPU | `auto`, `cuda`, `mps`, or `cpu` for the child process. |
| `--prompt-source` | `gt_bbox` | Oracle frame-specific boxes or YOLO detections. |
| `--video-prompt-stride` | 1 | Eligible prompt interval. Must be at least 1. |
| `--video-prompt-limit` | 0 | Maximum prompted frames per sequence; 0 is unlimited. |
| `--threshold-ablation` | false | Reuses candidates for gate thresholds 0.3/0.5/0.7. |
| `--no-generate-bboxes` | false | Makes missing bbox CSVs fatal instead of writing local generated files. |
| `MAX_YOLO_BOXES_PER_FRAME` | 1 | Single-object prompt cap. |
| `YOLO_CONF` | 0.5 | Detector threshold in YOLO mode. |
| `YOLO_IMGSZ` | 640 | Detector inference size in YOLO mode. |
| `RELIABILITY_THRESHOLD` | 0.35 | Active accept/hold cutoff. |
| `MAX_CONSECUTIVE_REJECTIONS` | 3 | Number of holds allowed before a forced current output. |
| `BLUR_REFERENCE` | 150 | Laplacian-variance normalization ceiling. |
| `MIN/MAX_MASK_AREA_RATIO` | 0.0005 / 0.80 | Implausible-size penalty boundaries. |

Architecture fields in the active YAML are checkpoint-coupled. Changing image
size, dimensions, depths, heads, or memory width requires compatible weights or
retraining.

## 11. Three-stride experiment command

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

This writes experiment artifacts into three isolated Google Drive directories.
`PYTHONDONTWRITEBYTECODE=1` avoids local Python bytecode writes during the run.
Google Drive for Desktop may still maintain its own transparent local cache;
the experiment itself does not place results in the repository's local
`outputs/` directory.

## 12. Bottlenecks and limitations

### Compute and memory

- Each prompted frame is encoded once by the image predictor to form a mask
  prompt and again by the video predictor.
- The video loader holds the complete resized sequence tensor; 250 frames at
  `[3,512,512]` float32 require roughly 750 MiB before model state.
- Spatial memory and unlimited conditioning frames can make dense stride-1
  inference expensive.
- CPU is the conservative default. MPS/CUDA can be faster but should be checked
  for backend compatibility and numerical comparability.

### I/O

- Candidate masks are saved, then reopened for gating.
- Direct Google Drive output can be slower because thousands of PNG/CSV files
  are synchronized.
- Source RGB and GT masks are reopened during gating/evaluation.

### Scientific limitations

- GT boxes use current-frame annotations and are an oracle prompt condition,
  not automatic inference.
- Reliability confidence is fixed at 0.5; learned image/video confidence is not
  propagated into the gate.
- Holding an unwarped prior mask can improve apparent temporal smoothness while
  becoming spatially stale under camera/object motion.
- Binary candidate masks discard logit uncertainty and prevent calibrated soft
  updates.
- The gate cannot repair boundaries; it chooses between current and previous
  binary outputs.
- Threshold tuning on the same 23 test sequences risks evaluation overfitting.

## 13. Recommended ablations and improvements

1. Compare stride 1, 5, and 10 using identical prompt source, checkpoint,
   threshold, and sequence set.
2. Add first-prompt-only (`--video-prompt-limit 1`) as a stronger memory stress
   test.
3. Compare GT boxes against YOLO boxes to separate oracle segmentation from
   detector errors.
4. Run `--threshold-ablation` on a validation subset before selecting a cutoff.
5. Carry image-predictor IoU scores, video object scores, and logits into the
   reliability computation instead of 0.5/binary masks.
6. Normalize centroid shift by image diagonal for cross-resolution comparison.
7. Add motion compensation before temporal IoU and prior-mask holding.
8. Cap conditioning frames or alter internal memory sampling to reduce dense
   stride-1 attention cost.
9. Replace the disk PNG round trip with per-sequence logits/arrays if Google
   Drive I/O becomes dominant.
10. If the research goal is truly reliability-gated SAM2 memory, apply the gate
    before encoding/storing a frame's internal memory and validate state
    consistency in both propagation directions.

## 14. Reproduction checklist

1. Use a fresh output root for each stride.
2. Record prompt source, stride, prompt limit, candidate device, checkpoint, and
   active YAML.
3. Keep `--no-generate-bboxes` when local data must remain read-only.
4. Verify `experiment_notes.json` reports `output_is_google_drive=true` and the
   expected sequence list/stride.
5. Confirm 23 sequences and 2,225 source frames appear in aggregate accounting.
6. Inspect `num_predictions_missing`; it should be zero after successful
   candidate and gated runs.
7. Compare per-sequence metrics as well as frame-weighted global means.
8. Do not describe GT-box results as fully automatic performance.
9. Do not describe the output-state gate as modifying SAM2 internal memory.

## 15. Final summary

- **Current pipeline:** one MedSAM2 candidate subprocess per stride, followed by
  gated-only postprocessing and evaluation.
- **Prompting:** frame-specific GT or YOLO boxes on stride-eligible frames;
  absent boxes fall back to propagation.
- **Gate:** effective thresholded accept/hold state machine with forced recovery
  after three holds.
- **Utilities:** shared metric/sort/bbox helpers plus dedicated mask and
  reliability utility modules.
- **Evaluation:** source-frame complete, missing-prediction aware, and NaN-safe
  for first-frame temporal metrics.
- **Outputs:** candidates, gated masks, logs, and summaries only; no baseline or
  comparison artifacts.
- **Main experiment:** full 23-sequence GT-box runs at prompt strides 1, 5, and
  10, each isolated under the Google Drive `AdeSEG/outputs` folder.

## 16. Completed-run analysis

The three full runs are complete: each contains 23 sequences, 2,225 evaluated
frames, and zero missing predictions.

| Stride | Dice | IoU | Temporal IoU | Prompt seeds | Rejected updates |
|---:|---:|---:|---:|---:|---:|
| 1 | **0.6919** | **0.6499** | 0.6750 | 1,710 | 291 |
| 5 | 0.6419 | 0.5948 | 0.6934 | 349 | 331 |
| 10 | 0.6255 | 0.5769 | **0.7051** | 174 | 340 |

The complete results audit is in
[`RESULTS_ANALYSIS_RELIABILITY_GATED_MEMORY.md`](RESULTS_ANALYSIS_RELIABILITY_GATED_MEMORY.md).
It includes candidate-versus-gated diagnostics, reliability calibration,
positive/empty-frame splits, all per-sequence results, paired statistics,
runtime analysis, diagrams, and prioritized corrective experiments. The key
finding is that the current output-state gate raises temporal IoU but lowers
candidate Dice at all three strides; it should not yet be interpreted as an
accuracy-improving reliability mechanism.
