# Experiment commands

This directory holds executable research entrypoints. The **method** is described under [docs/methodology](../docs/methodology/OVERVIEW.md), while commands, data and checkpoints are deliberately kept independent.

## Active Lean-SAM2 multiscale memory-write experiment

This is a **separate experimental path** from the original output-logit
RMFA refiner or TTMR token selector. It begins from an externally frozen
automatic **mask** prompt; YOLO/FGPI is not called inline.

| Command | Purpose |
|---|---|
| `python -m scripts.capture_lean_memory_train` | TRAIN-only predicted/native versus GT-conditioned teacher-memory pairs; GT never written into inference bank |
| `python -m training.lean_memory_fusion --input-mode full` | Train bounded residual memory fusion from sealed captures |
| `python -m scripts.run_lean_memory_fusion_ablation --arm native` | Native Lean baseline from frozen automatic masks |
| `python -m scripts.run_lean_memory_fusion_ablation --arm identity` | Zero-residual routing parity control |
| `python -m scripts.run_lean_memory_fusion_ablation --arm fusion` | Trained full or separately trained input-mode controls |
| `python -m scripts.score_lean_memory_ablations` | Independent paired video-level Dice/IoU, false alarms and memory telemetry |
| `python -m scripts.compare_native_tamp_indices` | Strict paired native TAMP retained-index overlap from opt-in dumps |

**Run with identical source commit, model checkpoint and frozen prompts.**
TAMP overlap analysis rejects no matches and incomplete event coverage
unless `--allow-incomplete-events` explicitly opts into an **exploratory**
partial comparison. It does not measure segmentation accuracy.

Use the exact commands and provenance requirements in the
[Lean memory-fusion experiment preflight](../docs/experiments/LEAN_MEMORY_FUSION_PREFLIGHT.md).
A trained checkpoint is not evidence of improved future-frame Dice.

## Historical / independent modules

## FGPI — automatic initialization

| Step | Command | Output |
|---|---|---|
| 1 | `python -m scripts.export_yolo_detections` | Target-class YOLO box JSONL and checkpoint/source hashes |
| 2 | `python -m scripts.infer_fgpi_video --mode box_only` | Matched detector-box SAM2 control |
| 3 | `python -m scripts.infer_fgpi_video --mode fgpi` | Joint YOLO-box and feature-derived point initialization |
| 4 | `python -m scripts.evaluate_fgpi_video` | Prompt-frame and post-prompt metrics |
| 5 | `python -m scripts.compare_fgpi_video` | Paired initial and propagated comparison |

Run each command with `--help` for exact required arguments. Both inference modes must use the **same frozen detections and pretrained checkpoint**. The current runner targets **MedSAM2 Hiera-T/512**. See [FGPI scientific formulation](../docs/methodology/FGPI.md) and [implementation notes](../docs/methodology/FGPI_IMPLEMENTATION_NOTES.md).

## RMFA — learned spatial refinement

| Command | Purpose |
|---|---|
| `python -m scripts.train_lean` | Train Lean-SAM2 structured channel-value aggregation on frozen TRAIN captures |
| `python -m scripts.ablate` | Matched-protocol study of native versus refined predictions |

RMFA's six computation blocks live under `modeling/aggregation_blocks/`; their trained checkpoint format and source identity checks are deliberately retained. Consult [RMFA](../docs/methodology/RMFA.md) and [training guidance](../training/README.md).

## TTMR — temporal selection

`adeseg.modules.TTMR` is an offline reference selector, not yet a full video CLI. Lean native memory functionality and a separate multiscale-anchor experiment remain under `modeling/lean_multiscale_anchor.py`. See [TTMR](../docs/methodology/TTMR.md).

## Baseline, reproducibility, and supplementary utilities

| Command | Role |
|---|---|
| `python -m scripts.compare_models` | Author SAM2, Lean-SAM2, SAM2Long and SAMURAI comparisons |
| `python -m scripts.evaluate_lean` | Oracle-GT-initialized author Lean comparison; not automatic FGPI |
| `python -m scripts.evaluate_cmsa` | CMSA-Net image-only evaluation |
| `python -m scripts.infer_med_video` | Native/adapter MedSAM2 oracle-first-mask study |
| `python -m scripts.evaluate_med_video` | Its strict video scorer |
| `python -m scripts.prepare_med_video_protocol` | Sealed oracle-video protocol |
| `python -m scripts.compare_med_runs` | Matched native/zero/learned adapter comparison |
| `python -m scripts.train_med` | MedSAM2 high-resolution skip adapter training |
| `python -m scripts.research` | Six-stage research workflow and receipt checks |
| `python -m scripts.validate_results` | Independent saved-prediction scoring |

The historical experiment details are on the [backup branch](https://github.com/aCoderChild/AdeSEG/tree/backup/experiment-full-20261118/reports). See [experiment preflight and controls](../docs/experiments/PLAN.md).
