#!/usr/bin/env python3
"""
Simple reliability-gated memory experiment for AdeSEG.

This script intentionally avoids a large config system. Constants below provide
defaults, while CLI arguments control run-specific paths and sequence selection:

    python experiments/reliability_gated_memory_experiment.py --help

What it does:
1. Runs MedSAM2 inference once per sequence into a scratch directory, with
   SAM2's internal memory bank disabled (SAM2_CFG points at num_maskmem=0)
   so each frame is an independent per-frame prediction, plus SAM2's own
   per-frame object-score confidence persisted to disk (see
   medsam2_infer_video_with_yolo.py). The raw per-frame baseline masks
   themselves are already produced/evaluated separately by
   scripts/experiments/run_experiment.py, so this script keeps only the
   confidence/prompt_confidence CSVs from that run and discards the masks.
2. Blends each frame's raw mask into a soft memory mask:
   M_t = r_t * current_mask + (1 - r_t) * M_(t-1), where r_t is a
   reliability score in [0, 1] (see scripts/utils/reliability_gate.py). This
   blend is the only cross-frame memory left in the pipeline.
3. Evaluates the blended (gated) masks against ground truth.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDSAM2_ROOT = PROJECT_ROOT / "external" / "MedSAM2"  # architecture from MedSAM
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MEDSAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDSAM2_ROOT))

from scripts.utils.eval_metrics import (
    calculate_dice,
    calculate_iou,
    calculate_temporal_iou,
    get_bbox_from_mask,
    get_numeric_sort_key,
)
from scripts.utils.mask_utils import (
    load_binary_mask,
    resize_binary_mask,
    resolve_mask_path,
    save_binary_mask,
)
from scripts.utils.reliability_gate import (
    ReliabilityConfig,
    apply_reliability_gate,
    area_change,
    area_fraction,
    centroid_shift,
    compute_reliability,
    safe_nanmean,
)

from external.MedSAM2.medsam2_infer_video_with_yolo import (
    get_frame_names,
    get_video_frame_dir,
    resolve_frame_path,
    write_sequence_log,
)

# ---------------------------------------------------------------------------
# Edit these constants instead of passing CLI arguments.
# ---------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "test" / "polypgen"
BBOX_ROOT = PROJECT_ROOT / "data" / "test" / "bbox"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "reliability_gated"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
SAM2_CFG = "configs/sam2.1_hiera_t512_no_memory.yaml"  # num_maskmem=0: no internal memory bank
SAM2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "MedSAM2_latest.pt"
YOLO_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "polypgen_yolov8n.pt"

# gt_bbox, yolo
PROMPT_SOURCE = "gt_bbox"  # Change to "yolo" to compare fully automatic prompting.
VIDEO_PROMPT_STRIDE = 1
VIDEO_PROMPT_LIMIT = 0

USE_BLUR_SCORE = True
SAVE_MASK_IMAGES = True  # also the source evaluate_variant reads predictions from

SEQUENCE_NAMES = None  # None means: scan DATA_ROOT and use every seq* folder.
MAX_YOLO_BOXES_PER_FRAME = 1  # only 1 box per frame
YOLO_CONF = 0.5
YOLO_IMGSZ = 640
ALLOW_BBOX_GENERATION = True
DEVICE = "cpu"


def parse_sequence_specs(specs: list[str] | None) -> list[str] | None:
    """Parse values such as ``1-23``, ``1,3,5``, or ``seq7``."""
    if not specs:
        return None
    sequence_ids = []
    for spec in specs:
        for token in spec.split(","):
            token = token.strip().lower().removeprefix("seq")
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", maxsplit=1)
                start, end = int(start_text), int(end_text)
                if start > end:
                    raise ValueError(f"Invalid descending sequence range: {spec}")
                sequence_ids.extend(range(start, end + 1))
            else:
                sequence_ids.append(int(token))
    return [f"seq{seq_id}" for seq_id in dict.fromkeys(sequence_ids)]


def is_google_drive_path(path: Path) -> bool:
    """Return whether a path is inside the macOS Google Drive CloudStorage mount."""
    parts = path.expanduser().absolute().parts
    return "CloudStorage" in parts and any(
        part.startswith("GoogleDrive-") for part in parts
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reliability-gated MedSAM2 inference and evaluation."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("ADESEG_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)),
        help="Root for every generated mask, confidence CSV, log, and metric artifact.",
    )
    parser.add_argument(
        "--require-google-drive-output",
        action="store_true",
        help="Fail unless --output-root is inside a GoogleDrive-* CloudStorage mount.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Sequence IDs/ranges, e.g. --sequences 1-23 or --sequences 1,3,5.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=DEVICE,
        help="Device used by the MedSAM2 inference subprocess.",
    )
    parser.add_argument(
        "--prompt-source",
        choices=["gt_bbox", "yolo"],
        default=PROMPT_SOURCE,
    )
    parser.add_argument(
        "--video-prompt-stride",
        type=int,
        default=VIDEO_PROMPT_STRIDE,
        help="Seed a prompt every Nth frame (1, 5, and 10 are typical ablations).",
    )
    parser.add_argument(
        "--video-prompt-limit",
        type=int,
        default=VIDEO_PROMPT_LIMIT,
        help="Maximum number of prompted frames per sequence; 0 means unlimited.",
    )
    parser.add_argument(
        "--no-generate-bboxes",
        action="store_true",
        help="Require existing bbox CSVs; do not write generated bbox files locally.",
    )
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global OUTPUT_ROOT, SEQUENCE_NAMES, DEVICE
    global PROMPT_SOURCE
    global ALLOW_BBOX_GENERATION, VIDEO_PROMPT_STRIDE, VIDEO_PROMPT_LIMIT

    output_root = args.output_root.expanduser().absolute()
    if args.require_google_drive_output and not is_google_drive_path(output_root):
        raise RuntimeError(
            "Refusing to run: --output-root is not inside a Google Drive "
            f"CloudStorage mount: {output_root}"
        )
    OUTPUT_ROOT = output_root
    SEQUENCE_NAMES = parse_sequence_specs(args.sequences)
    DEVICE = args.device
    PROMPT_SOURCE = args.prompt_source
    if args.video_prompt_stride < 1:
        raise ValueError("--video-prompt-stride must be at least 1")
    if args.video_prompt_limit < 0:
        raise ValueError("--video-prompt-limit cannot be negative")
    VIDEO_PROMPT_STRIDE = args.video_prompt_stride
    VIDEO_PROMPT_LIMIT = args.video_prompt_limit
    ALLOW_BBOX_GENERATION = not args.no_generate_bboxes


def list_sequences(data_root: Path) -> list[str]:
    if SEQUENCE_NAMES:
        missing = [name for name in SEQUENCE_NAMES if not (data_root / name).is_dir()]
        if missing:
            raise RuntimeError(f"Missing requested sequence directories: {missing}")
        return list(SEQUENCE_NAMES)
    return sorted(
        [
            path.name
            for path in data_root.iterdir()
            if path.is_dir() and path.name.startswith("seq")
        ],
        key=get_numeric_sort_key,
    )


def ensure_gt_bboxes(sequence_names: list[str]) -> None:
    if PROMPT_SOURCE != "gt_bbox":
        return

    for sequence_name in sequence_names:
        masks_dir = DATA_ROOT / sequence_name / "masks"
        bbox_dir = BBOX_ROOT / sequence_name
        bbox_csv = bbox_dir / "bboxes.csv"
        if bbox_csv.is_file():
            # Header-only CSVs are valid for sequences with no foreground masks.
            continue
        if not ALLOW_BBOX_GENERATION:
            raise RuntimeError(
                f"Missing {bbox_csv}; --no-generate-bboxes forbids creating it."
            )

        bbox_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        frame_names = get_frame_names(str(masks_dir))
        for frame_name in frame_names:
            mask_path = resolve_mask_path(masks_dir, frame_name)
            if mask_path is None:
                continue
            mask = load_binary_mask(mask_path)
            bbox = get_bbox_from_mask(mask)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            rows.append(
                {
                    "frame_idx": frame_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        with open(bbox_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["frame_idx", "x1", "y1", "x2", "y2"]
            )
            writer.writeheader()
            writer.writerows(rows)

def run_raw_inference(sequence_names: list[str], work_root: Path) -> None:
    """Run MedSAM2 for all sequences in one process so models load only once.

    Writes masks/confidence/prompt_confidence under work_root/raw/. The
    masks are scratch input for the gating blend below and get deleted once
    consumed; only the confidence/prompt_confidence CSVs are kept.
    """
    raw_logs_root = work_root / "logs"
    sequence_numbers = [name.removeprefix("seq") for name in sequence_names]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "external" / "MedSAM2" / "medsam2_infer_video_with_yolo.py"),
        "--base_video_dir",
        str(DATA_ROOT),
        "--seq_nums",
        *sequence_numbers,
        "--output_root",
        str(work_root),
        "--method_name",
        "raw",
        "--log_dir",
        str(raw_logs_root),
        "--sam2_cfg",
        SAM2_CFG,
        "--sam2_checkpoint",
        str(SAM2_CHECKPOINT),
        "--yolo_checkpoint",
        str(YOLO_CHECKPOINT),
        "--prompt_source",
        PROMPT_SOURCE,
        "--bbox_root",
        str(BBOX_ROOT),
        "--yolo_conf",
        str(YOLO_CONF),
        "--yolo_imgsz",
        str(YOLO_IMGSZ),
        "--max_yolo_boxes_per_frame",
        str(MAX_YOLO_BOXES_PER_FRAME),
        "--video_prompt_source",
        "box",
        "--video_prompt_stride",
        str(VIDEO_PROMPT_STRIDE),
        "--video_prompt_limit",
        str(VIDEO_PROMPT_LIMIT),
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    path_parts = [str(PROJECT_ROOT), str(MEDSAM2_ROOT)]
    if existing_pythonpath:
        path_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if DEVICE != "auto":
        env["SAM2_DEVICE"] = DEVICE
    else:
        env.pop("SAM2_DEVICE", None)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def resolve_raw_mask_path(
    work_root: Path, sequence_name: str, frame_name: str
) -> Path | None:
    mask_path = (
        work_root / "raw" / "masks" / sequence_name / "predicted" / f"{frame_name}.png"
    )
    return mask_path if mask_path.exists() else None


def load_confidence(confidence_root: Path, sequence_name: str) -> dict[str, float]:
    """Read the per-frame sigmoid(object_score_logits) CSV that
    medsam2_infer_video_with_yolo.py writes next to the raw masks."""
    csv_path = confidence_root / f"{sequence_name}.csv"
    if not csv_path.is_file():
        return {}
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["frame"]: float(row["confidence"]) for row in reader}


def load_prompt_confidence(
    prompt_confidence_root: Path, sequence_name: str
) -> dict[str, float]:
    """Read the per-frame prompt-box confidence CSV that
    medsam2_infer_video_with_yolo.py writes (YOLO detection score, or 1.0
    for a ground-truth box); frames with no prompt have no entry."""
    csv_path = prompt_confidence_root / f"{sequence_name}.csv"
    if not csv_path.is_file():
        return {}
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["frame"]: float(row["confidence"]) for row in reader}


def run_gated(sequence_names: list[str], output_root: Path, gated_root: Path) -> None:
    """Run raw inference once, then blend the raw masks into a soft memory mask per frame.

    Only the confidence/prompt_confidence CSVs from the raw inference run are
    kept (under output_root/confidence and output_root/prompt_confidence);
    the raw mask images are scratch input for the blend below and are
    deleted once every sequence has been processed.
    """
    work_root = output_root / "_work"
    confidence_root = output_root / "confidence"
    prompt_confidence_root = output_root / "prompt_confidence"
    run_raw_inference(sequence_names, work_root)
    shutil.copytree(work_root / "raw" / "confidence", confidence_root, dirs_exist_ok=True)
    shutil.copytree(
        work_root / "raw" / "prompt_confidence", prompt_confidence_root, dirs_exist_ok=True
    )

    masks_root = gated_root / "masks"
    logs_root = gated_root / "logs"
    reliability_config = ReliabilityConfig(use_blur_score=USE_BLUR_SCORE)

    for index, sequence_name in enumerate(sequence_names, start=1):
        print(f"[gated] {index}/{len(sequence_names)} {sequence_name}", flush=True)
        frame_dir = Path(get_video_frame_dir(str(DATA_ROOT), sequence_name))
        frame_names = get_frame_names(str(frame_dir))
        first_frame_path = Path(resolve_frame_path(str(frame_dir), frame_names[0]))
        height, width = np.array(Image.open(first_frame_path).convert("RGB")).shape[:2]

        memory_mask = None  # M_(t-1): float mask in [0, 1], None before frame 0
        rows = []
        empty_raw_count = 0
        raw_confidence = load_confidence(confidence_root, sequence_name)
        prompt_confidence = load_prompt_confidence(prompt_confidence_root, sequence_name)

        for frame_idx, frame_name in enumerate(frame_names):
            frame_path = Path(resolve_frame_path(str(frame_dir), frame_name))
            frame_rgb = np.array(Image.open(frame_path).convert("RGB"))
            frame_bgr = frame_rgb[:, :, ::-1]
            raw_mask_path = resolve_raw_mask_path(work_root, sequence_name, frame_name)
            current_mask = (
                load_binary_mask(raw_mask_path)
                if raw_mask_path is not None
                else np.zeros((height, width), dtype=np.uint8)
            )
            current_mask = resize_binary_mask(current_mask, (height, width))
            if current_mask.sum() == 0:
                empty_raw_count += 1
            previous_binary_mask = (
                None if memory_mask is None else (memory_mask >= 0.5).astype(np.uint8)
            )

            signals = compute_reliability(
                current_mask=current_mask,
                previous_mask=previous_binary_mask,
                frame_bgr=frame_bgr,
                mask_confidence=raw_confidence.get(frame_name),
                prompt_confidence=prompt_confidence.get(frame_name),
                config=reliability_config,
            )
            reliability = signals["reliability"]

            # Soft memory update (no internal SAM2/MedSAM2 tensors touched):
            # M_t = r_t * current_mask + (1 - r_t) * M_(t-1)
            memory_mask = apply_reliability_gate(
                current_mask=current_mask,
                previous_memory_mask=memory_mask,
                reliability=reliability,
            )
            final_mask = (memory_mask >= 0.5).astype(np.uint8)

            if SAVE_MASK_IMAGES:
                output_mask_path = (
                    masks_root / sequence_name / "predicted" / f"{frame_name}.png"
                )
                save_binary_mask(final_mask, output_mask_path)

            rows.append(
                {
                    "frame_idx": frame_idx,
                    "frame": frame_name,
                    "num_boxes": 1 if current_mask.sum() > 0 else 0,
                    "reliability": reliability,
                    "r_conf": signals["r_conf"],
                    "r_prompt": signals["r_prompt"],
                    "r_boundary": signals["r_boundary"],
                    "r_blur": signals["r_blur"],
                    "memory_area_frac": area_fraction(final_mask),
                }
            )

        if empty_raw_count > len(frame_names) / 2:
            print(
                f"[gated] {sequence_name}: {empty_raw_count}/{len(frame_names)} "
                "frames had no raw mask (expected if video_prompt_stride>1 "
                "with the memory bank disabled); relying on the reliability blend "
                "to carry memory across these gaps.",
                flush=True,
            )

        write_sequence_log(
            str(logs_root),
            sequence_name,
            {
                "video": sequence_name,
                "status": "success",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "height": int(height),
                "width": int(width),
                "prompt_source": PROMPT_SOURCE,
                "video_prompt_source": "box",
                "video_prompt_stride": VIDEO_PROMPT_STRIDE,
                "video_prompt_limit": VIDEO_PROMPT_LIMIT,
                "reliability_formula": (
                    "0.35*r_conf (SAM2 object-score confidence) + "
                    "0.25*r_prompt (prompt-box confidence, neutral 0.5 if no "
                    "prompt this frame) + 0.30*r_boundary + 0.10*r_blur, with "
                    "blank/tiny/huge-mask penalties; no temporal or area-ratio "
                    "term (memory bank disabled, num_maskmem=0)"
                ),
                "state_update": (
                    "soft memory blend: M_t = r_t*current_mask + (1-r_t)*M_(t-1); "
                    "saved/evaluated mask is M_t thresholded at 0.5"
                ),
                "internal_sam2_memory_modified": False,
                "rows": rows,
            },
        )

        csv_path = logs_root / f"{sequence_name}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    shutil.rmtree(work_root, ignore_errors=True)


def evaluate_variant(
    variant_name: str,
    sequence_names: list[str],
    masks_root: Path,
    logs_root: Path | None,
) -> None:
    """Score `masks_root/<seq>/predicted/*.png` against ground truth.

    `logs_root` supplies the per-frame reliability CSVs written by
    run_gated; pass None for variants with no reliability log, and the
    "reliability" column is left NaN for those.
    """
    eval_root = OUTPUT_ROOT / variant_name / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []
    summary_rows = []
    reliability_by_frame = {}

    if logs_root is not None:
        for sequence_name in sequence_names:
            log_csv = logs_root / f"{sequence_name}.csv"
            if not log_csv.is_file():
                continue
            with open(log_csv, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    reliability_by_frame[(sequence_name, row["frame"])] = float(
                        row["reliability"]
                    )

    for sequence_name in sequence_names:
        gt_dir = DATA_ROOT / sequence_name / "masks"
        frame_dir = Path(get_video_frame_dir(str(DATA_ROOT), sequence_name))
        pred_dir = masks_root / sequence_name / "predicted"
        # Evaluate the source-frame contract, not merely the predictions that
        # happen to exist. A missing prediction is an explicit blank failure.
        frame_names = get_frame_names(str(frame_dir))
        previous_pred = None
        sequence_rows = []
        for frame_name in frame_names:
            gt_path = None
            pred_path = None
            predicted_path = pred_dir / f"{frame_name}.png"
            if predicted_path.exists():
                pred_path = predicted_path
            if gt_dir.is_dir():
                gt_path = resolve_mask_path(gt_dir, frame_name)

            gt_mask = load_binary_mask(gt_path) if gt_path is not None else None
            prediction_missing = pred_path is None
            if pred_path is not None:
                pred_mask = load_binary_mask(pred_path)
            elif gt_mask is not None:
                pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
            else:
                frame_path = Path(resolve_frame_path(str(frame_dir), frame_name))
                with Image.open(frame_path) as frame_image:
                    pred_mask = np.zeros(
                        (frame_image.height, frame_image.width), dtype=np.uint8
                    )
            if gt_mask is not None and gt_mask.shape != pred_mask.shape:
                gt_mask = resize_binary_mask(gt_mask, pred_mask.shape)
            row = {
                "sequence": sequence_name,
                "variant": variant_name,
                "frame": frame_name,
                "frame_idx": int(frame_name)
                if str(frame_name).isdigit()
                else frame_name,
                "prediction_missing": prediction_missing,
                "dice": float("nan")
                if gt_mask is None
                else calculate_dice(pred_mask, gt_mask),
                "iou": float("nan")
                if gt_mask is None
                else calculate_iou(pred_mask, gt_mask),
                "pred_area_frac": area_fraction(pred_mask),
                "gt_area_frac": float("nan")
                if gt_mask is None
                else area_fraction(gt_mask),
                "temporal_iou": float("nan")
                if previous_pred is None
                else calculate_temporal_iou(pred_mask, previous_pred),
                "centroid_shift": float("nan")
                if previous_pred is None
                else centroid_shift(pred_mask, previous_pred),
                "area_change": float("nan")
                if previous_pred is None
                else area_change(pred_mask, previous_pred),
                "reliability": reliability_by_frame.get(
                    (sequence_name, frame_name), float("nan")
                ),
            }
            previous_pred = pred_mask
            sequence_rows.append(row)
            aggregate_rows.append(row)

        if not sequence_rows:
            continue

        seq_csv = eval_root / f"{sequence_name}.csv"
        with open(seq_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sequence_rows)

        summary_rows.append(
            {
                "video_id": sequence_name,
                "variant": variant_name,
                "num_frames": len(sequence_rows),
                "num_predictions_missing": int(
                    sum(1 for row in sequence_rows if row["prediction_missing"])
                ),
                "mean_dice": safe_nanmean(row["dice"] for row in sequence_rows),
                "mean_iou": safe_nanmean(row["iou"] for row in sequence_rows),
                "mean_temporal_iou": safe_nanmean(
                    row["temporal_iou"] for row in sequence_rows
                ),
                "mean_centroid_shift": safe_nanmean(
                    row["centroid_shift"] for row in sequence_rows
                ),
                "mean_area_change": safe_nanmean(
                    row["area_change"] for row in sequence_rows
                ),
                "mean_reliability": safe_nanmean(
                    row["reliability"] for row in sequence_rows
                ),
            }
        )

    if summary_rows:
        with open(
            eval_root / "summary.csv", "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if aggregate_rows:
        global_summary = {
            "variant": variant_name,
            "num_sequences": len(summary_rows),
            "num_frames": len(aggregate_rows),
            "num_predictions_missing": int(
                sum(1 for row in aggregate_rows if row["prediction_missing"])
            ),
            "mean_dice": safe_nanmean(row["dice"] for row in aggregate_rows),
            "mean_iou": safe_nanmean(row["iou"] for row in aggregate_rows),
            "mean_temporal_iou": safe_nanmean(
                row["temporal_iou"] for row in aggregate_rows
            ),
            "mean_centroid_shift": safe_nanmean(
                row["centroid_shift"] for row in aggregate_rows
            ),
            "mean_area_change": safe_nanmean(
                row["area_change"] for row in aggregate_rows
            ),
            "mean_reliability": safe_nanmean(
                row["reliability"] for row in aggregate_rows
            ),
        }
        with open(eval_root / "global_summary.json", "w", encoding="utf-8") as handle:
            json.dump(global_summary, handle, indent=2)


def main(args: argparse.Namespace | None = None) -> None:
    configure_from_args(parse_args() if args is None else args)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sequence_names = list_sequences(DATA_ROOT)
    if not sequence_names:
        raise RuntimeError(f"No seq* folders found under {DATA_ROOT}")
    ensure_gt_bboxes(sequence_names)

    print(f"Inference device: {DEVICE}")
    print(f"Video prompt stride: {VIDEO_PROMPT_STRIDE}")
    print(f"Video prompt limit: {VIDEO_PROMPT_LIMIT}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Sequences: {sequence_names}")
    print(f"Outputs: {OUTPUT_ROOT}")

    gated_root = OUTPUT_ROOT / "gated"
    run_gated(sequence_names, OUTPUT_ROOT, gated_root)

    evaluate_variant(
        "gated",
        sequence_names,
        masks_root=gated_root / "masks",
        logs_root=gated_root / "logs",
    )

    notes = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": str(DATA_ROOT),
        "bbox_root": str(BBOX_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "output_is_google_drive": is_google_drive_path(OUTPUT_ROOT),
        "prompt_source": PROMPT_SOURCE,
        "video_prompt_source": "box",
        "video_prompt_stride": VIDEO_PROMPT_STRIDE,
        "video_prompt_limit": VIDEO_PROMPT_LIMIT,
        "sequence_names": sequence_names,
        "device": DEVICE,
        "sam2_cfg": SAM2_CFG,
        "sam2_checkpoint": str(SAM2_CHECKPOINT),
        "yolo_checkpoint": str(YOLO_CHECKPOINT),
        "fallback_note": (
            "SAM2's internal memory bank is disabled (num_maskmem=0); "
            "each raw mask is an independent per-frame prediction. "
            "Reliability gating blends those raw masks into the only "
            "cross-frame memory in the pipeline: "
            "M_t = r_t*current_mask + (1-r_t)*M_(t-1)."
        ),
    }
    with open(OUTPUT_ROOT / "experiment_notes.json", "w", encoding="utf-8") as handle:
        json.dump(notes, handle, indent=2)

    print("Finished reliability-gated memory experiment.")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
