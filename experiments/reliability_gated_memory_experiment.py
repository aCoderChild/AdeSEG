#!/usr/bin/env python3
"""
Simple reliability-gated memory experiment for AdeSEG.

This script intentionally avoids a large config system. Constants below provide
defaults, while CLI arguments control run-specific paths and sequence selection:

    python experiments/reliability_gated_memory_experiment.py --help

What it does:
1. Runs a simple reliability-gated mask-memory variant.
2. Saves candidates, masks, logs, and metrics under a single output root.
3. Writes per-frame and aggregate metrics when ground-truth masks exist.

Important: reliability gating is applied to accepted output masks. It does not
modify SAM2's internal feature-memory tensors.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
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
SAM2_CFG = "configs/sam2.1_hiera_t512.yaml"
SAM2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "MedSAM2_latest.pt"
YOLO_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "polypgen_yolov8n.pt"

# gt_bbox, yolo
PROMPT_SOURCE = "gt_bbox"  # Change to "yolo" to compare fully automatic prompting.
VIDEO_PROMPT_SOURCE = "mask"
VIDEO_PROMPT_STRIDE = 1
VIDEO_PROMPT_LIMIT = 0
RELIABILITY_THRESHOLD = 0.35
RELIABILITY_THRESHOLDS = [0.3, 0.5, 0.7]
MAX_CONSECUTIVE_REJECTIONS = 3

USE_BLUR_SCORE = True
SAVE_MASK_IMAGES = True
SAVE_NUMPY_MASKS = False
SAVE_DEBUG_VISUALIZATIONS = False
NUM_DEBUG_SAMPLES = 20

RUN_THRESHOLD_ABLATION = False

SEQUENCE_NAMES = None  # None means: scan DATA_ROOT and use every seq* folder.
NUM_VIDEOS_TO_TEST = 0
MAX_YOLO_BOXES_PER_FRAME = 1  # only 1 box per frame
YOLO_CONF = 0.5
YOLO_IMGSZ = 640
BLUR_REFERENCE = 150.0
MIN_MASK_AREA_RATIO = 0.0005
MAX_MASK_AREA_RATIO = 0.80
ALLOW_BBOX_GENERATION = True
CANDIDATE_DEVICE = "cpu"


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
        help="Root for every generated candidate, mask, log, and metric artifact.",
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
        "--candidate-device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=CANDIDATE_DEVICE,
        help="Device used by the MedSAM2 candidate subprocess.",
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
        "--threshold-ablation",
        action="store_true",
        default=RUN_THRESHOLD_ABLATION,
        help="Postprocess the shared candidates at all RELIABILITY_THRESHOLDS.",
    )
    parser.add_argument(
        "--no-generate-bboxes",
        action="store_true",
        help="Require existing bbox CSVs; do not write generated bbox files locally.",
    )
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global OUTPUT_ROOT, SEQUENCE_NAMES, CANDIDATE_DEVICE
    global PROMPT_SOURCE, RUN_THRESHOLD_ABLATION
    global ALLOW_BBOX_GENERATION, VIDEO_PROMPT_STRIDE, VIDEO_PROMPT_LIMIT

    output_root = args.output_root.expanduser().absolute()
    if args.require_google_drive_output and not is_google_drive_path(output_root):
        raise RuntimeError(
            "Refusing to run: --output-root is not inside a Google Drive "
            f"CloudStorage mount: {output_root}"
        )
    OUTPUT_ROOT = output_root
    SEQUENCE_NAMES = parse_sequence_specs(args.sequences)
    CANDIDATE_DEVICE = args.candidate_device
    PROMPT_SOURCE = args.prompt_source
    if args.video_prompt_stride < 1:
        raise ValueError("--video-prompt-stride must be at least 1")
    if args.video_prompt_limit < 0:
        raise ValueError("--video-prompt-limit cannot be negative")
    VIDEO_PROMPT_STRIDE = args.video_prompt_stride
    VIDEO_PROMPT_LIMIT = args.video_prompt_limit
    RUN_THRESHOLD_ABLATION = args.threshold_ablation
    ALLOW_BBOX_GENERATION = not args.no_generate_bboxes


def list_sequences(data_root: Path) -> list[str]:
    if SEQUENCE_NAMES:
        missing = [name for name in SEQUENCE_NAMES if not (data_root / name).is_dir()]
        if missing:
            raise RuntimeError(f"Missing requested sequence directories: {missing}")
        return list(SEQUENCE_NAMES)
    sequence_names = sorted(
        [
            path.name
            for path in data_root.iterdir()
            if path.is_dir() and path.name.startswith("seq")
        ],
        key=get_numeric_sort_key,
    )
    if NUM_VIDEOS_TO_TEST > 0:
        return sequence_names[:NUM_VIDEOS_TO_TEST]
    return sequence_names


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


def save_debug_panel(
    output_path: Path,
    frame_rgb: np.ndarray,
    current_mask: np.ndarray,
    previous_mask: np.ndarray | None,
    final_mask: np.ndarray,
    gt_mask: np.ndarray | None,
    reliability: float,
    accepted_update: bool,
) -> None:
    def mask_to_rgb(mask: np.ndarray | None, size_hw):
        h, w = size_hw
        if mask is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        binary = resize_binary_mask(mask, size_hw)
        return np.repeat((binary * 255)[:, :, None], 3, axis=2).astype(np.uint8)

    h, w = frame_rgb.shape[:2]
    panels = [
        Image.fromarray(frame_rgb.astype(np.uint8)),
        Image.fromarray(mask_to_rgb(current_mask, (h, w))),
        Image.fromarray(mask_to_rgb(final_mask, (h, w))),
        Image.fromarray(mask_to_rgb(gt_mask, (h, w))),
    ]
    canvas = Image.new("RGB", (w * 4, h + 30), color=(0, 0, 0))
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (idx * w, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_candidate_inference(sequence_names: list[str], candidate_root: Path) -> None:
    """Generate all sequence candidates in one process so models load only once."""
    candidate_logs_root = candidate_root / "logs"
    sequence_numbers = [name.removeprefix("seq") for name in sequence_names]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "external" / "MedSAM2" / "medsam2_infer_video_with_yolo.py"),
        "--base_video_dir",
        str(DATA_ROOT),
        "--seq_nums",
        *sequence_numbers,
        "--output_root",
        str(candidate_root),
        "--method_name",
        "candidate",
        "--log_dir",
        str(candidate_logs_root),
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
        VIDEO_PROMPT_SOURCE,
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
    if CANDIDATE_DEVICE != "auto":
        env["SAM2_DEVICE"] = CANDIDATE_DEVICE
    else:
        env.pop("SAM2_DEVICE", None)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def resolve_candidate_mask_path(
    candidate_root: Path, sequence_name: str, frame_name: str
) -> Path | None:
    candidates = [
        candidate_root
        / "candidate"
        / "masks"
        / sequence_name
        / "predicted"
        / f"{frame_name}.png",
        candidate_root / "masks" / sequence_name / "predicted" / f"{frame_name}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_gated_variant(
    sequence_names: list[str],
    variant_name: str,
    reliability_threshold: float,
    candidate_root: Path,
) -> None:
    gated_root = OUTPUT_ROOT / variant_name
    masks_root = gated_root / "masks"
    logs_root = gated_root / "logs"
    debug_root = OUTPUT_ROOT / "debug_visualizations" / variant_name
    reliability_config = ReliabilityConfig(
        use_blur_score=USE_BLUR_SCORE,
        blur_reference=BLUR_REFERENCE,
        min_mask_area_ratio=MIN_MASK_AREA_RATIO,
        max_mask_area_ratio=MAX_MASK_AREA_RATIO,
    )
    for index, sequence_name in enumerate(sequence_names, start=1):
        print(f"[gated] {index}/{len(sequence_names)} {sequence_name}", flush=True)
        frame_dir = Path(get_video_frame_dir(str(DATA_ROOT), sequence_name))
        frame_names = get_frame_names(str(frame_dir))
        first_frame_path = Path(resolve_frame_path(str(frame_dir), frame_names[0]))
        height, width = np.array(Image.open(first_frame_path).convert("RGB")).shape[:2]

        previous_valid_mask = None
        consecutive_rejections = 0
        rows = []
        debug_budget = NUM_DEBUG_SAMPLES

        for frame_idx, frame_name in enumerate(frame_names):
            frame_path = Path(resolve_frame_path(str(frame_dir), frame_name))
            frame_rgb = np.array(Image.open(frame_path).convert("RGB"))
            frame_bgr = frame_rgb[:, :, ::-1]
            candidate_mask_path = resolve_candidate_mask_path(
                candidate_root, sequence_name, frame_name
            )
            current_mask = (
                load_binary_mask(candidate_mask_path)
                if candidate_mask_path is not None
                else np.zeros((height, width), dtype=np.uint8)
            )
            current_mask = resize_binary_mask(current_mask, (height, width))
            previous_mask_for_debug = (
                None if previous_valid_mask is None else previous_valid_mask.copy()
            )

            signals = compute_reliability(
                current_mask=current_mask,
                previous_mask=previous_valid_mask,
                frame_bgr=frame_bgr,
                mask_confidence=0.5,
                config=reliability_config,
            )
            reliability = signals["reliability"]

            # Output-state gate: direct SAM2/MedSAM2 feature-memory tensors are
            # not modified; low-reliability candidates hold the prior output.
            final_mask, accepted_update, consecutive_rejections = (
                apply_reliability_gate(
                    current_mask=current_mask,
                    previous_valid_mask=previous_valid_mask,
                    reliability=reliability,
                    reliability_threshold=reliability_threshold,
                    consecutive_rejections=consecutive_rejections,
                    max_consecutive_rejections=MAX_CONSECUTIVE_REJECTIONS,
                )
            )

            if final_mask.sum() > 0:
                previous_valid_mask = final_mask.astype(np.uint8)

            if SAVE_MASK_IMAGES:
                output_mask_path = (
                    masks_root / sequence_name / "predicted" / f"{frame_name}.png"
                )
                save_binary_mask(final_mask, output_mask_path)
            if SAVE_NUMPY_MASKS:
                npy_dir = gated_root / "masks_numpy" / sequence_name / "predicted"
                npy_dir.mkdir(parents=True, exist_ok=True)
                np.save(npy_dir / f"{frame_name}.npy", final_mask.astype(np.uint8))

            gt_mask = None
            gt_dir = DATA_ROOT / sequence_name / "masks"
            gt_candidate = resolve_mask_path(gt_dir, frame_name)
            if gt_candidate is not None:
                gt_mask = load_binary_mask(gt_candidate)
                gt_mask = resize_binary_mask(gt_mask, (height, width))

            if SAVE_DEBUG_VISUALIZATIONS and debug_budget > 0:
                iou_value = (
                    float("nan")
                    if gt_mask is None
                    else calculate_iou(final_mask, gt_mask)
                )
                if (
                    (accepted_update is False)
                    or (not math.isnan(iou_value) and iou_value < 0.2)
                    or reliability < 0.2
                    or reliability > 0.95
                ):
                    save_debug_panel(
                        debug_root / sequence_name / f"{frame_name}.png",
                        frame_rgb=frame_rgb,
                        current_mask=current_mask,
                        previous_mask=previous_mask_for_debug,
                        final_mask=final_mask,
                        gt_mask=gt_mask,
                        reliability=reliability,
                        accepted_update=accepted_update,
                    )
                    debug_budget -= 1

            rows.append(
                {
                    "frame_idx": frame_idx,
                    "frame": frame_name,
                    "num_boxes": 1 if current_mask.sum() > 0 else 0,
                    "reliability": reliability,
                    "accepted_update": bool(accepted_update),
                    "r_conf": signals["r_conf"],
                    "r_temporal": signals["r_temporal"],
                    "r_area": signals["r_area"],
                    "r_blur": signals["r_blur"],
                    "accepted_area_frac": area_fraction(final_mask),
                }
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
                "video_prompt_source": VIDEO_PROMPT_SOURCE,
                "video_prompt_stride": VIDEO_PROMPT_STRIDE,
                "video_prompt_limit": VIDEO_PROMPT_LIMIT,
                "reliability_threshold": reliability_threshold,
                "reliability_formula": "0.35*r_conf + 0.30*r_temporal + 0.25*r_area + 0.10*r_blur with blank/tiny/huge penalties",
                "state_update": "accept current binary mask when reliability >= threshold; otherwise hold the previous non-empty mask",
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


def run_gated(sequence_names: list[str]) -> list[str]:
    candidate_root = OUTPUT_ROOT / "_candidate"
    run_candidate_inference(sequence_names, candidate_root)
    variant_names = ["gated"]
    run_gated_variant(
        sequence_names=sequence_names,
        variant_name="gated",
        reliability_threshold=RELIABILITY_THRESHOLD,
        candidate_root=candidate_root,
    )
    if RUN_THRESHOLD_ABLATION:
        for threshold in RELIABILITY_THRESHOLDS:
            variant_name = f"gated_thr_{threshold:.1f}"
            run_gated_variant(
                sequence_names=sequence_names,
                variant_name=variant_name,
                reliability_threshold=threshold,
                candidate_root=candidate_root,
            )
            variant_names.append(variant_name)
    return variant_names


def evaluate_variant(variant_name: str, sequence_names: list[str]) -> None:
    variant_root = OUTPUT_ROOT / variant_name
    eval_root = variant_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []
    summary_rows = []
    reliability_by_frame = {}
    gated_log_csv_dir = variant_root / "logs"

    for sequence_name in sequence_names:
        log_csv = gated_log_csv_dir / f"{sequence_name}.csv"
        if not log_csv.is_file():
            continue
        with open(log_csv, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                reliability_by_frame[(sequence_name, row["frame"])] = {
                    "reliability": float(row["reliability"]),
                    "accepted_update": row.get("accepted_update", "").lower() == "true",
                }

    for sequence_name in sequence_names:
        gt_dir = DATA_ROOT / sequence_name / "masks"
        frame_dir = Path(get_video_frame_dir(str(DATA_ROOT), sequence_name))
        pred_dir = variant_root / "masks" / sequence_name / "predicted"
        # Evaluate the source-frame contract, not merely the predictions that
        # happen to exist. A missing prediction is an explicit blank failure.
        frame_names = get_frame_names(str(frame_dir))
        previous_pred = None
        sequence_rows = []
        for frame_name in frame_names:
            gt_path = None
            pred_path = None
            candidate_pred = pred_dir / f"{frame_name}.png"
            if candidate_pred.exists():
                pred_path = candidate_pred
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
            log_info = reliability_by_frame.get((sequence_name, frame_name), {})
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
                "reliability": log_info.get("reliability", float("nan")),
                "accepted_update": log_info.get("accepted_update", True),
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
                "num_updates_accepted": int(
                    sum(1 for row in sequence_rows if row["accepted_update"])
                ),
                "num_updates_rejected": int(
                    sum(1 for row in sequence_rows if not row["accepted_update"])
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
            "num_updates_accepted": int(
                sum(1 for row in aggregate_rows if row["accepted_update"])
            ),
            "num_updates_rejected": int(
                sum(1 for row in aggregate_rows if not row["accepted_update"])
            ),
        }
        with open(eval_root / "global_summary.json", "w", encoding="utf-8") as handle:
            json.dump(global_summary, handle, indent=2)


def write_project_level_reports(
    variant_names: list[str], sequence_names: list[str]
) -> None:
    all_rows = []
    summary_rows = []

    for variant_name in variant_names:
        eval_root = OUTPUT_ROOT / variant_name / "eval"
        for sequence_name in sequence_names:
            seq_csv = eval_root / f"{sequence_name}.csv"
            if not seq_csv.is_file():
                continue
            with open(seq_csv, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                all_rows.extend(reader)

        global_summary = eval_root / "global_summary.json"
        if not global_summary.is_file():
            continue
        with open(global_summary, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        summary_rows.append(
            {
                "variant": data["variant"],
                "mean_dice": data["mean_dice"],
                "mean_iou": data["mean_iou"],
                "mean_temporal_iou": data["mean_temporal_iou"],
                "mean_centroid_shift": data["mean_centroid_shift"],
                "mean_area_change": data["mean_area_change"],
                "mean_reliability": data["mean_reliability"],
                "num_frames": data["num_frames"],
                "num_predictions_missing": data["num_predictions_missing"],
                "num_updates_accepted": data["num_updates_accepted"],
                "num_updates_rejected": data["num_updates_rejected"],
            }
        )

    if all_rows:
        metrics_path = OUTPUT_ROOT / "metrics.csv"
        with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sequence",
                    "variant",
                    "frame",
                    "frame_idx",
                    "prediction_missing",
                    "dice",
                    "iou",
                    "pred_area_frac",
                    "gt_area_frac",
                    "temporal_iou",
                    "centroid_shift",
                    "area_change",
                    "reliability",
                    "accepted_update",
                ],
            )
            writer.writeheader()
            writer.writerows(all_rows)

    if summary_rows:
        summary_path = OUTPUT_ROOT / "summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def main(args: argparse.Namespace | None = None) -> None:
    configure_from_args(parse_args() if args is None else args)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sequence_names = list_sequences(DATA_ROOT)
    if not sequence_names:
        raise RuntimeError(f"No seq* folders found under {DATA_ROOT}")
    ensure_gt_bboxes(sequence_names)

    print(f"Candidate device: {CANDIDATE_DEVICE}")
    print(f"Video prompt stride: {VIDEO_PROMPT_STRIDE}")
    print(f"Video prompt limit: {VIDEO_PROMPT_LIMIT}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Sequences: {sequence_names}")
    print(f"Outputs: {OUTPUT_ROOT}")

    variant_names = run_gated(sequence_names)
    for variant_name in variant_names:
        evaluate_variant(variant_name, sequence_names)
    write_project_level_reports(variant_names, sequence_names)

    notes = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": str(DATA_ROOT),
        "bbox_root": str(BBOX_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "output_is_google_drive": is_google_drive_path(OUTPUT_ROOT),
        "prompt_source": PROMPT_SOURCE,
        "video_prompt_source": VIDEO_PROMPT_SOURCE,
        "video_prompt_stride": VIDEO_PROMPT_STRIDE,
        "video_prompt_limit": VIDEO_PROMPT_LIMIT,
        "reliability_threshold": RELIABILITY_THRESHOLD,
        "reliability_thresholds": RELIABILITY_THRESHOLDS,
        "run_threshold_ablation": RUN_THRESHOLD_ABLATION,
        "num_videos_to_test": NUM_VIDEOS_TO_TEST,
        "sequence_names": sequence_names,
        "candidate_device": CANDIDATE_DEVICE,
        "sam2_cfg": SAM2_CFG,
        "sam2_checkpoint": str(SAM2_CHECKPOINT),
        "yolo_checkpoint": str(YOLO_CHECKPOINT),
        "fallback_note": "Direct internal SAM2/MedSAM2 memory tensors are not modified. Reliability gating is applied to accepted binary output masks.",
    }
    with open(OUTPUT_ROOT / "experiment_notes.json", "w", encoding="utf-8") as handle:
        json.dump(notes, handle, indent=2)

    print("Finished reliability-gated memory experiment.")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
