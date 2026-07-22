#!/usr/bin/env python3
"""Reliability-gated SAM2 memory experiment -- single dynamic slot."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDSAM2_ROOT = PROJECT_ROOT / "external" / "MedSAM2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MEDSAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDSAM2_ROOT))

from sam2.build_sam import build_sam2_video_predictor

from scripts.utils.eval_metrics import (
    calculate_dice,
    calculate_fmeasure,
    calculate_iou,
    calculate_precision_recall,
    calculate_temporal_iou,
    get_bbox_from_mask,
    get_numeric_sort_key,
)
from scripts.utils.mask_utils import (
    load_binary_mask,
    make_overlay,
    resize_binary_mask,
    resolve_mask_path,
    save_binary_mask,
    save_overlay,
)
from scripts.utils.reliability_gate import (
    ReliabilityConfig,
    area_change,
    area_fraction,
    centroid_shift,
    compute_reliability,
    safe_nanmean,
)

from external.MedSAM2.medsam2_infer_video_with_yolo import (
    get_frame_names,
    get_video_frame_dir,
    get_yolo_boxes,
    load_gt_bboxes,
    resolve_frame_path,
    write_sequence_log,
)

DATA_ROOT = PROJECT_ROOT / "data" / "test" / "polypgen"
BBOX_ROOT = PROJECT_ROOT / "data" / "test" / "bbox"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "reliability_gated_video_memory"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
SAM2_CFG = "configs/sam2.1_hiera_t512.yaml"
SAM2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "MedSAM2_latest.pt"
YOLO_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "polypgen_yolov8n.pt"

PROMPT_SOURCE = "yolo"
VIDEO_PROMPT_STRIDE = 1
VIDEO_PROMPT_LIMIT = 0

USE_BLUR_SCORE = True
SAVE_MASK_IMAGES = True

SEQUENCE_NAMES = None
MAX_YOLO_BOXES_PER_FRAME = 1
YOLO_CONF = 0.5
YOLO_IMGSZ = 640
ALLOW_BBOX_GENERATION = True
DEVICE = "mps"


def is_google_drive_path(path: Path) -> bool:
    """Return whether a path is inside the macOS Google Drive CloudStorage mount."""
    parts = path.expanduser().absolute().parts
    return "CloudStorage" in parts and any(
        part.startswith("GoogleDrive-") for part in parts
    )


def parse_seqs(specs: list[str] | None) -> list[str] | None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reliability-gated SAM2 real-memory-bank inference and evaluation."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("ADESEG_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)),
    )
    parser.add_argument(
        "--require-google-drive-output",
        action="store_true",
        help="Fail unless --output-root is inside a GoogleDrive-* CloudStorage mount.",
    )
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=DEVICE)
    parser.add_argument("--prompt-source", choices=["gt_bbox", "yolo"], default=PROMPT_SOURCE)
    parser.add_argument("--video-prompt-stride", type=int, default=VIDEO_PROMPT_STRIDE)
    parser.add_argument("--video-prompt-limit", type=int, default=VIDEO_PROMPT_LIMIT)
    parser.add_argument("--no-generate-bboxes", action="store_true")
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global OUTPUT_ROOT, SEQUENCE_NAMES, DEVICE, PROMPT_SOURCE
    global ALLOW_BBOX_GENERATION, VIDEO_PROMPT_STRIDE, VIDEO_PROMPT_LIMIT

    output_root = args.output_root.expanduser().absolute()
    if args.require_google_drive_output and not is_google_drive_path(output_root):
        raise RuntimeError(
            "Refusing to run: --output-root is not inside a Google Drive "
            f"CloudStorage mount: {output_root}"
        )
    OUTPUT_ROOT = output_root
    SEQUENCE_NAMES = parse_seqs(args.sequences)
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
        bbox_csv = BBOX_ROOT / sequence_name / "bboxes.csv"
        if bbox_csv.is_file():
            continue
        if not ALLOW_BBOX_GENERATION:
            raise RuntimeError(f"Missing {bbox_csv}; --no-generate-bboxes forbids creating it.")
        masks_dir = DATA_ROOT / sequence_name / "masks"
        bbox_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for frame_name in get_frame_names(str(masks_dir)):
            mask_path = resolve_mask_path(masks_dir, frame_name)
            if mask_path is None:
                continue
            bbox = get_bbox_from_mask(load_binary_mask(mask_path))
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            rows.append({"frame_idx": frame_name, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        with open(bbox_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame_idx", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerows(rows)


def resolve_device() -> str | None:
    return None if DEVICE == "auto" else DEVICE


def gt_bbox_key(frame_name: str) -> str:
    return str(int(frame_name)) if frame_name.isdigit() else frame_name


def pick_box(
    frame_idx: int, frame_name: str, frame_path: Path, gt_bboxes: dict, yolo_model, seeded_count: int
) -> tuple[np.ndarray | None, float | None]:
    if VIDEO_PROMPT_LIMIT > 0 and seeded_count >= VIDEO_PROMPT_LIMIT:
        return None, None
    if VIDEO_PROMPT_STRIDE > 1 and frame_idx % VIDEO_PROMPT_STRIDE != 0:
        return None, None
    if PROMPT_SOURCE == "gt_bbox":
        boxes = gt_bboxes.get(gt_bbox_key(frame_name), [])
    else:
        boxes = get_yolo_boxes(
            yolo_model=yolo_model,
            frame_path=str(frame_path),
            yolo_imgsz=YOLO_IMGSZ,
            yolo_conf=YOLO_CONF,
            max_boxes=MAX_YOLO_BOXES_PER_FRAME,
        )
    if not boxes:
        return None, None
    box, confidence = boxes[0]
    return box, float(confidence)


def decode(
    predictor, inference_state, output_dict, frame_idx, batch_size, pointer_token=None,
    is_init_cond_frame=False,
):
    original_forward_sam_heads = predictor._forward_sam_heads
    original_decoder_forward = predictor.sam_mask_decoder.forward
    captured: dict = {}

    def capturing_forward_sam_heads(*args, **kwargs):
        result = original_forward_sam_heads(*args, **kwargs)
        captured["ious"] = result[2]
        return result

    def capturing_decoder_forward(*args, **kwargs):
        if pointer_token is not None:
            kwargs["sparse_prompt_embeddings"] = torch.cat(
                [kwargs["sparse_prompt_embeddings"], pointer_token], dim=1
            )
        return original_decoder_forward(*args, **kwargs)

    predictor._forward_sam_heads = capturing_forward_sam_heads
    predictor.sam_mask_decoder.forward = capturing_decoder_forward
    try:
        current_out, pred_masks_gpu = predictor._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=output_dict,
            frame_idx=frame_idx,
            batch_size=batch_size,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=True,
        )
    finally:
        predictor._forward_sam_heads = original_forward_sam_heads
        predictor.sam_mask_decoder.forward = original_decoder_forward

    mask_confidence = float(captured["ious"].mean().item())
    return current_out, pred_masks_gpu, mask_confidence

# single dynamic slot: exactly one stored frame, always at cond_frame_idx.
def prune_memory(inference_state: dict, cond_frame_idx: int) -> None:
    output_dict = inference_state["output_dict"]
    consolidated = inference_state["consolidated_frame_inds"]

    cond_storage = output_dict["cond_frame_outputs"]
    for stale_idx in [idx for idx in cond_storage if idx != cond_frame_idx]:
        del cond_storage[stale_idx]
    kept_cond = {cond_frame_idx} if cond_frame_idx is not None else set()
    consolidated["cond_frame_outputs"].intersection_update(kept_cond)

    output_dict["non_cond_frame_outputs"].clear()
    consolidated["non_cond_frame_outputs"].clear()

    for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
        for stale_idx in [idx for idx in point_inputs_per_frame if idx not in kept_cond]:
            del point_inputs_per_frame[stale_idx]
    for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
        for stale_idx in [idx for idx in mask_inputs_per_frame if idx not in kept_cond]:
            del mask_inputs_per_frame[stale_idx]


def infer_frame(
    predictor,
    inference_state,
    frame_idx: int,
    frame_rgb: np.ndarray,
    box: np.ndarray | None,
    running_memory_features,
    pointer,
    batch_size: int,
    config: ReliabilityConfig,
    memory_frame_idx: int | None,
) -> dict:
    output_dict = inference_state["output_dict"]

    if box is not None:
        predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=frame_idx, obj_id=1, box=box
        )
        predictor.propagate_in_video_preflight(inference_state)
        batch_size = predictor._get_obj_num(inference_state)

        current_out = output_dict["cond_frame_outputs"][frame_idx]
        running_memory_features = current_out["maskmem_features"].float()
        pointer = current_out["obj_ptr"].float()
        _, video_res_masks = predictor._get_orig_video_res_output(
            inference_state, current_out["pred_masks"]
        )
        final_mask = (video_res_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
        return {
            "mask": final_mask,
            "reliability": float("nan"),
            "signals": None,
            "running_memory_features": running_memory_features,
            "pointer": pointer,
            "batch_size": batch_size,
            "memory_frame_idx": frame_idx,
        }

    # No box this frame. If the single memory slot doesn't exist yet, this is
    # genuinely the first frame with no detection anywhere so far -- decode it
    # with NO prompt at all (is_init_cond_frame=True, no box/points), so we
    # never inject a box's false-positive "there is definitely something
    # here" evidence into a frame that may be a true negative. SAM2's own
    # object-presence head (pred_obj_scores) then gets an unbiased read: if
    # it says no object, low_res_multimasks is already driven to NO_OBJ_SCORE
    # by _forward_sam_heads, so the mask comes back genuinely empty.
    is_first_ever_frame = memory_frame_idx is None
    has_evidence = not is_first_ever_frame

    frame_bgr = frame_rgb[:, :, ::-1]
    pointer_token = None if pointer is None else pointer.unsqueeze(1)
    current_out, pred_masks_gpu, mask_confidence = decode(
        predictor, inference_state, output_dict, frame_idx, batch_size, pointer_token,
        is_init_cond_frame=is_first_ever_frame,
    )
    object_score = float(torch.sigmoid(current_out["object_score_logits"]).mean().item())
    _, video_res_masks = predictor._get_orig_video_res_output(inference_state, pred_masks_gpu)
    current_mask = (video_res_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)

    previous_binary_mask = None
    if not is_first_ever_frame:
        # The single dynamic slot still holds *last* frame's blended state
        # here -- this frame's write into it happens further down.
        previous_out = output_dict["cond_frame_outputs"].get(memory_frame_idx)
        if previous_out is not None:
            _, prev_res = predictor._get_orig_video_res_output(
                inference_state, previous_out["pred_masks"]
            )
            previous_binary_mask = (prev_res[0, 0] > 0).cpu().numpy().astype(np.uint8)

    signals = compute_reliability(
        current_mask=current_mask,
        previous_mask=previous_binary_mask,
        frame_bgr=frame_bgr,
        mask_confidence=mask_confidence,
        prompt_confidence=None,
        config=config,
        object_score=object_score,
        has_evidence=has_evidence,
    )
    reliability = signals["reliability"]

    features = current_out["maskmem_features"]
    if running_memory_features is None:
        blended = features
    else:
        blended = (
            reliability * features.float() + (1.0 - reliability) * running_memory_features
        ).to(features.dtype)
    running_memory_features = blended.float()
    current_out["maskmem_features"] = blended # memory M_t

    pointer_confidence = signals["r_conf"]
    current_pointer = current_out["obj_ptr"].float()
    previous_pointer = torch.zeros_like(current_pointer) if pointer is None else pointer
    pointer = pointer_confidence * current_pointer + (1 - pointer_confidence) * previous_pointer # object-pointer
    current_out["obj_ptr"] = pointer.to(current_out["obj_ptr"].dtype)

    new_memory_frame_idx = frame_idx if is_first_ever_frame else memory_frame_idx
    output_dict["cond_frame_outputs"][new_memory_frame_idx] = current_out

    return {
        "mask": current_mask,
        "reliability": reliability,
        "signals": signals,
        "running_memory_features": running_memory_features,
        "pointer": pointer,
        "batch_size": batch_size,
        "memory_frame_idx": new_memory_frame_idx,
    }


def run(predictor, sequence_names: list[str]) -> None:
    masks_root = OUTPUT_ROOT / "masks"
    overlays_root = OUTPUT_ROOT / "overlays"
    logs_root = OUTPUT_ROOT / "logs"
    reliability_config = ReliabilityConfig(use_blur_score=USE_BLUR_SCORE)

    yolo_model = None
    if PROMPT_SOURCE == "yolo":
        if YOLO is None:
            raise RuntimeError("PROMPT_SOURCE='yolo' requires the 'ultralytics' package.")
        yolo_model = YOLO(str(YOLO_CHECKPOINT))

    for index, sequence_name in enumerate(sequence_names, start=1):
        print(f"{index}/{len(sequence_names)} {sequence_name}", flush=True)
        frame_dir = Path(get_video_frame_dir(str(DATA_ROOT), sequence_name))
        frame_names = get_frame_names(str(frame_dir))
        if not frame_names:
            print(f"{sequence_name}: no frames found, skipping.", flush=True)
            continue
        gt_bboxes = (
            load_gt_bboxes(str(BBOX_ROOT), sequence_name) if PROMPT_SOURCE == "gt_bbox" else {}
        )
        gt_mask_dir = DATA_ROOT / sequence_name / "masks"

        inference_state = predictor.init_state(video_path=str(frame_dir))
        height = inference_state["video_height"]
        width = inference_state["video_width"]

        seeded_count = 0
        last_cond_frame_idx = None
        batch_size = 1
        running_memory_features = None
        pointer = None
        rows = []
        unprompted_count = 0

        for frame_idx, frame_name in enumerate(frame_names):
            frame_path = Path(resolve_frame_path(str(frame_dir), frame_name))
            frame_rgb = np.array(Image.open(frame_path).convert("RGB"))

            box, _ = pick_box(
                frame_idx, frame_name, frame_path, gt_bboxes, yolo_model, seeded_count
            )
            if box is not None:
                seeded_count += 1
            else:
                unprompted_count += 1
                if last_cond_frame_idx is None:
                    print(
                        f"{sequence_name}: no box prompt by frame {frame_name} yet; "
                        "decoding with no prompt at all (letting SAM2's own "
                        "object-presence head decide, unbiased by a fake box).",
                        flush=True,
                    )

            out = infer_frame(
                predictor, inference_state, frame_idx, frame_rgb, box,
                running_memory_features, pointer, batch_size, reliability_config,
                last_cond_frame_idx,
            )
            running_memory_features = out["running_memory_features"]
            pointer = out["pointer"]
            batch_size = out["batch_size"]
            final_mask = out["mask"]
            reliability = out["reliability"]
            signals = out["signals"]
            last_cond_frame_idx = out["memory_frame_idx"]

            prune_memory(inference_state, last_cond_frame_idx)

            real_box_found = box is not None

            if SAVE_MASK_IMAGES:
                output_mask_path = masks_root / sequence_name / "predicted" / f"{frame_name}.png"
                save_binary_mask(final_mask, output_mask_path)
                gt_path = resolve_mask_path(gt_mask_dir, frame_name)
                gt_mask = (
                    load_binary_mask(gt_path) if gt_path is not None else np.zeros_like(final_mask)
                )
                if gt_mask.shape != final_mask.shape:
                    gt_mask = resize_binary_mask(gt_mask, final_mask.shape)
                overlay = make_overlay(frame_rgb, gt_mask, final_mask)
                save_overlay(overlay, overlays_root / sequence_name / f"{frame_name}.png")

            rows.append(
                {
                    "frame_idx": frame_idx,
                    "frame": frame_name,
                    "num_boxes": 1 if real_box_found else 0,
                    "is_promptless_anchor": not real_box_found and out["memory_frame_idx"] == frame_idx,
                    "reliability": reliability,
                    "r_conf": signals["r_conf"] if signals else float("nan"),
                    "r_boundary": signals["r_boundary"] if signals else float("nan"),
                    "r_blur": signals["r_blur"] if signals else float("nan"),
                    "r_object": signals["r_object"] if signals else float("nan"),
                    "predicted_area_frac": area_fraction(final_mask),
                }
            )

        if unprompted_count > len(frame_names) / 2:
            print(
                f"{sequence_name}: {unprompted_count}/{len(frame_names)} frames had no box prompt.",
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
                "video_prompt_stride": VIDEO_PROMPT_STRIDE,
                "video_prompt_limit": VIDEO_PROMPT_LIMIT,
                "memory_mode": "single_dynamic_slot",
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


MEAN_METRIC_FIELDS = [
    "dice", "iou", "precision", "recall", "f2",
    "temporal_iou", "centroid_shift", "area_change", "reliability",
]


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "num_frames": len(rows),
        "num_predictions_missing": int(sum(1 for row in rows if row["prediction_missing"])),
        **{
            f"mean_{field}": safe_nanmean(row[field] for row in rows)
            for field in MEAN_METRIC_FIELDS
        },
    }


def evaluate(sequence_names: list[str], masks_root: Path, logs_root: Path | None) -> None:
    eval_root = OUTPUT_ROOT / "eval"
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
                for row in csv.DictReader(handle):
                    reliability_by_frame[(sequence_name, row["frame"])] = float(row["reliability"])

    for sequence_name in sequence_names:
        gt_dir = DATA_ROOT / sequence_name / "masks"
        frame_dir = Path(get_video_frame_dir(str(DATA_ROOT), sequence_name))
        pred_dir = masks_root / sequence_name / "predicted"
        frame_names = get_frame_names(str(frame_dir))
        previous_pred = None
        sequence_rows = []
        for frame_name in frame_names:
            predicted_path = pred_dir / f"{frame_name}.png"
            pred_path = predicted_path if predicted_path.exists() else None
            gt_path = resolve_mask_path(gt_dir, frame_name) if gt_dir.is_dir() else None
            gt_mask = load_binary_mask(gt_path) if gt_path is not None else None
            prediction_missing = pred_path is None
            if pred_path is not None:
                pred_mask = load_binary_mask(pred_path)
            elif gt_mask is not None:
                pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
            else:
                frame_path = Path(resolve_frame_path(str(frame_dir), frame_name))
                with Image.open(frame_path) as frame_image:
                    pred_mask = np.zeros((frame_image.height, frame_image.width), dtype=np.uint8)
            if gt_mask is not None and gt_mask.shape != pred_mask.shape:
                gt_mask = resize_binary_mask(gt_mask, pred_mask.shape)
            if gt_mask is None:
                precision = recall = f2 = float("nan")
            else:
                precision, recall = calculate_precision_recall(pred_mask, gt_mask)
                f2 = calculate_fmeasure(pred_mask, gt_mask, beta=2)
            row = {
                "sequence": sequence_name,
                "frame": frame_name,
                "frame_idx": int(frame_name) if str(frame_name).isdigit() else frame_name,
                "prediction_missing": prediction_missing,
                "dice": float("nan") if gt_mask is None else calculate_dice(pred_mask, gt_mask),
                "iou": float("nan") if gt_mask is None else calculate_iou(pred_mask, gt_mask),
                "precision": precision,
                "recall": recall,
                "f2": f2,
                "pred_area_frac": area_fraction(pred_mask),
                "gt_area_frac": float("nan") if gt_mask is None else area_fraction(gt_mask),
                "temporal_iou": float("nan") if previous_pred is None else calculate_temporal_iou(pred_mask, previous_pred),
                "centroid_shift": float("nan") if previous_pred is None else centroid_shift(pred_mask, previous_pred),
                "area_change": float("nan") if previous_pred is None else area_change(pred_mask, previous_pred),
                "reliability": reliability_by_frame.get((sequence_name, frame_name), float("nan")),
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
        summary_rows.append({"video_id": sequence_name, **summarize_rows(sequence_rows)})

    if summary_rows:
        with open(eval_root / "summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if aggregate_rows:
        global_summary = {
            "num_sequences": len(summary_rows),
            **summarize_rows(aggregate_rows),
        }
        with open(eval_root / "global_summary.json", "w", encoding="utf-8") as handle:
            json.dump(global_summary, handle, indent=2)


def main(args: argparse.Namespace | None = None) -> None:
    apply_args(parse_args() if args is None else args)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sequence_names = list_sequences(DATA_ROOT)
    if not sequence_names:
        raise RuntimeError(f"No seq* folders found under {DATA_ROOT}")
    ensure_gt_bboxes(sequence_names)

    print(f"Inference device: {DEVICE}")
    print(f"Video prompt stride: {VIDEO_PROMPT_STRIDE}")
    print(f"Sequences: {sequence_names}")
    print(f"Outputs: {OUTPUT_ROOT}")

    predictor = build_sam2_video_predictor(
        config_file=SAM2_CFG,
        ckpt_path=str(SAM2_CHECKPOINT),
        device=resolve_device(),
        apply_postprocessing=False,
    )

    run(predictor, sequence_names)
    evaluate(sequence_names, masks_root=OUTPUT_ROOT / "masks", logs_root=OUTPUT_ROOT / "logs")

    notes = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "prompt_source": PROMPT_SOURCE,
        "video_prompt_stride": VIDEO_PROMPT_STRIDE,
        "video_prompt_limit": VIDEO_PROMPT_LIMIT,
        "sequence_names": sequence_names,
        "device": DEVICE,
        "sam2_cfg": SAM2_CFG,
        "memory_mode": "single_dynamic_slot",
    }
    with open(OUTPUT_ROOT / "experiment_notes.json", "w", encoding="utf-8") as handle:
        json.dump(notes, handle, indent=2)

    print("Finished reliability-gated video memory experiment.")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
