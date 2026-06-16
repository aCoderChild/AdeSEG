#!/usr/bin/env python3
"""Video inference for the reliability-gated dynamic memory architecture."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.eval import run_sequence_evaluation
from scripts.utils.model_defaults import (
    DEFAULT_SAM2_CFG,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CODE_ROOT,
    DEFAULT_YOLO_CHECKPOINT,
)
from src import (
    ConceptPromptEncoder,
    PromptBatch,
    QDMNScoreReliabilityEstimator,
    ReliabilityGatedDynamicMemorySAM,
)

if DEFAULT_SAM2_CODE_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_SAM2_CODE_ROOT)

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms


IMAGE_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")


def get_numeric_sort_key(name: str):
    parts = re.split(r"(\d+)", name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[-1] in IMAGE_EXTS


def list_video_names(base_video_dir: str) -> list[str]:
    image_dir = os.path.join(base_video_dir, "images")
    if os.path.isdir(image_dir) and any(is_image_file(p) for p in os.listdir(image_dir)):
        return ["."]

    video_names = [
        p
        for p in os.listdir(base_video_dir)
        if os.path.isdir(os.path.join(base_video_dir, p))
    ]
    if video_names:
        return sorted(video_names, key=get_numeric_sort_key)

    if any(is_image_file(p) for p in os.listdir(base_video_dir)):
        return ["."]
    return []


def get_video_frame_dir(base_video_dir: str, video_name: str) -> str:
    if video_name == ".":
        image_dir = os.path.join(base_video_dir, "images")
        return image_dir if os.path.isdir(image_dir) else base_video_dir
    image_dir = os.path.join(base_video_dir, video_name, "images")
    return image_dir if os.path.isdir(image_dir) else os.path.join(base_video_dir, video_name)


def get_video_name(base_video_dir: str, video_name: str) -> str:
    if video_name != ".":
        return video_name
    current = os.path.basename(os.path.abspath(base_video_dir))
    parent = os.path.basename(os.path.dirname(os.path.abspath(base_video_dir)))
    return parent if current == "images" else current


def get_frame_names(video_dir: str) -> list[str]:
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if is_image_file(p)
    ]
    return sorted(frame_names, key=get_numeric_sort_key)


def resolve_frame_path(video_dir: str, frame_name: str) -> str:
    for ext in IMAGE_EXTS:
        path = os.path.join(video_dir, f"{frame_name}{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(os.path.join(video_dir, f"{frame_name}.jpg"))


def resolve_mask_path(mask_dir: str, video_name: str, frame_name: str) -> str | None:
    candidates = [
        os.path.join(mask_dir, video_name, f"{frame_name}.png"),
        os.path.join(mask_dir, f"{frame_name}.png"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_gt_bboxes(bbox_root: str | None, video_output_name: str) -> dict[str, list[tuple[np.ndarray, float]]]:
    if bbox_root is None:
        return {}
    bbox_file = os.path.join(bbox_root, video_output_name, "bboxes.csv")
    if not os.path.isfile(bbox_file):
        return {}

    bboxes: dict[str, list[tuple[np.ndarray, float]]] = {}
    with open(bbox_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_key = row.get("frame") or row.get("frame_name") or row["frame_idx"]
            frame_key = os.path.splitext(str(frame_key))[0]
            frame_name = str(int(frame_key)) if frame_key.isdigit() else frame_key
            box = np.array(
                [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                dtype=np.float32,
            )
            bboxes.setdefault(frame_name, []).append((box, 1.0))
    return bboxes


def load_point_prompts(csv_path: str | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if csv_path is None:
        return {}

    points_by_frame: dict[str, list[list[float]]] = {}
    labels_by_frame: dict[str, list[int]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = row.get("frame") or row.get("frame_name")
            if frame is None and row.get("frame_idx") is not None:
                frame = str(int(row["frame_idx"]))
            if frame is None:
                raise ValueError("Point CSV needs a frame, frame_name, or frame_idx column.")
            frame = os.path.splitext(str(frame))[0]
            points_by_frame.setdefault(frame, []).append([float(row["x"]), float(row["y"])])
            labels_by_frame.setdefault(frame, []).append(int(row.get("label", 1)))

    return {
        frame: (
            np.asarray(points_by_frame[frame], dtype=np.float32),
            np.asarray(labels_by_frame[frame], dtype=np.int32),
        )
        for frame in points_by_frame
    }


def get_yolo_boxes(yolo_model, frame_path: str, imgsz: int, conf: float, max_boxes: int):
    image = Image.open(frame_path).convert("RGB")
    results = yolo_model.predict([image], imgsz=imgsz, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    order = np.argsort(scores)[::-1]
    if max_boxes > 0:
        order = order[:max_boxes]
    return [(xyxy[i].astype(np.float32), float(scores[i])) for i in order]


def build_prompt_batch(
    frame_name: str,
    frame_path: str,
    video_name: str,
    boxes: list[tuple[np.ndarray, float]],
    point_prompts: dict[str, tuple[np.ndarray, np.ndarray]],
    prompt_mask_dir: str | None,
    concept_prompts: list[str] | None,
    prompt_mode: str,
    transforms: SAM2Transforms,
    prompt_encoder,
    orig_hw: tuple[int, int],
    device: torch.device,
) -> PromptBatch:
    use_boxes = prompt_mode in {"boxes", "both"}
    use_prompts = prompt_mode in {"prompts", "both"}

    box_tensor = None
    if use_boxes and boxes:
        box_np = boxes[0][0][None, :]
        box_tensor = torch.as_tensor(box_np, dtype=torch.float32, device=device)
        box_tensor = transforms.transform_boxes(box_tensor, normalize=True, orig_hw=orig_hw)

    point_coords = None
    point_labels = None
    if use_prompts and frame_name in point_prompts:
        coords_np, labels_np = point_prompts[frame_name]
        point_coords = torch.as_tensor(coords_np[None, ...], dtype=torch.float32, device=device)
        point_coords = transforms.transform_coords(point_coords, normalize=True, orig_hw=orig_hw)
        point_labels = torch.as_tensor(labels_np[None, ...], dtype=torch.int32, device=device)

    mask_tensor = None
    if use_prompts and prompt_mask_dir is not None:
        mask_path = resolve_mask_path(prompt_mask_dir, video_name, frame_name)
        if mask_path is not None:
            mask_np = np.array(Image.open(mask_path).convert("L"))
            mask_np = (mask_np > (0 if mask_np.max() <= 1 else 127)).astype(np.float32)
            mask_np = mask_np * 20.0 - 10.0
            mask_tensor = torch.from_numpy(mask_np)[None, None].to(device)
            mask_tensor = F.interpolate(
                mask_tensor,
                size=prompt_encoder.mask_input_size,
                mode="bilinear",
                align_corners=False,
            )

    concepts = concept_prompts if use_prompts else None
    return PromptBatch(
        point_coords=point_coords,
        point_labels=point_labels,
        boxes=box_tensor,
        mask=mask_tensor,
        concept_text=concepts,
    )


def mask_prompt_tensor(
    mask: np.ndarray,
    prompt_encoder,
    device: torch.device,
) -> torch.Tensor:
    mask_logits = mask.astype(np.float32) * 20.0 - 10.0
    mask_tensor = torch.as_tensor(mask_logits, device=device)[None, None]
    return F.interpolate(
        mask_tensor,
        size=prompt_encoder.mask_input_size,
        mode="bilinear",
        align_corners=False,
    )


def save_binary_mask(path: str, mask: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_bbox_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "box_id", "x_center", "y_center", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_reliability_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "reliability", "iou_score", "num_boxes", "has_points", "has_mask_prompt"],
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def run_sequence(
    model: ReliabilityGatedDynamicMemorySAM,
    transforms: SAM2Transforms,
    yolo_model,
    base_video_dir: str,
    output_mask_dir: str,
    video_name: str,
    prompt_mode: str,
    bbox_prompt_source: str,
    bbox_root: str | None,
    point_prompts: dict[str, tuple[np.ndarray, np.ndarray]],
    prompt_mask_dir: str | None,
    concept_prompts: list[str] | None,
    yolo_imgsz: int,
    yolo_conf: float,
    max_boxes_per_frame: int,
    prompt_stride: int,
    prompt_limit: int,
    use_previous_mask_prompt: bool,
    score_thresh: float,
    bbox_output_dir: str | None,
    reliability_output_dir: str | None,
    log_dir: str,
) -> str:
    video_dir = get_video_frame_dir(base_video_dir, video_name)
    video_output_name = get_video_name(base_video_dir, video_name)
    mask_output_name = os.path.join(video_output_name, "predicted")
    frame_names = get_frame_names(video_dir)
    gt_boxes = (
        load_gt_bboxes(bbox_root, video_output_name)
        if bbox_prompt_source == "gt_bbox"
        else {}
    )

    device = next(model.parameters()).device
    memory = None
    bbox_rows = []
    reliability_rows = []
    log = {
        "video": video_output_name,
        "status": "started",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_mode": prompt_mode,
        "bbox_prompt_source": bbox_prompt_source,
        "prompt_stride": prompt_stride,
        "prompt_limit": prompt_limit,
        "use_previous_mask_prompt": use_previous_mask_prompt,
        "num_frames": len(frame_names),
        "frames": [],
    }

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    prompt_count = 0
    previous_mask_prompt = None
    for frame_idx, frame_name in enumerate(frame_names):
        frame_path = resolve_frame_path(video_dir, frame_name)
        image = Image.open(frame_path).convert("RGB")
        orig_hw = (image.height, image.width)
        use_prompt_this_frame = (
            prompt_stride <= 1 or frame_idx % prompt_stride == 0
        ) and (prompt_limit <= 0 or prompt_count < prompt_limit)

        if use_prompt_this_frame and prompt_mode in {"boxes", "both"}:
            if bbox_prompt_source == "gt_bbox":
                boxes = list(gt_boxes.get(frame_name, []))
                if not boxes:
                    boxes = list(gt_boxes.get(str(int(frame_name)), [])) if frame_name.isdigit() else []
            else:
                boxes = get_yolo_boxes(yolo_model, frame_path, yolo_imgsz, yolo_conf, max_boxes_per_frame)
        else:
            boxes = []

        if max_boxes_per_frame > 0:
            boxes = boxes[:max_boxes_per_frame]

        prompt_batch = build_prompt_batch(
            frame_name=frame_name,
            frame_path=frame_path,
            video_name=video_output_name,
            boxes=boxes,
            point_prompts=point_prompts,
            prompt_mask_dir=prompt_mask_dir,
            concept_prompts=concept_prompts,
            prompt_mode=prompt_mode if use_prompt_this_frame else "none",
            transforms=transforms,
            prompt_encoder=model.prompt_encoder,
            orig_hw=orig_hw,
            device=device,
        )
        has_explicit_prompt = (
            bool(boxes)
            or prompt_batch.point_coords is not None
            or prompt_batch.mask is not None
            or (prompt_batch.concept_text is not None and len(prompt_batch.concept_text) > 0)
        )
        if (
            use_previous_mask_prompt
            and not has_explicit_prompt
            and previous_mask_prompt is not None
        ):
            prompt_batch.mask = mask_prompt_tensor(
                previous_mask_prompt,
                prompt_encoder=model.prompt_encoder,
                device=device,
            )

        has_any_prompt = (
            has_explicit_prompt
            or prompt_batch.mask is not None
        )
        if use_prompt_this_frame and has_any_prompt:
            prompt_count += 1

        input_tensor = transforms(image)[None].to(device)
        with autocast_context:
            output = model(input_tensor, memory=memory, prompts=prompt_batch)
        memory = output.memory

        mask_logits = F.interpolate(
            output.mask_logits.float(),
            size=orig_hw,
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = (mask_logits[0, 0] > score_thresh).cpu().numpy().astype(np.uint8)
        save_binary_mask(
            os.path.join(output_mask_dir, mask_output_name, f"{frame_name}.png"),
            pred_mask,
        )
        previous_mask_prompt = pred_mask

        for box_id, (box, _) in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            bbox_rows.append(
                {
                    "frame": frame_name,
                    "box_id": box_id,
                    "x_center": ((x1 + x2) / 2) / image.width,
                    "y_center": ((y1 + y2) / 2) / image.height,
                    "width": (x2 - x1) / image.width,
                    "height": (y2 - y1) / image.height,
                }
            )

        reliability = float(output.reliability.detach().flatten()[0].cpu())
        iou_score = float(output.iou_scores.detach().flatten()[0].cpu())
        has_points = prompt_batch.point_coords is not None
        has_mask_prompt = prompt_batch.mask is not None
        reliability_rows.append(
            {
                "frame": frame_name,
                "reliability": reliability,
                "iou_score": iou_score,
                "num_boxes": len(boxes),
                "has_points": int(has_points),
                "has_mask_prompt": int(has_mask_prompt),
            }
        )
        log["frames"].append(
            {
                "frame": frame_name,
                "frame_idx": frame_idx,
                "prompted": bool(use_prompt_this_frame and has_any_prompt),
                "num_boxes": len(boxes),
                "reliability": reliability,
                "iou_score": iou_score,
                "has_points": has_points,
                "has_mask_prompt": has_mask_prompt,
            }
        )

    if bbox_output_dir is not None:
        write_bbox_csv(os.path.join(bbox_output_dir, f"{video_output_name}.csv"), bbox_rows)
    if reliability_output_dir is not None:
        write_reliability_csv(
            os.path.join(reliability_output_dir, f"{video_output_name}.csv"),
            reliability_rows,
        )

    log["status"] = "success"
    log["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(os.path.join(log_dir, f"{video_output_name}.json"), log)
    return video_output_name


def build_model(args) -> tuple[ReliabilityGatedDynamicMemorySAM, SAM2Transforms]:
    sam2_model = build_sam2(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
    )

    concept_encoder = ConceptPromptEncoder(embed_dim=sam2_model.hidden_dim)
    reliability_estimator = None
    if args.reliability_estimator == "qdmn":
        reliability_estimator = QDMNScoreReliabilityEstimator(
            feature_dim=sam2_model.hidden_dim,
            qdmn_root=args.qdmn_root,
            checkpoint_path=args.qdmn_checkpoint,
        )

    model = ReliabilityGatedDynamicMemorySAM.from_sam2_base(
        sam2_model,
        concept_encoder=concept_encoder,
        reliability_estimator=reliability_estimator,
        memory_attention_strength=args.memory_attention_strength,
    )
    model = model.to(sam2_model.device)
    if args.adapter_checkpoint is not None:
        load_adapter_checkpoint(model, args.adapter_checkpoint)
    model.eval()
    transforms = SAM2Transforms(
        resolution=sam2_model.image_size,
        mask_threshold=args.score_thresh,
    )
    return model, transforms


def load_adapter_checkpoint(model: ReliabilityGatedDynamicMemorySAM, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    if "prompt_memory_attention" in checkpoint:
        model.prompt_memory_attention.load_state_dict(
            checkpoint["prompt_memory_attention"],
            strict=False,
        )
    if "reliability_estimator" in checkpoint:
        model.reliability_estimator.load_state_dict(
            checkpoint["reliability_estimator"],
            strict=False,
        )
    if "reliability_project" in checkpoint:
        model.reliability_estimator.project.load_state_dict(
            checkpoint["reliability_project"],
            strict=False,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_cfg", default=DEFAULT_SAM2_CFG)
    parser.add_argument("--sam2_checkpoint", default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("-i", "--base_video_dir", required=True)
    parser.add_argument("--video_list_file", default=None)
    parser.add_argument("--seq_nums", type=int, nargs="*", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--method_name", default="ReliabilityGatedSAM")
    parser.add_argument("-o", "--output_mask_dir", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--run_eval", action="store_true")

    parser.add_argument(
        "--prompt_mode",
        choices=["boxes", "prompts", "both"],
        default="both",
        help="boxes uses bbox prompts, prompts uses point/mask/concept prompts, both combines them.",
    )
    parser.add_argument(
        "--bbox_prompt_source",
        choices=["yolo", "gt_bbox"],
        default="yolo",
        help="Source for box prompts when --prompt_mode includes boxes.",
    )
    parser.add_argument(
        "--box_source",
        choices=["yolo", "gt_bbox"],
        default=None,
        help="Deprecated alias for --bbox_prompt_source.",
    )
    parser.add_argument("--bbox_root", default="data/bbox")
    parser.add_argument("--yolo_checkpoint", default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--yolo_conf", type=float, default=0.5)
    parser.add_argument("--max_boxes_per_frame", type=int, default=1)
    parser.add_argument(
        "--prompt_stride",
        type=int,
        default=1,
        help="Inject prompts every N frames; memory is used on frames between prompts.",
    )
    parser.add_argument(
        "--prompt_limit",
        type=int,
        default=0,
        help="Maximum number of prompted frames per sequence; 0 means no limit.",
    )
    parser.add_argument(
        "--no_previous_mask_prompt",
        action="store_true",
        help="Disable using the previous predicted mask as a mask prompt on frames without explicit prompts.",
    )

    parser.add_argument(
        "--point_prompt_csv",
        default=None,
        help="Optional CSV with frame/frame_idx,x,y,label columns.",
    )
    parser.add_argument(
        "--prompt_mask_dir",
        default=None,
        help="Optional mask prompt directory. Supports <dir>/<seq>/<frame>.png or <dir>/<frame>.png.",
    )
    parser.add_argument(
        "--concept_prompts",
        nargs="*",
        default=None,
        help="Optional conceptual text prompts, e.g. --concept_prompts adenoid tissue.",
    )

    parser.add_argument(
        "--reliability_estimator",
        choices=["adaptive", "qdmn"],
        default="qdmn",
    )
    parser.add_argument("--qdmn_root", default="external/QDMN")
    parser.add_argument(
        "--qdmn_checkpoint",
        default="checkpoints/QDMN.pth",
        help="Optional QDMN checkpoint. The score.* weights are loaded for reliability.",
    )
    parser.add_argument(
        "--adapter_checkpoint",
        default=None,
        help="Optional fine-tuned memory/reliability adapter checkpoint.",
    )
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument(
        "--memory_attention_strength",
        type=float,
        default=0.0,
        help="Blend strength for experimental previous-memory attention. 0 disables it.",
    )
    parser.add_argument("--apply_postprocessing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.box_source is not None:
        args.bbox_prompt_source = args.box_source
    if args.prompt_stride < 1:
        raise ValueError("--prompt_stride must be >= 1")
    if args.output_mask_dir is None:
        args.output_mask_dir = os.path.join(args.output_root, args.method_name, "masks")
    if args.log_dir is None:
        args.log_dir = os.path.join(args.output_root, args.method_name, "logs")

    if args.prompt_mode in {"boxes", "both"} and args.bbox_prompt_source == "yolo":
        from ultralytics import YOLO

        yolo_model = YOLO(args.yolo_checkpoint)
    else:
        yolo_model = None

    model, transforms = build_model(args)
    point_prompts = load_point_prompts(args.point_prompt_csv)

    if args.seq_nums:
        video_names = [f"seq{seq_num}" for seq_num in args.seq_nums]
    elif args.video_list_file is not None:
        with open(args.video_list_file, "r", encoding="utf-8") as f:
            video_names = [line.strip() for line in f if line.strip()]
    else:
        video_names = list_video_names(args.base_video_dir)

    bbox_output_dir = (
        os.path.join(args.output_root, args.method_name, "bbox")
        if args.prompt_mode in {"boxes", "both"} and args.bbox_prompt_source == "yolo"
        else None
    )
    reliability_output_dir = os.path.join(args.output_root, args.method_name, "reliability")

    print(f"running reliability-gated inference on {len(video_names)} videos: {video_names}")
    print(
        f"prompt_mode={args.prompt_mode}, "
        f"bbox_prompt_source={args.bbox_prompt_source}, "
        f"prompt_stride={args.prompt_stride}, prompt_limit={args.prompt_limit}"
    )

    completed = []
    for idx, video_name in enumerate(video_names):
        display_name = get_video_name(args.base_video_dir, video_name)
        print(f"\n{idx + 1}/{len(video_names)} - running on {display_name}")
        try:
            completed_name = run_sequence(
                model=model,
                transforms=transforms,
                yolo_model=yolo_model,
                base_video_dir=args.base_video_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                prompt_mode=args.prompt_mode,
                bbox_prompt_source=args.bbox_prompt_source,
                bbox_root=args.bbox_root,
                point_prompts=point_prompts,
                prompt_mask_dir=args.prompt_mask_dir,
                concept_prompts=args.concept_prompts,
                yolo_imgsz=args.yolo_imgsz,
                yolo_conf=args.yolo_conf,
                max_boxes_per_frame=args.max_boxes_per_frame,
                prompt_stride=args.prompt_stride,
                prompt_limit=args.prompt_limit,
                use_previous_mask_prompt=not args.no_previous_mask_prompt,
                score_thresh=args.score_thresh,
                bbox_output_dir=bbox_output_dir,
                reliability_output_dir=reliability_output_dir,
                log_dir=args.log_dir,
            )
            completed.append(completed_name)
        except Exception as exc:
            log = {
                "video": display_name,
                "status": "failed",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            write_json(os.path.join(args.log_dir, f"{display_name}.json"), log)
            print(f"Warning: failed {display_name}: {exc}")

    print(f"\ncompleted {len(completed)}/{len(video_names)} videos")
    print(f"masks saved to {args.output_mask_dir}")

    if args.run_eval:
        eval_prompt_source = (
            "yolo"
            if args.prompt_mode in {"boxes", "both"} and args.bbox_prompt_source == "yolo"
            else "gt_bbox"
        )
        run_sequence_evaluation(
            sequence_names=completed,
            data_root=args.base_video_dir,
            bbox_root=args.bbox_root,
            output_root=args.output_root,
            method_name=args.method_name,
            prompt_source=eval_prompt_source,
            verbose=True,
        )


if __name__ == "__main__":
    main()
