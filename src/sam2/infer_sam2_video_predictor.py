"""
Box-prompted SAM2 video inference and evaluation.

This script uses src/sam2/sam2_video_predictor.py through
build_sam2_video_predictor. It supports two prompt sources:

1. gt_box: boxes computed from ground-truth masks, or optionally loaded from CSV.
2. yolo: boxes predicted by a YOLOv8 model.

For each prompt stride, the script adds box prompts on frames:

    0, stride, 2 * stride, ...

Then SAM2 propagates through the whole sequence and masks are evaluated against
seq*/masks.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class FrameMetric:
    sequence: str
    frame_name: str
    stride: int
    prompt_source: str
    is_prompt_frame: bool
    dice: float
    iou: float


def natural_key(value: str | Path):
    name = Path(value).stem
    parts = re.split(r"(\d+)", name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def sequence_sort_key(path: Path):
    match = re.fullmatch(r"seq(\d+)", path.name)
    return int(match.group(1)) if match else path.name


def list_sequences(data_root: Path, seq_nums: list[int] | None):
    if seq_nums:
        sequences = [data_root / f"seq{seq_num}" for seq_num in seq_nums]
    else:
        sequences = [
            path
            for path in data_root.iterdir()
            if path.is_dir() and (path / "images").is_dir() and (path / "masks").is_dir()
        ]
        sequences = sorted(sequences, key=sequence_sort_key)

    missing = [str(path) for path in sequences if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing sequence directories: {missing}")
    return sequences


def list_frame_names(images_dir: Path) -> list[str]:
    # SAM2's frame loader sorts names lexicographically, so keep the same order.
    names = [path.name for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS]
    names.sort()
    if not names:
        raise RuntimeError(f"No image frames found in {images_dir}")
    return names


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    return np.asarray(mask) > 0


def mask_to_box(mask: np.ndarray, padding: int = 0):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    h, w = mask.shape[:2]
    x1 = max(float(xs.min() - padding), 0.0)
    y1 = max(float(ys.min() - padding), 0.0)
    x2 = min(float(xs.max() + 1 + padding), float(w))
    y2 = min(float(ys.max() + 1 + padding), float(h))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def load_gt_boxes_from_csv(bbox_root: Path, seq_name: str):
    csv_path = bbox_root / seq_name / "bboxes.csv"
    boxes = {}
    if not csv_path.exists():
        return boxes
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = row.get("frame_idx") or row.get("frame") or row.get("frame_name")
            if frame_idx is None:
                continue
            frame_key = str(frame_idx)
            try:
                if frame_key.replace(".", "", 1).isdigit():
                    frame_key = str(int(float(frame_key)))
                box = np.array(
                    [
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                    ],
                    dtype=np.float32,
                )
            except (KeyError, ValueError):
                continue
            boxes.setdefault(frame_key, []).append(box)
    return boxes


def get_gt_box(
    seq_dir: Path,
    frame_name: str,
    gt_box_source: str,
    bbox_root: Path | None,
    csv_boxes: dict[str, list[np.ndarray]],
    box_padding: int,
):
    if gt_box_source == "csv":
        stem = Path(frame_name).stem
        boxes = csv_boxes.get(stem, [])
        return boxes[0] if boxes else None

    mask = load_binary_mask(seq_dir / "masks" / frame_name)
    return mask_to_box(mask, padding=box_padding)


def get_yolo_box(yolo_model, image_path: Path, yolo_imgsz: int, yolo_conf: float):
    image = Image.open(image_path).convert("RGB")
    results = yolo_model.predict(
        [image],
        imgsz=yolo_imgsz,
        conf=yolo_conf,
        verbose=False,
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None
    boxes_np = boxes.xyxy.detach().cpu().numpy()
    conf_np = boxes.conf.detach().cpu().numpy()
    best_idx = int(np.argmax(conf_np))
    return boxes_np[best_idx].astype(np.float32), float(conf_np[best_idx])


def save_mask(mask_logits: torch.Tensor, output_path: Path, threshold: float):
    mask = (mask_logits > threshold).detach().cpu().numpy()
    mask = np.squeeze(mask).astype(np.uint8) * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)
    return mask > 0


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def write_metrics_csv(path: Path, rows: list[FrameMetric]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "frame_name",
                "stride",
                "prompt_source",
                "is_prompt_frame",
                "dice",
                "iou",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_summary_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["prompt_source", "stride", "sequence", "num_frames", "mean_dice", "mean_iou"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_saved_masks(
    seq_dir: Path,
    frame_names: list[str],
    masks_dir: Path,
    stride: int,
    prompt_source: str,
):
    prompt_indices = set(range(0, len(frame_names), stride))
    prompt_indices.add(0)
    rows = []
    for frame_idx, frame_name in enumerate(frame_names):
        pred_path = masks_dir / f"{Path(frame_name).stem}.png"
        if not pred_path.exists():
            continue
        pred = load_binary_mask(pred_path)
        gt = load_binary_mask(seq_dir / "masks" / frame_name)
        rows.append(
            FrameMetric(
                sequence=seq_dir.name,
                frame_name=frame_name,
                stride=stride,
                prompt_source=prompt_source,
                is_prompt_frame=frame_idx in prompt_indices,
                dice=dice_score(pred, gt),
                iou=iou_score(pred, gt),
            )
        )
    return rows


def prepare_prompt_boxes(
    seq_dir: Path,
    frame_names: list[str],
    stride: int,
    prompt_source: str,
    gt_box_source: str,
    bbox_root: Path | None,
    box_padding: int,
    yolo_model,
    yolo_imgsz: int,
    yolo_conf: float,
    prompt_csv_path: Path,
):
    prompt_indices = set(range(0, len(frame_names), stride))
    prompt_indices.add(0)
    csv_boxes = (
        load_gt_boxes_from_csv(bbox_root, seq_dir.name)
        if prompt_source == "gt_box" and gt_box_source == "csv" and bbox_root is not None
        else {}
    )

    prompt_boxes = {}
    yolo_rows = []
    for frame_idx in sorted(prompt_indices):
        frame_name = frame_names[frame_idx]
        if prompt_source == "gt_box":
            box = get_gt_box(
                seq_dir=seq_dir,
                frame_name=frame_name,
                gt_box_source=gt_box_source,
                bbox_root=bbox_root,
                csv_boxes=csv_boxes,
                box_padding=box_padding,
            )
            conf = None
        else:
            box, conf = get_yolo_box(
                yolo_model,
                seq_dir / "images" / frame_name,
                yolo_imgsz=yolo_imgsz,
                yolo_conf=yolo_conf,
            )

        if box is None:
            continue

        prompt_boxes[frame_idx] = box
        if prompt_source == "yolo":
            x1, y1, x2, y2 = box.tolist()
            yolo_rows.append(
                {
                    "frame_idx": frame_idx,
                    "frame_name": Path(frame_name).stem,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf,
                }
            )

    if prompt_source == "yolo":
        prompt_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_idx",
                    "frame_name",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "confidence",
                ],
            )
            writer.writeheader()
            writer.writerows(yolo_rows)

    return prompt_boxes


def run_sequence_stride(
    predictor,
    seq_dir: Path,
    stride: int,
    prompt_source: str,
    args,
    yolo_model=None,
):
    images_dir = seq_dir / "images"
    frame_names = list_frame_names(images_dir)
    method_name = f"{prompt_source}_stride{stride}"
    method_root = Path(args.output_root) / args.method_name / method_name
    masks_dir = method_root / "masks" / seq_dir.name / "predicted"
    bbox_csv_path = method_root / "bbox" / f"{seq_dir.name}.csv"

    if not args.overwrite and all((masks_dir / f"{Path(name).stem}.png").exists() for name in frame_names):
        rows = evaluate_saved_masks(
            seq_dir=seq_dir,
            frame_names=frame_names,
            masks_dir=masks_dir,
            stride=stride,
            prompt_source=prompt_source,
        )
        write_metrics_csv(method_root / "metrics" / f"{seq_dir.name}_frames.csv", rows)
        return rows

    prompt_boxes = prepare_prompt_boxes(
        seq_dir=seq_dir,
        frame_names=frame_names,
        stride=stride,
        prompt_source=prompt_source,
        gt_box_source=args.gt_box_source,
        bbox_root=Path(args.bbox_root) if args.bbox_root else None,
        box_padding=args.box_padding,
        yolo_model=yolo_model,
        yolo_imgsz=args.yolo_imgsz,
        yolo_conf=args.yolo_conf,
        prompt_csv_path=bbox_csv_path,
    )
    if 0 not in prompt_boxes:
        raise RuntimeError(
            f"{seq_dir.name}: no usable first-frame {prompt_source} box was found."
        )

    inference_state = predictor.init_state(
        video_path=str(images_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )

    for frame_idx, box in prompt_boxes.items():
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=args.obj_id,
            box=box,
            clear_old_points=True,
            normalize_coords=True,
        )

    predicted_by_idx = {}
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        frame_name = frame_names[out_frame_idx]
        output_path = masks_dir / f"{Path(frame_name).stem}.png"
        pred_mask = save_mask(out_mask_logits[0], output_path, args.score_threshold)
        predicted_by_idx[out_frame_idx] = pred_mask

    rows = []
    for frame_idx, frame_name in enumerate(frame_names):
        pred = predicted_by_idx.get(frame_idx)
        if pred is None:
            continue
        gt = load_binary_mask(seq_dir / "masks" / frame_name)
        rows.append(
            FrameMetric(
                sequence=seq_dir.name,
                frame_name=frame_name,
                stride=stride,
                prompt_source=prompt_source,
                is_prompt_frame=frame_idx in prompt_boxes,
                dice=dice_score(pred, gt),
                iou=iou_score(pred, gt),
            )
        )

    write_metrics_csv(method_root / "metrics" / f"{seq_dir.name}_frames.csv", rows)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM2VideoPredictor with GT-box or YOLO-box prompts."
    )
    parser.add_argument("--data-root", default="data/test/polypgen")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--method-name", default="sam2_video_box")
    parser.add_argument("--seq-nums", type=int, nargs="*")
    parser.add_argument(
        "--prompt-sources",
        nargs="+",
        choices=["gt_box", "yolo"],
        default=["gt_box", "yolo"],
    )
    parser.add_argument("--prompt-strides", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--sam2-cfg", default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--sam2-checkpoint", default="checkpoints/MedSAM2_latest.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--yolo-checkpoint", default="checkpoints/polypgen_yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument(
        "--gt-box-source",
        choices=["mask", "csv"],
        default="mask",
        help="Use boxes computed from GT masks or loaded from --bbox-root.",
    )
    parser.add_argument("--bbox-root", default=None)
    parser.add_argument("--box-padding", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--no-postprocessing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=not args.no_postprocessing,
    )
    predictor.eval()

    yolo_model = None
    if "yolo" in args.prompt_sources:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for --prompt-sources yolo. "
                "Install it or run with --prompt-sources gt_box."
            ) from exc
        yolo_model = YOLO(args.yolo_checkpoint)

    sequences = list_sequences(data_root, args.seq_nums)
    all_rows: list[FrameMetric] = []
    summary_rows = []

    with torch.inference_mode():
        for prompt_source in args.prompt_sources:
            for stride in args.prompt_strides:
                if stride <= 0:
                    raise ValueError(f"Prompt stride must be positive, got {stride}")
                combo_rows = []
                desc = f"{prompt_source} stride={stride}"
                for seq_dir in tqdm(sequences, desc=desc):
                    rows = run_sequence_stride(
                        predictor=predictor,
                        seq_dir=seq_dir,
                        stride=stride,
                        prompt_source=prompt_source,
                        args=args,
                        yolo_model=yolo_model,
                    )
                    combo_rows.extend(rows)
                    if rows:
                        summary_rows.append(
                            {
                                "prompt_source": prompt_source,
                                "stride": stride,
                                "sequence": seq_dir.name,
                                "num_frames": len(rows),
                                "mean_dice": float(np.mean([r.dice for r in rows])),
                                "mean_iou": float(np.mean([r.iou for r in rows])),
                            }
                        )

                all_rows.extend(combo_rows)
                if combo_rows:
                    summary_rows.append(
                        {
                            "prompt_source": prompt_source,
                            "stride": stride,
                            "sequence": "ALL",
                            "num_frames": len(combo_rows),
                            "mean_dice": float(np.mean([r.dice for r in combo_rows])),
                            "mean_iou": float(np.mean([r.iou for r in combo_rows])),
                        }
                    )

    output_root = Path(args.output_root) / args.method_name
    write_metrics_csv(output_root / "all_frame_metrics.csv", all_rows)
    write_summary_csv(output_root / "summary_metrics.csv", summary_rows)
    print(f"Saved masks and metrics under {output_root}")


if __name__ == "__main__":
    main()
