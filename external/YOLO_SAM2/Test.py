import argparse
import os
import shutil
import glob
import re
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import csv
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from scripts.utils.model_defaults import (
    DEFAULT_SAM2_CFG,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CODE_ROOT,
    DEFAULT_YOLO_CHECKPOINT,
)

if DEFAULT_SAM2_CODE_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_SAM2_CODE_ROOT)

from scripts.utils.eval import run_sequence_evaluation


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def get_numeric_sort_key(filename):
    basename = os.path.basename(filename)
    parts = re.split(r"(\d+)", os.path.splitext(basename)[0])
    return tuple(int(part) if part.isdigit() else part for part in parts)


def load_gt_bboxes(bbox_root, seq_name):
    bbox_file = os.path.join(bbox_root, seq_name, "bboxes.csv")
    bboxes = {}
    if not os.path.isfile(bbox_file):
        return bboxes
    with open(bbox_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_name = str(int(row["frame_idx"]))
            box = np.array(
                [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ],
                dtype=np.float32,
            )
            bboxes.setdefault(frame_name, []).append(box)
    return bboxes


def list_sequence_nums(base_dir):
    seq_nums = []
    for entry in os.listdir(base_dir):
        seq_path = os.path.join(base_dir, entry)
        match = re.fullmatch(r"seq(\d+)", entry)
        if match and os.path.isdir(seq_path):
            seq_nums.append(int(match.group(1)))
    return sorted(seq_nums)


def main():
    """
    YOLO_SAM2 Inference - Inference-Only
    
    Outputs:
    - predicted_masks/*.png: Binary mask images from SAM2
    - bbox.csv: YOLO-predicted boxes when --prompt_source yolo
    
    Does NOT compute metrics. Use --run_eval flag to delegate evaluation to eval_metrics.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default=DEFAULT_SAM2_CFG,
        help="SAM/MedSAM2 model configuration file.",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default=DEFAULT_SAM2_CHECKPOINT,
        help="Path to the SAM/MedSAM2 model checkpoint.",
    )
    parser.add_argument(
        "--yolo_checkpoint",
        "--checkpoint_add",
        dest="yolo_checkpoint",
        default=DEFAULT_YOLO_CHECKPOINT,
        help="YOLO pre-trained model checkpoint.",
    )
    parser.add_argument(
        "--seq_num",
        type=int,
        default=None,
        help="Single sequence number to process, e.g. --seq_num 5. If omitted, all sequences are processed.",
    )
    parser.add_argument(
        "--seq_nums",
        type=int,
        nargs="*",
        default=None,
        help="Optional sequence numbers to process, e.g. --seq_nums 1 5 23. If omitted, process all discovered sequences.",
    )
    parser.add_argument(
        "--base_dir",
        default=os.environ.get("POLYPGEN_DIR", "data/polypgen"),
        help="Dataset root containing seq*/images and seq*/masks directories.",
    )
    parser.add_argument(
        "--yolo_conf",
        type=float,
        default=0.5,
        help="YOLO confidence threshold for box prompts.",
    )
    parser.add_argument(
        "--yolo_imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--max_yolo_boxes_per_frame",
        type=int,
        default=1,
        help="Maximum YOLO boxes to use per frame; 1 matches the MedSAM2 pipeline.",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Run evaluation after inference.",
    )
    parser.add_argument(
        "--output_root",
        default="outputs",
        help="Root directory for method outputs.",
    )
    parser.add_argument(
        "--method_name",
        default="YOLO_SAM2",
        help="Output namespace under --output_root.",
    )
    parser.add_argument(
        "--prompt_source",
        choices=["yolo", "gt_bbox"],
        default="yolo",
        help="Use YOLO-predicted boxes or ground-truth boxes from --bbox_root.",
    )
    parser.add_argument(
        "--bbox_root",
        default="data/bbox",
        help="Root containing seq*/bboxes.csv for prompt_source=gt_bbox.",
    )
    args = parser.parse_args()

    sam2_model = build_sam2(args.sam2_cfg, args.sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    yolo_model = YOLO(args.yolo_checkpoint) if args.prompt_source == "yolo" else None

    base_dir = args.base_dir
    if args.seq_nums is not None and len(args.seq_nums) > 0:
        seq_nums = args.seq_nums
    elif args.seq_num is not None:
        seq_nums = [args.seq_num]
    else:
        seq_nums = list_sequence_nums(base_dir)
    if not seq_nums:
        raise FileNotFoundError(f"No seq* directories found under {base_dir}")

    save_prompt_bboxes = args.prompt_source == "yolo"
    print(f"Processing {len(seq_nums)} sequences: {[f'seq{n}' for n in seq_nums]}")
    for seq_num in seq_nums:
        seq_path = os.path.join(base_dir, f"seq{seq_num}")
        if not os.path.isdir(seq_path):
            raise FileNotFoundError(f"Requested sequence not found: {seq_path}")

        seq_dir = seq_path
        seq_name = os.path.basename(seq_dir)
        print(f"\nProcessing {seq_name}...")
        gt_bboxes = load_gt_bboxes(args.bbox_root, seq_name) if args.prompt_source == "gt_bbox" else {}

        # directory for predicted masks (save with same filename as source images)
        predicted_masks_dir = os.path.join(args.output_root, args.method_name, "masks", seq_name, "predicted")
        os.makedirs(predicted_masks_dir, exist_ok=True)

        images_dir = os.path.join(seq_dir, "images")
        image_files = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            image_files.extend(glob.glob(os.path.join(images_dir, pattern)))
        image_files = sorted(list(dict.fromkeys(image_files)), key=get_numeric_sort_key)

        seq_bbox_data = []
        if save_prompt_bboxes:
            bbox_dir = os.path.join(args.output_root, args.method_name, "bbox")
            os.makedirs(bbox_dir, exist_ok=True)
            bbox_csv = os.path.join(bbox_dir, f"{seq_name}.csv")

        for img_idx, img_path in enumerate(image_files):
            frame_name = os.path.basename(img_path).split(".")[0]
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
            img_h, img_w = image_np.shape[:2]
            if args.prompt_source == "gt_bbox":
                boxes_np = np.array(gt_bboxes.get(frame_name, []), dtype=np.float32)
                if args.max_yolo_boxes_per_frame > 0:
                    boxes_np = boxes_np[:args.max_yolo_boxes_per_frame]
            else:
                results = yolo_model.predict([image], imgsz=args.yolo_imgsz, conf=args.yolo_conf, verbose=False)
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    boxes_np = np.empty((0, 4), dtype=np.float32)
                else:
                    boxes_np = boxes.xyxy.detach().cpu().numpy()
                    box_conf = boxes.conf.detach().cpu().numpy()
                    order = np.argsort(box_conf)[::-1]
                    if args.max_yolo_boxes_per_frame > 0:
                        order = order[:args.max_yolo_boxes_per_frame]
                    boxes_np = boxes_np[order]

            # default prediction
            pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            n_boxes = len(boxes_np)
            if n_boxes == 0:
                pass
            elif n_boxes == 1:
                predictor.set_image(image_np)
                input_box = boxes_np[0]
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                pred_mask = masks[0]

                if save_prompt_bboxes:
                    x1, y1, x2, y2 = input_box
                    cx = (x1 + x2) / 2 / img_w
                    cy = (y1 + y2) / 2 / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    seq_bbox_data.append([frame_name, 0, cx, cy, w, h])
            else:
                predictor.set_image(image_np)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes_np,
                    multimask_output=False,
                )
                # combine masks
                pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                for m in masks:
                    # each m is (H,W) mask or tuple
                    mask_arr = m[0] if isinstance(m, (list, tuple)) else m
                    pred_mask = (pred_mask + mask_arr).clip(0, 1)

                if save_prompt_bboxes:
                    for box_idx, bbox in enumerate(boxes_np):
                        x1, y1, x2, y2 = bbox
                        cx = (x1 + x2) / 2 / img_w
                        cy = (y1 + y2) / 2 / img_h
                        w = (x2 - x1) / img_w
                        h = (y2 - y1) / img_h
                        seq_bbox_data.append([frame_name, box_idx, cx, cy, w, h])

            # normalize predicted mask: ensure 2D and same size as ground truth
            pred_mask = np.asarray(pred_mask)
            if pred_mask.ndim > 2:
                pred_mask = np.squeeze(pred_mask)
            if pred_mask.shape != (img_h, img_w):
                pred_mask = (pred_mask * 255).astype(np.uint8)
                pred_mask = cv2.resize(pred_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                pred_mask = pred_mask.astype(np.float32) / 255.0

            pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
            # save predicted mask with same filename as original image (use .png)
            mask_output_path = os.path.join(predicted_masks_dir, f"{frame_name}.png")
            cv2.imwrite(mask_output_path, pred_mask_binary * 255)

        if save_prompt_bboxes:
            with open(bbox_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "box_id", "x_center", "y_center", "width", "height"])
                w.writerows(seq_bbox_data)

    # Run evaluation if requested
    if args.run_eval:
        run_sequence_evaluation(
            seq_nums=seq_nums,
            data_root=args.base_dir,
            bbox_root=args.bbox_root,
            output_root=args.output_root,
            method_name=args.method_name,
            prompt_source=args.prompt_source,
            verbose=True,
        )

    print(f"Inference complete. Masks saved under: {os.path.join(args.output_root, args.method_name, 'masks')}")

if __name__ == "__main__":
    main()
