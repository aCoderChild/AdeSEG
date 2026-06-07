# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils.model_defaults import (
    DEFAULT_SAM2_CFG,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CODE_ROOT,
    DEFAULT_YOLO_CHECKPOINT,
)

if DEFAULT_SAM2_CODE_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_SAM2_CODE_ROOT)

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from scripts.utils.eval import run_sequence_evaluation

# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

def get_numeric_sort_key(name):
    """Natural sort key, e.g. 2 before 10 and seq2 before seq10."""
    parts = re.split(r"(\d+)", name)
    return tuple(int(part) if part.isdigit() else part for part in parts)

def is_image_file(path):
    return os.path.splitext(path)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]

def get_video_frame_dir(base_video_dir, video_name):
    """Support both DAVIS-style video dirs and PolypGen seq/images dirs."""
    if video_name == ".":
        image_dir = os.path.join(base_video_dir, "images")
        if os.path.isdir(image_dir):
            return image_dir
        return base_video_dir
    image_dir = os.path.join(base_video_dir, video_name, "images")
    if os.path.isdir(image_dir):
        return image_dir
    return os.path.join(base_video_dir, video_name)

def get_video_name(base_video_dir, video_name):
    if video_name != ".":
        return video_name
    parent = os.path.basename(os.path.dirname(os.path.abspath(base_video_dir)))
    current = os.path.basename(os.path.abspath(base_video_dir))
    return parent if current == "images" else current

def list_video_names(base_video_dir):
    """Return sub-video names or '.' when base_video_dir is itself an image folder."""
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

    frame_names = [p for p in os.listdir(base_video_dir) if is_image_file(p)]
    if frame_names:
        return ["."]
    return []

def get_frame_names(video_dir):
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if is_image_file(p)
    ]
    return sorted(frame_names, key=get_numeric_sort_key)

def resolve_frame_path(video_dir, frame_name):
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        frame_path = os.path.join(video_dir, f"{frame_name}{ext}")
        if os.path.exists(frame_path):
            return frame_path
    raise FileNotFoundError(os.path.join(video_dir, f"{frame_name}.jpg"))

# used for eval
def infer_gt_mask_dir(base_video_dir, video_name):
    if video_name == ".":
        parent = os.path.dirname(os.path.abspath(base_video_dir))
        if os.path.basename(os.path.abspath(base_video_dir)) == "images":
            return os.path.join(parent, "masks")
        return os.path.join(base_video_dir, "masks")
    return os.path.join(base_video_dir, video_name, "masks")

def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)

# used for future adenoid and nasopharynx segmentation
def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask

def save_palette_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)

def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
):
    """Save masks to a directory as greyscale PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask = (output_mask > 0).astype(np.uint8) * 255
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        assert output_mask.dtype == np.uint8
        assert output_mask.ndim == 2
        output_mask = Image.fromarray(output_mask)
        output_mask.save(output_mask_path)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask = (output_mask > 0).astype(np.uint8) * 255
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            assert output_mask.dtype == np.uint8
            assert output_mask.ndim == 2
            output_mask = Image.fromarray(output_mask)
            output_mask.save(output_mask_path)

# box generated from YOLOv8
def get_yolo_boxes(yolo_model, frame_path, yolo_imgsz, yolo_conf, max_boxes):
    image = Image.open(frame_path).convert("RGB")
    results = yolo_model.predict([image], imgsz=yolo_imgsz, conf=yolo_conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    order = np.argsort(conf)[::-1]
    if max_boxes > 0:
        order = order[:max_boxes]
    return [(xyxy[i].astype(np.float32), float(conf[i])) for i in order]

# use ground-truth boxes
def load_gt_bboxes(bbox_root, video_output_name):
    bbox_file = os.path.join(bbox_root, video_output_name, "bboxes.csv")
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
            bboxes.setdefault(frame_name, []).append((box, 1.0))
    return bboxes

# multle masks from the image predictor into 1 single foreground mask
# multiple masks do NOT have to be of 1 object
def combine_image_predictor_masks(masks):
    pred_mask = None
    for mask in masks:
        mask_arr = mask[0] if isinstance(mask, (list, tuple)) else mask
        mask_arr = np.squeeze(np.asarray(mask_arr)).astype(np.uint8)
        if pred_mask is None:
            pred_mask = mask_arr
        else:
            pred_mask = np.logical_or(pred_mask, mask_arr).astype(np.uint8)
    return pred_mask


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def save_intermediate_mask_prompt(
    binary_mask_dir,
    logits_dir,
    video_name,
    frame_name,
    mask_logits,
    scores,
    boxes,
):
    logits_arr = np.squeeze(np.asarray(mask_logits)).astype(np.float32)
    if logits_arr.ndim == 2:
        logits_arr = logits_arr[None, ...]
    probabilities = sigmoid(logits_arr).astype(np.float32)
    combined_logit = np.max(logits_arr, axis=0)
    combined_probability = np.max(probabilities, axis=0)
    combined_binary = (combined_logit > 0).astype(np.uint8)

    if binary_mask_dir is not None:
        binary_dir = os.path.join(binary_mask_dir, video_name)
        os.makedirs(binary_dir, exist_ok=True)
        Image.fromarray(combined_binary * 255).save(
            os.path.join(binary_dir, f"{frame_name}.png")
        )

    if logits_dir is not None:
        frame_logits_dir = os.path.join(logits_dir, video_name)
        os.makedirs(frame_logits_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(frame_logits_dir, f"{frame_name}.npz"),
            logits=logits_arr,
            probabilities=probabilities,
            combined_logit=combined_logit.astype(np.float32),
            combined_probability=combined_probability.astype(np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            boxes=np.asarray([box for box, _ in boxes], dtype=np.float32),
        )


def load_binary_mask(path):
    mask = Image.open(path).convert("L")
    mask = np.array(mask)
    threshold = 0 if mask.max() <= 1 else 127
    return (mask > threshold).astype(np.uint8)

def resolve_mask_path(mask_dir, frame_name):
    for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]:
        mask_path = os.path.join(mask_dir, f"{frame_name}{ext}")
        if os.path.exists(mask_path):
            return mask_path
    return None

# log for error handling
# status + error type
def write_sequence_log(log_dir, video_name, data):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{video_name}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return log_path

# runs MedSAM2 video segmentation for one sequence/video.
# uses YOLO BOUDNING BOXES as PROMTPS => SAM2 video predictor => masks
# also MASKS as PROMPTS: Bounding boxes == Image-predictor ==> Mask Prompts == Video-predictor ==> predicted masks
def yolo_vos_inference(
    predictor,
    yolo_model,
    base_video_dir,
    output_mask_dir,
    video_name,
    output_video_name=None,
    image_predictor=None,
    score_thresh=0.0,
    save_palette_png=False,
    yolo_imgsz=640,
    yolo_conf=0.5,
    yolo_iou_match_thresh=0.3,
    max_yolo_boxes_per_frame=1,
    bbox_output_dir=None,
    log_dir=None,
    intermediate_binary_mask_dir=None,
    intermediate_logits_dir=None,
    prompt_source="yolo",
    gt_bbox_root=None,
    sam2_cfg=None,
    sam2_checkpoint=None,
    yolo_checkpoint=None,
    video_prompt_source="mask",
    video_prompt_stride=1,
    video_prompt_limit=0,
):
    """Run video segmentation using YOLO-derived box or mask prompts."""
    video_dir = get_video_frame_dir(base_video_dir, video_name)
    video_output_name = get_video_name(base_video_dir, video_name)
    mask_output_name = output_video_name or video_output_name
    frame_names = get_frame_names(video_dir)
    gt_bboxes = load_gt_bboxes(gt_bbox_root, video_output_name) if prompt_source == "gt_bbox" else {}
    run_log = {
        "video": video_output_name,
        "status": "started",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video_dir": video_dir,
        "output_mask_dir": os.path.join(output_mask_dir, mask_output_name),
        "bbox_output_dir": bbox_output_dir,
        "intermediate_binary_mask_dir": intermediate_binary_mask_dir,
        "intermediate_logits_dir": intermediate_logits_dir,
        "num_frames": len(frame_names),
        "height": None,
        "width": None,
        "yolo_conf": yolo_conf,
        "yolo_imgsz": yolo_imgsz,
        "max_yolo_boxes_per_frame": max_yolo_boxes_per_frame,
        "prompt_source": prompt_source,
        "gt_bbox_root": gt_bbox_root,
        "sam2_cfg": sam2_cfg,
        "sam2_checkpoint": sam2_checkpoint,
        "yolo_checkpoint": yolo_checkpoint,
        "video_prompt_source": video_prompt_source, # what type of prompt is passed into the MedSAM2 video predictor
        "video_prompt_stride": video_prompt_stride,
        "video_prompt_limit": video_prompt_limit,
        "prompt_attempt_frames": [],
        "seeded_frames": [],
        "num_yolo_detections_total": 0,
        "num_saved_masks": 0,
        "error": None,
    }
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    run_log["height"] = int(height)
    run_log["width"] = int(width)

    seeded_frames = []
    bbox_rows = []
    # stride prompts
    for frame_idx, frame_name in enumerate(frame_names):
        if video_prompt_limit > 0 and len(seeded_frames) >= video_prompt_limit:
            continue
        if video_prompt_stride > 1 and frame_idx % video_prompt_stride != 0:
            continue
        frame_path = resolve_frame_path(video_dir, frame_name)
        frame_log = {
            "frame_idx": frame_idx,
            "frame": frame_name,
            "frame_path": frame_path,
            "num_boxes": 0,
            "box_confidences": [],
            "seeded": False,
        }
        run_log["prompt_attempt_frames"].append(frame_log)
        if prompt_source == "gt_bbox":
            boxes = list(gt_bboxes.get(frame_name, []))
            if max_yolo_boxes_per_frame > 0:
                boxes = boxes[:max_yolo_boxes_per_frame]
        else:
            boxes = get_yolo_boxes(
                yolo_model=yolo_model,
                frame_path=frame_path,
                yolo_imgsz=yolo_imgsz,
                yolo_conf=yolo_conf,
                max_boxes=max_yolo_boxes_per_frame,
            )
        frame_log["num_boxes"] = len(boxes)
        frame_log["box_confidences"] = [float(conf) for _, conf in boxes]
        run_log["num_yolo_detections_total"] += len(boxes)
        if not boxes:
            continue

        prompt_masks = None
        if video_prompt_source == "mask":
            if image_predictor is None:
                raise RuntimeError("video_prompt_source='mask' requires image_predictor")
            image = Image.open(frame_path).convert("RGB")
            image_np = np.array(image)
            boxes_np = np.stack([box for box, _ in boxes], axis=0)
            image_predictor.set_image(image_np)
            mask_logits, scores, _ = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_np,
                multimask_output=False,
                return_logits=True,
            )
            save_intermediate_mask_prompt(
                binary_mask_dir=intermediate_binary_mask_dir,
                logits_dir=intermediate_logits_dir,
                video_name=mask_output_name,
                frame_name=frame_name,
                mask_logits=mask_logits,
                scores=scores,
                boxes=boxes,
            )
            prompt_masks = [
                np.squeeze(np.asarray(mask_logit)) > 0
                for mask_logit in mask_logits
            ]

        assignments = [(1, box, conf) for box, conf in boxes]
        seeded_frames.append(frame_idx)
        frame_log["seeded"] = True
        run_log["seeded_frames"].append(
            {
                "frame_idx": frame_idx,
                "frame": frame_name,
                "num_boxes": len(boxes),
                "box_confidences": [float(conf) for _, conf in boxes],
            }
        )
        for box_id, (obj_id, box, conf) in enumerate(assignments):
            if video_prompt_source == "mask":
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=prompt_masks[box_id],
                )
            else:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=box,
                )
            if prompt_source == "yolo":
                x1, y1, x2, y2 = [float(v) for v in box]
                bbox_rows.append(
                    {
                        "frame": frame_name,
                        "box_id": box_id,
                        "x_center": ((x1 + x2) / 2) / width,
                        "y_center": ((y1 + y2) / 2) / height,
                        "width": (x2 - x1) / width,
                        "height": (y2 - y1) / height,
                    }
                )

    if bbox_output_dir is not None:
        os.makedirs(bbox_output_dir, exist_ok=True)
        bbox_path = os.path.join(bbox_output_dir, f"{video_output_name}.csv")
        with open(bbox_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame", "box_id", "x_center", "y_center", "width", "height"],
            )
            writer.writeheader()
            writer.writerows(bbox_rows)

    os.makedirs(os.path.join(output_mask_dir, mask_output_name), exist_ok=True)
    if not seeded_frames:
        print(
            f"Warning: in {video_output_name}, YOLO found no boxes at conf={yolo_conf}; "
            "writing blank masks so evaluation counts this failure."
        )
        for frame_name in frame_names:
            save_masks_to_dir(
                output_mask_dir=output_mask_dir,
                video_name=mask_output_name,
                frame_name=frame_name,
                per_obj_output_mask={},
                height=height,
                width=width,
                per_obj_png_file=False,
            )
        run_log["status"] = "no_prompts_blank_masks_written"
        run_log["num_saved_masks"] = len(frame_names)
        run_log["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if log_dir is not None:
            write_sequence_log(log_dir, video_output_name, run_log)
        return video_output_name, frame_names

    video_segments = {}
    first_seeded_frame = min(seeded_frames)
    run_log["first_seeded_frame_idx"] = int(first_seeded_frame)
    # for offline video recording
    for reverse in (False, True):
        direction = "backward" if reverse else "forward"
        propagated_frames = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=first_seeded_frame,
            reverse=reverse,
        ):
            propagated_frames.append(int(out_frame_idx))
            per_obj_output_mask = {
                out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            video_segments[out_frame_idx] = per_obj_output_mask
        run_log[f"propagated_frames_{direction}"] = propagated_frames

    output_palette = DAVIS_PALETTE
    saved_masks = 0
    for out_frame_idx in range(len(frame_names)):
        per_obj_output_mask = video_segments.get(out_frame_idx, {})
        if save_palette_png:
            save_palette_masks_to_dir(
                output_mask_dir=output_mask_dir,
                video_name=mask_output_name,
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
                per_obj_png_file=False,
                output_palette=output_palette,
            )
        else:
            save_masks_to_dir(
                output_mask_dir=output_mask_dir,
                video_name=mask_output_name,
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
                per_obj_png_file=False,
            )
        saved_masks += 1

    run_log["status"] = "success"
    run_log["num_video_segments"] = len(video_segments)
    run_log["num_saved_masks"] = saved_masks
    run_log["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if log_dir is not None:
        write_sequence_log(log_dir, video_output_name, run_log)

    return video_output_name, frame_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default=DEFAULT_SAM2_CFG,
        help="MedSAM2  model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default=DEFAULT_SAM2_CHECKPOINT,
        help="path to the MedSAM2 model checkpoint",
    )
    parser.add_argument(
        "-i",
        "--base_video_dir",
        type=str,
        required=True,
        help=(
            "directory containing image frames. This can be a single images/ folder "
            "or a dataset root containing seq*/images folders"
        ),
    )
    parser.add_argument(
        "--yolo_checkpoint",
        type=str,
        default=DEFAULT_YOLO_CHECKPOINT,
        help="YOLOv8 checkpoint used to generate box prompts",
    )
    parser.add_argument(
        "--gt_mask_dir",
        type=str,
        default=None,
        help=(
            "optional ground-truth mask directory for evaluation. If omitted, "
            "the script uses the sibling masks/ directory for a single images/ input "
            "or seq*/masks for a dataset root"
        ),
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run inference on",
    )
    parser.add_argument(
        "--seq_nums",
        type=int,
        nargs="*",
        default=None,
        help="optional sequence numbers to process, e.g. --seq_nums 1 5 23. If omitted, process all discovered sequences.",
    )
    parser.add_argument(
        "-o",
        "--output_mask_dir",
        type=str,
        default=None,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory for method outputs.",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="MedSAM2",
        help="Output namespace under --output_root.",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for video-mode output mask logits (default: 0.0)",
    )

    parser.add_argument(
        "--yolo_conf",
        type=float,
        default=0.5,
        help="YOLO confidence threshold for box prompts",
    )
    parser.add_argument(
        "--prompt_source",
        choices=["yolo", "gt_bbox"],
        default="yolo",
        help="Use YOLO-predicted boxes or ground-truth boxes from --bbox_root.",
    )
    parser.add_argument(
        "--bbox_root",
        type=str,
        default="data/bbox",
        help="Root containing seq*/bboxes.csv for prompt_source=gt_bbox.",
    )
    parser.add_argument(
        "--yolo_imgsz",
        type=int,
        default=640,
        help="YOLO inference image size",
    )
    parser.add_argument(
        "--yolo_iou_match_thresh",
        type=float,
        default=0.3,
        help="IoU threshold for matching YOLO boxes into stable object ids",
    )
    parser.add_argument(
        "--video_prompt_source",
        choices=["mask", "box"],
        default="mask",
        help=(
            "video mode prompt type. mask first converts YOLO boxes to MedSAM2 "
            "masks and injects those into the video predictor; box uses raw YOLO boxes."
        ),
    )
    parser.add_argument(
        "--video_prompt_stride",
        type=int,
        default=1,
        help="use YOLO-derived video prompts every N frames in video mode",
    )
    parser.add_argument(
        "--video_prompt_limit",
        type=int,
        default=0,
        help="maximum number of YOLO-derived video prompt frames; 0 means no limit",
    )
    parser.add_argument(
        "--max_yolo_boxes_per_frame",
        type=int,
        default=1,
        help="maximum YOLO boxes to use per frame; 1 is recommended for single-polyp videos",
    )
    parser.add_argument(
        "--save_palette_png",
        action="store_true",
        help="whether to save palette PNG files for output masks "
        "(default without this flag: all object masks are saved as grayscale PNG files (np.uint8) without palette)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--use_vos_optimized_video_predictor",
        action="store_true",
        help="whether to use vos optimized video predictor with all modules compiled",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Run evaluation after inference to generate metrics, bbox scores, and overlays",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for per-sequence JSON inference logs.",
    )
    parser.add_argument(
        "--save_intermediate_mask_prompts",
        action="store_true",
        help=(
            "When video_prompt_source=mask, save image-predictor prompt masks "
            "before video propagation: binary PNGs for evaluation and NPZ files "
            "with raw logits/probabilities."
        ),
    )
    args = parser.parse_args()
    if args.output_mask_dir is None:
        args.output_mask_dir = os.path.join(args.output_root, args.method_name, "masks")
    if args.log_dir is None:
        args.log_dir = os.path.join(args.output_root, args.method_name, "logs")

    # Initialize video predictor
    hydra_overrides_extra = ["++model.non_overlap_masks=true"]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
        vos_optimized=args.use_vos_optimized_video_predictor,
    )
    
    # Initialize image predictor if needed for mask prompts
    image_predictor = None
    if args.video_prompt_source == "mask":
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_model = build_sam2(
            config_file=args.sam2_cfg,
            ckpt_path=args.sam2_checkpoint,
            apply_postprocessing=args.apply_postprocessing,
        )
        image_predictor = SAM2ImagePredictor(sam2_model)
    
    yolo_model = YOLO(args.yolo_checkpoint) if args.prompt_source == "yolo" else None

    print(
        f"using {args.prompt_source} boxes as MedSAM2 prompts "
        f"with video mode inference"
    )
    # if a video list file is provided, read the video names from the file
    # (otherwise, we use all subdirectories in base_video_dir)
    if args.seq_nums is not None and len(args.seq_nums) > 0:
        video_names = [f"seq{seq_num}" for seq_num in args.seq_nums]
    elif args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = list_video_names(args.base_video_dir)
    print(f"running inference on {len(video_names)} videos:\n{video_names}")
    bbox_output_dir = (
        os.path.join(args.output_root, args.method_name, "bbox")
        if args.prompt_source == "yolo"
        else None
    )
    intermediate_method_name = f"{args.method_name}_IMAGE_PROMPT"
    save_intermediate_prompts = (
        args.save_intermediate_mask_prompts
        and args.video_prompt_source == "mask"
    )
    intermediate_binary_mask_dir = (
        os.path.join(args.output_root, intermediate_method_name, "masks")
        if save_intermediate_prompts
        else None
    )
    intermediate_logits_dir = (
        os.path.join(args.output_root, intermediate_method_name, "logits")
        if save_intermediate_prompts
        else None
    )

    for n_video, video_name in enumerate(video_names):
        display_name = get_video_name(args.base_video_dir, video_name)
        print(f"\n{n_video + 1}/{len(video_names)} - running on {display_name}")
        try:
            video_output_name, frame_names = yolo_vos_inference(
                predictor=predictor,
                yolo_model=yolo_model,
                base_video_dir=args.base_video_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                output_video_name=os.path.join(display_name, "predicted"),
                image_predictor=image_predictor,
                score_thresh=args.score_thresh,
                save_palette_png=args.save_palette_png,
                yolo_imgsz=args.yolo_imgsz,
                yolo_conf=args.yolo_conf,
                yolo_iou_match_thresh=args.yolo_iou_match_thresh,
                max_yolo_boxes_per_frame=args.max_yolo_boxes_per_frame,
                bbox_output_dir=bbox_output_dir,
                log_dir=args.log_dir,
                intermediate_binary_mask_dir=intermediate_binary_mask_dir,
                intermediate_logits_dir=intermediate_logits_dir,
                prompt_source=args.prompt_source,
                gt_bbox_root=args.bbox_root,
                sam2_cfg=args.sam2_cfg,
                sam2_checkpoint=args.sam2_checkpoint,
                yolo_checkpoint=args.yolo_checkpoint,
                video_prompt_source=args.video_prompt_source,
                video_prompt_stride=args.video_prompt_stride,
                video_prompt_limit=args.video_prompt_limit,
            )
        except Exception as e:
            log_data = {
                "video": display_name,
                "status": "failed",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
                "output_mask_dir": os.path.join(args.output_mask_dir, display_name, "predicted"),
                "bbox_output_dir": bbox_output_dir,
            }
            log_path = write_sequence_log(args.log_dir, display_name, log_data)
            print(f"Warning: failed {display_name}; details written to {log_path}: {e}")
            # Write blank masks after logging so evaluation counts this sequence as IoU=0
            # rather than silently excluding it from the aggregate mean.
            try:
                video_dir = get_video_frame_dir(args.base_video_dir, video_name)
                fail_frame_names = get_frame_names(video_dir)
                if fail_frame_names:
                    first_path = resolve_frame_path(video_dir, fail_frame_names[0])
                    with Image.open(first_path) as _img:
                        fail_height, fail_width = _img.height, _img.width
                    mask_output_name = os.path.join(display_name, "predicted")
                    for fn in fail_frame_names:
                        save_masks_to_dir(
                            output_mask_dir=args.output_mask_dir,
                            video_name=mask_output_name,
                            frame_name=fn,
                            per_obj_output_mask={},
                            height=fail_height,
                            width=fail_width,
                            per_obj_png_file=False,
                        )
                    log_data["status"] = "failed_blank_masks_written"
                    log_data["num_saved_masks"] = len(fail_frame_names)
                    log_data["num_frames"] = len(fail_frame_names)
                    write_sequence_log(args.log_dir, display_name, log_data)
                    print(f"  → Wrote {len(fail_frame_names)} blank masks for {display_name}")
            except Exception as blank_err:
                print(f"  → Could not write blank masks for {display_name}: {blank_err}")
            continue

    print(
        f"completed inference on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )

    # Run evaluation if requested
    if args.run_eval:
        run_sequence_evaluation(
            sequence_names=[
                get_video_name(args.base_video_dir, video_name)
                for video_name in video_names
            ],
            data_root=args.base_video_dir,
            bbox_root=args.bbox_root,
            output_root=args.output_root,
            method_name=args.method_name,
            prompt_source=args.prompt_source,
            verbose=True,
        )
        if save_intermediate_prompts:
            print("Evaluating intermediate image-predictor mask prompts")
            run_sequence_evaluation(
                sequence_names=[
                    get_video_name(args.base_video_dir, video_name)
                    for video_name in video_names
                ],
                data_root=args.base_video_dir,
                bbox_root=args.bbox_root,
                output_root=args.output_root,
                method_name=intermediate_method_name,
                prompt_source="gt_bbox",
                verbose=True,
            )

if __name__ == "__main__":
    main()
