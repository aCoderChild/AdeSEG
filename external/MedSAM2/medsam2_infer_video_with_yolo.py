# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import os
import re
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

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


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def assign_box_ids(boxes, tracked_boxes, next_obj_id, iou_thresh):
    """Assign stable ids to YOLO boxes. With one box per frame this stays obj_id=1."""
    if len(boxes) == 1 and len(tracked_boxes) <= 1:
        tracked_boxes[1] = boxes[0][0]
        return [(1, boxes[0][0], boxes[0][1])], max(next_obj_id, 2)

    assignments = []
    matched_ids = set()
    for box, conf in boxes:
        best_obj_id = None
        best_iou = 0.0
        for obj_id, tracked_box in tracked_boxes.items():
            if obj_id in matched_ids:
                continue
            iou = box_iou(box, tracked_box)
            if iou > best_iou:
                best_iou = iou
                best_obj_id = obj_id
        if best_obj_id is None or best_iou < iou_thresh:
            best_obj_id = next_obj_id
            next_obj_id += 1
        matched_ids.add(best_obj_id)
        tracked_boxes[best_obj_id] = box
        assignments.append((best_obj_id, box, conf))
    return assignments, next_obj_id


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


class StructureMeasure(object):
    def __init__(self):
        self.eps = np.finfo(np.double).eps

    def _object(self, gt, pred):
        x = np.mean(pred[gt])
        sigma_x = np.std(pred[gt])
        return 2 * x / (x * x + 1 + sigma_x + self.eps)

    def _s_object(self, gt, pred):
        pred_fg = pred.copy()
        pred_fg[~gt] = 0
        object_fg = self._object(gt, pred_fg)

        pred_bg = 1 - pred.copy()
        pred_bg[gt] = 0
        object_bg = self._object(~gt, pred_bg)

        foreground_ratio = np.mean(gt)
        return foreground_ratio * object_fg + (1 - foreground_ratio) * object_bg

    def _centroid(self, gt):
        rows, cols = gt.shape
        if np.sum(gt) == 0:
            return round(cols / 2), round(rows / 2)

        total = np.sum(gt)
        col_ids = range(cols)
        row_ids = range(rows)
        x = int(round(np.sum(np.sum(gt, axis=0) * col_ids) / total)) + 1
        y = int(round(np.sum(np.sum(gt, axis=1) * row_ids) / total)) + 1
        return x, y

    def _divide_gt(self, gt, x, y):
        rows, cols = gt.shape
        area = rows * cols
        lt = gt[0:y, 0:x]
        rt = gt[0:y, x:cols]
        lb = gt[y:rows, 0:x]
        rb = gt[y:rows, x:cols]

        w1 = (x * y) / area
        w2 = ((cols - x) * y) / area
        w3 = (x * (rows - y)) / area
        w4 = 1 - w1 - w2 - w3
        return lt, rt, lb, rb, w1, w2, w3, w4

    def _divide_pred(self, pred, x, y):
        rows, cols = pred.shape
        lt = pred[0:y, 0:x]
        rt = pred[0:y, x:cols]
        lb = pred[y:rows, 0:x]
        rb = pred[y:rows, x:cols]
        return lt, rt, lb, rb

    def _ssim(self, gt, pred):
        rows, cols = gt.shape
        n_pixels = rows * cols
        x = np.mean(pred)
        y = np.mean(gt)
        sigma_x2 = np.sum((pred - x) ** 2) / (n_pixels - 1 + self.eps)
        sigma_y2 = np.sum((gt - y) ** 2) / (n_pixels - 1 + self.eps)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (n_pixels - 1 + self.eps)
        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)
        if alpha != 0:
            return alpha / (beta + np.finfo(np.double).eps)
        if beta == 0:
            return 1.0
        return 0

    def _s_region(self, gt, pred):
        x, y = self._centroid(gt)
        gt_lt, gt_rt, gt_lb, gt_rb, w1, w2, w3, w4 = self._divide_gt(gt, x, y)
        pred_lt, pred_rt, pred_lb, pred_rb = self._divide_pred(pred, x, y)
        q1 = self._ssim(gt_lt, pred_lt)
        q2 = self._ssim(gt_rt, pred_rt)
        q3 = self._ssim(gt_lb, pred_lb)
        q4 = self._ssim(gt_rb, pred_rb)
        return w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4

    def __call__(self, gt, pred):
        gt = gt.astype(np.bool_)
        pred = pred.astype(np.double)
        mean_gt = np.mean(gt)
        if mean_gt == 0:
            return 1.0 - np.mean(pred)
        if mean_gt == 1:
            return np.mean(pred)

        score = 0.5 * self._s_object(gt, pred) + 0.5 * self._s_region(gt, pred)
        return max(score, 0)


class EnhancedAlignmentMeasure:
    def __init__(self):
        self.eps = np.finfo(np.double).eps

    def _enhanced_alignment_term(self, align_matrix):
        return ((align_matrix + 1) ** 2) / 4

    def _alignment_term(self, gt, pred):
        mean_pred = np.mean(pred)
        mean_gt = np.mean(gt)
        align_pred = pred - mean_pred
        align_gt = gt - mean_gt
        return 2 * (align_gt * align_pred) / (
            align_gt ** 2 + align_pred ** 2 + self.eps
        )

    def __call__(self, gt, pred):
        gt = gt.astype(np.bool_)
        pred = pred.astype(np.double)
        gt_float = gt.astype(np.float64)
        pred_float = pred.astype(np.float64)
        if np.sum(gt) == 0:
            enhanced_matrix = 1 - pred_float
        elif np.sum(~gt) == 0:
            enhanced_matrix = pred_float
        else:
            align_matrix = self._alignment_term(gt_float, pred_float)
            enhanced_matrix = self._enhanced_alignment_term(align_matrix)
        rows, cols = gt.shape
        return np.sum(enhanced_matrix) / (rows * cols - 1 + self.eps)


def calculate_sensitivity_specificity(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    sensitivity = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    return sensitivity, specificity


def calculate_dice(pred, gt):
    intersection = np.sum(pred * gt)
    return 2 * intersection / (np.sum(pred) + np.sum(gt) + 1e-7)


def calculate_iou(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(np.logical_or(pred, gt))
    return intersection / (union + 1e-7)


def calculate_fmeasure(pred, gt, beta=1):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (
        beta_sq * precision + recall + 1e-7
    )


def calculate_frame_metrics(pred, gt, structure_measure, enhanced_alignment):
    pred_soft = pred.astype(np.double)
    pred_binary = (pred > 0.5).astype(np.uint8)
    gt_binary = gt.astype(np.uint8)
    sensitivity, specificity = calculate_sensitivity_specificity(
        pred_binary, gt_binary
    )
    return {
        "structure_measure": float(structure_measure(gt_binary, pred_soft)),
        "enhanced_alignment": float(enhanced_alignment(gt_binary, pred_soft)),
        "fmeasure": float(calculate_fmeasure(pred_binary, gt_binary)),
        "sensitivity": float(sensitivity),
        "dice": float(calculate_dice(pred_binary, gt_binary)),
        "iou": float(calculate_iou(pred_binary, gt_binary)),
    }


def evaluate_video(
    output_mask_dir,
    video_output_name,
    frame_names,
    gt_mask_dir,
    artifacts_dir=None,
):
    if gt_mask_dir is None or not os.path.isdir(gt_mask_dir):
        print(f"Skipping evaluation for {video_output_name}: GT mask dir not found")
        return

    rows = []
    pred_dir = os.path.join(output_mask_dir, video_output_name)
    structure_measure = StructureMeasure()
    enhanced_alignment = EnhancedAlignmentMeasure()
    for frame_name in frame_names:
        pred_path = os.path.join(pred_dir, f"{frame_name}.png")
        gt_path = resolve_mask_path(gt_mask_dir, frame_name)
        if not os.path.exists(pred_path) or gt_path is None:
            continue
        pred = load_binary_mask(pred_path)
        gt = load_binary_mask(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch on {video_output_name}/{frame_name}: "
                f"pred={pred.shape}, gt={gt.shape}"
            )
        row = {"frame": frame_name}
        row.update(
            calculate_frame_metrics(
                pred=pred,
                gt=gt,
                structure_measure=structure_measure,
                enhanced_alignment=enhanced_alignment,
            )
        )
        rows.append(row)

    if not rows:
        print(f"Skipping evaluation for {video_output_name}: no matching masks found")
        return

    artifacts_dir = artifacts_dir or output_mask_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    metrics_path = os.path.join(artifacts_dir, "metrics.csv")
    avg_path = os.path.join(artifacts_dir, "metrics_avg.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg = {}
    for key in [
        "structure_measure",
        "enhanced_alignment",
        "fmeasure",
        "sensitivity",
        "dice",
        "iou",
    ]:
        avg[key] = float(np.mean([row[key] for row in rows]))
    with open(avg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg.keys()))
        writer.writeheader()
        writer.writerow(avg)
    print(f"Saved metrics for {video_output_name} to {metrics_path}")


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def yolo_image_inference(
    predictor,
    yolo_model,
    base_video_dir,
    output_mask_dir,
    video_name,
    yolo_imgsz=640,
    yolo_conf=0.5,
    max_yolo_boxes_per_frame=1,
    artifacts_dir=None,
):
    """Run per-frame MedSAM2 using YOLO boxes as prompts."""
    video_dir = get_video_frame_dir(base_video_dir, video_name)
    video_output_name = get_video_name(base_video_dir, video_name)
    frame_names = get_frame_names(video_dir)
    pred_dir = os.path.join(output_mask_dir, video_output_name)
    os.makedirs(pred_dir, exist_ok=True)

    artifacts_dir = artifacts_dir or output_mask_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    bbox_rows = []
    for frame_name in frame_names:
        frame_path = resolve_frame_path(video_dir, frame_name)
        image = Image.open(frame_path).convert("RGB")
        image_np = np.array(image)
        height, width = image_np.shape[:2]
        boxes = get_yolo_boxes(
            yolo_model=yolo_model,
            frame_path=frame_path,
            yolo_imgsz=yolo_imgsz,
            yolo_conf=yolo_conf,
            max_boxes=max_yolo_boxes_per_frame,
        )

        pred_mask = np.zeros((height, width), dtype=np.uint8)
        if boxes:
            boxes_np = np.stack([box for box, _ in boxes], axis=0)
            predictor.set_image(image_np)
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_np,
                multimask_output=False,
            )
            pred_mask = combine_image_predictor_masks(masks)
            if pred_mask is None:
                pred_mask = np.zeros((height, width), dtype=np.uint8)

        for box_id, (box, _) in enumerate(boxes):
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

        output_path = os.path.join(pred_dir, f"{frame_name}.png")
        Image.fromarray((pred_mask > 0).astype(np.uint8) * 255).save(output_path)

    bbox_path = os.path.join(artifacts_dir, "bbox.csv")
    with open(bbox_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "box_id", "x_center", "y_center", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(bbox_rows)

    return video_output_name, frame_names


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def yolo_vos_inference(
    predictor,
    yolo_model,
    base_video_dir,
    output_mask_dir,
    video_name,
    image_predictor=None,
    score_thresh=0.0,
    save_palette_png=False,
    yolo_imgsz=640,
    yolo_conf=0.25,
    yolo_iou_match_thresh=0.3,
    max_yolo_boxes_per_frame=1,
    artifacts_dir=None,
    video_prompt_source="mask",
    video_prompt_stride=1,
):
    """Run video segmentation using YOLO-derived box or mask prompts."""
    video_dir = get_video_frame_dir(base_video_dir, video_name)
    video_output_name = get_video_name(base_video_dir, video_name)
    frame_names = get_frame_names(video_dir)
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    tracked_boxes = {}
    next_obj_id = 1
    seeded_frames = []
    bbox_rows = []
    for frame_idx, frame_name in enumerate(frame_names):
        if video_prompt_stride > 1 and frame_idx % video_prompt_stride != 0:
            continue
        frame_path = resolve_frame_path(video_dir, frame_name)
        boxes = get_yolo_boxes(
            yolo_model=yolo_model,
            frame_path=frame_path,
            yolo_imgsz=yolo_imgsz,
            yolo_conf=yolo_conf,
            max_boxes=max_yolo_boxes_per_frame,
        )
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
            masks, _, _ = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_np,
                multimask_output=False,
            )
            prompt_masks = [
                np.squeeze(np.asarray(mask)).astype(bool)
                for mask in masks
            ]

        assignments, next_obj_id = assign_box_ids(
            boxes=boxes,
            tracked_boxes=tracked_boxes,
            next_obj_id=next_obj_id,
            iou_thresh=yolo_iou_match_thresh,
        )
        seeded_frames.append(frame_idx)
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

    if not seeded_frames:
        raise RuntimeError(
            f"In {video_output_name}, YOLO found no boxes at conf={yolo_conf}. "
            "Lower --yolo_conf or check the YOLO checkpoint/input images."
        )

    os.makedirs(os.path.join(output_mask_dir, video_output_name), exist_ok=True)
    artifacts_dir = artifacts_dir or output_mask_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    bbox_path = os.path.join(artifacts_dir, "bbox.csv")
    with open(bbox_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "box_id", "x_center", "y_center", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(bbox_rows)

    video_segments = {}
    first_seeded_frame = min(seeded_frames)
    for reverse in (False, True):
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=first_seeded_frame,
            reverse=reverse,
        ):
            per_obj_output_mask = {
                out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            video_segments[out_frame_idx] = per_obj_output_mask

    output_palette = DAVIS_PALETTE
    for out_frame_idx in range(len(frame_names)):
        per_obj_output_mask = video_segments.get(out_frame_idx, {})
        if save_palette_png:
            save_palette_masks_to_dir(
                output_mask_dir=output_mask_dir,
                video_name=video_output_name,
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
                video_name=video_output_name,
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
                per_obj_png_file=False,
            )

    return video_output_name, frame_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help="MedSAM2  model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="external/MedSAM2/checkpoints/checkpoints/MedSAM2_latest.pt",
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
        default="external/YOLO_SAM2/YOLO_Checkpoints/polypgen_yolov8n.pt",
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
        "-o",
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for video-mode output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--inference_mode",
        choices=["image", "video"],
        default="image",
        help=(
            "image runs MedSAM2 independently on each YOLO box prompt; video uses "
            "SAM2 video propagation. image is the stronger default for seq2."
        ),
    )
    parser.add_argument(
        "--yolo_conf",
        type=float,
        default=0.5,
        help="YOLO confidence threshold for box prompts",
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
    args = parser.parse_args()

    image_predictor = None
    if args.inference_mode == "image":
        sam2_model = build_sam2(
            config_file=args.sam2_cfg,
            ckpt_path=args.sam2_checkpoint,
            apply_postprocessing=args.apply_postprocessing,
        )
        predictor = SAM2ImagePredictor(sam2_model)
    else:
        hydra_overrides_extra = ["++model.non_overlap_masks=true"]
        predictor = build_sam2_video_predictor(
            config_file=args.sam2_cfg,
            ckpt_path=args.sam2_checkpoint,
            apply_postprocessing=args.apply_postprocessing,
            hydra_overrides_extra=hydra_overrides_extra,
            vos_optimized=args.use_vos_optimized_video_predictor,
        )
        if args.video_prompt_source == "mask":
            sam2_model = build_sam2(
                config_file=args.sam2_cfg,
                ckpt_path=args.sam2_checkpoint,
                apply_postprocessing=args.apply_postprocessing,
            )
            image_predictor = SAM2ImagePredictor(sam2_model)
    yolo_model = YOLO(args.yolo_checkpoint)

    print(
        f"using YOLOv8 boxes from {args.yolo_checkpoint} as MedSAM2 prompts "
        f"with {args.inference_mode} inference"
    )
    # if a video list file is provided, read the video names from the file
    # (otherwise, we use all subdirectories in base_video_dir)
    if args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = list_video_names(args.base_video_dir)
    print(f"running inference on {len(video_names)} videos:\n{video_names}")
    single_video_run = len(video_names) == 1

    for n_video, video_name in enumerate(video_names):
        display_name = get_video_name(args.base_video_dir, video_name)
        print(f"\n{n_video + 1}/{len(video_names)} - running on {display_name}")
        artifacts_dir = (
            args.output_mask_dir
            if single_video_run
            else os.path.join(args.output_mask_dir, display_name)
        )
        if args.inference_mode == "image":
            video_output_name, frame_names = yolo_image_inference(
                predictor=predictor,
                yolo_model=yolo_model,
                base_video_dir=args.base_video_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                yolo_imgsz=args.yolo_imgsz,
                yolo_conf=args.yolo_conf,
                max_yolo_boxes_per_frame=args.max_yolo_boxes_per_frame,
                artifacts_dir=artifacts_dir,
            )
        else:
            video_output_name, frame_names = yolo_vos_inference(
                predictor=predictor,
                yolo_model=yolo_model,
                base_video_dir=args.base_video_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                image_predictor=image_predictor,
                score_thresh=args.score_thresh,
                save_palette_png=args.save_palette_png,
                yolo_imgsz=args.yolo_imgsz,
                yolo_conf=args.yolo_conf,
                yolo_iou_match_thresh=args.yolo_iou_match_thresh,
                max_yolo_boxes_per_frame=args.max_yolo_boxes_per_frame,
                artifacts_dir=artifacts_dir,
                video_prompt_source=args.video_prompt_source,
                video_prompt_stride=args.video_prompt_stride,
            )
        gt_mask_dir = args.gt_mask_dir or infer_gt_mask_dir(args.base_video_dir, video_name)
        evaluate_video(
            output_mask_dir=args.output_mask_dir,
            video_output_name=video_output_name,
            frame_names=frame_names,
            gt_mask_dir=gt_mask_dir,
            artifacts_dir=artifacts_dir,
        )

    print(
        f"completed inference on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )


if __name__ == "__main__":
    main()
