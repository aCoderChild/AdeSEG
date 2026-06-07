#!/usr/bin/env python3
"""
Shared evaluation module for video segmentation pipelines.
Provides metrics computation, bounding box matching, and overlay visualization.
Used by: YOLO_SAM2 and MedSAM2
"""

import os
import csv
import glob
import json
import shutil
import argparse
import re
import numpy as np
import cv2
from PIL import Image

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ============================================================================
# Metric Calculation Classes and Functions
# ============================================================================

class StructureMeasure(object):
    """Structure Measure (S-measure) for segmentation evaluation."""
    def __init__(self):
        self.eps = np.finfo(np.double).eps

    def _Object(self, GT, pred):
        x = np.mean(pred[GT])
        sigma_x = np.std(pred[GT])
        score = 2 * x / (x * x + 1 + sigma_x + self.eps)
        return score

    def _S_object(self, GT, pred):
        pred_fg = pred.copy()
        pred_fg[~GT] = 0
        O_FG = self._Object(GT, pred_fg)

        pred_bg = 1 - pred.copy()
        pred_bg[GT] = 0
        O_BG = self._Object(~GT, pred_bg)

        u = np.mean(GT)
        Q = u * O_FG + (1 - u) * O_BG
        return Q

    def _centroid(self, GT):
        rows, cols = GT.shape
        if np.sum(GT) == 0:
            X = round(cols / 2)
            Y = round(rows / 2)
        else:
            total = np.sum(GT)
            i = range(cols)
            j = range(rows)
            X = int(round(np.sum(np.sum(GT, axis=0) * i) / total)) + 1
            Y = int(round(np.sum(np.sum(GT, axis=1) * j) / total)) + 1
        return (X, Y)

    def _divide_GT(self, GT, X, Y):
        rows, cols = GT.shape
        area = rows * cols
        LT = GT[0:Y, 0:X]
        RT = GT[0:Y, X:cols]
        LB = GT[Y:rows, 0:X]
        RB = GT[Y:rows, X:cols]

        w1 = ((X) * (Y)) / area
        w2 = ((cols - X) * (Y)) / area
        w3 = ((X) * (rows - Y)) / area
        w4 = 1 - w1 - w2 - w3
        return (LT, RT, LB, RB, w1, w2, w3, w4)

    def _divide_pred(self, pred, X, Y):
        rows, cols = pred.shape
        LT = pred[0:Y, 0:X]
        RT = pred[0:Y, X:cols]
        LB = pred[Y:rows, 0:X]
        RB = pred[Y:rows, X:cols]
        return (LT, RT, LB, RB)

    def _ssim(self, GT, pred):
        rows, cols = GT.shape
        N = rows * cols
        x = np.mean(pred)
        y = np.mean(GT)
        sigma_x2 = np.sum((pred - x) ** 2) / (N - 1 + self.eps)
        sigma_y2 = np.sum((GT - y) ** 2) / (N - 1 + self.eps)
        sigma_xy = np.sum((pred - x) * (GT - y)) / (N - 1 + self.eps)
        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)
        if alpha != 0:
            Q = alpha / (beta + np.finfo(np.double).eps)
        elif alpha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

    def _S_region(self, GT, pred):
        X, Y = self._centroid(GT)
        GT_LT, GT_RT, GT_LB, GT_RB, w1, w2, w3, w4 = self._divide_GT(GT, X, Y)

        Pred_LT, Pred_RT, Pred_LB, Pred_RB = self._divide_pred(pred, X, Y)
        Q1 = self._ssim(GT_LT, Pred_LT)
        Q2 = self._ssim(GT_RT, Pred_RT)
        Q3 = self._ssim(GT_LB, Pred_LB)
        Q4 = self._ssim(GT_RB, Pred_RB)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        return Q

    def _minmiax_norm(self, X, ymin=0, ymax=1):
        X = (ymax - ymin) * (X - np.min(X)) / (np.max(X) - np.min(X)) + ymin
        return X

    def _prepare_data(self, GT_path, pred_path):
        pred = np.array(Image.open(pred_path)).astype(np.double)
        GT = np.array(Image.open(GT_path)).astype(np.bool_)

        if len(pred.shape) != 2:
            pred = 0.2989 * pred[:, :, 0] + 0.5870 * pred[:, :, 1] + 0.1140 * pred[:, :, 2]
        if len(GT.shape) != 2:
            GT = GT[:, :, 0]
        assert len(pred.shape) == 2, "Pred should be one channel!"
        assert len(GT.shape) == 2, "Ground Truth should be one channel!"
        if np.max(pred) == 255:
            pred = (pred / 255)
        pred = self._minmiax_norm(pred, 0, 1)
        return GT, pred

    def __call__(self, GT_path, pred_path):
        GT = GT_path.astype(np.bool_)
        pred = pred_path.astype(np.double)
        meanGT = np.mean(GT)
        if meanGT == 0:
            x = np.mean(pred)
            Q = 1.0 - x
        elif meanGT == 1:
            x = np.mean(pred)
            Q = x
        else:
            alpha = 0.5
            Q = alpha * self._S_object(GT, pred) + (1 - alpha) * self._S_region(GT, pred)
            if Q < 0:
                Q = 0
        return Q


class EnhancedAlignmentMeasure:
    """Enhanced Alignment Measure (E-measure) for segmentation evaluation."""
    def __init__(self):
        self.eps = np.finfo(np.double).eps

    def _prepare_data(self, GT_path, pred_path):
        pred = np.array(Image.open(pred_path)).astype(np.bool_)
        GT = np.array(Image.open(GT_path)).astype(np.bool_)
        if len(pred.shape) != 2:
            pred = pred[:, :, 0]
        if len(GT.shape) != 2:
            GT = GT[:, :, 0]
        assert len(pred.shape) == 2, "Pred should be one channel!"
        assert len(GT.shape) == 2, "Ground Truth should be one channel!"
        return GT, pred

    def _EnhancedAlignmnetTerm(self, align_Matrix):
        enhanced = ((align_Matrix + 1) ** 2) / 4
        return enhanced

    def _AlignmentTerm(self, dGT, dpred):
        mean_dpred = np.mean(dpred)
        mean_dGT = np.mean(dGT)
        align_dpred = dpred - mean_dpred
        align_dGT = dGT - mean_dGT
        align_matrix = 2 * (align_dGT * align_dpred) / (align_dGT ** 2 + align_dpred ** 2 + self.eps)
        return align_matrix

    def __call__(self, GT_path, pred_path):
        GT = GT_path.astype(np.bool_)
        pred = pred_path.astype(np.double)
        dGT, dpred = GT.astype(np.float64), pred.astype(np.float64)
        if np.sum(GT) == 0:
            enhanced_matrix = 1 - dpred
        elif np.sum(~GT) == 0:
            enhanced_matrix = dpred
        else:
            align_matrix = self._AlignmentTerm(dGT, dpred)
            enhanced_matrix = self._EnhancedAlignmnetTerm(align_matrix)
        rows, cols = GT.shape
        score = np.sum(enhanced_matrix) / (rows * cols - 1 + self.eps)
        return score


def calculate_sensitivity_specificity(pred, gt):
    """Calculate sensitivity and specificity metrics."""
    tp = np.sum((pred == 1) & (gt == 1))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    sensitivity = 1.0 if tp + fn == 0 else tp / (tp + fn)
    specificity = 1.0 if tn + fp == 0 else tn / (tn + fp)
    return sensitivity, specificity


def calculate_dice(pred, gt):
    """Calculate Dice coefficient."""
    intersection = np.sum(pred * gt)
    denominator = np.sum(pred) + np.sum(gt)
    return 1.0 if denominator == 0 else 2 * intersection / denominator


def calculate_iou(pred, gt):
    """Calculate Intersection over Union (IoU)."""
    intersection = np.sum(pred * gt)
    union = np.sum(np.logical_or(pred, gt))
    return 1.0 if union == 0 else intersection / union


def calculate_fmeasure(pred, gt, beta=1):
    """Calculate F-measure."""
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    if tp + fp == 0 and tp + fn == 0:
        return 1.0
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def calculate_temporal_iou(current_mask, previous_mask):
    """IoU between adjacent predicted masks; 1.0 when both masks are empty."""
    current = current_mask.astype(bool)
    previous = previous_mask.astype(bool)
    union = np.logical_or(current, previous).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(current, previous).sum()
    return intersection / union


def calculate_centroid(mask):
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    return float(np.mean(coords[1])), float(np.mean(coords[0]))


def mean_numeric(values):
    numeric_values = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(value):
            continue
        numeric_values.append(value)
    if not numeric_values:
        return ""
    return float(np.mean(numeric_values))


def intersectionAndUnion(imPred, imLab, numClass):
    """Compute intersection and union for each class."""
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    area_sum = area_pred + area_lab
    return (area_intersection, area_union, area_sum)


# ============================================================================
# Bounding Box and Overlay Functions
# ============================================================================


def get_numeric_sort_key(name):
    """Natural sort key for frame names such as 2.jpg before 10.jpg."""
    basename = os.path.basename(str(name))
    stem = os.path.splitext(basename)[0]
    parts = re.split(r"(\d+)", stem)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def box_iou(box_a, box_b):
    """Compute IoU between two bounding boxes in (x1, y1, x2, y2) format."""
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


def compute_bbox_iou(box1, box2):
    """Compatibility wrapper for bbox IoU calculation."""
    return box_iou(box1, box2)


def match_bboxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Match predicted boxes to ground truth boxes using IoU."""
    matched = []
    used_pred = set()
    used_gt = set()
    pairs = []

    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            pairs.append((compute_bbox_iou(pred_box, gt_box), pred_idx, gt_idx))

    for iou, pred_idx, gt_idx in sorted(pairs, reverse=True):
        if iou < iou_threshold:
            break
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        matched.append((pred_boxes[pred_idx], gt_boxes[gt_idx], iou))
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)

    unmatched_gt = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in used_gt]
    return matched, unmatched_gt


def create_bbox_overlay(image_array, pred_bboxes, gt_bboxes):
    """Create overlay with predicted boxes in red and GT in green."""
    overlay = image_array.copy()
    for x1, y1, x2, y2 in gt_bboxes:
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    for x1, y1, x2, y2 in pred_bboxes:
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return overlay


def create_mask_overlay(image_array, pred_mask, gt_mask):
    """Create overlay with predicted mask in red and GT in green."""
    overlay = image_array.copy().astype(np.float32)
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + gt_mask * 150, 0, 255)
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] - gt_mask * 100, 0, 255)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] - gt_mask * 100, 0, 255)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + pred_mask * 150, 0, 255)
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] - pred_mask * 100, 0, 255)
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] - pred_mask * 100, 0, 255)
    return overlay.astype(np.uint8)


def get_bbox_from_mask(mask_array):
    """Extract bounding box from binary mask. Returns (x1, y1, x2, y2) or None if empty."""
    binary_mask = (mask_array > 0).astype(np.uint8)
    coords = np.where(binary_mask > 0)
    
    if len(coords[0]) == 0:
        return None
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    return int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1


def resolve_existing_path(base_without_ext, extensions):
    for ext in extensions:
        path = f"{base_without_ext}{ext}"
        if os.path.exists(path):
            return path
    return None


def save_bbox_iou_scores(bbox_iou_scores, output_dir, verbose=True):
    """Return mean bbox IoU without writing per-sequence bbox CSV files."""
    if not bbox_iou_scores:
        return None

    mean_iou = np.mean([s['iou'] for s in bbox_iou_scores])

    if verbose:
        print(f"  ✓ BBox IoU: {mean_iou:.4f}")
    
    return mean_iou


def load_gt_bboxes(seq_gt_bboxes_file):
    gt_bboxes_dict = {}
    with open(seq_gt_bboxes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            frame_idx = str(int(row['frame_idx']))
            bbox = (int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2']))
            gt_bboxes_dict.setdefault(frame_idx, []).append(bbox)
    return gt_bboxes_dict


def load_pred_yolo_bboxes(pred_bbox_file, seq_images_dir):
    pred_bboxes_dict = {}
    if not os.path.isfile(pred_bbox_file):
        return pred_bboxes_dict

    with open(pred_bbox_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            frame_name = str(int(row['frame']))
            image_path = resolve_existing_path(
                os.path.join(seq_images_dir, frame_name),
                [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
            )
            if image_path is None:
                continue

            image = cv2.imread(image_path)
            if image is None:
                continue
            image_h, image_w = image.shape[:2]

            x_center = float(row['x_center']) * image_w
            y_center = float(row['y_center']) * image_h
            width = float(row['width']) * image_w
            height = float(row['height']) * image_h
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            bbox = (
                int(round(max(0, x1))),
                int(round(max(0, y1))),
                int(round(min(image_w, x2))),
                int(round(min(image_h, y2))),
            )
            pred_bboxes_dict.setdefault(frame_name, []).append(bbox)

    return pred_bboxes_dict


def get_sequence_paths(seq_num, data_root, bbox_root, output_root, method_name):
    seq_images_dir = os.path.join(data_root, f"seq{seq_num}", "images")
    seq_masks_dir = os.path.join(data_root, f"seq{seq_num}", "masks")
    seq_gt_bboxes_file = os.path.join(bbox_root, f"seq{seq_num}", "bboxes.csv")
    pred_bbox_file = os.path.join(output_root, method_name, "bbox", f"seq{seq_num}.csv")
    plural_pred_masks_dir = os.path.join(
        output_root,
        method_name,
        "masks",
        f"seq{seq_num}",
        "predicted",
    )
    singular_pred_masks_dir = os.path.join(
        output_root,
        method_name,
        "mask",
        f"seq{seq_num}",
        "predicted",
    )
    pred_masks_dir = (
        plural_pred_masks_dir
        if os.path.isdir(plural_pred_masks_dir)
        else singular_pred_masks_dir
    )
    return seq_images_dir, seq_masks_dir, seq_gt_bboxes_file, pred_bbox_file, pred_masks_dir


def evaluate_bbox_sequence(seq_num, data_root="data/polypgen", bbox_root="data/bbox",
                           output_root="outputs", method_name="YOLO_SAM2",
                           eval_output_root=None, verbose=True):
    seq_images_dir, _, seq_gt_bboxes_file, pred_bbox_file, _ = get_sequence_paths(
        seq_num, data_root, bbox_root, output_root, method_name
    )

    method_output_root = eval_output_root or os.path.join(output_root, method_name, "eval")
    bbox_output_root = os.path.join(method_output_root, "bbox")
    os.makedirs(bbox_output_root, exist_ok=True)

    if not os.path.isfile(seq_gt_bboxes_file):
        if verbose:
            print(f"  ✗ Ground truth bboxes not found")
        return None

    try:
        gt_bboxes_dict = load_gt_bboxes(seq_gt_bboxes_file)
    except Exception as e:
        if verbose:
            print(f"  ✗ Error loading GT bboxes: {e}")
        return None
    if not gt_bboxes_dict:
        if verbose:
            print("  ⚠ No ground-truth boxes for this sequence")
        return None

    pred_yolo_bboxes_dict = load_pred_yolo_bboxes(pred_bbox_file, seq_images_dir)
    if not pred_yolo_bboxes_dict and verbose:
        print(f"  ⚠ No predicted YOLO boxes; scoring GT frames as IoU=0")
    
    bbox_iou_scores = []
    bbox_frame_names = sorted(
        gt_bboxes_dict,
        key=get_numeric_sort_key,
    )
    for frame_name in bbox_frame_names:
        pred_bboxes = pred_yolo_bboxes_dict.get(frame_name, [])
        gt_bboxes = gt_bboxes_dict.get(frame_name, [])
        if not gt_bboxes:
            continue
        if not pred_bboxes:
            bbox_iou_scores.append({'frame': frame_name, 'iou': 0.0})
            continue

        matched, unmatched_gt = match_bboxes(pred_bboxes, gt_bboxes, iou_threshold=0.0)
        for _, _, iou in matched:
            bbox_iou_scores.append({'frame': frame_name, 'iou': iou})
        for _ in unmatched_gt:
            bbox_iou_scores.append({'frame': frame_name, 'iou': 0.0})

    if bbox_iou_scores:
        seq_csv_path = os.path.join(bbox_output_root, f"seq{seq_num}.csv")
        with open(seq_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'iou'])
            writer.writeheader()
            writer.writerows(bbox_iou_scores)

    mean_iou = save_bbox_iou_scores(bbox_iou_scores, None, verbose=verbose)
    return mean_iou


def evaluate_mask_sequence(seq_num, data_root="data/polypgen", bbox_root="data/bbox",
                           output_root="outputs", method_name="YOLO_SAM2",
                           eval_output_root=None, save_bbox_overlays=True, verbose=True):
    seq_images_dir, seq_masks_dir, seq_gt_bboxes_file, pred_bbox_file, pred_masks_dir = (
        get_sequence_paths(seq_num, data_root, bbox_root, output_root, method_name)
    )
    
    method_output_root = eval_output_root or os.path.join(output_root, method_name)
    masks_output_dir = os.path.join(method_output_root, "mask", f"seq{seq_num}")
    overlays_output_dir = os.path.join(masks_output_dir, "overlays")
    bbox_overlays_output_dir = None
    if save_bbox_overlays:
        bbox_overlays_output_dir = os.path.join(
            method_output_root,
            "bbox",
            f"seq{seq_num}",
            "overlays",
        )
    
    output_dirs = [masks_output_dir, overlays_output_dir]
    if bbox_overlays_output_dir is not None:
        output_dirs.append(bbox_overlays_output_dir)
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
    
    if not os.path.isdir(pred_masks_dir):
        if verbose:
            print(f"  ✗ Predicted masks directory not found")
        return None
    
    if save_bbox_overlays and os.path.isfile(seq_gt_bboxes_file):
        try:
            gt_bboxes_dict = load_gt_bboxes(seq_gt_bboxes_file)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Error loading GT bboxes for overlays: {e}")
            gt_bboxes_dict = {}
    elif save_bbox_overlays:
        if verbose:
            print("  ⚠ Ground truth bboxes not found; mask metrics will still run")
        gt_bboxes_dict = {}
    else:
        gt_bboxes_dict = {}

    pred_yolo_bboxes_dict = (
        load_pred_yolo_bboxes(pred_bbox_file, seq_images_dir)
        if save_bbox_overlays
        else {}
    )
    
    gt_mask_files = sorted(
        glob.glob(os.path.join(seq_masks_dir, "*.png"))
        + glob.glob(os.path.join(seq_masks_dir, "*.jpg"))
        + glob.glob(os.path.join(seq_masks_dir, "*.jpeg"))
        + glob.glob(os.path.join(seq_masks_dir, "*.PNG"))
        + glob.glob(os.path.join(seq_masks_dir, "*.JPG"))
        + glob.glob(os.path.join(seq_masks_dir, "*.JPEG")),
        key=get_numeric_sort_key,
    )
    mask_metric_rows = []
    s_measure = StructureMeasure()
    e_measure = EnhancedAlignmentMeasure()
    previous_pred_mask = None
    previous_area_frac = None
    previous_centroid = None
    
    missing_predictions = 0
    for gt_mask_path in gt_mask_files:
        frame_name = os.path.splitext(os.path.basename(gt_mask_path))[0]
        image_path = resolve_existing_path(
            os.path.join(seq_images_dir, frame_name),
            [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
        )
        pred_mask_path = resolve_existing_path(
            os.path.join(pred_masks_dir, frame_name),
            [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"],
        )

        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_mask = (gt_mask > 127).astype(np.uint8)
        if pred_mask_path is None:
            missing_predictions += 1
            pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
        else:
            pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
            if pred_mask is None:
                missing_predictions += 1
                pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
            else:
                pred_mask = (pred_mask > 127).astype(np.uint8)
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(
                        pred_mask,
                        (gt_mask.shape[1], gt_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

        sensitivity, specificity = calculate_sensitivity_specificity(pred_mask, gt_mask)
        pred_area_frac = float(np.mean(pred_mask))
        temporal_iou_prev = None
        area_change_abs_prev = None
        centroid_shift_norm_prev = None
        current_centroid = calculate_centroid(pred_mask)
        if previous_pred_mask is not None:
            temporal_iou_prev = calculate_temporal_iou(pred_mask, previous_pred_mask)
            area_change_abs_prev = abs(pred_area_frac - previous_area_frac)
            if current_centroid is not None and previous_centroid is not None:
                dx = current_centroid[0] - previous_centroid[0]
                dy = current_centroid[1] - previous_centroid[1]
                diagonal = float(np.hypot(*pred_mask.shape))
                centroid_shift_norm_prev = float(np.hypot(dx, dy) / diagonal)

        mask_metric_rows.append({
            'frame': frame_name,
            'dice': calculate_dice(pred_mask, gt_mask),
            'iou': calculate_iou(pred_mask, gt_mask),
            'fmeasure': calculate_fmeasure(pred_mask, gt_mask),
            'sensitivity': sensitivity,
            'specificity': specificity,
            's_measure': s_measure(gt_mask, pred_mask),
            'e_measure': e_measure(gt_mask, pred_mask),
            'pred_area_frac': pred_area_frac,
            'temporal_iou_prev': temporal_iou_prev,
            'area_change_abs_prev': area_change_abs_prev,
            'centroid_shift_norm_prev': centroid_shift_norm_prev,
        })
        previous_pred_mask = pred_mask
        previous_area_frac = pred_area_frac
        previous_centroid = current_centroid
        
        if image_path is not None:
            image = cv2.imread(image_path)
            if image is not None:
                pred_bboxes = pred_yolo_bboxes_dict.get(frame_name, [])
                gt_bboxes = gt_bboxes_dict.get(frame_name, [])

                mask_overlay = create_mask_overlay(image, pred_mask, gt_mask)
                cv2.imwrite(os.path.join(overlays_output_dir, f"{frame_name}_mask_overlay.png"), mask_overlay)

                if bbox_overlays_output_dir is not None:
                    bbox_overlay = create_bbox_overlay(image, pred_bboxes, gt_bboxes)
                    cv2.imwrite(
                        os.path.join(bbox_overlays_output_dir, f"{frame_name}_bbox_overlay.png"),
                        bbox_overlay,
                    )

                if save_bbox_overlays:
                    combined_overlay = create_mask_overlay(image, pred_mask, gt_mask)
                    for x1, y1, x2, y2 in gt_bboxes:
                        cv2.rectangle(combined_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for x1, y1, x2, y2 in pred_bboxes:
                        cv2.rectangle(combined_overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(overlays_output_dir, f"{frame_name}_combined_overlay.png"), combined_overlay)
    
    # Save scores
    if mask_metric_rows:
        metrics_csv = os.path.join(masks_output_dir, "metrics.csv")
        metric_fields = [
            'frame', 'dice', 'iou', 'fmeasure', 'sensitivity', 'specificity',
            's_measure', 'e_measure', 'pred_area_frac', 'temporal_iou_prev',
            'area_change_abs_prev', 'centroid_shift_norm_prev'
        ]
        with open(metrics_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writeheader()
            writer.writerows(mask_metric_rows)

        metrics_avg_csv = os.path.join(masks_output_dir, "metrics_avg.csv")
        with open(metrics_avg_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields[1:])
            writer.writeheader()
            writer.writerow({
                field: mean_numeric([row[field] for row in mask_metric_rows])
                for field in metric_fields[1:]
            })

    if verbose:
        if missing_predictions:
            print(f"  ⚠ Missing predicted masks counted as empty: {missing_predictions}")
        print(f"  ✓ Overlays saved to: {overlays_output_dir}")
    
    return {
        'masks_dir': masks_output_dir,
        'overlays_dir': overlays_output_dir,
        'num_masks': len(mask_metric_rows),
    }


def aggregate_mask_metrics(output_root, method_name, num_sequences=23, sequence_ids=None, verbose=True):
    """
    Aggregate mask metrics from all sequences into summary files.

    Creates:
    - metrics_per_sequence.csv  — one row per found sequence
    - metrics_avg.csv           — mean across found sequences (backward-compatible schema)
    - metrics_stats.csv         — mean + std + n_sequences per metric
    - metrics_coverage.json     — lists found and missing sequence ids

    Returns a dict with keys 'found', 'missing', 'metric_values' on success, False on error.
    """
    masks_root = (
        os.path.join(output_root, "mask")
        if not method_name
        else os.path.join(output_root, method_name, "mask")
    )

    if not os.path.isdir(masks_root):
        if verbose:
            print(f"Masks root not found: {masks_root}")
        return False

    per_sequence_metrics = []
    metric_values = {}
    seq_ids = list(sequence_ids or range(1, num_sequences + 1))
    found_seqs = []
    missing_seqs = []

    for seq_num in seq_ids:
        seq_metrics_file = os.path.join(masks_root, f"seq{seq_num}", "metrics_avg.csv")

        if not os.path.isfile(seq_metrics_file):
            missing_seqs.append(seq_num)
            if verbose:
                print(f"  ⚠ metrics_avg.csv not found for seq{seq_num} (will be excluded from mean)")
            continue

        try:
            with open(seq_metrics_file, 'r') as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    row_dict = {'seq': seq_num}
                    for key, val in row.items():
                        key = key.strip()
                        try:
                            row_dict[key] = float(val) if val != "" else ""
                        except ValueError:
                            row_dict[key] = val
                    per_sequence_metrics.append(row_dict)
                    for key, value in row_dict.items():
                        if key == 'seq' or not isinstance(value, (int, float)) or np.isnan(value):
                            continue
                        metric_values.setdefault(key, []).append(value)
            found_seqs.append(seq_num)
        except Exception as e:
            missing_seqs.append(seq_num)
            if verbose:
                print(f"  ✗ Error reading metrics for seq{seq_num}: {e}")
            continue

    # Coverage report — written first so it's available even if later steps fail
    coverage = {
        "n_sequences_requested": len(seq_ids),
        "n_sequences_evaluated": len(found_seqs),
        "n_sequences_missing": len(missing_seqs),
        "found_sequences": sorted(found_seqs),
        "missing_sequences": sorted(missing_seqs),
    }
    coverage_file = os.path.join(masks_root, "metrics_coverage.json")
    try:
        with open(coverage_file, 'w') as f:
            json.dump(coverage, f, indent=2)
        if verbose:
            print(
                f"  ✓ Coverage: {len(found_seqs)}/{len(seq_ids)} sequences evaluated"
                + (f" (missing: {sorted(missing_seqs)})" if missing_seqs else "")
            )
    except Exception as e:
        if verbose:
            print(f"  ✗ Error writing coverage file: {e}")

    # Per-sequence CSV
    if per_sequence_metrics:
        per_seq_file = os.path.join(masks_root, "metrics_per_sequence.csv")
        fieldnames = ['seq'] + [k for k in per_sequence_metrics[0].keys() if k != 'seq']
        try:
            with open(per_seq_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_sequence_metrics)
            if verbose:
                print(f"  ✓ Per-sequence metrics: {per_seq_file}")
        except Exception as e:
            if verbose:
                print(f"  ✗ Error writing per-sequence metrics: {e}")
            return False

    # Mean-only CSV (backward-compatible schema)
    if metric_values:
        avg_metrics = {k: float(np.mean(v)) for k, v in metric_values.items()}
        avg_file = os.path.join(masks_root, "metrics_avg.csv")
        fieldnames = [k for k in avg_metrics.keys() if k != 'seq']
        try:
            with open(avg_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({k: avg_metrics[k] for k in fieldnames})
            if verbose:
                print(f"  ✓ Average metrics: {avg_file}")
        except Exception as e:
            if verbose:
                print(f"  ✗ Error writing average metrics: {e}")
            return False

        # Stats CSV — mean + std + n_sequences (ddof=1 where n>=2, else 0)
        stats_rows = []
        for k, vals in metric_values.items():
            n = len(vals)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
            stats_rows.append({"metric": k, "mean": mean_v, "std": std_v, "n_sequences": n})
        stats_file = os.path.join(masks_root, "metrics_stats.csv")
        try:
            with open(stats_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n_sequences"])
                writer.writeheader()
                writer.writerows(stats_rows)
            if verbose:
                print(f"  ✓ Metrics stats (mean+std): {stats_file}")
        except Exception as e:
            if verbose:
                print(f"  ✗ Error writing metrics stats: {e}")

    return {"found": found_seqs, "missing": missing_seqs, "metric_values": metric_values}


def compare_methods_paired(
    method_a,
    method_b,
    output_root="outputs",
    output_path=None,
    sequence_ids=None,
    num_sequences=23,
    verbose=True,
):
    """
    Compare two methods on their matched (common) sequences using a paired Wilcoxon
    signed-rank test per metric.

    Reads:
      {output_root}/{method}/eval/mask/seq{N}/metrics_avg.csv

    Writes (if output_path given):
      A CSV with columns: metric, {a}_mean, {a}_std, {b}_mean, {b}_std,
                          mean_delta (a-b), p_value, n_matched

    Returns the comparison dict keyed by metric name, or None on failure.
    Requires scipy; prints a warning and skips p-values when unavailable.
    """
    seq_ids = list(sequence_ids or range(1, num_sequences + 1))

    def _load_seq_metrics(method):
        mask_root = os.path.join(output_root, method, "eval", "mask")
        result = {}
        for seq_num in seq_ids:
            f = os.path.join(mask_root, f"seq{seq_num}", "metrics_avg.csv")
            if not os.path.isfile(f):
                continue
            try:
                with open(f, 'r') as fh:
                    reader = csv.DictReader(fh, skipinitialspace=True)
                    for row in reader:
                        row_clean = {}
                        for k, v in row.items():
                            k = k.strip()
                            try:
                                row_clean[k] = float(v)
                            except (ValueError, TypeError):
                                pass
                        result[seq_num] = row_clean
                        break
            except Exception:
                pass
        return result

    metrics_a = _load_seq_metrics(method_a)
    metrics_b = _load_seq_metrics(method_b)

    matched_seqs = sorted(set(metrics_a) & set(metrics_b))
    if not matched_seqs:
        if verbose:
            print(f"  ✗ No matched sequences between {method_a} and {method_b}")
        return None

    if verbose:
        only_a = sorted(set(metrics_a) - set(metrics_b))
        only_b = sorted(set(metrics_b) - set(metrics_a))
        print(f"  Matched sequences ({len(matched_seqs)}): {matched_seqs}")
        if only_a:
            print(f"  Only in {method_a}: {only_a}")
        if only_b:
            print(f"  Only in {method_b}: {only_b}")

    metric_names = [k for k in next(iter(metrics_a.values())).keys()]
    comparison = {}
    for metric in metric_names:
        paired = [
            (metrics_a[s][metric], metrics_b[s][metric])
            for s in matched_seqs
            if metric in metrics_a[s] and metric in metrics_b[s]
        ]
        if not paired:
            continue
        arr_a = np.array([p[0] for p in paired])
        arr_b = np.array([p[1] for p in paired])
        n = len(arr_a)
        mean_a = float(np.mean(arr_a))
        std_a = float(np.std(arr_a, ddof=1)) if n >= 2 else 0.0
        mean_b = float(np.mean(arr_b))
        std_b = float(np.std(arr_b, ddof=1)) if n >= 2 else 0.0
        delta = mean_a - mean_b

        p_value = None
        if _HAS_SCIPY and n >= 2 and not np.all(arr_a == arr_b):
            try:
                result = _scipy_stats.wilcoxon(arr_a, arr_b, alternative="two-sided")
                p_value = float(result.pvalue)
            except Exception:
                pass

        comparison[metric] = {
            f"{method_a}_mean": mean_a,
            f"{method_a}_std": std_a,
            f"{method_b}_mean": mean_b,
            f"{method_b}_std": std_b,
            "mean_delta": delta,
            "p_value": p_value,
            "n_matched": n,
        }

    if verbose:
        header = f"{'metric':<30} {method_a+' mean':>12} {method_b+' mean':>12} {'delta':>8} {'p-value':>10}"
        print(f"\n  Paired comparison ({method_a} vs {method_b}) on {len(matched_seqs)} matched sequences:")
        print("  " + header)
        print("  " + "-" * len(header))
        for metric, row in comparison.items():
            pv = f"{row['p_value']:.4f}" if row['p_value'] is not None else "n/a (no scipy)"
            print(
                f"  {metric:<30} {row[method_a+'_mean']:>12.4f} {row[method_b+'_mean']:>12.4f}"
                f" {row['mean_delta']:>+8.4f} {pv:>10}"
            )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fieldnames = (
            ["metric"]
            + [f"{method_a}_mean", f"{method_a}_std"]
            + [f"{method_b}_mean", f"{method_b}_std"]
            + ["mean_delta", "p_value", "n_matched"]
        )
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for metric, row in comparison.items():
                    writer.writerow({"metric": metric, **row})
            if verbose:
                print(f"\n  ✓ Paired comparison written: {output_path}")
        except Exception as e:
            if verbose:
                print(f"  ✗ Error writing comparison: {e}")

    return comparison


def aggregate_bbox_metrics(output_root, method_name, num_sequences=23, sequence_ids=None, verbose=True):
    """
    Aggregate bounding box metrics from all sequences into summary files.
    
    Creates:
    - {output_root}/{method}/boxes/iou_per_sequence.csv (aggregated mean_iou from each sequence)
    - {output_root}/{method}/boxes/iou_avg.csv (average IoU across all sequences)
    
    Args:
        output_root: Root output directory
        method_name: Name of method (YOLO_SAM2, MedSAM2, etc.)
        num_sequences: Number of sequences to aggregate (default 23)
        verbose: Print progress messages
    """
    boxes_root = (
        os.path.join(output_root, "boxes")
        if not method_name
        else os.path.join(output_root, method_name, "boxes")
    )
    
    if not os.path.isdir(boxes_root):
        if verbose:
            print(f"Boxes root not found: {boxes_root}")
        return False
    
    per_sequence_ious = []
    all_ious = []
    
    # Read mean_iou.csv from each sequence
    seq_ids = sequence_ids or range(1, num_sequences + 1)
    for seq_num in seq_ids:
        seq_iou_file = os.path.join(boxes_root, f"seq{seq_num}", "mean_iou.csv")
        
        if not os.path.isfile(seq_iou_file):
            if verbose:
                print(f"  ⚠ mean_iou.csv not found for seq{seq_num}")
            continue
        
        try:
            with open(seq_iou_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        iou_val = float(row.get('mean_iou', 0))
                        per_sequence_ious.append({'seq': seq_num, 'mean_iou': iou_val})
                        all_ious.append(iou_val)
                    except ValueError:
                        pass
        except Exception as e:
            if verbose:
                print(f"  ⚠ Error reading IoU for seq{seq_num}: {e}")
            continue
    
    # Write per-sequence IoU aggregation
    if per_sequence_ious:
        per_seq_file = os.path.join(boxes_root, "iou_per_sequence.csv")
        
        try:
            with open(per_seq_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['seq', 'mean_iou'])
                writer.writeheader()
                writer.writerows(per_sequence_ious)
            
            if verbose:
                print(f"  ✓ Per-sequence IoU aggregated: {per_seq_file}")
        except Exception as e:
            if verbose:
                print(f"  ✗ Error writing per-sequence IoU: {e}")
            return False
    
    # Compute and write average IoU
    if all_ious:
        avg_iou = np.mean(all_ious)
        
        avg_file = os.path.join(boxes_root, "iou_avg.csv")
        
        try:
            with open(avg_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iou_avg'])
                writer.writerow([avg_iou])
            
            if verbose:
                print(f"  ✓ Average IoU computed: {avg_file}")
        except Exception as e:
            if verbose:
                print(f"  ✗ Error writing average IoU: {e}")
            return False
    
    return True


def evaluate_method(method_name, data_root="data/polypgen", bbox_root="data/bbox",
                    outputs_root="outputs", num_sequences=23, eval_type="both",
                    sequence_ids=None, verbose=True):
    method_root = os.path.join(outputs_root, method_name)
    eval_root = os.path.join(method_root, "eval")

    if verbose:
        print(f"\nEvaluating {method_name}")
        print(f"  Data root: {data_root}")
        print(f"  Eval output: {eval_root}")

    per_sequence_ious = []

    seq_ids = sequence_ids or list(range(1, num_sequences + 1))
    for seq_num in seq_ids:
        if verbose:
            print(f"seq{seq_num}:")

        if eval_type in ("bbox", "both"):
            mean_iou = evaluate_bbox_sequence(
                seq_num=seq_num,
                data_root=data_root,
                bbox_root=bbox_root,
                output_root=outputs_root,
                method_name=method_name,
                eval_output_root=eval_root,
                verbose=verbose,
            )
            if mean_iou is not None:
                per_sequence_ious.append({'seq': seq_num, 'mean_iou': mean_iou})

        if eval_type in ("mask", "both"):
            evaluate_mask_sequence(
                seq_num=seq_num,
                data_root=data_root,
                bbox_root=bbox_root,
                output_root=outputs_root,
                method_name=method_name,
                eval_output_root=eval_root,
                save_bbox_overlays=eval_type == "both",
                verbose=verbose,
            )

    if eval_type in ("mask", "both"):
        aggregate_mask_metrics(
            eval_root,
            "",
            num_sequences=num_sequences,
            sequence_ids=seq_ids,
            verbose=verbose,
        )

    if eval_type in ("bbox", "both") and per_sequence_ious:
        bbox_root_out = os.path.join(eval_root, "bbox")
        os.makedirs(bbox_root_out, exist_ok=True)
        per_seq_file = os.path.join(bbox_root_out, "iou_per_sequence.csv")
        with open(per_seq_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['seq', 'mean_iou'])
            writer.writeheader()
            writer.writerows(per_sequence_ious)

        avg_file = os.path.join(bbox_root_out, "iou_avg.csv")
        with open(avg_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iou_avg'])
            writer.writerow([np.mean([row['mean_iou'] for row in per_sequence_ious])])

        if verbose:
            print(f"  ✓ Per-sequence IoU aggregated: {per_seq_file}")
            print(f"  ✓ Average IoU computed: {avg_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation masks and bbox IoU for method outputs."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["YOLO_SAM2", "MedSAM2"],
        help="Method names under outputs/ to evaluate.",
    )
    parser.add_argument(
        "--data_root",
        default="data/polypgen",
        help="Input dataset root containing seq*/images and seq*/masks.",
    )
    parser.add_argument(
        "--bbox_root",
        default="data/bbox",
        help="Ground-truth bbox root containing seq*/bboxes.csv.",
    )
    parser.add_argument(
        "--outputs_root",
        default="outputs",
        help="Root containing method output folders.",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=23,
        help="Number of sequences to evaluate.",
    )
    parser.add_argument(
        "--seqs",
        nargs="+",
        type=int,
        help="Optional explicit sequence ids to evaluate.",
    )
    parser.add_argument(
        "--eval_type",
        choices=["bbox", "mask", "both"],
        default="both",
        help="Run bbox IoU only, final mask evaluation only, or both.",
    )
    args = parser.parse_args()

    for method_name in args.methods:
        evaluate_method(
            method_name=method_name,
            data_root=args.data_root,
            bbox_root=args.bbox_root,
            outputs_root=args.outputs_root,
            num_sequences=args.num_sequences,
            eval_type=args.eval_type,
            sequence_ids=args.seqs,
            verbose=True,
        )


if __name__ == "__main__":
    main()

# python3 scripts/utils/eval_metrics.py \
#   --methods YOLO_SAM2 MedSAM2 \
#   --data_root data/polypgen \
#   --outputs_root outputs
