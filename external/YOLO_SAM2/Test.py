import argparse
import os
import shutil
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import csv
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class StructureMeasure(object):
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


def intersectionAndUnion(imPred, imLab, numClass):
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    area_sum = area_pred + area_lab
    return (area_intersection, area_union, area_sum)


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
    dice = 2 * intersection / (np.sum(pred) + np.sum(gt) + 1e-7)
    return dice


def calculate_iou(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(np.logical_or(pred, gt))
    iou = intersection / (union + 1e-7)
    return iou


def calculate_fmeasure(pred, gt, beta=1):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    beta_sq = beta ** 2
    fmeasure = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-7)
    return fmeasure

# device selection
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

sam2_checkpoint = "external/YOLO_SAM2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_add",
        default="external/YOLO_SAM2/YOLO_Checkpoints/polypgen_yolov8n.pt",
        help="YOLO pre-trained model address.",
    )
    parser.add_argument(
        "--seq_num",
        type=int,
        default="2",
        help="Sequence number to process (e.g. 1 for seq1). If omitted, all sequences are processed.",
    )
    args = parser.parse_args()

    yolo_model = YOLO(args.checkpoint_add)

    outputs_dir = f"external/YOLO_SAM2/polypgen_vid_seq/seq{args.seq_num}"
    os.makedirs(outputs_dir, exist_ok=True)

    structure_measure = StructureMeasure()
    enhanced_alignment = EnhancedAlignmentMeasure()

    base_dir = "external/YOLO_SAM2/polypgen_vid_seq"
    if args.seq_num is not None:
        seq_path = os.path.join(base_dir, f"seq{args.seq_num}")
        if not os.path.isdir(seq_path):
            raise FileNotFoundError(f"Requested sequence not found: {seq_path}")

    all_metrics = {}

    seq_dir = seq_path
    seq_name = os.path.basename(seq_dir)
    print(f"\nProcessing {seq_name}...")

    seq_outputs_dir = os.path.join(outputs_dir, "outputs")
    os.makedirs(seq_outputs_dir, exist_ok=True)

    # directory for predicted masks (save with same filename as source images)
    predicted_masks_dir = os.path.join(seq_outputs_dir, "predicted_masks")
    os.makedirs(predicted_masks_dir, exist_ok=True)

    images_dir = os.path.join(seq_dir, "images")
    masks_dir = os.path.join(seq_dir, "masks")

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    seq_metrics_list = []
    seq_bbox_data = []

    for img_idx, img_path in enumerate(image_files):
        frame_name = os.path.basename(img_path).split(".")[0]
        # try common mask extensions
        mask_candidates = [
            os.path.join(masks_dir, f"{frame_name}.png"),
            os.path.join(masks_dir, f"{frame_name}.jpg"),
            os.path.join(masks_dir, f"{frame_name}.jpeg"),
        ]
        mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)

        if mask_path is None:
            print(f"Warning: Ground truth mask not found for {img_path}")
            continue

        ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
        ground_truth_binary = (ground_truth > 0.5).astype(np.uint8)

        image = Image.open(img_path)
        results = yolo_model.predict([image], imgsz=640, conf=0.5)
        boxes = results[0].boxes

        # default prediction
        pred_mask = np.zeros(ground_truth.shape)

        n_boxes = len(boxes)
        if n_boxes == 0:
            pass
        elif n_boxes == 1:
            boxes_tensor = boxes.xyxy[0]
            image_np = np.array(image.convert("RGB"))
            predictor.set_image(image_np)
            input_box = np.array(boxes_tensor)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            pred_mask = masks[0]

            img_h, img_w = image_np.shape[:2]
            x1, y1, x2, y2 = input_box
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            seq_bbox_data.append([frame_name, 0, cx, cy, w, h])
        else:
            boxes_np = np.array(boxes.xyxy)
            image_np = np.array(image.convert("RGB"))
            predictor.set_image(image_np)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_np,
                multimask_output=False,
            )
            # combine masks
            pred_mask = np.zeros(ground_truth.shape)
            for m in masks:
                # each m is (H,W) mask or tuple
                mask_arr = m[0] if isinstance(m, (list, tuple)) else m
                pred_mask = (pred_mask + mask_arr).clip(0, 1)

            img_h, img_w = image_np.shape[:2]
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
        if pred_mask.shape != ground_truth.shape:
            pred_mask = (pred_mask * 255).astype(np.uint8)
            pred_mask = cv2.resize(pred_mask, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred_mask = pred_mask.astype(np.float32) / 255.0

        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
        # save predicted mask with same filename as original image (use .png)
        mask_output_path = os.path.join(predicted_masks_dir, f"{frame_name}.png")
        cv2.imwrite(mask_output_path, pred_mask_binary * 255)

        # metrics
        s_score = structure_measure(ground_truth_binary, pred_mask)
        e_score = enhanced_alignment(ground_truth_binary, pred_mask)
        f_score = calculate_fmeasure(pred_mask_binary, ground_truth_binary)
        sensitivity, specificity = calculate_sensitivity_specificity(pred_mask_binary, ground_truth_binary)
        dice = calculate_dice(pred_mask_binary, ground_truth_binary)
        iou = calculate_iou(pred_mask_binary, ground_truth_binary)

        seq_metrics_list.append({
            "frame": frame_name,
            "structure_measure": float(s_score),
            "enhanced_alignment": float(e_score),
            "fmeasure": float(f_score),
            "sensitivity": float(sensitivity),
            "dice": float(dice),
            "iou": float(iou),
        })

        print(f"  {seq_name}/{frame_name}: Sα={s_score:.4f} E={e_score:.4f} F={f_score:.4f} Sens={sensitivity:.4f} Dice={dice:.4f} IoU={iou:.4f}")

        # save per-frame metrics
        if seq_metrics_list:
            metrics_csv = os.path.join(seq_outputs_dir, "metrics.csv")
            with open(metrics_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(seq_metrics_list[0].keys()))
                writer.writeheader()
                writer.writerows(seq_metrics_list)

            # save averages
            avg = {}
            for k in ["structure_measure", "enhanced_alignment", "fmeasure", "sensitivity", "dice", "iou"]:
                avg[k] = float(np.mean([m[k] for m in seq_metrics_list]))
            avg_csv = os.path.join(seq_outputs_dir, "metrics_avg.csv")
            with open(avg_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(avg.keys()))
                writer.writeheader()
                writer.writerow(avg)
            all_metrics[seq_name] = avg

        # save bbox.csv
        if seq_bbox_data:
            bbox_csv = os.path.join(seq_outputs_dir, "bbox.csv")
            with open(bbox_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "box_id", "x_center", "y_center", "width", "height"])
                w.writerows(seq_bbox_data)

    # summary
    print("\nProcessing complete. Sequence averages:")
    for seq, metrics in all_metrics.items():
        print(f"{seq}: {metrics}")

if __name__ == "__main__":
    main()
