import os
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io, transform
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import importlib.util
import cv2

# Load config from YAML
import yaml
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'infer.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

VAL_IMG = config['VAL_IMG']
VAL_BBOX = config['VAL_BBOX_MANUAL']
VAL_MASK = config['VAL_MASK']
CHECKPOINT = config['CHECKPOINT']
RESULTS_DIR = config['RESULTS_DIR_MANUAL']
VIS_DIR = os.path.join(RESULTS_DIR, config['VIS_SUBDIR'])
METRICS_DIR = os.path.join(RESULTS_DIR, config['METRICS_SUBDIR'])
PER_SAMPLE_CSV = os.path.join(METRICS_DIR, config['PER_SAMPLE_CSV'])
AVG_CSV = os.path.join(METRICS_DIR, config['AVG_CSV'])

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Load MedSAM model
def load_medsam_model(checkpoint_path, device):
    from segment_anything.build_sam import sam_model_registry
    medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    medsam_model.eval()
    medsam_model.to(device)
    return medsam_model

# Inference function (from MedSAM_Inference.py)
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None)
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    low_res_pred = low_res_pred.squeeze().detach().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

# Visualisation helpers
def overlay_mask(img, mask, color, alpha=0.5):
    img = img.copy()
    mask = mask.astype(bool)
    img[mask] = (1 - alpha) * img[mask] + alpha * np.array(color)
    return img

def draw_bbox(img, bbox, color, lw=2):
    img = img.copy()
    x0, y0, x1, y1 = [int(v) for v in bbox]
    img = cv2.rectangle(img, (x0, y0), (x1, y1), color, lw)
    return img


# Load metrics from utils/metrics.py
import sys
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'utils')
if METRICS_PATH not in sys.path:
    sys.path.append(METRICS_PATH)
import utils.metrics as medsam_metrics

# Main batch inference and evaluation
def main():
    import cv2
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device selected: {device}")
    print(f"Loading MedSAM model from: {CHECKPOINT}")
    medsam_model = load_medsam_model(CHECKPOINT, device)
    results = []
    img_names = sorted(os.listdir(VAL_IMG))
    print(f"Found {len(img_names)} images in {VAL_IMG}")
    processed_count = 0
    for img_name in tqdm(img_names):
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(VAL_IMG, img_name)
        bbox_path = os.path.join(VAL_BBOX, base + '.csv')
        mask_path = os.path.join(VAL_MASK, base + '.jpg')
        if not (os.path.exists(bbox_path) and os.path.exists(mask_path)):
            print(f"Skipping {img_name}: bbox or mask not found. Bbox: {bbox_path}, Mask: {mask_path}")
            continue
        # Read image
        print(f"Processing image: {img_path}")
        img_np = io.imread(img_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape
        # Preprocess
        img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Read bbox in CSV format: class_name,xmin,ymin,xmax,ymax
        with open(bbox_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            # If header is not numeric, skip it
            if not header[1].replace('.', '', 1).isdigit():
                row = next(reader)
            else:
                row = header
            if len(row) == 5:
                # Format: class_name,xmin,ymin,xmax,ymax
                box = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
            else:
                raise ValueError(f"Unexpected bbox format in {bbox_path}: {row}")
        print(f"Loaded bbox from {bbox_path}: {box}")
        box_np = np.array([box])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        # Inference
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)
        pred_mask = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
        # Load GT mask
        gt_mask = io.imread(mask_path)
        print(f"Loaded GT mask from {mask_path}, shape: {gt_mask.shape}")
        if gt_mask.max() > 1:
            gt_mask = (gt_mask > 0).astype(np.uint8)
        # Create visualizations
        vis1 = overlay_mask(img_3c/255.0, gt_mask, [0,1,0], 0.5) # GT green
        vis1 = draw_bbox((vis1*255).astype(np.uint8), box, (0,255,0))
        vis2 = overlay_mask(img_3c/255.0, pred_mask, [1,0,0], 0.5) # Pred red
        vis2 = draw_bbox((vis2*255).astype(np.uint8), box, (0,0,255))
        vis3 = overlay_mask(img_3c/255.0, gt_mask, [0,1,0], 0.3)
        vis3 = overlay_mask(vis3, pred_mask, [1,0,0], 0.3)
        vis3 = (vis3*255).astype(np.uint8)

        # Combine the three images horizontally with a separator
        separator = np.full((vis1.shape[0], 5, 3), 255, dtype=np.uint8)
        combined = np.hstack((vis1, separator, vis2, separator, vis3))

        # Save the combined image
        combined_path = os.path.join(VIS_DIR, f'{base}_combined.jpg')
        io.imsave(combined_path, combined)
        print(f"Saved combined visualization: {combined_path}")
        # Evaluate using metrics.py
        dice_score = medsam_metrics.dice_coefficient(pred_mask, gt_mask)
        iou = medsam_metrics.iou_score(pred_mask, gt_mask)
        f1 = medsam_metrics.f_measure(pred_mask, gt_mask)
        s_measure = medsam_metrics.structure_measure(pred_mask, gt_mask)
        wfb = medsam_metrics.weighted_f_measure(pred_mask, gt_mask)
        emeasure = medsam_metrics.enhanced_alignment_measure(pred_mask, gt_mask)
        mae_val = medsam_metrics.mae(pred_mask, gt_mask)
        precision, recall, specificity, sensitivity = medsam_metrics.precision_recall_specificity(pred_mask, gt_mask)
        print(f"Dice: {dice_score:.4f}, IoU: {iou:.4f}, F1: {f1:.4f}, S-measure: {s_measure:.4f}, WFb: {wfb:.4f}, E-measure: {emeasure:.4f}, MAE: {mae_val:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}")
        results.append({
            'image': img_name,
            'dice': dice_score,
            'iou': iou,
            'f1': f1,
            's_measure': s_measure,
            'wfb': wfb,
            'emeasure': emeasure,
            'mae': mae_val,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'sensitivity': sensitivity
        })
        processed_count += 1
    # Save per-sample metrics
    print(f"Processed {processed_count} images.")
    print(f"Saving per-sample metrics to {PER_SAMPLE_CSV}")
    metric_fields = ['image', 'dice', 'iou', 'f1', 's_measure', 'wfb', 'emeasure', 'mae', 'precision', 'recall', 'specificity', 'sensitivity']
    with open(PER_SAMPLE_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metric_fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    # Save average metrics
    if results:
        avg_metrics = {k: np.mean([r[k] for r in results]) for k in metric_fields if k != 'image'}
    else:
        print("Warning: No images were processed. Creating empty metrics files.")
        avg_metrics = {k: 0 for k in metric_fields if k != 'image'}
    print(f"Saving average metrics to {AVG_CSV}")
    with open(AVG_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['metric', 'average'])
        for k, v in avg_metrics.items():
            writer.writerow([k, v])
    print('Done!')

if __name__ == '__main__':
    main()
