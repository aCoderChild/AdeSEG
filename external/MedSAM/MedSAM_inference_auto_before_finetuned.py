import os
import numpy as np
import torch
from skimage import io, transform
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import cv2

# Load config from YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'infer.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

VAL_IMG = config.get('VAL_IMG', 'data/segmentation/val_data/images')
VAL_MASK = config.get('VAL_MASK', 'data/segmentation/val_data/masks')
CHECKPOINT = config.get('CHECKPOINT')
RESULTS_DIR = config.get('RESULTS_DIR_AUTO_BEFORE_FINETUNED', 'results/auto/original_checkpoint_encoder')
VIS_DIR = os.path.join(RESULTS_DIR, config.get('VIS_SUBDIR', 'vis'))
NO_MASK_DIR = os.path.join(RESULTS_DIR, config.get('NO_MASK_SUBDIR', 'no_masks'))
METRICS_DIR = os.path.join(RESULTS_DIR, config.get('METRICS_SUBDIR', 'metrics'))
PER_SAMPLE_CSV = os.path.join(METRICS_DIR, config.get('PER_SAMPLE_CSV', 'per_sample.csv'))
AVG_CSV = os.path.join(METRICS_DIR, config.get('AVG_CSV', 'avg.csv'))
MASK_OUT_DIR = config.get('MASK_OUT_DIR', 'data/segmentation/val_data/bbox')

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MASK_OUT_DIR, exist_ok=True)

# Load MedSAM model and automatic mask generator
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

def load_medsam_model(checkpoint_path, device):
    # If checkpoint_path is falsy (None or empty), or does not exist, raise error
    if not checkpoint_path:
        raise ValueError("No checkpoint path provided. Please specify a valid checkpoint file.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} should be a state_dict, not a dict with 'model' key. Please provide the correct .pth file.")
    medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
    print(f"Loaded checkpoint keys: {list(checkpoint.keys())[:10]} ... total {len(checkpoint)}")
    print(f"Model state_dict keys: {list(medsam_model.state_dict().keys())[:10]} ... total {len(medsam_model.state_dict())}")
    medsam_model.load_state_dict(checkpoint)
    medsam_model.eval()
    medsam_model.to(device)
    return medsam_model

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

def main():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device selected: {device}")
    print(f"Loading MedSAM model from: {CHECKPOINT}")
    medsam_model = load_medsam_model(CHECKPOINT, device)
    mask_generator = SamAutomaticMaskGenerator(
        medsam_model,
        points_per_side=32, # default
        pred_iou_thresh=0.5,  # default: 0.88
        stability_score_thresh=0.5,  # default: 0.95
        min_mask_region_area=0,
    )

    results = []
    img_names = sorted(os.listdir(VAL_IMG))
    print(f"Found {len(img_names)} images in {VAL_IMG}")
    os.makedirs(NO_MASK_DIR, exist_ok=True)
    for img_name in tqdm(img_names):
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(VAL_IMG, img_name)
        mask_path = os.path.join(VAL_MASK, base + '.jpg')
        if not os.path.exists(mask_path):
            print(f"No mask file found for {img_name}")
            # Move image to no-mask folder
            os.rename(img_path, os.path.join(NO_MASK_DIR, img_name))
            continue
        mask = io.imread(mask_path)
        if mask.max() == 0:
            print(f"No mask (all zeros) for {img_name}")
            os.rename(img_path, os.path.join(NO_MASK_DIR, img_name))
            continue
        # Read image
        img_np = io.imread(img_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        # Ensure float32 for MPS compatibility
        img_3c = img_3c.astype(np.float32)
        H, W, _ = img_3c.shape
        # Print image statistics for debugging
        print(f"{img_name}: min={img_3c.min()}, max={img_3c.max()}, mean={img_3c.mean()}, shape={img_3c.shape}")
        # Run automatic mask generator
        masks = mask_generator.generate(img_3c)
        # Take the largest mask (by area) as the main prediction
        if len(masks) == 0:
            print(f"No masks found for {img_name}")
            os.rename(img_path, os.path.join(NO_MASK_DIR, img_name))
            continue
        main_mask = max(masks, key=lambda m: m['area'])['segmentation']
        # Save mask
        mask_save_path = os.path.join(MASK_OUT_DIR, base + '.jpg')
        io.imsave(mask_save_path, (main_mask * 255).astype(np.uint8))
        # Load GT mask
        gt_mask = io.imread(mask_path)
        if gt_mask.max() > 1:
            gt_mask = (gt_mask > 0).astype(np.uint8)
        # Create visualizations
        vis1 = overlay_mask(img_3c/255.0, gt_mask, [0,1,0], 0.5) # GT green
        vis2 = overlay_mask(img_3c/255.0, main_mask, [1,0,0], 0.5) # Pred red
        vis3 = overlay_mask(img_3c/255.0, gt_mask, [0,1,0], 0.3)
        vis3 = overlay_mask(vis3, main_mask, [1,0,0], 0.3)
        vis3 = (vis3*255).astype(np.uint8)
        separator = np.full((vis1.shape[0], 5, 3), 255, dtype=np.uint8)
        combined = np.hstack((vis1*255, separator, vis2*255, separator, vis3))
        combined_path = os.path.join(VIS_DIR, f'{base}_combined.jpg')
        io.imsave(combined_path, combined.astype(np.uint8))
        # Evaluate using metrics.py
        dice_score = medsam_metrics.dice_coefficient(main_mask, gt_mask)
        iou = medsam_metrics.iou_score(main_mask, gt_mask)
        f1 = medsam_metrics.f_measure(main_mask, gt_mask)
        s_measure = medsam_metrics.structure_measure(main_mask, gt_mask)
        wfb = medsam_metrics.weighted_f_measure(main_mask, gt_mask)
        emeasure = medsam_metrics.enhanced_alignment_measure(main_mask, gt_mask)
        mae_val = medsam_metrics.mae(main_mask, gt_mask)
        precision, recall, specificity, sensitivity = medsam_metrics.precision_recall_specificity(main_mask, gt_mask)
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
    # Save per-sample metrics
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(PER_SAMPLE_CSV, index=False)
    # Save average metrics
    avg_metrics = df.mean(numeric_only=True)
    avg_metrics.to_csv(AVG_CSV)
    print(f"Saved per-sample metrics to {PER_SAMPLE_CSV}")
    print(f"Saved average metrics to {AVG_CSV}")

if __name__ == "__main__":
    main()
