#!/usr/bin/env python3
"""
Generate ground truth bounding boxes from mask images.
For each mask, creates the tightest bounding box that covers all foreground pixels.
Outputs CSV files with frame_idx, x1, y1, x2, y2 for each frame.
"""

import argparse
import os
import re
import glob
import csv
import numpy as np
from PIL import Image


def get_numeric_sort_key(filename):
    """Natural sort key for files like frame_001.png, frame_10.png"""
    basename = os.path.basename(filename)
    parts = re.split(r'(\d+)', basename)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def get_bbox_from_mask(mask_array):
    """
    Get the tightest bounding box that covers all foreground pixels in a mask.
    Returns (x1, y1, x2, y2) or None if mask is empty.
    """
    # Convert to binary
    binary_mask = (mask_array > 0).astype(np.uint8)
    
    # Find all non-zero coordinates
    coords = np.where(binary_mask > 0)
    
    if len(coords[0]) == 0:
        # Empty mask
        return None
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Ensure valid box
    x1, y1 = int(x_min), int(y_min)
    x2, y2 = int(x_max) + 1, int(y_max) + 1  # +1 to include the edge pixel
    
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth bounding boxes from masks"
    )
    parser.add_argument(
        "--seq_num",
        type=int,
        required=True,
        help="Sequence number (e.g., 1 for seq1)"
    )
    parser.add_argument(
        "--data_root",
        default="data/polypgen",
        help="Root directory for polypgen data"
    )
    parser.add_argument(
        "--output_root",
        default="data/bbox",
        help="Root directory for output bounding box data"
    )
    args = parser.parse_args()

    # Construct paths
    seq_dir = os.path.join(args.data_root, f"seq{args.seq_num}")
    masks_dir = os.path.join(seq_dir, "masks")
    output_dir = os.path.join(args.output_root, f"seq{args.seq_num}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if masks directory exists
    if not os.path.isdir(masks_dir):
        print(f"Error: Masks directory not found: {masks_dir}")
        return False
    
    # Find all mask files
    mask_files = glob.glob(os.path.join(masks_dir, "*.png")) + \
                 glob.glob(os.path.join(masks_dir, "*.jpg")) + \
                 glob.glob(os.path.join(masks_dir, "*.jpeg"))
    
    if not mask_files:
        print(f"Warning: No mask files found in {masks_dir}")
        return False
    
    # Remove duplicates while preserving order, then sort numerically
    mask_files = list(dict.fromkeys(mask_files))
    mask_files = sorted(mask_files, key=get_numeric_sort_key)
    
    # Process each mask file
    bbox_data = []
    for mask_path in mask_files:
        frame_name = os.path.splitext(os.path.basename(mask_path))[0]
        
        try:
            # Load mask
            mask_img = Image.open(mask_path).convert("L")
            mask_array = np.array(mask_img)
            
            # Get bounding box
            bbox = get_bbox_from_mask(mask_array)
            
            if bbox is None:
                # Empty masks are skipped - evaluation pipeline handles missing frames gracefully
                print(f"Warning: Empty mask for {frame_name}, skipping")
                continue
            
            x1, y1, x2, y2 = bbox
            bbox_data.append([frame_name, x1, y1, x2, y2])
            
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
            continue
    
    # Always write the CSV. Header-only files mean the sequence has no
    # foreground boxes, which is valid for mask evaluation.
    output_csv = os.path.join(output_dir, "bboxes.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "x1", "y1", "x2", "y2"])
        writer.writerows(bbox_data)

    if bbox_data:
        print(f"Saved {len(bbox_data)} bounding boxes to {output_csv}")
    else:
        print(f"Saved header-only bbox file for seq{args.seq_num}: no foreground masks")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

# How to run
# for i in {1..23}; do python3 scripts/utils/ground_truth_bbox_gen.py --seq_num $i; done
