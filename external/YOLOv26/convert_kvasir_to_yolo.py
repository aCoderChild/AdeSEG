import os
import cv2
import numpy as np
from glob import glob

# TODO: conver the future adenoid dataset to YOLO format
# YOLO format:
# - images/
# - labels

# Paths
# TODO: create config files for this

KVASIR_IMG_DIR = '/Users/maianhpham/Documents/adenoid_segmentation/data/segmentation/Kvasir-SEG/images'
KVASIR_BBOX_DIR = '/Users/maianhpham/Documents/adenoid_segmentation/data/segmentation/Kvasir-SEG/bbox'
TRAIN_IMG_DIR = '/Users/maianhpham/Documents/adenoid_segmentation/external/YOLOv26/train_data/images'
TRAIN_LABEL_DIR = '/Users/maianhpham/Documents/adenoid_segmentation/external/YOLOv26/train_data/labels'
VAL_IMG_DIR = '/Users/maianhpham/Documents/adenoid_segmentation/external/YOLOv26/val_data/images'
VAL_LABEL_DIR = '/Users/maianhpham/Documents/adenoid_segmentation/external/YOLOv26/val_data/labels'
TRAIN_TXT = '/Users/maianhpham/Documents/adenoid_segmentation/data/segmentation/Kvasir-SEG/train.txt'
VAL_TXT = '/Users/maianhpham/Documents/adenoid_segmentation/data/segmentation/Kvasir-SEG/val.txt'

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)


# TODO: in future adenoid dataset, we have 2 classes:
# - adenoid
# - nasopharynx
def convert_bbox_to_yolo(bbox, img_w, img_h, class_idx):
    # bbox: [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return [class_idx, x_center, y_center, width, height]

def main():

    # Read train.txt and val.txt for sample names
    with open(TRAIN_TXT, 'r') as f:
        train_samples = set(line.strip() for line in f if line.strip())
    with open(VAL_TXT, 'r') as f:
        val_samples = set(line.strip() for line in f if line.strip())

    # one class: polyp (class 0)
    class_map = {"polyp": 0}

    def process_samples(samples, img_dir, label_dir):
        for img_name in samples:
            img_path = os.path.join(KVASIR_IMG_DIR, img_name + '.jpg')
            bbox_path = os.path.join(KVASIR_BBOX_DIR, img_name + '.csv')
            if not os.path.exists(img_path):
                img_path = os.path.join(KVASIR_IMG_DIR, img_name + '.png')
            if not os.path.exists(img_path):
                print(f"Image not found for {img_name}")
                continue
            if not os.path.exists(bbox_path):
                print(f"BBox not found for {img_name}")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read {img_path}")
                continue
            img_h, img_w = img.shape[:2]
            cv2.imwrite(os.path.join(img_dir, os.path.basename(img_path)), img)
            # Read bbox CSV and convert to YOLO format
            label_path = os.path.join(label_dir, img_name + '.txt')
            with open(bbox_path, 'r') as f_bbox, open(label_path, 'w') as f_label:
                lines = f_bbox.readlines()
                for line in lines[1:]:  # skip header
                    parts = line.strip().split(',')
                    if len(parts) != 5:
                        continue
                    class_name, x_min, y_min, x_max, y_max = parts
                    class_idx = class_map.get(class_name, 0)
                    x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
                    yolo_bbox = convert_bbox_to_yolo([x_min, y_min, x_max, y_max], img_w, img_h, class_idx)
                    f_label.write(' '.join(map(str, yolo_bbox)) + '\n')

    # Process train and val samples
    process_samples(train_samples, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
    process_samples(val_samples, VAL_IMG_DIR, VAL_LABEL_DIR)

if __name__ == '__main__':
    main()
