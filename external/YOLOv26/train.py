
# YOLOv26 Training Script
import os
from ultralytics import YOLO


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv26 Training Script")
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to your YOLO-format data.yaml')
    parser.add_argument('--model_cfg', type=str, required=True, help='Pretrained YOLOv26 checkpoint (.pt)')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save results and checkpoints')
    args = parser.parse_args()

    DATA_YAML = args.data_yaml
    MODEL_CFG = args.model_cfg
    RESULTS_DIR = args.results_dir

    # Create output directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load YOLOv26 model
    model = YOLO(MODEL_CFG)

    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        momentum=0.937,
        warmup_epochs=10.0,
        plots=True,
        cos_lr=False,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        close_mosaic=10,
        patience=15,
        save_period=10,
        project=RESULTS_DIR,
        name="exp",
    )
    print("Training completed.")