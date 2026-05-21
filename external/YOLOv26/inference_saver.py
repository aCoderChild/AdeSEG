from ultralytics import YOLO
from functions import InferenceSaver
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv26 inference and save predictions.")
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to YOLOv26 model checkpoint (.pt)')
    parser.add_argument('--image_folder', type=str, required=True, help='Folder containing images for inference')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save YOLO-format predictions')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = YOLO(args.model_checkpoint)
    saver = InferenceSaver(
        model=model,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou
    )
    saver.save_all_inferences()