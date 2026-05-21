from ultralytics import YOLO
from functions import ResultPlotter
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLOv26 predictions vs. ground truth.")
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to YOLOv26 model checkpoint (.pt)')
    parser.add_argument('--image_folder', type=str, required=True, help='Folder containing images to visualize')
    parser.add_argument('--label_folder', type=str, required=True, help='Folder containing YOLO-format ground truth labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualization results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for predictions')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS')
    parser.add_argument('--all', action='store_true', help='Visualize all images in the folder')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = YOLO(args.model_checkpoint)
    plotter = ResultPlotter(
        model=model,
        image_folder=args.image_folder,
        label_folder=args.label_folder,
        results_fig_root=args.output_dir
    )
    if args.all:
        plotter.plot_all_images()
    else:
        plotter.prepare_single_image()
        plotter.plot_single_image()