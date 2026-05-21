
# YOLOv26 Evaluation Script
import torch
from ultralytics import YOLO

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv26 Evaluation Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained YOLOv26 model (.pt)')
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to YOLO-format data.yaml')
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    DATA_YAML = args.data_yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(MODEL_PATH)

    # Evaluate
    metrics = model.val(data=DATA_YAML, device=device)
    print("Evaluation metrics:")
    print(metrics)

    # Save metrics to results/metrics/metrics.csv only
    import os
    import csv
    metrics_dir = 'results/metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_csv_path = os.path.join(metrics_dir, 'metrics.csv')

    # Try to extract results_dict if present, else save all metrics
    results_dict = getattr(metrics, 'results_dict', None)
    if results_dict is None:
        # Fallback: try to convert metrics to dict
        try:
            results_dict = dict(metrics)
        except Exception:
            results_dict = {'raw_metrics': str(metrics)}

    # Save as CSV (flat key-value pairs)
    with open(metrics_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['metric', 'value'])
        for k, v in results_dict.items():
            writer.writerow([k, v])
    print(f"[INFO] Metrics saved to {metrics_csv_path}")