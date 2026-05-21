import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import math

from ultralytics import YOLO

def load_model_train(model_name):
    """
    Loads a YOLOv26 model for training.
    """
    if model_name == "yolo26n":
        return YOLO("external/checkpoints/yolo26n.pt")
    elif model_name == "yolo26s":
        return YOLO("external/checkpoints/yolo26s.pt")
    else:
        raise Exception("Model missing: " + model_name)


def load_model_test(model_name):
    """
    Loads a YOLOv26 model for testing.
    """
    def inner(checkpoints_name):
        if model_name in ["yolo26n", "yolo26s"]:
            ckpt_path = f"external/YOLOv26/runs/train/{checkpoints_name}/best.pt"
            return YOLO(ckpt_path)
        else:
            raise Exception("Model missing: " + model_name)
    return inner


class ModelLoader:
    def draw_boxes(self, image, boxes, colors=None, class_names=None):
        """
        Draws bounding boxes on the image.
        boxes: list of (class_id, x_center, y_center, width, height) in normalized coordinates (0-1)
        colors: list or dict of BGR tuples, or None for default
        class_names: list of class names, or None
        """
        img = image.copy()
        h, w = img.shape[:2]
        if colors is None:
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for box in boxes:
            class_id, xc, yc, bw, bh = box
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            color = colors[class_id % len(colors)] if isinstance(colors, (list, tuple)) else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = str(class_id)
            if class_names is not None and class_id < len(class_names):
                label = str(class_names[class_id])
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img
        
    def load_yolo_labels(self, label_path):
        """
        Loads YOLO-format bounding box labels from a .txt file.
        Returns a list of (class_id, x_center, y_center, width, height) as floats (all normalized 0-1).
        """
        boxes = []
        if not os.path.exists(label_path):
            return boxes
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x, y, w, h = map(float, parts)
                    boxes.append((int(class_id), x, y, w, h))
            return boxes
        
    def get_file_names(self, folder, max_files=None):
        """
            Returns a sorted list of image file names in the given folder. Optionally limits to max_files.
        """
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in valid_exts]
        files.sort()
        if max_files is not None:
            files = files[:max_files]
        return files
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        if model_name in ["yolo26n", "yolo26s"]:
            ckpt_path = f"external/YOLOv26/runs/train/{checkpoints_name}/best.pt"
            return YOLO(ckpt_path)
        else:
            raise Exception("Model missing: " + model_name)



import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class ModelEvaluator(ModelLoader):
    def __init__(self, MODEL_NAME, data_yaml, image_folder, label_folder, device, conf=0.5, iou=0.5):
        super().__init__(MODEL_NAME)
        self.data_yaml = data_yaml
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.device = device
        self.conf = conf
        self.iou = iou
        self.combined_images = None

    def evaluate(self, results_path):
        """
        Evaluate YOLO model and return results.
        """
        self.model.to(self.device)
        val_results = self.model.val(
            data=self.data_yaml,
            project=results_path,
            conf=self.conf,
            iou=self.iou,
            split='test'
        )
        return val_results

    def prepare_images(self, nr_images=4):
        images = self.get_file_names(self.image_folder, nr_images)
        combined_images = []
        for img_file in images:
            img_path = os.path.join(self.image_folder, img_file)
            label_path = os.path.join(self.label_folder, os.path.splitext(img_file)[0] + ".txt")
            img = cv2.imread(img_path)
            img = cv2.resize(img, (640, 640))
            img_gt = img.copy()
            img_pred = img.copy()
            gt_boxes = self.load_yolo_labels(label_path)
            img_gt = self.draw_boxes(img_gt, gt_boxes, self.colors, class_names=self.model.names)
            cv2.putText(img_gt, "Ground Truth", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            results = self.model(img_path)[0]
            pred_boxes = []
            for cls, xywh in zip(results.boxes.cls, results.boxes.xywh):
                cls = int(cls.item())
                x, y, w, h = xywh.tolist()
                pred_boxes.append((cls, x / results.orig_shape[1], y / results.orig_shape[0], w / results.orig_shape[1], h / results.orig_shape[0]))
            img_pred = self.draw_boxes(img_pred, pred_boxes, self.colors, class_names=self.model.names)
            cv2.putText(img_pred, "Prediction", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            combined = np.hstack((img_gt, img_pred))
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            combined_images.append(combined_rgb)
        self.combined_images = combined_images

    def plot(self, results_fig_root):
        num_images = 4
        cols = 2
        rows = math.ceil(len(self.combined_images) / cols)
        grid_rows = []
        for i in range(rows):
            row_imgs = self.combined_images[i * cols:(i + 1) * cols]
            while len(row_imgs) < cols:
                h, w, _ = row_imgs[0].shape
                empty_img = np.zeros((h, w, 3), dtype=np.uint8)
                row_imgs.append(empty_img)
            row = np.hstack(row_imgs)
            grid_rows.append(row)
        final_image = np.vstack(grid_rows)
        plt.figure(figsize=(16, num_images * 5))
        plt.imshow(final_image)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f"external/YOLOv26/results/plot_{self.model_name}.png", bbox_inches="tight", pad_inches=0)
        plt.close()


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class ResultPlotter(ModelLoader):
    # make_combined_image removed; logic moved to prepare_single_image
    def plot_all_images(self):
        images = self.get_file_names(self.image_folder)
        if not images:
            print(f"[ERROR] No images found in folder: {self.image_folder}")
            return
        for idx, img_file in enumerate(images):
            print(f"[INFO] Processing image {idx+1}/{len(images)}: {img_file}")
            self.prepare_single_image(image_index=idx)
            if self.combined_images:
                self.plot_single_image()
                
    def __init__(self, model, image_folder, label_folder, results_fig_root):
        # Try to extract model_name from model or checkpoint path if possible
        model_name = getattr(model, 'model_name', None)
        if model_name is None and hasattr(model, 'ckpt_path'):
            # Try to extract from checkpoint path
            ckpt_path = model.ckpt_path if hasattr(model, 'ckpt_path') else ''
            if 'yolo26n' in ckpt_path:
                model_name = 'yolo26n'
            elif 'yolo26s' in ckpt_path:
                model_name = 'yolo26s'
            else:
                model_name = 'unknown'
        if model_name is None:
            model_name = 'unknown'
        super().__init__(model_name)
        self.model = model
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.results_fig_root = results_fig_root
        self.combined_images = None
        self.image_filename = None
        # Default colors for drawing boxes (BGR)
        self.colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
        ]

    def prepare_single_image(self, image_index=0):
        images = self.get_file_names(self.image_folder, 10)
        print(f"[INFO] Images found in {self.image_folder}:")
        for img_name in images:
            print(f"  - {img_name}")
        if not images:
            print(f"[ERROR] No images found in folder: {self.image_folder}")
            self.combined_images = []
            self.image_filename = None
            return
        checked = 0
        found = False
        while checked < len(images):
            idx = (image_index + checked) % len(images)
            img_file = images[idx]
            img_path = os.path.join(self.image_folder, img_file)
            label_path = os.path.join(self.label_folder, os.path.splitext(img_file)[0] + ".txt")
            print(f"[INFO] Trying image: {img_file}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Could not read image: {img_path}")
                print(f"[DEBUG] Image file does not exist: {not os.path.exists(img_path)}")
                print(f"[DEBUG] Current working directory: {os.getcwd()}")
                checked += 1
                continue
            print(f"[INFO] Successfully loaded image: {img_file}")
            found = True
            break
        if not found:
            print(f"[ERROR] No valid images to process in {self.image_folder}")
            self.combined_images = []
            self.image_filename = None
            return
        img = cv2.resize(img, (640, 640))
        img_gt = img.copy()
        img_pred = img.copy()

        # Load ground truth boxes and draw on GT image
        gt_boxes = self.load_yolo_labels(label_path)
        img_gt = self.draw_boxes(img_gt, gt_boxes, self.colors, class_names=self.model.names)
        if img_gt is None:
            print(f"[ERROR] draw_boxes returned None for image: {img_file}")
            self.combined_images = []
            self.image_filename = img_file
            return

        # Run inference to get predicted boxes
        results = self.model(img_path)[0]
        pred_boxes = []
        for cls, xywh in zip(results.boxes.cls, results.boxes.xywh):
            cls = int(cls.item())
            x, y, w, h = xywh.tolist()
            pred_boxes.append((cls, x / results.orig_shape[1], y / results.orig_shape[0], w / results.orig_shape[1], h / results.orig_shape[0]))
        img_pred = self.draw_boxes(img_pred, pred_boxes, self.colors, class_names=self.model.names)

        # Prepare overlay: GT in green, prediction in red
        img_overlay = img.copy()
        img_overlay = self.draw_boxes(img_overlay, gt_boxes, colors=[(0,255,0)], class_names=self.model.names)
        img_overlay = self.draw_boxes(img_overlay, pred_boxes, colors=[(0,0,255)], class_names=self.model.names)

        # Add labels to each panel
        cv2.putText(img_gt, "Ground Truth", (10, img_gt.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(img_pred, "Prediction", (10, img_pred.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(img_overlay, "Overlay", (10, img_overlay.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # Combine all three images horizontally
        separator = np.full((img_gt.shape[0], 1, img_gt.shape[2]), 255, dtype=img_gt.dtype)
        combined = np.hstack((img_gt, separator, img_pred, separator, img_overlay))
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        self.combined_images = [combined_rgb]
        self.image_filename = img_file

    def plot_single_image(self):
        if not self.combined_images:
            print("No images to plot.")
            return
        final_image = self.combined_images[0]
        # Save the combined image (GT | Prediction | Overlay)
        plt.figure(figsize=(18, 6))
        plt.imshow(final_image)
        plt.axis("off")
        plt.tight_layout(pad=0)
        safe_filename = os.path.splitext(self.image_filename)[0]
        # Ensure output directory exists
        output_dir = self.results_fig_root if hasattr(self, 'results_fig_root') and self.results_fig_root else "external/YOLOv26/results"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"visualization_{safe_filename}.png")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close()
        print(f"Saved plot to {out_path}")


class InferenceSaver:
    """
    Save model inferences as YOLO-format .txt files for a single model.
    """
    def __init__(self, model, image_folder, output_dir, conf=0.25, iou=0.5):
        self.model = model
        self.image_folder = image_folder
        self.output_dir = output_dir
        self.conf = conf
        self.iou = iou

    def save_all_inferences(self):
        os.makedirs(self.output_dir, exist_ok=True)
        image_files = [f for f in os.listdir(self.image_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]
        print(f"Processing {len(image_files)} images...")
        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(self.image_folder, img_file)
            results = self.model(img_path, conf=self.conf, iou=self.iou, verbose=False)[0]
            base_name = os.path.splitext(img_file)[0]
            output_path = os.path.join(self.output_dir, f"{base_name}.txt")
            self._save_inference_yolo(results, output_path)
            if idx % 10 == 0 or idx == len(image_files):
                print(f"  Processed {idx}/{len(image_files)} images")
        print(f"Inferences saved to: {self.output_dir}\n")

    def _save_inference_yolo(self, results, output_path):
        with open(output_path, 'w') as f:
            if len(results.boxes) == 0:
                return
            for cls, xywh, conf in zip(results.boxes.cls, results.boxes.xywh, results.boxes.conf):
                cls_id = int(cls.item())
                x, y, w, h = xywh.tolist()
                x_norm = x / results.orig_shape[1]
                y_norm = y / results.orig_shape[0]
                w_norm = w / results.orig_shape[1]
                h_norm = h / results.orig_shape[0]
                f.write(f"{cls_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {conf:.4f}\n")