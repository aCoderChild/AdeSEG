"""Frame-independent MedSAM2 inference with YOLOv8 or dataset box prompts.

Each frame is embedded and decoded independently with ``SAM2ImagePredictor``.
This script deliberately does not construct video inference state, call
``propagate_in_video``, or carry masks/features between frames.
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.model_defaults import (
    DEFAULT_SAM2_CFG,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CODE_ROOT,
    DEFAULT_YOLO_CHECKPOINT,
)
from utils.mask_utils import save_masks_to_dir

if DEFAULT_SAM2_CODE_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_SAM2_CODE_ROOT)

from MedSAM2.sam2.build_sam import build_sam2
from MedSAM2.sam2.sam2_image_predictor import SAM2ImagePredictor


LOCAL_SAM2_CHECKPOINT = os.path.join(project_root, "checkpoints", "MedSAM2_latest.pt")
LOCAL_YOLO_CHECKPOINT = os.path.join(project_root, "checkpoints", "polypgen_yolov8n.pt")
SAM2_CHECKPOINT_DEFAULT = (
    DEFAULT_SAM2_CHECKPOINT
    if os.path.isfile(DEFAULT_SAM2_CHECKPOINT)
    else LOCAL_SAM2_CHECKPOINT
)
YOLO_CHECKPOINT_DEFAULT = (
    DEFAULT_YOLO_CHECKPOINT
    if os.path.isfile(DEFAULT_YOLO_CHECKPOINT)
    else LOCAL_YOLO_CHECKPOINT
)


def get_numeric_sort_key(name):
    """Natural sort key, e.g. 2 before 10 and seq2 before seq10."""
    parts = re.split(r"(\d+)", name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def is_image_file(path):
    return os.path.splitext(path)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]


def get_video_frame_dir(base_video_dir, video_name):
    """Support DAVIS-style video dirs and PolypGen ``seq/images_seq`` dirs."""
    if video_name == ".":
        image_dir = os.path.join(base_video_dir, "images")
        return image_dir if os.path.isdir(image_dir) else base_video_dir

    image_dir = os.path.join(base_video_dir, video_name, "images")
    if os.path.isdir(image_dir):
        return image_dir

    polypgen_image_dir = os.path.join(base_video_dir, video_name, f"images_{video_name}")
    if os.path.isdir(polypgen_image_dir):
        return polypgen_image_dir
    return os.path.join(base_video_dir, video_name)


def get_video_name(base_video_dir, video_name):
    if video_name != ".":
        return video_name
    parent = os.path.basename(os.path.dirname(os.path.abspath(base_video_dir)))
    current = os.path.basename(os.path.abspath(base_video_dir))
    return parent if current == "images" else current


def list_video_names(base_video_dir):
    """Return sub-video names or ``.`` when ``base_video_dir`` contains images."""
    image_dir = os.path.join(base_video_dir, "images")
    if os.path.isdir(image_dir) and any(is_image_file(p) for p in os.listdir(image_dir)):
        return ["."]

    video_names = [
        name
        for name in os.listdir(base_video_dir)
        if os.path.isdir(os.path.join(base_video_dir, name))
    ]
    if video_names:
        return sorted(video_names, key=get_numeric_sort_key)

    if any(is_image_file(p) for p in os.listdir(base_video_dir)):
        return ["."]
    return []


def get_frame_names(video_dir):
    frame_names = [
        os.path.splitext(name)[0]
        for name in os.listdir(video_dir)
        if is_image_file(name)
    ]
    return sorted(frame_names, key=get_numeric_sort_key)


def resolve_frame_path(video_dir, frame_name):
    for extension in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        frame_path = os.path.join(video_dir, f"{frame_name}{extension}")
        if os.path.exists(frame_path):
            return frame_path
    raise FileNotFoundError(os.path.join(video_dir, f"{frame_name}.jpg"))


def get_yolo_boxes(yolo_model, frame_path, yolo_imgsz, yolo_conf, max_boxes):
    """Return up to ``max_boxes`` YOLO detections as absolute xyxy prompts."""
    image = Image.open(frame_path).convert("RGB")
    results = yolo_model.predict([image], imgsz=yolo_imgsz, conf=yolo_conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    order = np.argsort(confidence)[::-1]
    if max_boxes > 0:
        order = order[:max_boxes]
    return [(xyxy[index].astype(np.float32), float(confidence[index])) for index in order]


def get_data_boxes(bbox_dir, frame_name, max_boxes):
    """Load absolute xyxy boxes from ``bbox_seqN/<frame_name>.txt``."""
    bbox_path = os.path.join(bbox_dir, f"{frame_name}.txt")
    if not os.path.isfile(bbox_path):
        return []

    boxes = []
    with open(bbox_path, "r", encoding="utf-8") as file:
        for line in file:
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"Expected 5 fields in {bbox_path}: {line.rstrip()}")
            _, x1, y1, x2, y2 = fields
            boxes.append((np.array([x1, y1, x2, y2], dtype=np.float32), 1.0))
    return boxes[:max_boxes] if max_boxes > 0 else boxes


@torch.inference_mode()
def frame_inference(
    predictor,
    yolo_model,
    box_source,
    base_video_dir,
    output_mask_dir,
    video_name,
    yolo_imgsz=640,
    yolo_conf=0.5,
    max_yolo_boxes_per_frame=1,
):
    """Segment every frame independently from that frame's available box prompts."""
    video_dir = get_video_frame_dir(base_video_dir, video_name)
    video_output_name = get_video_name(base_video_dir, video_name)
    frame_names = get_frame_names(video_dir)
    if not frame_names:
        raise RuntimeError(f"Found no image frames in {video_dir}")

    data_box_dir = os.path.join(base_video_dir, video_name, f"bbox_{video_name}")
    for frame_idx, frame_name in enumerate(frame_names):
        frame_path = resolve_frame_path(video_dir, frame_name)
        if box_source == "yolo":
            boxes = get_yolo_boxes(
                yolo_model,
                frame_path,
                yolo_imgsz,
                yolo_conf,
                max_yolo_boxes_per_frame,
            )
        else:
            boxes = get_data_boxes(data_box_dir, frame_name, max_yolo_boxes_per_frame)

        with Image.open(frame_path) as source_image:
            image = source_image.convert("RGB")
        width, height = image.size

        # An image predictor cannot decode a prompted object when no box exists.
        # Saving an empty mask is intentional and does not transfer information
        # from any preceding frame.
        if not boxes:
            save_masks_to_dir(
                output_mask_dir,
                video_output_name,
                frame_name,
                {},
                height,
                width,
                False,
            )
            continue

        predictor.set_image(image)
        per_obj_mask = {}
        for object_id, (box, _) in enumerate(boxes, start=1):
            masks, _, _ = predictor.predict(
                box=box,
                multimask_output=False,
                return_logits=False,
            )
            per_obj_mask[object_id] = masks[0]
        save_masks_to_dir(
            output_mask_dir,
            video_output_name,
            frame_name,
            per_obj_mask,
            height,
            width,
            False,
        )

        print(
            f"{video_output_name}: frame {frame_idx + 1}/{len(frame_names)} "
            f"({len(boxes)} box prompt(s))"
        )

    return video_output_name, frame_names


def main():
    parser = argparse.ArgumentParser(
        description="Frame-independent MedSAM2 inference with box prompts."
    )
    parser.add_argument("--sam2_cfg", default=DEFAULT_SAM2_CFG)
    parser.add_argument("--sam2_checkpoint", default=SAM2_CHECKPOINT_DEFAULT)
    parser.add_argument(
        "-i",
        "--base_video_dir",
        required=True,
        help="dataset root containing seq*/images_seq* folders, or an image folder",
    )
    parser.add_argument("--yolo_checkpoint", default=YOLO_CHECKPOINT_DEFAULT)
    parser.add_argument(
        "--box_source",
        choices=["yolo", "data"],
        default="yolo",
        help="use YOLOv8 detections or PolypGen bbox_seqN annotation files",
    )
    parser.add_argument("--video_list_file", default=None)
    parser.add_argument("--seq_nums", type=int, nargs="*", default=None)
    parser.add_argument("-o", "--output_mask_dir", required=True)
    parser.add_argument("--yolo_conf", type=float, default=0.5)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--max_yolo_boxes_per_frame", type=int, default=1)
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="apply the model's optional image-mask postprocessing",
    )
    args = parser.parse_args()

    model = build_sam2(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
    )
    predictor = SAM2ImagePredictor(model)

    yolo_model = None
    if args.box_source == "yolo":
        if YOLO is None:
            raise RuntimeError(
                "YOLO inference requires the 'ultralytics' package, but it is not installed."
            )
        yolo_model = YOLO(args.yolo_checkpoint)

    if args.seq_nums:
        video_names = [f"seq{sequence_number}" for sequence_number in args.seq_nums]
    elif args.video_list_file:
        with open(args.video_list_file, "r", encoding="utf-8") as file:
            video_names = [line.strip() for line in file if line.strip()]
    else:
        video_names = list_video_names(args.base_video_dir)

    print(
        f"Running frame-independent MedSAM2 inference on {len(video_names)} sequence(s) "
        f"with {args.box_source} boxes."
    )
    for index, video_name in enumerate(video_names, start=1):
        print(f"{index}/{len(video_names)}: {get_video_name(args.base_video_dir, video_name)}")
        frame_inference(
            predictor=predictor,
            yolo_model=yolo_model,
            box_source=args.box_source,
            base_video_dir=args.base_video_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            yolo_imgsz=args.yolo_imgsz,
            yolo_conf=args.yolo_conf,
            max_yolo_boxes_per_frame=args.max_yolo_boxes_per_frame,
        )


if __name__ == "__main__":
    main()
