"""Frame discovery, image decoding, and box-prompt preprocessing."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG")


def get_numeric_sort_key(name: str):
    parts = re.split(r"(\d+)", name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def is_image_file(path: str) -> bool:
    return Path(path).suffix in IMAGE_EXTENSIONS


def get_video_frame_dir(base_video_dir: str | Path, video_name: str) -> str:
    base_video_dir = Path(base_video_dir)
    if video_name == ".":
        images = base_video_dir / "images"
        return str(images if images.is_dir() else base_video_dir)
    images = base_video_dir / video_name / "images"
    if images.is_dir():
        return str(images)
    polypgen_images = base_video_dir / video_name / f"images_{video_name}"
    return str(polypgen_images if polypgen_images.is_dir() else base_video_dir / video_name)


def get_video_name(base_video_dir: str | Path, video_name: str) -> str:
    if video_name != ".":
        return video_name
    base = Path(base_video_dir).resolve()
    return base.parent.name if base.name == "images" else base.name


def list_video_names(base_video_dir: str | Path) -> list[str]:
    base = Path(base_video_dir)
    images = base / "images"
    if images.is_dir() and any(is_image_file(path.name) for path in images.iterdir()):
        return ["."]
    videos = [path.name for path in base.iterdir() if path.is_dir()]
    if videos:
        return sorted(videos, key=get_numeric_sort_key)
    return ["."] if any(is_image_file(path.name) for path in base.iterdir()) else []


def get_frame_names(video_dir: str | Path) -> list[str]:
    return sorted(
        (path.stem for path in Path(video_dir).iterdir() if is_image_file(path.name)),
        key=get_numeric_sort_key,
    )


def resolve_frame_path(video_dir: str | Path, frame_name: str) -> str:
    for extension in IMAGE_EXTENSIONS:
        path = Path(video_dir) / f"{frame_name}{extension}"
        if path.exists():
            return str(path)
    raise FileNotFoundError(Path(video_dir) / f"{frame_name}.jpg")


def load_rgb_image(frame_path: str | Path) -> Image.Image:
    with Image.open(frame_path) as image:
        return image.convert("RGB")


def load_bgr_image(frame_path: str | Path) -> np.ndarray:
    return np.asarray(load_rgb_image(frame_path))[:, :, ::-1].copy()


def get_data_box_dir(base_video_dir: str | Path, video_name: str) -> Path:
    return Path(base_video_dir) / video_name / f"bbox_{video_name}"


def get_data_boxes(bbox_dir: str | Path, frame_name: str, max_boxes: int):
    bbox_path = Path(bbox_dir) / f"{frame_name}.txt"
    if not bbox_path.is_file():
        return []
    boxes = []
    for line in bbox_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected 5 fields in {bbox_path}: {line}")
        _, x1, y1, x2, y2 = fields
        boxes.append((np.array([x1, y1, x2, y2], dtype=np.float32), 1.0))
    return boxes[:max_boxes] if max_boxes > 0 else boxes


def get_yolo_boxes(yolo_model, frame_path: str | Path, yolo_imgsz: int, yolo_conf: float, max_boxes: int):
    image = load_rgb_image(frame_path)
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


def select_video_names(base_video_dir: str | Path, seq_nums=None, video_list_file: str | Path | None = None) -> list[str]:
    if seq_nums:
        return [f"seq{sequence_number}" for sequence_number in seq_nums]
    if video_list_file:
        return [line.strip() for line in Path(video_list_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    return list_video_names(base_video_dir)
