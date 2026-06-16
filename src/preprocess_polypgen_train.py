#!/usr/bin/env python3
"""Create an augmented PolypGen training folder without modifying the test data."""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def numeric_sort_key(name: str):
    parts = re.split(r"(\d+)", name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def list_sequences(data_root: Path) -> list[str]:
    return sorted(
        [p.name for p in data_root.iterdir() if p.is_dir() and (p / "images").is_dir()],
        key=numeric_sort_key,
    )


def list_frame_names(image_dir: Path) -> list[str]:
    return sorted(
        [p.stem for p in image_dir.iterdir() if p.suffix in IMAGE_EXTS],
        key=numeric_sort_key,
    )


def resolve_file(folder: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(folder / f"{stem}.jpg")


def load_mask(path: Path) -> Image.Image:
    mask = Image.open(path).convert("L")
    arr = np.array(mask)
    binary = (arr > (0 if arr.max() <= 1 else 127)).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def augment_pair(
    image: Image.Image,
    mask: Image.Image,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image]:
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    if rng.random() < 0.2:
        image = ImageOps.flip(image)
        mask = ImageOps.flip(mask)

    angle = rng.uniform(-12.0, 12.0)
    translate = (
        rng.uniform(-0.04, 0.04) * image.width,
        rng.uniform(-0.04, 0.04) * image.height,
    )
    scale = rng.uniform(0.90, 1.10)
    image = affine(image, angle, translate, scale, resample=Image.BILINEAR, fill=0)
    mask = affine(mask, angle, translate, scale, resample=Image.NEAREST, fill=0)

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.80, 1.20))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.80, 1.20))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.85, 1.15))
    return image, mask


def affine(
    image: Image.Image,
    angle: float,
    translate: tuple[float, float],
    scale: float,
    resample: int,
    fill: int | tuple[int, int, int],
) -> Image.Image:
    # PIL's high-level rotate is enough here and keeps the output size fixed.
    image = image.rotate(angle, resample=resample, fillcolor=fill)
    if scale != 1.0:
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        scaled = image.resize(new_size, resample=resample)
        canvas = Image.new(image.mode, image.size, fill)
        left = (image.width - scaled.width) // 2
        top = (image.height - scaled.height) // 2
        canvas.paste(scaled, (left, top))
        image = canvas
    return ImageOps.expand(image, border=0).transform(
        image.size,
        Image.AFFINE,
        (1, 0, -translate[0], 0, 1, -translate[1]),
        resample=resample,
        fillcolor=fill,
    )


def bbox_from_mask(mask: Image.Image) -> tuple[int, int, int, int] | None:
    arr = np.array(mask) > 0
    ys, xs = np.where(arr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def save_pair(
    image: Image.Image,
    mask: Image.Image,
    image_path: Path,
    mask_path: Path,
) -> tuple[int, int, int, int] | None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, quality=95)
    mask.save(mask_path, quality=95)
    return bbox_from_mask(mask)


def write_bboxes(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)


def create_augmented_train(args) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    bbox_output_root = Path(args.bbox_output_root)

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    if bbox_output_root.exists() and args.overwrite:
        shutil.rmtree(bbox_output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bbox_output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    sequences = [f"seq{n}" for n in args.seq_nums] if args.seq_nums else list_sequences(input_root)

    total_written = 0
    for seq_name in sequences:
        image_dir = input_root / seq_name / "images"
        mask_dir = input_root / seq_name / "masks"
        out_image_dir = output_root / seq_name / "images"
        out_mask_dir = output_root / seq_name / "masks"
        bbox_rows = []

        for frame_name in list_frame_names(image_dir):
            image = Image.open(resolve_file(image_dir, frame_name)).convert("RGB")
            mask = load_mask(resolve_file(mask_dir, frame_name))

            variants = []
            if args.include_original:
                variants.append((frame_name, image, mask))
            for aug_idx in range(args.augmentations_per_frame):
                aug_image, aug_mask = augment_pair(image.copy(), mask.copy(), rng)
                variants.append((f"{frame_name}_aug{aug_idx:02d}", aug_image, aug_mask))

            for out_name, out_image, out_mask in variants:
                bbox = save_pair(
                    out_image,
                    out_mask,
                    out_image_dir / f"{out_name}.jpg",
                    out_mask_dir / f"{out_name}.jpg",
                )
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                bbox_rows.append(
                    {"frame_idx": out_name, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                )
                total_written += 1

        write_bboxes(bbox_output_root / seq_name / "bboxes.csv", bbox_rows)
        print(f"{seq_name}: wrote {len(bbox_rows)} training frames")

    print(f"Done. Train images/masks: {output_root}")
    print(f"Done. Train bboxes: {bbox_output_root}")
    print(f"Total frame variants with foreground boxes: {total_written}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="data/polypgen")
    parser.add_argument("--output_root", default="data/polypgen_train")
    parser.add_argument("--bbox_output_root", default="data/bbox_train")
    parser.add_argument("--seq_nums", type=int, nargs="*", default=None)
    parser.add_argument("--augmentations_per_frame", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--include_original", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    create_augmented_train(parse_args())
