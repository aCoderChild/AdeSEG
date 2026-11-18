#!/usr/bin/env python3
"""GT-content-free CMSA-Net comparison using the pinned, unchanged author model.

The original author test loop loads GT masks and opens GT files to resize saved
predictions. This wrapper supplies zero placeholders instead and uses the
corresponding *source image* size for export. No evaluation mask bytes are read
until after inference, when independent scoring is performed.

This is a NEW comparison protocol. Do not assign the old author-runner SHA or
reuse its saved Dice without rerunning.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.video import load_manifest
from scripts import run_cmsa_polypgen as historical


class ImageOnlyCMSADataset(Dataset):
    """Apply the original image transforms without opening GT masks."""

    def __init__(self, original_dataset):
        self.original = original_dataset

    def __len__(self):
        return len(self.original.video_train_list)

    def __getitem__(self, index):
        pairs = self.original.video_train_list[index]
        images = []
        for image_path, _mask_path in pairs:
            with Image.open(image_path) as im:
                images.append(im.convert("RGB"))
        # Only dummy labels reach the author's paired transforms; images remain
        # identical to those of the original inference dataloader.
        placeholders = [Image.new("L", im.size, 0) for im in images]
        transformed = self.original.img_label_transform(images, placeholders)
        if not isinstance(transformed, tuple) or len(transformed) != 2:
            raise RuntimeError("Unexpected CMSA-Net author transform output")
        images, dummy_labels = transformed
        if not all(isinstance(x, torch.Tensor) for x in (*images, *dummy_labels)):
            raise TypeError("CMSA transforms must return torch tensors")
        return torch.stack(images), torch.stack(dummy_labels), pairs


def image_size_index(manifest):
    """Build output-size lookups from RGB images only; never open a mask."""
    mapping = {}
    for split in ("dev", "test"):
        for frames in load_manifest(manifest, split).values():
            for item in frames:
                key = str(item.mask_path.resolve())
                image = str(item.image_path.resolve())
                old = mapping.setdefault(key, image)
                if old != image:
                    raise ValueError("Mask path maps to multiple source images")
    return mapping


def run(args):
    native_images = image_size_index(args.manifest)
    old_dataset = historical.author_dataset
    old_author = historical.load_author
    patched_author = []

    def image_only_dataset(videos, normalization="released"):
        return ImageOnlyCMSADataset(old_dataset(videos, normalization))

    def verified_author():
        module = old_author()
        original_save = module.safe_save
        patched_author.append((module, original_save))

        def save_from_source_image(prediction, save_path, mask_path, to_resize=True):
            key = str(Path(mask_path).resolve())
            if key not in native_images:
                raise ValueError("No source image for output frame")
            with Image.open(native_images[key]) as image:
                image_size = image.size
            if to_resize:
                prediction = prediction.resize(image_size)
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            prediction.save(out)

        module.safe_save = save_from_source_image
        return module

    try:
        with patch.object(historical, "author_dataset", image_only_dataset), \
             patch.object(historical, "load_author", verified_author):
            historical.run(args)
    finally:
        for module, old_save in patched_author:
            module.safe_save = old_save

    receipt = args.output / "setup.json"
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data.update({
        "protocol": "image_only_cmsa_comparison",
        "inference_reads_ground_truth_mask_bytes": False,
        "export_resize_reference": "source_rgb_image",
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "historical_runner_unmodified": True,
        "note_new_protocol": (
            "Dummy GT tensors are passed to the author test loop. The author "
            "prediction and dynamic reference update use image tensors only; "
            "saved output size is derived from the original RGB image. "
            "Historical scores cannot be claimed without rerunning."
        ),
    })
    receipt.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=historical.ROOT / "data/polypgen_sequence.jsonl")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--normalization", choices=("released", "channel"),
        default="released")
    parser.add_argument("--sequence", action="append")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
