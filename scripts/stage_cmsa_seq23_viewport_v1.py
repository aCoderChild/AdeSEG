#!/usr/bin/env python3
"""Stage label-free, lossless sequence 23 inputs for the released CMSA loop.

The author loader requires a label-shaped image to resize saved predictions.
The staged black dummy is generated from known geometry and is not an annotation.
This utility does not run the model or score predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.video import load_manifest
from modeling.seq23_viewport import VIEWPORT, crop_viewport

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (Path.home() / "Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com"
               / "My Drive/AdeSEG/outputs/experiment").resolve()
RESERVE = 23 * 1024**3


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def save_crop(source: Path, destination: Path) -> dict:
    before = sha(source)
    image = image_pixels(source)
    if image.shape != (720, 1280, 3):
        raise ValueError(f"Unexpected source geometry: {source}: {image.shape}")
    cropped = crop_viewport(image)
    Image.fromarray(cropped).save(destination, format="PNG")
    restored = image_pixels(destination)
    if not np.array_equal(cropped, restored) or sha(source) != before:
        raise RuntimeError(f"Crop pixel mismatch or source changed: {source}")
    return {"source_sha256": before, "crop_sha256": sha(destination),
            "crop_rgb_sha256": hashlib.sha256(cropped.tobytes()).hexdigest()}


def stage(manifest: Path, output: Path) -> None:
    output = output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or output.exists():
        raise ValueError("Output must be a new directory under Drive experiment storage")
    manifest_sha = sha(manifest)
    videos = {**load_manifest(manifest, "dev"), **load_manifest(manifest, "test")}
    frames = videos["seq23"]
    if len(frames) < 4:
        raise ValueError("CMSA author loop requires at least four frames")
    if os.statvfs(OUTPUT_ROOT).f_bavail * os.statvfs(OUTPUT_ROOT).f_frsize < RESERVE:
        raise RuntimeError("Insufficient free space to stage CMSA viewport")
    output.mkdir(parents=True)
    images = output / "inputs" / "seq23"
    images.mkdir(parents=True)
    dummy = output / "viewport_dummy_mask.png"
    Image.new("L", (VIEWPORT[2] - VIEWPORT[0], VIEWPORT[3] - VIEWPORT[1]), 0).save(dummy)
    records = []
    for i, frame in enumerate(frames):
        if os.statvfs(OUTPUT_ROOT).f_bavail * os.statvfs(OUTPUT_ROOT).f_frsize < RESERVE:
            raise RuntimeError("Disk reserve reached; partial staging preserved")
        source = frame.image_path.resolve(strict=True)
        source.relative_to((ROOT / "data").resolve())
        target = images / f"{i:06d}.png"
        info = save_crop(source, target)
        records.append({"frame": i, "source_frame_index": frame.frame_index,
                        "source": str(source), "crop": str(target), **info})
    if sha(manifest) != manifest_sha:
        raise RuntimeError("Manifest changed during staging")
    receipt = {"passed": True, "sequence": "seq23", "scope": "Image-only staging; no annotation read.",
               "viewport_xyxy": list(VIEWPORT), "source_size": [1280, 720],
               "manifest": str(manifest.resolve()), "manifest_sha256": manifest_sha,
               "stager_sha256": sha(Path(__file__)), "crop_helper_sha256": sha(ROOT / "modeling/seq23_viewport.py"),
               "dummy_mask": str(dummy), "dummy_mask_sha256": sha(dummy), "records": records}
    (output / "stage.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": True, "frames": len(records), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/polypgen_sequence.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage(args.manifest, args.output)
