"""REFUGE2 image/mask adapter with explicit disc and cup labels."""

from pathlib import Path

import numpy as np
from PIL import Image

from datasets.common import FrameSample


def iter_refuge2_samples(data_root: str | Path, split: str):
    root = Path(data_root) / split
    image_dir, mask_dir = root / "images", root / "mask"
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        mask_path = next((mask_dir / f"{image_path.stem}{suffix}" for suffix in (".png", ".bmp", ".jpg") if (mask_dir / f"{image_path.stem}{suffix}").is_file()), None)
        if mask_path is None:
            raise FileNotFoundError(f"Missing REFUGE2 mask for {image_path.name}")
        labels = np.asarray(Image.open(mask_path).convert("L"))
        # REFUGE2 masks encode cup=0, disc rim=128, and background=255.
        yield FrameSample(0, image_path, {"disc": labels != 255, "cup": labels == 0}, {"split": split, "image_id": image_path.stem})
