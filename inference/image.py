"""Oracle-box MedSAM2 inference for two-region static images."""

from pathlib import Path

import numpy as np
from PIL import Image

from adenoid.io import make_overlay, save_overlay
from datasets.refuge2 import iter_refuge2_samples


def box_from_mask(mask: np.ndarray) -> np.ndarray:
    y, x = np.where(np.asarray(mask) > 0)
    if not x.size:
        raise ValueError("A GT-derived box requires a non-empty mask.")
    return np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float32)


def run_refuge2_oracle_boxes(predictor, data_root, split, output_dir, start=0, limit=None, save_overlays=True):
    """Predict disc and cup separately from their GT-derived boxes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    for index, sample in enumerate(iter_refuge2_samples(data_root, split)):
        if index < start:
            continue
        if limit is not None and index >= start + limit:
            break
        image = np.asarray(Image.open(sample.image_path).convert("RGB"))
        predictor.set_image(image)
        labels = np.full(image.shape[:2], 255, dtype=np.uint8)
        predictions = {}
        for name, value in (("disc", 128), ("cup", 0)):
            masks, _, _ = predictor.predict(box=box_from_mask(sample.masks[name]), multimask_output=False)
            predictions[name] = masks[0] > 0
            labels[predictions[name]] = value
            if save_overlays:
                save_overlay(make_overlay(image, sample.masks[name], predictions[name]), overlay_dir / f"{sample.metadata['image_id']}_{name}.png")
        Image.fromarray(labels).save(output_dir / f"{sample.metadata['image_id']}.png")
