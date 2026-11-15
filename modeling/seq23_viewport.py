"""Fixed, visually inspected main viewport for PolypGen sequence 23.

This is a sequence-specific preprocessing recipe, not a viewport detector.
Coordinates use half-open intervals in the original 1280 by 720 image.
"""
from pathlib import Path
import numpy as np
from PIL import Image

IMAGE_SIZE = (1280, 720)
VIEWPORT = (446, 0, 1280, 720)

def crop_viewport(image):
    image = np.asarray(image)
    if image.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3) or image.dtype != np.uint8:
        raise ValueError('Sequence 23 requires a 1280x720 RGB uint8 image')
    x0, y0, x1, y1 = VIEWPORT
    return image[y0:y1, x0:x1].copy()

def restore_mask(mask):
    mask = np.asarray(mask)
    x0, y0, x1, y1 = VIEWPORT
    if mask.shape != (y1-y0, x1-x0):
        raise ValueError('Mask does not match the fixed viewport')
    full = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=mask.dtype)
    full[y0:y1, x0:x1] = mask
    return full

def stage_cropped_frames(frames, directory):
    """Write lossless cropped images; SAM2 requires numeric .jpg filenames."""
    directory = Path(directory)
    for index, frame in enumerate(frames):
        with Image.open(frame.image_path) as image:
            cropped = crop_viewport(np.asarray(image.convert('RGB')))
        Image.fromarray(cropped).save(directory / f'{index:06d}.jpg', format='PNG')
