"""Read one source-bound frame in the exact refiner viewport, without labels."""
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from modeling.seq23_viewport import VIEWPORT


def load_aligned_rgb(frame, source_record, sequence, ordinal, roi_xyxy, expected_view_shape):
    if frame.video_id != sequence or type(ordinal) is not int or ordinal < 0 or source_record.get('frame') != ordinal \
            or source_record.get('source_frame_index') != frame.frame_index:
        raise ValueError('RGB frame ordinal or source index differs')
    expected = source_record.get('image_sha256')
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('RGB frame hash is missing')
    path = Path(frame.image_path).resolve(strict=True)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('RGB frame bytes differ')
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert('RGB')
        width, height = image.size
        expected_roi = list(VIEWPORT) if sequence == 'seq23' else [0, 0, width, height]
        if list(roi_xyxy) != expected_roi or not (0 <= expected_roi[0] < expected_roi[2] <= width and
                                                   0 <= expected_roi[1] < expected_roi[3] <= height):
            raise ValueError('RGB viewport differs')
        pixels = np.asarray(image.crop(tuple(expected_roi)), dtype=np.uint8).copy()
    if pixels.shape[:2] != tuple(expected_view_shape):
        raise ValueError('RGB and refiner viewport dimensions differ')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('RGB frame changed during decoding')
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float() / 255.0
