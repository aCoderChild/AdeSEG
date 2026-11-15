"""Use one archived box, point pair and mask across distinct video predictors.

No detector, annotation, confidence rule or history pruning is introduced.
The generated point mask is diagnostic; the archived mask is always committed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from modeling.warm_automatic_prompt_v2 import apply_archived_warm_prompt_v2
from scripts.memory_token_io import digest


def archived_warm_components(sequence, start, mask, provenance):
    """Read only the hash-bound automatic prompt, in its recorded view."""
    path = Path(provenance['prompt_file']).resolve(strict=True).parent/'prompt.json'
    expected = provenance['prompt_metadata_sha256']
    if digest(path) != expected: raise ValueError('Archived prompt metadata changed')
    raw = path.read_bytes();data = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != expected or digest(path) != expected or data.get('first_detection_index') != start:
        raise ValueError('Archived prompt metadata or initialization frame differs')
    box_key = 'box_viewport_xyxy' if sequence == 'seq23' else 'box'
    points_key = 'points_viewport' if sequence == 'seq23' else 'points'
    box = np.asarray(data.get(box_key), dtype=np.float32)
    points = np.asarray(data.get(points_key), dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if box.shape != (4,) or points.shape != (2,2) or mask.ndim != 2 or not mask.any():
        raise ValueError('Archived box, point pair or mask is unavailable')
    if not np.isfinite(box).all() or not np.isfinite(points).all():
        raise ValueError('Archived coordinates are nonfinite')
    height, width = mask.shape
    roi = provenance['roi_xyxy']
    if (len(roi) != 4 or any(type(x) is not int for x in roi) or
            roi[2]-roi[0] != width or roi[3]-roi[1] != height):
        raise ValueError('Archived viewport and mask geometry differ')
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height and
            ((points[:,0] >= 0) & (points[:,0] < width) &
             (points[:,1] >= 0) & (points[:,1] < height)).all()):
        raise ValueError('Archived coordinates lie outside the current view')
    return {'box': box, 'points': points, 'mask': mask.copy(),
            'prompt_metadata_sha256': expected, 'roi_xyxy': list(roi)}


def apply_warm_author_initialization(model, state, frame, components):
    """Apply author prompt APIs; returned masks only document the warm call."""
    return apply_archived_warm_prompt_v2(model, state, frame, components['box'],
                                         components['points'], components['mask'])
