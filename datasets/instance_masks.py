"""Read persistent instance IDs without binary conversion or uint8 truncation."""

import numpy as np
from PIL import Image


def load_instance_mask(path):
    with Image.open(path) as image:
        if image.format not in ("PNG", "TIFF"):
            raise ValueError("Instance IDs require lossless indexed PNG or TIFF masks")
        labels = np.array(image)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError("Expected a 2-D nonnegative integer instance mask")
    return labels


def instance_prompts(frames, *, object_ids=None, prompt_frames=None, void_id=None):
    """Load only explicit first prompts; later annotations do not guide inference."""
    by_index = {frame.frame_index: index for index, frame in enumerate(frames)}
    if prompt_frames is None:
        initial = load_instance_mask(frames[0].mask_path)
        ids = object_ids if object_ids is not None else [
            int(i) for i in np.unique(initial) if i > 0 and i != void_id]
        prompt_frames = {i: frames[0].frame_index for i in ids}
    elif object_ids is not None and set(object_ids) != set(prompt_frames):
        raise ValueError("Object IDs differ from the explicit prompt schedule")
    prompts = {}
    for object_id, source_frame in prompt_frames.items():
        if (not isinstance(object_id, int) or isinstance(object_id, bool)
                or object_id <= 0 or object_id == void_id):
            raise ValueError("Expected positive, non-void persistent instance IDs")
        if source_frame not in by_index:
            raise ValueError("Prompt refers to a frame outside this manifest sequence")
        index = by_index[source_frame]
        labels = load_instance_mask(frames[index].mask_path)
        mask = labels == object_id
        if not mask.any():
            raise ValueError(f"Instance {object_id} is absent from its prompt frame")
        prompts.setdefault(index, {})[object_id] = mask
    if not prompts:
        raise ValueError("No prompted instances")
    return prompts
