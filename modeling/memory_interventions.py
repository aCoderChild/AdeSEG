"""Native SAM2 mask corruption and spatial-memory re-encoding.

Extracted unchanged from the superseded utility-label runner.
"""

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from datasets.io import load_label_mask


def corrupt_mask(mask, level, kind, frame_index):
    """Apply the frozen cumulative c1--c4 E3 corruption schedule.

    ``level`` uses the preregistered normalized encoding: 0.25, 0.50, 0.75,
    and 1.00 correspond to c1, c2, c3, and c4.  Named single operators remain
    available for diagnostic ablations, but ``mixed`` is always cumulative.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    if not 0 <= level <= 1:
        raise ValueError("--corruption-level must be in [0, 1]")
    if level == 0:
        return mask.astype(bool)
    if kind == "mixed":
        stage = min(4, max(1, int(np.ceil(level * 4))))
        result = mask.copy()
        # c1: deterministic +/- 3 px boundary perturbation at memory-mask
        # resolution. Alternate the sign across frames without changing stage.
        boundary_kernel = np.ones((7, 7), np.uint8)
        result = (
            cv2.erode(result, boundary_kernel)
            if frame_index % 2 == 0 else cv2.dilate(result, boundary_kernel)
        )
        if stage >= 2:
            rows, columns = np.nonzero(result)
            if rows.size:
                height = max(1, int(rows.max() - rows.min() + 1))
                width = max(1, int(columns.max() - columns.min() + 1))
                fraction = 0.05 + 0.05 * ((frame_index % 3) / 2)
                dy, dx = max(1, round(height * fraction)), max(1, round(width * fraction))
                shifted = np.zeros_like(result)
                shifted[dy:, dx:] = result[:-dy, :-dx]
                result = shifted
        if stage >= 3:
            columns = np.flatnonzero(result.any(axis=0))
            if columns.size:
                deletion_fraction = 0.20 + 0.05 * (frame_index % 3)
                delete_width = max(1, round(columns.size * deletion_fraction))
                result[:, columns[-delete_width:]] = 0
        if stage >= 4:
            # Deterministic false-positive blob. The current implementation has
            # no image-similarity search, so record this as the c4 geometric
            # proxy rather than claiming a distractor-selected blob.
            centre = (int(mask.shape[1] * 0.2), int(mask.shape[0] * 0.2))
            axes = (max(1, round(mask.shape[1] * 0.05)),
                    max(1, round(mask.shape[0] * 0.05)))
            cv2.ellipse(result, centre, axes, 0, 0, 360, 1, -1)
        return result.astype(bool)
    radius = max(1, round(1 + 10 * level))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    if kind == "erosion":
        result = cv2.erode(mask, kernel)
    elif kind == "dilation":
        result = cv2.dilate(mask, kernel)
    elif kind == "shift":
        dy, dx = round(mask.shape[0] * 0.2 * level), round(mask.shape[1] * 0.2 * level)
        result = np.zeros_like(mask)
        result[dy:, dx:] = mask[: mask.shape[0] - dy, : mask.shape[1] - dx]
    elif kind == "blob":
        result = mask.copy()
        centre = (int(mask.shape[1] * 0.2), int(mask.shape[0] * 0.2))
        axes = (max(1, round(mask.shape[1] * 0.12 * level)),
                max(1, round(mask.shape[0] * 0.12 * level)))
        cv2.ellipse(result, centre, axes, 0, 0, 360, 1, -1)
    elif kind == "deletion":
        result = mask.copy()
        cutoff = round(mask.shape[1] * (1 - 0.8 * level))
        result[:, cutoff:] = 0
    else:
        raise ValueError(kind)
    return result.astype(bool)


def replace_spatial_memory(predictor, state, output_dict, frames, frame_index, label_id,
                           all_label_ids, source, corruption_level, corruption_type,
                           mask_override=None):
    if source == "predicted":
        return
    target = (load_label_mask(frames[frame_index].mask_path, all_label_ids) == label_id
              if mask_override is None else np.asarray(mask_override, dtype=bool))
    if source == "corrupted" and mask_override is None:
        target = corrupt_mask(target, corruption_level, corruption_type, frame_index)
    logits = torch.from_numpy(target).to(predictor.device).float()[None, None]
    # ``_run_memory_encoder`` receives the high-resolution SAM mask tensor,
    # not an original-video-resolution mask.  ``add_new_mask`` normally does
    # this resize internally; direct counterfactual memory replacement must
    # preserve the same contract explicitly.
    logits = F.interpolate(
        logits, size=(predictor.image_size, predictor.image_size),
        mode="bilinear", align_corners=False,
    ) * 20 - 10
    score = torch.full((1, 1), 10.0 if target.any() else -10.0, device=predictor.device)
    features, position = predictor._run_memory_encoder(
        inference_state=state, frame_idx=frame_index, batch_size=1,
        high_res_masks=logits, object_score_logits=score, is_mask_from_pts=True,
    )
    storage = "cond_frame_outputs" if frame_index in output_dict["cond_frame_outputs"] else "non_cond_frame_outputs"
    output_dict[storage][frame_index]["maskmem_features"] = features
    output_dict[storage][frame_index]["maskmem_pos_enc"] = position
