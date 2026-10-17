"""Region taxonomy and budget matching for the existing memory-attention hook."""

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

CATEGORIES = ("boundary", "foreground", "near_background", "far_background")


def token_categories(mask, grid=(64, 64), band=1, thresholds=(0.1, 0.9), image=None):
    coverage = cv2.resize(
        np.asarray(mask, np.float32), grid[::-1], interpolation=cv2.INTER_AREA
    )
    low, high = thresholds
    boundary = ((coverage > low) & (coverage < high)).astype(np.uint8)
    if band:
        boundary = cv2.dilate(boundary, np.ones((2 * band + 1,) * 2, np.uint8))
    boundary = boundary.astype(bool)
    near = cv2.dilate(boundary.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    categories = np.full(grid, 3, dtype=np.uint8)
    categories[(coverage >= high) & ~boundary] = 1
    categories[(coverage <= low) & near & ~boundary] = 2
    categories[boundary] = 0
    artifact = np.zeros(grid, bool)
    if image is not None:
        # Reuse analyze_illumination.image_stats's actual luminance rule.
        image = np.asarray(image)
        luma = np.sum(image * np.array([0.2126, 0.7152, 0.0722]), axis=-1) / 255
        fraction = cv2.resize(
            (luma > 0.95).astype(np.float32), grid[::-1], interpolation=cv2.INTER_AREA
        )
        artifact = fraction >= 0.25
    return {"categories": categories, "coverage": coverage, "artifact": artifact}


def matched_indices(categories, category, k, seed):
    """No replacement and no silent budget reduction. None means infeasible."""
    candidates = np.flatnonzero(np.asarray(categories).ravel() == category)
    if k <= 0 or len(candidates) < k:
        return None
    return np.random.default_rng(seed).choice(candidates, size=k, replace=False)


def spatially_matched_indices(categories, reference, seed):
    """Random positions matched to each reference's integer distance from object.

    Sampling includes the reference support; excluding boundary tokens would
    make exact distance matching impossible in many frames. Report overlap.
    """
    foreground = np.isin(categories, [0, 1])
    frames = foreground if foreground.ndim == 3 else foreground[None]
    distance = np.concatenate(
        [
            np.floor(distance_transform_edt(~frame)).astype(int).ravel()
            for frame in frames
        ]
    )
    rng = np.random.default_rng(seed)
    selected = []
    for value, count in zip(*np.unique(distance[reference], return_counts=True)):
        candidates = np.flatnonzero(distance == value)
        if len(candidates) < count:
            return None
        selected.extend(rng.choice(candidates, count, replace=False))
    return np.asarray(selected, dtype=int)


def indices_to_masks(units, selected):
    masks = {}
    offset = 0
    for unit in units:
        length = unit["end"] - unit["start"]
        mask = (selected >= offset) & (selected < offset + length)
        local = np.zeros(length, bool)
        local[selected[mask] - offset] = True
        masks[unit["frame_idx"]] = torch.from_numpy(local)
        offset += length
    return masks


def indices_to_value_replacements(units, selected, replacement_values):
    """Map selected global spatial indices to per-frame value replacements.

    ``replacement_values`` contains one ``memory_dim`` vector per selected key,
    in the order supplied by ``selected``.  Unselected entries are ignored by
    the attention hook, so they are left zero rather than copied from memory.
    """
    selected = np.asarray(selected, dtype=int).reshape(-1)
    values = torch.as_tensor(replacement_values)
    if values.ndim != 2 or values.shape[0] != len(selected):
        raise ValueError("Need one replacement vector for every selected token")
    order = np.argsort(selected)
    selected, values = selected[order], values[torch.as_tensor(order, device=values.device)]
    replacements = {}
    offset = 0
    cursor = 0
    for unit in units:
        length = unit["end"] - unit["start"]
        local = np.zeros(length, dtype=bool)
        fill = torch.zeros(length, values.shape[1], dtype=values.dtype)
        while cursor < len(selected) and selected[cursor] < offset + length:
            if selected[cursor] < offset:
                raise ValueError("Selected token index is outside the memory spans")
            index = selected[cursor] - offset
            local[index] = True
            fill[index] = values[cursor]
            cursor += 1
        replacements[unit["frame_idx"]] = (torch.from_numpy(local), fill)
        offset += length
    if cursor != len(selected):
        raise ValueError("Selected token index is outside the memory spans")
    return replacements


def query_regions(foreground, height, width):
    """Three disjoint query regions from previous prediction, without GT."""
    mask = foreground.reshape(-1, 1, height, width).float()
    outer = torch.nn.functional.max_pool2d(mask, 3, 1, 1) > 0
    inner = ~(torch.nn.functional.max_pool2d(1 - mask, 3, 1, 1) > 0)
    boundary = outer & ~inner
    return {
        "foreground": inner.flatten(1),
        "boundary": boundary.flatten(1),
        "background": (~outer).flatten(1),
    }


def category_attention(layers, units, maps, queries, conditioning=None):
    """Per-layer/query-region key mass; pointer tokens are reported separately."""
    output = []
    for weights in layers:
        regions = {}
        for name, query_mask in queries.items():
            query_mask = query_mask.to(weights.device).bool()
            selected = weights.permute(0, 2, 1, 3)[query_mask]
            if not selected.numel():
                regions[name] = None
                continue
            masses = np.zeros(4)
            spatial_end = 0
            for unit in units:
                spatial_end = max(spatial_end, unit["end"])
                if (
                    conditioning is not None
                    and bool(unit["conditioning"]) != conditioning
                ):
                    continue
                keys = selected[..., unit["start"] : unit["end"]]
                labels = torch.as_tensor(
                    maps[unit["frame_idx"]], device=keys.device
                ).flatten()
                for category in range(4):
                    masses[category] += float(
                        keys[..., labels == category].sum(-1).mean()
                    )
            regions[name] = dict(zip(CATEGORIES, masses.tolist()))
            regions[name]["pointers"] = float(
                selected[..., spatial_end:].sum(-1).mean()
            )
        output.append(regions)
    return output


def persistent_region_rollout(
    predictor,
    state,
    controller,
    mask_provider,
    category,
    conditioning=False,
    seed=0,
    diagnostics=None,
):
    """Keep the policy active while native propagation writes its predictions.

    ``mask_provider(frame_idx)`` supplies a stored predicted mask for a deployable
    policy, or an oracle mask for a labeled analysis. It is called at read time,
    so newly written, possibly erroneous predictions enter future decisions.
    This function does not restore or replace memory after each prediction.
    """
    if category not in (*CATEGORIES, "random"):
        raise ValueError(f"Unknown region category: {category}")
    previous_policy = controller.token_mask_policy
    previous_masks = controller.excluded_token_masks
    previous_value_policy = controller.token_value_replacement_policy

    def policy(units):
        chosen = [unit for unit in units if bool(unit["conditioning"]) == conditioning]
        if not chosen:
            return {}
        maps = []
        for unit in chosen:
            side = int(np.sqrt(unit["end"] - unit["start"]))
            if side * side != unit["end"] - unit["start"]:
                raise ValueError(
                    "Persistent region policy requires a square spatial grid"
                )
            maps.append(
                token_categories(mask_provider(unit["frame_idx"]), (side, side))[
                    "categories"
                ].ravel()
            )
        labels = np.concatenate(maps)
        k = int((labels == 0).sum())
        target = max(unit["frame_idx"] + unit["temporal_distance"] for unit in chosen)
        if category == "random":
            selected = (
                np.random.default_rng(seed + target).choice(
                    len(labels), k, replace=False
                )
                if k
                else None
            )
        else:
            selected = matched_indices(
                labels, CATEGORIES.index(category), k, seed + target
            )
        if diagnostics is not None:
            diagnostics.append(
                {"frame": target, "k": k, "feasible": selected is not None}
            )
        return {} if selected is None else indices_to_masks(chosen, selected)

    controller.token_mask_policy = policy
    try:
        yield from predictor.propagate_in_video(state)
    finally:
        controller.token_mask_policy = previous_policy
        controller.excluded_token_masks = previous_masks
        controller.token_value_replacement_policy = previous_value_policy
