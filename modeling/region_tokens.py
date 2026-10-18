"""Region taxonomy and budget matching for the existing memory-attention hook."""

import cv2
from contextlib import contextmanager
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

CATEGORIES = ("boundary", "foreground", "near_background", "far_background")


@contextmanager
def capture_chunked_region_attention(
    memory_attention, units, maps, chunk_size=32, failure_dir=None
):
    """Observe native post-RoPE attention without retaining Q×K matrices.

    Adapt the bundled SAM2 Attention._attention reconstruction, in query chunks.
    Always return its original SDPA output; this is measurement, not a new
    attention implementation. Mass averages all queries/heads, not GT queries.
    """
    records, originals = [], []
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    try:
        for layer_index, layer in enumerate(memory_attention.layers):
            module = layer.cross_attn_image
            originals.append((module, module.__dict__.get("_attention")))
            native = module._attention

            def observe(
                q,
                k,
                v,
                attn_mask=None,
                _native=native,
                _index=layer_index,
                _module=module,
            ):
                if _module.training:
                    raise ValueError("Attention observation requires eval mode")
                out = _native(q, k, v, attn_mask=attn_mask)
                key_mass = torch.zeros(
                    k.shape[-2], device=q.device, dtype=torch.float32
                )
                maximum_error = 0.0
                worst_start = 0
                with torch.no_grad():
                    for start in range(0, q.shape[-2], chunk_size):
                        end = start + chunk_size
                        logits = q[..., start:end, :].float() @ k.float().transpose(
                            -2, -1
                        )
                        logits /= q.shape[-1] ** 0.5
                        if attn_mask is not None:
                            mask = (
                                attn_mask
                                if attn_mask.shape[-2] == 1
                                else attn_mask[..., start:end, :]
                            )
                            if mask.dtype == torch.bool:
                                logits.masked_fill_(~mask, float("-inf"))
                            else:
                                logits += mask
                        weights = logits.softmax(-1)
                        key_mass += weights.sum(dim=(0, 1, 2))
                        manual = weights @ v.float()
                        if (
                            not torch.isfinite(manual).all()
                            or not torch.isfinite(out[..., start:end, :]).all()
                        ):
                            raise ValueError("Nonfinite attention reconstruction")
                        error = float(
                            (manual - out[..., start:end, :].float()).abs().max()
                        )
                        if error > maximum_error:
                            maximum_error, worst_start = error, start
                if failure_dir is not None and maximum_error > 1e-4:
                    from pathlib import Path

                    destination = Path(failure_dir)
                    destination.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "q": q[..., worst_start : worst_start + chunk_size, :]
                            .detach()
                            .cpu(),
                            "k": k.detach().cpu(),
                            "v": v.detach().cpu(),
                            "native": out[
                                ..., worst_start : worst_start + chunk_size, :
                            ]
                            .detach()
                            .cpu(),
                            "attn_mask": attn_mask.detach().cpu()
                            if attn_mask is not None
                            else None,
                            "query_start": worst_start,
                            "mps_max_abs": maximum_error,
                        },
                        destination / f"layer{_index}.pt",
                    )
                key_mass /= q.shape[0] * q.shape[1] * q.shape[-2]
                key_mass = key_mass.cpu().numpy()
                regions = []
                for unit in units:
                    labels = np.asarray(maps[unit["frame_idx"]]).reshape(-1)
                    mass = key_mass[unit["start"] : unit["end"]]
                    if labels.size != mass.size:
                        raise ValueError("Region labels do not match spatial keys")
                    regions.append(
                        {
                            "frame": unit["frame_idx"],
                            "conditioning": bool(unit["conditioning"]),
                            "regions": {
                                name: {
                                    "tokens": int((labels == i).sum()),
                                    "mass": float(mass[labels == i].sum()),
                                }
                                for i, name in enumerate(CATEGORIES)
                            },
                        }
                    )
                spatial_end = max(u["end"] for u in units)
                records.append(
                    {
                        "layer": _index,
                        "units": regions,
                        "pointer_tokens": len(key_mass) - spatial_end,
                        "pointer_mass": float(key_mass[spatial_end:].sum()),
                        "total_mass": float(key_mass.sum()),
                        "reconstruction_max_abs": maximum_error,
                        "query_pool": "all native queries, averaged over heads",
                    }
                )
                return out

            module._attention = observe
        yield records
    finally:
        for module, previous in originals:
            if previous is None:
                module.__dict__.pop("_attention", None)
            else:
                module._attention = previous


def region_geometry(mask, grid=(64, 64), thresholds=(0.1, 0.9)):
    """V2 symmetric distance strata, reusing the area-coverage taxonomy.

    Distance is between token centres and the inner contour of coverage >= .5.
    Empty masks have undefined distances, not an invented foreground contour.
    Mixed coverage is a mask property, not an image-edge detector.
    """
    coverage = token_categories(mask, grid, thresholds=thresholds)["coverage"]
    inside = coverage >= 0.5
    mixed = (coverage > thresholds[0]) & (coverage < thresholds[1])
    if not inside.any():
        distance = np.full(grid, np.nan)
    else:
        eroded = cv2.erode(
            inside.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        contour = inside & ~eroded
        distance = distance_transform_edt(~contour)
    # Symmetric absolute bins: 0, (0,1], (1,3], >3 token-centre units.
    bins = np.digitize(distance, [0, 1, 3], right=True).astype(np.int8)
    bins[~np.isfinite(distance)] = -1
    return {
        "coverage": coverage,
        "inside": inside,
        "mixed": mixed,
        "absolute_distance": distance,
        "distance_bin": bins,
        "signed_distance": np.where(inside, -distance, distance),
    }


def disjoint_distance_match(target, eligible, distance_bins, seed=0):
    """Match counts per frame/bin without replacement, overlap or relaxation.

    Arrays are [H,W] or [frames,H,W]. None means the full budget is infeasible.
    """
    target = np.asarray(target, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    bins = np.asarray(distance_bins)
    if target.shape != eligible.shape or bins.shape != target.shape:
        raise ValueError("Target, eligibility and distance shapes must match")
    if target.ndim not in (2, 3):
        raise ValueError("Expected [H,W] or [frames,H,W]")
    if not target.any() or (bins[target] < 0).any():
        return None
    if target.ndim == 2:
        target, eligible, bins = target[None], eligible[None], bins[None]
    rng, selected = np.random.default_rng(seed), []
    for frame, (support, pool, strata) in enumerate(zip(target, eligible, bins)):
        pool = pool & ~support
        for value, count in zip(*np.unique(strata[support], return_counts=True)):
            candidates = np.flatnonzero(pool.ravel() & (strata.ravel() == value))
            if len(candidates) < count:
                return None
            selected.extend(
                rng.choice(candidates, int(count), replace=False) + frame * support.size
            )
    return np.asarray(selected, dtype=int)


def subsampled_disjoint_match(target, eligible, strata, cap=64, seed=0):
    """Explicit common-support subsampling, not a relaxed full-budget match.

    Reuse disjoint_distance_match after sampling at most cap target keys with
    quotas supported by BOTH groups in each memory-frame/exact-distance stratum.
    """
    target, eligible, strata = (
        np.asarray(target, bool),
        np.asarray(eligible, bool),
        np.asarray(strata),
    )
    if (
        target.shape != eligible.shape
        or target.shape != strata.shape
        or target.ndim != 3
        or cap < 1
    ):
        raise ValueError("Expected matching [frames,H,W] arrays and a positive cap")
    eligible = eligible & ~target
    rng = np.random.default_rng(seed)
    groups, slots = [], []
    for frame in range(len(target)):
        for value in np.unique(strata[frame][target[frame]]):
            if value < 0:
                continue
            left = np.flatnonzero(target[frame] & (strata[frame] == value))
            right = np.flatnonzero(eligible[frame] & (strata[frame] == value))
            capacity = min(len(left), len(right))
            if capacity:
                slots.extend([len(groups)] * capacity)
                groups.append((frame, left))
    metadata = {
        "target_support": int(target.sum()),
        "control_support": int(eligible.sum()),
        "common_capacity": len(slots),
        "cap": cap,
        "selected": min(cap, len(slots)),
    }
    if not slots:
        return None, None, metadata
    quotas = np.bincount(
        rng.choice(slots, metadata["selected"], replace=False), minlength=len(groups)
    )
    sampled = np.zeros_like(target)
    for quota, (frame, candidates) in zip(quotas, groups):
        sampled[frame].flat[rng.choice(candidates, int(quota), replace=False)] = True
    control = disjoint_distance_match(sampled, eligible, strata, seed=seed)
    if control is None:
        raise RuntimeError("Common-support quota construction failed")
    return np.flatnonzero(sampled), control, metadata


def regional_bias_candidates(categories, seed=None):
    """Eight ±1-logit regional candidates; optional within-frame permutation.

    Same candidate count, per-frame token norm and value histogram in the null.
    Zero is evaluated separately and available to both searches. No GT masks.
    """
    result = []
    for category in range(4):
        for direction in (-1, 1):
            biases = {}
            for index, labels in categories.items():
                value = direction * (np.asarray(labels).reshape(-1) == category).astype(
                    np.float32
                )
                value -= value.mean()
                if seed is not None:
                    value = np.random.default_rng(
                        np.random.SeedSequence(
                            [seed, category, direction + 1, int(index)]
                        )
                    ).permutation(value)
                biases[index] = value
            result.append((f"{CATEGORIES[category]}_{direction:+d}", biases))
    return result




def matched_experiments(labels, seeds=(0, 1, 2, 3, 4), spatial_match="pooled_distance"):
    """One shared budget for the four categories and both random controls."""
    if spatial_match not in {"pooled_distance", "frame_distance"}:
        raise ValueError(spatial_match)
    labels = np.asarray(labels)
    reference = np.flatnonzero(labels.ravel() == 0)
    k = len(reference)
    experiments = [
        (name, 0, matched_indices(labels, i, k, 0)) for i, name in enumerate(CATEGORIES)
    ]
    for seed in seeds:
        uniform = (
            np.random.default_rng(seed).choice(labels.size, k, replace=False)
            if k
            else None
        )
        spatial = (
            spatially_matched_indices(
                labels,
                reference,
                seed,
                frame_stratified=spatial_match == "frame_distance",
            )
            if k
            else None
        )
        experiments.extend(
            [("random", seed, uniform), ("spatial_random", seed, spatial)]
        )
    return k, reference, experiments


def region_memory_mask(mask, condition, frame_index, conditioning=False):
    """Reuse the frozen corruption operators at the registered 64x64 resolution.

    Conditioning memory is never corrupted. Natural memory is supplied by the
    caller from the model prediction rather than reconstructed from labels.
    """
    if condition not in {"c0", "c1", "c2"}:
        raise ValueError(f"Not an oracle/corrupted memory condition: {condition}")
    if conditioning or condition == "c0":
        return np.asarray(mask, dtype=bool)
    from modeling.memory_interventions import corrupt_mask

    grid = cv2.resize(
        np.asarray(mask, np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST
    )
    return corrupt_mask(grid, {"c1": 0.25, "c2": 0.5}[condition], "mixed", frame_index)


def reliability_bias(units, taxonomy, weights, scale, ablation="full", seed=0):
    """E6 fixed formula; no labels are accessed by this function.

    r = clip(predicted IoU,.01,1) * sigmoid(object score) *
        exp(-age/6) * (1-.75*artifact). Missing confidence is an error.
    Random control permutes the same vector, preserving its norm exactly.
    """
    if ablation not in {
        "full",
        "no_reliability",
        "boundary_only",
        "artifact_off",
        "random",
    }:
        raise ValueError(ablation)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (4,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Four finite positive category weights are required")
    if float(scale) == 0:
        return {}
    if ablation == "boundary_only":
        weights = np.array([weights[0], 1.0, 1.0, 1.0])
    result = {}
    for unit in units:
        item = taxonomy[unit["frame_idx"]]
        values = weights[np.asarray(item["categories"]).ravel()].copy()
        if ablation != "no_reliability":
            if unit.get("predicted_iou") is None or unit.get("object_score") is None:
                raise ValueError("E6 requires recorded predicted IoU and object score")
            iou = float(torch.as_tensor(unit["predicted_iou"]).float().mean())
            presence = float(
                torch.as_tensor(unit["object_score"]).float().sigmoid().mean()
            )
            reliability = (
                np.clip(iou, 0.01, 1.0)
                * presence
                * np.exp(-abs(unit["temporal_distance"]) / 6.0)
            )
            values *= reliability
            if ablation != "artifact_off":
                values *= 1.0 - 0.75 * np.asarray(item["artifact"]).ravel()
        bias = float(scale) * np.log(np.maximum(values, 1e-8))
        if ablation == "random":
            bias = np.random.default_rng(seed + int(unit["frame_idx"])).permutation(
                bias
            )
        result[unit["frame_idx"]] = bias
    return result


def token_categories(
    mask,
    grid=(64, 64),
    band=1,
    thresholds=(0.1, 0.9),
    image=None,
    boundary_definition="mixed_coverage",
):
    coverage = cv2.resize(
        np.asarray(mask, np.float32), grid[::-1], interpolation=cv2.INTER_AREA
    )
    low, high = thresholds
    boundary = ((coverage > low) & (coverage < high)).astype(np.uint8)
    if boundary_definition == "contour_union":
        # Prospective sensitivity: include the inner contour even when a hard
        # object edge aligns exactly with grid cells. Padding treats outside
        # the image as background. Keep fractional-coverage edge cells too.
        occupied = (coverage >= 0.5).astype(np.uint8)
        interior = cv2.erode(
            occupied,
            np.ones((3, 3), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        boundary |= occupied & ~interior
    elif boundary_definition != "mixed_coverage":
        raise ValueError(boundary_definition)
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


def spatially_matched_indices(categories, reference, seed, frame_stratified=False):
    """Random positions matched to each reference's integer distance from object.

    Sampling includes the reference support; excluding boundary tokens would
    make exact distance matching impossible in many frames. Report overlap.
    """
    foreground = np.isin(categories, [0, 1])
    frames = foreground if foreground.ndim == 3 else foreground[None]
    if frame_stratified:
        # Same frame and same integer-distance histogram as the reference.
        # This preserves temporal composition without excluding overlap.
        selected = []
        offset = 0
        for frame_index, frame in enumerate(frames):
            local = (
                reference[(reference >= offset) & (reference < offset + frame.size)]
                - offset
            )
            if len(local):
                result = spatially_matched_indices(
                    np.asarray(categories).reshape(frames.shape)[frame_index],
                    local,
                    seed + frame_index,
                )
                if result is None:
                    return None
                selected.extend(result + offset)
            offset += frame.size
        return np.asarray(selected, dtype=int)
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
    selected, values = (
        selected[order],
        values[torch.as_tensor(order, device=values.device)],
    )
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
