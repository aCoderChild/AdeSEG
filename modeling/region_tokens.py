"""Three-region spatial attribution
Reference: external/boundary-iou-api"""

import cv2
import numpy as np
import torch

# Exact author Git commit; checked before boundary extraction for reproducibility.
BOUNDARY_IOU_REVISION = "37d25586a677b043ed585f10e5c42d4e80176ea9"

REGION_NAMES = ("background", "foreground_interior", "boundary")


def query_regions(foreground, height, width):
    mask = foreground.reshape(-1, 1, height, width).float()
    outer = torch.nn.functional.max_pool2d(mask, 3, 1, 1) > 0
    inner = ~(torch.nn.functional.max_pool2d(1 - mask, 3, 1, 1) > 0)
    boundary = outer & ~inner
    return {
        "foreground": inner.flatten(1),
        "boundary": boundary.flatten(1),
        "background": (~outer).flatten(1),
    }


def reference_token_regions(mask, grid, dilation_ratio=0.02):
    """Boundary IoU geometry + explicit project-specific grid mapping."""
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path
    import subprocess

    repo = Path(__file__).resolve().parents[1] / "external/boundary-iou-api"
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != BOUNDARY_IOU_REVISION:
        raise ValueError("Boundary IoU author revision differs")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "boundary_iou/utils/boundary_utils.py",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    spec = spec_from_file_location(
        "adeseg_boundary_iou", repo / "boundary_iou/utils/boundary_utils.py"
    )
    author = module_from_spec(spec)
    spec.loader.exec_module(author)
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all() or dilation_ratio <= 0:
        raise ValueError("Expected binary 2-D mask and positive boundary ratio")
    mask = mask.astype(np.uint8)
    width = max(1, int(round(dilation_ratio * np.hypot(*mask.shape))))
    pad = width + 1
    padded = np.pad(mask, pad)
    ratio = width / np.hypot(*padded.shape)
    boundary = (
        author.mask_to_boundary(padded, ratio)
        | author.mask_to_boundary(1 - padded, ratio)
    )[pad:-pad, pad:-pad]
    coverage = cv2.resize(
        mask.astype(np.float32), grid[::-1], interpolation=cv2.INTER_AREA
    )
    boundary_coverage = cv2.resize(
        boundary.astype(np.float32), grid[::-1], interpolation=cv2.INTER_AREA
    )
    mixed = (coverage > 1e-6) & (coverage < 1 - 1e-6)
    labels = np.where(coverage >= 0.5, 1, 0).astype(np.int64)
    labels[(boundary_coverage > 0) | mixed] = 2
    return {
        "labels": labels,
        "foreground_coverage": coverage,
        "boundary_coverage": boundary_coverage,
        "mixed": mixed,
        "width_pixels": width,
        "dilation_ratio": dilation_ratio,
    }


def group_memory_vectors(memory, labels):
    """Read-only grouping by row-major spatial index; pointers are not inputs."""
    labels = torch.as_tensor(labels, device=memory.device)
    if (
        memory.ndim != 4
        or memory.shape[0] != 1
        or tuple(labels.shape) != tuple(memory.shape[-2:])
    ):
        raise ValueError("Expected one spatial memory tensor and matching region grid")
    if (
        not torch.isfinite(memory).all()
        or not ((labels >= 0) & (labels <= 2) & (labels == labels.long())).all()
    ):
        raise ValueError("Nonfinite memory or invalid three-region labels")
    vectors = memory.detach().flatten(2).transpose(1, 2)[0].clone()
    groups = {}
    for index, name in enumerate(REGION_NAMES):
        indices = torch.where(labels.flatten() == index)[0]
        groups[name] = {"indices": indices.cpu(), "vectors": vectors[indices].cpu()}
    joined = torch.cat([group["indices"] for group in groups.values()])
    if not torch.equal(joined.sort().values, torch.arange(vectors.shape[0])):
        raise ValueError("Region groups must be disjoint and exhaustive")
    return groups
