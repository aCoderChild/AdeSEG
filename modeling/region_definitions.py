"""Spatial region definitions from pinned author code; no learned classifier.

Primary: exact mixed 4x4 cells, using Efficient-SAM2's foreground/background
window occupancy. Comparators: Mask Transfiner incoherence, MatteFormer
trimaps and the retained v3 inner band. Inputs are native stored mask logits,
not a full-resolution PNG re-encoded as a mask prompt.
"""

import ast
from functools import lru_cache
import random
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from modeling.region_tokens import TOKEN_GRID, _author, _verified_source, object_tokens

TRANSFINER_REVISION = "5b61fb53d8df5484f44c8b7d8415f398fd283ddc"
MATTEFORMER_REVISION = "f09f090a7e8069f213ea2671c49c52516f755bb5"
PRIMARY_ASSIGNMENT = "efficient_sam2_exact_mixed_cells_native256_v1"
TRIMAP_WIDTHS = (3, 7, 15)


def native_logits(value):
    value = torch.as_tensor(value).detach().float().cpu()
    if value.shape != (256, 256) or not torch.isfinite(value).all():
        raise ValueError("Expected finite native stored 256x256 logits")
    return value


@lru_cache(maxsize=1)
def author_incoherence():
    path, text = _verified_source(
        "Mask-Transfiner",
        TRANSFINER_REVISION,
        "detectron2/modeling/roi_heads/mask_head.py",
    )
    node = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_incoherent_mask"
    )
    namespace = {"F": F}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[node.name]


@lru_cache(maxsize=3)
def _matteformer_generator(width):
    """Unchanged GenMask with fixed RNG responses for erosion sizes.

    Author generation resolution is 640x640. Fix equal FG/BG widths instead
    of sampling; disable random interpolation. np.int compatibility is local.
    The author's unrelated segmentation augmentation output is discarded.
    """
    if width not in TRIMAP_WIDTHS:
        raise ValueError("Use one of the prespecified trimap kernel sizes")
    path, text = _verified_source(
        "MatteFormer", MATTEFORMER_REVISION, "dataloader/data_generator.py"
    )
    tree = ast.parse(text)
    nodes = [
        next(
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
        )
        for name in ("maybe_random_interp", "GenMask")
    ]
    # Execute author code unchanged, without importing its training stack or
    # modifying NumPy's global namespace/RNG.
    numpy_compat = SimpleNamespace(
        int=int,
        uint8=np.uint8,
        ones_like=np.ones_like,
        random=SimpleNamespace(randint=lambda low, high: width),
    )
    namespace = {
        "cv2": cv2,
        "np": numpy_compat,
        "random": random.Random(0),
        "CONFIG": SimpleNamespace(data=SimpleNamespace(random_interp=False)),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["GenMask"]()


def matteformer_trimap(binary, width):
    return _matteformer_generator(width)({"alpha": np.asarray(binary, np.float32)})[
        "trimap"
    ]


def mixed_cell_regions(low_res_logits):
    """Exhaustive pure BG / pure FG / mixed partition at native token scale."""
    low_res_logits = native_logits(low_res_logits)
    foreground = object_tokens(low_res_logits)
    background = object_tokens(torch.where(low_res_logits > 0, -1.0, 1.0))
    if not (foreground | background).all():
        raise RuntimeError("Author occupancy failed to cover every spatial token")
    labels = np.zeros(TOKEN_GRID, np.int64)
    labels[foreground & ~background] = 1
    labels[foreground & background] = 2
    return {
        "labels": labels,
        "assignment": PRIMARY_ASSIGNMENT,
        "object_tokens": foreground,
        "mixed": foreground & background,
        "label_source": "native stored pred_masks > 0",
        "boundary_meaning": "mixed cells, not all geometric contour cells",
    }


def region_label_schemes(low_res_logits):
    """One stored prediction, all definitions; never modify native tensors."""
    logits = native_logits(low_res_logits)
    binary = logits > 0
    primary = mixed_cell_regions(logits)
    objects = primary["object_tokens"]
    scores = author_incoherence()(binary.float()[None, None], 4)[0, 0]
    # The author saturates residues >= 0.01 to one. Retain the native score
    # as well; converting saturated cells to a categorical label is our adapter.
    incoherent = scores.numpy() == 1
    labels = np.where(objects, 1, 0).astype(np.int64)
    labels[incoherent] = 2
    schemes = {
        "mixed_cells": primary,
        "transfiner_incoherent": {
            "labels": labels,
            "scores": scores.numpy(),
            "assignment": "transfiner_get_incoherent_mask_sfact4_saturated_v1",
            "adaptation": "author function uses sfact=2 in its training caller; here 4 maps native256 to memory64",
        },
    }
    for width in TRIMAP_WIDTHS:
        trimap = matteformer_trimap(binary.numpy(), width)
        # Same nearest-neighbor token-map operation as the author encoder;
        # native256 -> memory64, rather than its full image -> stage-0 grid.
        coarse = F.interpolate(
            torch.from_numpy(trimap)[None, None], scale_factor=1 / 4, mode="nearest"
        )[0, 0].numpy()
        labels = np.zeros(TOKEN_GRID, np.int64)
        labels[coarse == 255] = 1
        labels[coarse == 128] = 2
        schemes[f"matteformer_k{width}"] = {
            "labels": labels,
            "assignment": f"matteformer_fixed_k{width}_native256_v1",
            "kernel_size_at_author640": width,
        }
    boundary = _author()["boundary"](objects.astype(np.uint8), 0.02).astype(bool)
    labels = np.where(objects, 1, 0).astype(np.int64)
    labels[boundary] = 2
    schemes["inner_band_0.02"] = {
        "labels": labels,
        "assignment": "v3_inner_band_direct_native256_v1",
        "note": "same v3 token geometry, without full-res PNG round trip",
    }
    for record in schemes.values():
        if (
            record["labels"].shape != TOKEN_GRID
            or not np.isin(record["labels"], [0, 1, 2]).all()
        ):
            raise RuntimeError("Invalid three-region token partition")
    return schemes
