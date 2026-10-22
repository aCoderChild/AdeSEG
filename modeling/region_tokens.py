"""Three-region spatial attribution composed only of pinned author code.

Mask -> memory-token regions, step by step (no project geometry):

1. SAM2 ``SAM2VideoPredictor.add_new_mask``: resize the mask to the model's
   square input (bilinear, antialiased) and binarize at 0.5.
2. SAM2 ``SAM2Base._use_mask_as_output``: turn it into SAM2's 256x256
   low-resolution mask logits, exactly as a mask prompt is stored.
3. Efficient-SAM2 ``select_mask_windows``: map the low-resolution mask onto
   the 64x64 memory grid; a token is object if any mask pixel falls in it.
4. Boundary IoU ``mask_to_boundary``: the object tokens within distance d of
   the object contour (G_d intersected with G in the paper) are boundary.

Foreground = object tokens minus boundary; background = non-object tokens.
Steps 1, 2 and 4 run unchanged author source; step 3 runs Efficient-SAM2's
author method with window size 1 so that each window is one memory token.
"""

import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

# Exact author Git commits; checked before any author source is executed.
BOUNDARY_IOU_REVISION = "37d25586a677b043ed585f10e5c42d4e80176ea9"
SAM2_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
EFFICIENT_SAM2_REVISION = "8e1ad06089b9c958ff5ebfc72b9b7d0b904f5ce5"

REGION_NAMES = ("background", "foreground_interior", "boundary")

# v1: any band pixel or mixed footprint -> boundary (project rule).
# v2: area-plurality over a symmetric pixel band (project rule).
# v3: composition of unchanged SAM2, Efficient-SAM2 and Boundary IoU code.
REGION_ASSIGNMENT = "sam2_efficient_sam2_boundary_iou_v3"

SAM2_IMAGE_SIZE = 1024
TOKEN_GRID = (64, 64)

ROOT = Path(__file__).resolve().parents[1]
_AUTHOR = {}


def _verified_source(name, revision, source):
    repo = ROOT / "external" / name
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if head != revision:
        raise ValueError(f"{name} author revision differs")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", source],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    path = repo / source
    return path, path.read_text()


def _author_method(name, revision, source, class_name, method_name):
    """Compile one unchanged method from pinned author source."""
    path, text = _verified_source(name, revision, source)
    cls = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    namespace = {"torch": torch, "F": F}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _author_resize_block():
    """SAM2 add_new_mask's unchanged mask-resize statement."""
    source = "sam2/sam2_video_predictor.py"
    path, text = _verified_source("sam2", SAM2_REVISION, source)
    cls = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.ClassDef) and n.name == "SAM2VideoPredictor"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "add_new_mask"
    )
    block = next(
        n
        for n in method.body
        if isinstance(n, ast.If) and "mask_H != self.image_size" in ast.unparse(n.test)
    )
    return compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec")


def _boundary_iou():
    from importlib.util import module_from_spec, spec_from_file_location

    path, _ = _verified_source(
        "boundary-iou-api",
        BOUNDARY_IOU_REVISION,
        "boundary_iou/utils/boundary_utils.py",
    )
    spec = spec_from_file_location("adeseg_boundary_iou", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mask_to_boundary


def _author():
    if not _AUTHOR:
        _AUTHOR.update(
            resize=_author_resize_block(),
            low_res=_author_method(
                "sam2",
                SAM2_REVISION,
                "sam2/modeling/sam2_base.py",
                "SAM2Base",
                "_use_mask_as_output",
            ),
            windows=_author_method(
                "Efficient-SAM2",
                EFFICIENT_SAM2_REVISION,
                "sam2/sam2_video_predictor_with_bypass_stg.py",
                "SAM2VideoPredictor_bypass_stg",
                "select_mask_windows",
            ),
            boundary=_boundary_iou(),
        )
    return _AUTHOR


def sam2_low_res_mask(mask):
    """SAM2's 256x256 low-resolution logits for a binary mask prompt."""
    author = _author()
    mask = torch.as_tensor(np.asarray(mask), dtype=torch.float32)
    # add_new_mask: mask_inputs_orig = mask[None, None].float()
    state = {
        "mask_inputs_orig": mask[None, None],
        "mask_H": mask.shape[0],
        "mask_W": mask.shape[1],
        "self": SimpleNamespace(image_size=SAM2_IMAGE_SIZE),
        "torch": torch,
    }
    exec(author["resize"], state)
    model = SimpleNamespace(
        use_obj_ptrs_in_encoder=False, hidden_dim=256, pred_obj_scores=False
    )
    low_res = author["low_res"](model, None, None, state["mask_inputs"])[0]
    return low_res[0, 0]


def object_tokens(low_res_mask, grid=TOKEN_GRID):
    """Efficient-SAM2's mask-to-window selection at one token per window."""
    side = low_res_mask.shape[-1] // grid[1]
    if tuple(low_res_mask.shape) != (grid[0] * side, grid[1] * side):
        raise ValueError("Low-resolution mask does not tile the memory grid")
    # window_size = mem_info['window_sizes'][-2] * 4 in the author method.
    if side % 4:
        raise ValueError("Author window scaling expects a 4x mask-to-grid ratio")
    predictor = SimpleNamespace(mem_info={"window_sizes": [side // 4, side // 4]})
    selected = _author()["windows"](predictor, low_res_mask > 0)
    tokens = torch.zeros(grid[0] * grid[1], dtype=torch.bool)
    tokens[selected] = True
    return tokens.reshape(grid).numpy()


def reference_token_regions(mask, grid=TOKEN_GRID, dilation_ratio=0.02):
    """Background / foreground / boundary memory tokens for one binary mask."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all() or dilation_ratio <= 0:
        raise ValueError("Expected binary 2-D mask and positive boundary ratio")
    if tuple(grid) != TOKEN_GRID:
        raise ValueError("SAM2 Hiera-B+ memory uses a 64x64 token grid")
    low_res = sam2_low_res_mask(mask)
    objects = object_tokens(low_res, grid).astype(np.uint8)
    boundary = _author()["boundary"](objects, dilation_ratio).astype(bool)
    labels = np.zeros(grid, np.int64)
    labels[(objects == 1) & ~boundary] = 1
    labels[boundary] = 2
    width = max(1, int(round(dilation_ratio * np.hypot(*grid))))
    return {
        "labels": labels,
        "object_tokens": objects.astype(bool),
        "low_res_logits": low_res.numpy(),
        "assignment": REGION_ASSIGNMENT,
        "width_tokens": width,
        "width_pixels": width * SAM2_IMAGE_SIZE // grid[0],
        "dilation_ratio": dilation_ratio,
    }


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
