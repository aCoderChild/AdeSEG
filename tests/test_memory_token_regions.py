"""Memory-token regions from unchanged author code, alignment and grouping."""

import ast
import subprocess

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from modeling import region_tokens
from modeling.region_tokens import (
    REGION_NAMES,
    ROOT,
    group_memory_vectors,
    object_tokens,
    reference_token_regions,
    sam2_low_res_mask,
)

FRAME = (1080, 1350)  # PolypGen-like, non-square: SAM2 squashes to 1024x1024


def rectangle(top, left, bottom, right, shape=FRAME):
    mask = np.zeros(shape, np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


def author_text(name, revision, source):
    return subprocess.check_output(
        ["git", "show", f"{revision}:{source}"], cwd=ROOT / "external" / name, text=True
    )


def method_source(text, class_name, method_name):
    cls = next(
        n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    return ast.get_source_segment(text, method)


def test_executed_code_is_unchanged_author_source():
    sam2 = author_text("sam2", region_tokens.SAM2_REVISION, "sam2/modeling/sam2_base.py")
    efficient = author_text(
        "Efficient-SAM2",
        region_tokens.EFFICIENT_SAM2_REVISION,
        "sam2/sam2_video_predictor_with_bypass_stg.py",
    )
    predictor = author_text(
        "sam2", region_tokens.SAM2_REVISION, "sam2/sam2_video_predictor.py"
    )
    assert "F.interpolate" in method_source(sam2, "SAM2Base", "_use_mask_as_output")
    assert "torch.any(masks_padded" in method_source(
        efficient, "SAM2VideoPredictor_bypass_stg", "select_mask_windows"
    )
    resize = method_source(predictor, "SAM2VideoPredictor", "add_new_mask")
    assert "mask_inputs = (mask_inputs >= 0.5).float()" in resize
    block = region_tokens._author_resize_block()
    assert "mask_inputs" in block.co_names


def test_low_res_mask_matches_sam2_mask_prompt_path():
    mask = rectangle(300, 400, 800, 1000)
    low_res = sam2_low_res_mask(mask)
    # Expected behaviour of the two SAM2 steps, written out for the test only.
    resized = F.interpolate(
        torch.tensor(mask, dtype=torch.float32)[None, None],
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    logits = (resized >= 0.5).float() * 20 - 10
    expected = F.interpolate(
        logits, size=(256, 256), mode="bilinear", align_corners=False, antialias=True
    )[0, 0]
    assert low_res.shape == (256, 256)
    assert torch.equal(low_res, expected)


def test_object_token_is_any_low_res_pixel_in_its_cell():
    low_res = torch.full((256, 256), -10.0)
    low_res[9, 130] = 10.0  # one positive pixel -> token (2, 32)
    low_res[100:108, 40:48] = 10.0  # 8x8 block -> tokens (25..26, 10..11)
    tokens = object_tokens(low_res)
    assert tokens.shape == (64, 64)
    expected = (low_res > 0).reshape(64, 4, 64, 4).any(3).any(1).numpy()
    assert np.array_equal(tokens, expected)
    assert tokens[2, 32] and tokens[25:27, 10:12].all() and tokens.sum() == 5


def test_empty_mask_is_all_background():
    result = reference_token_regions(np.zeros(FRAME, bool))
    assert (result["labels"] == 0).all()
    assert not result["object_tokens"].any()


def test_three_regions_follow_boundary_iou_inner_band():
    result = reference_token_regions(rectangle(300, 400, 800, 1000))
    labels, objects = result["labels"], result["object_tokens"]
    assert result["assignment"] == "sam2_efficient_sam2_boundary_iou_v3"
    assert set(np.unique(labels)) == {0, 1, 2}
    # Boundary IoU's band lies inside the object (G_d intersected with G).
    assert np.array_equal(labels > 0, objects)
    assert result["width_tokens"] == 2  # round(0.02 * hypot(64, 64))
    rows = np.where(objects.any(1))[0]
    top, bottom = rows.min(), rows.max()
    column = np.where(objects.any(0))[0].mean().astype(int)
    assert (labels[top : top + 2, column] == 2).all()
    assert labels[top + 2, column] == 1
    assert (labels[bottom - 1 : bottom + 1, column] == 2).all()
    assert labels[top - 1, column] == 0


def test_rows_and_columns_keep_image_orientation():
    labels = reference_token_regions(rectangle(0, 0, 300, 300))["labels"]
    assert labels[:10, :10].any() and not labels[40:, 40:].any()


def test_boundary_width_is_monotonic_in_ratio():
    mask = rectangle(200, 300, 900, 1100)
    bands = [
        reference_token_regions(mask, dilation_ratio=ratio)["labels"] == 2
        for ratio in (0.01, 0.02, 0.04)
    ]
    assert all(np.all(~small | large) for small, large in zip(bands, bands[1:]))
    assert bands[0].sum() < bands[1].sum() < bands[2].sum()


def test_small_object_has_no_foreground_interior():
    result = reference_token_regions(rectangle(500, 600, 540, 650))
    assert result["object_tokens"].any()
    assert not (result["labels"] == 1).any()


def test_rejects_non_memory_grid_and_non_binary_masks():
    with pytest.raises(ValueError, match="64x64"):
        reference_token_regions(rectangle(0, 0, 10, 10), (8, 12))
    with pytest.raises(ValueError, match="binary"):
        reference_token_regions(np.full(FRAME, 2))


def test_exactly_three_region_types_and_no_learned_head():
    assert REGION_NAMES == ("background", "foreground_interior", "boundary")
    assert not hasattr(region_tokens, "build_linear_region_probe")
    memory = torch.zeros(1, 4, 1, 3)
    assert tuple(group_memory_vectors(memory, [[0, 1, 2]])) == REGION_NAMES


def test_memory_vectors_reconstruct_exactly_in_row_major_order_without_mutation():
    memory = torch.arange(24).reshape(1, 4, 2, 3).float()
    original = memory.clone()
    labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
    groups = group_memory_vectors(memory, labels)
    assert groups["foreground_interior"]["indices"].tolist() == [1, 4]
    reconstructed = torch.empty(6, 4)
    for group in groups.values():
        reconstructed[group["indices"]] = group["vectors"]
        group["vectors"].zero_()
    assert torch.equal(reconstructed, original.flatten(2).transpose(1, 2)[0])
    assert torch.equal(memory, original)
    with pytest.raises(ValueError, match="invalid"):
        group_memory_vectors(memory, labels + 1)
