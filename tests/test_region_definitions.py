import ast

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from modeling.region_definitions import (
    TRANSFINER_REVISION,
    author_incoherence,
    matteformer_trimap,
    mixed_cell_regions,
    region_label_schemes,
)
from modeling.region_tokens import ROOT


def test_primary_matches_exact_block_occupancy_and_preserves_input():
    logits = torch.full((256, 256), -10.0)
    logits[0, 0] = 10
    logits[80:160, 80:160] = 10
    before = logits.clone()
    labels = mixed_cell_regions(logits)["labels"]
    blocks = (logits > 0).reshape(64, 4, 64, 4)
    foreground, background = blocks.any(3).any(1), (~blocks).any(3).any(1)
    expected = np.zeros((64, 64), np.int64)
    expected[(foreground & ~background).numpy()] = 1
    expected[(foreground & background).numpy()] = 2
    assert np.array_equal(labels, expected)
    assert labels[0, 0] == 2 and labels[25, 25] == 1
    assert torch.equal(logits, before)


def test_author_incoherence_function_is_executed_unchanged():
    path = ROOT / "external/Mask-Transfiner/detectron2/modeling/roi_heads/mask_head.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_incoherent_mask"
    )
    context = {"F": F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), context)
    mask = torch.zeros(1, 1, 256, 256)
    mask[..., 81:151, 49:167] = 1
    assert torch.equal(author_incoherence()(mask, 4), context[node.name](mask, 4))
    assert len(TRANSFINER_REVISION) == 40


def test_small_cell_can_be_missed_by_author_bilinear_sampling():
    mask = torch.zeros(256, 256)
    mask[0, 0] = 1
    labels = mixed_cell_regions(mask * 20 - 10)["labels"]
    assert labels[0, 0] == 2
    assert not author_incoherence()(mask[None, None], 4).any()


@pytest.mark.parametrize("present,expected", [(False, 0), (True, 1)])
def test_empty_and_full_masks_all_schemes(present, expected):
    logits = torch.full((256, 256), 10.0 if present else -10.0)
    for name, record in region_label_schemes(logits).items():
        labels = record["labels"]
        assert labels.shape == (64, 64)
        if name == "inner_band_0.02" and present:
            assert (labels == 2).any()  # Author explicitly counts the image border.
        else:
            assert (labels == expected).all(), name


def test_trimap_is_deterministic_two_sided_and_does_not_change_global_rng():
    binary = np.zeros((256, 256), np.float32)
    binary[64:192, 64:192] = 1
    state = np.random.get_state()
    trimap = matteformer_trimap(binary, 15)
    assert np.array_equal(trimap, matteformer_trimap(binary, 15))
    assert set(np.unique(trimap)) == {0, 128, 255}
    assert ((trimap == 128) & (binary == 0)).any()
    assert ((trimap == 128) & (binary == 1)).any()
    assert all(np.array_equal(a, b) for a, b in zip(state, np.random.get_state()))


def test_bad_native_shape_and_nonfinite_logits_fail():
    with pytest.raises(ValueError, match="256x256"):
        mixed_cell_regions(torch.zeros(1024, 1024))
    with pytest.raises(ValueError, match="finite"):
        mixed_cell_regions(torch.full((256, 256), float("nan")))
