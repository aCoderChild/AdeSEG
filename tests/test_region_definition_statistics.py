import numpy as np
import pytest
import torch

from modeling.region_definitions import mixed_cell_regions
from scripts.region_definition_statistics import (
    contour_coverage,
    size_bin,
    summarize_frames,
)


def test_aligned_contour_is_measurably_missed_not_silently_expanded():
    mask = np.zeros((256, 256), np.uint8)
    mask[64:128, 64:128] = 1
    labels = mixed_cell_regions(torch.from_numpy(mask).float() * 20 - 10)["labels"]
    result = contour_coverage(mask, labels)
    assert not (labels == 2).any()
    assert result["including_border"]["missed_fraction"] == 1
    assert result["excluding_border"]["contour_pixels"] > 0


def test_non_aligned_contour_has_complete_mixed_cell_coverage():
    mask = np.zeros((256, 256), np.uint8)
    mask[65:127, 65:127] = 1
    labels = mixed_cell_regions(torch.from_numpy(mask).float() * 20 - 10)["labels"]
    result = contour_coverage(mask, labels)
    assert result["including_border"]["missed_fraction"] == 0


def test_empty_and_border_only_contours_are_undefined_when_excluded():
    labels = np.ones((64, 64), int)
    empty = contour_coverage(np.zeros((256, 256), np.uint8), labels)
    assert empty["including_border"]["missed_fraction"] is None
    full = contour_coverage(np.ones((256, 256), np.uint8), labels)
    assert full["image_border_contour_pixels"] == 1020
    assert full["including_border"]["missed_fraction"] == 1
    assert full["excluding_border"]["missed_fraction"] is None


def test_shape_guard_quartile_ties_and_empty_summary():
    with pytest.raises(ValueError, match="native256"):
        contour_coverage(np.zeros((64, 64)), np.zeros((64, 64)))
    assert [size_bin(a, [0.1, 0.2, 0.3]) for a in [0, 0.1, 0.11, 0.2, 0.3, 0.4]] == [
        "absent",
        "Q1",
        "Q2",
        "Q2",
        "Q3",
        "Q4",
    ]
    assert summarize_frames([]) == {}
