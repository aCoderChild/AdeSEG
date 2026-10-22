import numpy as np
import torch

from modeling import region_definitions
from scripts.relabel_region_trimaps import amended_adapter
from scripts.region_definition_statistics import contour_coverage


def test_amended_configuration_is_isolated_and_preserves_shared_width():
    low = torch.full((256, 256), -10.0)
    low[65:191, 67:189] = 10
    original = region_definitions.region_label_schemes(low)
    changed = amended_adapter().region_label_schemes(low)
    assert region_definitions.TRIMAP_WIDTHS == (3, 7, 15)
    assert amended_adapter().TRIMAP_WIDTHS == (15, 21, 29)
    assert np.array_equal(
        original["matteformer_k15"]["labels"], changed["matteformer_k15"]["labels"]
    )
    assert "matteformer_k3" in original and "matteformer_k3" not in changed
    assert np.array_equal(
        original["mixed_cells"]["labels"], changed["mixed_cells"]["labels"]
    )


def test_wider_author_trimaps_reduce_missed_contours_on_test_rectangle():
    low = torch.full((256, 256), -10.0)
    low[65:191, 67:189] = 10
    schemes = amended_adapter().region_label_schemes(low)
    misses = []
    for width in (15, 21, 29):
        labels = schemes[f"matteformer_k{width}"]["labels"]
        assert labels.shape == (64, 64)
        assert set(np.unique(labels)) == {0, 1, 2}
        misses.append(
            contour_coverage((low > 0).numpy(), labels)["including_border"][
                "missed_pixels"
            ]
        )
    assert misses[2] <= misses[1] <= misses[0]
