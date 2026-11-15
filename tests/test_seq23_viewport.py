from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from modeling.seq23_viewport import VIEWPORT, crop_viewport, restore_mask, stage_cropped_frames

def test_crop_excludes_sidebar_and_restores_coordinates():
    image = np.zeros((720, 1280, 3), np.uint8)
    image[:170, 230:426] = 255
    image[200, 500] = (1, 2, 3)
    crop = crop_viewport(image)
    assert crop.shape == (720, 834, 3)
    assert crop.max() == 3
    mask = np.zeros(crop.shape[:2], np.uint8)
    mask[200, 500 - VIEWPORT[0]] = 1
    restored = restore_mask(mask)
    assert restored.sum() == 1 and restored[200, 500] == 1
    assert not restored[:, :VIEWPORT[0]].any()

def test_lossless_staging_and_bad_shapes(tmp_path):
    image = np.random.default_rng(1).integers(0, 256, (720, 1280, 3), np.uint8)
    path = tmp_path / 'original.png'
    Image.fromarray(image).save(path)
    stage_cropped_frames([SimpleNamespace(image_path=path)], tmp_path)
    assert np.array_equal(np.asarray(Image.open(tmp_path / '000000.jpg')), crop_viewport(image))
    with pytest.raises(ValueError):
        crop_viewport(image[:719])
    with pytest.raises(ValueError):
        restore_mask(np.zeros((720, 833)))
