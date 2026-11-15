import numpy as np
import pytest
from PIL import Image

from scripts.stage_cmsa_seq23_viewport_v1 import image_pixels, save_crop


def test_lossless_crop_excludes_inset(tmp_path):
    pixels = np.zeros((720, 1280, 3), dtype=np.uint8)
    pixels[:, :446] = (255, 0, 0)
    pixels[:, 446:] = (0, 1, 254)
    source, crop = tmp_path / "frame.png", tmp_path / "crop.png"
    Image.fromarray(pixels).save(source)
    receipt = save_crop(source, crop)
    result = image_pixels(crop)
    assert result.shape == (720, 834, 3)
    assert np.array_equal(result, pixels[:, 446:])
    assert len(receipt["crop_sha256"]) == 64
    assert len(receipt["source_sha256"]) == 64


def test_unexpected_geometry_fails_before_creating_crop(tmp_path):
    source, crop = tmp_path / "bad.png", tmp_path / "crop.png"
    Image.new("RGB", (640, 480)).save(source)
    with pytest.raises(ValueError, match="geometry"):
        save_crop(source, crop)
    assert not crop.exists()
