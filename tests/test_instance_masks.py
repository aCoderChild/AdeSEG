import numpy as np
from PIL import Image
import pytest

from datasets.instance_masks import instance_prompts, load_instance_mask
from datasets.video import ManifestFrame


def test_indexed_colors_are_ids_and_16_bit_ids_are_not_truncated(tmp_path):
    palette_path = tmp_path / "palette.png"
    image = Image.fromarray(np.array([[0, 7], [42, 7]], np.uint8)).convert("P")
    image.putpalette([255, 0, 0] * 256)
    image.save(palette_path)
    assert load_instance_mask(palette_path).tolist() == [[0, 7], [42, 7]]
    large_path = tmp_path / "large.png"
    Image.fromarray(np.array([[0, 300], [1024, 300]], np.uint16)).save(large_path)
    assert load_instance_mask(large_path).tolist() == [[0, 300], [1024, 300]]


def test_rejects_rgb_or_lossy_instance_masks(tmp_path):
    for suffix in ("png", "jpg"):
        path = tmp_path / f"rgb.{suffix}"
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(path)
        with pytest.raises(ValueError):
            load_instance_mask(path)


def test_only_scheduled_prompt_annotations_are_read(tmp_path):
    path = tmp_path / "first.png"
    Image.fromarray(np.array([[0, 7], [42, 255]], np.uint8)).save(path)
    frames = [ManifestFrame("video", 100, tmp_path / "frame.jpg", path),
              ManifestFrame("video", 200, tmp_path / "later.jpg", tmp_path / "missing.png")]
    prompts = instance_prompts(frames, void_id=255)
    assert set(prompts[0]) == {7, 42} and set(prompts) == {0}
    assert not (prompts[0][7] & prompts[0][42]).any()
    with pytest.raises(ValueError, match="absent"):
        instance_prompts(frames, object_ids=[99])


def test_delayed_prompts_use_manifest_frame_ids_not_list_positions(tmp_path):
    paths = []
    for frame in (10, 20):
        path = tmp_path / f"{frame}.png"
        Image.fromarray(np.array([[0, 7], [42, 0]], np.uint8)).save(path)
        paths.append(ManifestFrame("video", frame, path, path))
    prompts = instance_prompts(paths, prompt_frames={7: 10, 42: 20})
    assert set(prompts[0]) == {7} and set(prompts[1]) == {42}
    with pytest.raises(ValueError, match="outside"):
        instance_prompts(paths, prompt_frames={7: 0})
