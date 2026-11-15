import numpy as np
from PIL import Image

from datasets.video import ManifestFrame
from scripts.stage_tcmn_polypgen import stage_sequence


def test_staging_exposes_only_initial_visible_mask_and_keeps_order(tmp_path):
    frames = []
    for index in range(3):
        image = tmp_path / f"{index}.jpg"
        mask = tmp_path / f"{index}.png"
        Image.new("RGB", (3, 2)).save(image)
        values = np.zeros((2, 3), dtype=np.uint8)
        if index:
            values[0, 1] = 1
        Image.fromarray(values).save(mask)
        frames.append(ManifestFrame("seq", 10 + index, image, mask))

    result = stage_sequence(frames, tmp_path / "staged")
    assert result["source_prompt_frame"] == 11
    assert [row["source_frame_index"] for row in result["frames"]] == [11, 12]
    images = tmp_path / "staged/JPEGImages/seq"
    annotations = tmp_path / "staged/Annotations/seq"
    assert sorted(p.name for p in images.iterdir()) == ["000000.jpg", "000001.jpg"]
    assert (images / "000000.jpg").resolve() == frames[1].image_path
    assert [p.name for p in annotations.iterdir()] == ["000000.png"]
    prompt = Image.open(annotations / "000000.png")
    assert prompt.mode == "P"
    assert np.array_equal(np.asarray(prompt), np.asarray(Image.open(frames[1].mask_path)))
