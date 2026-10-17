import numpy as np
from PIL import Image

from datasets.video import ManifestFrame
from scripts.audit_region_initialization import audit_initialization


def test_audit_distinguishes_delayed_and_unpromptable_labels(tmp_path):
    videos = {}
    for name, visible in (("visible", 0), ("delayed", 2), ("unpromptable", None)):
        frames = []
        for index in range(3):
            path = tmp_path / f"{name}_{index}.png"
            mask = np.zeros((8, 8), np.uint8)
            if visible is not None and index >= visible:
                mask[2:6, 2:6] = 1
            Image.fromarray(mask).save(path)
            frames.append(ManifestFrame(name, index, path, path))
        videos[name] = frames
    result = audit_initialization(videos)
    assert {
        row["video"]: row["first_visible_frame"] for row in result["sequences"]
    } == {"visible": 0, "delayed": 2, "unpromptable": None}
