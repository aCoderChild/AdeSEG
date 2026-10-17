import numpy as np
from PIL import Image

from datasets.video import ManifestFrame
from scripts.audit_region_initialization import audit_initialization
from scripts.audit_region_initialization import prompt_index, target_indices
import pytest


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
    amended = {"initialization": "first_labeled_frame"}
    assert prompt_index(videos["delayed"], amended) == 2
    assert prompt_index(videos["visible"], {}) == 0
    with pytest.raises(ValueError):
        prompt_index(videos["delayed"], {})
    with pytest.raises(ValueError):
        prompt_index(videos["unpromptable"], amended)


def test_targets_respect_full_post_prompt_memory_bank():
    assert target_indices(15, 0, 10) == list(range(7, 15))
    targets = target_indices(91, 24, 10)
    assert len(targets) == 10
    assert targets[0] == 31 and targets[-1] == 90
    with pytest.raises(ValueError):
        target_indices(10, 4, 10)
