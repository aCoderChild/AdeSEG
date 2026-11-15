import random

import numpy as np
from PIL import Image
import pytest

from datasets.sam2_clip_sampler import sample_clip
from datasets.video import ManifestFrame


def test_author_sampler_reproducible_and_preserves_rng(tmp_path):
    frames = []
    for i in range(8):
        mask = tmp_path / f'{i}.png'
        Image.fromarray(np.full((3, 3), i >= 3, dtype=np.uint8)).save(mask)
        frames.append(ManifestFrame('video', i, tmp_path / f'{i}.jpg', mask))
    random.seed(42)
    before = random.getstate()
    first = sample_clip(frames, 3, 17)
    assert random.getstate() == before
    assert first == sample_clip(frames, 3, 17)
    assert [f.frame_index for f in first] == list(range(first[0].frame_index, first[0].frame_index + 3))
    assert first[0].frame_index >= 3
    with pytest.raises(ValueError):
        sample_clip(frames, 9, 17)
