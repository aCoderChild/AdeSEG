"""Curated smoke selection must not use test data or whole benchmark."""
import json
import numpy as np
from PIL import Image
import pytest

from scripts.check_med_gpu_smoke import validate_curated_manifest


def make_manifest(tmp_path, *, n=3, split="dev", video="s1", present=(True, True, True)):
    records=[]
    for i in range(n):
        image=tmp_path/f"{i}.jpg"
        mask=tmp_path/f"{i}.png"
        Image.fromarray(np.full((8,8,3),20,np.uint8)).save(image)
        pixels=np.ones((8,8),np.uint8) if present[i] else np.zeros((8,8),np.uint8)
        Image.fromarray(pixels).save(mask)
        records.append({"split":split, "video_id":video, "frame_index":i,
                        "image":image.name, "mask":mask.name})
    manifest=tmp_path/"manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(x) for x in records))
    return manifest


def test_short_dev_sequence_is_eligible(tmp_path):
    manifest=make_manifest(tmp_path)
    summary=validate_curated_manifest(manifest,max_frames=3)
    assert summary["frames"]==3
    assert summary["prompt_position"]==0


def test_test_split_is_rejected(tmp_path):
    manifest=make_manifest(tmp_path,split="test")
    with pytest.raises(ValueError,match="No valid frames"):
        validate_curated_manifest(manifest)


def test_late_first_foreground_is_rejected(tmp_path):
    manifest=make_manifest(tmp_path,present=(False,False,True))
    with pytest.raises(ValueError,match="prompt"):
        validate_curated_manifest(manifest)


def test_oversized_smoke_is_rejected(tmp_path):
    manifest=make_manifest(tmp_path)
    with pytest.raises(ValueError,match="2..2"):
        validate_curated_manifest(manifest,max_frames=2)
