"""Conversion must be explicit, lossless for indexed labels and non-destructive."""
import json
import numpy as np
from PIL import Image
import pytest
from scripts.build_mask_manifest import normalized_mask, convert_manifest
from datasets.io import load_label_mask


def test_convert_binary_png_0_255_to_0_1(tmp_path):
    path = tmp_path / "binary.png"
    Image.fromarray(np.array([[0, 255], [255, 0]], np.uint8)).save(path)
    assert normalized_mask(path, kind="binary").tolist() == [[0, 1], [1, 0]]


def test_preserve_semantic_ids_and_reject_unknown(tmp_path):
    path = tmp_path / "semantic.png"
    Image.fromarray(np.array([[0, 1], [2, 0]], np.uint8)).save(path)
    assert normalized_mask(path, kind="semantic", labels=(0,1,2)).tolist() == [[0,1],[2,0]]
    with pytest.raises(ValueError, match="Unexpected semantic IDs"):
        normalized_mask(path, kind="semantic", labels=(0, 1))


def test_reject_ambiguous_color_code(tmp_path):
    path = tmp_path / "color.png"
    Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]]], np.uint8)).save(path)
    with pytest.raises(ValueError, match="Color-coded"):
        normalized_mask(path, kind="binary")


def test_convert_manifest_without_overwriting_source(tmp_path):
    image, mask = tmp_path / "frame.jpg", tmp_path / "mask.png"
    Image.fromarray(np.full((2, 2, 3), 42, np.uint8)).save(image)
    Image.fromarray(np.array([[0,255],[0,255]], np.uint8)).save(mask)
    before = mask.read_bytes()
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps({"video_id":"seq1","frame_index":0,
                     "split":"dev","image":image.name,"mask":mask.name})+"\n")
    created = convert_manifest(manifest, tmp_path/"out", kind="binary")
    result = json.loads(created.read_text())
    assert np.unique(load_label_mask(created.parent/result["mask"], [1])).tolist() == [0,1]
    assert mask.read_bytes() == before
    assert len(result["original_mask_sha256"]) == 64
    with pytest.raises(FileExistsError):
        convert_manifest(manifest, tmp_path/"out", kind="binary")
