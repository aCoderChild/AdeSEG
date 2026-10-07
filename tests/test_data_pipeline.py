"""Guards for the data path: frame order, mask decoding and two-region scoring."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adenoid.io import load_semantic_mask  # noqa: E402
from evaluation.two_region import evaluate  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_polypgen_frames_are_in_numeric_order():
    order = load_script("build_polypgen_manifest").numeric_order
    paths = [Path(f"0_endocv2021_positive_{n}.jpg") for n in (104, 69, 7, 1000)]
    assert [int(p.stem.rsplit("_", 1)[1]) for p in order(paths)] == [7, 69, 104, 1000]


def test_inference_and_evaluation_use_the_same_frame_order(tmp_path):
    """infer.py follows the manifest (numeric_order); temporal.py sorts mask files with
    natural_sort_key. Reappearance metrics depend on row order, so the two must agree."""
    order = load_script("build_polypgen_manifest").numeric_order
    from evaluation.temporal import frame_stem, image_files, sequence_directories

    for stem in ("seq1_C6_10", "seq1_C6_9", "seq1_C6_100", "seq1_C6_0"):
        (tmp_path / f"{stem}.jpg").touch()
        (tmp_path / f"{stem}_mask.jpg").touch()
    images = [p for p in tmp_path.iterdir() if not p.stem.endswith("_mask")]
    masks = [p for p in image_files(tmp_path) if p.stem.endswith("_mask")]
    assert [p.stem for p in order(images)] == [frame_stem(p) for p in masks]

    root = ROOT / "data/PolypGen2021_MultiCenterData_v3/sequenceData/positive"
    for sequence in sorted(root.glob("seq*")) if root.is_dir() else []:
        image_dir, mask_dir = sequence_directories(root, sequence.name)
        assert [p.stem for p in order(image_dir.glob("*.jpg"))] == [frame_stem(p) for p in image_files(mask_dir)]


def test_binary_jpeg_masks_load_as_zero_one(tmp_path):
    mask = np.zeros((64, 64), np.uint8)
    mask[16:48, 16:48] = 255
    for mode in ("L", "RGB"):
        path = tmp_path / f"mask_{mode}.jpg"
        Image.fromarray(mask).convert(mode).save(path, quality=75)
        loaded = load_semantic_mask(path)
        assert set(np.unique(loaded)) <= {0, 1}
        assert abs(int(loaded.sum()) - 32 * 32) < 64


def test_wrongly_encoded_masks_raise(tmp_path):
    from adenoid.io import load_label_mask

    binary = np.zeros((32, 32), np.uint8)
    binary[8:24, 8:24] = 255
    Image.fromarray(binary).save(tmp_path / "binary_255.png")  # 0/255 PNG read as label 255
    with pytest.raises(ValueError):
        load_label_mask(tmp_path / "binary_255.png", [1])
    grey = np.full((96, 96), 255, np.uint8)
    grey[:, 32:64] = 128
    grey[:, 64:] = 0
    Image.fromarray(grey).save(tmp_path / "three_class.jpg", quality=75)  # would merge two regions
    with pytest.raises(ValueError):
        load_semantic_mask(tmp_path / "three_class.jpg")


def test_indexed_png_masks_keep_their_labels(tmp_path):
    mask = np.zeros((16, 16), np.uint8)
    mask[:, 8:] = 1
    mask[4:8, 4:8] = 2
    Image.fromarray(mask).save(tmp_path / "mask.png")
    assert np.array_equal(load_semantic_mask(tmp_path / "mask.png"), mask)


def test_three_class_grey_jpeg_maps_to_labels(tmp_path):
    build = load_script("build_mask_manifest")
    grey = np.full((96, 96), 255, np.uint8)   # REFUGE coding: 255 background
    grey[:, 32:64] = 128                       # 128 disc rim
    grey[:, 64:] = 0                           # 0 cup
    Image.fromarray(grey).save(tmp_path / "mask.jpg", quality=60)
    values, labels = build.parse_value_map(["255:0", "128:1", "0:2"])
    out = build.to_labels(tmp_path / "mask.jpg", values, labels)
    assert set(np.unique(out)) == {0, 1, 2}
    centres = out[:, [16, 48, 80]]
    assert (centres == np.array([0, 1, 2])).all()


def test_two_region_evaluation_is_perfect_on_identical_masks(tmp_path):
    rows = []
    for i, (adenoid, airway) in enumerate(((600, 400), (100, 900))):
        mask = np.zeros((40, 50), np.uint8)
        mask.flat[:airway] = 1
        mask.flat[airway:airway + adenoid] = 2
        Image.fromarray(np.zeros((40, 50, 3), np.uint8)).save(tmp_path / f"img{i}.png")
        Image.fromarray(mask).save(tmp_path / f"gt{i}.png")
        (tmp_path / "pred" / "masks" / f"v{i}").mkdir(parents=True)
        Image.fromarray(mask).save(tmp_path / "pred" / "masks" / f"v{i}" / f"img{i}.png")
        rows.append({"split": "test", "video_id": f"v{i}", "frame_index": 0,
                     "image": f"img{i}.png", "mask": f"gt{i}.png"})
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    frames, summary = evaluate(tmp_path / "manifest.jsonl", "test", tmp_path / "pred",
                               ("adenoid", 2), ("airway", 1), "fraction_of_total", [0.5, 0.75])
    assert summary["adenoid_dice"] == summary["airway_dice"] == 1.0
    assert summary["frame_ratio"]["mae"] == 0.0
    assert [round(f["ground_truth_ratio"], 2) for f in frames] == [0.6, 0.1]
    assert summary["grade"]["accuracy"] == 1.0
