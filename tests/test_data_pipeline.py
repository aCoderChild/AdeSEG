"""Guards for the data path: frame order and mask decoding."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_semantic_mask  # noqa: E402
from scripts.build_polypgen_manifest import numeric_order  # noqa: E402
from scripts.build_polypgen_kaggle_manifest import verified_frame_order  # noqa: E402


def test_polypgen_frames_are_in_numeric_order():
    order = numeric_order
    paths = [Path(f"0_endocv2021_positive_{n}.jpg") for n in (104, 69, 7, 1000)]
    assert [int(p.stem.rsplit("_", 1)[1]) for p in order(paths)] == [7, 69, 104, 1000]


def test_staging_preserves_manifest_order_and_source_files(tmp_path):
    from datasets.video import ManifestFrame, stage_frames

    sources = [tmp_path / name for name in ("source_104.jpg", "source_7.jpg")]
    for path in sources:
        path.write_bytes(path.name.encode())
    frames = [
        ManifestFrame("seq", i, path, tmp_path / "mask.png")
        for i, path in enumerate(sources)
    ]
    staged = tmp_path / "staged"
    staged.mkdir()
    stage_frames(frames, staged)
    assert sorted(path.name for path in staged.iterdir()) == [
        "000000.jpg",
        "000001.jpg",
    ]
    for index, source in enumerate(sources):
        link = staged / f"{index:06d}.jpg"
        assert link.is_symlink()
        assert link.resolve() == source.resolve()
        assert link.read_bytes() == source.read_bytes()


def test_inference_and_evaluation_use_the_same_frame_order(tmp_path):
    """Manifest and temporal-evaluation frame order must agree."""
    order = numeric_order
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
        assert [p.stem for p in order(image_dir.glob("*.jpg"))] == [
            frame_stem(p) for p in image_files(mask_dir)
        ]


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
    from datasets.io import load_label_mask

    binary = np.zeros((32, 32), np.uint8)
    binary[8:24, 8:24] = 255
    Image.fromarray(binary).save(
        tmp_path / "binary_255.png"
    )  # 0/255 PNG read as label 255
    with pytest.raises(ValueError):
        load_label_mask(tmp_path / "binary_255.png", [1])
    grey = np.full((96, 96), 255, np.uint8)
    grey[:, 32:64] = 128
    grey[:, 64:] = 0
    Image.fromarray(grey).save(
        tmp_path / "three_class.jpg", quality=75
    )  # would merge two regions
    with pytest.raises(ValueError):
        load_semantic_mask(tmp_path / "three_class.jpg")


def test_indexed_png_masks_keep_their_labels(tmp_path):
    mask = np.zeros((16, 16), np.uint8)
    mask[:, 8:] = 1
    mask[4:8, 4:8] = 2
    Image.fromarray(mask).save(tmp_path / "mask.png")
    assert np.array_equal(load_semantic_mask(tmp_path / "mask.png"), mask)


def test_reindexed_files_require_provenance(tmp_path):
    files = [tmp_path / "0.jpg", tmp_path / "1.jpg"]
    with pytest.raises(ValueError, match="do not establish chronology"):
        verified_frame_order(files, "seq13")
    _, source = verified_frame_order(files, "seq13", allow_reindexed=True)
    assert source == "unverified_reindexed_diagnostic_only"


def test_original_numeric_suffix_order_unchanged(tmp_path):
    files = [tmp_path / f"EndoCV2021_seq13_{n}.jpg" for n in [100, 12, 0]]
    ordered, source = verified_frame_order(files, "seq13")
    assert [p.name for p in ordered] == [
        f"EndoCV2021_seq13_{n}.jpg" for n in [0, 12, 100]
    ]
    assert source == "original_numeric_suffix"


def test_mixed_names_cannot_bypass_reindexed_guard(tmp_path):
    with pytest.raises(ValueError, match="Mixed original and reindexed"):
        verified_frame_order(
            [tmp_path / "0.jpg", tmp_path / "original_12.jpg"], "seq13"
        )


def test_verified_mapping_binds_files_and_prevents_duplicate_frames(tmp_path):
    entries, files = {}, []
    for name, original in [("0.jpg", 100), ("1.jpg", 12)]:
        path = tmp_path / name
        path.write_bytes(name.encode())
        files.append(path)
        entries[name] = {
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "original_frame_index": original,
        }
    mapping = {"sequences": {"seq13": entries}}
    ordered, _ = verified_frame_order(files, "seq13", mapping)
    assert [p.name for p in ordered] == ["1.jpg", "0.jpg"]
    entries["1.jpg"]["original_frame_index"] = 100
    with pytest.raises(ValueError, match="one-to-one"):
        verified_frame_order(files, "seq13", mapping)
    entries["1.jpg"]["image_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="content mismatch"):
        verified_frame_order(files, "seq13", mapping)
