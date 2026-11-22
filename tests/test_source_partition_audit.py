"""Adversarial cross-partition provenance checks."""
import json
import numpy as np
from PIL import Image
import pytest
from scripts.audit_source_partitions import audit_partition_integrity


def make_partitions(tmp_path):
    manifests = {}
    groups = {"train": ["t"], "dev": ["d"], "test": ["s"]}
    owners = {"train_video": "t", "dev_video": "d", "test_video": "s"}
    for i, split in enumerate(("train", "dev", "test")):
        image = tmp_path / f"{split}.jpg"
        Image.fromarray(np.full((8, 8, 3), i * 50, np.uint8)).save(image)
        row = {
            "split": split, "video_id": f"{split}_video",
            "frame_index": 0, "image": image.name,
        }
        if split == "train":
            mask = tmp_path / "train_gt.png"
            Image.fromarray(np.zeros((8, 8), np.uint8)).save(mask)
            row["mask"] = mask.name
        manifest = tmp_path / f"{split}.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        manifests[split] = manifest
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(groups))
    group_map = tmp_path / "group_map.json"
    group_map.write_text(json.dumps(owners))
    return manifests, split_path, group_map


def audit(m, split, groups):
    return audit_partition_integrity(
        m["train"], m["dev"], m["test"], split, groups)


def test_source_split_audit_accepts_distinct_bytes_and_groups(tmp_path):
    m, split, groups = make_partitions(tmp_path)
    record = audit(m, split, groups)
    assert record["exact_cross_group_image_duplicates"] == 0
    assert record["image_count_checked"] == 3
    assert record["video_count_by_split"] == {
        "train": 1, "dev": 1, "test": 1,
    }


def test_source_split_audit_rejects_byte_duplicate_in_different_group(tmp_path):
    m, split, groups = make_partitions(tmp_path)
    # Distinct path names and video names, but literally identical image bytes.
    (tmp_path / "dev.jpg").write_bytes((tmp_path / "train.jpg").read_bytes())
    with pytest.raises(ValueError, match="Identical image bytes"):
        audit(m, split, groups)


def test_source_split_audit_rejects_duplicate_video_group_or_gt_dev(tmp_path):
    m, split, groups = make_partitions(tmp_path)
    data = json.loads(groups.read_text())
    data["dev_video"] = "t"
    groups.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="wrong split"):
        audit(m, split, groups)

    _, split, groups = make_partitions(tmp_path)
    row = json.loads(m["dev"].read_text())
    row["mask"] = "train_gt.png"
    m["dev"].write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="image-only inference manifest"):
        audit(m, split, groups)


def test_source_split_audit_rejects_cross_split_video_name_reuse(tmp_path):
    m, split, groups = make_partitions(tmp_path)
    row = json.loads(m["test"].read_text())
    row["video_id"] = "dev_video"
    m["test"].write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Video ID appears"):
        audit(m, split, groups)
