"""Check the public trainer refuses mislabeled and modified captures."""
import hashlib
import json

import pytest

from training.split_guard import validate_train_sources


def test_train_partition_and_capture_hash(tmp_path):
    payload = b"capture bytes"
    p = tmp_path / "capture.pt"
    p.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = tmp_path / "manifest.json"
    lock = tmp_path / "split.json"
    lock.write_text(json.dumps({
        "train": ["patient_A"], "dev": ["patient_B"], "test": ["patient_C"]
    }))
    rows = [{"split": "train", "source_group": "patient_A",
             "tensor_file": "capture.pt", "capture_sha256": digest}]
    manifest.write_text(json.dumps(rows))
    verified = validate_train_sources(manifest, lock, file_field="tensor_file")
    assert verified["train_source_groups"] == ["patient_A"]
    assert verified["capture_count"] == 1
    rows[0]["source_group"] = "patient_C"
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="partition"):
        validate_train_sources(manifest, lock, file_field="tensor_file")
    rows[0]["source_group"] = "patient_A"
    manifest.write_text(json.dumps(rows))
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        validate_train_sources(manifest, lock, file_field="tensor_file")


def test_overlapping_patient_groups_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    lock = tmp_path / "split.json"
    p = tmp_path / "x.pt"
    p.write_bytes(b"test")
    manifest.write_text(json.dumps([{
        "split": "train", "source_group": "patient_A",
        "file": "x.pt", "capture_sha256": hashlib.sha256(b"test").hexdigest()
    }]))
    lock.write_text(json.dumps({
        "train": ["patient_A"], "dev": ["patient_A"], "test": []
    }))
    with pytest.raises(ValueError, match="Overlapping"):
        validate_train_sources(manifest, lock, file_field="file")
