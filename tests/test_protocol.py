import json
from pathlib import Path

import pytest

from training.protocol import evaluation_role, load_protocol, verify_training


def protocol(tmp_path: Path, locked_test=None):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({
        "training_videos": ["train_a", "train_b"],
        "validation_videos": ["val_a"],
        "exploratory_videos": ["explore_a"],
        "locked_test": locked_test,
    }))
    return load_protocol(path)


def test_protocol_requires_the_exact_development_partition(tmp_path):
    current = protocol(tmp_path)
    verify_training(current, {"train_a": [] , "train_b": []}, {"val_a": []})
    with pytest.raises(ValueError, match="training_videos"):
        verify_training(current, {"train_a": []}, {"val_a": []})


def test_protocol_marks_reused_test_data_as_exploratory(tmp_path):
    current = protocol(tmp_path)
    assert evaluation_role(current, ["explore_a"]) == "exploratory"
    with pytest.raises(ValueError, match="declared protocol role"):
        evaluation_role(current, ["train_a"])


def test_protocol_accepts_a_disjoint_locked_test_only(tmp_path):
    current = protocol(tmp_path, ["locked_a"])
    assert evaluation_role(current, ["locked_a"]) == "locked_test"
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps({
        "training_videos": ["train_a"], "validation_videos": ["val_a"],
        "exploratory_videos": ["explore_a"], "locked_test": ["val_a"],
    }))
    with pytest.raises(ValueError, match="disjoint"):
        load_protocol(path)
