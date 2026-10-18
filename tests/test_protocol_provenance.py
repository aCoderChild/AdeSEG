import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "protocols"


@pytest.mark.parametrize("path", sorted(PROTOCOLS.glob("*.json")), ids=lambda p: p.name)
def test_protocol_documents_and_parent_links_are_readable(path):
    """Preserve frozen filenames used by pinned jobs and protocol ancestry."""
    document = json.loads(path.read_text())
    assert isinstance(document, dict)
    for key in ("parent_protocol", "parent"):
        if key in document:
            parent = path.parent / document[key]
            assert parent != path
            assert isinstance(json.loads(parent.read_text()), dict)


def test_v2_reuses_the_frozen_explicit_v1_davis_partition():
    v2 = json.loads((ROOT / "protocols/latent_utility_v2.json").read_text())
    split_path = ROOT / "protocols" / v2["datasets"]["davis_train"]["sequence_partition_file"]
    split = json.loads(split_path.read_text())["davis_2017"]

    assert len(split["train_sequences"]) == 60
    assert len(split["model_selection_holdout"]) == 10
    assert len(split["validation_sequences"]) == 30
    assert split["model_selection_holdout"] == v2["datasets"]["davis_train"]["model_selection_holdout"]

    train = set(split["train_sequences"])
    holdout = set(split["model_selection_holdout"])
    validation = set(split["validation_sequences"])
    assert holdout <= train
    assert train.isdisjoint(validation)
    assert len(train - holdout) == 50
