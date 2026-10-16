import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v2_reuses_the_frozen_explicit_v1_davis_partition():
    v2 = json.loads((ROOT / "protocols/latent_utility_v2.json").read_text())
    split_path = ROOT / "protocols" / v2["datasets"]["davis_train"]["sequence_partition_file"]
    split = json.loads(split_path.read_text())["davis_2017"]

    assert len(split["train_sequences"]) == 60
    assert len(split["model_selection_holdout"]) == 10
    assert len(split["validation_sequences"]) == 30
    assert split["model_selection_holdout"] == v2["datasets"]["davis_train"]["model_selection_holdout"]
