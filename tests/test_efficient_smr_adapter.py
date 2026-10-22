from pathlib import Path

import pytest

from scripts.run_efficient_smr import (
    ROOT,
    SETTINGS,
    verify_source,
    check_pruning,
)


def test_vendor_change_is_device_allocation_only():
    repo, revision = verify_source(ROOT / "external/Efficient-SAM2")
    assert repo.name == "Efficient-SAM2"
    assert len(revision) == 40


def test_smr_is_enabled_and_encoder_routing_and_frame_pruning_are_disabled():
    assert SETTINGS["prune_memory"]
    assert SETTINGS["apply_WB"] and SETTINGS["disable_WB"]
    assert SETTINGS["selected_layers"] == []
    assert not SETTINGS["Mem_Frame_Prune"]
    assert SETTINGS["num_frame_to_prune"] == 0
    assert SETTINGS["topk_mask"] and SETTINGS["set_drop_ratio"] == 0.95
    assert not SETTINGS["random_mask"] and not SETTINGS["uniform_mask"]


def test_each_layer_must_demonstrate_real_key_reduction():
    rows = [
        {"layer": i, "original_keys": 28676, "retained_keys": 9216, "pointer_keys": 4}
        for i in range(4)
    ]
    check_pruning(rows)
    with pytest.raises(RuntimeError, match="layer 3"):
        check_pruning(rows[:-1])
    with pytest.raises(RuntimeError, match="inactive"):
        check_pruning([{**r, "retained_keys": r["original_keys"]} for r in rows])
    with pytest.raises(RuntimeError, match="Invalid"):
        check_pruning(
            rows
            + [{"layer": 0, "original_keys": 10, "retained_keys": 1, "pointer_keys": 4}]
        )


def test_adapter_uses_author_propagation_and_native_retrieval():
    source = (ROOT / "scripts/run_efficient_smr.py").read_text()
    assert "model.propagate_in_video_wj" in source
    assert "build_WB_model" in source
    assert "Hiera.forward(module, args[0])" in source
    assert "author.main()" in source
    assert Path(ROOT / "patches/efficient-sam2-device.md").exists()
