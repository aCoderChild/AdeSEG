import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v2_protocol_is_frozen_and_uses_hiera_base_plus():
    protocol = json.loads((ROOT / "protocols/latent_utility_v2.json").read_text())

    assert protocol["status"].startswith("frozen before")
    assert protocol["models"]["M1"]["config"] == "configs/sam2.1_hiera_b+1024.yaml"
    assert protocol["failure_definition"]["threshold"] == 0.5
    assert protocol["failure_definition"]["minimum_consecutive_frames"] == 3
    assert protocol["statistics"]["probe_seeds"] == [0, 1, 2, 3, 4]
    assert protocol["statistics"]["readout_seeds"] == [0, 1, 2]
    assert protocol["evidence_bars"]["E1_attention_or_both_share_for_broad_framing"] == 0.20


def test_v2_protocol_keeps_pointer_units_separate_and_has_all_corruptions():
    protocol = json.loads((ROOT / "protocols/latent_utility_v2.json").read_text())
    measurements = protocol["measurements"]

    assert "separate" in measurements["units"]["pointers"]
    assert list(measurements["corruption"]) == ["c0", "c1", "c2", "c3", "c4", "natural"]
    assert protocol["priority"]["must"] == [
        "sanity",
        "E1-light",
        "E2-frame",
        "E3-c0-c2-c4-natural",
        "E4-linear-all-controls",
        "E5-core-comparisons",
        "E7-zero-shot-PolypGen",
    ]
