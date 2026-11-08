import json
import random

import numpy as np
import torch

from scripts.run_token_conditioned_key_adapter import (
    MANIFEST,
    PROTOCOL,
    SOURCES,
    _release_gate,
    _capture_rng,
    _chunk_frames,
    _restore_rng,
    _source_hashes,
    _summarize,
    _tree_digest,
)
from scripts.memory_token_io import digest


def _protocol():
    return {
        "training": {"seed": 20261108, "schedule": "four-frame chunking"},
        "data": {"validation_sequences": ["seq13", "seq14", "seq15"]},
        "evaluation": {
            "release_rule": (
                "gain at least 0.01 equal-sequence present Dice; wins at least 2/3 sequences; "
                "false alarm increase no more than 0.02 absolute versus C0"
            )
        },
    }


def _groups(token_dice=0.72, token_fa=0.51, global_fa=0.10):
    groups = {arm: {} for arm in ("native", "token_conditioned", "global_gate")}
    for index, sequence in enumerate(("seq13", "seq14", "seq15")):
        groups["native"][sequence] = {
            "present_dice": 0.70, "present_iou": 0.55, "present_boundary_f": 0.60,
            "gt_empty_false_alarm_rate": 0.50,
        }
        groups["token_conditioned"][sequence] = {
            "present_dice": token_dice + index * 0.001, "present_iou": 0.56,
            "present_boundary_f": 0.61, "gt_empty_false_alarm_rate": token_fa,
        }
        groups["global_gate"][sequence] = {
            "present_dice": 0.70, "present_iou": 0.55, "present_boundary_f": 0.60,
            "gt_empty_false_alarm_rate": global_fa,
        }
    return groups


def test_chunk_size_matches_frozen_schedule():
    assert _chunk_frames(_protocol()) == 4


def test_actual_frozen_protocol_declares_expected_schedule_and_screen():
    protocol = json.loads(PROTOCOL.read_text())
    assert _chunk_frames(protocol) == 4
    assert "at least 0.01 equal-sequence present Dice" in protocol["evaluation"]["release_rule"]
    assert "wins at least 2/3 sequences" in protocol["evaluation"]["release_rule"]
    assert "no more than 0.02 absolute" in protocol["evaluation"]["release_rule"]


def test_chunk_size_fails_closed_if_protocol_schedule_changes():
    protocol = _protocol()
    protocol["training"]["schedule"] = "six-frame chunking"
    try:
        _chunk_frames(protocol)
    except RuntimeError as error:
        assert "four-frame" in str(error)
    else:
        raise AssertionError("Changed schedule was silently accepted")


def test_release_screen_uses_the_frozen_native_false_alarm_comparator(tmp_path):
    result = _summarize(_groups(), ["seq13", "seq14", "seq15"], tmp_path,
                        _protocol(), validation=True)
    assert result["validation_release_screen"]["passed"] is True
    assert np.isclose(result["validation_release_screen"]["false_alarm_increase_vs_native"], 0.01)
    assert json.loads((tmp_path / "validation_results.json").read_text())[
        "validation_release_screen"]["passed"] is True


def test_release_screen_fails_when_dice_margin_is_not_met(tmp_path):
    result = _summarize(_groups(token_dice=0.705), ["seq13", "seq14", "seq15"],
                        tmp_path, _protocol(), validation=True)
    assert result["validation_release_screen"]["passed"] is False


def test_summary_reports_leave_one_sequence_out_sensitivity(tmp_path):
    result = _summarize(_groups(), ["seq13", "seq14", "seq15"], tmp_path,
                        _protocol(), validation=False)
    contrast = result["contrasts"]["token_conditioned_minus_native"]
    expected = {"seq13": 0.0215, "seq14": 0.021, "seq15": 0.0205}
    assert contrast["leave_one_sequence_out_mean_delta"].keys() == expected.keys()
    assert all(np.isclose(contrast["leave_one_sequence_out_mean_delta"][name], value)
               for name, value in expected.items())


def test_release_gate_reads_the_gate_completion_receipt_and_checks_shared_sources(tmp_path):
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"frozen checkpoint receipt")
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    current_sources = _source_hashes()
    required = {
        "protocols/token_conditioned_key_adapter_polypgen_v1.json",
        "protocols/token_conditioned_key_adapter_engineering_gate_amendment_2026-11-09.json",
        "modeling/token_conditioned_key_adapter.py",
        "modeling/decoder_training_mask.py",
        "scripts/verify_token_conditioned_key_adapter.py",
        "scripts/run_author_baseline.py",
        "datasets/video.py",
    }
    setup = {
        "protocol_sha256": digest(PROTOCOL),
        "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": digest(MANIFEST),
        "sam2_revision": json.loads(PROTOCOL.read_text())["model"]["sam2_revision"],
        "sources": {name: current_sources[name] for name in required},
    }
    identity = {
        "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": digest(MANIFEST),
        "sam2_revision": setup["sam2_revision"],
    }
    completion = {
        "passed": True,
        "full_cohort_training_started": False,
        "efficacy_inference_started": False,
        "polypgen": [identity],
        "davis_multi_object": identity,
    }
    (gate_dir / "setup.json").write_text(json.dumps(setup))
    (gate_dir / "completion.json").write_text(json.dumps(completion))
    receipt = _release_gate(gate_dir, checkpoint, current_sources)
    assert receipt["completion"]["passed"] is True
    assert receipt["results_sha256"] == digest(gate_dir / "completion.json")


def test_release_gate_rejects_a_changed_shared_adapter_source(tmp_path):
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"frozen checkpoint receipt")
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    current_sources = _source_hashes()
    setup = {
        "protocol_sha256": digest(PROTOCOL),
        "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": digest(MANIFEST),
        "sam2_revision": json.loads(PROTOCOL.read_text())["model"]["sam2_revision"],
        "sources": {
            name: current_sources[name]
            for name in (
                "protocols/token_conditioned_key_adapter_polypgen_v1.json",
                "protocols/token_conditioned_key_adapter_engineering_gate_amendment_2026-11-09.json",
                "modeling/token_conditioned_key_adapter.py",
                "modeling/decoder_training_mask.py",
                "scripts/verify_token_conditioned_key_adapter.py",
                "scripts/run_author_baseline.py",
                "datasets/video.py",
            )
        },
    }
    setup["sources"]["modeling/token_conditioned_key_adapter.py"] = "0" * 64
    identity = {
        "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": digest(MANIFEST),
        "sam2_revision": setup["sam2_revision"],
    }
    (gate_dir / "setup.json").write_text(json.dumps(setup))
    (gate_dir / "completion.json").write_text(json.dumps({
        "passed": True,
        "full_cohort_training_started": False,
        "efficacy_inference_started": False,
        "polypgen": [identity],
        "davis_multi_object": identity,
    }))
    try:
        _release_gate(gate_dir, checkpoint, current_sources)
    except RuntimeError as error:
        assert "source hashes differ" in str(error)
    else:
        raise AssertionError("A gate with a stale adapter source hash was accepted")


def test_rng_state_round_trip_is_exact(tmp_path):
    torch.manual_seed(123)
    random.seed(456)
    np.random.seed(789)
    state = _capture_rng()
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    expected_torch = torch.rand(4)
    expected_python = random.random()
    expected_numpy = np.random.random(4)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    _restore_rng(restored)
    assert torch.equal(torch.rand(4), expected_torch)
    assert random.random() == expected_python
    assert np.array_equal(np.random.random(4), expected_numpy)


def test_nested_memory_bank_digest_is_order_stable_and_content_sensitive():
    first = {"a": torch.tensor([1.0]), "nested": {"b": torch.tensor([2.0])}}
    second = {"nested": {"b": torch.tensor([2.0])}, "a": torch.tensor([1.0])}
    assert _tree_digest(first) == _tree_digest(second)
    second["nested"]["b"][0] = 3.0
    assert _tree_digest(first) != _tree_digest(second)


def test_all_frozen_source_receipt_paths_exist():
    assert SOURCES
    assert set(_source_hashes()) == set(SOURCES)
