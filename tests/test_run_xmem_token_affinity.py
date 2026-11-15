import json
import hashlib
import zipfile

import pytest
import torch
from torch import nn

from scripts import run_xmem_token_affinity as runner


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.xmem_strength = nn.Parameter(torch.tensor(0.0))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _Attention()


def _arm_row(dice, false_alarms=0, undetected=0):
    return {
        "present_dice": dice,
        "present_iou": dice / (2 - dice),
        "present_boundary_f": dice,
        "elapsed_seconds": 10.0,
        "fps": 2.0,
        "gt_empty_false_alarm_count": false_alarms,
        "gt_empty_frames": 10,
        "propagation": {"undetected_returns": undetected, "return_count": 1},
    }


def test_adapter_state_round_trip_copies_only_added_tensors():
    model = _Model()
    expected = runner.adapter_state(model)
    expected["layer.xmem_strength"] = torch.tensor(0.125)

    runner.load_adapter_state(model, expected)

    assert model.layer.xmem_strength.item() == pytest.approx(0.125)
    assert model.layer.xmem_strength.grad is None


def test_gradient_receipts_localize_differences_to_named_parameters():
    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    second = nn.Parameter(torch.tensor([3.0]))
    first.grad = torch.tensor([0.5, 0.25])
    second.grad = torch.tensor([0.75])
    left = runner.gradient_tensor_hashes([
        ("adapter.first", first), ("adapter.second", second),
    ])

    second.grad = torch.tensor([0.5])
    right = runner.gradient_tensor_hashes([
        ("adapter.first", first), ("adapter.second", second),
    ])

    assert left.keys() == right.keys() == {"adapter.first", "adapter.second"}
    assert left["adapter.first"] == right["adapter.first"]
    assert left["adapter.second"] != right["adapter.second"]


def test_parameter_diagnostic_amendment_is_bound_to_reduction_amendment():
    amendment = json.loads(runner.PARAMETER_DIAGNOSTIC_AMENDMENT.read_text())
    assert amendment["parent_amendment_sha256"] == runner.digest(
        runner.REDUCTION_AMENDMENT
    )


def test_training_checkpoint_restores_adapter_optimizer_rng_and_plan(tmp_path):
    model = _Model()
    with torch.no_grad():
        model.layer.xmem_strength.fill_(0.5)
    optimizer = torch.optim.AdamW([model.layer.xmem_strength], lr=0.01)
    (model.layer.xmem_strength.square()).backward()
    optimizer.step()
    plan = [{"epoch": 1, "sequence": "seq2", "indices": [0, 1, 2]}]
    core_hash = "core"
    arm = "token_affinity"
    amendment_hash = "amendment"
    rng_amendment_hash = "rng-amendment"
    autograd_amendment_hash = "autograd-amendment"
    reduction_amendment_hash = "reduction-amendment"
    saved_rng_state = runner.capture_rng_state()
    payload = {
        "core_sha256": core_hash,
        "arm": arm,
        "protocol_amendment_sha256": amendment_hash,
        "rng_amendment_sha256": rng_amendment_hash,
        "autograd_determinism_amendment_sha256": autograd_amendment_hash,
        "reduction_determinism_amendment_sha256": reduction_amendment_hash,
        "adapter": runner.adapter_state(model),
        "optimizer": optimizer.state_dict(),
        "rng_state": saved_rng_state,
        "next_clip_index": 1,
        "completed_clip_rows": [{"plan": plan[0], "result": {"loss": 1.0}}],
        "best_epoch": None,
        "best_adapter_sha256": None,
    }
    checkpoint = tmp_path / "resume.pt"
    receipt_path = tmp_path / "resume_receipt.json"
    runner.atomic_torch_save(checkpoint, payload)
    runner.atomic_json(receipt_path, {
        "sha256": runner.digest(checkpoint),
        "core_sha256": core_hash,
        "arm": arm,
        "next_clip_index": 1,
        "rng_amendment_sha256": rng_amendment_hash,
        "autograd_determinism_amendment_sha256": autograd_amendment_hash,
        "reduction_determinism_amendment_sha256": reduction_amendment_hash,
    })

    resumed = _Model()
    resumed_optimizer = torch.optim.AdamW([resumed.layer.xmem_strength], lr=0.01)
    restored = runner.restore_training_checkpoint(
        checkpoint,
        receipt_path,
        resumed,
        resumed_optimizer,
        tmp_path,
        core_hash=core_hash,
        arm=arm,
        plan=plan,
        amendment_sha256=amendment_hash,
        rng_amendment_sha256=rng_amendment_hash,
        autograd_determinism_amendment_sha256=autograd_amendment_hash,
        reduction_determinism_amendment_sha256=reduction_amendment_hash,
    )

    assert runner.state_digest(runner.adapter_state(resumed)) == runner.state_digest(payload["adapter"])
    assert runner._tree_equal(resumed_optimizer.state_dict(), payload["optimizer"])
    assert runner.rng_state_digest(runner.capture_rng_state()) == runner.rng_state_digest(
        saved_rng_state
    )
    assert restored["next_clip_index"] == 1


def test_training_checkpoint_rejects_cursor_or_receipt_mismatch(tmp_path):
    model = _Model()
    optimizer = torch.optim.AdamW([model.layer.xmem_strength], lr=0.01)
    plan = [{"epoch": 1, "sequence": "seq2"}]
    payload = {
        "core_sha256": "core", "arm": "token_affinity",
        "protocol_amendment_sha256": "amendment",
        "rng_amendment_sha256": "rng-amendment",
        "autograd_determinism_amendment_sha256": "autograd-amendment",
        "reduction_determinism_amendment_sha256": "reduction-amendment",
        "adapter": runner.adapter_state(model), "optimizer": optimizer.state_dict(),
        "rng_state": runner.capture_rng_state(), "next_clip_index": 0,
        "completed_clip_rows": [], "best_epoch": None,
        "best_adapter_sha256": None,
    }
    checkpoint = tmp_path / "resume.pt"
    receipt_path = tmp_path / "resume_receipt.json"
    runner.atomic_torch_save(checkpoint, payload)
    runner.atomic_json(receipt_path, {"sha256": runner.digest(checkpoint),
                                      "core_sha256": "other", "arm": "token_affinity",
                                      "next_clip_index": 0,
                                      "rng_amendment_sha256": "rng-amendment",
                                      "autograd_determinism_amendment_sha256": "autograd-amendment"})
    resumed = _Model()
    with pytest.raises(RuntimeError, match="receipt provenance"):
        runner.restore_training_checkpoint(
            checkpoint, receipt_path, resumed,
            torch.optim.AdamW([resumed.layer.xmem_strength], lr=0.01), tmp_path,
            core_hash="core", arm="token_affinity", plan=plan,
            amendment_sha256="amendment",
            rng_amendment_sha256="rng-amendment",
            autograd_determinism_amendment_sha256="autograd-amendment",
            reduction_determinism_amendment_sha256="reduction-amendment",
        )


def test_runner_failure_is_receipted_before_error_propagation(tmp_path):
    class IntegrityError(RuntimeError):
        pass

    previous = {"phase": "training", "clip": 1, "expected_clips": 480}
    failure = runner.write_failure_receipt(
        tmp_path,
        phase="training",
        arm="token_affinity",
        error=IntegrityError("memory order mismatch"),
        core_hash="core",
        gate_receipt={"checks": {"native_bank_integrity": True}},
        last_progress=previous,
        traceback_text="traceback fixture",
        failed_at=123.0,
    )

    progress = json.loads((tmp_path / "progress.json").read_text())
    saved_failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure == saved_failure
    assert saved_failure["status"] == "stopped"
    assert saved_failure["exception_type"] == "IntegrityError"
    assert saved_failure["last_successful_progress"] == previous
    assert progress["status"] == "stopped"
    assert progress["failure_file"] == "failure.json"


def test_result_rule_uses_frozen_false_alarm_and_return_safety_limits():
    names = [f"seq{i}" for i in range(16, 24)]
    per_arm = {
        "token_affinity": {name: _arm_row(0.62) for name in names},
        "frame_broadcast": {name: _arm_row(0.60) for name in names},
        "native": {name: _arm_row(0.60) for name in names},
    }
    amendment = {
        "data": {"test": names},
        "training": {"seed": 4},
        "stopping_rule": {
            "continue_only_if": "frozen test rule",
            "max_absolute_false_alarm_rate_increase": 0.02,
        },
    }

    result = runner.summarize_results(per_arm, amendment)
    assert result["frozen_continuation_rule_passed"] is True

    per_arm["token_affinity"][names[0]]["gt_empty_false_alarm_count"] = 2
    result = runner.summarize_results(per_arm, amendment)
    assert result["frozen_continuation_rule_passed"] is False

    per_arm["token_affinity"][names[0]]["gt_empty_false_alarm_count"] = 0
    per_arm["token_affinity"][names[0]]["propagation"]["undetected_returns"] = 1
    result = runner.summarize_results(per_arm, amendment)
    assert result["frozen_continuation_rule_passed"] is False


def test_result_rule_fails_if_a_test_sequence_has_no_present_frames():
    names = [f"seq{i}" for i in range(16, 24)]
    per_arm = {
        arm: {name: _arm_row(0.6) for name in names}
        for arm in ("token_affinity", "frame_broadcast", "native")
    }
    per_arm["native"][names[0]]["present_dice"] = None
    amendment = {
        "data": {"test": names},
        "training": {"seed": 4},
        "stopping_rule": {"continue_only_if": "frozen test rule",
                           "max_absolute_false_alarm_rate_increase": 0.02},
    }

    with pytest.raises(RuntimeError, match="undefined"):
        runner.summarize_results(per_arm, amendment)


def test_gate_receipt_rejects_the_saved_partial_probe(tmp_path):
    (tmp_path / "completion.json").write_text(json.dumps({
        "status": "partial_engineering_pass",
        "protocol_complete": False,
    }))
    (tmp_path / "setup.json").write_text(json.dumps({
        "checkpoint_sha256": "checkpoint",
        "protocol_sha256": "protocol",
    }))

    with pytest.raises(RuntimeError, match="partial"):
        runner.verify_gate_receipt(tmp_path, _expected_gate_core())


def test_manifest_split_roles_match_the_frozen_manifest_inventory():
    amendment = json.loads(runner.EFFICACY_AMENDMENT.read_text())
    split_amendment = json.loads(runner.MANIFEST_SPLIT_AMENDMENT.read_text())
    train_and_validation_and_excluded = (
        amendment["data"]["train"]
        + amendment["data"]["validation"]
        + amendment["data"]["excluded"]
    )
    videos_by_split = {
        "dev": {sequence: [] for sequence in train_and_validation_and_excluded},
        "test": {sequence: [] for sequence in amendment["data"]["test"]},
    }
    videos = {sequence: frames for split in videos_by_split.values()
              for sequence, frames in split.items()}

    runner.validate_roles(amendment, videos, videos_by_split, split_amendment)


def test_manifest_split_roles_reject_test_sequences_loaded_as_dev():
    amendment = json.loads(runner.EFFICACY_AMENDMENT.read_text())
    split_amendment = json.loads(runner.MANIFEST_SPLIT_AMENDMENT.read_text())
    dev_ids = (amendment["data"]["train"] + amendment["data"]["validation"]
               + amendment["data"]["excluded"] + amendment["data"]["test"])
    videos_by_split = {"dev": {sequence: [] for sequence in dev_ids}, "test": {}}
    videos = dict(videos_by_split["dev"])

    with pytest.raises(RuntimeError, match="Manifest split 'dev'"):
        runner.validate_roles(amendment, videos, videos_by_split, split_amendment)


def _expected_gate_core(source_hashes=None):
    return {
        "checkpoint_sha256": "checkpoint",
        "method_protocol_sha256": "protocol",
        "efficacy_amendment_sha256": "amendment",
        "rng_amendment_sha256": "rng-amendment",
        "autograd_determinism_amendment_sha256": "autograd-amendment",
        "reduction_determinism_amendment_sha256": "reduction-amendment",
        "manifest_split_amendment_sha256": "split-amendment",
        "manifest_sha256": "manifest",
        "sam2_revision": "sam2-revision",
        "python": "python-version",
        "torch": "torch-version",
        "source_hashes": source_hashes or {},
    }


def test_gate_receipt_requires_every_check_and_verified_source_snapshot(tmp_path):
    schema = json.loads(runner.EFFICACY_AMENDMENT.read_text())["gate_completion_schema"]
    rng_schema = json.loads(runner.RNG_AMENDMENT.read_text())
    source = b"verified fixture source\n"
    source_hash = hashlib.sha256(source).hexdigest()
    snapshot = tmp_path / "source_snapshot.zip"
    with zipfile.ZipFile(snapshot, "w") as archive:
        archive.writestr("example.py", source)
    setup = {
        "checkpoint_sha256": "checkpoint",
        "protocol_sha256": "protocol",
        "efficacy_amendment_sha256": "amendment",
        "rng_amendment_sha256": "rng-amendment",
        "autograd_determinism_amendment_sha256": "autograd-amendment",
        "reduction_determinism_amendment_sha256": "reduction-amendment",
        "manifest_split_amendment_sha256": "split-amendment",
        "manifest_sha256": "manifest",
        "sam2_revision": "sam2-revision",
        "environment": {"python": "python-version", "torch": "torch-version"},
        "source_snapshot_sha256": runner.digest(snapshot),
        "sources": {"example.py": source_hash},
    }
    runner.atomic_json(tmp_path / "setup.json", setup)
    results_path = tmp_path / "results.json"
    runner.atomic_json(results_path, {"gate": "passed"})
    completion = {
        "status": "complete",
        "protocol_complete": True,
        "protocol_sha256": setup["protocol_sha256"],
        "efficacy_amendment_sha256": setup["efficacy_amendment_sha256"],
        "rng_amendment_sha256": setup["rng_amendment_sha256"],
        "autograd_determinism_amendment_sha256": setup["autograd_determinism_amendment_sha256"],
        "reduction_determinism_amendment_sha256": setup["reduction_determinism_amendment_sha256"],
        "manifest_split_amendment_sha256": setup["manifest_split_amendment_sha256"],
        "manifest_sha256": setup["manifest_sha256"],
        "sam2_revision": setup["sam2_revision"],
        "setup_sha256": runner.digest(tmp_path / "setup.json"),
        "source_snapshot_sha256": runner.digest(snapshot),
        "results_sha256": runner.digest(results_path),
        "checks": {name: True for name in (
            *schema["required_boolean_checks"],
            *rng_schema["required_engineering_gate_checks"],
        )},
    }
    runner.atomic_json(tmp_path / "completion.json", completion)

    receipt = runner.verify_gate_receipt(
        tmp_path, _expected_gate_core({"example.py": source_hash})
    )
    assert receipt["checks"] == completion["checks"]

    completion["checks"]["persistent_multi_object_id_isolation"] = False
    runner.atomic_json(tmp_path / "completion.json", completion)
    with pytest.raises(RuntimeError, match="required checks"):
        runner.verify_gate_receipt(
            tmp_path, _expected_gate_core({"example.py": source_hash})
        )


def test_gate_receipt_rejects_a_changed_efficacy_amendment_or_runner_source(tmp_path):
    schema = json.loads(runner.EFFICACY_AMENDMENT.read_text())["gate_completion_schema"]
    rng_schema = json.loads(runner.RNG_AMENDMENT.read_text())
    source = b"verified fixture source\n"
    source_hash = hashlib.sha256(source).hexdigest()
    snapshot = tmp_path / "source_snapshot.zip"
    with zipfile.ZipFile(snapshot, "w") as archive:
        archive.writestr("example.py", source)
    setup = {
        "checkpoint_sha256": "checkpoint",
        "protocol_sha256": "protocol",
        "efficacy_amendment_sha256": "amendment",
        "rng_amendment_sha256": "rng-amendment",
        "autograd_determinism_amendment_sha256": "autograd-amendment",
        "reduction_determinism_amendment_sha256": "reduction-amendment",
        "manifest_split_amendment_sha256": "split-amendment",
        "manifest_sha256": "manifest",
        "sam2_revision": "sam2-revision",
        "environment": {"python": "python-version", "torch": "torch-version"},
        "source_snapshot_sha256": runner.digest(snapshot),
        "sources": {"example.py": source_hash},
    }
    runner.atomic_json(tmp_path / "setup.json", setup)
    results_path = tmp_path / "results.json"
    runner.atomic_json(results_path, {"gate": "passed"})
    runner.atomic_json(tmp_path / "completion.json", {
        "status": "complete",
        "protocol_complete": True,
        "protocol_sha256": setup["protocol_sha256"],
        "efficacy_amendment_sha256": setup["efficacy_amendment_sha256"],
        "rng_amendment_sha256": setup["rng_amendment_sha256"],
        "autograd_determinism_amendment_sha256": setup["autograd_determinism_amendment_sha256"],
        "reduction_determinism_amendment_sha256": setup["reduction_determinism_amendment_sha256"],
        "manifest_split_amendment_sha256": setup["manifest_split_amendment_sha256"],
        "manifest_sha256": setup["manifest_sha256"],
        "sam2_revision": setup["sam2_revision"],
        "setup_sha256": runner.digest(tmp_path / "setup.json"),
        "source_snapshot_sha256": runner.digest(snapshot),
        "results_sha256": runner.digest(results_path),
        "checks": {name: True for name in (
            *schema["required_boolean_checks"],
            *rng_schema["required_engineering_gate_checks"],
        )},
    })

    with pytest.raises(RuntimeError, match="efficacy_amendment_sha256"):
        runner.verify_gate_receipt(
            tmp_path, _expected_gate_core({"example.py": source_hash} | {
                "protocols/xmem_token_affinity_efficacy_amendment_2026-11-08.json": "x"
            }) | {"efficacy_amendment_sha256": "changed-amendment"}
        )
    with pytest.raises(RuntimeError, match="source inventory"):
        runner.verify_gate_receipt(
            tmp_path, _expected_gate_core({"example.py": "changed-source-hash"})
        )
