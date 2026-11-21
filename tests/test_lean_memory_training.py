"""Capture alignment, sealed TRAIN distillation and live-checkpoint contracts."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from modeling.lean_memory_feature_fusion import (
    EarlyFeatureMemoryFusion, MemoryFeatureFusionConfig,
)
from scripts.capture_lean_memory_train import downsample_capture_features
from scripts.run_lean_memory_fusion_ablation import file_sha256, get_fusion
from training.lean_memory_fusion import (
    train, read_capture, TRAINING_PROTOCOL, audit_teacher_feasibility,
)
from training.split_guard import sha256
from adeseg.inference.lean_memory_feature_fusion import route_memory_feature_fusion


def real_geometry_capture():
    torch.manual_seed(11)
    native = torch.randn(1, 64, 2, 2)
    fpn = [torch.randn(64, 1, 32), torch.randn(16, 1, 64),
           torch.randn(4, 1, 256)]
    shape = [(8, 8), (4, 4), (2, 2)]
    mask = torch.randn(1, 1, 8, 8)
    extra = downsample_capture_features(
        fpn, shape, native, mask, torch.ones(1),
        MemoryFeatureFusionConfig(),
    )
    return {
        "early": extra["early"], "native_memory": native,
        "teacher_memory": native + 0.1,
        "predicted_mask_logits": extra["predicted_mask_logits"],
        "object_score_logits": torch.ones(1),
        "video_id": "train_case", "frame_index": 1,
    }


def test_oracle_teacher_reachability_audit_distinguishes_bound_and_gate():
    config = MemoryFeatureFusionConfig()
    data = real_geometry_capture()
    data["teacher_memory"] = data["native_memory"].clone()
    feasible = audit_teacher_feasibility([data], config)
    assert feasible["mean_fraction_unattainable_elements"] == 0
    assert feasible["fraction_predicted_absent"] == 0

    data["teacher_memory"] = data["native_memory"] + 1e3
    infeasible = audit_teacher_feasibility([data], config)
    assert infeasible["fraction_captures_with_unattainable_target"] == 1
    assert infeasible["max_required_to_allowed_ratio"] > 1

    data["teacher_memory"] = data["native_memory"] + 0.001
    data["object_score_logits"] = -torch.ones(1)
    blocked = audit_teacher_feasibility([data], config)
    assert blocked["fraction_predicted_absent"] == 1
    assert blocked["mean_fraction_unattainable_elements"] == 1


def test_pinned_feature_alignment_matches_native_grid():
    record = real_geometry_capture()
    assert [x.shape for x in record["early"]] == [(1, 32, 2, 2), (1, 64, 2, 2)]
    assert record["predicted_mask_logits"].shape == (1, 1, 2, 2)
    model = EarlyFeatureMemoryFusion().eval()
    assert torch.equal(model(
        record["early"], record["native_memory"],
        record["predicted_mask_logits"], record["object_score_logits"]),
        record["native_memory"])


def setup_training(tmp_path):
    capture = tmp_path / "capture.pt"
    torch.save(real_geometry_capture(), capture)
    manifest = tmp_path / "train_manifest.json"
    manifest.write_text(json.dumps([{
        "split": "train", "source_group": "group_A",
        "tensor_file": "capture.pt", "capture_sha256": sha256(capture),
        "capture_source_sha256": sha256(
            Path(__file__).resolve().parents[1] / "scripts/capture_lean_memory_train.py"),
        "video_id": "train_case", "frame_index": 1,
    }]))
    split = tmp_path / "groups.json"
    split.write_text(json.dumps({
        "train": ["group_A"], "dev": ["group_B"], "test": ["group_C"],
    }))
    return manifest, split, capture


def test_capture_training_produces_valid_authenticated_checkpoint(tmp_path):
    manifest, split, capture = setup_training(tmp_path)
    record = read_capture(capture, MemoryFeatureFusionConfig())
    assert record["video_id"] == "train_case"
    output = tmp_path / "trained.pt"
    args = SimpleNamespace(
        manifest=manifest, split_file=split, output=output,
        backbone_sha256="a" * 64, epochs=2, lr=1e-2,
        preservation_weight=0.1, seed=42, device="cpu",
    )
    receipt = train(args)
    assert receipt["training_steps"] == 2
    assert receipt["training_protocol"] == TRAINING_PROTOCOL
    assert file_sha256(output) == receipt["checkpoint_sha256"]
    model, _ = get_fusion(
        "fusion", output, receipt["checkpoint_sha256"],
        "a" * 64, sha256(manifest), "cpu")
    assert model._trained_checkpoint_loaded
    assert not model.is_identity_initialized()


def test_separately_trained_control_cannot_be_evaluated_as_full(tmp_path):
    manifest, split, _ = setup_training(tmp_path)
    output = tmp_path / "no_early.pt"
    args = SimpleNamespace(
        manifest=manifest, split_file=split, output=output,
        backbone_sha256="a" * 64, epochs=2, lr=1e-2,
        preservation_weight=0.1, seed=42, device="cpu",
        input_mode="no_early",
    )
    receipt = train(args)
    assert receipt["fusion_input_mode"] == "no_early"
    checkpoint, _ = get_fusion(
        "fusion", output, receipt["checkpoint_sha256"],
        "a" * 64, sha256(manifest), "cpu", input_mode="no_early")
    assert checkpoint._trained_checkpoint_loaded
    with pytest.raises(ValueError, match="training protocol"):
        get_fusion("fusion", output, receipt["checkpoint_sha256"],
                   "a" * 64, sha256(manifest), "cpu", input_mode="full")


def test_hard_no_object_gate_rejects_unlearnable_training_capture(tmp_path):
    manifest, split, capture = setup_training(tmp_path)
    data = torch.load(capture, map_location="cpu", weights_only=True)
    data["object_score_logits"] = -torch.ones_like(data["object_score_logits"])
    torch.save(data, capture)
    rows = json.loads(manifest.read_text("utf8"))
    rows[0]["capture_sha256"] = sha256(capture)
    manifest.write_text(json.dumps(rows), encoding="utf8")
    args = SimpleNamespace(
        manifest=manifest, split_file=split,
        output=tmp_path / "no_object_checkpoint.pt",
        backbone_sha256="a" * 64, epochs=2, lr=1e-2,
        preservation_weight=0.1, seed=42, device="cpu",
        input_mode="full",
    )
    with pytest.raises(ValueError, match="No learnable memory captures"):
        train(args)
    assert not args.output.exists()


def test_explicit_no_object_gate_ablation_can_learn_from_absent_capture(tmp_path):
    manifest, split, capture = setup_training(tmp_path)
    data = torch.load(capture, map_location="cpu", weights_only=True)
    data["object_score_logits"] = -torch.ones_like(data["object_score_logits"])
    torch.save(data, capture)
    rows = json.loads(manifest.read_text("utf8"))
    rows[0]["capture_sha256"] = sha256(capture)
    manifest.write_text(json.dumps(rows), encoding="utf8")
    args = SimpleNamespace(
        manifest=manifest, split_file=split,
        output=tmp_path / "allow_absent.pt",
        backbone_sha256="a" * 64, epochs=2, lr=1e-2,
        preservation_weight=0.1, seed=42, device="cpu",
        input_mode="full", allow_no_object_memory_fusion=True,
    )
    receipt = train(args)
    assert receipt["no_object_fusion_ablation"] is True
    assert receipt["trainable_capture_count"] == 1
    assert receipt["skipped_predicted_absent_updates"] == 0
    assert receipt["training_steps"] == 2
    loaded, _ = get_fusion(
        "fusion", args.output, receipt["checkpoint_sha256"],
        "a" * 64, sha256(manifest), "cpu")
    assert loaded.config.preserve_native_no_object is False


def test_invalid_capture_shape_and_sealed_group_rejected(tmp_path):
    manifest, split, capture = setup_training(tmp_path)
    payload = torch.load(capture, map_location="cpu", weights_only=True)
    payload["early"][0] = payload["early"][0][:, :31]
    torch.save(payload, capture)
    with pytest.raises(ValueError, match="Early capture"):
        read_capture(capture, MemoryFeatureFusionConfig())
    args = SimpleNamespace(
        manifest=manifest, split_file=split, output=tmp_path/"out.pt",
        backbone_sha256="a"*64, epochs=1, lr=1e-2,
        preservation_weight=0.0, seed=123, device="cpu",
    )
    with pytest.raises(ValueError, match="SHA256"):
        train(args)


def test_unverified_trainer_checkpoint_cannot_be_loaded_for_live_inference(tmp_path):
    manifest, split, capture = setup_training(tmp_path)
    model = EarlyFeatureMemoryFusion()
    from modeling.lean_memory_feature_fusion import implementation_sha256
    payload = {
        "format": model.CHECKPOINT_FORMAT, "config": asdict(model.config),
        "implementation_sha256": implementation_sha256(),
        "backbone_sha256": "a"*64,
        "train_manifest_sha256": sha256(manifest),
        "training_steps": 2,
        "model_state_dict": model.state_dict(),
    }
    payload["model_state_dict"]["output_projection.bias"].fill_(1.0)
    bad = tmp_path / "pretend_trained.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="training protocol"):
        get_fusion("fusion", bad, file_sha256(bad), "a"*64,
                   sha256(manifest), "cpu")


class RealGeometryPredictor:
    def __init__(self):
        self.training = False

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats,
                   feat_sizes, pred_masks_high_res, object_score_logits):
        return self._encode_new_memory(
            current_vision_feats=current_vision_feats, feat_sizes=feat_sizes,
            pred_masks_high_res=pred_masks_high_res,
            object_score_logits=object_score_logits, is_mask_from_pts=False)

    def _encode_new_memory(self, current_vision_feats, feat_sizes,
                           pred_masks_high_res, object_score_logits,
                           is_mask_from_pts):
        return torch.ones(1, 64, 2, 2), [torch.zeros(1, 64, 2, 2)]


def test_live_scoped_hook_with_32_64_fpn_outputs_preserves_positions_and_tokens():
    p = RealGeometryPredictor()
    model = EarlyFeatureMemoryFusion().eval()
    fpn = [torch.randn(64, 1, 32), torch.randn(16, 1, 64),
           torch.randn(4, 1, 256)]
    tel = []
    with route_memory_feature_fusion(
        p, model, identity_control=True, telemetry=tel
    ):
        out, pos = p.track_step(
            1, False, fpn, [(8,8), (4,4), (2,2)],
            torch.randn(1,1,8,8), torch.ones(1),
        )
    assert out.shape == (1,64,2,2) and pos[0].shape == out.shape
    assert len(tel) == 1
    assert tel[0]["native_token_count"] == tel[0]["enriched_token_count"] == 4
    assert tel[0]["changed_elements"] == 0
