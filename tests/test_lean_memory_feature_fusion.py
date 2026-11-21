"""CPU contractual tests for Lean-SAM2 early-FPN memory enrichment."""
from dataclasses import asdict

import pytest
import torch

from modeling.lean_memory_feature_fusion import (
    EarlyFeatureMemoryFusion, MemoryFeatureFusionConfig, implementation_sha256,
)
from adeseg.inference.lean_memory_feature_fusion import route_memory_feature_fusion


CFG = MemoryFeatureFusionConfig(early_channels=(8, 8), memory_channels=4,
                                hidden_channels=8, max_relative_residual=0.25)


def inputs(dtype=torch.float32):
    torch.manual_seed(20261120)
    early = [torch.randn(1, 8, 8, 8, dtype=dtype),
             torch.randn(1, 8, 4, 4, dtype=dtype)]
    memory = torch.randn(1, 4, 2, 2, dtype=dtype)
    mask = torch.randn(1, 1, 8, 8, dtype=dtype)
    score = torch.ones(1, dtype=dtype)
    return early, memory, mask, score


class FakePinnedLean:
    def __init__(self):
        self.training = False
        self.calls = 0

    def _encode_new_memory(self, current_vision_feats, feat_sizes,
                           pred_masks_high_res, object_score_logits,
                           is_mask_from_pts=False):
        self.calls += 1
        memory = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2) / 10
        position = [torch.randn_like(memory)]
        return memory, position

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats,
                   feat_sizes, pred_masks_high_res, object_score_logits):
        return self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=pred_masks_high_res,
            object_score_logits=object_score_logits,
        )


def fake_inputs():
    levels = [torch.randn(64, 1, 8), torch.randn(16, 1, 8),
              torch.randn(4, 1, 8)]
    return (levels, [(8, 8), (4, 4), (2, 2)],
            torch.randn(1, 1, 8, 8), torch.ones(1))


def make_trained(fusion):
    # Test fixture only: imitate loading a source-checked trained checkpoint.
    state = {k: v.clone() for k, v in fusion.state_dict().items()}
    state['output_projection.bias'].fill_(2)
    payload = {
        'format': fusion.CHECKPOINT_FORMAT,
        'config': asdict(fusion.config),
        'implementation_sha256': implementation_sha256(),
        'backbone_sha256': 'a' * 64,
        'train_manifest_sha256': 'b' * 64,
        'training_steps': 1,
        'model_state_dict': state,
    }
    fusion.load_trained_state(payload, backbone_sha256='a' * 64,
                              train_manifest_sha256='b' * 64)
    return payload


def test_pinned_lean_hiera_bplus_fpn_channels_and_memory_identity():
    # SAM2's decoder conv_s0/conv_s1 produce 32/64, memory encoder 64.
    cfg = MemoryFeatureFusionConfig()
    assert cfg.early_channels == (32, 64)
    assert cfg.memory_channels == 64
    model = EarlyFeatureMemoryFusion(cfg).eval()
    early = [torch.randn(1, 32, 8, 8), torch.randn(1, 64, 4, 4)]
    native = torch.randn(1, 64, 2, 2)
    mask, presence = torch.randn(1, 1, 8, 8), torch.ones(1)
    assert torch.equal(model(early, native, mask, presence), native)
    with pytest.raises(ValueError, match='level 0'):
        model([torch.randn(1, 256, 8, 8), early[1]], native, mask, presence)


def test_all_matched_input_modes_preserve_native_memory_contract():
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    make_trained(fusion)
    early, native, mask, score = inputs()
    for mode in fusion.INPUT_MODES:
        enriched = fusion(early, native, mask, score, input_mode=mode)
        assert enriched.shape == native.shape
        assert enriched.dtype == native.dtype
        assert torch.isfinite(enriched).all()
    with pytest.raises(ValueError, match="input_mode"):
        fusion(early, native, mask, score, input_mode="invalid")
    with pytest.raises(ValueError, match="at least two"):
        single = EarlyFeatureMemoryFusion(
            MemoryFeatureFusionConfig(early_channels=(8,),
                                      memory_channels=4, hidden_channels=8)
        )
        single(early[:1], native, mask, score, input_mode="early1_only")


def test_both_early_scales_have_a_differentiable_path_to_memory():
    fusion = EarlyFeatureMemoryFusion(CFG).train()
    # At identity initialization the last projection is zero, so gradients
    # reach the early branches only after that projection starts learning.
    with torch.no_grad():
        fusion.output_projection.weight.fill_(0.02)
    early, native, mask, presence = inputs()
    early = [x.detach().requires_grad_(True) for x in early]
    out = fusion(early, native, mask, presence)
    out.square().mean().backward()
    for feature in early:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
        assert feature.grad.abs().sum() > 0


def test_target_anchor_similarity_drift_is_a_separate_diagnostic():
    class AnchorAwarePredictor(FakePinnedLean):
        def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats,
                       feat_sizes, pred_masks_high_res, object_score_logits,
                       output_dict):
            return self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes, pred_masks_high_res=pred_masks_high_res,
                object_score_logits=object_score_logits,
            )

    predictor = AnchorAwarePredictor()
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    make_trained(fusion)
    early, shapes, mask, scores = fake_inputs()
    telemetry = []
    anchors = {"anchor_mgr": {"anchor_tokens": torch.randn(3, 4)}}
    with route_memory_feature_fusion(
            predictor, fusion, telemetry=telemetry, input_mode="early0_only"):
        predictor.track_step(1, False, early, shapes, mask, scores, anchors)
    assert len(telemetry) == 1
    row = telemetry[0]
    assert row["fusion_input_mode"] == "early0_only"
    assert row["anchor_reference_available"] is True
    assert -1.0 <= row["anchor_similarity_native_mean"] <= 1.0
    assert -1.0 <= row["anchor_similarity_enriched_mean"] <= 1.0
    assert row["anchor_similarity_mean_abs_shift"] >= 0.0


def test_checkpoint_rejects_dtype_conversion_overflow():
    model = EarlyFeatureMemoryFusion(CFG)
    payload = make_trained(EarlyFeatureMemoryFusion(CFG))
    bad = dict(payload['model_state_dict'])
    bad['output_projection.bias'] = torch.full(
        bad['output_projection.bias'].shape, 1e300, dtype=torch.float64)
    with pytest.raises(ValueError, match='tensors'):
        model.load_trained_state(
            dict(payload, model_state_dict=bad),
            backbone_sha256='a' * 64, train_manifest_sha256='b' * 64)
    assert not model._trained_checkpoint_loaded


def test_zero_residual_identity_and_no_mutation():
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    early, memory, mask, score = inputs()
    before = memory.clone()
    out = fusion(early, memory, mask, score)
    assert torch.equal(out, memory)
    assert torch.equal(before, memory)
    assert out.dtype == memory.dtype and out.shape == memory.shape


def test_native_precision_and_presence_gate():
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    make_trained(fusion)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        early, memory, mask, score = inputs(dtype)
        out = fusion(early, memory, mask, score)
        assert out.dtype == dtype and out.shape == memory.shape
        assert not torch.equal(out, memory)
        absent = fusion(early, memory, mask, -score)
        assert torch.equal(absent, memory)
        assert torch.isfinite(out).all()
        # Residual cannot exceed its stated per-entry amplitude bound.
        bound = 0.25 * float(memory.float().std(unbiased=False))
        assert float((out.detach().float() - memory.float()).abs().amax()) <= bound + 0.002


def test_gradients_begin_at_zero_output_head():
    fusion = EarlyFeatureMemoryFusion(CFG).train()
    early, memory, mask, score = inputs()
    out = fusion(early, memory, mask, score)
    out.square().sum().backward()
    assert fusion.output_projection.weight.grad is not None
    assert fusion.output_projection.weight.grad.abs().sum() > 0


def test_nonfinite_invalid_geometry_and_config_fail_closed():
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    early, memory, mask, score = inputs()
    with pytest.raises(ValueError, match='levels'):
        fusion(early[:1], memory, mask, score)
    with pytest.raises(ValueError, match='level 0'):
        fusion([early[0][:, :7], early[1]], memory, mask, score)
    with pytest.raises(ValueError, match='Non-finite'):
        fusion(early, memory, mask * float('nan'), score)
    with pytest.raises(ValueError, match='one native predicted-mask'):
        fusion(early, memory, mask[:, :0], score)
    with pytest.raises(ValueError, match='configuration'):
        MemoryFeatureFusionConfig(early_channels=(), memory_channels=4)


def test_source_bound_training_load_does_not_partially_mutate_on_failure():
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    payload = make_trained(fusion)
    fusion2 = EarlyFeatureMemoryFusion(CFG)
    before = {k: v.clone() for k, v in fusion2.state_dict().items()}
    invalid = dict(payload, implementation_sha256='0' * 64)
    with pytest.raises(ValueError, match='provenance'):
        fusion2.load_trained_state(invalid, backbone_sha256='a' * 64,
                                   train_manifest_sha256='b' * 64)
    assert all(torch.equal(v, before[k]) for k, v in fusion2.state_dict().items())
    bad_tensors = dict(payload['model_state_dict'])
    bad_tensors['output_projection.weight'] = bad_tensors['output_projection.weight'][:, :1]
    with pytest.raises(ValueError, match='tensors'):
        fusion2.load_trained_state(dict(payload, model_state_dict=bad_tensors),
                                   backbone_sha256='a' * 64,
                                   train_manifest_sha256='b' * 64)
    assert all(torch.equal(v, before[k]) for k, v in fusion2.state_dict().items())


def test_routing_exact_identity_conditional_bypass_and_restoration():
    predictor = FakePinnedLean()
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    original_track = predictor.track_step.__func__
    original_encode = predictor._encode_new_memory.__func__
    levels, sizes, mask, score = fake_inputs()
    history = []
    with route_memory_feature_fusion(predictor, fusion, identity_control=True,
                                     telemetry=history):
        for frame, cond in ((0, True), (1, False)):
            mem, pos = predictor.track_step(frame, cond, levels, sizes, mask, score)
            assert torch.equal(mem, torch.arange(16).reshape(1, 4, 2, 2)/10)
            assert isinstance(pos, list) and pos[0].shape == mem.shape
        # Re-encoding outside the native track_step stays unchanged.
        mem, _ = predictor._encode_new_memory(levels, sizes, mask, score)
        assert torch.equal(mem, torch.arange(16).reshape(1, 4, 2, 2)/10)
        assert len(history) == 1 and history[0]['changed_elements'] == 0
        assert history[0]['changed_elements_after_bf16'] == 0
        assert history[0]['changed_spatial_positions_after_bf16'] == 0
        assert history[0]['mean_abs_delta_after_bf16'] == 0
        assert history[0]['native_token_count'] == history[0]['enriched_token_count'] == 4
    assert predictor.track_step.__func__ is original_track
    assert predictor._encode_new_memory.__func__ is original_encode


def test_routing_learned_changes_memory_only_without_extra_tokens():
    predictor = FakePinnedLean()
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    make_trained(fusion)
    levels, sizes, mask, score = fake_inputs()
    history = []
    with route_memory_feature_fusion(predictor, fusion, telemetry=history):
        cond, cond_pos = predictor.track_step(0, True, levels, sizes, mask, score)
        memory, pos = predictor.track_step(1, False, levels, sizes, mask, score)
        assert torch.equal(cond, torch.arange(16).reshape(1, 4, 2, 2)/10)
        assert memory.shape == cond.shape
        assert not torch.equal(memory, cond)
        assert len(pos) == len(cond_pos) == 1
        assert history[0]['changed_elements'] > 0
        assert history[0]['changed_elements_after_bf16'] > 0
        assert history[0]['changed_spatial_positions_after_bf16'] > 0
        assert 0 < history[0]['fraction_spatial_positions_changed_after_bf16'] <= 1
        assert history[0]['relative_l2_delta_after_bf16'] > 0
        assert len(history[0]['early_fpn_rms']) == 2
    assert 'track_step' not in vars(predictor)
    assert '_encode_new_memory' not in vars(predictor)


def test_live_routing_rejects_corrupted_memory_position_and_anchor():
    class MalformedPositionPredictor(FakePinnedLean):
        def _encode_new_memory(self, current_vision_feats, feat_sizes,
                               pred_masks_high_res, object_score_logits,
                               is_mask_from_pts=False):
            native = torch.ones(1, 4, 2, 2)
            return native, [torch.ones(1, 4, 3, 2)]

    levels, sizes, mask, score = fake_inputs()
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    bad_pos = MalformedPositionPredictor()
    with route_memory_feature_fusion(bad_pos, fusion, identity_control=True):
        with pytest.raises(ValueError, match="positional-encoding"):
            bad_pos.track_step(1, False, levels, sizes, mask, score)

    class BadAnchorsPredictor(FakePinnedLean):
        def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats,
                       feat_sizes, pred_masks_high_res, object_score_logits,
                       output_dict):
            return self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes, pred_masks_high_res=pred_masks_high_res,
                object_score_logits=object_score_logits,
            )

    bad_anchor_predictor = BadAnchorsPredictor()
    with route_memory_feature_fusion(bad_anchor_predictor, fusion,
                                     identity_control=True):
        with pytest.raises(ValueError, match="target anchors"):
            bad_anchor_predictor.track_step(
                1, False, levels, sizes, mask, score,
                {"anchor_mgr": {"initialized": True,
                                "anchor_tokens": torch.randn(2, 8)}})


def test_live_routing_rejects_malformed_source_scale_order():
    predictor = FakePinnedLean()
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    _, _, mask, score = fake_inputs()
    early = [torch.randn(16, 1, 8), torch.randn(64, 1, 8),
             torch.randn(4, 1, 8)]
    with route_memory_feature_fusion(predictor, fusion, identity_control=True):
        with pytest.raises(ValueError, match="scale hierarchy"):
            predictor.track_step(
                1, False, early, [(4, 4), (8, 8), (2, 2)], mask, score)


def test_small_fp32_residual_can_disappear_after_bf16_storage():
    class ConstantMemoryLean(FakePinnedLean):
        def _encode_new_memory(self, current_vision_feats, feat_sizes,
                               pred_masks_high_res, object_score_logits,
                               is_mask_from_pts=False):
            base = torch.ones(1, 4, 2, 2)
            return base, [torch.zeros_like(base)]

    predictor = ConstantMemoryLean()
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    make_trained(fusion)
    levels, sizes, mask, score = fake_inputs()
    history = []
    with route_memory_feature_fusion(predictor, fusion, telemetry=history):
        predictor.track_step(1, False, levels, sizes, mask, score)
    assert history[0]["changed_elements"] > 0
    assert history[0]["changed_elements_after_bf16"] == 0
    assert history[0]["fraction_spatial_positions_changed_after_bf16"] == 0.0


def test_hook_conflicts_untrained_and_errors():
    p = FakePinnedLean()
    fusion = EarlyFeatureMemoryFusion(CFG).eval()
    with pytest.raises(ValueError, match='trained'):
        with route_memory_feature_fusion(p, fusion):
            pass
    p._track_step = lambda: None
    with pytest.raises(RuntimeError, match='Conflicting'):
        with route_memory_feature_fusion(p, fusion, identity_control=True):
            pass
    del p._track_step
    levels, sizes, mask, score = fake_inputs()
    with pytest.raises(ValueError, match='level count'):
        with route_memory_feature_fusion(p, fusion, identity_control=True):
            p.track_step(1, False, levels[:2], sizes[:2], mask, score)
    assert 'track_step' not in vars(p)
    assert '_encode_new_memory' not in vars(p)
