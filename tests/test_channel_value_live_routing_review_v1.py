"""Independent synthetic review; this is not a real-MPS propagation test."""

import pytest
import torch

from modeling.lean_channel_value_routing_v1 import route_channel_value_refinement
from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1
from test_boundary_video_routing_v1 import NumericPredictor
from test_lean_contrast_refinement_integration import Refiner, run


@pytest.mark.parametrize('arm,output_delta,memory_delta', [
    ('A', 0, 0), ('B', 1, 0), ('C', 0, 1), ('D', 1, 1),
])
def test_routing_keeps_cached_mask_pointer_score_and_conditioning_native(arm, output_delta, memory_delta):
    predictor = NumericPredictor().eval()
    original, native = run(predictor)
    telemetry = []
    with route_channel_value_refinement(predictor, Refiner().eval(), arm,
                                        test_only=True, telemetry=telemetry):
        result, output = run(predictor)
        assert torch.equal(result['pred_masks'], original['pred_masks'])
        assert result['obj_ptr'] == original['obj_ptr']
        assert torch.equal(result['object_score_logits'], original['object_score_logits'])
        assert torch.equal(output, native + output_delta)
        assert torch.allclose(result['memory'], original['memory'] + memory_delta)
        conditioned, conditioned_output = run(predictor, conditioning=True,
                                               mask_inputs=torch.ones(1, 1, 4, 4))
        assert torch.equal(conditioned_output, native)
        assert torch.equal(conditioned['memory'], original['memory'])
    assert telemetry[1]['conditioning']
    assert telemetry[1]['native']['sha256'] == telemetry[1]['output']['sha256']
    assert not any(x in vars(predictor) for x in ('_track_step', '_encode_memory_in_output',
                                                   '_run_single_frame_inference'))


def test_learned_model_without_authorized_runtime_fails_closed():
    predictor = NumericPredictor().eval()
    untrained = MultiscaleChannelValueRefinerV1(mode='foreground_background_asinh').eval()
    untrained._trained_checkpoint_loaded = True
    with pytest.raises(RuntimeError, match='runtime provenance'):
        with route_channel_value_refinement(predictor, untrained, 'D'):
            pass
    with pytest.raises(RuntimeError, match='runtime provenance'):
        with route_channel_value_refinement(predictor, untrained, 'D', test_only=True):
            pass


def test_postprompt_corrections_fail_and_restore_hooks():
    predictor = NumericPredictor().eval()
    with route_channel_value_refinement(predictor, Refiner().eval(), 'B', test_only=True):
        for override in ({'point_inputs': {'x': 1}}, {'mask_inputs': torch.ones(1)},
                         {'prev_sam_mask_logits': torch.ones(1)}, {'store_as_cond': True}):
            with pytest.raises(ValueError, match='one-shot protocol'):
                run(predictor, **override)
        assert run(predictor)[1].shape == (1, 1, 2, 2)
    assert '_track_step' not in vars(predictor)


def test_no_object_sentinel_is_observed_not_a_refiner_gate():
    class SuppressedPredictor(NumericPredictor):
        def _track_step(self, frame_idx, is_init_cond_frame, current_vision_feats,
                        current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
                        output_dict, num_frames, track_in_reverse, prev_sam_mask_logits,
                        inference_state=None, obj_idx=None):
            current, sam, early, deep = super()._track_step(
                frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
                feat_sizes, point_inputs, mask_inputs, output_dict, num_frames,
                track_in_reverse, prev_sam_mask_logits, inference_state, obj_idx)
            low = torch.full_like(sam[3], -1024.)
            high = torch.full_like(sam[4], -1024.)
            return current, (*sam[:3], low, high, *sam[5:]), early, deep

    predictor = SuppressedPredictor().eval()
    telemetry = []
    # Synthetic delta verifies routing semantics, not trained-model behavior.
    with route_channel_value_refinement(predictor, Refiner(delta=2048).eval(), 'D',
                                        test_only=True, telemetry=telemetry):
        state, output = run(predictor)
    assert (output > 0).all() and (state['memory'] > 0).all()
    assert state['object_score_logits'].item() == -2
    assert telemetry[0]['native_exact_suppressed']
    assert telemetry[0]['output']['positive_pixels'] == output.numel()
    assert not telemetry[0]['native_object_present']
