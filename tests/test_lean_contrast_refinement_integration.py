import pytest
import torch
import torch.nn.functional as F

from modeling.lean_contrast_refinement_integration import route_lean_contrast_refinement
from modeling.multilevel_contrast_refiner import MultilevelContrastRefiner


class Refiner(torch.nn.Module):
    def __init__(self, delta=1):
        super().__init__()
        self.delta = delta
        self._trained_checkpoint_loaded = True

    def forward(self, early, deep, low):
        assert early.shape == (1, 32, 2, 2)
        assert deep.shape == (1, 256, 1, 1)
        return low + self.delta


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fill_hole_area = 0
        self.last_memory = None

    def _track_step(self, frame_idx, is_init_cond_frame, current_vision_feats,
                    current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
                    output_dict, num_frames, track_in_reverse, prev_sam_mask_logits,
                    inference_state=None, obj_idx=None):
        low = torch.tensor([[[[-1., 0.], [1., 2.]]]])
        high = F.interpolate(low, size=(4, 4), mode='bilinear', align_corners=False)
        return {}, (None, None, None, low, high, 'pointer', 'score'), [torch.zeros(1, 32, 2, 2)], torch.zeros(1, 256, 1, 1)

    def _encode_memory_in_output(self, current_vision_feats, feat_sizes, point_inputs,
                                 run_mem_encoder, high_res_masks, object_score_logits,
                                 current_out):
        assert object_score_logits == 'score'
        self.last_memory = high_res_masks.clone() if run_mem_encoder else None
        current_out['memory'] = self.last_memory

    def _run_single_frame_inference(self, inference_state, obj_idx, output_dict,
                                    frame_idx, batch_size, is_init_cond_frame,
                                    point_inputs, mask_inputs, reverse, run_mem_encoder,
                                    store_as_cond=False, prev_sam_mask_logits=None):
        out, sam, _, _ = self._track_step(
            frame_idx, is_init_cond_frame, None, None, None, point_inputs, mask_inputs,
            output_dict, 2, reverse, prev_sam_mask_logits, inference_state, obj_idx)
        low, high, ptr, score = sam[3:]
        self._encode_memory_in_output(None, None, point_inputs, run_mem_encoder,
                                      high, score, out)
        return {'pred_masks': low.clone(), 'memory': out['memory'], 'obj_ptr': ptr,
                'object_score_logits': score}, low


class SWRSignaturePredictor(Predictor):
    # The official-aligned SWR subclass omits the base method's store_as_cond.
    def _run_single_frame_inference(self, inference_state, obj_idx, output_dict,
                                    frame_idx, batch_size, is_init_cond_frame,
                                    point_inputs, mask_inputs, reverse, run_mem_encoder,
                                    prev_sam_mask_logits=None):
        return super()._run_single_frame_inference(
            inference_state, obj_idx, output_dict, frame_idx, batch_size,
            is_init_cond_frame, point_inputs, mask_inputs, reverse, run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits)


def run(model, *, conditioning=False, run_mem_encoder=True, **overrides):
    kwargs = dict(inference_state={'obj_ids': [1]}, obj_idx=0, output_dict={},
                  frame_idx=0 if conditioning else 1, batch_size=1,
                  is_init_cond_frame=conditioning, point_inputs=None, mask_inputs=None,
                  reverse=False, run_mem_encoder=run_mem_encoder)
    kwargs.update(overrides)
    return model._run_single_frame_inference(**kwargs)


@pytest.mark.parametrize('arm,output_delta,memory_delta', [
    ('A', 0, 0), ('B', 1, 0), ('C', 0, 1), ('D', 1, 1),
])
def test_routes_returned_output_and_memory_independently(arm, output_delta, memory_delta):
    model = Predictor().eval()
    original, original_return = run(model)
    with route_lean_contrast_refinement(model, Refiner().eval(), arm, test_only=True):
        compact, output = run(model)
        assert torch.equal(compact['pred_masks'], original['pred_masks'])
        assert torch.equal(output, original_return + output_delta)
        assert torch.allclose(compact['memory'], original['memory'] + memory_delta)
        assert compact['obj_ptr'] == original['obj_ptr']
        assert compact['object_score_logits'] == original['object_score_logits']
        conditioned, conditioned_output = run(model, conditioning=True)
        assert torch.equal(conditioned['memory'], original['memory'])
        assert torch.equal(conditioned_output, original_return)
    assert '_track_step' not in vars(model)
    assert '_encode_memory_in_output' not in vars(model)
    assert '_run_single_frame_inference' not in vars(model)


def test_identity_and_skipped_encoder():
    model = Predictor().eval()
    baseline, _ = run(model)
    with route_lean_contrast_refinement(model, Refiner(delta=0).eval(), 'D', test_only=True):
        compact, _ = run(model)
        assert torch.equal(compact['memory'], baseline['memory'])
        compact, _ = run(model, run_mem_encoder=False)
        assert compact['memory'] is None


def test_routing_rejects_replaced_internal_hook():
    from types import MethodType

    model = Predictor().eval()
    with route_lean_contrast_refinement(model, Refiner().eval(), 'B', test_only=True):
        model._track_step = MethodType(lambda self, *args, **kwargs: None, model)
        with pytest.raises(RuntimeError, match='routing hook changed'):
            run(model)


def test_swr_subclass_signature():
    model = SWRSignaturePredictor().eval()
    with route_lean_contrast_refinement(model, Refiner().eval(), 'B', test_only=True):
        compact, output = run(model)
        assert torch.equal(output, compact['pred_masks'] + 1)


def test_unbound_learned_refiner_fails_closed_even_with_test_override():
    model = Predictor().eval()
    learned = MultilevelContrastRefiner().eval()
    # This flag alone previously allowed using arbitrary caller-asserted hashes
    # against an unrelated predictor and runtime.
    learned._trained_checkpoint_loaded = True
    with pytest.raises(RuntimeError, match='runtime provenance'):
        with route_lean_contrast_refinement(model, learned, 'D'):
            pass
    with pytest.raises(RuntimeError, match='runtime provenance'):
        with route_lean_contrast_refinement(model, learned, 'D', test_only=True):
            pass


def test_bound_arm_a_checks_runtime_at_entry_and_each_frame(monkeypatch):
    from modeling.lean_verified_runtime_lease import _RuntimeLease

    model = Predictor().eval()
    learned = MultilevelContrastRefiner().eval()
    lease = object.__new__(_RuntimeLease)
    calls = []
    changed = {'value': False}

    def validate(self, active_model, active_refiner, *, tensors):
        assert active_model is model and active_refiner is learned
        calls.append(tensors)
        if changed['value']:
            raise RuntimeError('Verified memory policy changed')

    monkeypatch.setattr(_RuntimeLease, 'validate_live', validate)
    with route_lean_contrast_refinement(model, learned, 'A', runtime_lease=lease):
        assert calls == [True]
        run(model)
        assert calls == [True, False]
        changed['value'] = True
        with pytest.raises(RuntimeError, match='policy changed'):
            run(model)
    with pytest.raises(RuntimeError, match='runtime provenance'):
        with route_lean_contrast_refinement(model, Refiner().eval(), 'B'):
            pass
    # The native no-refiner arm remains available.
    with route_lean_contrast_refinement(model, None, 'A'):
        assert run(model)[1].shape == (1, 1, 2, 2)


def test_fail_closed_and_restore_on_exception():
    model = Predictor().eval()
    refiner = Refiner().eval()
    refiner._trained_checkpoint_loaded = False
    with pytest.raises(ValueError):
        with route_lean_contrast_refinement(model, refiner, 'B', test_only=True):
            pass
    refiner._trained_checkpoint_loaded = True
    with pytest.raises(ValueError):
        with route_lean_contrast_refinement(model, refiner, 'B', test_only=True):
            run(model, point_inputs={'point_labels': [1]})
    assert '_track_step' not in vars(model)
    with route_lean_contrast_refinement(model, refiner, 'B', test_only=True):
        with pytest.raises(ValueError):
            run(model, inference_state={'obj_ids': [1, 2]})
        with pytest.raises(RuntimeError):
            with route_lean_contrast_refinement(model, refiner, 'B', test_only=True):
                pass
        assert run(model)[1].shape == (1, 1, 2, 2)
