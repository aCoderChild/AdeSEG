import torch
import pytest
from modeling.cfbi_training import capture_training_read, clone_tensors


class FakeModel:
    def _get_image_feature(self, state, frame, batch_size):
        return None, None, [torch.tensor([1.])], [torch.tensor([2.])], [(1, 1)]


def test_capture_excludes_current_future_and_clones_native_memory():
    source = {'maskmem_features': torch.ones(1), 'pred_masks': torch.ones(1)}
    state = {'cached_features': {3: 'cached'}, 'num_frames': 9,
             'output_dict_per_obj': {0: {'cond_frame_outputs': {0: source},
                                          'non_cond_frame_outputs': {2: source, 3: source, 4: source}}}}
    with torch.inference_mode():
        sample = capture_training_read(FakeModel(), state, 3)
    assert not torch.is_inference(sample['vision'][0])
    assert list(sample['output_dict']['cond_frame_outputs']) == [0]
    assert list(sample['output_dict']['non_cond_frame_outputs']) == [2]
    sample['output_dict']['cond_frame_outputs'][0]['pred_masks'].zero_()
    assert source['pred_masks'].item() == 1
    assert sample['frame'] == 3 and sample['num_frames'] == 9
    with pytest.raises(ValueError, match='image cache'):
        capture_training_read(FakeModel(), state, 5)


def test_clone_converts_inference_tensors_for_backward():
    with torch.inference_mode():
        inference = torch.ones(2)
    result = clone_tensors({'feature': inference})['feature']
    assert not torch.is_inference(result)
    weight = torch.ones(2, requires_grad=True)
    (weight * result).sum().backward()
    assert torch.equal(weight.grad, result)
