import pytest
import torch

from modeling.lean_initial_multimask import initial_multimask


class FakeModel:
    def _use_multimask(self, is_init_cond_frame, point_inputs):
        return False

    def _forward_sam_heads(self, *, multimask_output):
        assert multimask_output
        low = torch.arange(6).reshape(1, 3, 1, 2)
        high = low.clone()
        scores = torch.tensor([[0.1, 0.8, 0.3]])
        return low, high, scores, low[:, 1:2], high[:, 1:2], None, None


def test_author_selection_and_restoration():
    model = FakeModel()
    old_choose, old_head = model._use_multimask, model._forward_sam_heads
    events = []
    with initial_multimask(model, events):
        assert model._use_multimask(True, {'point_labels': torch.tensor([[2, 3]])})
        model._forward_sam_heads(multimask_output=True)
        assert events[0]['selected'] == 1
    assert model._use_multimask == old_choose and model._forward_sam_heads == old_head
    assert not model._use_multimask(True, {})


def test_rejects_noninitial_calls_and_restores_on_failure():
    model = FakeModel()
    old_choose, old_head = model._use_multimask, model._forward_sam_heads
    with pytest.raises(ValueError, match='restricted'):
        with initial_multimask(model, []):
            model._use_multimask(False, {})
    assert model._use_multimask == old_choose and model._forward_sam_heads == old_head
