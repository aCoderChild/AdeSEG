import types

import numpy as np
import pytest
import torch

from modeling.lean_early_contrast_anchor import blend_anchor_with_early_contrast, early_contrast_anchor


def test_fixed_blend_preserves_noop_and_expected_token_order():
    feature = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
    positive = torch.tensor([1., 0.])
    negative = torch.tensor([0., 1.])
    anchor = torch.tensor([0.2, 0.3, 0.4, 0.5])
    assert torch.equal(blend_anchor_with_early_contrast(anchor, feature, positive, negative, (2, 2), 0), anchor)
    pure = blend_anchor_with_early_contrast(anchor, feature, positive, negative, (2, 2), 1)
    assert torch.allclose(pure, torch.tensor([.5, -.5, -.5, .5]))
    mixed = blend_anchor_with_early_contrast(anchor, feature, positive, negative, (2, 2), .5)
    assert torch.allclose(mixed, (anchor + pure) / 2)
    with pytest.raises(ValueError, match='Blend strength'):
        blend_anchor_with_early_contrast(anchor, feature, positive, negative, (2, 2), 1.1)


class FakeModel:
    def __init__(self):
        x = torch.arange(64).reshape(8, 8).float()
        self.feature = torch.stack((x + 1, 65 - x))[None]
        self.queued = []

    def _get_image_feature(self, state, frame_idx, batch_size):
        return None, {'backbone_fpn': [self.feature]}, None

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        return 'mask added'

    def _cache_tarr_affinity_from_new_memory(self, current_out, output_dict, frame_idx, obj_idx, is_init_cond_frame, run_mem_encoder, reverse):
        base = torch.tensor([.1, .2, .3, .4])
        current_out['anchor_similarity_cache'] = base
        self.queued.append(base)


def test_changes_cached_anchor_without_changing_queued_routing_and_restores():
    model = FakeModel()
    old = model._get_image_feature, model.add_new_mask, model._cache_tarr_affinity_from_new_memory
    events = []
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 0] = True
    with early_contrast_anchor(model, .5, events):
        model._get_image_feature({}, 3, 1)
        assert model.add_new_mask({}, 3, 1, mask) == 'mask added'
        out = {'maskmem_features': torch.zeros(1, 2, 2, 2)}
        model._cache_tarr_affinity_from_new_memory(out, {}, 3, 0, False, True, False)
        assert not torch.equal(out['anchor_similarity_cache'], model.queued[0])
        assert torch.equal(model.queued[0], torch.tensor([.1, .2, .3, .4]))
    assert (model._get_image_feature, model.add_new_mask, model._cache_tarr_affinity_from_new_memory) == old
    assert events[0]['frame_index'] == 3 and events[0]['tokens'] == 4


def test_rejects_wrong_frame_and_restores_on_error():
    model = FakeModel()
    old = model.add_new_mask
    with pytest.raises(ValueError, match='frame differ'):
        with early_contrast_anchor(model, .5, []):
            model._get_image_feature({}, 3, 1)
            model.add_new_mask({}, 4, 1, np.ones((8, 8), dtype=bool))
    assert model.add_new_mask == old
