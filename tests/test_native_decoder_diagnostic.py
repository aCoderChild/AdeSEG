import pytest
import torch

from scripts.diagnose_native_decoder_masks import verify_stored


@pytest.mark.parametrize('key', ['pred_masks', 'object_score_logits', 'obj_ptr'])
def test_replay_rejects_changed_native_output(key):
    stored = {name: torch.ones(1) for name in
              ['pred_masks', 'object_score_logits', 'obj_ptr', 'maskmem_features']}
    stored['maskmem_features'] = stored['maskmem_features'].bfloat16()
    stored['maskmem_pos_enc'] = [torch.ones(1)]
    output = {**stored, 'maskmem_features': torch.ones(1)}
    verify_stored(output, stored)
    output[key] = torch.zeros(1)
    with pytest.raises(RuntimeError, match=key):
        verify_stored(output, stored)


def test_replay_rejects_changed_position_or_mask_dtype():
    stored = {name: torch.ones(1) for name in
              ['pred_masks', 'object_score_logits', 'obj_ptr', 'maskmem_features']}
    stored['maskmem_pos_enc'] = [torch.ones(1)]
    with pytest.raises(RuntimeError, match='positions'):
        verify_stored({**stored, 'maskmem_pos_enc': [torch.zeros(1)]}, stored)
    with pytest.raises(RuntimeError, match='pred_masks'):
        verify_stored({**stored, 'pred_masks': stored['pred_masks'].half()}, stored)
