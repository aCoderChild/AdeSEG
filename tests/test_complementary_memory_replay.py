"""Replay rejects mutation, writeback, changed scores/pointers and unmatched reads."""
from contextlib import AbstractContextManager

import pytest
import torch
from torch import nn

from scripts import run_complementary_memory_replay as replay


class Observer(AbstractContextManager):
    def __init__(self, *args, **kwargs):
        self.calls = [{'layers': [0, 1, 2, 3]}]

    def __exit__(self, *args):
        return False


class Decoder(nn.Module):
    def forward(self, value):
        return value, torch.ones(1, 1), torch.ones(1, 1, 2), torch.ones(1, 1)


class Model(nn.Module):
    def __init__(self, behavior='clean'):
        super().__init__()
        self.sam_mask_decoder = Decoder()
        self.behavior = behavior

    def _encode_new_memory(self):
        raise AssertionError('Actual encoder should never be called')

    def track_step(self, **kwargs):
        assert kwargs['run_mem_encoder'] is False
        assert kwargs['point_inputs'] is None and kwargs['mask_inputs'] is None
        value = kwargs['current_vision_feats'][0]
        if self.behavior == 'mutate':
            value.add_(1)
        if self.behavior == 'encode':
            self._encode_new_memory()
        masks, _, _, score = self.sam_mask_decoder(value)
        if self.behavior == 'nonfinite':
            masks = masks * float('nan')
        return {'pred_masks': masks, 'obj_ptr': torch.ones(1, 2), 'object_score_logits': score,
                'maskmem_features': None, 'maskmem_pos_enc': None}


def sample():
    return {'frame': 2, 'num_frames': 4, 'vision': [torch.ones(1, 1, 2, 2)],
            'positions': [torch.zeros(1, 1, 2, 2)], 'sizes': [(2, 2)],
            'output_dict': {'cond_frame_outputs': {0: {'pred_masks': torch.ones(1, 1, 2, 2)}},
                            'non_cond_frame_outputs': {1: {'pred_masks': torch.zeros(1, 1, 2, 2)}}}}


def test_clean_replay_repeat_and_hooks_removed(monkeypatch):
    monkeypatch.setattr(replay, 'NativeComplementaryRead', Observer)
    model, data = Model(), sample()
    before = replay.content_hash(data)
    first = replay.replay(model, data)
    second = replay.replay(model, data)
    replay.check_exact(first[0], second[0])
    assert replay.content_hash(data) == before
    assert len(model.sam_mask_decoder._forward_hooks) == 0
    assert torch.equal(first[2]['raw_decoder_logits'], data['vision'][0])


@pytest.mark.parametrize('behavior,match', [('mutate', 'mutated input'), ('encode', 'encoded new memory'),
                                          ('nonfinite', 'Nonfinite')])
def test_rejects_invalid_replay_and_protects_original(monkeypatch, behavior, match):
    monkeypatch.setattr(replay, 'NativeComplementaryRead', Observer)
    model, data = Model(behavior), sample()
    before = replay.content_hash(data)
    with pytest.raises(RuntimeError, match=match):
        replay.replay(model, data)
    assert replay.content_hash(data) == before
    assert len(model.sam_mask_decoder._forward_hooks) == 0


@pytest.mark.parametrize('key', replay.OUTPUT_KEYS)
def test_identity_checks_logits_scores_and_pointers(key):
    reference = {k: torch.zeros(1) for k in replay.OUTPUT_KEYS}
    changed = {**reference, key: torch.tensor([1e-6])}
    with pytest.raises(RuntimeError, match=key):
        replay.check_exact(changed, reference)


def test_layout_rejects_changed_pointer_and_missing_layer():
    reference = {k: True for k in replay.LAYOUT_KEYS}
    reference['layers'] = [0, 1, 2, 3]
    replay.check_layout(reference, reference)
    with pytest.raises(RuntimeError, match='pointer_input'):
        replay.check_layout({**reference, 'pointer_input_sha256': 'changed'}, reference)
    with pytest.raises(RuntimeError, match='layer'):
        replay.check_layout({**reference, 'layers': [0, 1, 2]}, reference)


def test_summary_keeps_absent_denominator_and_present_only_dice():
    rows = [{'present': True, 'absent_fp': None, 'scores': {'dice': .8, 'iou': .7, 'boundary_f': .6}},
            {'present': False, 'absent_fp': True, 'scores': {'dice': 0., 'iou': 0., 'boundary_f': 0.}},
            {'present': False, 'absent_fp': False, 'scores': {'dice': 1., 'iou': 1., 'boundary_f': 1.}}]
    result = replay.arm_summary(rows, 'weights')
    assert result['present_dice'] == .8
    assert result['gt_empty_false_alarm_count'] == 1
    assert result['gt_empty_false_alarm_rate'] == .5
    assert result['gt_empty_frames'] == 2
