from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modeling.native_stage_capture import NativeStageCapture


class Attention(nn.Module):
    def forward(self, q, k, v, num_k_exclude_rope=0):
        return q


class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({'cross_attn_image': Attention()}) for _ in range(4)])

    def forward(self, *, curr):
        value = curr[0].transpose(0, 1)
        for layer in self.layers:
            value = layer['cross_attn_image'](q=value, k=value, v=value, num_k_exclude_rope=1)
        return value.transpose(0, 1)


class Decoder(nn.Module):
    def forward(self, *, image_embeddings, high_res_features):
        return image_embeddings[:, :1], torch.ones(1, 1), torch.ones(1, 1, 8), torch.ones(1, 1)


def model():
    return SimpleNamespace(memory_attention=Memory(), sam_mask_decoder=Decoder())


def test_stage_capture_is_read_only_detached_and_cleans_up():
    m = model()
    features = torch.arange(32.).reshape(4, 1, 8).requires_grad_()
    plain = m.memory_attention(curr=[features])
    with NativeStageCapture(m) as observer:
        observer.begin(True)
        conditioned = m.memory_attention(curr=[features])
        image = conditioned.permute(1, 2, 0).reshape(1, 8, 2, 2)
        m.sam_mask_decoder(image_embeddings=image, high_res_features=[image])
        assert torch.equal(plain, conditioned)
        assert torch.equal(observer.tensors['current_image'], features)
        assert not observer.tensors['current_image'].requires_grad
        assert [r['layer'] for r in observer.layers] == [0, 1, 2, 3]
        with torch.no_grad():
            features.add_(1)
        assert not torch.equal(observer.tensors['current_image'], features)
    assert not m.memory_attention._forward_hooks and not m.sam_mask_decoder._forward_pre_hooks


def test_stage_capture_rejects_wrong_decoder_features_and_cleans_up():
    m = model()
    with pytest.raises(RuntimeError, match='Decoder input'):
        with NativeStageCapture(m) as observer:
            observer.begin(True)
            m.memory_attention(curr=[torch.zeros(4, 1, 8)])
            m.sam_mask_decoder(image_embeddings=torch.ones(1, 8, 2, 2), high_res_features=[])
    assert not m.memory_attention._forward_hooks and not m.sam_mask_decoder._forward_pre_hooks


def test_inactive_stage_capture_does_not_keep_tensors():
    m = model()
    with NativeStageCapture(m) as observer:
        observer.begin(False)
        m.memory_attention(curr=[torch.zeros(4, 1, 8)])
        assert not observer.tensors
