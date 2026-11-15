import sys
import types

import pytest
import torch

from modeling.lean_indexed_spatial_rope import indexed_spatial_rope


def fake_model(monkeypatch):
    pkg = types.ModuleType('sam2')
    modeling = types.ModuleType('sam2.modeling')
    module = types.ModuleType('sam2.modeling.memory_attention')
    pkg.modeling = modeling
    modeling.memory_attention = module
    monkeypatch.setitem(sys.modules, 'sam2', pkg)
    monkeypatch.setitem(sys.modules, 'sam2.modeling', modeling)
    monkeypatch.setitem(sys.modules, 'sam2.modeling.memory_attention', module)
    module._apply_rotary_repeat_2d = lambda x, freq, n: (x, freq, n)
    module._repeat_freqs_for_k = lambda freq, repeats: freq.repeat(repeats, 1)

    class Encoding:
        def __call__(self, _):
            values = torch.arange(4096, dtype=torch.float32).reshape(1, 1, 64, 64)
            return torch.cat([values, values * 2], dim=1)

    class Cross:
        def compute_cis(self, end_x, end_y):
            return torch.arange(end_x * end_y, dtype=torch.float32).to(torch.complex64)[:, None]

    class Attention:
        layers = [types.SimpleNamespace(toma_drop_ratio_ca_kv=0, cross_attn_image=Cross())]

        def __call__(self, **kwargs):
            return self.forward(**kwargs)

        def forward(self, *, curr, memory_pos, num_obj_ptr_tokens):
            del curr
            length = memory_pos.shape[0] - num_obj_ptr_tokens
            base = torch.full((1, 1), 99 + 0j)
            q = module._apply_rotary_repeat_2d(torch.zeros(1, 1, 4096, 2), base, 1)
            k = module._repeat_freqs_for_k(base, length)
            return q, k

    return types.SimpleNamespace(device=torch.device('cpu'), mem_dim=2,
        memory_encoder=types.SimpleNamespace(position_encoding=Encoding()),
        maskmem_tpos_enc=torch.tensor([[[[0.25, 0.5]]]]), memory_attention=Attention()), module


def test_query_grid_and_selected_keys_keep_original_positions(monkeypatch):
    model, module = fake_model(monkeypatch)
    prior_forward, prior_q, prior_k = model.memory_attention.forward, module._apply_rotary_repeat_2d, module._repeat_freqs_for_k
    selected = torch.tensor([4, 3, 22, 1])
    pos = torch.stack((selected.float(), selected.float() * 2), dim=1) + torch.tensor([0.25, 0.5])
    pos = torch.cat((pos, torch.tensor([[9999., 9999.]])))[:, None, :]
    events = []
    with indexed_spatial_rope(model, events):
        q, k = model.memory_attention(curr=[torch.zeros(4096, 1, 2)], memory_pos=pos, num_obj_ptr_tokens=1)
        assert q[2] == 4096
        assert torch.equal(q[1][:, 0].real, torch.arange(4096).float())
        assert torch.equal(k[:, 0].real, selected.float())
    assert model.memory_attention.forward == prior_forward
    assert module._apply_rotary_repeat_2d is prior_q and module._repeat_freqs_for_k is prior_k
    assert events[0]['spatial_keys'] == 4 and events[0]['pointer_keys'] == 1


def test_restores_hooks_when_original_attention_raises(monkeypatch):
    model, module = fake_model(monkeypatch)
    def fail(**kwargs):
        raise RuntimeError('injected failure')
    model.memory_attention.forward = fail
    prior_q, prior_k = module._apply_rotary_repeat_2d, module._repeat_freqs_for_k
    with pytest.raises(RuntimeError, match='injected'):
        with indexed_spatial_rope(model, []):
            model.memory_attention(curr=[torch.zeros(4096, 1, 2)], memory_pos=torch.tensor([[[0.25, 0.5]]]), num_obj_ptr_tokens=0)
    assert model.memory_attention.forward is fail
    assert module._apply_rotary_repeat_2d is prior_q and module._repeat_freqs_for_k is prior_k
