from types import SimpleNamespace
import torch
from modeling.lean_utility_source_capture import capture_utility_sources


class FakeModel:
    mem_dim = 4
    def __init__(self):
        def grid(_):
            x = torch.arange(4096, dtype=torch.float32).reshape(1, 1, 64, 64)
            return torch.cat((x, x * 2, x * 3, x * 4), dim=1)
        self.memory_encoder = SimpleNamespace(position_encoding=grid)
        self.maskmem_tpos_enc = torch.zeros(2, 1, 1, 4)
        self.memory_attention = SimpleNamespace(forward=lambda **kw: kw['memory'])
    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        return None, {'backbone_fpn': [torch.full((1, 32, 256, 256), float(frame_idx))]}
    def _materialize_memory_entry(self, prev, device, *, is_cond=False):
        return prev['features'], prev['positions']
    def _prepare_memory_conditioned_features(self, **kwargs):
        entry = kwargs['output_dict']['non_cond_frame_outputs'][1]
        f, p = self._materialize_memory_entry(entry, 'cpu')
        cond = torch.zeros(4096, 1, 4)
        return self.memory_attention.forward(memory=torch.cat((cond, f)), memory_pos=torch.cat((cond, p)), num_obj_ptr_tokens=0)


def test_source_identity_and_positions_are_captured_without_changing_read(tmp_path):
    model = FakeModel(); events = []
    grid = model.memory_encoder.position_encoding(torch.zeros(1, 4, 64, 64)).flatten(2).permute(2, 0, 1)
    entry = {'frame_idx': 1, 'features': torch.ones(2, 1, 4), 'positions': grid[[4, 9]]}
    outputs = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {1: entry}}
    original = model._prepare_memory_conditioned_features(frame_idx=2, output_dict=outputs)
    with capture_utility_sources(model, {2}, tmp_path, events):
        model._get_image_feature({}, 1, 1)
        result = model._prepare_memory_conditioned_features(frame_idx=2, output_dict=outputs)
    assert torch.equal(result, original)
    source = torch.load(tmp_path / 'source_snapshots/000002.pt', weights_only=True)['sources'][0]
    assert source['frame'] == 1 and source['start'] == 4096
    assert source['token_ids'].tolist() == [4, 9]
    assert events[0]['attributed_tokens'] == 2


def test_ambiguous_source_is_not_attributed(tmp_path):
    model = FakeModel(); events = []
    grid = model.memory_encoder.position_encoding(torch.zeros(1, 4, 64, 64)).flatten(2).permute(2, 0, 1)
    entry = {'frame_idx': 1, 'features': torch.ones(2, 1, 4), 'positions': grid[[4, 9]]}
    outputs = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {1: entry}}
    def duplicate(**kwargs):
        f, p = model._materialize_memory_entry(entry, 'cpu')
        cond = torch.zeros(4096, 1, 4)
        return model.memory_attention.forward(memory=torch.cat((cond, f, f)), memory_pos=torch.cat((cond, p, p)), num_obj_ptr_tokens=0)
    model._prepare_memory_conditioned_features = duplicate
    with capture_utility_sources(model, {2}, tmp_path, events):
        model._get_image_feature({}, 1, 1)
        model._prepare_memory_conditioned_features(frame_idx=2, output_dict=outputs)
    saved = torch.load(tmp_path/'source_snapshots/000002.pt', weights_only=True)
    assert saved['sources'] == []
    assert saved['rejected'][0]['frame'] == 1
