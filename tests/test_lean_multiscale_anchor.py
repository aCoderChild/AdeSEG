import torch
import torch.nn.functional as F
import pytest

from modeling.lean_multiscale_anchor import install_multiscale_anchor, multiscale_anchor_score


class FakeLean:
    def __init__(self):
        self.tarr = []
        self.mem_keep_ratio = .25
        self.dump_calls = 0

    def _compute_anchor_similarity_maxcos(self, tokens, anchors):
        vectors = F.normalize(tokens[:, 0].float(), dim=-1)
        refs = F.normalize(anchors.reshape(-1, tokens.shape[-1]).float(), dim=-1)
        return (vectors @ refs.T).max(dim=-1).values

    def _cache_tarr_affinity_from_new_memory(self, current_out, output_dict, frame_idx, obj_idx,
                                              is_init_cond_frame, run_mem_encoder, reverse):
        memory = current_out['maskmem_features']
        local = self._compute_anchor_similarity_maxcos(memory.flatten(2).permute(2, 0, 1),
                                                        output_dict['anchor_mgr']['anchor_tokens'])
        current_out['anchor_similarity_cache'] = local.clone()
        self.tarr.append(local.clone())

    def _dump_retained_tokens_record(self, **kwargs):
        self.dump_calls += 1


def test_zero_strength_preserves_original_cache_and_tarr():
    model = FakeLean()
    memory = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4) + 1
    anchors = torch.tensor([[1., 0.]])
    expected = model._compute_anchor_similarity_maxcos(memory.flatten(2).permute(2, 0, 1), anchors)
    out = {'maskmem_features': memory}
    events = install_multiscale_anchor(model, 0.)
    model._cache_tarr_affinity_from_new_memory(out, {'anchor_mgr': {'anchor_tokens': anchors}}, 1, 0, False, True, False)
    assert torch.equal(out['anchor_similarity_cache'], expected)
    assert torch.equal(model.tarr[0], expected)
    assert not events


def test_multiscale_changes_tamp_cache_but_not_tarr_or_memory():
    model = FakeLean()
    torch.manual_seed(3)
    memory = torch.randn(1, 2, 8, 8)
    original_memory = memory.clone()
    anchors = torch.tensor([[1., 0.], [0., 1.]])
    local = model._compute_anchor_similarity_maxcos(memory.flatten(2).permute(2, 0, 1), anchors)
    out = {'maskmem_features': memory}
    events = install_multiscale_anchor(model, .5)
    model._cache_tarr_affinity_from_new_memory(out, {'anchor_mgr': {'anchor_tokens': anchors}}, 2, 0, False, True, False)
    assert torch.equal(model.tarr[0], local)
    assert torch.equal(memory, original_memory)
    assert out['anchor_similarity_cache'].shape == local.shape
    assert not torch.equal(out['anchor_similarity_cache'], local)
    assert events[0]['tokens'] == 64
    assert events[0]['mean_abs_score_change'] > 0


def test_rejects_misaligned_grids_and_invalid_strength():
    model = FakeLean()
    with pytest.raises(ValueError):
        multiscale_anchor_score(model, torch.ones(1, 2, 7, 8), torch.ones(1, 2), torch.ones(56), .5)
    with pytest.raises(ValueError):
        install_multiscale_anchor(model, 1.1)


def test_selection_trace_verifies_original_budget_and_spatial_indices():
    model = FakeLean()
    trace = []
    install_multiscale_anchor(model, .5, selection_events=trace)
    kwargs = {'usage_scores': torch.ones(16), 'keep_idx': torch.tensor([2, 5, 7, 9]),
              'main_keep_idx': torch.tensor([2, 5, 7, 9]), 'latest_prev': {'frame_idx': 4}, 'frame_idx': 5}
    model._dump_retained_tokens_record(**kwargs)
    assert model.dump_calls == 1
    assert trace[0]['candidates'] == 16 and trace[0]['kept'] == 4
    with pytest.raises(RuntimeError):
        model._dump_retained_tokens_record(**{**kwargs, 'keep_idx': torch.tensor([2, 5, 7, 16])})
