import ast
from functools import partial
import math
from pathlib import Path
from types import FunctionType, MethodType, SimpleNamespace

import pytest
import torch

from modeling.regional_token_selection import author_selectors, prompt_anchors, select_tokens
from modeling.sparse_memory import native_positions, sparse_forward


def test_stage_continuation_stops_on_role_control_loss_or_pooled_harm():
    from scripts.run_regional_token_selection import diagnostic_continuation, full_screen
    contrasts = {a: {'mean_dice_difference': .03, 'sequence_bootstrap_95_ci': [.01, .04],
        'wins': 15, 'equal_sequence_false_alarm_difference': 0., 'pooled_false_alarm_difference': 0.}
        for a in ('native', 'foreground', 'role_blind')}
    result = {'contrasts': contrasts}
    assert diagnostic_continuation(result) and full_screen(result, 21)
    contrasts['role_blind']['mean_dice_difference'] = -.001
    assert not diagnostic_continuation(result) and not full_screen(result, 21)
    contrasts['role_blind']['mean_dice_difference'] = .03
    contrasts['native']['pooled_false_alarm_difference'] = .06
    assert not diagnostic_continuation(result) and not full_screen(result, 21)


def test_statistics_refuse_unmatched_target_denominators():
    from scripts.run_regional_token_selection import ARMS, summarize
    rows = {a: {'seq2': {'target_frames': 10, 'present_frames': 8, 'gt_empty_frames': 2,
                         'original_sam2_parameter_sha256': 'same'}} for a in ARMS}
    rows['regional']['seq2']['gt_empty_frames'] = 1
    with pytest.raises(RuntimeError, match='denominator'):
        summarize(rows)


def test_published_usage_selector_is_reused_exactly():
    _, topk = author_selectors()
    usage = torch.arange(16, dtype=torch.float32)
    actual = select_tokens(torch.randn(16, 4), torch.full((16,), .5), usage,
                           None, "usage", .25)
    assert torch.equal(actual, topk(usage, 4))
    assert torch.where(actual)[0].tolist() == [12, 13, 14, 15]


def test_context_consistency_changes_ranking_at_the_same_budget():
    # All four tokens have the same attention importance. The fourth token is
    # reliable context; the third contradicts its background role.
    features = torch.tensor([[1., 0.], [.9, .1], [1., 0.], [0., 1.]])
    p = torch.tensor([1., 1., 0., 0.])
    anchors = (features[:1], features[-1:])
    usage = torch.tensor([1., 1., 1.01, 1.02])
    fg = select_tokens(features, p, usage, anchors, "foreground", .5)
    regional = select_tokens(features, p, usage, anchors, "regional", .5)
    assert fg[2] and not regional[2]
    assert regional[3] and not fg[3]
    assert fg.sum() == regional.sum() == 2


def test_full_retention_is_identity_for_every_policy():
    for mode in ("identity", "usage", "foreground", "role_blind", "regional"):
        assert select_tokens(torch.randn(16, 8), torch.rand(16), torch.rand(16),
                             None, mode, 1.).all()


def test_role_blind_control_uses_same_anchors_but_ignores_occupancy():
    f = torch.tensor([[1., 0.], [0., 1.], [.5, .5], [-1., 0.]])
    anchors = (f[:1], f[1:2])
    u = torch.tensor([1., 1.01, 1.02, 1.03])
    first = select_tokens(f, torch.zeros(4), u, anchors, "role_blind", .5)
    second = select_tokens(f, torch.ones(4), u, anchors, "role_blind", .5)
    assert torch.equal(first, second)
    assert first[:2].all()


def test_empty_context_has_a_defined_foreground_fallback():
    f = torch.randn(16, 8)
    anchors = (f[:2], f[:0])
    p, usage = torch.ones(16), torch.rand(16)
    assert torch.equal(select_tokens(f, p, usage, anchors, "regional"),
                       select_tokens(f, p, usage, anchors, "foreground"))


def test_invalid_support_fails_before_selection():
    with pytest.raises(ValueError, match="Invalid selector"):
        select_tokens(torch.randn(4, 2), torch.full((4,), float('nan')),
                      torch.ones(4), None, "usage")


def test_prompt_anchor_selection_is_fixed_and_diverse():
    torch.manual_seed(7)
    features = torch.randn(4096, 8)
    probabilities = torch.zeros(4096)
    probabilities[:1024] = 1.
    usage = torch.arange(4096, dtype=torch.float32)
    first = prompt_anchors(features, probabilities, usage)
    repeated = prompt_anchors(features, probabilities, usage)
    assert [len(x) for x in first] == [51, 64]
    assert all(torch.equal(a, b) for a, b in zip(first, repeated))


@pytest.mark.parametrize("frames,pointers", [(1, 0), (2, 4), (7, 12)])
def test_local_sdpa_capture_preserves_native_bit_parity(frames, pointers):
    # Execute unchanged native projection/RoPE/SDPA with a private capture
    # namespace; ensure installing it never modifies torch's global SDPA.
    torch.manual_seed(12)
    path = Path(__file__).resolve().parents[1] / 'external/sam2/sam2/modeling/sam/transformer.py'
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name in ('Attention', 'RoPEAttention')]
    pos = native_positions()
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, Tensor=torch.Tensor,
        math=math, partial=partial, compute_axial_cis=pos.compute_axial_cis,
        apply_rotary_enc=pos.apply_rotary_enc)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    attention = namespace['RoPEAttention'](embedding_dim=16, num_heads=1, kv_in_dim=8,
                                          rope_k_repeat=True, feat_sizes=(4, 4)).eval()
    q = torch.randn(1, 16, 16)
    k, v = torch.randn(1, frames*4096+pointers, 8), torch.randn(1, frames*4096+pointers, 8)
    sdpa = torch.nn.functional.scaled_dot_product_attention
    calls = []
    def captured(q, k, v, **kwargs):
        calls.append(k.shape[-2])
        return sdpa(q, k, v, **kwargs)
    with torch.inference_mode():
        expected = attention(q, k, v, num_k_exclude_rope=pointers)
        function = sparse_forward()
        env = dict(function.__globals__, F=SimpleNamespace(scaled_dot_product_attention=captured))
        actual_function = FunctionType(function.__code__, env, function.__name__, function.__defaults__)
        attention._sparse_mask = torch.ones(max(0, frames-2)*4096, dtype=torch.bool)
        actual = MethodType(actual_function, attention)(q, k, v, num_k_exclude_rope=pointers)
    assert torch.equal(expected, actual)
    assert calls == [frames*4096+pointers]
    assert torch.nn.functional.scaled_dot_product_attention is sdpa
