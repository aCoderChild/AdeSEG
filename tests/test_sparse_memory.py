import ast
import math
from functools import partial
from pathlib import Path
from types import MethodType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
from modeling.sparse_memory import author_sparse_parts, sparse_forward, temporal_slot, retained_cells, native_positions


@pytest.fixture
def native_class():
    # Load unchanged author classes without polluting other authors' `sam2`
    # package namespace during pytest collection.
    path = ROOT / "external/sam2/sam2/modeling/sam/transformer.py"
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name in ("Attention", "RoPEAttention")]
    pos = native_positions()
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, Tensor=torch.Tensor,
                     math=math, partial=partial, compute_axial_cis=pos.compute_axial_cis,
                     apply_rotary_enc=pos.apply_rotary_enc)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["RoPEAttention"]


@pytest.mark.parametrize("frames,pointers", [(1, 0), (1, 4), (2, 4), (3, 4), (7, 12)])
def test_real_projection_sdpa_all_token_path_is_bit_identical(frames, pointers, native_class):
    torch.manual_seed(19)
    model = native_class(embedding_dim=16, num_heads=1, kv_in_dim=8,
                          rope_k_repeat=True, feat_sizes=(4, 4)).eval()
    q, k, v = torch.randn(1, 16, 16), torch.randn(1, frames*4096+pointers, 8), torch.randn(1, frames*4096+pointers, 8)
    with torch.inference_mode():
        native = model(q, k, v, num_k_exclude_rope=pointers)
        model._sparse_mask = torch.ones(max(0, (frames-2)*4096), dtype=torch.bool)
        adapted = MethodType(sparse_forward(), model)(q, k, v, num_k_exclude_rope=pointers)
    assert torch.equal(native, adapted)


def test_author_pruned_rope_keeps_original_cell_positions():
    gather, rotary = author_sparse_parts()
    torch.manual_seed(20)
    q, k, v = torch.randn(1, 1, 16, 8), torch.randn(1, 1, 12288, 8), torch.randn(1, 12292, 8)
    mask = torch.arange(4096) % 3 == 0
    indices = torch.cat([torch.arange(4096), torch.arange(4096, 8192)[mask], torch.arange(8192, 12288)])
    cis = native_positions().compute_axial_cis(8, 4, 4)
    native_q, native_k = native_positions().apply_rotary_enc(q, k, cis, repeat_freqs_k=True)
    actual_q, actual_k = rotary(q, k[:, :, indices], cis, 12288, True, mask, 4096, 8192)
    assert torch.equal(native_q, actual_q)
    assert torch.equal(native_k[:, :, indices], actual_k)
    raw = torch.randn_like(v)
    selected_k, selected_v = gather(raw, v, mask, 4096, 8192)
    assert torch.equal(selected_k[:, -4:], raw[:, -4:])
    assert torch.equal(selected_v[:, -4:], v[:, -4:])


@pytest.mark.parametrize("frame,current,prompt,slot", [(0, 1, 0, 6), (9, 10, 0, 0), (4, 10, 0, 5), (1, 30, 0, 5)])
def test_fixed_temporal_saturation(frame, current, prompt, slot):
    assert temporal_slot(frame, current, prompt) == slot


def test_empty_selection_and_future_rejection():
    assert retained_cells(torch.zeros(64, 64, dtype=torch.long)).sum() == 0
    with pytest.raises(ValueError, match="future"):
        temporal_slot(3, 3, 0)
