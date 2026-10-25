"""Pinned author eligibility reuse, branch provenance and spatial budgets."""

from types import SimpleNamespace

import pytest
import torch

from modeling.author_sparse_memory import (
    AuthorSparseMemory, adapted_prepare, author_nodes, branch_memory,
    eligibility_function,
)


def stored(frame, branches=False):
    features = torch.full((1, 64, 1, 1), float(frame)).expand(1, 64, 64, 64)
    masks = torch.full((1, 1, 256, 256), -1.0)
    masks[:, :, 110:130, 110:130] = 1
    row = {"maskmem_features": features, "pred_masks": masks,
           "best_iou_score": torch.tensor([0.9]), "object_score_logits": torch.tensor([[1.0]]),
           "ious": torch.tensor([[0.9]])}
    if branches:
        for key in ("maskmem_features", "pred_masks", "object_score_logits", "ious"):
            row[key] = torch.stack([row[key], row[key]], -1)
    return row


def controller(method, identity=False):
    obj = AuthorSparseMemory.__new__(AuthorSparseMemory)
    obj.model = SimpleNamespace(max_obj_ptrs_in_encoder=16,
        memory_bank_iou_threshold=0.5, memory_bank_obj_score_threshold=0.0,
        memory_bank_kf_score_threshold=0.0)
    obj.method, obj.prompt, obj.identity = method, 0, identity
    obj.eligible = eligibility_function(method, {})
    obj.label_cache = {}
    return obj


@pytest.mark.parametrize("method", ["samurai", "sam2long"])
def test_pinned_author_functions_compile_without_editing_repositories(method):
    path, _, _, _, _ = author_nodes(method)
    assert path.exists()
    assert callable(adapted_prepare(method, {}))


@pytest.mark.parametrize("method", ["samurai", "sam2long"])
def test_author_eligibility_and_pointers_remain_separate_from_extended_spatial_plan(method):
    obj = controller(method)
    outputs = {"cond_frame_outputs": {0: stored(0)},
               "non_cond_frame_outputs": {i: stored(i, method == "sam2long") for i in range(1, 40)}}
    selection = {i: 1 for i in range(1, 40)} if method == "sam2long" else 0
    ctx = dict(frame_idx=40, output_dict=outputs, num_frames=50)
    if method == "sam2long":
        ctx.update(mem_pick_index=selection, start_frame_idx=0, iou_thre=0.3)
    native = obj.eligible(obj.model, **ctx, limit=15)
    ctx['valid_indices'] = native.copy()
    ctx['unselected_cond_outputs'] = {}
    plan = obj.plan(ctx)
    assert ctx['valid_indices'] == native
    assert len(plan) == 31 and len(obj.units) == 32
    assert plan[-1] == (6, 39)
    assert plan[0][0] == 1
    assert [u['branch'] for u in obj.units[1:]] == [1 if method == "sam2long" else None]*31
    assert all(f < 40 for _, f in plan)


@pytest.mark.parametrize("method", ["samurai", "sam2long"])
def test_identity_plan_has_exact_native_six_slots(method):
    obj = controller(method, True)
    ctx = dict(frame_idx=10, num_frames=20, output_dict={"cond_frame_outputs": {0: stored(0)},
        "non_cond_frame_outputs": {i: stored(i, method == "sam2long") for i in range(1, 10)}})
    if method == "sam2long":
        ctx.update(mem_pick_index={i: 0 for i in range(1, 10)}, start_frame_idx=0, iou_thre=0.3)
    ctx['valid_indices'] = obj.eligible(obj.model, **ctx, limit=15)
    ctx['unselected_cond_outputs'] = {}
    assert obj.plan(ctx) == list(zip(range(1, 7), range(4, 10)))


def test_ineligible_samurai_frames_not_admitted_by_expansion():
    obj = controller('samurai')
    outputs = {"cond_frame_outputs": {0: stored(0)}, "non_cond_frame_outputs": {i: stored(i) for i in range(1, 20)}}
    outputs['non_cond_frame_outputs'][10]['best_iou_score'] = torch.tensor([0.1])
    outputs['non_cond_frame_outputs'][11]['object_score_logits'] = torch.tensor([[-1.0]])
    ids = obj.eligible(obj.model, frame_idx=20, output_dict=outputs, num_frames=30, limit=31)
    assert 10 not in ids and 11 not in ids and ids[-1] == 19


def test_sam2long_branch_is_author_selected_not_rechosen():
    row = stored(3, True)
    row['maskmem_features'][..., 1] = 42
    feats, _, branch = branch_memory(row, 3, {3: 1})
    assert branch == 1 and (feats == 42).all()
    with pytest.raises(RuntimeError, match='branch'):
        branch_memory(row, 3, 0)


def test_sparse_budget_stops_at_first_overflow():
    obj = controller('samurai')
    outputs = {"cond_frame_outputs": {0: stored(0)}, "non_cond_frame_outputs": {i: stored(i) for i in range(1, 12)}}
    for row in outputs['non_cond_frame_outputs'].values():
        row['pred_masks'].fill_(1)
    ctx = dict(frame_idx=12, num_frames=20, output_dict=outputs)
    ctx['valid_indices'] = obj.eligible(obj.model, **ctx, limit=15)
    ctx['unselected_cond_outputs'] = {}
    plan = obj.plan(ctx)
    assert len(plan) == 6 and len(obj.units) == 7
    assert [f for _, f in plan] == list(range(6, 12))


def test_first_samurai_read_does_not_duplicate_conditioning_prompt():
    obj = controller('samurai')
    ctx = dict(frame_idx=1, num_frames=20, output_dict={"cond_frame_outputs": {0: stored(0)}, "non_cond_frame_outputs": {}})
    ctx['valid_indices'] = obj.eligible(obj.model, **ctx, limit=15)
    ctx['unselected_cond_outputs'] = {}
    assert ctx['valid_indices'] == [0]
    assert obj.plan(ctx) == []
    assert len(obj.units) == 1
