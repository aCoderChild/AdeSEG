import json

import pytest
import torch
import torch.nn.functional as F

from scripts.measure_author_regions import (
    AuthorRegionObserver,
    selected_memory,
    verify_predictions,
)
from test_region_bias import setup


def test_candidate_selection_uses_author_branch_and_rejects_missing_identity():
    features = torch.randn(1, 64, 64, 64, 3)
    masks = torch.randn(1, 1, 32, 32, 3)
    output = {"maskmem_features": features, "pred_masks": masks}
    actual, label_mask, branch = selected_memory(output, 5, {5: 2})
    assert torch.equal(actual, features[..., 2])
    assert torch.equal(label_mask, masks[..., 2]) and branch == 2
    with pytest.raises(RuntimeError, match="branch identity"):
        selected_memory(output, 5, 0)


@pytest.mark.parametrize("candidate", [None, 1])
def test_read_observer_preserves_native_outputs_and_restores_hooks(candidate):
    model, state, logits, q, v = setup()
    outputs = state["output_dict_per_obj"][0]
    outputs["cond_frame_outputs"][0]["pred_masks"] = logits
    selection = 0
    if candidate is not None:
        first = outputs["cond_frame_outputs"][0]
        outputs["non_cond_frame_outputs"][1] = {
            "maskmem_features": torch.stack(
                [first["maskmem_features"] + 1, first["maskmem_features"] + 2], -1
            ),
            "pred_masks": torch.stack([-logits, logits], -1),
        }
        chosen = outputs["non_cond_frame_outputs"][1]["maskmem_features"][..., 1]
        v = torch.cat([v[:, :-4], chosen.flatten(2).transpose(1, 2), v[:, -4:]], 1)
        selection = {0: 0, 1: 1}

    def prepare(frame_idx, is_init_cond_frame, output_dict, mem_pick_index=0):
        result = None
        for block in model.memory_attention.layers:
            result = block.cross_attn_image(q=q, k=v, v=v, num_k_exclude_rope=4)
        return result

    model._prepare_memory_conditioned_features = prepare
    model._get_orig_video_res_output = lambda state, masks: (masks, masks)
    expected = prepare(2, False, outputs, selection)
    original_sdpa = F.scaled_dot_product_attention
    with AuthorRegionObserver(model, "seq2", 0, state) as observer:
        actual = model._prepare_memory_conditioned_features(
            2, False, outputs, selection
        )
        assert torch.equal(expected, actual)
        assert len(observer.measurements) == 4
        assert [r["layer"] for r in observer.measurements] == [0, 1, 2, 3]
        if candidate is not None:
            assert observer.measurements[0]["memory_units"][1]["candidate"] == 1
        assert all(r["pointer_tokens"] == 4 for r in observer.measurements)
    assert F.scaled_dot_product_attention is original_sdpa
    assert model._prepare_memory_conditioned_features is prepare


def test_prediction_parity_requires_exact_hash_source_and_presence(tmp_path):
    reference = tmp_path / "frames.jsonl"
    rows = [{"source_frame_index": 12, "prediction_sha256": "abc", "present": True}]
    reference.write_text(json.dumps(rows[0]) + "\n")
    verify_predictions(rows, reference)
    with pytest.raises(RuntimeError, match="predictions"):
        verify_predictions([{**rows[0], "prediction_sha256": "changed"}], reference)
