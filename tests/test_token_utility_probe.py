"""Engineering tests for an offline probe; never tests for a positive real result."""
import copy
import json

import numpy as np
import pytest
import torch

from evaluation.token_utility_probe import (
    ARMS, MODES, block_indices, field_features, fit_probe, nested_predictions,
    sequence_scores, summarize,
)
from modeling.complementary_memory.integration import _tensor_sha256
from scripts.run_token_utility_probe import intervention_features


def test_block_mapping_matches_original_unfold():
    grid = torch.arange(4096).reshape(64,64)
    author_cells = grid.unfold(0,4,4).unfold(1,4,4).reshape(-1,16)
    ids = [0,1,15,16,17,31,32,48,100,128,130,160,200,240,254,255]
    np.testing.assert_array_equal(block_indices(ids), author_cells[ids].flatten().numpy())


@pytest.mark.parametrize("ids", [[0]*16, list(range(15)), list(range(15))+[256], list(range(15))+[15.5]])
def test_bad_blocks_fail(ids):
    with pytest.raises(ValueError):
        block_indices(ids)


@pytest.mark.parametrize("value", [0.,1.])
def test_empty_or_full_support_is_finite(value):
    vectors = np.arange(256*64).reshape(256,64)
    result = field_features(vectors, np.full(256,value))
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[1:], np.zeros(65))


def fixture():
    generator = torch.Generator().manual_seed(12)
    outputs = {i:{"maskmem_features":torch.randn(1,64,64,64,generator=generator).bfloat16(),
                  "pred_masks":torch.randn(1,1,256,256,generator=generator),
                  "object_score_logits":torch.tensor([[1.]])} for i in range(4)}
    read = {"layers":[0,1,2,3], "memory_frames":[{"frame":i} for i in range(3)],
            "pointer_keys":12,"pointer_input_sha256":"pointer","pointer_position_sha256":"pos",
            "stored_mask_alignment":True,"native_positions_verified":True,"bank_read_unchanged":True}
    native = {"frame":3,"object_score_logit":1.,"read":read}
    arm = copy.deepcopy(native)
    indices = block_indices(list(range(16)))
    removed = torch.zeros(4096,dtype=torch.bool); removed[indices] = True
    vectors = outputs[1]["maskmem_features"][0].float().flatten(1).T
    p = torch.nn.functional.avg_pool2d(outputs[1]["pred_masks"].sigmoid(),4).flatten()
    group = {"memory_frame":1,"block_ids":list(range(16)),"removed":256,
             "mask_sha256":_tensor_sha256(removed),"first_read_usage_mass":.1,
             "mean_soft_foreground":float(p[removed].mean()),
             "mean_vector_norm":float(vectors[removed].norm(dim=-1).mean())}
    arm['read']['selection_layers'] = [{"layer":i,"frame_ids":[0,1,2],"counts":[4096,3840,4096],
        "spatial_keys_before":12288,"spatial_keys_after":12032,"pointer_keys":12,
        "intervention_groups":[copy.deepcopy(group)],
        "selected_indices_sha256":["unused",_tensor_sha256(torch.where(~removed)[0]),"unused"]} for i in range(4)]
    return native,arm,outputs


def test_only_past_vectors_and_shared_native_first_pass_are_inputs():
    native, arm, outputs = fixture()
    original = intervention_features(native,arm,outputs)
    arm["object_score_logit"] = -1000
    arm["scores"] = {"dice":1.}
    arm["present"] = False
    outputs[3]["maskmem_features"].fill_(999.)
    changed = intervention_features(native,arm,outputs)
    for mode in MODES:
        np.testing.assert_array_equal(original[mode],changed[mode])
    assert {m:len(v) for m,v in original.items()} == {"features":803,"geometry":1331,"regional":1331}


@pytest.mark.parametrize("corruption", ["future","pointer","budget","descriptor","block_hash","layer"])
def test_corrupt_intervention_fails(corruption):
    native,arm,outputs = fixture()
    layer = arm["read"]["selection_layers"][0]
    if corruption == "future":
        layer["intervention_groups"][0]["memory_frame"] = 3
    elif corruption == "pointer":
        arm["read"]["pointer_input_sha256"] = "bad"
    elif corruption == "budget":
        layer["counts"][1] = 3839
    elif corruption == "descriptor":
        layer["intervention_groups"][0]["mean_vector_norm"] = 100.
    elif corruption == "block_hash":
        layer["intervention_groups"][0]["mask_sha256"] = "bad"
    else:
        layer["layer"] = 3
    with pytest.raises(RuntimeError):
        intervention_features(native,arm,outputs)


def test_scaler_fits_only_supplied_training_rows():
    x = np.array([[1.,3.],[3.,5.],[20.,30.]])
    model = fit_probe(x[:2],np.array([0.,1.]),np.array(['a','b']),1)
    np.testing.assert_allclose(model['scale'].mean_,[2,4])


def test_held_sequence_labels_cannot_change_its_predictions():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(18,3)); y = x[:,0]*.01
    groups = np.repeat(['a','b','c','d','e','f'],3)
    matrices = {m:x.copy() for m in MODES}
    seen=[]
    first = nested_predictions(matrices,y,groups,alphas=[1,10],inner_folds=2,
                               on_fold=lambda record, *_:seen.append(record))
    y2=y.copy(); y2[groups=='a'] += 100
    second = nested_predictions(matrices,y2,groups,alphas=[1,10],inner_folds=2)
    for m in MODES:
        np.testing.assert_array_equal(first[m][groups=='a'],second[m][groups=='a'])
    assert all(r['test_sequence'] not in r['train_sequences'] for r in seen)


def test_utility_direction_and_tie_handling():
    rows=[{'sequence':'a','frame':1,'arm':a} for a in ARMS]
    y=np.array([.02,-.03,0.])
    best=sequence_scores(rows,y,y)['a']
    assert best['chosen_minus_native_dice'] == .03
    tied=sequence_scores(rows,y,np.zeros(3))['a']
    assert tied['chosen_minus_uniform_arm_dice'] == 0
    assert tied['pair_concordance'] == .5


def test_all_zero_effects_are_valid_negative_result():
    rows=[{'sequence':s,'frame':1,'arm':a} for s in ['a','b'] for a in ARMS]
    y=np.zeros(6)
    result=summarize(rows,y,{m:y.copy() for m in (*MODES,'zero','train_mean')},1)
    assert result['continuation_passed'] is False
    assert result['means']['regional']['pair_concordance'] is None
    json.dumps(result,allow_nan=False)
