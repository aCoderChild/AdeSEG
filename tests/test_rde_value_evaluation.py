import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from datasets.video import ManifestFrame
from evaluation.segmentation import calculate_scores
from evaluation.davis2017.metrics import db_eval_boundary
from scripts import rde_value_evaluation as evaluation
from scripts.memory_token_io import digest,parameter_digest
from scripts.mavos_value_utils import atomic_json
from scripts.run_token_conditioned_key_adapter import _atomic_jsonl,_summary_from_rows


def fixture(tmp_path,monkeypatch):
    frames=[]
    for i in range(3):
        truth=np.zeros((8,8),dtype=np.uint8);truth[2:5,2:5]=1
        image=tmp_path/f'image{i}.jpg';mask=tmp_path/f'gt{i}.png'
        Image.fromarray(np.zeros((8,8,3),dtype=np.uint8)).save(image);Image.fromarray(truth).save(mask)
        frames.append(ManifestFrame('fixture',i,image,mask))
    model=torch.nn.Linear(1,1).requires_grad_(False).eval();calls=[]
    def fake(model,frames,heads,name,out,setup):
        calls.append(name);directory=out/'evaluation'/name;(directory/'masks').mkdir(parents=True)
        rows=[]
        for i in [1,2]:
            truth=np.asarray(Image.open(frames[i].mask_path))==1
            path=directory/'masks'/f'{i:06d}.png';Image.fromarray(truth.astype(np.uint8)*255).save(path)
            rows.append({'frame':i,'present':True,'scores':{**calculate_scores(truth,truth),'boundary_f':float(db_eval_boundary(truth,truth))},
                'absent_fp':None,'prediction_sha256':'unit_fixture','mask_sha256':digest(path)})
        summary=_summary_from_rows(name,0,rows,1.,'native',parameter_digest(model),setup,[])
        _atomic_jsonl(directory/'frames.jsonl',rows);atomic_json(directory/'summary.json',summary)
        return summary
    monkeypatch.setattr(evaluation,'author_evaluate',fake)
    return model,frames,calls


def test_cache_is_rescored_and_wrong_setup_or_mask_rejected(tmp_path,monkeypatch):
    model,frames,calls=fixture(tmp_path,monkeypatch);out=tmp_path/'result'
    a=evaluation.evaluate_condition(model,frames,None,'fixture',out,'setup')
    assert a['present_dice']==1 and len(calls)==1
    assert evaluation.evaluate_condition(model,frames,None,'fixture',out,'setup')==a
    assert len(calls)==1
    with pytest.raises(RuntimeError,match='identity'):evaluation.evaluate_condition(model,frames,None,'fixture',out,'other')
    path=out/'evaluation/fixture/masks/000001.png';Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(path)
    with pytest.raises(RuntimeError,match='mask integrity'):evaluation.evaluate_condition(model,frames,None,'fixture',out,'setup')


def test_partial_evaluation_is_preserved_not_silently_reused(tmp_path,monkeypatch):
    model,frames,calls=fixture(tmp_path,monkeypatch);out=tmp_path/'result'
    (out/'staging/fixture').mkdir(parents=True)
    with pytest.raises(RuntimeError,match='Uncommitted'):evaluation.evaluate_condition(model,frames,None,'fixture',out,'setup')
    assert not calls


def test_coherent_file_checksums_do_not_hide_wrong_saved_metrics(tmp_path,monkeypatch):
    model,frames,calls=fixture(tmp_path,monkeypatch);out=tmp_path/'result'
    evaluation.evaluate_condition(model,frames,None,'fixture',out,'setup')
    directory=out/'evaluation/fixture'
    rows=[json.loads(x) for x in (directory/'frames.jsonl').read_text().splitlines()]
    rows[0]['scores']['dice']=.5
    _atomic_jsonl(directory/'frames.jsonl',rows)
    receipt=json.loads((directory/'completion.json').read_text());receipt['frames_sha256']=digest(directory/'frames.jsonl')
    atomic_json(directory/'completion.json',receipt)
    with pytest.raises(RuntimeError,match='metrics do not match saved masks'):
        evaluation.evaluate_condition(model,frames,None,'fixture',out,'setup')


def test_training_cache_cannot_be_loaded_for_inference(tmp_path):
    path=tmp_path/'unsafe.pt';torch.save({'adapter':{},'identity_sha256':'id','arm':'arm','frames_seen':1,'teacher_cache':{}},path)
    with pytest.raises(RuntimeError,match='unexpected'):evaluation.load_adapter(path,'id','arm')


def test_pointer_layer_and_count_corruption_rejected():
    row={'frame':2,'memory_read':{'frame':2,'object_id':1,'layers':[0,1,2,3],'stored_mask_alignment':True,
        'native_positions_verified':True,'spatial_keys':8192,'pointer_keys':8},
        'value_calls':[{'layer':i,'pointer_tail_bit_identical':True,'spatial_tokens':8192,'pointer_tokens':8} for i in range(4)]}
    evaluation.verify_value_rows([row],True)
    for change in ['layer','pointer','count','owner']:
        r=copy.deepcopy(row)
        if change=='layer':r['value_calls'].pop()
        elif change=='pointer':r['value_calls'][0]['pointer_tail_bit_identical']=False
        elif change=='count':r['value_calls'][0]['spatial_tokens']=4096
        else:r['memory_read']['object_id']=2
        with pytest.raises(RuntimeError):evaluation.verify_value_rows([r],True)
