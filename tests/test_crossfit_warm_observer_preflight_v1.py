"""CPU tests for archived automatic warm-prompt preflight."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts import preflight_crossfit_warm_observer_v1 as warm
from scripts.memory_token_io import digest


def fixture_prompt(monkeypatch,tmp_path,sequence,*,bad_points=False):
    prompt={'first_detection_index':0,'box_viewport_xyxy' if sequence=='seq23' else 'box':[10,20,90,80],
            'points_viewport' if sequence=='seq23' else 'points':[[20,30],[30,40]] if not bad_points else [[-1,30],[30,40]]}
    path=tmp_path/'prompt.json';path.write_text(json.dumps(prompt));mask=tmp_path/'prompt.png';mask.write_bytes(b'archived')
    prov={'prompt_file':str(mask),'prompt_metadata_sha256':digest(path),'prompt_mask_sha256':digest(mask),'roi_xyxy':[0,0,100,100]}
    info={'prompt_provenance':prov,'view':{'kind':'full_source_frame'},'baseline_completion_sha256':'frozen'}
    monkeypatch.setattr(warm,'sequence_inputs',lambda plan,seq:([1,2],[],0,[1],np.ones((100,100),bool),None,None,info))
    return path


def test_viewport_coordinates_are_selected_without_a_detector(monkeypatch,tmp_path):
    fixture_prompt(monkeypatch,tmp_path,'seq23')
    result=warm.checked_prompt({},'seq23')
    assert result['box_field']=='box_viewport_xyxy' and result['points_field']=='points_viewport'
    assert result['target_frames']==[1]


def test_out_of_bounds_prompt_fails_closed(monkeypatch,tmp_path):
    fixture_prompt(monkeypatch,tmp_path,'seq6',bad_points=True)
    with pytest.raises(ValueError,match='outside the view'):warm.checked_prompt({},'seq6')


def test_changed_prompt_metadata_fails_closed(monkeypatch,tmp_path):
    path=fixture_prompt(monkeypatch,tmp_path,'seq6');path.write_text('{}')
    with pytest.raises(ValueError,match='metadata changed'):warm.checked_prompt({},'seq6')


def test_v3_seq6_shared_identity_and_overlap_are_required():
    from scripts.preflight_crossfit_warm_observer_v3 import verify_seq6_identity
    common={'runtime_environment':{},'source_freeze_sha256':'freeze','runner_sha256':'old-runner',
            'preflight_sha256':'old-preflight','model_state_sha256':'model','checkpoint_sha256':'checkpoint',
            'anchor_sha256':'anchor'}
    seq6={'identity':{'runner_sha256':'seq6-runner','preflight_sha256':'seq6-preflight',
                      'model_state_sha256':'model','checkpoint_sha256':'checkpoint','anchor_sha256':'anchor'},
          'targets':[24,90]}
    audit={'auditor_sha256':'auditor'}
    verify_seq6_identity(common,seq6,audit,[24,90], 'seq6-runner','auditor')
    seq6['identity']['anchor_sha256']='changed'
    with pytest.raises(ValueError,match='shared identity'):verify_seq6_identity(common,seq6,audit,[24,90],'seq6-runner','auditor')
    seq6['identity']['anchor_sha256']='anchor'
    with pytest.raises(ValueError,match='overlap'):verify_seq6_identity(common,seq6,audit,[24],'seq6-runner','auditor')
    with pytest.raises(ValueError,match='auditor'):verify_seq6_identity(common,seq6,audit,[24,90],'seq6-runner','bad')
