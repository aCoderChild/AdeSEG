import json
import sys
from pathlib import Path

import pytest

from scripts import run_extended_boundary_refiner_v1 as runner
from scripts import verify_extended_boundary_video_short_gate_v1 as gate


def argv(tmp_path, *, sequence='seq8', limit=None):
    result=['run']
    for name in ('study','audit','split-preflight','extended-preflight','checkpoint','detector','baseline-reference','seq4-capture','engineering-gate'):
        result += ['--'+name,str(tmp_path/name)]
    result += ['--output',str(tmp_path/'out'),'--sequence',sequence,'--arm','A']
    if limit is not None:result += ['--limit',str(limit)]
    return result


def test_full_video_rejects_missing_both_new_gates_before_inputs(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',argv(tmp_path))
    with pytest.raises(ValueError,match='short gate and its independent audit'):runner.main()
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('seq,limit',[('seq16',4),('seq8',1),('seq17',5),('seq22',100)])
def test_only_two_predeclared_four_frame_engineering_clips(tmp_path,monkeypatch,seq,limit):
    monkeypatch.setattr(sys,'argv',argv(tmp_path,sequence=seq,limit=limit))
    with pytest.raises(ValueError,match='Only seq8/seq17'):runner.main()
    assert not (tmp_path/'out').exists()


def test_extended_gate_rejects_failed_or_incomplete_receipts(tmp_path):
    p=tmp_path/'gate.json';p.write_text(json.dumps({'passed':False}))
    with pytest.raises(ValueError,match='incomplete'):gate.require_gate(p,extended_preflight=tmp_path,engineering_gate=tmp_path)
    with pytest.raises(ValueError,match='Two full short-run'):gate.verify_mapping({'seq8':{}},tmp_path,tmp_path)


def test_short_gate_rejects_aliases_before_reading_artifacts(tmp_path):
    mapping={s:{a:str(tmp_path/'same') for a in gate.ARMS} for s in gate.SEQS}
    with pytest.raises(ValueError,match='not distinct'):gate.verify_mapping(mapping,tmp_path,tmp_path)


def test_independent_audit_must_bind_correct_runner_gate_and_preflight(tmp_path):
    from scripts.memory_token_io import digest
    pre=tmp_path/'pre';pre.mkdir();(pre/'manifest.json').write_text('{}')
    ext=tmp_path/'extension.json';ext.write_text('{}')
    original=tmp_path/'original.json';original.write_text('{}')
    audit=tmp_path/'audit.json'
    data={'passed':True,'extension_gate_sha256':digest(ext),'extended_preflight_sha256':digest(pre/'manifest.json'),
          'engineering_gate_sha256':digest(original),'auditor_sha256':digest(runner.ROOT/'scripts/audit_extended_boundary_video_short_v1.py'),
          'runner_sha256':digest(Path(runner.__file__)),'runs_verified':10}
    audit.write_text(json.dumps(data))
    assert runner.validate_independent_short_audit(audit,ext,pre,original)['passed']
    data['runs_verified']=9;audit.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='audit differs'):runner.validate_independent_short_audit(audit,ext,pre,original)
    data['runs_verified']=10;data['extension_gate_sha256']='bad';audit.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='audit differs'):runner.validate_independent_short_audit(audit,ext,pre,original)


def test_gate_demonstrates_routes_and_rejects_prompt_or_conditioning_tamper(tmp_path,monkeypatch):
    pre=tmp_path/'pre';pre.mkdir(); original=tmp_path/'original.json'; original.write_text('{}')
    items={s:{'first_detection_index':0,'prompt_metadata_sha256':'p','prompt_mask_sha256':'pm',
              'view':{'kind':'full_source_frame'},'frames':[{'frame':0,'source_frame_index':0,'image_sha256':'i'}]}
           for s in gate.SEQS}
    (pre/'manifest.json').write_text(json.dumps({'sequences':items,'checkpoint_sha256':'c','detector_sha256':'d'}))
    mapping={s:{a:str(tmp_path/s/a) for a in gate.ARMS} for s in gate.SEQS}
    entries={}
    common={'prompt_frame':0,'prompt_sha256':'p','prompt_mask_sha256':'pm','checkpoint_sha256':'c',
            'detector_sha256':'d','candidate_checkpoint_sha256':'x','control_checkpoint_sha256':'y',
            'view':{'kind':'full_source_frame'},'input_images':[{'frame':0,'source_frame_index':0,'sha256':'i'}]}
    initial={'frame':0,'mask_sha256':'mask','pred_masks_sha256':'pred0','maskmem_features_sha256':'mem0',
             'obj_ptr_sha256':'ptr0','object_score_logits_sha256':'score0','output_logits_sha256':'out0'}
    for seq in gate.SEQS:
        for arm in gate.ARMS:
            output='out1' if arm in ('B','D') else 'out0'; mem='mem1' if arm in ('C','D') else 'mem0'
            first={'frame':1,'mask_sha256':'mask','pred_masks_sha256':'pred0','maskmem_features_sha256':mem,
                   'obj_ptr_sha256':'ptr0','object_score_logits_sha256':'score0','output_logits_sha256':output}
            rec={**common,'masks':[{'frame':0,'sha256':'a'}], 'post_prompt_state':[dict(initial),first]}
            entries[str(Path(mapping[seq][arm]).resolve())]={'folder':mapping[seq][arm],'rec':rec,'runtime':{'passed':True}}
    monkeypatch.setattr(gate,'verify_short_run',lambda folder,seq,arm,**kwargs:entries[str(Path(folder).resolve())])
    saved_read=gate.read
    monkeypatch.setattr(gate,'read',lambda path:saved_read(path) if Path(path)==pre/'manifest.json' else [])
    gate.verify_mapping(mapping,pre,original)
    victim=entries[str(Path(mapping['seq17']['C']).resolve())]['rec']['post_prompt_state'][0]
    victim['obj_ptr_sha256']='tampered'
    with pytest.raises(ValueError,match='conditioning state'):gate.verify_mapping(mapping,pre,original)
    victim['obj_ptr_sha256']='ptr0'
    entries[str(Path(mapping['seq8']['A']).resolve())]['rec']['prompt_sha256']='bad'
    with pytest.raises(ValueError,match='prompt or images'):gate.verify_mapping(mapping,pre,original)
