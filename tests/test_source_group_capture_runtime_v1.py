import numpy as np
import pytest
from scripts.freeze_source_group_runtime_v1 import verify_fork_state
from scripts.run_source_group_capture_v1 import ensure_storage_headroom,verify_conditioning_mask


def test_fork_must_match_revision_and_exact_dirty_paths():
    patches={'a.py':'x','b.py':'y'}
    verify_fork_state('rev',[' M a.py',' M b.py'],'rev',patches)
    with pytest.raises(ValueError):verify_fork_state('other',[' M a.py',' M b.py'],'rev',patches)
    with pytest.raises(ValueError):verify_fork_state('rev',[' M a.py',' M b.py','?? extra'],'rev',patches)


def test_storage_and_conditioning_fail_closed():
    ensure_storage_headroom(22,22)
    with pytest.raises(ValueError,match='Free space'):ensure_storage_headroom(21,22)
    a=np.array([[False,True]])
    verify_conditioning_mask(a,a.copy())
    with pytest.raises(ValueError,match='conditioning'):verify_conditioning_mask(a,~a)
    with pytest.raises(ValueError,match='conditioning'):verify_conditioning_mask(a,np.zeros((2,1),bool))


def test_source_freeze_rejects_changed_dependency():
    from scripts.freeze_source_group_runtime_v1 import validate_source_freeze
    freeze={'passed':True,'identity':{'source_sha256':{'helper.py':'a'}}}
    done={'passed':True,'source_freeze_sha256':'f'}
    validate_source_freeze(freeze,done,'f',freeze['identity'])
    with pytest.raises(ValueError):validate_source_freeze(freeze,done,'f',{'source_sha256':{'helper.py':'changed'}})
    with pytest.raises(ValueError):validate_source_freeze(freeze,done,'changed',freeze['identity'])


def test_trajectory_and_observation_mutations_fail():
    import copy
    from scripts.audit_source_group_capture_v1 import verify_trajectory,verify_observation_coverage,verify_sample_file
    spec={'frames':5,'prompt_frame':0,'targets':[1,2,3,4]}
    ref={'sequence':'seq3','limit':5,'full_sequence':False,'prompt_frame':0,'targets':[1,2,3,4],
         'rows':[{'frame':i,'mask_sha256':'x'} for i in range(5)],'positions':[1],'selection':[1],'anchor_events':[1],'token_window_events':[1]}
    cap=copy.deepcopy(ref);assert verify_trajectory(ref,cap,spec)==(5,[1,2,3,4])
    cap['positions']=[2]
    with pytest.raises(ValueError):verify_trajectory(ref,cap,spec)
    cap=copy.deepcopy(ref);cap['rows'][1]['mask_sha256']='changed'
    with pytest.raises(ValueError):verify_trajectory(ref,cap,spec)
    verify_observation_coverage([{'frame':i} for i in (1,2,3,4)],[1,2,3,4])
    with pytest.raises(ValueError):verify_observation_coverage([{'frame':i} for i in (1,2,3,3)],[1,2,3,4])
    with pytest.raises(ValueError):verify_observation_coverage([{'frame':i} for i in (1,2,3)],[1,2,3,4])


def test_observation_file_hash_and_location(tmp_path):
    from scripts.audit_source_group_capture_v1 import verify_sample_file
    from scripts.memory_token_io import digest
    root=tmp_path/'run';root.mkdir();sample=root/'feature.pt';sample.write_bytes(b'feature');sha=digest(sample)
    verify_sample_file(sample,root,sha)
    sample.write_bytes(b'altered')
    with pytest.raises(ValueError):verify_sample_file(sample,root,sha)
    outside=tmp_path/'outside';outside.write_bytes(b'feature')
    with pytest.raises(ValueError):verify_sample_file(outside,root,sha)


def test_python_audit_guard_blocks_mask_opens(tmp_path):
    import json,subprocess,sys
    img=tmp_path/'frame.jpg';img.write_bytes(b'image')
    mask=tmp_path/'mask.jpg';mask.write_bytes(b'mask')
    manifest=tmp_path/'data.jsonl';manifest.write_text(json.dumps({'mask':str(mask),'image':str(img)})+'\n')
    code='''from pathlib import Path\nfrom scripts.source_group_label_guard_v1 import install_mask_open_guard\nimport sys\ng=install_mask_open_guard(Path(sys.argv[1]))\nassert Path(sys.argv[2]).read_bytes()==b"image"\ntry: Path(sys.argv[3]).read_bytes()\nexcept PermissionError: pass\nelse: raise AssertionError("mask opened")\nassert g["blocked_paths"]==1 and g["open_attempts"]==1\n'''
    subprocess.run([sys.executable,'-c',code,str(manifest),str(img),str(mask)],check=True)


def test_full_capture_gate_binds_receipts_trajectory_and_observation_files(tmp_path):
    import copy,json
    from scripts.memory_token_io import digest
    from scripts.run_source_group_capture_v1 import verify_short_gate
    refdir=tmp_path/'ref';capdir=tmp_path/'cap';repdir=tmp_path/'rep';refdir.mkdir();capdir.mkdir();repdir.mkdir()
    identity={'preflight_sha256':'p','source_freeze_sha256':'f','runner_sha256':'r','runtime_environment':{'device':'mps'}}
    rows=[]
    for i in range(5):
        for root in (refdir,capdir,repdir):(root/f'{i:06d}.png').write_bytes(str(i).encode())
        rows.append({'frame':i,'mask_sha256':digest(refdir/f'{i:06d}.png')})
    for root in (refdir,capdir,repdir):(root/'conditioning_mask.png').write_bytes(b'condition')
    cond=digest(refdir/'conditioning_mask.png');observations=[]
    for i in range(1,5):
        f=capdir/f'feature{i}';s=capdir/f'snapshot{i}';f.write_bytes(b'f');s.write_bytes(b's')
        observations.append({'frame':i,'feature_file':str(f),'feature_sha256':digest(f),'snapshot_file':str(s),'snapshot_sha256':digest(s)})
    repeat_observations=[]
    for i in range(1,5):
        f=repdir/f'feature{i}';s=repdir/f'snapshot{i}';f.write_bytes(b'f');s.write_bytes(b's')
        repeat_observations.append({'frame':i,'feature_file':str(f),'feature_sha256':digest(f),'snapshot_file':str(s),'snapshot_sha256':digest(s)})
    common={'sequence':'seq3','identity':identity,'limit':5,'targets':[1,2,3,4],'rows':rows,'conditioning_sha256':cond,
            'positions':[],'selection':[],'anchor_events':[],'token_window_events':[]}
    ref={**common,'mode':'reference'};cap={**copy.deepcopy(common),'mode':'capture','observations':observations}
    repeat={**copy.deepcopy(common),'mode':'capture','observations':repeat_observations}
    rp=refdir/'receipt.json';cp=capdir/'receipt.json';xp=repdir/'receipt.json';rp.write_text(json.dumps(ref));cp.write_text(json.dumps(cap));xp.write_text(json.dumps(repeat))
    gate={'passed':True,'sequence':'seq3','limit':5,'full_sequence':False,'frames':5,'targets':[1,2,3,4],
          'auditor_sha256':'a','identity':identity,'label_guard':{'open_attempts':0},'conditioning_sha256':cond,
          'reference_receipt_path':str(rp),'reference_receipt_sha256':digest(rp),'capture_receipt_path':str(cp),'capture_receipt_sha256':digest(cp),
          'repeat_capture_receipt_path':str(xp),'repeat_capture_receipt_sha256':digest(xp),'observed_tensor_repeat_equal':True}
    verify_short_gate(gate,'p','f','r','a',{'device':'mps'})
    with pytest.raises(ValueError):verify_short_gate(gate,'p','f','changed','a',{'device':'mps'})
    (capdir/'feature1').write_bytes(b'changed')
    with pytest.raises(ValueError):verify_short_gate(gate,'p','f','r','a',{'device':'mps'})


def test_short_mode_cannot_bypass_full_gate():
    from scripts.run_source_group_capture_v1 import verify_limit
    assert verify_limit('seq3',5,15)==5 and verify_limit('seq3',0,15)==15
    for seq,limit in [('seq3',15),('seq3',14),('seq3',-1),('seq4',5)]:
        with pytest.raises(ValueError):verify_limit(seq,limit,15)


def test_nested_tensor_replay_is_exact():
    import torch
    from scripts.audit_source_group_capture_v1 import exact_nested
    a={'x':(torch.tensor([1.,2.]),{'y':3})};b={'x':(torch.tensor([1.,2.]),{'y':3})}
    assert exact_nested(a,b)
    b['x'][0][1]=3
    assert not exact_nested(a,b)
