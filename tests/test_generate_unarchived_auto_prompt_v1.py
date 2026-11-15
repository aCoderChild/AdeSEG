import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

import scripts.generate_unarchived_auto_prompt_v1 as module


def test_all_static_project_imports_are_source_bound():
    roots={'scripts','modeling','datasets','evaluation','training'}
    seen=set();pending=['scripts/generate_unarchived_auto_prompt_v1.py']
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse((module.ROOT/path).read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom) and node.level==0 and node.module:names=[node.module]
            for name in names:
                if name.split('.')[0] not in roots:continue
                candidate=name.replace('.','/')+'.py'
                if (module.ROOT/candidate).is_file():pending.append(candidate)
    assert seen.issubset(set(module.FILES))


def make_frames(tmp_path, count=3):
    frames=[]
    for i in range(count):
        path=tmp_path/f'input{i}.jpg'
        Image.fromarray(np.full((4,6,3),i*20,dtype=np.uint8)).save(path)
        frames.append(SimpleNamespace(frame_index=i,image_path=path))
    return frames


def test_scan_stops_at_first_valid_box_and_never_uses_future(tmp_path):
    frames=make_frames(tmp_path);inventory,size=module.image_inventory(frames)
    calls=[]
    def detect(_detector,_image):
        calls.append(len(calls))
        return (None,None) if len(calls)==1 else ([0.,0.,4.,3.],.7)
    frame,box,confidence,attempts=module.scan_first(frames,inventory,object(),detect)
    assert frame==1 and box==[0.,0.,4.,3.] and confidence==.7
    assert calls==[0,1] and len(attempts)==2 and size==(6,4)


def test_no_detection_materializes_four_exact_empty_video_outputs(tmp_path):
    frames=make_frames(tmp_path);inventory,size=module.image_inventory(frames)
    frame,box,confidence,attempts=module.scan_first(frames,inventory,object(),lambda *_:(None,None))
    assert frame is box is confidence is None and len(attempts)==len(frames)
    out=tmp_path/'output';out.mkdir()
    arms=module.save_no_detection_outputs(out,inventory,size)
    assert set(arms)=={'A','B','C','D'}
    for rows in arms.values():
        assert [r['frame'] for r in rows]==[0,1,2]
    for arm in arms:
        for row in arms[arm]:
            f=out/arm/f"{row['frame']:06d}.png"
            assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
            with Image.open(f) as image:
                assert image.size==size and not np.asarray(image).any()
    assert len({row['sha256'] for rows in arms.values() for row in rows})==1


@pytest.mark.parametrize('detection', [([0.,0.,9.,3.],.5),([2.,0.,2.,3.],.5),([0.,0.,4.,3.],float('nan')), (None,.5)])
def test_scan_rejects_bad_detection_without_gt(tmp_path,detection):
    frames=make_frames(tmp_path);inventory,_=module.image_inventory(frames)
    with pytest.raises(ValueError):module.scan_first(frames,inventory,object(),lambda *_:detection)


class FakeModel:
    def __init__(self, mask):
        self.mask=mask
        self.box_calls=0
        self.point_calls=0
        self.reset=False
    def state_dict(self):return {'weight':torch.tensor([1.0]),'buffer':torch.tensor([2.0])}
    def init_state(self,video_path,**kwargs):return {'num_frames':len(list(Path(video_path).glob('*.jpg')))}
    def add_new_points_or_box(self,state,frame_idx,obj_id,**kwargs):
        if 'box' in kwargs:self.box_calls+=1
        else:self.point_calls+=1
        return frame_idx,[1],self.mask.clone()
    def _get_image_feature(self,state,frame_idx,batch_size):
        return None,{'backbone_fpn':[torch.ones((1,32,2,2))]},None,None,None
    def reset_state(self,state):self.reset=True


@pytest.mark.parametrize('value,status',[(1.,'detected_nonempty'),(-1.,'detected_empty_box_only')])
def test_degenerate_points_keep_box_mask_and_empty_is_not_no_detection(tmp_path,monkeypatch,value,status):
    frames=make_frames(tmp_path);out=tmp_path/'out';out.mkdir()
    model=FakeModel(torch.full((1,1,4,6),value))
    monkeypatch.setattr(module,'install_multiscale_anchor',lambda *args,**kwargs:[])
    monkeypatch.setattr(module,'early_feature_points',lambda *args,**kwargs:(None,'degenerate_prototypes'))
    result=module.generate_prompt(model,frames,0,[0,0,4,3],out,(6,4))
    assert result['status']==status and result['points'] is None
    assert result['point_status']=='degenerate_prototypes' and model.point_calls==0
    assert model.box_calls==1 and model.reset
    assert result['lean_mask_status']==('automatic_mask' if value>0 else 'empty_prediction_box_only')
    assert not (out/'A').exists()


def test_box_points_adds_existing_box_and_checks_geometry(tmp_path,monkeypatch):
    frames=make_frames(tmp_path);out=tmp_path/'out';out.mkdir()
    model=FakeModel(torch.ones((1,1,4,6)))
    monkeypatch.setattr(module,'install_multiscale_anchor',lambda *args,**kwargs:[])
    monkeypatch.setattr(module,'early_feature_points',lambda *args,**kwargs:({'feature_points':[[1,1],[2,2]]},'available'))
    result=module.generate_prompt(model,frames,0,[0,0,4,3],out,(6,4))
    assert result['points']==[[1,1],[2,2]] and model.point_calls==1


def test_nonfinite_logits_fail_closed():
    with pytest.raises(ValueError):module.binary_mask(torch.full((1,1,4,6),float('nan')),(4,6))
    with pytest.raises(ValueError):module.binary_mask(torch.zeros((1,1,3,6)),(4,6))


def test_reference_gate_rejects_copied_process_and_tampering(tmp_path,monkeypatch):
    archive=tmp_path/'archive';(archive/'seq6').mkdir(parents=True)
    prompt={'first_detection_index':0,'box':[0,0,4,3],'confidence':.5,'point_status':'available',
            'points':[[1,1],[2,2]],'lean_mask_status':'automatic_mask','feature_shape':[1,32,2,2],
            'early_feature_sha256':'f'*64}
    saved=Image.fromarray(np.full((4,6),255,dtype=np.uint8))
    saved.save(archive/'seq6/prompt_refined.png');saved.save(archive/'seq6/prompt_provisional.png')
    prompt['refined_mask_sha256']=module.digest(archive/'seq6/prompt_refined.png')
    prompt['provisional_mask_sha256']=module.digest(archive/'seq6/prompt_provisional.png')
    (archive/'seq6/prompt.json').write_text(json.dumps(prompt))
    monkeypatch.setattr(module,'ARCHIVE',archive)
    monkeypatch.setattr(module,'BASE',tmp_path)
    sig={'frozen':'signature'}
    directories=[]
    for i in range(2):
        out=tmp_path/f'ref{i}';out.mkdir();directories.append(out)
        saved.save(out/'prompt_refined.png');saved.save(out/'prompt_provisional.png')
        (out/'model_state.json').write_text('{}')
        p={**prompt,'model_state_sha256':module.digest(out/'model_state.json')}
        (out/'prompt.json').write_text(json.dumps(p))
        (out/'frames.json').write_text(json.dumps([{'frame':0,'image_sha256':'a'*64}]))
        (out/'detections.json').write_text(json.dumps([{'frame':0,'image_sha256':'a'*64,'box':[0,0,4,3],'confidence':.5}]))
        rec={'passed':True,'mode':'reference','sequence':'seq6','signature':sig,
             'prompt_sha256':module.digest(out/'prompt.json'),'frames_sha256':module.digest(out/'frames.json'),
             'detections_sha256':module.digest(out/'detections.json'),'label_guard':{'open_attempts':0},
             'pid':i+10,'run_uuid':str(i)}
        (out/'completion.json').write_text(json.dumps(rec))
    assert module.verify_reference_pair(*directories,sig)['passed']
    b=directories[1];rec=json.loads((b/'completion.json').read_text());rec['pid']=10
    (b/'completion.json').write_text(json.dumps(rec))
    with pytest.raises(ValueError,match='distinct process'):
        module.verify_reference_pair(*directories,sig)
    rec['pid']=11;(b/'completion.json').write_text(json.dumps(rec))
    (b/'detections.json').write_text('[1]')
    with pytest.raises(ValueError,match='artifacts changed'):
        module.verify_reference_pair(*directories,sig)


def test_independent_audit_must_bind_gate_source_and_both_reference_receipts(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'BASE',tmp_path)
    gate_dir=tmp_path/'gate';gate_dir.mkdir();(gate_dir/'gate.json').write_text('{}')
    refs=[];rows=[]
    for i in range(2):
        folder=tmp_path/f'ref{i}';folder.mkdir();refs.append(folder)
        (folder/'frames.json').write_text('[{}]');(folder/'detections.json').write_text('[{}]')
        prompt={'model_state_sha256':f'model{i}'}
        (folder/'prompt.json').write_text(json.dumps(prompt))
        completion={'pid':i+2,'run_uuid':f'uuid{i}','frames_sha256':module.digest(folder/'frames.json'),
                    'detections_sha256':module.digest(folder/'detections.json'),'prompt_sha256':module.digest(folder/'prompt.json')}
        (folder/'completion.json').write_text(json.dumps(completion))
        rows.append({'completion_sha256':module.digest(folder/'completion.json'),**completion,**prompt})
    gate={'reference_a':str(refs[0]),'reference_b':str(refs[1])}
    audit={'passed':True,'gate_sha256':module.digest(gate_dir/'gate.json'),
           'audit_source_sha256':module.digest(module.ROOT/'scripts/audit_unarchived_prompt_gate_v1.py'),
           'input_frames_checked':1,'detector_attempts_checked':1,'reference_runs_checked':2,
           'label_guard':{'open_attempts':0,'blocked_paths':1},'runs':rows}
    path=tmp_path/'audit.json';path.write_text(json.dumps(audit))
    assert module.verify_independent_audit(path,gate_dir,gate)['passed']
    audit['runs'][1]['completion_sha256']='bad';path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='audited reference identity'):
        module.verify_independent_audit(path,gate_dir,gate)
    audit['runs'][1]['completion_sha256']=module.digest(refs[1]/'completion.json')
    audit['gate_sha256']='bad';path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='gate audit differs'):
        module.verify_independent_audit(path,gate_dir,gate)


def test_target_mode_refuses_to_run_without_independent_audit(monkeypatch):
    monkeypatch.setattr(module.sys,'argv',['generator','--mode','target','--sequence','seq1',
        '--checkpoint','unused','--detector','unused','--gate','unused','--output','unused'])
    with pytest.raises(ValueError,match='independent audit'):
        module.main()


def test_trace_rejects_resealed_earlier_detection_or_wrong_input():
    inventory=[{'frame':0,'image_sha256':'a'}, {'frame':1,'image_sha256':'b'}]
    prompt={'first_detection_index':1,'box':[0,0,4,3],'confidence':.7}
    attempts=[{'frame':0,'image_sha256':'a','box':None,'confidence':None},
              {'frame':1,'image_sha256':'b','box':[0,0,4,3],'confidence':.7}]
    module.validate_detection_trace(attempts,inventory,prompt)
    changed=[dict(a) for a in attempts];changed[0]['box']=[0,0,1,1];changed[0]['confidence']=.4
    with pytest.raises(ValueError,match='continued'):module.validate_detection_trace(changed,inventory,prompt)
    changed=[dict(a) for a in attempts];changed[1]['image_sha256']='wrong'
    with pytest.raises(ValueError,match='input order'):module.validate_detection_trace(changed,inventory,prompt)


def test_duplicate_screen_detects_target_cross_sequence_and_frozen_tamper(tmp_path,monkeypatch):
    data=tmp_path/'manifest.jsonl';data.write_text('test')
    p=tmp_path/'training';p.mkdir();(p/'manifest.json').write_text('{}')
    monkeypatch.setattr(module,'DATA',data);monkeypatch.setattr(module,'TRAIN_PREFLIGHT',p)
    monkeypatch.setattr(module,'EXPECTED_TRAIN',{'seq2'});monkeypatch.setattr(module,'EXPECTED_VALIDATION',{'seq9'})
    monkeypatch.setattr(module,'ALL_SEQUENCES',{'seq1','seq7','seq2','seq9'})
    videos={}
    for n,sequence in enumerate(('seq1','seq7','seq2','seq9')):
        path=tmp_path/f'{sequence}.png'
        Image.fromarray(np.full((4,6,3),n*50,dtype=np.uint8)).save(path)
        videos[sequence]=[SimpleNamespace(frame_index=0,image_path=path)]
    frozen={'manifest_sha256':module.digest(data),'sequences':{s:{'frames':1,'image_sha256':[module.digest(videos[s][0].image_path)]} for s in ('seq2','seq9')}}
    assert module.screen_target_duplicates(videos,frozen)['target_cross_sequence_exact_or_decoded_matches']==0
    with pytest.raises(ValueError,match='coverage differs'):
        module.screen_target_duplicates({k:v for k,v in videos.items() if k!='seq9'},frozen)
    frozen['sequences']['seq2']['image_sha256'][0]='0'*64
    with pytest.raises(ValueError,match='training/validation image differs'):
        module.screen_target_duplicates(videos,frozen)
    frozen['sequences']['seq2']['image_sha256'][0]=module.digest(videos['seq2'][0].image_path)
    videos['seq7'][0].image_path.write_bytes(videos['seq1'][0].image_path.read_bytes())
    with pytest.raises(ValueError,match='duplicates a different sequence'):
        module.screen_target_duplicates(videos,frozen)
