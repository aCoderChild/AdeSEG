import json
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.memory_token_io import digest
from scripts import verify_warm_author_comparator_gate_v1 as verifier


@pytest.fixture
def smokes(tmp_path,monkeypatch):
    monkeypatch.setattr(verifier,'EXPERIMENT_ROOT',tmp_path)
    frames={};smokes={}
    for seq,start in [('seq6',2),('seq23',0)]:
        path=tmp_path/seq;path.mkdir();frames[seq]=[];rows=[]
        for i in range(start+5):
            image=path/f'image{i}.png';prediction=path/f'{i:06d}.png'
            Image.new('RGB',(8,5)).save(image)
            Image.fromarray(np.full((5,8),0 if i<start else 255,np.uint8)).save(prediction)
            frames[seq].append(SimpleNamespace(frame_index=i*2,image_path=image))
            rows.append({'frame':i,'source_frame_index':i*2,'image_sha256':digest(image),
                         'mask_sha256':digest(prediction),'post_prompt':i>=start,
                         **({'logits_sha256':'d'*64} if i>=start else {})})
        (path/'receipt.json').write_text(json.dumps({'passed':True,'full_video':False,'method':'lean',
            'sequence':seq,'identity':{'method':'lean','sequence':seq,'source':'same'},
            'prompt':{'seq':seq},'prompt_frame':start,'noncausal':False,'label_guard':{'open_attempts':0},
            'process_id':1 if seq=='seq6' else 2,'state_sha256':'same',
            'warm_feature_callback':{'early_sha256':'a'*64,'positive_sha256':'b'*64,'negative_sha256':'c'*64,
                                     'cache_before':[0],'cache_after':[0],'early_shape':[32,256,256]},
            'positions':[{'spatial_keys':4096}],'prediction_wall_seconds':1.0,
            'feature_callback_enabled':True,'records':rows}))
        smokes[seq]=path
    monkeypatch.setattr(verifier,'input_data',lambda seq,plan:(frames[seq],2 if seq=='seq6' else 0,{'seq':seq},None))
    monkeypatch.setattr(verifier,'identity',lambda method,seq,plan:{'method':method,'sequence':seq,'source':'same'})
    return smokes


def test_warm_gate_rejects_changed_mask(smokes):
    result=verifier.audit('lean',smokes,{})
    assert result['passed'] and len(result['smokes'])==2
    Image.new('L',(8,5)).save(smokes['seq6']/'000002.png')
    with pytest.raises(ValueError,match='input or output'):
        verifier.audit('lean',smokes,{})


def test_warm_gate_rejects_wrong_causal_claim(smokes):
    path=smokes['seq23']/'receipt.json';rec=json.loads(path.read_text());rec['noncausal']=True
    path.write_text(json.dumps(rec))
    with pytest.raises(ValueError,match='identity or coverage'):
        verifier.audit('lean',smokes,{})


def test_warm_feature_callback_checks_cache_and_prototypes():
    from scripts.run_warm_author_comparator_v1 import warm_feature_callback
    class Fake:
        def _get_image_feature(self,state,frame,batch_size):
            state['cached_features']={frame:True}
            feature=torch.arange(3*16*16,dtype=torch.float32).reshape(1,3,16,16)+1
            return None,{'backbone_fpn':[feature]},None,None,None
    state={'cached_features':{0:True}};mask=np.zeros((16,16),bool);mask[8,8]=True;record={}
    warm_feature_callback(Fake(),state,3,mask,record)
    assert record['cache_before']==[0] and record['cache_after']==[3]
    assert record['early_shape']==[3,16,16] and len(record['positive_sha256'])==64


def test_launcher_freezes_all_84_videos_before_scoring(tmp_path):
    from scripts.run_warm_author_study_v1 import commands
    names=[name for name,_ in commands(tmp_path)]
    assert len([x for x in names if x.startswith('smoke_')])==8
    assert len([x for x in names if x.startswith('gate_')])==4
    assert len([x for x in names if x.startswith('control_')])==4
    assert len([x for x in names if x.startswith('featureaudit_')])==4
    assert len([x for x in names if x.startswith('full_')])==84
    assert names[-3:]==['verify','score','audit'] and all(x.startswith('full_') for x in names[20:-3])


def test_warm_scoring_blocks_labels_until_verification(tmp_path,monkeypatch):
    from scripts import score_warm_author_study_v1 as scorer
    folder=tmp_path/'run';(folder/'verification').mkdir(parents=True)
    (folder/'verification/verification.json').write_text('{}')
    monkeypatch.setattr(scorer,'new_output',lambda p:p)
    monkeypatch.setattr(scorer,'install_phase_mask_guard',lambda _: {'open_attempts':0,'active':True})
    monkeypatch.setattr(scorer,'phase_bound_sha',lambda *_:'expected')
    monkeypatch.setattr(scorer,'read_bound',lambda *_:{'passed':True,'label_guard':{'open_attempts':0},
        'scorer_sha256':digest(scorer.Path(scorer.__file__))})
    monkeypatch.setattr(scorer,'validate_frozen',lambda *_:(_ for _ in ()).throw(ValueError('not frozen')))
    monkeypatch.setattr(scorer,'load_label_mask',lambda *_:pytest.fail('label opened before freeze'))
    with pytest.raises(ValueError,match='not frozen'):scorer.score(folder)


def test_phase_proof_is_bound_and_seq23_outside_view_is_empty(tmp_path):
    from scripts.score_warm_author_study_v1 import phase_bound_sha,require_viewport_mask
    from scripts.score_author_lean_automatic_v1 import read_bound
    proof=tmp_path/'proof.json';proof.write_text('{"passed": true}')
    (tmp_path/'phases.jsonl').write_text(json.dumps({'phase':'verify','passed':True,'artifact_sha256':digest(proof)})+'\n')
    sha=phase_bound_sha(tmp_path,'verify');assert read_bound(proof,sha)['passed']
    proof.write_text('{"passed": false}')
    with pytest.raises(ValueError,match='Artifact changed'):
        read_bound(proof,phase_bound_sha(tmp_path,'verify'))
    mask=np.zeros((720,1280),np.uint8);mask[0,446]=255;require_viewport_mask(mask)
    mask[0,445]=255
    with pytest.raises(ValueError,match='outside'):
        require_viewport_mask(mask)


def test_feature_access_control_reports_difference_without_rejecting(smokes,tmp_path,monkeypatch):
    from scripts import audit_warm_author_feature_access_v1 as control_audit
    root=tmp_path/'control';root.mkdir();enabled=smokes['seq6'];rec=json.loads((enabled/'receipt.json').read_text())
    for row in rec['records']:shutil.copyfile(enabled/f"{row['frame']:06d}.png",root/f"{row['frame']:06d}.png")
    rec['feature_callback_enabled']=False;rec['warm_feature_callback']={};rec['process_id']=9
    Image.new('L',(8,5)).save(root/'000002.png')
    rec['records'][2]['mask_sha256']=digest(root/'000002.png');rec['records'][2]['logits_sha256']='e'*64
    (root/'receipt.json').write_text(json.dumps(rec));gate=tmp_path/'gate.json';gate.write_text('{}')
    monkeypatch.setattr(control_audit,'EXPERIMENT_ROOT',tmp_path)
    monkeypatch.setattr(control_audit,'identity',verifier.identity)
    monkeypatch.setattr(control_audit,'input_data',verifier.input_data)
    monkeypatch.setattr(control_audit,'verify_gate',lambda *_:{'state_sha256':'same',
        'smokes':[{'sequence':'seq6','receipt_path':str(enabled/'receipt.json'),
                   'receipt_sha256':digest(enabled/'receipt.json')}]})
    result=control_audit.audit('lean',gate,root,{})
    assert result['passed'] and result['differences'][0]['mask_disagree_pixels']==40
    assert not result['differences'][0]['logits_hash_equal']
    monkeypatch.setattr(control_audit,'verify_gate',lambda *_:{'state_sha256':'same',
        'smokes':[{'sequence':'seq6','receipt_path':str(enabled/'receipt.json'),'receipt_sha256':'0'*64}]})
    with pytest.raises(ValueError,match='reconstructed gate'):
        control_audit.audit('lean',gate,root,{})
