import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import scripts.score_unarchived_no_detection_v1 as module


def fixture_target(tmp_path):
    images=[];frames=[];inventory=[];attempts=[]
    for i in range(2):
        image=tmp_path/f'image{i}.png';Image.fromarray(np.full((3,4,3),i*20,dtype=np.uint8)).save(image)
        label=tmp_path/f'label{i}.png';Image.fromarray(np.zeros((3,4),dtype=np.uint8)).save(label)
        frames.append(SimpleNamespace(frame_index=i,image_path=image,mask_path=label))
        inventory.append({'frame':i,'source_frame_index':i,'image_path':str(image.resolve()),'image_sha256':module.digest(image)})
        attempts.append({'frame':i,'image_sha256':module.digest(image),'box':None,'confidence':None})
    folder=tmp_path/'target';folder.mkdir();masks={}
    for arm in module.ARMS:
        (folder/arm).mkdir();masks[arm]=[]
        for i in range(2):
            f=folder/arm/f'{i:06d}.png';Image.fromarray(np.zeros((3,4),dtype=np.uint8)).save(f)
            masks[arm].append({'frame':i,'sha256':module.digest(f)})
    prompt={'status':'no_detection','first_detection_index':None,'box':None,'confidence':None,'points':None,
            'point_status':'no_detection','lean_mask_status':'no_detection','post_prompt_frames':0,'empty_arm_masks':masks}
    for name,obj in (('prompt.json',prompt),('frames.json',inventory),('detections.json',attempts),('duplicate_screen.json',{'passed':True})):
        (folder/name).write_text(json.dumps(obj))
    rec={'passed':True,'mode':'target','sequence':'seq1','signature':{'test':'sig'},'reference_gate_sha256':'gate',
         'independent_audit_sha256':'ref','label_guard':{'open_attempts':0},
         **{field:module.digest(folder/name) for field,name in [('prompt_sha256','prompt.json'),('frames_sha256','frames.json'),
              ('detections_sha256','detections.json'),('duplicate_screen_sha256','duplicate_screen.json')]}}
    (folder/'completion.json').write_text(json.dumps(rec))
    return folder,frames,prompt,rec


def test_no_detection_evidence_and_changed_mask_fail_before_label_scoring(tmp_path):
    folder,frames,prompt,rec=fixture_target(tmp_path)
    got=module.verify_one_target(folder,'seq1',frames,{'test':'sig'},'gate','ref')
    assert got[1]['status']=='no_detection'
    Image.fromarray(np.full((3,4),255,dtype=np.uint8)).save(folder/'B/000001.png')
    with pytest.raises(ValueError,match='Prediction hash differs'):
        module.verify_one_target(folder,'seq1',frames,{'test':'sig'},'gate','ref')


def test_resealed_earlier_detection_is_not_no_detection(tmp_path):
    folder,frames,prompt,rec=fixture_target(tmp_path)
    attempts=json.loads((folder/'detections.json').read_text());attempts[0]['box']=[0,0,2,2];attempts[0]['confidence']=.8
    (folder/'detections.json').write_text(json.dumps(attempts));rec['detections_sha256']=module.digest(folder/'detections.json')
    (folder/'completion.json').write_text(json.dumps(rec))
    with pytest.raises(ValueError,match='continued after'):
        module.verify_one_target(folder,'seq1',frames,{'test':'sig'},'gate','ref')


def test_all_frame_summary_excludes_undefined_post_prompt_and_counts_miss(tmp_path,monkeypatch):
    folder,frames,prompt,rec=fixture_target(tmp_path)
    # Dataset loader supports indexed 0/1 PNG masks.
    label=np.zeros((3,4),dtype=np.uint8);label[1,2]=1
    Image.fromarray(label).save(frames[1].mask_path)
    monkeypatch.setattr(module,'SEQUENCES',('seq1',))
    inventory=json.loads((folder/'frames.json').read_text())
    rows,summaries=module.score({'seq1':frames},{'seq1':{'folder':str(folder),'frames':inventory,'prompt':prompt}})
    s=summaries['seq1']
    assert s['post_prompt_frame_count']==0 and s['post_prompt_present_dice'] is None
    assert s['all_frame_present']==1 and s['pre_initialization_present_missed']==1 and s['all_frame_present_dice']==0
    assert s['labeled_empty_false_alarms']==0 and s['labeled_empty_denominator']==1
    assert rows['seq1'][0]['dice_on_present'] is None and rows['seq1'][1]['dice_on_present']==0


@pytest.mark.parametrize('mutation,expected',[('image','Source image changed before scoring'),('prediction','Prediction changed before scoring')])
def test_score_rejects_mutated_frozen_input_before_reading_label(tmp_path,monkeypatch,mutation,expected):
    folder,frames,prompt,rec=fixture_target(tmp_path)
    monkeypatch.setattr(module,'SEQUENCES',('seq1',))
    inventory=json.loads((folder/'frames.json').read_text())
    if mutation=='image':Image.fromarray(np.ones((3,4,3),dtype=np.uint8)).save(frames[0].image_path)
    else:Image.fromarray(np.ones((3,4),dtype=np.uint8)).save(folder/'A/000000.png')
    monkeypatch.setattr(module,'load_label_mask',lambda *args:pytest.fail('label was opened before input checks'))
    with pytest.raises(ValueError,match=expected):
        module.score({'seq1':frames},{'seq1':{'folder':str(folder),'frames':inventory,'prompt':prompt}})
