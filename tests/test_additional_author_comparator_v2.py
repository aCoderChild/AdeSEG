import json
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import run_additional_author_comparator_v2 as run
from scripts import verify_additional_author_comparator_gate_v2 as verify
from scripts import run_additional_author_comparator_study_v2 as study
from scripts.memory_token_io import digest


def test_all_study_uses_three_mps_smokes_then_54_full_videos(tmp_path):
    phases,mapping=study.commands(tmp_path)
    names=[x[0] for x in phases]
    assert len(mapping)==18 and all(len(v)==3 for v in mapping.values())
    assert sum(n.startswith('short_') for n in names)==3
    assert sum(n.startswith('gate_') for n in names)==3
    assert sum(n.startswith('full_') for n in names)==54
    assert names[-3:]==['verify','score','audit']
    assert all('--cpu' not in cmd and ('cpu' not in cmd) for _,cmd in phases)


def test_active_phase_stops_if_reserve_is_lost(tmp_path,monkeypatch):
    events=[]
    class FakeProcess:
        def __init__(self,*args,**kwargs):pass
        def wait(self,timeout=None):
            if not events:raise subprocess.TimeoutExpired('child',timeout)
            return -15
        def terminate(self):events.append('terminated')
        def kill(self):events.append('killed')
    monkeypatch.setattr(study.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(study.shutil,'disk_usage',lambda _:SimpleNamespace(free=study.GUARD-1))
    with pytest.raises(RuntimeError,match='reserve lost'):
        study.run_phase(['child'],tmp_path/'phase.log',tmp_path,[])
    assert events==['terminated']


def test_source_manifest_detects_lock_helper_change(monkeypatch):
    before=study.source_hashes();real_digest=study.digest
    def altered(path):
        return 'changed' if str(path).endswith('scripts/resume_source_group_capture_v1.py') else real_digest(path)
    monkeypatch.setattr(study,'digest',altered)
    assert study.source_hashes()['lock_helper']!=before['lock_helper']


def test_resume_rechecks_prefix_and_preserves_partial(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'OUT',tmp_path)
    monkeypatch.setattr(study,'source_hashes',lambda:{'sources':'frozen'})
    checkpoint=tmp_path/'checkpoint';checkpoint.write_text('checkpoint')
    detector=tmp_path/'detector';detector.write_text('detector')
    monkeypatch.setattr(study,'CHECKPOINT',checkpoint);monkeypatch.setattr(study,'DETECTOR',detector)
    for attr in ('EXTENDED','REMAINING','TRAIN_PREFLIGHT'):
        folder=tmp_path/attr;folder.mkdir();(folder/'manifest.json').write_text(attr)
        monkeypatch.setattr(study,attr,folder)
    phases,mapping=study.commands(tmp_path,('seq10',))
    manifest={'sources':{'sources':'frozen'},'run_map':mapping,'methods':list(study.METHODS),
              'representative_for_method':study.REPRESENTATIVE,'phases':[n for n,_ in phases],
              'sequences':['seq10'],'checkpoint_sha256':digest(checkpoint),'detector_sha256':digest(detector),
              'extended_preflight_sha256':digest(study.EXTENDED/'manifest.json'),
              'remaining_preflight_sha256':digest(study.REMAINING/'manifest.json'),
              'train_preflight_sha256':digest(study.TRAIN_PREFLIGHT/'manifest.json'),
              'launcher_label_guard':{'open_attempts':0},'python_hash_seed':'0','mps_fallback':'0','mps_fast_math':'0'}
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest));(tmp_path/'run_map.json').write_text(json.dumps(mapping))
    name=phases[0][0];file=study.artifact(tmp_path,name);file.parent.mkdir(parents=True);file.write_text('{}')
    log=tmp_path/(name+'.log');log.write_text('done')
    row={'phase':name,'passed':True,'artifact_sha256':digest(file),'log_sha256':digest(log)}
    (tmp_path/'phases.jsonl').write_text(json.dumps(row)+'\n')
    assert study.validate_resume(tmp_path)[-1]==1
    next_name=phases[1][0];partial=study.artifact(tmp_path,next_name).parent;partial.mkdir(parents=True)
    (partial/'keep').write_text('kept');(tmp_path/(next_name+'.log')).write_text('failed')
    archived=study.preserve_partial(tmp_path,next_name)
    assert (archived/'output/keep').read_text()=='kept' and (archived/'phase.log').read_text()=='failed'
    file.write_text('changed')
    with pytest.raises(ValueError,match='artifact or log changed'):study.validate_resume(tmp_path)


def test_one_mps_smoke_receipt_reconstructed_and_tamper_rejected(tmp_path,monkeypatch):
    seq='seq6';method='native';start=2;frames=[];records=[]
    for i in range(7):
        img=tmp_path/f'source{i}.png';Image.fromarray(np.zeros((5,6,3),np.uint8)).save(img)
        frames.append(SimpleNamespace(image_path=img,frame_index=100+i))
        mask=tmp_path/f'{i:06d}.png';Image.fromarray(np.zeros((5,6),np.uint8)).save(mask)
        records.append({'frame':i,'source_frame_index':100+i,'image_sha256':digest(img),
                        'mask_sha256':digest(mask),'post_prompt':i>=start})
    identity={'method':method,'sequence':seq,'device':'mps'};prompt={'p':1}
    monkeypatch.setattr(verify,'identity',lambda *args:identity)
    monkeypatch.setattr(verify,'input_data',lambda *args:(frames,start,None,prompt))
    monkeypatch.setattr(run,'EXPERIMENT_ROOT',tmp_path)
    receipt={'passed':True,'full_video':False,'deterministic':True,'identity':identity,'device':'mps',
             'label_guard':{'open_attempts':0},'method':method,'sequence':seq,'prompt_frame':start,
             'prompt':prompt,'state_sha256':'state','noncausal':False,'records':records}
    (tmp_path/'receipt.json').write_text(json.dumps(receipt))
    gate=verify.audit(tmp_path,'c','d','e','r');gate['label_guard']={'open_attempts':0}
    path=tmp_path/'gate.json';path.write_text(json.dumps(gate))
    assert run.verify_gate(path,identity,prompt,'c','d','e','r')['passed']
    with pytest.raises(ValueError,match='Matching|matching'):
        run.verify_gate(path,dict(identity,method='samurai'),prompt,'c','d','e','r')
    Image.fromarray(np.ones((5,6),np.uint8)).save(tmp_path/'000003.png')
    with pytest.raises(ValueError,match='prediction changed'):
        run.verify_gate(path,identity,prompt,'c','d','e','r')
