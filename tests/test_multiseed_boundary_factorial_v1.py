import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import run_multiseed_boundary_factorial_v1 as launcher
from scripts import score_multiseed_boundary_factorial_v1 as scorer
from scripts import audit_multiseed_boundary_factorial_v1 as auditor


def test_plan_freezes_all_32_before_scoring_and_binds_each_seed(tmp_path):
    gates={seed:(tmp_path/str(seed)/'gate',tmp_path/str(seed)/'audit') for seed in launcher.SEEDS}
    phases,mapping=launcher.commands(tmp_path/'run',gates)
    assert len(phases)==33 and phases[-1][0]=='score'
    assert [(n,s) for n,s,_ in phases[:-1]]==[(f'{seed}_{seq}_{arm}',seed)
        for seed in launcher.SEEDS for seq in launcher.SEQUENCES for arm in launcher.ARMS]
    for name,seed,cmd in phases[:-1]:
        _,seq,arm=name.split('_')
        assert cmd[cmd.index('--seed')+1]==str(seed)
        assert cmd[cmd.index('--engineering-gate')+1]==str(gates[seed][0])
        assert cmd[cmd.index('--engineering-audit')+1]==str(gates[seed][1])
        assert ('--matched-a' in cmd) is (arm!='A')
        if arm!='A':assert cmd[cmd.index('--matched-a')+1]==mapping[str(seed)][seq]['A']


def test_source_closure_includes_metric_and_audit_helpers():
    source=launcher.sources()
    for key in ('primary_metric','primary_boundary_metric','independent_metric','mask_loader','video_loader','auditor'):
        assert source[key].is_file()


def test_map_rejects_aliases_and_missing_seed(tmp_path):
    data={}
    for seed in launcher.SEEDS:
        data[str(seed)]={}
        for seq in launcher.SEQUENCES:
            data[str(seed)][seq]={}
            for arm in launcher.ARMS:
                folder=tmp_path/str(seed)/seq/arm;folder.mkdir(parents=True)
                data[str(seed)][seq][arm]=str(folder)
    path=tmp_path/'run_map.json';path.write_text(json.dumps(data))
    assert len(scorer.mapped_runs(tmp_path)[1])==2
    data[str(launcher.SEEDS[1])]['seq8']['B']=data[str(launcher.SEEDS[0])]['seq8']['A']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='distinct'):scorer.mapped_runs(tmp_path)
    del data[str(launcher.SEEDS[1])];path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Seed coverage'):scorer.mapped_runs(tmp_path)


def test_primary_and_independent_audits_do_not_open_labels_on_failed_precheck(tmp_path,monkeypatch):
    monkeypatch.setattr(auditor,'precheck',lambda *_: (_ for _ in ()).throw(ValueError('incomplete')))
    monkeypatch.setattr(auditor,'label',lambda *_: (_ for _ in ()).throw(AssertionError('label opened')))
    with pytest.raises(ValueError,match='incomplete'):auditor.audit(tmp_path)


def test_map_rejects_duplicate_json_keys(tmp_path):
    (tmp_path/'run_map.json').write_text('{"20261125":{},"20261125":{}}')
    with pytest.raises(ValueError,match='Duplicate'):scorer.mapped_runs(tmp_path)


def test_invalid_output_is_rejected_before_label_access(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir()
    monkeypatch.setattr(scorer,'BASE',root)
    monkeypatch.setattr(scorer,'validate_output',lambda path:Path(path).resolve())
    monkeypatch.setattr(scorer,'load_label_mask',lambda *_: (_ for _ in ()).throw(AssertionError('label opened')))
    monkeypatch.setattr(scorer.sys,'argv',['score','--run',str(root),'--output',str(tmp_path/'outside')])
    with pytest.raises(ValueError,match='new folder within'):scorer.main()


def test_prelabel_guard_blocks_and_counts_mask_open(tmp_path,monkeypatch):
    label=tmp_path/'label';label.write_bytes(b'x'); other=tmp_path/'other';other.write_bytes(b'y')
    monkeypatch.setattr(scorer,'mask_path_sets',lambda *_:({str(label.resolve())},{str(other.resolve())}))
    guard=scorer.install_prelabel_guard()
    try:
        with pytest.raises(PermissionError,match='Label mask blocked'):label.read_bytes()
        assert guard['open_attempts']==1
    finally:guard['active']=False
    assert label.read_bytes()==b'x'


def test_label_changed_during_decode_is_rejected(tmp_path,monkeypatch):
    label=tmp_path/'label';label.write_bytes(b'original')
    from scripts.memory_token_io import digest
    frame=SimpleNamespace(mask_path=label,frame_index=0)
    metadata={'seq8':[{'label_sha256':digest(label)}]}
    monkeypatch.setattr(scorer,'SEEDS',(20261125,));monkeypatch.setattr(scorer,'SEQUENCES',('seq8',))
    monkeypatch.setattr(scorer,'ARMS',('A',))
    monkeypatch.setattr(scorer,'load_label_mask',lambda *_:(label.write_bytes(b'changed'),np.zeros((2,2)))[1])
    receipts={'20261125':{'seq8':{'A':{'prompt_frame':0}}}}
    with pytest.raises(ValueError,match='Label changed while decoding'):
        scorer.score_all({}, {'seq8':[frame]}, receipts, metadata)


def test_prediction_changed_during_decode_is_rejected(tmp_path,monkeypatch):
    label=tmp_path/'label';label.write_bytes(b'original')
    image=tmp_path/'000000.png';Image.fromarray(np.zeros((2,2),dtype=np.uint8)).save(image)
    from scripts.memory_token_io import digest as real_digest
    frame=SimpleNamespace(mask_path=label,frame_index=0)
    metadata={'seq8':[{'label_sha256':real_digest(label)}]}; mask_sha=real_digest(image)
    monkeypatch.setattr(scorer,'SEEDS',(20261125,));monkeypatch.setattr(scorer,'SEQUENCES',('seq8',))
    monkeypatch.setattr(scorer,'ARMS',('A',));monkeypatch.setattr(scorer,'load_label_mask',lambda *_:np.zeros((2,2)))
    calls={'image':0}
    def altered(path):
        if Path(path)==image:
            calls['image']+=1
            return mask_sha if calls['image']==1 else 'tampered'
        return real_digest(path)
    monkeypatch.setattr(scorer,'digest',altered)
    receipts={'20261125':{'seq8':{'A':{'prompt_frame':0,'masks':[{'sha256':mask_sha}]}}}}
    mapping={'20261125':{'seq8':{'A':tmp_path}}}
    with pytest.raises(ValueError,match='Prediction changed while decoding'):
        scorer.score_all(mapping, {'seq8':[frame]}, receipts, metadata)


def test_validate_all_uses_each_seed_gate_and_never_opens_labels(tmp_path,monkeypatch):
    from scripts.memory_token_io import digest
    study=tmp_path/'study';pre=tmp_path/'pre';pre.mkdir();(pre/'manifest.json').write_text('{}');(pre/'completion.json').write_text('{}')
    checkpoint=tmp_path/'backbone';checkpoint.write_bytes(b'backbone')
    detector=tmp_path/'detector';detector.write_bytes(b'detector')
    images=[]
    for i in range(2):
        path=tmp_path/f'image{i}.png';Image.fromarray(np.full((2,2),i,dtype=np.uint8)).save(path)
        images.append(SimpleNamespace(image_path=path,frame_index=i))
    prompt={'first_detection_index':0};provenance={'prompt_mask_sha256':'mask'}
    monkeypatch.setattr(scorer,'SEQUENCES',('seq8',));monkeypatch.setattr(scorer,'STUDY',study)
    monkeypatch.setattr(scorer,'SEQ8_PRE',pre);monkeypatch.setattr(scorer,'CHECKPOINT',checkpoint)
    monkeypatch.setattr(scorer,'DETECTOR',detector);monkeypatch.setattr(scorer,'DETECTOR_SHA256',digest(detector))
    monkeypatch.setattr(scorer,'expected_inputs',lambda *_:(images,prompt,'prompt',{'kind':'full'},None,provenance))
    monkeypatch.setattr(scorer,'_check_events',lambda folder,rec:{'seed':rec['seed']})
    monkeypatch.setattr(scorer,'check_telemetry',lambda *_:({},{}))
    monkeypatch.setattr(scorer,'load_label_mask',lambda *_: (_ for _ in ()).throw(AssertionError('label opened')))
    mapping={};gates={}
    for seed in scorer.SEEDS:
        key=str(seed);g=tmp_path/key/'gate.json';g.parent.mkdir(parents=True);g.write_text(json.dumps({'runtime_identity':{'seed':seed}}))
        audit=tmp_path/key/'audit.json';audit.write_text('{}');gates[seed]=(g,audit)
        for kind in ('region_boundary','mask_only_boundary'):
            folder=study/key/'runs'/kind;folder.mkdir(parents=True);(folder/'checkpoint.pt').write_bytes(kind.encode())
        mapping[key]={'seq8':{}}
        for arm in scorer.ARMS:
            folder=tmp_path/key/'seq8'/arm;folder.mkdir(parents=True);mapping[key]['seq8'][arm]=folder
            masks=[]
            for i in range(2):
                path=folder/f'{i:06d}.png';Image.fromarray(np.full((2,2),i*255,dtype=np.uint8)).save(path)
                masks.append({'frame':i,'sha256':digest(path)})
            initial={'frame':0,'mask_sha256':masks[0]['sha256'],'pred_masks_sha256':'p0','maskmem_features_sha256':'m0',
                     'obj_ptr_sha256':'o0','object_score_logits_sha256':'s0'}
            first={'frame':1,'mask_sha256':masks[1]['sha256'],'pred_masks_sha256':'p1',
                   'maskmem_features_sha256':'m1x' if arm in 'CD' else 'm1',
                   'obj_ptr_sha256':'o1','object_score_logits_sha256':'s1',
                   'output_logits_sha256':'out1x' if arm in 'BD' else 'out1'}
            rec={'passed':True,'full_video':True,'limit':0,'seed':seed,'sequence':'seq8','arm':arm,
                'runner_source_sha256':digest(scorer.RUNNER),'prompt_frame':0,'prompt_sha256':'prompt',
                'prompt_mask_sha256':'mask','view':{'kind':'full'},'checkpoint_sha256':digest(checkpoint),
                'detector_sha256':digest(detector),'candidate_checkpoint_sha256':digest(study/key/'runs/region_boundary/checkpoint.pt'),
                'control_checkpoint_sha256':digest(study/key/'runs/mask_only_boundary/checkpoint.pt'),
                'engineering_gate_sha256':digest(g),'engineering_audit_sha256':digest(audit),
                'seq8_preflight_sha256':digest(pre/'manifest.json'),'seq8_preflight_completion_sha256':digest(pre/'completion.json'),
                'label_guard':{'open_attempts':0},'input_images':[{'frame':i,'sha256':digest(f.image_path)} for i,f in enumerate(images)],
                'masks':masks,'post_prompt_state':[initial,first]}
            (folder/'completion.json').write_text(json.dumps(rec))
            for name in ('memory_events.json','positions.json','selections.json'):(folder/name).write_text('[]')
    _,receipts,_,_=scorer.validate_all(mapping,gates)
    assert receipts['20261125']['seq8']['A']['seed']==20261125
    victim=mapping['20261135']['seq8']['D']/'completion.json';bad=json.loads(victim.read_text());bad['seed']=20261125
    victim.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='identity or guarded inputs'):
        scorer.validate_all(mapping,gates)
