import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import preflight_remaining_boundary_video_v1 as pre


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n')


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root=tmp_path/'archive'; seq='seq8'; source=root/'run'; folder=source/seq
    folder.mkdir(parents=True)
    manifest=tmp_path/'data.jsonl'; manifest.write_text('{}\n')
    checkpoint=tmp_path/'checkpoint.pt'; checkpoint.write_bytes(b'checkpoint')
    detector=tmp_path/'detector.pt'; detector.write_bytes(b'detector')
    image=tmp_path/'image.png'; Image.fromarray(np.full((8, 10, 3), 25, dtype=np.uint8)).save(image)
    mask=folder/'prompt_refined.png'; a=np.zeros((8,10),dtype=np.uint8); a[2:4,3:6]=255; Image.fromarray(a).save(mask)
    prompt={'first_detection_index':0,'box':[1,1,8,7],'points':[[2,2],[3,3]],'refined_mask_sha256':sha(mask)}
    save_json(folder/'prompt.json',prompt)
    source_frames=[{'source_frame_index':i,'image_sha256':sha(image),'dice':0.123+i} for i in range(5)]
    save_json(folder/'frames.json',source_frames)
    records=[{'sequence':seq,'frames':5,'first_detection_index':0,'point_status':'available',
              'lean_mask_status':'automatic_mask','prompt_sha256':sha(folder/'prompt.json'),
              'frames_sha256':sha(folder/'frames.json')}]
    save_json(source/'records.json',records); save_json(source/'completion.json',{'passed':True,'records_sha256':sha(source/'records.json')})
    save_json(source/'setup.json',{'sequences':[seq],'initialization':'first chronological YOLO detection; no GT prompt',
              'signature':{'checkpoint_sha256':sha(checkpoint),'detector_sha256':sha(detector),'manifest_sha256':sha(manifest)}})
    locations=root/'locations.json'; save_json(locations,{seq:str(source)})
    train=tmp_path/'train'; train.mkdir()
    config=tmp_path/'train_config.json';save_json(config,{'duplicate_screen_sequences':['seq8']})
    save_json(train/'manifest.json',{'manifest_sha256':sha(manifest),'sequences':{'seq2':{'split':'train'},'seq9':{'split':'validation'}},
              'config_sha256':sha(config),'source_sha256':{'scripts/preflight_remaining_boundary_video_v1.py':sha(pre.__file__)},
              'duplicates':{'cross_group_exact_byte':0,'cross_group_decoded_rgb':0}})
    save_json(train/'completion.json',{'passed':True,'manifest_sha256':sha(train/'manifest.json')})
    frames=[SimpleNamespace(frame_index=i,image_path=image,mask_path=tmp_path/'never-open-label.png') for i in range(5)]
    monkeypatch.setattr(pre,'SEQUENCES',(seq,));monkeypatch.setattr(pre,'EXPECTED_TRAIN',{'seq2'});monkeypatch.setattr(pre,'EXPECTED_VALIDATION',{'seq9'})
    monkeypatch.setattr(pre,'ARCHIVE',root);monkeypatch.setattr(pre,'LOCATIONS',locations);monkeypatch.setattr(pre,'DATA',manifest)
    monkeypatch.setattr(pre,'TRAIN_CONFIG',config)
    monkeypatch.setattr(pre,'CHECKPOINT_SHA',sha(checkpoint));monkeypatch.setattr(pre,'DETECTOR_SHA',sha(detector))
    train_image=tmp_path/'train.png'; Image.fromarray(np.full((8,10,3), 91, dtype=np.uint8)).save(train_image)
    val_image=tmp_path/'validation.png'; Image.fromarray(np.full((8,10,3), 142, dtype=np.uint8)).save(val_image)
    frozen= json.loads((train/'manifest.json').read_text())
    frozen['sequences']['seq2'].update(frames=1,image_sha256=[sha(train_image)])
    frozen['sequences']['seq9'].update(frames=1,image_sha256=[sha(val_image)])
    save_json(train/'manifest.json',frozen); save_json(train/'completion.json',{'passed':True,'manifest_sha256':sha(train/'manifest.json')})
    train_frames=[SimpleNamespace(frame_index=0,image_path=train_image,mask_path=tmp_path/'never-open-train-label.png')]
    val_frames=[SimpleNamespace(frame_index=0,image_path=val_image,mask_path=tmp_path/'never-open-validation-label.png')]
    monkeypatch.setattr(pre,'load_manifest',lambda path,split: {seq:frames,'seq2':train_frames,'seq9':val_frames} if split=='dev' else {})
    return SimpleNamespace(root=root,source=source,folder=folder,frames=frames,train=train,config=config,checkpoint=checkpoint,detector=detector,image=image,mask=mask)


def test_selective_metadata_ignores_outcome_fields(tmp_path):
    p=tmp_path/'frames.json'; p.write_text('[\n{\n"source_frame_index": 2,\n"dice": "not parsed",\n"image_sha256": "'+'a'*64+'"\n}\n]')
    assert pre.image_metadata_only(p)==([2],['a'*64])
    p.write_text('{"source_frame_index": 2, "image_sha256": "'+'a'*64+'"}')
    with pytest.raises(ValueError,match='syntax'): pre.image_metadata_only(p)


def test_realistic_archived_chain_and_sanitized_inputs(fixture,tmp_path,monkeypatch):
    f=fixture
    manifest=pre.build_manifest(f.checkpoint,f.detector,f.train)
    manifest['label_guard']={'blocked_paths':1,'open_attempts':0}
    assert manifest['sequences']['seq8']['frame_count']==5
    assert manifest['duplicate_screen_scope']==['seq2','seq8','seq9']
    assert manifest['duplicate_screen_image_counts']=={'seq2':1,'seq9':1,'seq8':5}
    assert all('dice' not in row for row in manifest['sequences']['seq8']['frames'])
    folder=tmp_path/'output';folder.mkdir(); save_json(folder/'manifest.json',manifest)
    save_json(folder/'completion.json',{'passed':True,'manifest_sha256':sha(folder/'manifest.json')})
    frames,prompt,prompt_sha,view,mask,provenance=pre.extended_priority_inputs('seq8',folder,f.checkpoint,f.detector)
    assert len(frames)==5 and prompt['first_detection_index']==0 and mask.sum()==6
    assert view=={'kind':'full_source_frame'} and provenance['roi_xyxy']==[0,0,10,8]
    assert prompt_sha==sha(f.folder/'prompt.json')
    f.image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='image identity'):pre.extended_priority_inputs('seq8',folder,f.checkpoint,f.detector)


def test_rejects_changed_archive_or_prompt(fixture):
    f=fixture
    (f.folder/'frames.json').write_text('changed')
    with pytest.raises(ValueError,match='metadata changed'):pre.build_manifest(f.checkpoint,f.detector,f.train)


def test_rejects_empty_and_invalid_prompts(fixture):
    f=fixture
    prompt=json.loads((f.folder/'prompt.json').read_text())
    prompt['points']=[[2,2],[100,100]]
    with pytest.raises(ValueError,match='outside'):pre.validate_prompt(prompt,f.mask,(10,8))
    Image.fromarray(np.zeros((8,10),dtype=np.uint8)).save(f.mask)
    prompt['points']=[[2,2],[3,3]]
    with pytest.raises(ValueError,match='empty'):pre.validate_prompt(prompt,f.mask,(10,8))


def test_rejects_train_validation_or_duplicate_overlap(fixture,monkeypatch):
    f=fixture
    monkeypatch.setattr(pre,'EXPECTED_TRAIN',{'seq8'})
    with pytest.raises(ValueError,match='split or duplicate'):pre.checked_training_preflight(f.train)


def test_rejects_exact_or_decoded_cross_sequence_duplicate():
    seen={}
    pre.register_cross_sequence_image(seen,'hash','seq8')
    pre.register_cross_sequence_image(seen,'hash','seq8')
    with pytest.raises(ValueError,match='duplicates'):pre.register_cross_sequence_image(seen,'hash','seq16')


def test_rejects_changed_config(fixture):
    f=fixture
    save_json(f.config,{'duplicate_screen_sequences':[]})
    with pytest.raises(ValueError,match='duplicate-screen'):pre.checked_training_preflight(f.train)


def test_target_duplicate_of_training_image_is_rejected(fixture, monkeypatch):
    f=fixture
    original=pre.load_manifest
    def changed(path,split):
        data=original(path,split)
        if split=='dev':
            data['seq2'][0].image_path=f.image
        return data
    monkeypatch.setattr(pre,'load_manifest',changed)
    frozen=json.loads((f.train/'manifest.json').read_text())
    frozen['sequences']['seq2']['image_sha256']=[sha(f.image)]
    save_json(f.train/'manifest.json',frozen);save_json(f.train/'completion.json',{'passed':True,'manifest_sha256':sha(f.train/'manifest.json')})
    with pytest.raises(ValueError,match='duplicates'):
        pre.build_manifest(f.checkpoint,f.detector,f.train)


def test_rejects_archive_change_after_preflight(fixture,tmp_path):
    f=fixture
    manifest=pre.build_manifest(f.checkpoint,f.detector,f.train)
    manifest['label_guard']={'blocked_paths':1,'open_attempts':0}
    folder=tmp_path/'out';folder.mkdir();save_json(folder/'manifest.json',manifest)
    save_json(folder/'completion.json',{'passed':True,'manifest_sha256':sha(folder/'manifest.json')})
    (f.folder/'frames.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        pre.load_extended_preflight(folder,f.checkpoint,f.detector)


def test_rejects_frozen_training_image_changed(fixture):
    f=fixture
    image=f.train.parent/'train.png'
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='frozen image differs'):
        pre.build_manifest(f.checkpoint,f.detector,f.train)


def test_rejects_tampered_train_screen_count(fixture,tmp_path):
    f=fixture
    manifest=pre.build_manifest(f.checkpoint,f.detector,f.train)
    manifest['label_guard']={'blocked_paths':1,'open_attempts':0}
    manifest['duplicate_screen_image_counts']['seq2']=2
    folder=tmp_path/'out';folder.mkdir();save_json(folder/'manifest.json',manifest)
    save_json(folder/'completion.json',{'passed':True,'manifest_sha256':sha(folder/'manifest.json')})
    with pytest.raises(ValueError,match='count changed'):
        pre.load_extended_preflight(folder,f.checkpoint,f.detector)


def test_rejects_target_image_changed_while_decoding(fixture,monkeypatch):
    f=fixture
    original=pre.digest
    count=0
    def racing_digest(path):
        nonlocal count
        result=original(path)
        if Path(path)==f.image:
            count+=1
            if count==2: return 'f'*64
        return result
    monkeypatch.setattr(pre,'digest',racing_digest)
    with pytest.raises(ValueError,match='changed while decoding'):
        pre.build_manifest(f.checkpoint,f.detector,f.train)


def test_rejects_resealed_guard_or_noncanonical_checkpoint(fixture,tmp_path):
    f=fixture
    manifest=pre.build_manifest(f.checkpoint,f.detector,f.train)
    folder=tmp_path/'out'; folder.mkdir()
    def seal():
        save_json(folder/'manifest.json',manifest)
        save_json(folder/'completion.json',{'passed':True,'manifest_sha256':sha(folder/'manifest.json')})
    manifest['label_guard']={'blocked_paths':1,'open_attempts':1};seal()
    with pytest.raises(ValueError,match='identity differs'):
        pre.load_extended_preflight(folder,f.checkpoint,f.detector)
    manifest['label_guard']['open_attempts']=0
    alternate=tmp_path/'alternate.pt';alternate.write_bytes(b'alternate')
    manifest['checkpoint_sha256']=sha(alternate);seal()
    with pytest.raises(ValueError,match='identity differs'):
        pre.load_extended_preflight(folder,alternate,f.detector)
