import copy,json
from pathlib import Path
import pytest
from PIL import Image
from scripts.preflight_source_group_capture_v1 import CONFIG,verify_config,decoded_sha,validate_replay_identity,validate_feature_pipeline,read_frame_source_projection,ensure_distinct_paths


def test_config_excludes_priority_and_enforces_source_group_partition():
    config=json.loads(CONFIG.read_text());verify_config(config)
    bad=copy.deepcopy(config);bad['sequences']['seq9']['source_group_hypothesis']='prefix1'
    with pytest.raises(ValueError):verify_config(bad)
    bad=copy.deepcopy(config);bad['sequences']['seq4']['targets']=[0]
    with pytest.raises(ValueError,match='capture indices'):verify_config(bad)


def test_decoded_digest_ignores_file_encoding_but_checks_pixels(tmp_path):
    image=Image.new('RGB',(3,4),(10,20,30));a=tmp_path/'a.png';b=tmp_path/'b.bmp'
    image.save(a);image.save(b)
    assert decoded_sha(a)==decoded_sha(b)
    image.putpixel((1,1),(11,20,30));image.save(b)
    assert decoded_sha(a)!=decoded_sha(b)


def test_sampling_protected_and_receipt_binding():
    config=json.loads(CONFIG.read_text());bad=copy.deepcopy(config)
    bad['sequences']['seq2']['targets'][1]+=1
    with pytest.raises(ValueError):verify_config(bad)
    bad=copy.deepcopy(config);bad['protected_sequences'].remove('seq20')
    with pytest.raises(ValueError):verify_config(bad)
    receipt={'passed':True,'device':'mps','sequence':'seq4','prompt_frame':9,'checkpoint_sha256':'a'}
    expected={'sequence':'seq4','prompt_frame':9,'checkpoint_sha256':'a'}
    validate_replay_identity(receipt,expected,'h','h','mps')
    receipt['prompt_frame']=10
    with pytest.raises(ValueError):validate_replay_identity(receipt,expected,'h','h','mps')
    with pytest.raises(ValueError):validate_replay_identity(receipt,expected,'h','other','mps')


def test_pipeline_hash_verifies_actual_structure():
    import hashlib
    pipeline={'x':1};h=hashlib.sha256(json.dumps(pipeline,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    ref={'pipeline':pipeline,'feature_pipeline_sha256':h};validate_feature_pipeline(ref)
    ref['pipeline']['x']=2
    with pytest.raises(ValueError):validate_feature_pipeline(ref)


def test_frame_projection_ignores_ground_truth_fields(tmp_path):
    import json
    path=tmp_path/'frames.json';path.write_text(json.dumps([{'index':0,'source_frame_index':12,'image_sha256':'a'*64,'present':True,'dice':0.3}]))
    assert read_frame_source_projection(path)==[{'index':0,'source_frame_index':12,'image_sha256':'a'*64}]
    path.write_text('[{"index":0}]')
    with pytest.raises(ValueError):read_frame_source_projection(path)


def test_predeclared_sequence_roles_cannot_be_swapped():
    config=json.loads(CONFIG.read_text());bad=copy.deepcopy(config)
    bad['sequences']['seq5']['split']='validation';bad['sequences']['seq15']['split']='train'
    with pytest.raises(ValueError,match='roles'):verify_config(bad)


def test_repeat_receipt_paths_cannot_alias(tmp_path):
    a=tmp_path/'a';a.write_text('receipt');b=tmp_path/'b';b.write_text('receipt')
    ensure_distinct_paths(a,b)
    alias=tmp_path/'alias';alias.symlink_to(a)
    with pytest.raises(ValueError,match='alias'):ensure_distinct_paths(a,alias)
