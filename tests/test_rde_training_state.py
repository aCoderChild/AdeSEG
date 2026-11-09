import json
import torch
import pytest
from scripts.rde_training_state import load_checkpoint,publish_checkpoint,append_frame,verify_journal,next_sequence
from scripts.mavos_value_utils import atomic_torch_save


def state(index=0,tip=None):
    return {'identity_sha256':'run','frames_seen':index,'journal_tip':tip,'weights':torch.tensor([index])}


def test_inactive_slot_crash_preserves_committed_checkpoint(tmp_path):
    publish_checkpoint(tmp_path,state())
    # Simulate a crash after the next slot is written, before pointer publish.
    atomic_torch_save(tmp_path/'checkpoint_1.pt',state(1,'new'))
    assert load_checkpoint(tmp_path,'run')['frames_seen']==0
    publish_checkpoint(tmp_path,state(1,'new'))
    assert load_checkpoint(tmp_path,'run')['frames_seen']==1
    with pytest.raises(RuntimeError,match='backward'):publish_checkpoint(tmp_path,state())
    with pytest.raises(RuntimeError,match='journal tip'):publish_checkpoint(tmp_path,state(1,'other'))
    assert load_checkpoint(tmp_path,'run')['frames_seen']==1
    with pytest.raises(RuntimeError,match='identity'):load_checkpoint(tmp_path,'other')
    (tmp_path/'checkpoint_1.pt').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError,match='checksum'):load_checkpoint(tmp_path,'run')


def test_orphan_receipt_must_replay_and_hash_chain_is_verified(tmp_path):
    a=append_frame(tmp_path,'run',1,None,{'loss':1.})
    assert verify_journal(tmp_path,'run',0,None)==1
    assert append_frame(tmp_path,'run',1,None,{'loss':1.})==a
    with pytest.raises(RuntimeError,match='replay'):append_frame(tmp_path,'run',1,None,{'loss':2.})
    b=append_frame(tmp_path,'run',2,a,{'loss':.9})
    assert verify_journal(tmp_path,'run',2,b)==0
    with pytest.raises(RuntimeError,match='coverage'):verify_journal(tmp_path,'run',0,None)
    p=tmp_path/'000001.json';v=json.loads(p.read_text());v['record']['loss']=3;p.write_text(json.dumps(v))
    with pytest.raises(RuntimeError,match='chain'):verify_journal(tmp_path,'run',2,b)


def test_epoch_cursor_covers_all_fixed_sequences_then_terminates():
    cursor={'epoch':0,'sequence_index':0,'next_frame':None};seen=[]
    while cursor['epoch']<3:
        seen.append((cursor['epoch'],cursor['sequence_index']))
        cursor=next_sequence(cursor['epoch'],cursor['sequence_index'],2,3)
    assert seen==[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
    assert cursor=={'epoch':3,'sequence_index':0,'next_frame':None}
