"""Small training transactions using the project's atomic Torch/JSON writers.

Write only the inactive checkpoint slot, then atomically publish latest.json.
A crash cannot overwrite the current committed slot. Scientific frame receipts
form a hash chain; one pre-checkpoint orphan must replay exactly on resume.
"""
import json
from pathlib import Path
import torch
from scripts.mavos_value_utils import atomic_json,atomic_torch_save
from scripts.memory_token_io import digest


def load_checkpoint(directory,identity):
    directory=Path(directory);pointer=directory/'latest.json'
    if not pointer.exists():return None
    meta=json.loads(pointer.read_text())
    if meta['identity_sha256']!=identity or meta['slot'] not in (0,1):raise RuntimeError('Checkpoint identity/slot mismatch')
    path=directory/f"checkpoint_{meta['slot']}.pt"
    if digest(path)!=meta['checkpoint_sha256']:raise RuntimeError('Checkpoint checksum mismatch')
    state=torch.load(path,weights_only=True,map_location='cpu')
    if state['identity_sha256']!=identity or state['frames_seen']!=meta['frames_seen'] or state['journal_tip']!=meta['journal_tip']:
        raise RuntimeError('Checkpoint header differs from its committed pointer')
    return state


def publish_checkpoint(directory,state):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    pointer=directory/'latest.json';slot=0
    if pointer.exists():
        previous=json.loads(pointer.read_text())
        if previous['identity_sha256']!=state['identity_sha256']:raise RuntimeError('Cannot replace another run checkpoint')
        if previous['slot'] not in (0,1) or state['frames_seen']<previous['frames_seen']:
            raise RuntimeError('Checkpoint slot/cursor moved backward')
        if state['frames_seen']==previous['frames_seen'] and state['journal_tip']!=previous['journal_tip']:
            raise RuntimeError('Unchanged checkpoint frame count has a different journal tip')
        slot=1-previous['slot']
    path=directory/f'checkpoint_{slot}.pt';atomic_torch_save(path,state)
    atomic_json(pointer,{'slot':slot,'checkpoint_sha256':digest(path),'identity_sha256':state['identity_sha256'],
        'frames_seen':state['frames_seen'],'journal_tip':state['journal_tip']})


def append_frame(directory,identity,index,previous,row):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{index:06d}.json'
    payload={'identity_sha256':identity,'index':index,'previous_sha256':previous,'record':row}
    if path.exists():
        if json.loads(path.read_text())!=payload:raise RuntimeError('Uncommitted frame did not replay exactly')
    else:atomic_json(path,payload)
    return digest(path)


def verify_journal(directory,identity,count,tip):
    directory=Path(directory);paths=sorted(directory.glob('*.json'));previous=None
    if len(paths)<count or len(paths)>count+1:raise RuntimeError('Journal coverage differs from committed cursor')
    for index,path in enumerate(paths,1):
        if path.name!=f'{index:06d}.json':raise RuntimeError('Journal frame index gap')
        row=json.loads(path.read_text())
        if row['identity_sha256']!=identity or row['index']!=index or row['previous_sha256']!=previous:
            raise RuntimeError('Journal hash chain or identity differs')
        previous=digest(path)
        if index==count and previous!=tip:raise RuntimeError('Committed journal tip differs')
    if count==0 and tip is not None:raise RuntimeError('Initial checkpoint has a nonempty journal tip')
    return len(paths)-count


def next_sequence(epoch,index,length,epochs):
    if not 0<=epoch<epochs or not 0<=index<length:raise RuntimeError('Invalid training sequence cursor')
    index+=1
    if index==length:epoch,index=epoch+1,0
    return {'epoch':epoch,'sequence_index':index,'next_frame':None}
