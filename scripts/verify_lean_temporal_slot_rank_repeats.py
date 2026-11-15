#!/usr/bin/env python3
"""Independently reconstruct temporal-slot substitutions and audit repeated gates."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_position_lookup import MemoryPositionLookup
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_lean_temporal_spread_window import FORK

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('first','second','capture','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();x=read(a.first/'completion.json');y=read(a.second/'completion.json');c=read(a.capture/'spread_capture_receipt.json')
if x!=y or not x['passed'] or x['mode']!='candidate' or c['entry']!='crop' or x['capture_receipt_sha256']!=digest(a.capture/'spread_capture_receipt.json') or x['checkpoint_sha256']!=digest(a.checkpoint) or x['runner_sha256']!=digest(ROOT/'scripts/replay_lean_temporal_slot_rank.py') or x['module_sha256']!=digest(ROOT/'modeling/lean_temporal_rank_positions.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_temporal_slot_rank_diagnostic_v1.json') or [e['frame'] for e in x['events']]!=[8,20,35,48,51]:raise ValueError('Replay provenance or repeat differs')
torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK)
with torch.inference_mode():
 grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1)[:,0,:].contiguous();offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).cpu();lookup=MemoryPositionLookup(grid,offsets)
 for e in x['events']:
  frame=e['frame'];snapshot_path=a.capture/'snapshots'/f'{frame:06d}.pt';expected=next(z['read_sha256'] for z in c['snapshots'] if z['frame']==frame)
  if digest(snapshot_path)!=expected:raise ValueError('Snapshot hash changed')
  snapshot=torch.load(snapshot_path,map_location='cpu',weights_only=True);args=snapshot['attention_inputs'];pos=args['memory_pos'];ptr=int(args['num_obj_ptr_tokens']);idx=lookup.decode(pos,ptr);hist=e['history'];start=4096+e['insurance_tokens'];end=len(pos)-ptr
  if len(pos)!=e['total_keys'] or ptr!=e['pointer_keys'] or start+sum(z[1] for z in hist)!=end:raise ValueError('Memory geometry differs')
  for mode in ('noop','rank','permuted'):
   expected=pos.clone();cursor=start;slots=[]
   for j,(source,n) in enumerate(hist):
    actual=min(frame-source,6)-1;slot=actual if mode=='noop' else (len(hist)-1-j if mode=='rank' else (j+1 if j<len(hist)-1 else 0))
    spatial=grid[idx[cursor:cursor+n]]
    if not torch.equal(pos[cursor:cursor+n],(spatial+offsets[actual]).unsqueeze(1)):raise ValueError('Source temporal position differs')
    expected[cursor:cursor+n]=(spatial+offsets[slot]).unsqueeze(1);slots.append(slot);cursor+=n
   if hashlib.sha256(expected.contiguous().numpy().tobytes()).hexdigest()!=e['position_sha256'][mode] or slots!=[z['assigned_slot'] for z in e['segments'][mode]]:raise ValueError('Substituted positions differ')
   rot=e['rotary'][mode]
   if rot['spatial_keys']!=end or rot['pointer_keys']!=ptr or rot['query_rotations']!=4 or rot['key_rotations']!=4 or rot['indices_sha256']!=e['rotary']['noop']['indices_sha256']:raise ValueError('Spatial rotary geometry differs')
   for folder in (a.first,a.second):
    if digest(folder/f'{frame:06d}_{mode}.png')!=e['masks'][mode]:raise ValueError('Mask changed')
  if e['masks']['noop']!=digest(a.capture/f'{frame:06d}.png') or sorted(z['assigned_slot'] for z in e['segments']['rank'])!=sorted(z['assigned_slot'] for z in e['segments']['permuted']):raise ValueError('No-op or slot multiset differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'runner_sha256':x['runner_sha256'],'module_sha256':x['module_sha256'],'protocol_sha256':x['protocol_sha256'],'verifier_sha256':digest(Path(__file__)),'frames':[8,20,35,48,51]},indent=2)+'\n');print('Repeated sequence 23 masks, exact position tensors and spatial indices verified.')
