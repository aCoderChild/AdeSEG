#!/usr/bin/env python3
"""Independently audit repeated cross-trajectory same-read gates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_cross_trajectory_blocks import BLOCKS
from modeling.lean_cross_trajectory_pairs import PAIRS,swap_pair
from scripts.memory_token_io import digest,validate_output
from scripts.run_lean_offline_memory_ablation import equal

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('first','second','raw-capture','fused-capture','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();x=read(a.first/'completion.json');y=read(a.second/'completion.json');r=read(a.raw_capture/'raw_two_capture_receipt.json');f=read(a.fused_capture/'fusion_capture_receipt.json')
if not all(z['passed'] for z in (x,y,r,f)) or x!=y or x['mode']!='candidate' or x['raw_capture_sha256']!=digest(a.raw_capture/'raw_two_capture_receipt.json') or x['fused_capture_sha256']!=digest(a.fused_capture/'fusion_capture_receipt.json') or x['swap_sha256']!=digest(ROOT/'modeling/lean_cross_trajectory_pairs.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_seq23_memory_block_pairs_v1.json') or x['runner_sha256']!=digest(ROOT/'scripts/replay_lean_cross_trajectory_pairs.py') or [e['frame'] for e in x['events']]!=[8,20,35,48,51]:raise ValueError('Repeated swap gates differ')
if digest(a.raw_capture/'prompt.json')!=digest(a.fused_capture/'prompt.json') or digest(a.raw_capture/'prompt.png')!=digest(a.fused_capture/'prompt.png'):raise ValueError('Initial cropped prompt differs')
rs={z['frame']:z for z in r['snapshots']};fs={z['frame']:z for z in f['snapshots']}
for e in x['events']:
 i=e['frame'];paths=[a.raw_capture/'snapshots'/f'{i:06d}.pt',a.fused_capture/'snapshots'/f'{i:06d}.pt']
 if digest(paths[0])!=rs[i]['read_sha256'] or digest(paths[1])!=fs[i]['read_sha256']:raise ValueError('Snapshot changed')
 aa,bb=[torch.load(path,map_location='cpu',weights_only=True)['attention_inputs'] for path in paths]
 if not equal(aa['curr'],bb['curr']) or not equal(aa['curr_pos'],bb['curr_pos']) or not torch.equal(aa['memory'][:4096],bb['memory'][:4096]) or not torch.equal(aa['memory_pos'][:4096],bb['memory_pos'][:4096]):raise ValueError('Query or condition differs')
 ptr=int(aa['num_obj_ptr_tokens']);end=len(aa['memory'])-ptr if ptr else len(aa['memory'])
 if len(aa['memory'])!=len(bb['memory']) or ptr!=int(bb['num_obj_ptr_tokens']) or len(aa['memory'])!=e['total_keys'] or ptr!=e['pointer_keys']:raise ValueError('Key count differs')
 source=torch.load(a.fused_capture/'fusion_sources'/f'{i:06d}.pt',map_location='cpu',weights_only=True)
 if source['start']!=7168 or not torch.equal(source['fused'],bb['memory'][7168:8192]):raise ValueError('Fused block changed')
 for receiver,donor in ((aa,bb),(bb,aa)):
  for name,pair in PAIRS.items():
   swap=swap_pair(receiver,donor,name)
   for key in ('memory','memory_pos'):
    for block,(start,stop) in BLOCKS.items():
     expected=donor if block in pair else receiver
     if not torch.equal(swap[key][start:stop],expected[key][start:stop]):raise ValueError('Incorrect pair segment')
    if not torch.equal(swap[key][:4096],receiver[key][:4096]) or not torch.equal(swap[key][12288:],receiver[key][12288:]):raise ValueError('Condition or pointers changed')
 for arm,sha in e['masks'].items():
  for folder in (a.first,a.second):
   if digest(folder/f'{i:06d}_{arm}.png')!=sha:raise ValueError('Mask changed')
  rot=e['rotary'][arm]
  if rot['spatial_keys']!=end or rot['pointer_keys']!=ptr or rot['query_rotations']!=4 or rot['key_rotations']!=4:raise ValueError('Rotary geometry changed')
 if e['masks']['raw_noop']!=digest(a.raw_capture/f'{i:06d}.png') or e['masks']['fused_noop']!=digest(a.fused_capture/f'{i:06d}.png'):raise ValueError('No-op mask differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'runner_sha256':x['runner_sha256'],'swap_sha256':x['swap_sha256'],'protocol_sha256':x['protocol_sha256'],'frames':[8,20,35,48,51],'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Sequence 23 repeated pair swaps, positions, pointers and no-op masks verified.')
