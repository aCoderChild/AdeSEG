#!/usr/bin/env python3
"""Independently verify deterministic block swaps and saved gate outputs."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_cross_trajectory_blocks import BLOCKS
from modeling.lean_cross_trajectory_pairs import PAIRS,swap_pair
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('first','second','raw-capture','fused-capture','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text())
x=read(a.first/'completion.json');y=read(a.second/'completion.json')
if x!=y or not x['passed'] or x['mode']!='gate' or [e['frame'] for e in x['events']]!=[8,12]:raise ValueError('Gate repeats differ')
if x['runner_sha256']!=digest(ROOT/'scripts/replay_lean_cross_trajectory_pairs.py') or x['swap_sha256']!=digest(ROOT/'modeling/lean_cross_trajectory_pairs.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_seq23_memory_block_pairs_v1.json'):raise ValueError('Code or protocol changed')
for e in x['events']:
 i=e['frame'];raw=torch.load(a.raw_capture/'snapshots'/f'{i:06d}.pt',map_location='cpu',weights_only=True)['attention_inputs'];fused=torch.load(a.fused_capture/'snapshots'/f'{i:06d}.pt',map_location='cpu',weights_only=True)['attention_inputs'];source=torch.load(a.fused_capture/'fusion_sources'/f'{i:06d}.pt',map_location='cpu',weights_only=True)
 if source['start']!=7168 or not torch.equal(source['fused'],fused['memory'][7168:8192]):raise ValueError('Fused segment mismatch')
 for recipient,donor in ((raw,fused),(fused,raw)):
  for name,pair in PAIRS.items():
   swapped=swap_pair(recipient,donor,name)
   for k in ('memory','memory_pos'):
    for block,(start,end) in BLOCKS.items():
     expected=donor if block in pair else recipient
     if not torch.equal(swapped[k][start:end],expected[k][start:end]):raise ValueError('Incorrect pair segment')
    if not torch.equal(swapped[k][:4096],recipient[k][:4096]) or not torch.equal(swapped[k][12288:],recipient[k][12288:]):raise ValueError('Condition or pointers changed')
 for arm,sha in e['masks'].items():
  if any(digest(folder/f'{i:06d}_{arm}.png')!=sha for folder in (a.first,a.second)):raise ValueError('Output changed')
  rot=e['rotary'][arm]
  if rot['spatial_keys']!=12288 or rot['pointer_keys']!=e['pointer_keys'] or rot['query_rotations']!=4 or rot['key_rotations']!=4:raise ValueError('Rotary geometry changed')
 for name,folder in (('raw',a.raw_capture),('fused',a.fused_capture)):
  if e['masks'][name+'_noop']!=digest(folder/f'{i:06d}.png'):raise ValueError('No-op differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'runner_sha256':x['runner_sha256'],'swap_sha256':x['swap_sha256'],'protocol_sha256':x['protocol_sha256'],'verifier_sha256':digest(Path(__file__)),'frames':[8,12]},indent=2)+'\n')
print('Repeated pair-swap gates verified')
