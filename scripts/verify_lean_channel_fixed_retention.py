#!/usr/bin/env python3
"""Audit repeated frozen-read retention outputs, counts and RoPE receipts."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('capture','scores-a','scores-b','retention-gate','first','second','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();cap=read(a.capture/'priority_capture_receipt.json');sa=read(a.scores_a/'completion.json');sb=read(a.scores_b/'completion.json');gate=read(a.retention_gate/'verification.json');x=read(a.first/'completion.json');y=read(a.second/'completion.json')
if not all(v['passed'] for v in (cap,sa,sb,gate,x,y)) or sa!=sb or x!=y or x['capture_sha256']!=digest(a.capture/'priority_capture_receipt.json') or x['score_receipt_sha256']!=digest(a.scores_a/'completion.json') or x['score_repeat_sha256']!=digest(a.scores_b/'completion.json') or x['retention_gate_sha256']!=digest(a.retention_gate/'verification.json') or x['runner_sha256']!=digest(ROOT/'scripts/replay_lean_channel_fixed_retention.py') or x['retention_sha256']!=digest(ROOT/'modeling/lean_fixed_read_retention.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_channel_fixed_retention_v1.json'):raise ValueError('Receipts or implementation changed')
methods={'usage','contrast','distributed','three_region','foreground','shuffled_contrast','rgb_contrast','prompt_conditioned','descriptor_independent','mismatched_prompt'}
snaps={e['frame']:e for e in cap['snapshots']}
if {e['frame'] for e in x['events']}!=set(snaps):raise ValueError('Frame coverage changed')
for e in x['events']:
 f=e['frame'];s=a.capture/'source_snapshots'/f'{f:06d}.pt';snap=a.capture/'snapshots'/f'{f:06d}.pt'
 if digest(s)!=snaps[f]['sources_sha256'] or digest(snap)!=snaps[f]['read_sha256']:raise ValueError('Snapshot changed')
 src=torch.load(s,map_location='cpu',weights_only=True);data=torch.load(snap,map_location='cpu',weights_only=True);args=data['attention_inputs'];ptr=int(args['num_obj_ptr_tokens']);eligible=set()
 for v in src['sources']:
  if v['frame']>=f-1:raise ValueError('Newest source included')
  eligible.update(range(v['start'],v['start']+len(v['token_ids'])))
 if len(eligible)!=5120 or set(e['kept'])!=methods or set(e['deleted'])!=methods or set(e['masks'])!=methods|{'noop'}:raise ValueError('Scope changed')
 total=int(args['memory'].shape[0])
 for name in methods:
  kept=e['kept'][name];deleted=e['deleted'][name]
  if len(kept)!=4096 or len(set(kept))!=4096 or len(deleted)!=1024 or len(set(deleted))!=1024 or set(kept)&set(deleted) or set(kept)|set(deleted)!=eligible:raise ValueError('Retention complement changed')
  if e['key_counts'][name]!=total-1024:raise ValueError('Key count changed')
  rope=e['rotary'][name]
  if rope['spatial_keys']!=total-1024-ptr or rope['pointer_keys']!=ptr or rope['query_rotations']!=4 or rope['key_rotations']!=4:raise ValueError('RoPE alignment receipt changed')
 if e['key_counts']['noop']!=total or e['rotary']['noop']['spatial_keys']!=total-ptr or e['rotary']['noop']['pointer_keys']!=ptr:raise ValueError('No-op attention geometry changed')
 for name,h in e['masks'].items():
  for folder in (a.first,a.second):
   if digest(folder/f'{f:06d}_{name}.png')!=h:raise ValueError('Mask differs')
 if e['masks']['noop']!=digest(a.capture/f'{f:06d}.png'):raise ValueError('No-op differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'sequence':cap['sequence'],'frames':cap['targets'],'first_sha256':digest(a.first/'completion.json'),'second_sha256':digest(a.second/'completion.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Retention complements, attention geometry, RoPE receipts and repeated masks verified.')
