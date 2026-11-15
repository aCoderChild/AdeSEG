#!/usr/bin/env python3
"""Independently verify the repeated label-free retention engineering gate."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('capture','first','second','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();cap=read(a.capture/'region_capture_receipt.json');x=read(a.first/'completion.json');y=read(a.second/'completion.json')
if not cap['passed'] or not x['passed'] or x!=y or x['capture_sha256']!=digest(a.capture/'region_capture_receipt.json') or x['runner_sha256']!=digest(ROOT/'scripts/gate_lean_retention_1024.py') or x['retention_sha256']!=digest(ROOT/'modeling/lean_fixed_read_retention.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_channel_retention_1024_v1.json'):raise ValueError('Gate provenance or repetition changed')
snaps={e['frame']:e for e in cap['snapshots']}
if {e['frame'] for e in x['events']}!=set(snaps):raise ValueError('Coverage changed')
methods={'usage','contrast','distributed','three_region','foreground','shuffled_contrast'}
for e in x['events']:
 f=e['frame'];s=a.capture/'source_snapshots'/f'{f:06d}.pt';snap=a.capture/'snapshots'/f'{f:06d}.pt'
 if digest(s)!=snaps[f]['sources_sha256'] or digest(snap)!=snaps[f]['read_sha256']:raise ValueError('Snapshot changed')
 src=torch.load(s,map_location='cpu',weights_only=True);eligible=set()
 for v in src['sources']:eligible.update(range(v['start'],v['start']+len(v['token_ids'])))
 if len(eligible)!=5120 or set(e['deleted'])!=methods:raise ValueError('Scope changed')
 for ids in e['deleted'].values():
  if len(ids)!=4096 or len(set(ids))!=4096 or not set(ids)<=eligible:raise ValueError('Deletion complement invalid')
 for name,h in e['masks'].items():
  for folder in (a.first,a.second):
   if digest(folder/f'{f:06d}_{name}.png')!=h:raise ValueError('Mask changed')
 if e['masks']['noop']!=digest(a.capture/f'{f:06d}.png'):raise ValueError('No-op differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'capture_sha256':digest(a.capture/'region_capture_receipt.json'),'checkpoint_sha256':read(a.capture/'pool_observation.json')['identity']['checkpoint_sha256'],'retention_sha256':digest(ROOT/'modeling/lean_fixed_read_retention.py'),'protocol_sha256':digest(ROOT/'protocols/lean_channel_retention_1024_v1.json'),'runner_sha256':digest(ROOT/'scripts/gate_lean_retention_1024.py'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n')
print('Independent seq3 retention gates passed on both captured reads.')
