#!/usr/bin/env python3
"""Independently verify repeated read-only capture on seq3."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.run_lean_offline_memory_ablation import equal

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('first','second','snapshot-gate','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();x=read(a.first/'condition_validation_capture.json');y=read(a.second/'condition_validation_capture.json');old=read(a.snapshot_gate/'region_capture_receipt.json')
if not x['passed'] or x!=y or x['mode']!='gate' or x['sequence']!='seq3' or x['targets']!=old['targets'] or x['runner_sha256']!=digest(ROOT/'scripts/capture_lean_condition_validation.py') or x['memory_capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_condition_region_validation_v1.json'):raise ValueError('Gate receipt changed')
for e in x['snapshots']:
 i=e['frame'];path=a.first/'snapshots'/f'{i:06d}.pt';path2=a.second/'snapshots'/f'{i:06d}.pt';old_entry=next(z for z in old['snapshots'] if z['frame']==i);ref=a.snapshot_gate/'snapshots'/f'{i:06d}.pt'
 if digest(path)!=e['read_sha256'] or digest(path2)!=e['read_sha256'] or digest(ref)!=old_entry['read_sha256'] or not equal(torch.load(path,map_location='cpu',weights_only=True),torch.load(ref,map_location='cpu',weights_only=True)):raise ValueError('Captured read changed')
for directory in (a.first,a.second):
 if digest(directory/'records.json')!=x['records_sha256']:raise ValueError('Records changed')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'capture_runner_sha256':x['runner_sha256'],'memory_capture_sha256':x['memory_capture_sha256'],'preflight_sha256':x['preflight_sha256'],'protocol_sha256':x['protocol_sha256'],'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Two seq3 observer runs reproduce previous full reads exactly.')
