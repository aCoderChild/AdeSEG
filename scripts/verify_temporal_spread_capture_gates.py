#!/usr/bin/env python3
"""Verify two raw-two observer replays against archived seq3 output."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.run_lean_offline_memory_ablation import equal

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('first','second','reference','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();x=read(a.first/'spread_capture_receipt.json');y=read(a.second/'spread_capture_receipt.json');reference=read(a.reference/'token_window_observation.json')
if not x['passed'] or x!=y or x['entry']!='gate' or x['mode']!='gate' or x['targets']!=[8,12] or x['reference_observation_sha256']!=digest(a.reference/'token_window_observation.json') or x['runner_sha256']!=digest(ROOT/'scripts/capture_lean_temporal_spread_reads.py') or x['capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_temporal_slot_rank_diagnostic_v1.json') or not reference['passed']:raise ValueError('Capture gate differs')
for item in x['snapshots']:
 i=item['frame'];first=a.first/'snapshots'/f'{i:06d}.pt';second=a.second/'snapshots'/f'{i:06d}.pt'
 if digest(first)!=item['read_sha256'] or digest(second)!=item['read_sha256'] or not equal(torch.load(first,map_location='cpu',weights_only=True),torch.load(second,map_location='cpu',weights_only=True)):raise ValueError('Snapshots differ')
for name in ('records.json','positions.json','selections.json','memory_events.json','token_window_events.json'):
 if digest(a.first/name)!=digest(a.reference/name) or digest(a.second/name)!=digest(a.reference/name):raise ValueError('Reference trace differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'runner_sha256':x['runner_sha256'],'capture_sha256':x['capture_sha256'],'protocol_sha256':x['protocol_sha256'],'first_sha256':digest(a.first/'spread_capture_receipt.json'),'second_sha256':digest(a.second/'spread_capture_receipt.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Spread seq3 read-only captures reproduced exactly.')
