#!/usr/bin/env python3
"""Independently audit two local deletion replays before labels are opened."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_source_group_deletion import partition_history
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for name in ('first','second','capture','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());x=read(a.first/'completion.json');y=read(a.second/'completion.json');u=read(a.capture/'utility_capture_receipt.json');s=read(a.capture/'source_capture_receipt.json')
if x!=y or not x['passed'] or x['capture_sha256']!=digest(a.capture/'source_capture_receipt.json') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_source_group_deletion_v1.json') or x['partition_sha256']!=digest(ROOT/'modeling/lean_source_group_deletion.py') or x['deletion_sha256']!=digest(ROOT/'modeling/lean_utility_group_deletion.py') or x['runner_sha256']!=digest(ROOT/'scripts/replay_lean_source_group_deletion.py') or x['sequence']!=u['sequence'] or not s['passed']:raise ValueError('Replay identity differs')
if [e['frame'] for e in x['events']]!=[e['frame'] for e in s['observations']]:raise ValueError('Frame coverage differs')
count=0
for event in x['events']:
 frame=event['frame'];name=f'{frame:06d}.pt'
 if digest(a.capture/'source_snapshots'/name)!=s['source_snapshots'][name]:raise ValueError('Source snapshot changed')
 src=torch.load(a.capture/'source_snapshots'/name,map_location='cpu',weights_only=True);groups=partition_history(src['sources'],frame,u['sequence'])
 if groups is None:
  if event.get('excluded')!='five eligible 1024-key sources not available':raise ValueError('Incorrect exclusion')
  continue
 if groups!=event['groups'] or len({j for g in groups for j in g})!=5120:raise ValueError('Group assignment differs')
 counts=event['key_counts'];baseline=counts['noop']
 if any(counts[f'group{i}']!=baseline-1024 for i in range(5)):raise ValueError('Key counts differ')
 for arm in ('noop',*[f'group{i}' for i in range(5)]):
  fname=f'{frame:06d}_{arm}.png'
  if any(digest(folder/fname)!=event['masks'][arm] for folder in (a.first,a.second)):raise ValueError('Replay mask differs')
  if arm=='noop' and not np.array_equal(np.asarray(Image.open(a.first/fname))>0,np.asarray(Image.open(a.capture/f'{frame:06d}.png'))>0):raise ValueError('No-op mask differs')
 count+=1
if not count:raise ValueError('No eligible reads to verify')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
result={'passed':True,'sequence':u['sequence'],'eligible_reads':count,'excluded_reads':len(x['events'])-count,'first_sha256':digest(a.first/'completion.json'),'second_sha256':digest(a.second/'completion.json'),'capture_sha256':digest(a.capture/'source_capture_receipt.json'),'verifier_sha256':digest(Path(__file__)),'meaning':'Deterministic local interventions only; target labels unopened.'}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
