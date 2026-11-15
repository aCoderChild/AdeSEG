#!/usr/bin/env python3
"""Independently verify frozen Lean snapshot and offline ablation gates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.run_lean_offline_memory_ablation import equal

PROTOCOL=ROOT/'protocols/lean_memory_offline_ablation_v1.json'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('capture-a','capture-b','offline-a','offline-b','reference','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();ref=read(a.reference/'records.json');refdone=read(a.reference/'completion.json')
 if not refdone['passed'] or digest(a.reference/'records.json')!=refdone['records_sha256']:raise ValueError('Reference invalid')
 captures=[];offline=[]
 for capture,run in ((a.capture_a,a.offline_a),(a.capture_b,a.offline_b)):
  c=read(capture/'capture_observation.json');cr=read(capture/'records.json');r=read(run/'completion.json')
  if not c['passed'] or c['sequence']!='seq3' or c['reference_records_sha256']!=refdone['records_sha256'] or c['records_sha256']!=digest(capture/'records.json') or c['protocol_sha256']!=digest(PROTOCOL) or c['source_sha256']!=digest(ROOT/'scripts/capture_lean_memory_reads.py') or c['capture_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py'):
   raise ValueError('Capture identity invalid')
  for key in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256'):
   if cr[key]!=ref[key]:raise ValueError('Capture trajectory differs: '+key)
  for i,h in enumerate(cr['mask_sha256']):
   if digest(capture/f'{i:06d}.png')!=h or digest(a.reference/f'{i:06d}.png')!=h:raise ValueError('Replay mask changed')
  if not read(capture/'parameter_integrity_after.json')['unchanged']:raise ValueError('Capture model changed')
  if not r['passed'] or r['sequence']!='seq3' or r['capture_receipt_sha256']!=digest(capture/'capture_observation.json') or r['source_sha256']!=digest(ROOT/'scripts/run_lean_offline_memory_ablation.py') or not read(run/'parameter_integrity_after.json')['unchanged']:
   raise ValueError('Offline run invalid')
  if [x['frame'] for x in r['events']]!=c['target_frames'] or [x['frame'] for x in c['snapshots']]!=c['target_frames']:raise ValueError('Frame coverage differs')
  for snap,event in zip(c['snapshots'],r['events']):
   if digest(capture/'snapshots'/snap['snapshot_file'])!=snap['sha256']:raise ValueError('Snapshot changed')
   for arm,h in event['masks'].items():
    if digest(run/f"{event['frame']:06d}_{arm}.png")!=h:raise ValueError('Output mask changed')
   if not np.array_equal(np.asarray(Image.open(run/f"{event['frame']:06d}_noop.png"))>0,np.asarray(Image.open(a.reference/f"{event['frame']:06d}.png"))>0):raise ValueError('No-op mask differs')
  captures.append(c);offline.append(r)
 if captures[0]!=captures[1] or offline[0]!=offline[1]:raise ValueError('Independent engineering gates differ')
 for snap in captures[0]['snapshots']:
  x=torch.load(a.capture_a/'snapshots'/snap['snapshot_file'],map_location='cpu',weights_only=True)
  y=torch.load(a.capture_b/'snapshots'/snap['snapshot_file'],map_location='cpu',weights_only=True)
  if not equal(x,y):raise ValueError('Independent tensor snapshots differ')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'passed':True,'sequence':'seq3','frames':captures[0]['target_frames'],'capture_receipt_sha256':digest(a.capture_a/'capture_observation.json'),
         'offline_completion_sha256':digest(a.offline_a/'completion.json'),'protocol_sha256':digest(PROTOCOL),'source_sha256':digest(Path(__file__))}
 (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
