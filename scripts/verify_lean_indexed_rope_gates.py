#!/usr/bin/env python3
"""Independently verify label-free position-preserving Lean engineering runs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p): return json.loads(p.read_text())

def main():
 p=argparse.ArgumentParser()
 for name in ('gate-a','gate-b','reference','observation','output'): p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args(); observation=read(a.observation/'position_observation.json'); ref=read(a.reference/'records.json'); runs=[]
 if not observation['passed'] or digest(a.reference/'records.json')!=observation['reference_records_sha256']:
  raise ValueError('Observed reference invalid')
 for gate in (a.gate_a,a.gate_b):
  s,r,d=read(gate/'setup.json'),read(gate/'records.json'),read(gate/'completion.json')
  if (not d['passed'] or s['mode']!='gate' or s['sequence']!='seq3' or digest(gate/'records.json')!=d['records_sha256']
      or not read(gate/'parameter_integrity_after.json')['unchanged'] or r['frames_sha256'] is not None
      or r['reference_records_sha256']!=digest(a.reference/'records.json')): raise ValueError('Gate invalid')
  for key,file in [('positions_sha256','positions.json'),('selections_sha256','selections.json'),('memory_events_sha256','memory_events.json')]:
   if digest(gate/file)!=r[key]: raise ValueError(f'{file} changed')
  for i,h in enumerate(r['mask_sha256']):
   if digest(gate/f'{i:06d}.png')!=h: raise ValueError('Mask changed')
  if r['mask_sha256'][:3]!=ref['mask_sha256'][:3] or r['changed_masks']!=sum(x!=y for x,y in zip(r['mask_sha256'],ref['mask_sha256'])) or r['changed_masks']<=0:
   raise ValueError('Unexpected prefix or ineffective intervention')
  selections=read(gate/'selections.json'); positions=read(gate/'positions.json')
  if not selections or any(x['kept']!=1024 or x['candidates']!=4096 for x in selections): raise ValueError('Selection budget differs')
  if not positions or len(positions)!=observation['attention_calls'] or any(x['query_rotations']!=4 or x['key_rotations']!=4 for x in positions): raise ValueError('Rotary calls differ')
  if any(positions[i]['indices_sha256']!=observation['observations'][i]['position_indices_sha256'] for i in range(2)):
   raise ValueError('First positions differ before intervention')
  runs.append((s,r))
 if runs[0]!=runs[1]: raise ValueError('Replays differ')
 out=validate_output(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists(): raise ValueError('Output exists')
 payload={'passed':True,'scope':'Engineering checks only; no labels, no accuracy or generalization claim.',
  'identity':runs[0][0]['identity'],'gate_records_sha256':digest(a.gate_a/'records.json'),
  'reference_records_sha256':digest(a.reference/'records.json'),'observation_sha256':digest(a.observation/'position_observation.json'),
  'frames':runs[0][1]['frames'],'changed_masks':runs[0][1]['changed_masks'],'identical_prefix_frames':3,
  'position_events':len(read(a.gate_a/'positions.json')),'source_sha256':digest(Path(__file__))}
 out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({k:v for k,v in payload.items() if k!='identity'},indent=2))
if __name__=='__main__':main()
