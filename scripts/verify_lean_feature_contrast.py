#!/usr/bin/env python3
"""Verify repeatable label-free FPN observation and exact reference replay."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('gate-a','gate-b','reference','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args();ref=read(a.reference/'records.json');runs=[]
 for gate in (a.gate_a,a.gate_b):
  o=read(gate/'contrast_observation.json');r=read(gate/'records.json');d=read(gate/'completion.json')
  if (not o['passed'] or not o['label_free_gate'] or o['sequence']!='seq3' or not d['passed'] or digest(gate/'records.json')!=d['records_sha256']
      or o['source_sha256']!=digest(ROOT/'scripts/observe_lean_feature_contrast.py') or o['protocol_sha256']!=digest(ROOT/'protocols/lean_early_feature_contrast_diagnostic_v1.json')
      or o['prototype_source_sha256']!=digest(ROOT/'modeling/prompt_feature_refinement.py') or o['reference_records_sha256']!=digest(a.reference/'records.json')
      or not read(gate/'parameter_integrity_after.json')['unchanged']):raise ValueError('Observation invalid')
  for key in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256'):
   if r[key]!=ref[key]:raise ValueError('Reference replay differs')
  for i,h in enumerate(r['mask_sha256']):
   if digest(gate/f'{i:06d}.png')!=h:raise ValueError('Mask changed')
  if [c['frame'] for c in o['captures']]!=o['target_frames']:raise ValueError('Observation coverage differs')
  for c in o['captures']:
   path=gate/'contrast'/f"{c['frame']:06d}.npz"
   if digest(path)!=c['map_sha256']:raise ValueError('Map changed')
   with np.load(path,allow_pickle=False) as arrays:
    if set(arrays.files)!=set(c['arrays']):raise ValueError('Array names changed')
    for name,metadata in c['arrays'].items():
     z=arrays[name]
     if list(z.shape)!=metadata['shape'] or hashlib.sha256(z.tobytes()).hexdigest()!=metadata['sha256'] or not np.isfinite(z).all():raise ValueError('Array changed')
  runs.append(o)
 if runs[0]!=runs[1]:raise ValueError('Observations differ')
 out=validate_output(a.output)
 if out.exists():raise ValueError('Output exists')
 result={'passed':True,'scope':'Label-free observation only; no accuracy claim.','observation_sha256':digest(a.gate_a/'contrast_observation.json'),
  'reference_records_sha256':digest(a.reference/'records.json'),'protocol_sha256':runs[0]['protocol_sha256'],
  'observer_sha256':runs[0]['source_sha256'],'source_sha256':digest(Path(__file__)),'target_frames':runs[0]['target_frames']}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
