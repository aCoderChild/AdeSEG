#!/usr/bin/env python3
"""Independently verify repeatable read-only Lean attention observation."""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('gate-a','gate-b','reference','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args();ref=read(a.reference/'records.json');runs=[]
 for gate in (a.gate_a,a.gate_b):
  o=read(gate/'attention_observation.json');r=read(gate/'records.json');done=read(gate/'completion.json');events=read(gate/'attention_events.json')
  if (not o['passed'] or not o['label_free_gate'] or o['sequence']!='seq3' or not done['passed'] or digest(gate/'records.json')!=done['records_sha256']
      or o['source_sha256']!=digest(ROOT/'scripts/observe_lean_attention_mass.py') or o['protocol_sha256']!=digest(ROOT/'protocols/lean_attention_mass_diagnostic_v1.json')
      or o['summary_source_sha256']!=digest(ROOT/'modeling/lean_attention_mass.py') or o['reference_records_sha256']!=digest(a.reference/'records.json')
      or o['attention_events_sha256']!=digest(gate/'attention_events.json') or len(events)!=o['events'] or not read(gate/'parameter_integrity_after.json')['unchanged']):raise ValueError('Observation invalid')
  for key in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256'):
   if r[key]!=ref[key]:raise ValueError('Reference replay differs')
  for i,h in enumerate(r['mask_sha256']):
   if digest(gate/f'{i:06d}.png')!=h:raise ValueError('Mask changed')
  if sum(x['latest_full_memory'] is not None for x in events)!=o['selection_events']:raise ValueError('Selection event count differs')
  for e in events:
   fractions=[e['conditioning_mass'],e['other_spatial_mass'],e['pointer_mass']]
   if e['conditioning_keys']!=4096 or any(not math.isfinite(x) or x<0 or x>1 for x in fractions) or abs(sum(fractions)-1)>1e-5:raise ValueError('Usage mass invalid')
   latest=e['latest_full_memory']
   if latest is not None and (latest['end']-latest['start']!=4096 or latest['start']<4096 or latest['end']!=e['keys']-e['pointer_keys'] or not 0<=latest['retained_mass_at_this_read']<=latest['mass_before_pruning']<=1):raise ValueError('Latest segment invalid')
  runs.append(o)
 if runs[0]!=runs[1]:raise ValueError('Observations differ')
 out=validate_output(a.output)
 if out.exists():raise ValueError('Output exists')
 z={'passed':True,'scope':'Label-free attention observation; not a causal ablation.', 'observation_sha256':digest(a.gate_a/'attention_observation.json'),
    'reference_records_sha256':digest(a.reference/'records.json'),'protocol_sha256':runs[0]['protocol_sha256'],'observer_sha256':runs[0]['source_sha256'],
    'events_sha256':runs[0]['attention_events_sha256'],'source_sha256':digest(Path(__file__)),'events':runs[0]['events'],'selection_events':runs[0]['selection_events']}
 out.write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z,indent=2))
if __name__=='__main__':main()
