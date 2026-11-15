#!/usr/bin/env python3
"""Independently verify label-free multimask-prompt full-video replays."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('gate-a','gate-b','reference','prompt-record','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args();ref=read(a.reference/'records.json');prompt=read(a.prompt_record);runs=[]
 for gate in (a.gate_a,a.gate_b):
  s,r,d=read(gate/'setup.json'),read(gate/'records.json'),read(gate/'completion.json')
  if (s['mode']!='gate' or s['sequence']!='seq3' or not d['passed'] or digest(gate/'records.json')!=d['records_sha256']
      or not read(gate/'parameter_integrity_after.json')['unchanged'] or r['frames_sha256'] is not None
      or r['indexed_reference_sha256']!=digest(a.reference/'records.json') or r['prompt_record_sha256']!=digest(a.prompt_record)):raise ValueError('Gate invalid')
  for name in ('positions','selections','memory_events','prompt_events'):
   if digest(gate/f'{name}.json')!=r[name+'_sha256']:raise ValueError('Event trace changed')
  if read(gate/'prompt_events.json')!=prompt['candidate']['decoder_events']:raise ValueError('Prompt decoder differs')
  for i,h in enumerate(r['mask_sha256']):
   if digest(gate/f'{i:06d}.png')!=h:raise ValueError('Mask changed')
  positions=read(gate/'positions.json');selections=read(gate/'selections.json')
  if not positions or any(x['query_rotations']!=4 or x['key_rotations']!=4 for x in positions) or not selections or any(x['candidates']!=4096 or x['kept']!=1024 for x in selections):raise ValueError('Position or budget changed')
  if r['changed_masks']!=sum(x!=y for x,y in zip(r['mask_sha256'],ref['mask_sha256'])) or not r['changed_masks']:raise ValueError('No measured effect')
  runs.append((s,r))
 if runs[0]!=runs[1]:raise ValueError('Independent replays differ')
 out=validate_output(a.output)
 if out.exists():raise ValueError('Output exists')
 payload={'passed':True,'scope':'Label-free engineering verification; not an accuracy result.','identity':runs[0][0]['identity'],
  'records_sha256':digest(a.gate_a/'records.json'),'reference_sha256':digest(a.reference/'records.json'),
  'prompt_record_sha256':digest(a.prompt_record),'frames':runs[0][1]['frames'],'changed_masks':runs[0][1]['changed_masks'],
  'source_sha256':digest(Path(__file__))}
 out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({k:v for k,v in payload.items() if k!='identity'},indent=2))
if __name__=='__main__':main()
