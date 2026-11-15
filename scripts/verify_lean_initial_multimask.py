#!/usr/bin/env python3
"""Independently verify the label-free native multimask prompt replays."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p): return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('gate-a','gate-b','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args(); runs=[]
 for gate in (a.gate_a,a.gate_b):
  setup,done,records=read(gate/'setup.json'),read(gate/'completion.json'),read(gate/'records.json')
  if setup['mode']!='gate' or not done['passed'] or digest(gate/'records.json')!=done['records_sha256'] or len(records)!=1 or records[0]['sequence']!='seq3': raise ValueError('Invalid gate')
  item=records[0]; folder=gate/'seq3'
  if item!=read(folder/'record.json') or any(not read(folder/f'parameter_integrity_{c}.json')['unchanged'] for c in ('baseline','candidate')): raise ValueError('Record or parameters changed')
  for condition in ('baseline','candidate'):
   for name in ('provisional','refined'):
    if digest(folder/f'{condition}_{name}.png')!=item[condition][name+'_sha256']: raise ValueError('Mask changed')
  events=item['candidate']['decoder_events']
  if not events or any(len(x['predicted_iou'])!=3 or x['selected']!=max(range(3),key=lambda i:x['predicted_iou'][i]) for x in events): raise ValueError('Wrong decoder selection')
  if item['baseline']['decoder_events']: raise ValueError('Baseline was modified')
  runs.append((setup['identity'],records))
 if runs[0]!=runs[1]: raise ValueError('Replays differ')
 out=validate_output(a.output)
 if out.exists(): raise ValueError('Output exists')
 payload={'passed':True,'scope':'Label-free engineering verification only.', 'identity':runs[0][0], 'records_sha256':digest(a.gate_a/'records.json'),
  'baseline_refined_sha256':runs[0][1][0]['baseline']['refined_sha256'], 'candidate_refined_sha256':runs[0][1][0]['candidate']['refined_sha256'],
  'candidate_decoder_calls':len(runs[0][1][0]['candidate']['decoder_events']),'source_sha256':digest(Path(__file__))}
 out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({k:v for k,v in payload.items() if k!='identity'},indent=2))
if __name__=='__main__':main()
