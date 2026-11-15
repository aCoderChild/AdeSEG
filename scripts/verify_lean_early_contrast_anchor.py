#!/usr/bin/env python3
"""Verify no-op parity and deterministic active early-contrast token ranking."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('no-op-a','no-op-b','active-a','active-b','reference','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args();ref=read(a.reference/'records.json');runs=[]
 for gate,mode in ((a.no_op_a,'no_op'),(a.no_op_b,'no_op'),(a.active_a,'active_gate'),(a.active_b,'active_gate')):
  z=read(gate/'contrast_intervention.json');r=read(gate/'records.json');done=read(gate/'completion.json')
  if (not z['passed'] or z['mode']!=mode or z['sequence']!='seq3' or not done['passed'] or digest(gate/'records.json')!=done['records_sha256']
      or z['model_records_sha256']!=done['records_sha256'] or z['contrast_events_sha256']!=digest(gate/'contrast_events.json')
      or z['reference_records_sha256']!=digest(a.reference/'records.json') or not read(gate/'parameter_integrity_after.json')['unchanged']
      or z['identity']['runner_sha256']!=digest(ROOT/'scripts/run_lean_early_contrast_anchor.py')
      or z['identity']['adapter_sha256']!=digest(ROOT/'modeling/lean_early_contrast_anchor.py')):raise ValueError('Gate invalid')
  for name in ('positions','selections','memory_events'):
   if digest(gate/f'{name}.json')!=r[name+'_sha256']:raise ValueError('Trace changed')
  for i,h in enumerate(r['mask_sha256']):
   if digest(gate/f'{i:06d}.png')!=h:raise ValueError('Mask changed')
  select=read(gate/'selections.json');events=read(gate/'contrast_events.json')
  if not select or any(x['kept']!=1024 or x['candidates']!=4096 for x in select) or len(events)!=z['score_events'] or any(x['tokens']!=4096 for x in events):raise ValueError('Budget or score events changed')
  if mode=='no_op':
   if z['changed_masks'] or z['changed_selections'] or z['changed_score_events'] or any(r[k]!=ref[k] for k in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256')):raise ValueError('No-op differs')
  elif z['changed_masks']<=0 or not z['changed_selections'] or z['changed_score_events']<=0:raise ValueError('No active effect')
  runs.append((z,r,select))
 if runs[0]!=runs[1] or runs[2]!=runs[3] or runs[0][0]['identity']!=runs[2][0]['identity']:raise ValueError('Replays differ')
 changed=sum(x['indices_sha256']!=y['indices_sha256'] for x,y in zip(runs[0][2],runs[2][2]))
 out=validate_output(a.output)
 if out.exists():raise ValueError('Output exists')
 result={'passed':True,'scope':'Label-free engineering verification, not an efficacy result.','identity':runs[0][0]['identity'],
  'no_op_intervention_sha256':digest(a.no_op_a/'contrast_intervention.json'),'active_intervention_sha256':digest(a.active_a/'contrast_intervention.json'),
  'reference_records_sha256':digest(a.reference/'records.json'),'active_changed_masks':runs[2][0]['changed_masks'],
  'active_changed_selection_events':changed,'selection_events':len(runs[0][2]),'source_sha256':digest(Path(__file__))}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='identity'},indent=2))
if __name__=='__main__':main()
