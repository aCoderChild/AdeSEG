#!/usr/bin/env python3
"""Evaluate a fixed early-FPN contrast blend in existing Lean token ranking."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_early_contrast_anchor import early_contrast_anchor
from scripts.memory_token_io import digest
import scripts.run_lean_indexed_spatial_rope as runner
PROTOCOL=ROOT/'protocols/lean_early_contrast_anchor_v1.json'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint','merged','reference','indexed-reference','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--mode',choices=('no_op','active_gate','study'),required=True);p.add_argument('--sequence',required=True)
 for x in ('no-op-a','no-op-b','active-a','active-b'):p.add_argument('--'+x,type=Path)
 a=p.parse_args();protocol=read(PROTOCOL)
 if (a.mode!='study' and a.sequence!='seq3') or (a.mode=='study' and a.sequence not in protocol['cohort']):raise ValueError('Out-of-scope sequence')
 ref=read(a.indexed_reference/'records.json');ref_done=read(a.indexed_reference/'completion.json')
 if not ref_done['passed'] or digest(a.indexed_reference/'records.json')!=ref_done['records_sha256']:raise ValueError('Reference invalid')
 identity={'runner_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),'adapter_sha256':digest(ROOT/'modeling/lean_early_contrast_anchor.py'),
  'prototype_sha256':digest(ROOT/'modeling/prompt_feature_refinement.py'),'indexed_runner_sha256':digest(ROOT/'scripts/run_lean_indexed_spatial_rope.py'),
  'indexed_adapter_sha256':digest(ROOT/'modeling/lean_indexed_spatial_rope.py'),'cpu_verification_sha256':digest(a.cpu_verification)}
 if a.mode=='study':
  if not all((a.no_op_a,a.no_op_b,a.active_a,a.active_b)):raise ValueError('Two no-op and active gates required')
  pairs=[]
  for x,y,mode in ((a.no_op_a,a.no_op_b,'no_op'),(a.active_a,a.active_b,'active_gate')):
   records=[]
   for g in (x,y):
    z=read(g/'contrast_intervention.json')
    if not z['passed'] or z['identity']!=identity or z['mode']!=mode or z['sequence']!='seq3' or z['model_records_sha256']!=digest(g/'records.json') or z['contrast_events_sha256']!=digest(g/'contrast_events.json'):
     raise ValueError('Engineering replay invalid')
    records.append(z)
   if records[0]!=records[1]:raise ValueError('Engineering replays differ')
   pairs.append(records[0])
  if pairs[0]['changed_masks']!=0 or pairs[0]['changed_selections'] or pairs[1]['changed_masks']<=0 or not pairs[1]['changed_selections'] or not pairs[1]['changed_score_events']:
   raise ValueError('No-op or active intervention is not verified')
 blend=0.0 if a.mode=='no_op' else .5;contrast=[];contexts=[];install=runner.install_multiscale_anchor
 def install_with_contrast(model,strength,selection_events=None):
  events=install(model,strength,selection_events=selection_events)
  ctx=early_contrast_anchor(model,blend,contrast);ctx.__enter__();contexts.append(ctx)
  return events
 runner.install_multiscale_anchor=install_with_contrast
 args=['run_lean_indexed_spatial_rope.py','--mode','gate' if a.mode!='study' else 'candidate','--sequence',a.sequence]
 for key in ('checkpoint','merged','reference','cpu_verification','observed_reference','output'):args.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
 args.extend(['--gate-a',str(a.indexed_gate_a),'--gate-b',str(a.indexed_gate_b)]);sys.argv=args
 try:runner.main()
 finally:
  for ctx in reversed(contexts):ctx.__exit__(None,None,None)
 record=read(a.output/'records.json')
 if a.mode=='no_op':
  for key in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256'):
   if record[key]!=ref[key]:raise ValueError(f'No-op differs: {key}')
 if not contrast:raise ValueError('Contrast branch never received an existing anchor score')
 if any(x['tokens']!=4096 for x in contrast):raise ValueError('Unexpected score-grid size')
 if a.mode=='no_op' and any(x['mean_abs_score_change']!=0 for x in contrast):raise ValueError('No-op altered score')
 changed=sum(x!=y for x,y in zip(record['mask_sha256'],ref['mask_sha256']))
 (a.output/'contrast_events.json').write_text(json.dumps(contrast,indent=2)+'\n')
 result={'passed':True,'mode':a.mode,'sequence':a.sequence,'strength':blend,'identity':identity,'reference_records_sha256':ref_done['records_sha256'],
  'model_records_sha256':digest(a.output/'records.json'),'contrast_events_sha256':digest(a.output/'contrast_events.json'),
  'changed_score_events':sum(x['mean_abs_score_change']>0 for x in contrast),'score_events':len(contrast),
  'changed_masks':changed,'changed_selections':record['selections_sha256']!=ref['selections_sha256']}
 (a.output/'contrast_intervention.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='identity'},indent=2))
if __name__=='__main__':main()
