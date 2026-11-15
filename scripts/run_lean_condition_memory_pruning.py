#!/usr/bin/env python3
"""Evaluate fixed-budget read-time conditioning memory selection in Lean."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_condition_memory_pruning import condition_memory_pruning
from scripts.memory_token_io import digest
import scripts.run_lean_condition_indexed_engine as runner
PROTOCOL=ROOT/'protocols/lean_condition_memory_pruning_v1.json'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint','merged','reference','indexed-reference','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--mode',choices=('gate','study'),required=True);p.add_argument('--strategy',choices=('full','usage','uniform'),required=True);p.add_argument('--sequence',required=True)
 for x in ('full-a','full-b','usage-a','usage-b','uniform-a','uniform-b'):p.add_argument('--'+x,type=Path)
 a=p.parse_args();protocol=read(PROTOCOL)
 if (a.mode=='gate' and a.sequence!='seq3') or (a.mode=='study' and (a.sequence not in protocol['cohort'] or a.strategy=='full')):raise ValueError('Out-of-scope sequence or strategy')
 ref=read(a.indexed_reference/'records.json');ref_done=read(a.indexed_reference/'completion.json')
 if not ref_done['passed'] or digest(a.indexed_reference/'records.json')!=ref_done['records_sha256']:raise ValueError('Reference invalid')
 identity={'runner_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),'adapter_sha256':digest(ROOT/'modeling/lean_condition_memory_pruning.py'),
  'engine_sha256':digest(ROOT/'scripts/run_lean_condition_indexed_engine.py'), 'indexed_runner_sha256':digest(ROOT/'scripts/run_lean_indexed_spatial_rope.py'),
  'indexed_adapter_sha256':digest(ROOT/'modeling/lean_indexed_spatial_rope.py'),'cpu_verification_sha256':digest(a.cpu_verification)}
 if a.mode=='study':
  if not all((a.full_a,a.full_b,a.usage_a,a.usage_b,a.uniform_a,a.uniform_b)):raise ValueError('Two replays per strategy required')
  pairs={}
  for x,y,strategy in ((a.full_a,a.full_b,'full'),(a.usage_a,a.usage_b,'usage'),(a.uniform_a,a.uniform_b,'uniform')):
   records=[]
   for g in (x,y):
    z=read(g/'condition_intervention.json')
    if not z['passed'] or z['identity']!=identity or z['mode']!='gate' or z['strategy']!=strategy or z['sequence']!='seq3' or z['model_records_sha256']!=digest(g/'records.json') or z['condition_events_sha256']!=digest(g/'condition_events.json'):raise ValueError('Engineering replay invalid')
    records.append(z)
   if records[0]!=records[1]:raise ValueError('Engineering replays differ')
   pairs[strategy]=records[0]
  if pairs['full']['changed_masks'] or pairs['full']['changed_selections'] or pairs['usage']['changed_masks']<=0 or pairs['uniform']['changed_masks']<=0 or pairs['usage']['indices_sha256']==pairs['uniform']['indices_sha256']:
   raise ValueError('No-op or active interventions not verified')
 events=[];contexts=[];install=runner.install_multiscale_anchor
 def install_with_condition(model,strength,selection_events=None):
  memory_events=install(model,strength,selection_events=selection_events);ctx=condition_memory_pruning(model,a.strategy,events);ctx.__enter__();contexts.append(ctx);return memory_events
 runner.install_multiscale_anchor=install_with_condition
 args=['run_lean_indexed_spatial_rope.py','--mode','gate' if a.mode=='gate' else 'candidate','--sequence',a.sequence]
 for key in ('checkpoint','merged','reference','cpu_verification','observed_reference','output'):args.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
 args.extend(['--gate-a',str(a.indexed_gate_a),'--gate-b',str(a.indexed_gate_b)]);sys.argv=args
 try:runner.main()
 finally:
  for ctx in reversed(contexts):ctx.__exit__(None,None,None)
 record=read(a.output/'records.json');selection=[x for x in events if x['event']=='selection'];reads=[x for x in events if x['event']=='conditioning_read']
 prompt_folder=Path(read(a.merged/'locations.json')[a.sequence])/a.sequence
 first=read(prompt_folder/'prompt.json')['first_detection_index']
 if record['mask_sha256'][:first+2]!=ref['mask_sha256'][:first+2]:raise ValueError('Masks changed before the first compressed read')
 if len(selection)!=1 or not reads:raise ValueError('Conditioning selection not observed')
 expected=4096 if a.strategy=='full' else 1024
 if selection[0]['retained_tokens']!=expected or reads[0]['tokens']!=4096 or any(x['tokens']!=expected for x in reads[1:]):raise ValueError('Conditioning budget differs')
 if a.strategy=='full':
  for key in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256'):
   if record[key]!=ref[key]:raise ValueError(f'Full-retention no-op differs: {key}')
 changed=sum(x!=y for x,y in zip(record['mask_sha256'],ref['mask_sha256']))
 (a.output/'condition_events.json').write_text(json.dumps(events,indent=2)+'\n')
 z={'passed':True,'mode':a.mode,'strategy':a.strategy,'sequence':a.sequence,'identity':identity,'reference_records_sha256':ref_done['records_sha256'],
  'model_records_sha256':digest(a.output/'records.json'),'condition_events_sha256':digest(a.output/'condition_events.json'),
  'changed_masks':changed,'changed_selections':record['selections_sha256']!=ref['selections_sha256'],
  'indices_sha256':selection[0]['selected_indices_sha256'],'retained_tokens':expected,'condition_reads':len(reads),
  'selected_usage_fraction':selection[0]['selected_usage_fraction_of_conditioning']}
 (a.output/'condition_intervention.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps({k:v for k,v in z.items() if k!='identity'},indent=2))
if __name__=='__main__':main()
