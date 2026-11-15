#!/usr/bin/env python3
"""Independently validate conditioning-memory no-op and matched-budget replays."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('full-a','full-b','usage-a','usage-b','uniform-a','uniform-b','reference','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args();ref=read(a.reference/'records.json');refpos=read(a.reference/'positions.json');runs={}
 for x,y,strategy in ((a.full_a,a.full_b,'full'),(a.usage_a,a.usage_b,'usage'),(a.uniform_a,a.uniform_b,'uniform')):
  pair=[]
  for gate in (x,y):
   z=read(gate/'condition_intervention.json');r=read(gate/'records.json');done=read(gate/'completion.json');setup=read(gate/'setup.json');events=read(gate/'condition_events.json')
   if (not z['passed'] or z['mode']!='gate' or z['strategy']!=strategy or z['sequence']!='seq3' or not done['passed']
       or digest(gate/'records.json')!=done['records_sha256'] or z['model_records_sha256']!=done['records_sha256']
       or z['condition_events_sha256']!=digest(gate/'condition_events.json') or z['reference_records_sha256']!=digest(a.reference/'records.json')
       or z['identity']['runner_sha256']!=digest(ROOT/'scripts/run_lean_condition_memory_pruning.py')
       or z['identity']['adapter_sha256']!=digest(ROOT/'modeling/lean_condition_memory_pruning.py')
       or z['identity']['engine_sha256']!=digest(ROOT/'scripts/run_lean_condition_indexed_engine.py')
       or setup['engine_variant_sha256']!=z['identity']['engine_sha256'] or not read(gate/'parameter_integrity_after.json')['unchanged']):raise ValueError('Invalid gate')
   for key in ('positions','selections','memory_events'):
    if digest(gate/f'{key}.json')!=r[key+'_sha256']:raise ValueError('Trace changed')
   for i,h in enumerate(r['mask_sha256']):
    if digest(gate/f'{i:06d}.png')!=h:raise ValueError('Mask changed')
   if r['mask_sha256'][:2]!=ref['mask_sha256'][:2] or z['changed_masks']!=sum(x!=y for x,y in zip(r['mask_sha256'],ref['mask_sha256'])):raise ValueError('Causal prefix or mask count invalid')
   positions=read(gate/'positions.json');sels=read(gate/'selections.json')
   if len(positions)!=len(refpos) or any(x['pointer_keys']!=y['pointer_keys'] for x,y in zip(positions,refpos)) or any(x['candidates']!=4096 or x['kept']!=1024 for x in sels):raise ValueError('Pointer or noncondition budget differs')
   selected=[e for e in events if e['event']=='selection'];reads=[e for e in events if e['event']=='conditioning_read']
   if len(selected)!=1 or not reads or reads[0]['tokens']!=4096 or len(reads)!=z['condition_reads']:raise ValueError('Missing condition events')
   ev=selected[0];expected=4096 if strategy=='full' else 1024
   if ev['retained_tokens']!=expected or any(t['tokens']!=expected for t in reads[1:]):raise ValueError('Condition budget differs')
   usage=torch.tensor(ev['first_read_usage'],dtype=torch.float32);indices=torch.tensor(ev['selected_indices'],dtype=torch.long)
   if usage.numel()!=4096 or indices.numel()!=expected or torch.unique(indices).numel()!=expected or not torch.isfinite(usage).all():raise ValueError('Indices or usage invalid')
   want=torch.arange(4096) if strategy=='full' else torch.topk(usage,1024).indices if strategy=='usage' else torch.arange(4096).reshape(64,64)[::2,::2].reshape(-1)
   if not torch.equal(indices,want) or hashlib.sha256(indices.numpy().tobytes()).hexdigest()!=ev['selected_indices_sha256'] or ev['selected_indices_sha256']!=z['indices_sha256']:
    raise ValueError('Author usage or uniform selection differs')
   actual=float((usage[indices].sum()/usage.sum()).item())
   if abs(actual-ev['selected_usage_fraction_of_conditioning'])>1e-7:raise ValueError('Selected usage mass differs')
   if strategy=='full' and (z['changed_masks'] or z['changed_selections'] or any(r[k]!=ref[k] for k in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256'))):raise ValueError('Full-retention no-op differs')
   if strategy!='full' and not z['changed_masks']:raise ValueError('Reduced budget had no output effect')
   pair.append((z,r))
  if pair[0]!=pair[1]:raise ValueError('Independent gate replays differ')
  runs[strategy]=pair[0]
 if not (runs['full'][0]['identity']==runs['usage'][0]['identity']==runs['uniform'][0]['identity']) or runs['usage'][0]['indices_sha256']==runs['uniform'][0]['indices_sha256']:raise ValueError('Gate identities or controls differ')
 out=validate_output(a.output)
 if out.exists():raise ValueError('Output exists')
 z={'passed':True,'scope':'Label-free engineering checks; not an efficacy result.','identity':runs['full'][0]['identity'],
  **{strategy+'_intervention_sha256':digest(path/'condition_intervention.json') for strategy,path in [('full',a.full_a),('usage',a.usage_a),('uniform',a.uniform_a)]},
  'reference_records_sha256':digest(a.reference/'records.json'),'selected_usage_fraction':{k:runs[k][0]['selected_usage_fraction'] for k in runs},
  'changed_masks':{k:runs[k][0]['changed_masks'] for k in runs},'source_sha256':digest(Path(__file__))}
 out.write_text(json.dumps(z,indent=2)+'\n');print(json.dumps({k:v for k,v in z.items() if k!='identity'},indent=2))
if __name__=='__main__':main()
