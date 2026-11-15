#!/usr/bin/env python3
"""Compare saved, label-free CPU/MPS token-window engineering replays."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('cpu','mps-a','mps-b','reference','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();paths={'cpu':a.cpu,'mps_a':a.mps_a,'mps_b':a.mps_b};runs={k:read(p/'receipt.json') for k,p in paths.items()}
 for key,x in runs.items():
  if not x['passed'] or x['device']!=('cpu' if key=='cpu' else 'mps') or x['prompt_frame']!=9 or x['end_frame']!=20:raise ValueError('Unexpected replay')
  if any(digest(paths[key]/f"{r['frame']:06d}.png")!=r['mask_sha256'] for r in x['rows']):raise ValueError('Mask changed')
  if any(digest(paths[key]/f"features_{r['frame']:06d}.pt")!=r['sha256'] for r in x['features']):raise ValueError('Feature changed')
  writes=[e for e in x['token_window_events'] if e['stage']=='after_write']
  if len(writes)!=11 or any(e['total_tokens']>16384 or e['insurance_tokens']>3072 for e in writes):raise ValueError('Budget or coverage changed')
 for key in ('checkpoint_sha256','prompt_sha256','fork_patches','runner_sha256'):
  if len({json.dumps(x[key],sort_keys=True) for x in runs.values()})!=1:raise ValueError(f'Cross-device input differs: {key}')
 old=read(a.reference/'records.json');old_events=[e for e in read(a.reference/'token_window_events.json') if e['frame']<=20]
 if any(r['mask_sha256']!=old['mask_sha256'][r['frame']] for r in runs['cpu']['rows']) or old_events!=runs['cpu']['token_window_events']:
  raise ValueError('CPU replay differs from archived masks or token-window events')
 tensors={k:{i:torch.load(path/f'features_{i:06d}.pt',map_location='cpu',weights_only=True) for i in (10,19)} for k,path in paths.items()}
 for i in (10,19):
  old_feature=torch.load(a.reference/'features'/f'{i:06d}.pt',map_location='cpu',weights_only=True)['feature'].unsqueeze(0)
  snapshot=torch.load(a.reference/'snapshots'/f'{i:06d}.pt',map_location='cpu',weights_only=True)
  for k,v in [('early',old_feature),('deep',snapshot['decoder_inputs']['backbone_features']),('low',snapshot['decoder_outputs'][3])]:
   if not torch.equal(tensors['cpu'][i][k],v):raise ValueError(f'CPU {k} differs from archived capture')
 def measure(left,right):
  masks=[]
  for i in range(9,21):
   x=np.asarray(Image.open(paths[left]/f'{i:06d}.png'))>0;y=np.asarray(Image.open(paths[right]/f'{i:06d}.png'))>0
   if x.shape!=y.shape:raise ValueError('Mask geometry mismatch')
   inter=int((x&y).sum());union=int((x|y).sum());den=int(x.sum()+y.sum())
   masks.append({'frame':i,'iou':inter/union if union else 1.0,'dice':2*inter/den if den else 1.0,'disagree_pixels':int((x!=y).sum())})
  features=[]
  for i in (10,19):
   for name in ('early','deep','low'):
    x=tensors[left][i][name];y=tensors[right][i][name]
    if x.shape!=y.shape:raise ValueError('Feature geometry mismatch')
    d=(x-y).abs();features.append({'frame':i,'kind':name,'shape':list(x.shape),'mean_abs':float(d.mean()),'max_abs':float(d.max()),'exact':torch.equal(x,y)})
  l,r=runs[left],runs[right]
  return {'masks':masks,'features':features,'token_window_events_equal':l['token_window_events']==r['token_window_events'],
          'selection_events_equal':l['selection']==r['selection'],
          'position_geometry_equal':[(x['spatial_keys'],x['pointer_keys']) for x in l['positions']]==[(x['spatial_keys'],x['pointer_keys']) for x in r['positions']],
          'position_index_hashes_equal':[x['indices_sha256'] for x in l['positions']]==[x['indices_sha256'] for x in r['positions']]}
 result={'scope':'No-label engineering comparison on one short seq4 trajectory. No predeclared equivalence tolerance; no learned model, efficacy or generalization claim.',
  'cpu_matches_archived_captures':True,'cpu_vs_mps':measure('cpu','mps_a'),'mps_repeat':measure('mps_a','mps_b'),
  'budget':{'capacity':16384,'insurance_reserve':3072,**{k:{'max_total_tokens':max(e['total_tokens'] for e in x['token_window_events'] if e['stage']=='after_write'),
    'eviction_events':sum(bool(e['evict_frames']) for e in x['token_window_events'] if e['stage']=='admission_plan')} for k,x in runs.items()}},
  'input_receipt_sha256':{k:digest(path/'receipt.json') for k,path in paths.items()},'runner_sha256':digest(Path(__file__))}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
