#!/usr/bin/env python3
"""Independently audit two label-free conditioning region replays."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_condition_region_ablation import prompt_region_scores,condition_ablation_indices
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('capture','first','second','merged','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();cap=read(a.capture/'region_capture_receipt.json');x=read(a.first/'completion.json');y=read(a.second/'completion.json')
if not cap['passed'] or not x['passed'] or x!=y or x['mode']!='gate' or x['sequence']!='seq3' or x['capture_sha256']!=digest(a.capture/'region_capture_receipt.json') or x['runner_sha256']!=digest(ROOT/'scripts/replay_lean_condition_region.py') or x['module_sha256']!=digest(ROOT/'modeling/lean_condition_region_ablation.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_condition_region_ablation_v1.json'):raise ValueError('Gate provenance or repetition changed')
folder=Path(read(a.merged/'locations.json')['seq3'])/'seq3';prompt=folder/'prompt_refined.png'
if digest(prompt)!=x['prompt_sha256']:raise ValueError('Prompt changed')
occ=prompt_region_scores(np.asarray(Image.open(prompt))>0);snapshots={s['frame']:s for s in cap['snapshots']};methods={'foreground','background','boundary','uniform','shuffled','attention_oracle'}
for e in x['events']:
 i=e['frame'];path=a.capture/'snapshots'/f'{i:06d}.pt'
 if digest(path)!=snapshots[i]['read_sha256']:raise ValueError('Snapshot changed')
 data=torch.load(path,map_location='cpu',weights_only=True);u=data['attention_outputs'][1].double().numpy();total=len(u);ptr=int(data['attention_inputs']['num_obj_ptr_tokens'])
 if set(e['selected'])!=methods or set(e['masks'])!=methods|{'noop'}:raise ValueError('Missing method')
 for method in methods:
  ids=e['selected'][method]
  if ids!=condition_ablation_indices(occ,u[:4096],method).tolist() or len(ids)!=1024 or len(set(ids))!=1024:raise ValueError('Selection differs')
  if e['key_counts'][method]!=total-1024 or e['rotary'][method]['pointer_keys']!=ptr or e['rotary'][method]['spatial_keys']!=total-1024-ptr:raise ValueError('Attention geometry changed')
  if abs(e['removed_attention_mass'][method]-float(u[ids].sum()/u.sum()))>1e-12:raise ValueError('Mass differs')
 for name,h in e['masks'].items():
  for folder in (a.first,a.second):
   if digest(folder/f'{i:06d}_{name}.png')!=h:raise ValueError('Mask changed')
 if e['masks']['noop']!=digest(a.capture/f'{i:06d}.png'):raise ValueError('No-op mask differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'checkpoint_sha256':x['checkpoint_sha256'],'module_sha256':x['module_sha256'],'protocol_sha256':x['protocol_sha256'],'runner_sha256':x['runner_sha256'],'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Condition-region engineering replays and ranks verified.')
