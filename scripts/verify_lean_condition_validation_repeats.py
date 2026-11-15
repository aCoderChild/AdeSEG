#!/usr/bin/env python3
"""Verify repeated prompt-region interventions and independently recompute ranks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_condition_region_ablation import prompt_region_scores
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('capture','first','second','merged','gate-verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();cap=read(a.capture/'condition_validation_capture.json');x=read(a.first/'completion.json');y=read(a.second/'completion.json');g=read(a.gate_verification/'verification.json')
if not all(t['passed'] for t in (cap,x,y,g)) or x!=y or x['mode']!='candidate' or x['sequence']!=cap['sequence'] or x['capture_sha256']!=digest(a.capture/'condition_validation_capture.json') or x['runner_sha256']!=digest(ROOT/'scripts/replay_lean_condition_validation.py') or x['module_sha256']!=digest(ROOT/'modeling/lean_condition_region_ablation.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_condition_region_ablation_v1.json') or g['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_condition_region_gate.py'):raise ValueError('Receipts changed')
seq=cap['sequence'];prompt=a.capture/'prompt.png' if cap.get('viewport',False) else Path(read(a.merged/'locations.json')[seq])/seq/'prompt_refined.png'
if digest(prompt)!=x['prompt_sha256']:raise ValueError('Prompt changed')
mask=np.asarray(Image.open(prompt))>0
if cap.get('viewport',False):x0,y0,x1,y1=VIEWPORT;mask=mask[y0:y1,x0:x1]
original_occ=prompt_region_scores(mask);occ=original_occ.astype(np.float64);ids=np.arange(4096,dtype=np.int64);reverse=np.zeros(4096,dtype=np.int64)
for bit in range(12):reverse|=((ids>>bit)&1)<<(11-bit)
tie=reverse;snapshots={s['frame']:s for s in cap['snapshots']};methods={'foreground','background','boundary','uniform','shuffled','attention_oracle'}
if {e['frame'] for e in x['events']}!=set(snapshots):raise ValueError('Frames missing')
for e in x['events']:
 i=e['frame'];path=a.capture/'snapshots'/f'{i:06d}.pt'
 if digest(path)!=snapshots[i]['read_sha256']:raise ValueError('Snapshot changed')
 data=torch.load(path,map_location='cpu',weights_only=True);u=data['attention_outputs'][1].double().numpy();total=len(u);ptr=int(data['attention_inputs']['num_obj_ptr_tokens'])
 vals={'foreground':occ,'background':1-occ,'boundary':occ*(1-occ),'shuffled':occ[np.random.default_rng(20261114).permutation(4096)],'attention_oracle':u[:4096]}
 if set(e['selected'])!=methods or set(e['masks'])!=methods|{'noop'}:raise ValueError('Method scope changed')
 for method in methods:
  chosen=e['selected'][method]
  ranked=np.argsort(tie,kind='stable')[:1024] if method=='uniform' else np.lexsort((tie,-vals[method]))[:1024]
  if chosen!=np.sort(ranked).tolist() or len(set(chosen))!=1024:raise ValueError('Selection ranks differ')
  if e['key_counts'][method]!=total-1024 or e['rotary'][method]['pointer_keys']!=ptr or e['rotary'][method]['spatial_keys']!=total-1024-ptr or e['rotary'][method]['key_rotations']!=4:raise ValueError('Attention geometry differs')
  if abs(e['removed_attention_mass'][method]-float(u[chosen].sum()/u.sum()))>1e-12 or abs(e['mean_prompt_occupancy'][method]-float(original_occ[chosen].mean()))>1e-12:raise ValueError('Mass or occupancy differs')
 for name,h in e['masks'].items():
  for folder in (a.first,a.second):
   if digest(folder/f'{i:06d}_{name}.png')!=h:raise ValueError('Mask differs')
 if e['masks']['noop']!=digest(a.capture/f'{i:06d}.png'):raise ValueError('No-op differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'sequence':seq,'frames':sorted(snapshots),'first_sha256':digest(a.first/'completion.json'),'second_sha256':digest(a.second/'completion.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print(seq,'repeat, roles, attention and no-op verified')
