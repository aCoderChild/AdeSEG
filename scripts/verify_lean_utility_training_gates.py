#!/usr/bin/env python3
"""Verify nested feature/read observers and exact saved-state model replay."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_offline_memory_ablation import equal
from scripts.run_lean_token_window import FORK,PATCHED
p=argparse.ArgumentParser()
for k in ('first','second','reference','model-verification','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());x=read(a.first/'utility_capture_receipt.json');y=read(a.second/'utility_capture_receipt.json');v=read(a.model_verification/'verification.json');ident=v['identity']
if x!=y or not x['passed'] or x['mode']!='gate' or x['sequence']!='seq3' or x['runner_sha256']!=digest(ROOT/'scripts/capture_lean_utility_training_states.py') or x['read_observer_sha256']!=digest(ROOT/'modeling/lean_memory_capture.py') or x['protocol_sha256']!=digest(ROOT/'protocols/lean_utility_training_capture_v1.json') or x['reference_receipt_sha256']!=digest(a.reference/'channel_feature_receipt.json') or not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or ident['checkpoint_sha256']!=digest(a.checkpoint) or any(ident['patch_sha256'][k]!=digest(FORK/k) for k in PATCHED):raise ValueError('Capture or model verification changed')
for folder in (a.first,a.second):
 for name in ('records.json','positions.json','selections.json','memory_events.json','token_window_events.json'):
  if digest(folder/name)!=digest(a.reference/name):raise ValueError(f'Trajectory {name} differs')
 masks=sorted(p.name for p in a.reference.glob('[0-9]*.png'))
 if not masks or sorted(p.name for p in folder.glob('[0-9]*.png'))!=masks or any(digest(folder/name)!=digest(a.reference/name) for name in masks):raise ValueError('Full mask trajectory differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK);before=parameter_fingerprints(model);rope=[]
with torch.inference_mode(),indexed_spatial_rope(model,rope):
 for z in x['snapshots']:
  frame=z['frame'];files=[folder/'snapshots'/f'{frame:06d}.pt' for folder in (a.first,a.second)]
  if any(digest(f)!=z['sha256'] for f in files):raise ValueError('Snapshot hash differs')
  a0,b0=[torch.load(f,map_location='cpu',weights_only=True) for f in files]
  if not equal(a0,b0) or a0['frame']!=frame or not equal(model.memory_attention(**a0['attention_inputs']),a0['attention_outputs']) or not equal(model._forward_sam_heads(**a0['decoder_inputs']),a0['decoder_outputs']):raise ValueError('Nested no-op differs')
  for folder in (a.first,a.second):
   if digest(folder/'features'/f'{frame:06d}.pt')!=digest(a.reference/'features'/f'{frame:06d}.pt') or digest(folder/f'{frame:06d}.png')!=digest(a.reference/f'{frame:06d}.png'):raise ValueError('Feature or mask differs')
assert_parameter_integrity(model,before,out,'after')
(out/'verification.json').write_text(json.dumps({'passed':True,'runner_sha256':x['runner_sha256'],'protocol_sha256':x['protocol_sha256'],'reference_receipt_sha256':x['reference_receipt_sha256'],'first_receipt_sha256':digest(a.first/'utility_capture_receipt.json'),'second_receipt_sha256':digest(a.second/'utility_capture_receipt.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Nested observers, features and no-op model replay independently verified.')
