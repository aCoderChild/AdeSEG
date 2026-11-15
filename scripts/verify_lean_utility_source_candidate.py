#!/usr/bin/env python3
"""Audit source identities against the previously captured frozen read."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_position_lookup import MemoryPositionLookup
from modeling.lean_region_read_intervention import locate_verified_span
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_token_window import FORK,PATCHED
from scripts.run_lean_offline_memory_ablation import equal
p=argparse.ArgumentParser()
for name in ('capture','reference','source-verification','model-verification','checkpoint','output'):
 p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text())
r=read(a.capture/'source_capture_receipt.json');u=read(a.capture/'utility_capture_receipt.json');ref=read(a.reference/'utility_capture_receipt.json');v=read(a.source_verification/'verification.json');m=read(a.model_verification/'verification.json')
if not r['passed'] or not u['passed'] or not ref['passed'] or u!=ref or r['sequence']!=u['sequence'] or r['nested_receipt_sha256']!=digest(a.capture/'utility_capture_receipt.json') or r['source_gate_sha256']!=digest(a.source_verification/'verification.json') or not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_utility_source_gates.py') or r['observer_sha256']!=digest(ROOT/'modeling/lean_utility_source_capture.py') or r['runner_sha256']!=digest(ROOT/'scripts/capture_lean_utility_source_candidate.py') or not m['passed'] or m['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or m['identity']['checkpoint_sha256']!=digest(a.checkpoint) or any(m['identity']['patch_sha256'][k]!=digest(FORK/k) for k in PATCHED):raise ValueError('Capture or engineering provenance changed')
for group in ('source_features','source_snapshots'):
 if not r[group] or {x.name for x in (a.capture/group).glob('*.pt')}!=set(r[group]) or any(digest(a.capture/group/n)!=h for n,h in r[group].items()):raise ValueError('Source file identity differs')
for name in ('records.json','positions.json','selections.json','memory_events.json','token_window_events.json'):
 if digest(a.capture/name)!=digest(a.reference/name):raise ValueError('Trajectory differs')
masks=sorted(p.name for p in a.reference.glob('[0-9]*.png'))
if not masks or masks!=sorted(p.name for p in a.capture.glob('[0-9]*.png')) or any(digest(a.capture/n)!=digest(a.reference/n) for n in masks):raise ValueError('Masks differ')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK);before=parameter_fingerprints(model)
with torch.no_grad():grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1)[:,0].cpu().float()
lookup=MemoryPositionLookup(grid,torch.zeros(1,model.mem_dim));offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).cpu()
shots={z['frame']:z['sha256'] for z in u['snapshots']}
for event in r['observations']:
 frame=event['frame'];path=a.capture/'snapshots'/f'{frame:06d}.pt'
 if digest(path)!=shots[frame] or digest(a.reference/'snapshots'/path.name)!=shots[frame]:raise ValueError('Attention snapshot differs')
 data=torch.load(path,map_location='cpu',weights_only=True);source=torch.load(a.capture/'source_snapshots'/path.name,map_location='cpu',weights_only=True)
 if source['frame']!=frame or source['pointer_count']!=data['attention_inputs']['num_obj_ptr_tokens']:raise ValueError('Source snapshot differs')
 args=data['attention_inputs'];protected=[(0,4096)];count=0
 for s in source['sources']:
  if s['frame']>=frame or not (a.capture/'source_features'/f"{s['frame']:06d}.pt").is_file():raise ValueError('Missing source features')
  start=locate_verified_span(args['memory'],args['memory_pos'],source['pointer_count'],s['features'],s['positions'],offsets,protected);ids=lookup.decode(s['positions'].float(),0)
  if start!=s['start'] or not torch.equal(ids,s['token_ids']) or len(torch.unique(ids))!=len(ids):raise ValueError('Source identity or coordinates differ')
  protected.append((start,start+len(ids)));count+=len(ids)
 if count!=event['attributed_tokens'] or len(source['sources'])!=event['attributed_sources'] or source['rejected']!=event['rejected']:raise ValueError('Source summary differs')
assert_parameter_integrity(model,before,out,'after')
result={'passed':True,'sequence':u['sequence'],'source_receipt_sha256':digest(a.capture/'source_capture_receipt.json'),'reference_receipt_sha256':digest(a.reference/'utility_capture_receipt.json'),'runner_sha256':digest(Path(__file__)),'events':r['observations'],'meaning':'Captured-source engineering audit; not deletion utility or a segmentation result.'}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'sequence':u['sequence']}))
