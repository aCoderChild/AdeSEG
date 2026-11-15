#!/usr/bin/env python3
"""Summarize verified attention, pre-suppression masks and object logits."""
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('replay','verification','binary-audit','binary-verification','binary-replay','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());r=read(a.replay/'completion.json');v=read(a.verification/'verification.json');b=read(a.binary_audit/'results.json');bv=read(a.binary_verification/'verification.json')
if not all(x['passed'] for x in (r,v,bv)) or r['mode']!='candidate' or v['mode']!='candidate' or v['first_sha256']!=digest(a.replay/'completion.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_temporal_slot_continuous_gates.py') or b['replay_sha256']!=digest(a.binary_replay/'completion.json') or bv['audit_sha256']!=digest(a.binary_audit/'results.json') or bv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_temporal_slot_rank_audit.py'):raise ValueError('Verified inputs differ')
prior=read(a.binary_replay/'completion.json')
rows=[]
for e,br in zip(r['events'],b['frames'],strict=True):
 i=e['frame']
 if i!=br['frame'] or e['masks']!=next(z['masks'] for z in prior['events'] if z['frame']==i) or e['masks']!={name:digest(a.replay/f'{i:06d}_{name}.png') for name in ('noop','rank','permuted')}:raise ValueError('Mask identity differs')
 z={}
 for arm in ('noop','rank','permuted'):
  path=a.replay/f'{i:06d}_{arm}.pt'
  if digest(path)!=e['tensors'][arm]:raise ValueError('Tensor changed')
  z[arm]=torch.load(path,map_location='cpu',weights_only=True)
  if any(not torch.isfinite(t).all() for t in z[arm].values()) or z[arm]['object_logits'].numel()!=1:raise ValueError('Unexpected decoder tensor')
 raw={}
 for arm,t in z.items():
  result={'object_logit':float(t['object_logits'].item()),'post_mask_min':float(t['post_masks'].min()),'post_mask_max':float(t['post_masks'].max()),'mask_sha256':e['masks'][arm]}
  for key in ('attention','raw_masks','post_masks'):
   delta=(t[key]-z['noop'][key]).double();result[key+'_delta_rms']=float(torch.sqrt(torch.mean(delta*delta)));result[key+'_delta_max']=float(delta.abs().max())
  if any(not math.isfinite(x) for k,x in result.items() if isinstance(x,float)):raise ValueError('Nonfinite diagnostic')
  raw[arm]=result
 rows.append({'frame':i,'label_present':br['present'],'binary_scores':br['scores'],'arms':raw})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
d={'scope':'Post-hoc frozen reads. Raw decoder masks are pre-presence observations only and are not used for inference. Native object-presence and mask rules are unchanged.','rows':rows,'replay_sha256':digest(a.replay/'completion.json'),'verification_sha256':digest(a.verification/'verification.json'),'binary_audit_sha256':digest(a.binary_audit/'results.json'),'runner_sha256':digest(Path(__file__))}
(out/'results.json').write_text(json.dumps(d,indent=2)+'\n');lines=['# Sequence 23: temporal-slot effects before and after presence suppression','',d['scope'],'','| Frame | Arm | Native object logit | Attention Δ RMS | Raw mask-logit Δ RMS | Post-suppression mask Δ RMS | Binary Dice / FA |','| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
for x in rows:
 for arm,z in x['arms'].items():
  score=x['binary_scores'][arm];b=f"{score['dice']:.3f}" if score['dice'] is not None else 'FA='+str(score['false_alarm']);lines.append(f"| {x['frame']} | {arm} | {z['object_logit']:.6f} | {z['attention_delta_rms']:.3e} | {z['raw_masks_delta_rms']:.3e} | {z['post_masks_delta_rms']:.3e} | {b} |")
lines+=['','Deltas are computed against the no-op on the same saved read. A negative native object logit causes the existing model to suppress its post-head mask; inspecting the raw mask does not alter that decision. Effects on internal tensors do not prove any full-video benefit, and these previously inspected frames are not an independent test.','']
(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
