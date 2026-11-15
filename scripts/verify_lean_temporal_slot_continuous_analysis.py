#!/usr/bin/env python3
"""Independently recompute diagnostic statistics from verified saved tensors."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('analysis','replay','verification','binary-audit','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());done=read(a.analysis/'completion.json');d=read(a.analysis/'results.json');r=read(a.replay/'completion.json');v=read(a.verification/'verification.json');b=read(a.binary_audit/'results.json')
if not all(x['passed'] for x in (done,r,v)) or done['results_sha256']!=digest(a.analysis/'results.json') or done['report_sha256']!=digest(a.analysis/'report.md') or d['runner_sha256']!=digest(ROOT/'scripts/analyze_lean_temporal_slot_continuous.py') or d['replay_sha256']!=digest(a.replay/'completion.json') or d['verification_sha256']!=digest(a.verification/'verification.json') or d['binary_audit_sha256']!=digest(a.binary_audit/'results.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_temporal_slot_continuous_gates.py'):raise ValueError('Source or verification differs')
for row,e,br in zip(d['rows'],r['events'],b['frames'],strict=True):
 i=row['frame']
 if i!=e['frame'] or i!=br['frame'] or row['label_present']!=br['present'] or row['binary_scores']!=br['scores']:raise ValueError('Frame or binary score differs')
 tensors={}
 for arm in ('noop','rank','permuted'):
  path=a.replay/f'{i:06d}_{arm}.pt'
  if digest(path)!=e['tensors'][arm] or row['arms'][arm]['mask_sha256']!=e['masks'][arm]:raise ValueError('Tensor or mask hash differs')
  tensors[arm]=torch.load(path,map_location='cpu',weights_only=True)
 for arm in tensors:
  t=tensors[arm];z=row['arms'][arm]
  for key,value in (('object_logit',float(t['object_logits'].item())),('post_mask_min',float(t['post_masks'].min())),('post_mask_max',float(t['post_masks'].max()))):
   if abs(z[key]-value)>1e-12:raise ValueError('Scalar differs')
  for key in ('attention','raw_masks','post_masks'):
   diff=(t[key].numpy()-tensors['noop'][key].numpy()).astype(np.float64)
   rms=float(np.sqrt(np.mean(diff*diff)));maximum=float(np.abs(diff).max())
   if abs(z[key+'_delta_rms']-rms)>1e-12 or abs(z[key+'_delta_max']-maximum)>1e-12:raise ValueError('Continuous difference differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'frames':len(d['rows']),'analysis_sha256':digest(a.analysis/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All attention, raw-mask and object-logit statistics independently recomputed.')
