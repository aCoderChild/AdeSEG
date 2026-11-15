#!/usr/bin/env python3
"""Independently recompute per-mask and aggregate temporal-spread scores."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('audit','run','recent','historical','verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());done=read(a.audit/'completion.json');r=read(a.audit/'results.json');v=read(a.verification/'verification.json')
if not done['passed'] or not v['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or done['runner_sha256']!=digest(ROOT/'scripts/audit_lean_temporal_spread.py') or r['verification_sha256']!=digest(a.verification/'verification.json'):raise ValueError('Audit provenance changed')
seq=r['sequence'];frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}[seq];target=[json.loads(x) for x in (a.historical/'full'/seq/'frames.jsonl').read_text().splitlines()];selected={x['frame'] for x in target if x['present']}
if len(frames)!=len(r['frames']):raise ValueError('Frame count differs')
for i,(f,row) in enumerate(zip(frames,r['frames'],strict=True)):
 if row['frame']!=i or row['matched']!=(i in selected) or digest(f.image_path)!=row['image_sha256'] or digest(f.mask_path)!=row['label_sha256']:raise ValueError('Input or matched frame differs')
 gt=load_label_mask(f.mask_path,[1])==1;present=bool(gt.any())
 if row['present']!=present:raise ValueError('Label presence differs')
 for arm,folder in (('recent',a.recent),('spread',a.run)):
  path=folder/f'{i:06d}.png';z=row['scores'][arm]
  if digest(path)!=z['mask_sha256']:raise ValueError('Mask hash differs')
  pred=np.asarray(Image.open(path))>0
  if pred.shape!=gt.shape:raise ValueError('Mask shape differs')
  den=np.count_nonzero(pred)+np.count_nonzero(gt);dice=2*np.count_nonzero(pred&gt)/den if den else 1.;false=bool(pred.any()) if not present else None
  if abs(dice-z['dice'])>1e-12 or false!=z['false_alarm']:raise ValueError('Mask score differs')
for arm in ('recent','spread'):
 vals=[x['scores'][arm]['dice'] for x in r['frames'] if x['matched']];empty=[x['scores'][arm]['false_alarm'] for x in r['frames'] if not x['present']];z=r['summary'][arm]
 if len(vals)!=z['matched_present_count'] or abs(float(np.mean(vals))-z['matched_present_dice'])>1e-12 or sum(empty)!=z['false_alarms'] or len(empty)!=z['empty_count']:raise ValueError('Aggregate differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'sequence':seq,'frames':len(frames),'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All per-frame and aggregate scores independently verified.')
