#!/usr/bin/env python3
"""Independently recompute confusion counts from repeated local fusion masks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('audit','first','second','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());done=read(a.audit/'completion.json');r=read(a.audit/'results.json');one=read(a.first/'completion.json');two=read(a.second/'completion.json')
if not done['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or r['audit_sha256']!=digest(ROOT/'scripts/audit_lean_same_read_fusion_priority.py') or not one['passed'] or one!=two or r['replay_sha256']!=digest(a.first/'completion.json') or r['repeat_sha256']!=digest(a.second/'completion.json'):raise ValueError('Audit or repeats changed')
seq=r['sequence'];videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos[seq];events={e['frame']:e for e in one['events']};checks=0
for row in r['frames']:
 i=row['frame']
 if digest(frames[i].mask_path)!=row['label_sha256']:raise ValueError('Label changed')
 gt=load_label_mask(frames[i].mask_path,[1])==1
 if bool(gt.any())!=row['present']:raise ValueError('Presence differs')
 masks={}
 for name,score in row['scores'].items():
  f=a.first/f'{i:06d}_{name}.png';g=a.second/f.name
  if digest(f)!=events[i]['masks'][name] or digest(g)!=events[i]['masks'][name]:raise ValueError('Mask hash differs')
  pred=np.asarray(Image.open(f))>0
  if pred.shape!=gt.shape:raise ValueError('Mask geometry differs')
  tp=int(np.count_nonzero(pred & gt));fp=int(np.count_nonzero(pred & ~gt));fn=int(np.count_nonzero(~pred & gt));den=2*tp+fp+fn;dice=2*tp/den if den else 1.
  if abs(score['dice']-dice)>1e-12 or score['predicted_pixels']!=tp+fp or score['false_alarm']!=(bool(tp+fp) if not gt.any() else None):raise ValueError('Score differs')
  masks[name]=pred;checks+=1
 if np.count_nonzero(masks['noop']^masks['raw_skeleton'])!=row['raw_changed_pixels'] or not np.array_equal(masks['noop'],masks['fused_reconstructed']):raise ValueError('Pixel comparison differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'passed':True,'sequence':seq,'frames':len(r['frames']),'mask_checks':checks,'results_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))};(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
