#!/usr/bin/env python3
"""Independently rescore all saved factorial output masks and changed pixels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('audit','replay','verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda p:json.loads(p.read_text());d=read(a.audit/'results.json');done=read(a.audit/'completion.json');r=read(a.replay/'completion.json');v=read(a.verification/'verification.json')
if not all(x['passed'] for x in (done,r,v)) or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or d['runner_sha256']!=digest(ROOT/'scripts/audit_lean_empty_encoder_components.py') or d['replay_sha256']!=digest(a.replay/'completion.json') or d['verification_sha256']!=digest(a.verification/'verification.json'):raise ValueError('Audit provenance differs')
frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq23']
for row,e in zip(d['rows'],r['events'],strict=True):
 i=row['target_frame'];f=frames[i]
 if i!=e['target_frame'] or row['source_frame']!=e['source_frame'] or digest(f.image_path)!=row['image_sha256'] or digest(f.mask_path)!=row['label_sha256'] or row['object_logits']!=e['object_logits']:raise ValueError('Input or logits differ')
 gt=load_label_mask(f.mask_path,[1])==1;present=bool(gt.any());masks={}
 if present or present!=row['present']:raise ValueError('Selected control is no longer labeled empty')
 for arm,sha in e['masks'].items():
  path=a.replay/f'{i:06d}_{arm}.png'
  if digest(path)!=sha:raise ValueError('Mask changed')
  pred=np.asarray(Image.open(path))>0;n=int(pred.sum());g=int(gt.sum());tp=int((pred&gt).sum());dice=2*tp/(n+g) if present else None;fa=bool(n) if not present else None;z=row['scores'][arm]
  if z['predicted_pixels']!=n or z['false_alarm']!=fa or (z['dice'] is None)!=(dice is None) or dice is not None and abs(z['dice']-dice)>1e-12:raise ValueError('Score differs')
  masks[arm]=pred
 for arm,count in row['changed_pixels'].items():
  base=arm.split('_')[0]+'_rrr'
  if count!=int(np.count_nonzero(masks[arm]!=masks[base])):raise ValueError('Changed pixels differ')
for arm,count in d['summary'].items():
 if count!=sum(row['scores'][arm]['false_alarm'] for row in d['rows']):raise ValueError('False-alarm total differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'frames':len(d['rows']),'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All factorial mask scores and changed pixels independently recomputed.')
