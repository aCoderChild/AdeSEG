#!/usr/bin/env python3
"""Recompute local deletion scores independently from masks and labels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('capture','replay','replay-verification','audit','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();cap=read(a.capture/'priority_capture_receipt.json');replay=read(a.replay/'completion.json');vr=read(a.replay_verification/'verification.json');done=read(a.audit/'completion.json');audit=read(a.audit/'results.json')
if not all(x['passed'] for x in (cap,replay,vr,done)) or vr['first_sha256']!=digest(a.replay/'completion.json') or vr['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_channel_priority_repeats.py') or done['results_sha256']!=digest(a.audit/'results.json') or audit['source_sha256']!=digest(ROOT/'scripts/audit_lean_channel_priority_intervention.py'):raise ValueError('Provenance changed')
seq=cap['sequence'];videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos[seq]
for row,event in zip(audit['frames'],replay['events'],strict=True):
 i=event['frame'];frame=frames[i]
 if row['frame']!=i or digest(frame.mask_path)!=row['label_sha256']:raise ValueError('Label identity changed')
 gt=load_label_mask(frame.mask_path,[1])==1;present=bool(gt.any())
 if present!=row['present']:raise ValueError('Present status changed')
 for name,info in row['scores'].items():
  path=a.replay/f'{i:06d}_{name}.png'
  if digest(path)!=event['masks'][name]:raise ValueError('Mask changed')
  prediction=np.asarray(Image.open(path))>0;n=int(np.count_nonzero(prediction))+int(np.count_nonzero(gt));dice=2*int(np.count_nonzero(prediction & gt))/n if n else 1.
  if abs(dice-info['dice'])>1e-12 or int(prediction.sum())!=info['predicted_pixels'] or (bool(prediction.any()) if not present else None)!=info['false_alarm']:raise ValueError('Score differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'sequence':seq,'frames':cap['targets'],'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All masks independently rescored.')
