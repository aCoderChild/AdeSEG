#!/usr/bin/env python3
"""Recompute saved prompt-region intervention Dice and false alarms."""
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
for k in ('capture','replay','verification','audit','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();cap=read(a.capture/'priority_capture_receipt.json');r=read(a.replay/'completion.json');v=read(a.verification/'verification.json');done=read(a.audit/'completion.json');d=read(a.audit/'results.json')
if not all(x['passed'] for x in (cap,r,v,done)) or v['first_sha256']!=digest(a.replay/'completion.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_condition_region_repeats.py') or done['results_sha256']!=digest(a.audit/'results.json') or d['runner_sha256']!=digest(ROOT/'scripts/audit_lean_condition_region.py'):raise ValueError('Audit provenance changed')
seq=cap['sequence'];videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
for row,e in zip(d['frames'],r['events'],strict=True):
 i=row['frame'];frame=videos[seq][i]
 if i!=e['frame'] or digest(frame.mask_path)!=row['label_sha256']:raise ValueError('Frame changed')
 gt=load_label_mask(frame.mask_path,[1])==1
 if bool(gt.any())!=row['present']:raise ValueError('Presence differs')
 for name,info in row['scores'].items():
  path=a.replay/f'{i:06d}_{name}.png'
  if digest(path)!=e['masks'][name]:raise ValueError('Mask changed')
  pred=np.asarray(Image.open(path))>0;den=int(np.count_nonzero(pred))+int(np.count_nonzero(gt));dice=2*int(np.count_nonzero(pred&gt))/den if den else 1.
  if abs(dice-info['dice'])>1e-12 or int(pred.sum())!=info['predicted_pixels'] or (bool(pred.any()) if not gt.any() else None)!=info['false_alarm']:raise ValueError('Score differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'sequence':seq,'frames':len(d['frames']),'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print(seq,'scores independently verified')
