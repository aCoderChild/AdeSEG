#!/usr/bin/env python3
"""Independently recompute fixed-frame one-shot segmentation metrics."""
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
for k in ('audit','masks','oracle','preflight','crop-reference','priority-audit','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();done=read(a.audit/'completion.json');d=read(a.audit/'results.json');mask=read(a.masks/'completion.json');oracle=read(a.oracle/'completion.json');pre=read(a.preflight/'manifest.json');prior=read(a.priority_audit/'results.json')
if not done['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or d['runner_sha256']!=digest(ROOT/'scripts/audit_polypgen_original_finetune_masks.py') or d['mask_receipt_sha256']!=digest(a.masks/'completion.json') or d['oracle_sha256']!=digest(a.oracle/'completion.json'):raise ValueError('Mask audit provenance changed')
videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
for row,m,o in zip(d['results'],mask['events'],oracle['events'],strict=True):
 seq=row['sequence'];i=row['frame'];frame=videos[seq][i]
 if (seq,i)!=(m['sequence'],m['frame']) or (seq,i)!=(o['sequence'],o['frame']) or digest(frame.mask_path)!=row['label_sha256']:raise ValueError('Label identity differs')
 gt=load_label_mask(frame.mask_path,[1])==1;present=bool(gt.any())
 if present!=row['present']:raise ValueError('Presence differs')
 source=Path(pre['sequences'][seq]['source'])/seq;original=a.crop_reference/'prompt.png' if m['viewport'] else source/'prompt_refined.png';old=next(x for x in prior['results'] if x['sequence']==seq)
 if old['frame']!=i or digest(original)!=old['prompt_mask_sha256']:raise ValueError('Original prompt changed')
 paths={'saved_lean_prompt':original}
 for name,sha in m['masks'].items():
  path=a.masks/f'{seq}_{i:06d}_{name}.png'
  if digest(path)!=sha:raise ValueError('Candidate mask changed')
  paths[name]=path
 if o['mask_sha256'] is not None:
  path=a.oracle/f'{seq}_{i:06d}_oracle_box.png'
  if digest(path)!=o['mask_sha256']:raise ValueError('Oracle mask changed')
  paths['oracle_box']=path
 elif row['scores']['oracle_box'] is not None:raise ValueError('Empty oracle score')
 for name,path in paths.items():
  pred=np.asarray(Image.open(path))>0;n=int(np.count_nonzero(pred));truth=int(np.count_nonzero(gt));tp=int(np.count_nonzero(pred&gt));info=row['scores'][name]
  dice=2*tp/(n+truth) if present else None;precision=(tp/n if n else 0.) if present else None;recall=tp/truth if present else None;false=bool(n) if not present else None
  if info['predicted_pixels']!=n or info['false_alarm']!=false:raise ValueError('Mask count or false alarm differs')
  for key,value in (('dice',dice),('precision',precision),('recall',recall)):
   if (info[key] is None)!=(value is None) or value is not None and abs(info[key]-value)>1e-12:raise ValueError('Mask metric differs: '+key)
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'frames':len(d['results']),'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All original, candidate and oracle mask scores independently recomputed.')
