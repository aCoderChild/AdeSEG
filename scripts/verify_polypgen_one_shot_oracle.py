#!/usr/bin/env python3
"""Verify repeated explicit ground-truth box controls."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('first','second','predictions','masks','mask-verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();x=read(a.first/'completion.json');y=read(a.second/'completion.json');pred=read(a.predictions/'completion.json');mv=read(a.mask_verification/'verification.json')
if not x['passed'] or x!=y or x['predictions_sha256']!=digest(a.predictions/'completion.json') or x['mask_receipt_sha256']!=digest(a.masks/'completion.json') or not mv['passed'] or mv['mask_receipt_sha256']!=x['mask_receipt_sha256'] or mv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_one_shot_masks.py') or x['runner_sha256']!=digest(ROOT/'scripts/predict_polypgen_one_shot_oracle.py') or x['protocol_sha256']!=digest(ROOT/'protocols/polypgen_one_shot_detector_pilot_v1.json'):raise ValueError('Oracle provenance changed')
videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
for e,prediction in zip(x['events'],pred['events'],strict=True):
 seq=e['sequence'];i=e['frame'];f=videos[seq][i]
 if (seq,i)!=(prediction['sequence'],prediction['frame']) or digest(f.mask_path)!=e['label_sha256']:raise ValueError('Label changed')
 gt=load_label_mask(f.mask_path,[1])==1
 if prediction['viewport']:
  x0,y0,x1,y1=VIEWPORT
  if gt[:,:x0].any():raise ValueError('Label outside viewport')
  gt=gt[y0:y1,x0:x1]
 yy,xx=np.nonzero(gt);box=None if len(xx)==0 else [float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)]
 if box!=e['oracle_box_viewport']:raise ValueError('Oracle box differs')
 if box is None:
  if e['mask_sha256'] is not None:raise ValueError('Oracle output on empty label')
 else:
  for folder in (a.first,a.second):
   path=folder/f'{seq}_{i:06d}_oracle_box.png'
   if digest(path)!=e['mask_sha256']:raise ValueError('Oracle mask changed')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'frames':len(x['events']),'oracle_sha256':digest(a.first/'completion.json'),'repeat_sha256':digest(a.second/'completion.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Explicit oracle boxes and repeated masks verified.')
