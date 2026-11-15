#!/usr/bin/env python3
"""Independently recompute frozen one-shot detector box statistics."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('audit','predictions','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();done=read(a.audit/'completion.json');d=read(a.audit/'results.json');pred=read(a.predictions/'completion.json')
if not done['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or d['prediction_sha256']!=digest(a.predictions/'completion.json') or d['runner_sha256']!=digest(ROOT/'scripts/audit_polypgen_original_finetune_boxes.py'):raise ValueError('Audit changed')
videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
for row,e in zip(d['results'],pred['events'],strict=True):
 seq=row['sequence'];i=row['frame'];f=videos[seq][i]
 if (seq,i)!=(e['sequence'],e['frame']) or digest(f.mask_path)!=row['label_sha256'] or digest(f.image_path)!=row['image_sha256']:raise ValueError('Input changed')
 gt=load_label_mask(f.mask_path,[1])==1;present=bool(gt.any());h,w=gt.shape;ys,xs=np.nonzero(gt)
 if present!=row['present']:raise ValueError('Presence differs')
 target=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1] if present else None
 original=e['original_box_viewport']
 if e['viewport']:original=[original[0]+VIEWPORT[0],original[1]+VIEWPORT[1],original[2]+VIEWPORT[0],original[3]+VIEWPORT[1]]
 if target!=row['oracle_box'] or row['candidate_box']!=e['candidate_box_full'] or row['original_box']!=original:raise ValueError('Box differs')
 for arm in ('original','candidate'):
  box=row[arm+'_box'];stat=row[arm]
  if stat['detected']!=(box is not None):raise ValueError('Detection status differs')
  if box is None:
   if stat['box_fraction']!=0 or (present and (stat['box_iou']!=0 or stat['box_truth_coverage']!=0)) or (not present and stat['labeled_empty_false_alarm'] is not False):raise ValueError('Missing box scores differ')
   continue
  x0,y0,x1,y1=box;bx=(np.arange(w)+.5>=x0)&(np.arange(w)+.5<x1);by=(np.arange(h)+.5>=y0)&(np.arange(h)+.5<y1);area=int(bx.sum())*int(by.sum())
  if abs(stat['box_fraction']-area/(h*w))>1e-12:raise ValueError('Box fraction differs')
  if present:
   cover=float(np.count_nonzero(gt & (by[:,None]&bx[None,:]))/np.count_nonzero(gt));ix=max(0,min(x1,target[2])-max(x0,target[0]));iy=max(0,min(y1,target[3])-max(y0,target[1]));inter=ix*iy;union=(x1-x0)*(y1-y0)+(target[2]-target[0])*(target[3]-target[1])-inter;iou=inter/union if union else 0.
   if abs(stat['box_truth_coverage']-cover)>1e-12 or abs(stat['box_iou']-iou)>1e-12:raise ValueError('Box scores differ')
  elif stat['labeled_empty_false_alarm'] is not True:raise ValueError('Empty false alarm differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'frames':len(d['results']),'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All frozen box scores independently reproduced.')
