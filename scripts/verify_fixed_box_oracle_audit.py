#!/usr/bin/env python3
"""Independently recalculate image diagnostic scores from saved inputs."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('fixed','oracle','audit','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();done=json.loads((a.audit/'completion.json').read_text());data=json.loads((a.audit/'results.json').read_text())
if not done['passed'] or done['results_sha256']!=digest(a.audit/'results.json'):raise ValueError('Audit changed')
videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};fixed=json.loads((a.fixed/'completion.json').read_text());oracle=json.loads((a.oracle/'completion.json').read_text())
for r,fe,oe in zip(data['results'],fixed['events'],oracle['events'],strict=True):
 seq=r['sequence'];i=r['frame'];frame=videos[seq][i]
 if (seq,i)!=(fe['sequence'],fe['frame']) or (seq,i)!=(oe['sequence'],oe['frame']) or digest(frame.mask_path)!=oe['label_sha256']:raise ValueError('Case identity')
 gt=load_label_mask(frame.mask_path,[1])==1;present=bool(gt.any())
 fp=a.fixed/f'{seq}_{i:06d}_fixed_initial.png'
 if digest(fp)!=fe['masks']['fixed_initial']:raise ValueError('Fixed mask hash')
 pred=np.asarray(Image.open(fp))>0
 if r['present']!=present:raise ValueError('Presence mismatch')
 if present:
  oracle_path=a.oracle/f'{seq}_{i:06d}_oracle.png'
  if digest(oracle_path)!=oe['mask_sha256']:raise ValueError('Oracle hash')
  op=np.asarray(Image.open(oracle_path))>0
  for k,m in [('fixed_dice',pred),('oracle_dice',op)]:
   dice=float(2*np.count_nonzero(m & gt)/(np.count_nonzero(m)+np.count_nonzero(gt)))
   if abs(dice-r[k])>1e-12:raise ValueError(k)
  view=gt
  if seq=='seq23':x0,y0,x1,y1=VIEWPORT;view=gt[y0:y1,x0:x1]
  ys,xs=np.where(view);b=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1];v=fe['box']
  inter=max(0,min(v[2],b[2])-max(v[0],b[0]))*max(0,min(v[3],b[3])-max(v[1],b[1]));union=(v[2]-v[0])*(v[3]-v[1])+(b[2]-b[0])*(b[3]-b[1])-inter
  if abs(inter/union-r['fixed_box_iou'])>1e-12:raise ValueError('Box IoU')
 elif bool(pred.any())!=r['fixed_false_alarm']:raise ValueError('False alarm')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'case_count':len(data['results']),'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n')
print('All frozen image diagnostic results independently reproduced.')
