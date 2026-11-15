#!/usr/bin/env python3
"""Independently recompute paired C6 detector metrics from frozen boxes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('audit','predictions','inputs','dataset','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();done=read(a.audit/'completion.json');d=read(a.audit/'results.json');pred=read(a.predictions/'completion.json');src=read(a.inputs/'inputs.json');dataset=read(a.dataset/'dataset.json')
if not done['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or d['predictions_sha256']!=digest(a.predictions/'completion.json') or d['inputs_sha256']!=digest(a.inputs/'inputs.json') or d['dataset_sha256']!=digest(a.dataset/'dataset.json') or d['runner_sha256']!=digest(ROOT/'scripts/audit_polypgen_c6_detector.py'):raise ValueError('Audit provenance changed')
source={r['staged_image']:r for r in dataset['included'] if r['split']=='val'};computed=[]
for row,event,inp in zip(d['rows'],pred['events'],src['inputs'],strict=True):
 if row['id']!=event['id'] or event['id']!=inp['id'] or row['image_sha256']!=inp['image_sha256'] or digest(Path(inp['image']))!=inp['image_sha256']:raise ValueError('Image identity changed')
 item=source[inp['image']];mp=Path(item['source_mask'])
 if digest(mp)!=row['mask_sha256'] or row['gt_boxes']!=item['boxes_xyxy'] or [digest(Path(f)) for f in item['source_boxes']]!=item['source_boxes_sha256']:raise ValueError('Target changed')
 with Image.open(mp) as im:gray=np.asarray(im.convert('L'))
 gt=gray>(0 if gray.max()<=1 else 127);present=bool(gt.any());h,w=gt.shape
 if present!=row['present'] or (h,w)!=(inp['height'],inp['width']):raise ValueError('Target geometry changed')
 scores={}
 for name in ('original','candidate'):
  box=event['boxes'][name]['box'];stat=row['scores'][name]
  if box is None:
   iou=0. if present else None;coverage=0. if present else None;false=False if not present else None
  else:
   x0,y0,x1,y1=box;bx=(np.arange(w)+.5>=x0)&(np.arange(w)+.5<x1);by=(np.arange(h)+.5>=y0)&(np.arange(h)+.5<y1)
   coverage=float(np.count_nonzero(gt & (by[:,None]&bx[None,:]))/np.count_nonzero(gt)) if present else None
   vals=[]
   for q in item['boxes_xyxy']:
    ix=max(0,min(x1,q[2])-max(x0,q[0]));iy=max(0,min(y1,q[3])-max(y0,q[1]));inter=ix*iy;union=(x1-x0)*(y1-y0)+(q[2]-q[0])*(q[3]-q[1])-inter;vals.append(inter/union if union else 0.)
   iou=max(vals) if present else None;false=True if not present else None
  if stat['detected']!=(box is not None) or stat['empty_false_alarm']!=false or (stat['best_gt_box_iou'] is None)!=(iou is None) or iou is not None and abs(stat['best_gt_box_iou']-iou)>1e-12 or (stat['mask_coverage'] is None)!=(coverage is None) or coverage is not None and abs(stat['mask_coverage']-coverage)>1e-12:raise ValueError('Per-image score differs')
  scores[name]={'detected':box is not None,'iou':iou,'coverage':coverage,'false':false}
 computed.append({'present':present,'scores':scores})
pos=[r for r in computed if r['present']];neg=[r for r in computed if not r['present']]
for name in ('original','candidate'):
 expected={'positive_detections':sum(r['scores'][name]['detected'] for r in pos),'positive_total':len(pos),'mean_best_box_iou':float(np.mean([r['scores'][name]['iou'] for r in pos])),'mean_mask_coverage':float(np.mean([r['scores'][name]['coverage'] for r in pos])),'empty_false_alarms':sum(r['scores'][name]['false'] for r in neg),'empty_total':len(neg)}
 for k,v in expected.items():
  if abs(d['summary'][name][k]-v)>1e-12:raise ValueError('Aggregate differs')
diff=np.array([r['scores']['candidate']['iou']-r['scores']['original']['iou'] for r in pos]);expected={'candidate_wins':int((diff>0).sum()),'ties':int((diff==0).sum()),'candidate_losses':int((diff<0).sum()),'mean_iou_difference':float(diff.mean())}
for k,v in expected.items():
 if abs(d['paired'][k]-v)>1e-12:raise ValueError('Paired difference differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'images':len(computed),'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('All 88 per-image and aggregate detector metrics independently recomputed.')
