#!/usr/bin/env python3
"""Independently recheck supplied box annotations and YOLO conversion."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics.utils.ops import xyxy2xywhn,xywhn2xyxy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('audit','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();done=read(a.audit/'completion.json');d=read(a.audit/'results.json')
if not done['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or d['runner_sha256']!=digest(ROOT/'scripts/audit_polypgen_detector_boxes.py'):raise ValueError('Audit changed')
counts={};checked=0;max_error=0.
for r in d['records']:
 image=Path(r['image']);mask=Path(r['mask']) if r['mask'] else None;c=r['center'];key=str(c)
 if digest(image)!=r['image_sha256'] or (mask and digest(mask)!=r['mask_sha256']):raise ValueError('Image or mask changed')
 with Image.open(image) as im:w,h=im.size
 present=None;gt=None
 if mask:
  with Image.open(mask) as im:m=np.asarray(im.convert('L'))
  gt=m>(0 if m.max()<=1 else 127);present=bool(gt.any())
 if r['mask_present']!=present:raise ValueError('Mask class differs')
 folder=image.parent.parent/f'bbox_C{c}';files=[x for x in (folder/(image.stem+'.txt'),folder/(image.stem+'_mask.txt')) if x.is_file()]
 if [str(x) for x in files]!=r['box_files'] or [digest(x) for x in files]!=r['box_file_sha256']:raise ValueError('Box file changed')
 boxes=[];reason=None
 if not files:status='missing'
 elif len(files)>1:status='ambiguous'
 else:
  for line in files[0].read_text(errors='strict').splitlines():
   f=line.split()
   if not f:continue
   if len(f)!=5 or f[0]!='polyp':reason='malformed';break
   try:x=np.array([float(v) for v in f[1:]],dtype=np.float64)
   except ValueError:reason='malformed';break
   if not np.isfinite(x).all():reason='malformed';break
   if not (0<=x[0]<x[2]<=w and 0<=x[1]<x[3]<=h):reason='out_of_bounds';break
   boxes.append(x.tolist())
  status=reason or ('boxes' if boxes else 'empty')
 if status!=r['status'] or boxes!=r['boxes']:raise ValueError('Box status or coordinates differ')
 cov=None
 if status in ('boxes','empty') and gt is not None and gt.any():
  cover=np.zeros((h,w),bool)
  for x0,y0,x1,y1 in boxes:cover[int(np.floor(y0)):int(np.ceil(y1)),int(np.floor(x0)):int(np.ceil(x1))]=1
  cov=float(np.sum(cover&gt)/np.sum(gt))
 if (cov is None)!=(r['foreground_coverage'] is None) or cov is not None and abs(cov-r['foreground_coverage'])>1e-12:raise ValueError('Mask coverage differs')
 if status=='boxes':
  xy=np.array(boxes,dtype=np.float64);norm=xyxy2xywhn(xy.copy(),w=w,h=h,clip=False);back=xywhn2xyxy(norm.copy(),w=w,h=h)
  err=float(np.abs(back-xy).max());max_error=max(max_error,err)
  if not np.isfinite(norm).all() or np.any(norm<0) or np.any(norm>1) or err>1e-9:raise ValueError('YOLO coordinate roundtrip failed')
  checked+=len(boxes)
 counts.setdefault(key,0);counts[key]+=1
if {k:v['images'] for k,v in d['counts'].items()}!=counts:raise ValueError('Counts differ')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'image_counts':counts,'valid_box_count':checked,'max_roundtrip_error_pixels':max_error,'audit_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Existing annotations and official YOLO box-coordinate roundtrips verified.')
