#!/usr/bin/env python3
"""Audit existing single-frame PolypGen VOC-style text box annotations."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_binary_mask
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('input-audit','verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();d=read(a.input_audit/'results.json');done=read(a.input_audit/'completion.json');v=read(a.verification/'verification.json')
if not done['passed'] or not v['passed'] or done['results_sha256']!=digest(a.input_audit/'results.json') or v['results_sha256']!=done['results_sha256'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_training_inputs.py'):raise ValueError('Input preflight changed')
results=[];counts={}
for item in d['single_frame_files']:
 c=item['center'];key=str(c);image=Path(item['image']);mask=Path(item['mask']) if item['mask'] else None
 if digest(image)!=item['image_sha256']:raise ValueError('Image changed')
 with Image.open(image) as im:w,h=im.size
 mask_data=load_binary_mask(mask).astype(bool) if mask else None
 if mask_data is not None and mask_data.shape!=(h,w):raise ValueError('Mask geometry changed')
 folder=image.parent.parent/f'bbox_C{c}';candidates=[x for x in (folder/(image.stem+'.txt'),folder/(image.stem+'_mask.txt')) if x.is_file()]
 rec={'center':c,'image':str(image),'image_sha256':item['image_sha256'],'mask':str(mask) if mask else None,'mask_sha256':digest(mask) if mask else None,'box_files':[str(x) for x in candidates],'box_file_sha256':[digest(x) for x in candidates],'status':None,'mask_present':None if mask_data is None else bool(mask_data.any()),'boxes':[],'foreground_coverage':None}
 if len(candidates)==0:rec['status']='missing'
 elif len(candidates)>1:rec['status']='ambiguous'
 else:
  path=candidates[0];parsed=[];reason=None
  for line in path.read_text(errors='strict').splitlines():
   fields=line.split()
   if not fields:continue
   if len(fields)!=5 or fields[0]!='polyp':reason='malformed';break
   try:coords=[float(x) for x in fields[1:]]
   except ValueError:reason='malformed';break
   if not all(np.isfinite(coords)):reason='malformed';break
   x0,y0,x1,y1=coords
   if not (0<=x0<x1<=w and 0<=y0<y1<=h):reason='out_of_bounds';break
   parsed.append(coords)
  rec['status']=reason or ('boxes' if parsed else 'empty');rec['boxes']=parsed
  if reason is None and mask_data is not None and mask_data.any():
   cover=np.zeros((h,w),dtype=bool)
   for x0,y0,x1,y1 in parsed:cover[int(np.floor(y0)):int(np.ceil(y1)),int(np.floor(x0)):int(np.ceil(x1))]=True
   rec['foreground_coverage']=float(np.count_nonzero(cover & mask_data)/np.count_nonzero(mask_data))
 counts.setdefault(key,{'images':0,'boxes':0,'empty':0,'missing':0,'ambiguous':0,'malformed':0,'out_of_bounds':0,'missing_mask':0,'positive_mask_empty_box':0,'empty_mask_positive_box':0,'positive_mask_missing_box':0,'positive_mask_coverage_lt_99pct':0})
 n=counts[key];n['images']+=1;n[rec['status']]+=1;n['missing_mask']+=mask_data is None
 if mask_data is not None:
  present=bool(mask_data.any());n['positive_mask_empty_box']+=present and rec['status']=='empty';n['empty_mask_positive_box']+=(not present) and rec['status']=='boxes';n['positive_mask_missing_box']+=present and rec['status']=='missing';n['positive_mask_coverage_lt_99pct']+=rec['foreground_coverage'] is not None and rec['foreground_coverage']<.99
 results.append(rec)
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
summary={'scope':'Read-only audit of candidate single-frame annotations; VOC x/y box convention assumed half-open for coverage check. Threshold 0.99 is a post-hoc annotation audit, not an inference gate.','input_sha256':digest(a.input_audit/'results.json'),'verification_sha256':digest(a.verification/'verification.json'),'runner_sha256':digest(Path(__file__)),'counts':counts,'records':results}
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Supplied single-frame detector-box audit','',summary['scope'],'','| Center | Images | Box files | Empty box | Missing box | Invalid or ambiguous | Positive mask, no box file | Positive mask, empty box | Empty mask, nonempty box | Positive mask coverage below 99% |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for c,n in counts.items():
 invalid=n['ambiguous']+n['malformed']+n['out_of_bounds'];lines.append(f"| {c} | {n['images']} | {n['boxes']} | {n['empty']} | {n['missing']} | {invalid} | {n['positive_mask_missing_box']} | {n['positive_mask_empty_box']} | {n['empty_mask_positive_box']} | {n['positive_mask_coverage_lt_99pct']} |")
lines+=['','Images without an associated box file must not automatically become negative training examples. Mask-coverage differences can reflect annotation conventions, distinct instances, incomplete labels or coordinate format; they require investigation rather than automatic correction. The checkpoint’s original training split and patient overlap remain unknown.','']
(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
