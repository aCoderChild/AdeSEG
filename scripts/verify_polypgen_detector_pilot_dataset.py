#!/usr/bin/env python3
"""Independently verify curated split and each normalized detector label."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('dataset','input-audit','box-audit','duplicate-audit','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();d=read(a.dataset/'dataset.json');done=read(a.dataset/'completion.json');inputs=read(a.input_audit/'results.json');boxes=read(a.box_audit/'results.json');dups=read(a.duplicate_audit/'results.json')
if not done['passed'] or done['dataset_sha256']!=digest(a.dataset/'dataset.json') or done['config_sha256']!=digest(a.dataset/'data.yaml') or d['config_sha256']!=done['config_sha256'] or d['input_sha256']!=digest(a.input_audit/'results.json') or d['box_audit_sha256']!=digest(a.box_audit/'results.json') or d['duplicate_audit_sha256']!=digest(a.duplicate_audit/'results.json') or d['runner_sha256']!=digest(ROOT/'scripts/prepare_polypgen_detector_pilot.py') or d['protocol_sha256']!=digest(ROOT/'protocols/polypgen_one_shot_detector_pilot_v1.json'):raise ValueError('Dataset provenance changed')
cfg=yaml.safe_load((a.dataset/'data.yaml').read_text())
if cfg!={'path':str(a.dataset.resolve()),'train':'images/train','val':'images/val','names':{0:'polyp'}}:raise ValueError('Ultralytics dataset configuration changed')
box_by={r['image']:r for r in boxes['records']};single={r['image']:r for r in inputs['single_frame_files']};dupe={x['image'] for g in dups['groups'] for x in g['items']};inc={r['source_image']:r for r in d['included']};exc={r['image']:r for r in d['excluded']}
if set(inc)&set(exc) or set(inc)|set(exc)!=set(box_by) or set(box_by)!=set(single) or inputs['overlaps']:raise ValueError('Source coverage differs')
for src,record in box_by.items():
 reasons=[]
 if src in dupe:reasons.append('exact_duplicate_with_different_annotations')
 if record['mask'] is None:reasons.append('missing_mask')
 if record['status'] not in ('boxes','empty'):reasons.append('box_'+record['status'])
 if record['mask_present'] is True and record['status']=='empty':reasons.append('positive_mask_empty_box')
 if record['mask_present'] is False and record['status']=='boxes':reasons.append('empty_mask_positive_box')
 if record['mask_present'] is True and record['foreground_coverage'] is not None and record['foreground_coverage']<.99:reasons.append('box_coverage_below_0.99')
 if bool(reasons)!=(src in exc) or reasons and exc[src]['reasons']!=reasons:raise ValueError('Exclusion policy changed')
counts={s:{'images':0,'positive':0,'negative':0} for s in ('train','val')};expected_files={s:{'images':set(),'labels':set()} for s in ('train','val')};pixels={s:set() for s in ('train','val')};max_error=0.
for src,r in inc.items():
 item=box_by[src];split='val' if item['center']==6 else 'train';image=Path(src);mask=Path(r['source_mask']);staged=Path(r['staged_image']);label=Path(r['staged_label'])
 if r['split']!=split or r['center']!=item['center'] or staged.is_symlink() or not staged.is_file() or staged.stat().st_ino==image.stat().st_ino or not staged.is_relative_to(a.dataset/'images'/split) or not label.is_relative_to(a.dataset/'labels'/split):raise ValueError('Split or staged image copy differs')
 if digest(image)!=r['image_sha256'] or digest(staged)!=r['image_sha256'] or digest(mask)!=r['mask_sha256'] or digest(label)!=r['label_sha256'] or r['pixel_sha256']!=single[src]['pixel_sha256'] or r['boxes_xyxy']!=item['boxes'] or [digest(Path(x)) for x in r['source_boxes']]!=r['source_boxes_sha256']:raise ValueError('Source, staged image or label changed')
 with Image.open(image) as im:
  w,h=im.size
  if im.getexif().get(274,1)!=1:raise ValueError('Unexpected image orientation')
 with staged.open('rb') as f:
  f.seek(-2,2)
  if f.read()!=b'\xff\xd9':raise ValueError('Incomplete staged JPEG')
 if (w,h)!=(r['image_width'],r['image_height']):raise ValueError('Image size differs')
 lines=[x.split() for x in label.read_text().splitlines() if x.strip()]
 if len(lines)!=len(item['boxes']):raise ValueError('Box count differs')
 for fields,box in zip(lines,item['boxes'],strict=True):
  if len(fields)!=5 or fields[0]!='0':raise ValueError('Class or label malformed')
  xywh=np.array([float(z) for z in fields[1:]],dtype=np.float64)
  if not np.isfinite(xywh).all() or (xywh<0).any() or (xywh>1).any():raise ValueError('Normalized label invalid')
  x,y,bw,bh=xywh;decoded=np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]);err=float(np.abs(decoded-np.asarray(box)).max());max_error=max(max_error,err)
  if err>1e-6:raise ValueError('Converted box differs')
 counts[split]['images']+=1;counts[split]['positive' if lines else 'negative']+=1;pixels[split].add(r['pixel_sha256']);expected_files[split]['images'].add(staged);expected_files[split]['labels'].add(label)
if pixels['train']&pixels['val'] or counts!=d['split_counts']:raise ValueError('Split collision or counts differ')
for split in ('train','val'):
 if set((a.dataset/'images'/split).glob('*.jpg'))!=expected_files[split]['images'] or set((a.dataset/'labels'/split).glob('*.txt'))!=expected_files[split]['labels']:raise ValueError('Unexpected staged files')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'dataset_sha256':digest(a.dataset/'dataset.json'),'config_sha256':digest(a.dataset/'data.yaml'),'split_counts':counts,'excluded':len(exc),'max_box_error_pixels':max_error,'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print(json.dumps({'passed':True,'split_counts':counts,'excluded':len(exc),'max_box_error_pixels':max_error}))
