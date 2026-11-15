#!/usr/bin/env python3
"""Stage a provenance-pinned center-separated YOLO detector pilot dataset."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
import yaml
from ultralytics.utils.ops import xyxy2xywhn
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

PROTOCOL=ROOT/'protocols/polypgen_one_shot_detector_pilot_v1.json'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('input-audit','input-verification','box-audit','box-verification','duplicate-audit','duplicate-verification','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();inputs=read(a.input_audit/'results.json');iv=read(a.input_verification/'verification.json');boxes=read(a.box_audit/'results.json');bv=read(a.box_verification/'verification.json');dups=read(a.duplicate_audit/'results.json');dv=read(a.duplicate_verification/'verification.json')
 if not iv['passed'] or iv['results_sha256']!=digest(a.input_audit/'results.json') or iv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_training_inputs.py') or not bv['passed'] or bv['audit_sha256']!=digest(a.box_audit/'results.json') or bv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_boxes.py') or not dv['passed'] or dv['results_sha256']!=digest(a.duplicate_audit/'results.json') or dv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_duplicate_annotations.py') or boxes['input_sha256']!=digest(a.input_audit/'results.json') or dups['box_audit_sha256']!=digest(a.box_audit/'results.json') or inputs['overlaps']:raise ValueError('Verified input provenance changed')
 single={x['image']:x for x in inputs['single_frame_files']};excluded_duplicates={item['image'] for group in dups['groups'] for item in group['items']}
 if set(single)!={x['image'] for x in boxes['records']}:raise ValueError('Input coverage changed')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 for split in ('train','val'):
  (out/'images'/split).mkdir(parents=True);(out/'labels'/split).mkdir(parents=True)
 included=[];excluded=[]
 for rec in boxes['records']:
  src=Path(rec['image']);meta=single[str(src)];center=rec['center'];mask=Path(rec['mask']) if rec['mask'] else None;reasons=[]
  if str(src) in excluded_duplicates:reasons.append('exact_duplicate_with_different_annotations')
  if mask is None:reasons.append('missing_mask')
  if rec['status'] not in ('boxes','empty'):reasons.append('box_'+rec['status'])
  if rec['mask_present'] is True and rec['status']=='empty':reasons.append('positive_mask_empty_box')
  if rec['mask_present'] is False and rec['status']=='boxes':reasons.append('empty_mask_positive_box')
  if rec['mask_present'] is True and rec['foreground_coverage'] is not None and rec['foreground_coverage']<.99:reasons.append('box_coverage_below_0.99')
  if reasons:excluded.append({'image':str(src),'center':center,'reasons':reasons});continue
  if digest(src)!=rec['image_sha256'] or mask is None or digest(mask)!=rec['mask_sha256'] or any(digest(Path(x))!=sha for x,sha in zip(rec['box_files'],rec['box_file_sha256'],strict=True)):raise ValueError('Source changed during staging')
  with Image.open(src) as im:
   w,h=im.size
   if im.getexif().get(274,1)!=1:raise ValueError('Unexpected image orientation')
  with src.open('rb') as f:
   f.seek(-2,2)
   if f.read()!=b'\xff\xd9':raise ValueError('Incomplete JPEG source')
  split='val' if center==6 else 'train';name=f'C{center}_{src.stem}';staged=out/'images'/split/(name+'.jpg');label=out/'labels'/split/(name+'.txt')
  if staged.exists() or label.exists():raise ValueError('Staged filename collision')
  shutil.copyfile(src,staged)
  if staged.is_symlink() or digest(staged)!=rec['image_sha256']:raise ValueError('Staged image copy differs')
  if rec['boxes']:
   vals=xyxy2xywhn(np.asarray(rec['boxes'],dtype=np.float64).copy(),w=w,h=h,clip=False)
   if not np.isfinite(vals).all() or (vals<0).any() or (vals>1).any():raise ValueError('Invalid normalized box')
   label.write_text(''.join('0 '+' '.join(f'{float(v):.12f}' for v in row)+'\n' for row in vals))
  else:label.write_text('')
  included.append({'split':split,'center':center,'source_image':str(src),'image_sha256':rec['image_sha256'],'pixel_sha256':meta['pixel_sha256'],'source_mask':str(mask),'mask_sha256':rec['mask_sha256'],'source_boxes':rec['box_files'],'source_boxes_sha256':rec['box_file_sha256'],'boxes_xyxy':rec['boxes'],'image_width':w,'image_height':h,'staged_image':str(staged),'staged_label':str(label),'label_sha256':digest(label)})
 splits={s:[r for r in included if r['split']==s] for s in ('train','val')}
 if not all(splits.values()) or {x['center'] for x in splits['val']}!={6} or any(x['center']==6 for x in splits['train']) or {x['pixel_sha256'] for x in splits['train']}&{x['pixel_sha256'] for x in splits['val']}:raise ValueError('Center split or exact image separation failed')
 config={'path':str(out),'train':'images/train','val':'images/val','names':{0:'polyp'}};(out/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 result={'included':included,'excluded':excluded,'split_counts':{s:{'images':len(rows),'positive':sum(bool(x['boxes_xyxy']) for x in rows),'negative':sum(not x['boxes_xyxy'] for x in rows)} for s,rows in splits.items()},'input_sha256':digest(a.input_audit/'results.json'),'box_audit_sha256':digest(a.box_audit/'results.json'),'duplicate_audit_sha256':digest(a.duplicate_audit/'results.json'),'protocol_sha256':digest(PROTOCOL),'config_sha256':digest(out/'data.yaml'),'runner_sha256':digest(Path(__file__))}
 (out/'dataset.json').write_text(json.dumps(result,indent=2)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'dataset_sha256':digest(out/'dataset.json'),'config_sha256':digest(out/'data.yaml')},indent=2)+'\n');print(json.dumps({'passed':True,'split_counts':result['split_counts'],'excluded':len(excluded)},indent=2))
if __name__=='__main__':main()
