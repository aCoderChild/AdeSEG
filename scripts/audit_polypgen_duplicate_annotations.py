#!/usr/bin/env python3
"""Compare labels for exactly duplicated single-frame RGB inputs."""
import argparse,collections,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_binary_mask
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('input-audit','input-verification','box-audit','box-verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();d=read(a.input_audit/'results.json');iv=read(a.input_verification/'verification.json');boxes=read(a.box_audit/'results.json');bv=read(a.box_verification/'verification.json')
if not iv['passed'] or iv['results_sha256']!=digest(a.input_audit/'results.json') or iv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_training_inputs.py') or not bv['passed'] or bv['audit_sha256']!=digest(a.box_audit/'results.json') or bv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_boxes.py') or boxes['input_sha256']!=digest(a.input_audit/'results.json'):raise ValueError('Preflight evidence changed')
by_image={x['image']:x for x in boxes['records']};groups=collections.defaultdict(list)
for r in d['single_frame_files']:groups[r['pixel_sha256']].append(r)
results=[]
for pixel,items in sorted(groups.items()):
 if len(items)<2:continue
 rows=[];image_arrays=[];mask_arrays=[]
 for item in items:
  rec=by_image[item['image']];image=Path(item['image']);mask=Path(rec['mask']) if rec['mask'] else None
  if digest(image)!=rec['image_sha256'] or not mask or digest(mask)!=rec['mask_sha256']:raise ValueError('Duplicate input or label changed')
  with Image.open(image) as f:image_arrays.append(np.asarray(f.convert('RGB')).copy())
  mask_arrays.append(load_binary_mask(mask).astype(bool));rows.append({'center':item['center'],'image':str(image),'mask':str(mask),'image_sha256':rec['image_sha256'],'mask_sha256':rec['mask_sha256'],'box_file_sha256':rec['box_file_sha256'],'boxes':rec['boxes']})
 if any(not np.array_equal(image_arrays[0],x) for x in image_arrays[1:]):raise ValueError('Decoded pixel collision')
 pairs=[]
 for i in range(len(rows)):
  for j in range(i+1,len(rows)):
   x,y=mask_arrays[i],mask_arrays[j]
   if x.shape!=y.shape:raise ValueError('Duplicate mask dimensions differ')
   union=int((x|y).sum());pairs.append({'first':i,'second':j,'mask_iou':float((x&y).sum()/union) if union else 1.,'first_mask_pixels':int(x.sum()),'second_mask_pixels':int(y.sum()),'same_boxes':rows[i]['boxes']==rows[j]['boxes']})
 results.append({'pixel_sha256':pixel,'items':rows,'pairs':pairs})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'results.json').write_text(json.dumps({'groups':results,'input_sha256':digest(a.input_audit/'results.json'),'box_audit_sha256':digest(a.box_audit/'results.json'),'runner_sha256':digest(Path(__file__))},indent=2)+'\n')
lines=['# Exactly duplicated detector-training inputs','','Images are grouped by identical decoded RGB pixels at their original dimensions. Masks and existing boxes are independently hashed; mask IoU is descriptive and uses the repository’s binary JPEG decoding. No automatic decision about which annotation is correct is made.','','| Center | Image pair | Mask IoU | Same box annotations? |','| --- | --- | ---: | :---: |']
for g in results:
 for p0 in g['pairs']:
  x=g['items'][p0['first']];y=g['items'][p0['second']];lines.append(f"| {x['center']} / {y['center']} | {Path(x['image']).name} / {Path(y['image']).name} | {p0['mask_iou']:.6f} | {'yes' if p0['same_boxes'] else 'no'} |")
lines+=['','Identical image inputs with conflicting annotations should not be naively duplicated during training. The source annotations require review; their differing labels do not reveal which is correct.','']
(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
