#!/usr/bin/env python3
"""Independently recompute single-frame mask coverage and exact video overlap."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output

DATA=ROOT/'data/PolypGen2021_MultiCenterData_v3';MANIFEST=ROOT/'data/polypgen_sequence.jsonl'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def picture(path):
 with Image.open(path) as f:
  rgb=f.convert('RGB');return hashlib.sha256(rgb.tobytes()).hexdigest(),(rgb.height,rgb.width)
p=argparse.ArgumentParser()
for k in ('audit','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();done=json.loads((a.audit/'completion.json').read_text());d=json.loads((a.audit/'results.json').read_text())
if not done['passed'] or done['results_sha256']!=sha(a.audit/'results.json') or done['report_sha256']!=sha(a.audit/'report.md') or d['runner_sha256']!=sha(ROOT/'scripts/audit_polypgen_detector_training_inputs.py') or d['manifest_sha256']!=sha(MANIFEST) or d['dataset_readme_sha256']!=sha(DATA/'readme.md') or d['detector_sha256']!=sha(a.checkpoint):raise ValueError('Source changed')
videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')};all_byte={};all_pixel={}
if {k:len(v) for k,v in videos.items()}!=d['video_frame_counts']:raise ValueError('Video scope changed')
for name,frames in sorted(videos.items()):
 for i,f in enumerate(frames):
  pix,shape=picture(f.image_path);value=(name,i,str(f.image_path),shape)
  all_byte.setdefault(sha(f.image_path),[]).append(value);all_pixel.setdefault(pix,[]).append(value)
record={x['image']:x for x in d['single_frame_files']}
if len(record)!=len(d['single_frame_files']):raise ValueError('Duplicate image records')
seen=set();count_per_center={};matched=[]
for center in range(1,7):
 path=DATA/f'data_C{center}';images=path/f'images_C{center}';masks=path/f'masks_C{center}';counts={'images':0,'paired_masks':0,'missing_masks':0,'dimension_mismatches':0,'empty_masks':0,'present_masks':0,'mask_errors':0,'exact_byte_images_overlapping_video':0,'exact_pixel_images_overlapping_video':0}
 for image in sorted(images.glob('*.jpg')):
  key=str(image);seen.add(key);item=record[key];h=sha(image);pix,shape=picture(image);mask=masks/(image.stem+'_mask.jpg');counts['images']+=1
  if item['center']!=center or item['image_sha256']!=h or item['pixel_sha256']!=pix or item['mask']!=(str(mask) if mask.is_file() else None):raise ValueError('Single frame changed')
  if not mask.is_file():counts['missing_masks']+=1
  else:
   counts['paired_masks']+=1
   try:
    with Image.open(mask) as im:arr=np.asarray(im.convert('L'))
    if arr.shape!=shape:counts['dimension_mismatches']+=1
    binary=arr>(0 if arr.max()<=1 else 127)
    counts['present_masks' if binary.any() else 'empty_masks']+=1
   except (OSError,ValueError):counts['mask_errors']+=1
  bm=all_byte.get(h,[]);pm=all_pixel.get(pix,[]);counts['exact_byte_images_overlapping_video']+=bool(bm);counts['exact_pixel_images_overlapping_video']+=bool(pm)
  if bm or pm:matched.append((center,key,sorted(bm),sorted(pm)))
 if counts!=d['centers'][str(center)]['counts']:raise ValueError('Center count differs')
 count_per_center[str(center)]=counts['images']
if seen!=set(record):raise ValueError('Single-image coverage changed')
reported=[]
for x in d['overlaps']:
 def fmt(k):return sorted((y['sequence'],y['frame'],y['path'],tuple(y['shape'])) for y in x[k])
 reported.append((x['center'],x['image'],fmt('byte_matches'),fmt('pixel_matches')))
if sorted(matched)!=sorted(reported):raise ValueError('Video overlap differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'video_frames':sum(map(len,videos.values())),'single_frame_counts':count_per_center,'overlapping_single_images':len(matched),'results_sha256':sha(a.audit/'results.json'),'verifier_sha256':sha(Path(__file__))},indent=2)+'\n');print('All video and single-frame hashes, pairings, mask counts, and exact matches independently verified.')
