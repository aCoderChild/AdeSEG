#!/usr/bin/env python3
"""Read-only forensic audit of detector training candidates and video overlap."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_binary_mask
from scripts.memory_token_io import digest,validate_output

DATA=ROOT/'data/PolypGen2021_MultiCenterData_v3';MANIFEST=ROOT/'data/polypgen_sequence.jsonl'
def pixels(path):
 with Image.open(path) as image:
  array=np.asarray(image.convert('RGB'));shape=array.shape
  return hashlib.sha256(array.tobytes()).hexdigest(),shape[:2]

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 from ultralytics import YOLO
 detector=YOLO(str(a.checkpoint));ckpt=detector.ckpt;train=ckpt.get('train_args') or {}
 videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')};video_bytes={};video_pixels={};video_summary={}
 for name,frames in sorted(videos.items()):
  for i,frame in enumerate(frames):
   b=digest(frame.image_path);h,shape=pixels(frame.image_path);entry={'sequence':name,'frame':i,'path':str(frame.image_path),'shape':list(shape)}
   video_bytes.setdefault(b,[]).append(entry);video_pixels.setdefault(h,[]).append(entry)
  video_summary[name]=len(frames)
 centers={};matches=[];files=[]
 for center in range(1,7):
  directory=DATA/f'data_C{center}';images=directory/f'images_C{center}';masks=directory/f'masks_C{center}'
  counts={'images':0,'paired_masks':0,'missing_masks':0,'dimension_mismatches':0,'empty_masks':0,'present_masks':0,'mask_errors':0,'exact_byte_images_overlapping_video':0,'exact_pixel_images_overlapping_video':0};missing=[];errors=[]
  for image in sorted(images.glob('*.jpg')):
   counts['images']+=1;byte=digest(image);pixel,shape=pixels(image);mask=masks/(image.stem+'_mask.jpg')
   if not mask.is_file():counts['missing_masks']+=1;missing.append(str(image))
   else:
    counts['paired_masks']+=1
    try:
     m=load_binary_mask(mask)
     if m.shape!=shape:counts['dimension_mismatches']+=1
     if np.any(m):counts['present_masks']+=1
     else:counts['empty_masks']+=1
    except (OSError,ValueError) as exc:counts['mask_errors']+=1;errors.append({'image':str(image),'error':type(exc).__name__})
   bm=video_bytes.get(byte,[]);pm=video_pixels.get(pixel,[])
   counts['exact_byte_images_overlapping_video']+=bool(bm);counts['exact_pixel_images_overlapping_video']+=bool(pm)
   if bm or pm:matches.append({'center':center,'image':str(image),'image_sha256':byte,'pixel_sha256':pixel,'byte_matches':bm,'pixel_matches':pm})
   files.append({'center':center,'image':str(image),'image_sha256':byte,'pixel_sha256':pixel,'mask':str(mask) if mask.is_file() else None})
  centers[str(center)]={'counts':counts,'missing':missing,'errors':errors}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'Read-only forensic audit; decoded RGB equality is exact at original resolution. No sequence labels were opened.','detector_sha256':digest(a.checkpoint),'detector_checkpoint_metadata':{'epoch':ckpt.get('epoch'),'date':ckpt.get('date'),'version':ckpt.get('version'),'task':detector.task,'names':detector.names,'data_argument':train.get('data'),'model_argument':train.get('model'),'epochs_argument':train.get('epochs'),'image_size_argument':train.get('imgsz'),'seed_argument':train.get('seed')},'manifest_sha256':digest(MANIFEST),'dataset_readme_sha256':digest(DATA/'readme.md'),'video_frame_counts':video_summary,'centers':centers,'overlaps':matches,'single_frame_files':files,'runner_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# PolypGen detector input provenance and exact overlap','','The existing detector checkpoint records a `polypgen.yaml` data argument, but that YAML and the identity of its training examples have not been recovered. Checkpoint metadata alone cannot establish the training split. This audit compares candidate center-wise single-frame RGB images against the sequence manifest without reading sequence masks. Exact file-byte and exact decoded-pixel equality were checked; absence of an exact match does not exclude different frames from the same video, re-encoding, patient overlap or approximate visual duplicates.','','| Center | Images | Paired masks | Missing masks | Dimension mismatch | Present / empty masks | Exact-pixel images matching any sequence frame |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for c,v in centers.items():
  z=v['counts'];lines.append(f"| {c} | {z['images']} | {z['paired_masks']} | {z['missing_masks']} | {z['dimension_mismatches']} | {z['present_masks']} / {z['empty_masks']} | {z['exact_pixel_images_overlapping_video']} |")
 priority=('seq8','seq11','seq20','seq23')
 for name in priority:
  hits=[m for m in matches if any(e['sequence']==name for e in m['pixel_matches'])];lines.append(f'\n{name}: {len(hits)} single-frame images match one or more video frames exactly in decoded RGB.')
 lines+=['','Patient and original-video identifiers were not established by this exact-image comparison. The reported metadata are not evidence that any particular split was used to train the checkpoint. Additional detector training or evaluation must avoid claiming patient independence without stronger provenance.','']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
