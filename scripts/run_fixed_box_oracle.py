#!/usr/bin/env python3
"""Explicit target-label box oracle, only after both automatic predictions exist."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.seq23_viewport import crop_viewport,restore_mask,VIEWPORT
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import parameter_fingerprints,assert_parameter_integrity
from scripts.run_prompt_feature_refinement import REPO,gt_box,predict_mask

def read(x):return json.loads(x.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('fixed-a','fixed-b','gate-verification','checkpoint','preflight','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();ra=read(a.fixed_a/'completion.json');rb=read(a.fixed_b/'completion.json');g=read(a.gate_verification/'verification.json')
 if not all(r['passed'] and r['mode']=='target' for r in (ra,rb)) or ra['identity']!=rb['identity'] or ra['events']!=rb['events'] or len(ra['events'])!=20:raise ValueError('Fixed predictions differ')
 if not g['passed'] or g['identity']!=ra['identity'] or g['verifier_sha256']!=digest(ROOT/'scripts/verify_fixed_box_prompt_gates.py') or ra['identity']['checkpoint_sha256']!=digest(a.checkpoint) or ra['identity']['runner_sha256']!=digest(ROOT/'scripts/run_fixed_box_prompt_diagnostic.py'):raise ValueError('Gate or checkpoint changed')
 for folder in (a.fixed_a,a.fixed_b):
  for e in ra['events']:
   if digest(folder/f"{e['sequence']}_{e['frame']:06d}_fixed_initial.png")!=e['masks']['fixed_initial']:raise ValueError('Fixed prediction changed')
 manifest=read(a.preflight/'manifest.json')
 if digest(a.preflight/'manifest.json')!=ra['identity']['preflight_sha256']:raise ValueError('Selection changed')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(20261114)
 sys.path.insert(0,str(REPO))
 from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(a.checkpoint),device='cpu',mode='eval',apply_postprocessing=False).eval().requires_grad_(False)
 before=parameter_fingerprints(model);predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0);events=[]
 with torch.inference_mode():
  for e in ra['events']:
   seq=e['sequence'];i=e['frame'];frame=videos[seq][i];meta=manifest['sequences'][seq];folder=Path(meta['source'])/seq
   if digest(folder/'frames.json')!=meta['frames_metadata_sha256']:raise ValueError('Frame metadata changed')
   rows=read(folder/'frames.json')
   if digest(frame.image_path)!=e['image_sha256'] or digest(frame.mask_path)!=rows[i]['label_sha256']:raise ValueError('Image or label changed')
   gt=load_label_mask(frame.mask_path,[1])==1
   with Image.open(frame.image_path) as im:image=np.array(im.convert('RGB'),copy=True)
   if seq=='seq23':
    x0,y0,x1,y1=VIEWPORT
    if np.any(gt & ~restore_mask(gt[y0:y1,x0:x1])):raise ValueError('Labeled foreground outside viewport')
    gt=gt[y0:y1,x0:x1];image=crop_viewport(image)
   box=gt_box(gt);item={'sequence':seq,'frame':i,'label_sha256':digest(frame.mask_path),'oracle_box':box,'mask_sha256':None}
   if box is not None:
    predictor.set_image(image);mask=predict_mask(predictor,image.shape[:2],box)
    if seq=='seq23':mask=restore_mask(mask)
    path=out/f'{seq}_{i:06d}_oracle.png';Image.fromarray(mask.astype(np.uint8)*255).save(path);item['mask_sha256']=digest(path)
   events.append(item)
 assert_parameter_integrity(model,before,out,'after')
 result={'passed':True,'identity':ra['identity'],'fixed_completion_sha256':digest(a.fixed_a/'completion.json'),'fixed_repeat_sha256':digest(a.fixed_b/'completion.json'),'events':events,'runner_sha256':digest(Path(__file__)),'scope':'Explicit label-derived box oracle only; not automatic inference or propagation.'}
 (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'cases':len(events),'oracle_present':sum(x['oracle_box'] is not None for x in events)}))
if __name__=='__main__':main()
