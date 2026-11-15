#!/usr/bin/env python3
"""Decode explicit GT-box controls only after automatic predictions are frozen."""
import argparse,json,subprocess,sys
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
from scripts.run_prompt_feature_refinement import REPO,REVISION,predict_mask,gt_box

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('checkpoint','checkpoint-verification','predictions','prediction-verification','masks','mask-verification','preflight','crop-reference','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();ck=read(a.checkpoint_verification/'verification.json');pred=read(a.predictions/'completion.json');pv=read(a.prediction_verification/'verification.json');masks=read(a.masks/'completion.json');mv=read(a.mask_verification/'verification.json');pre=read(a.preflight/'manifest.json');crop=read(a.crop_reference/'completion.json')
 if not ck['passed'] or ck['identity']['checkpoint_sha256']!=digest(a.checkpoint) or ck['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or not pred['passed'] or not pv['passed'] or pv['predictions_sha256']!=digest(a.predictions/'completion.json') or pv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_original_finetune_predictions.py') or not masks['passed'] or not mv['passed'] or mv['mask_receipt_sha256']!=digest(a.masks/'completion.json') or mv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_original_finetune_masks.py') or masks['predictions_sha256']!=pv['predictions_sha256'] or pred['preflight_sha256']!=digest(a.preflight/'manifest.json') or not crop['passed'] or crop['frames_sha256']!=digest(a.crop_reference/'frames.json'):raise ValueError('Automatic outputs or model changed')
 rev=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True)
 if rev!=REVISION or dirty:raise ValueError('SAM2 reference source changed')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(20261114);sys.path.insert(0,str(REPO))
 import sam2
 if not Path(sam2.__file__).resolve().is_relative_to(REPO.resolve()):raise ValueError('Wrong SAM2 source')
 from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(a.checkpoint),device='cpu',mode='eval',apply_postprocessing=False).eval().requires_grad_(False);before=parameter_fingerprints(model);predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0);events=[]
 with torch.inference_mode():
  for e in pred['events']:
   seq=e['sequence'];i=e['frame'];frame=videos[seq][i];meta=pre['sequences'][seq];source=Path(meta['source'])/seq
   if digest(source/'frames.json')!=meta['frames_metadata_sha256'] or digest(frame.image_path)!=e['image_sha256']:raise ValueError('Source changed')
   rows=read(a.crop_reference/'frames.json') if e['viewport'] else read(source/'frames.json')
   if digest(frame.mask_path)!=rows[i]['label_sha256']:raise ValueError('Target label changed')
   gt=load_label_mask(frame.mask_path,[1])==1
   with Image.open(frame.image_path) as im:image=np.array(im.convert('RGB'),copy=True)
   if e['viewport']:
    x0,y0,x1,y1=VIEWPORT
    if gt[:,:x0].any():raise ValueError('Label outside viewport')
    gt=gt[y0:y1,x0:x1];image=crop_viewport(image)
   box=gt_box(gt);masksha=None
   if box is not None:
    predictor.set_image(image);mask=predict_mask(predictor,image.shape[:2],box)
    if e['viewport']:mask=restore_mask(mask)
    path=out/f'{seq}_{i:06d}_oracle_box.png';Image.fromarray(mask.astype(np.uint8)*255).save(path);masksha=digest(path)
   events.append({'sequence':seq,'frame':i,'label_sha256':rows[i]['label_sha256'],'oracle_box_viewport':box,'mask_sha256':masksha})
 assert_parameter_integrity(model,before,out,'after')
 result={'passed':True,'events':events,'sam2_checkpoint_sha256':digest(a.checkpoint),'predictions_sha256':digest(a.predictions/'completion.json'),'mask_receipt_sha256':digest(a.masks/'completion.json'),'protocol_sha256':digest(ROOT/'protocols/polypgen_original_detector_finetune_v1.json'),'runner_sha256':digest(Path(__file__)),'scope':'Explicit GT box oracle only; not automatic inference or video propagation.'}
 (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'frames':len(events),'present_oracle':sum(e['oracle_box_viewport'] is not None for e in events)}))
if __name__=='__main__':main()
