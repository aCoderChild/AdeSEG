#!/usr/bin/env python3
"""Decode fixed detector boxes with matched SAM2 box and early-feature points."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.early_feature_prompt import early_feature_points
from modeling.seq23_viewport import crop_viewport,restore_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import parameter_fingerprints,assert_parameter_integrity
from scripts.run_prompt_feature_refinement import REPO,REVISION,predict_mask

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','candidate'),required=True)
 for k in ('checkpoint','checkpoint-verification','fixed-box-gate','fixed-box-reference','predictions','prediction-verification','mask-gate','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();ck=read(a.checkpoint_verification/'verification.json');old_gate=read(a.fixed_box_gate/'verification.json')
 if not ck['passed'] or ck['identity']['checkpoint_sha256']!=digest(a.checkpoint) or ck['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or not old_gate['passed'] or old_gate['identity']['checkpoint_sha256']!=digest(a.checkpoint) or old_gate['verifier_sha256']!=digest(ROOT/'scripts/verify_fixed_box_prompt_gates.py'):raise ValueError('Verified model or earlier box-only gate changed')
 rev=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True)
 if rev!=REVISION or dirty:raise ValueError('SAM2 reference source changed')
 if a.mode=='gate':
  old=read(a.fixed_box_reference/'completion.json')
  if not old['passed'] or old['mode']!='gate' or old['identity']!=old_gate['identity'] or len(old['events'])<1:raise ValueError('Earlier image gate changed')
  e=old['events'][0];cases=[{'sequence':'seq3','frame':e['frame'],'image_sha256':e['image_sha256'],'viewport':False,'original_box_viewport':e['box'],'candidate_box_viewport':e['box'],'masks':e['masks']}]
 else:
  pred=read(a.predictions/'completion.json');pv=read(a.prediction_verification/'verification.json');mg=read(a.mask_gate/'verification.json')
  if not pred['passed'] or pred['mode']!='candidate' or not pv['passed'] or pv['predictions_sha256']!=digest(a.predictions/'completion.json') or pv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_original_finetune_predictions.py') or not mg['passed'] or mg['runner_sha256']!=digest(Path(__file__)) or mg['checkpoint_sha256']!=digest(a.checkpoint) or mg['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_original_finetune_masks_gate.py'):raise ValueError('Predictions or mask engineering gate changed')
  cases=pred['events']
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(20261114);sys.path.insert(0,str(REPO))
 import sam2
 if not Path(sam2.__file__).resolve().is_relative_to(REPO.resolve()):raise ValueError('Wrong SAM2 source')
 from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(a.checkpoint),device='cpu',mode='eval',apply_postprocessing=False).eval().requires_grad_(False);before=parameter_fingerprints(model);predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0);events=[]
 with torch.inference_mode():
  for e in cases:
   seq=e['sequence'];i=e['frame'];path=videos[seq][i].image_path
   if digest(path)!=e['image_sha256']:raise ValueError('Image changed')
   with Image.open(path) as im:image=np.array(im.convert('RGB'),copy=True)
   if e['viewport']:image=crop_viewport(image)
   predictor.set_image(image);feature=predictor._features['high_res_feats'][0]
   if feature.shape!=(1,32,256,256) or not torch.isfinite(feature).all():raise ValueError('Unexpected early image feature')
   feature_sha=hashlib.sha256(feature.detach().cpu().contiguous().numpy().tobytes()).hexdigest();masks={};points={}
   for name in ('original','candidate'):
    box=e[name+'_box_viewport'];provisional=predict_mask(predictor,image.shape[:2],box)
    if box is None:point_value=None;status='no_box';refined=provisional
    else:
     proposal,status=early_feature_points({'backbone_fpn':[feature]},provisional,box,seed=20261114)
     point_value=None if proposal is None else proposal['feature_points']
     refined=provisional if point_value is None else predict_mask(predictor,image.shape[:2],box,point_value)
    points[name]={'status':status,'coordinates':point_value}
    for suffix,mask in (('box',provisional),('points',refined)):
     full=restore_mask(mask) if e['viewport'] else mask
     path=out/f'{seq}_{i:06d}_{name}_{suffix}.png';Image.fromarray(full.astype(np.uint8)*255).save(path);masks[name+'_'+suffix]=digest(path)
   if a.mode=='gate' and masks['original_box']!=e['masks']['fixed_initial']:raise ValueError('Box-only image predictor differs from earlier verified gate')
   events.append({'sequence':seq,'frame':i,'image_sha256':e['image_sha256'],'viewport':e['viewport'],'original_box_viewport':e['original_box_viewport'],'candidate_box_viewport':e['candidate_box_viewport'],'feature_sha256':feature_sha,'points':points,'masks':masks})
 assert_parameter_integrity(model,before,out,'after')
 result={'passed':True,'mode':a.mode,'events':events,'sam2_checkpoint_sha256':digest(a.checkpoint),'sam2_revision':rev,'fixed_box_gate_sha256':digest(a.fixed_box_gate/'verification.json'),'predictions_sha256':digest(a.predictions/'completion.json') if a.mode=='candidate' else None,'protocol_sha256':digest(ROOT/'protocols/polypgen_original_detector_finetune_v1.json'),'feature_helper_sha256':digest(ROOT/'modeling/early_feature_prompt.py'),'selector_sha256':digest(ROOT/'modeling/prompt_feature_refinement.py'),'runner_sha256':digest(Path(__file__)),'scope':'SAM2 image-level box and point prompts on fixed initial frames; no video propagation, detector search or target labels.'}
 (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'mode':a.mode,'frames':len(events)}))
if __name__=='__main__':main()
