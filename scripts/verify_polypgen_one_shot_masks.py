#!/usr/bin/env python3
"""Audit repeated box-only and early-feature point masks without labels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('first','second','predictions','prediction-verification','mask-gate','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();x=read(a.first/'completion.json');y=read(a.second/'completion.json');pred=read(a.predictions/'completion.json');pv=read(a.prediction_verification/'verification.json');g=read(a.mask_gate/'verification.json')
if not all(t['passed'] for t in (x,y,pred,pv,g)) or x!=y or x['mode']!='candidate' or x['predictions_sha256']!=digest(a.predictions/'completion.json') or pv['predictions_sha256']!=x['predictions_sha256'] or pv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_one_shot_predictions.py') or x['sam2_checkpoint_sha256']!=digest(a.checkpoint) or g['checkpoint_sha256']!=x['sam2_checkpoint_sha256'] or x['runner_sha256']!=digest(ROOT/'scripts/predict_polypgen_one_shot_masks.py') or g['runner_sha256']!=x['runner_sha256'] or g['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_one_shot_masks_gate.py') or x['feature_helper_sha256']!=digest(ROOT/'modeling/early_feature_prompt.py') or x['selector_sha256']!=digest(ROOT/'modeling/prompt_feature_refinement.py') or x['protocol_sha256']!=digest(ROOT/'protocols/polypgen_one_shot_detector_pilot_v1.json'):raise ValueError('Mask provenance changed')
if len(x['events'])!=4 or len(pred['events'])!=4:raise ValueError('Scope differs')
for e,prediction in zip(x['events'],pred['events'],strict=True):
 seq=e['sequence'];i=e['frame']
 if (seq,i,e['image_sha256'],e['viewport'])!=(prediction['sequence'],prediction['frame'],prediction['image_sha256'],prediction['viewport']) or e['original_box_viewport']!=prediction['original_box_viewport'] or e['candidate_box_viewport']!=prediction['candidate_box_viewport']:raise ValueError('Prompt input differs')
 if set(e['masks'])!={'original_box','original_points','candidate_box','candidate_points'}:raise ValueError('Arm scope changed')
 for name in ('original','candidate'):
  box=e[name+'_box_viewport'];point=e['points'][name]
  if box is None:
   if point['status']!='no_box' or point['coordinates'] is not None:raise ValueError('Missing box generated points')
  elif point['status']=='available':
   points=point['coordinates']
   if len(points)!=2 or any(len(xy)!=2 or not(box[0]<=xy[0]<=box[2] and box[1]<=xy[1]<=box[3]) for xy in points):raise ValueError('Feature point outside prompt box')
  elif point['status'] not in ('empty_provisional_mask','degenerate_prototypes','insufficient_box_cells') or point['coordinates'] is not None:raise ValueError('Invalid feature-point status')
  for suffix in ('box','points'):
   method=name+'_'+suffix;sha=e['masks'][method]
   for folder in (a.first,a.second):
    path=folder/f'{seq}_{i:06d}_{method}.png'
    if digest(path)!=sha:raise ValueError('Mask changed')
    if box is None and np.asarray(Image.open(path)).any():raise ValueError('Missing box produced mask')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'frames':4,'mask_receipt_sha256':digest(a.first/'completion.json'),'repeat_sha256':digest(a.second/'completion.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Repeated candidate masks and point geometry verified.')
