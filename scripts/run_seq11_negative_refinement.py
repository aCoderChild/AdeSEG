#!/usr/bin/env python3
"""Run native SAM2 with one fixed negative point derived from saved scores."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import parameter_fingerprints,assert_parameter_integrity
from scripts.run_prompt_feature_refinement import predict_mask

REPO=ROOT/'external/sam2_diagnostic_reference';REVISION='2b90b9f5ceec907a1c18123530e92e794ad901a4'
PROTOCOL=ROOT/'protocols/seq11_temporal_negative_v1.json'
ARMS=('random_negative','foreground_negative','contrast_negative','rgb_contrast_negative')
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','pilot'),required=True)
 for k in ('checkpoint','prior','reference','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--gate-a',type=Path);p.add_argument('--gate-b',type=Path);a=p.parse_args()
 prior=read(a.prior/'predictions.json');rd=read(a.reference/'completion.json');rec=read(a.reference/'records.json')
 if not rd['passed'] or digest(a.reference/'records.json')!=rd['records_sha256'] or prior['identity']['reference_sha256']!=rd['records_sha256'] or prior['identity']['checkpoint_sha256']!=digest(a.checkpoint):raise ValueError('Reference identity changed')
 if prior['identity']['runner_sha256']!=digest(ROOT/'scripts/run_seq11_temporal_refinement.py') or not read(a.prior/'parameter_integrity_after.json')['unchanged']:raise ValueError('Prior pilot source or parameters changed')
 rev=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
 if rev!=REVISION or subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True):raise ValueError('Native model source changed')
 identity={'runner_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),'prior_sha256':digest(a.prior/'predictions.json'),
           'reference_sha256':rd['records_sha256'],'checkpoint_sha256':digest(a.checkpoint),'revision':rev,'torch':torch.__version__,'python':sys.version}
 gates=[]
 if a.mode=='pilot':
  if not a.gate_a or not a.gate_b:raise ValueError('Two gates required')
  for path in (a.gate_a,a.gate_b):
   gate=read(path/'gate.json')
   if gate['identity']!=identity or not read(path/'parameter_integrity_after.json')['unchanged']:raise ValueError('Gate identity invalid')
   for row in gate['records']:
    for arm,h in row['masks'].items():
     if digest(path/f"{row['target']:06d}_{arm}.png")!=h:raise ValueError('Gate masks changed')
   gates.append(gate)
  if gates[0]!=gates[1]:raise ValueError('Gates differ')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(REPO));from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 torch.set_num_threads(2);torch.manual_seed(20261114)
 model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(a.checkpoint),device='cpu',mode='eval',apply_postprocessing=False).eval().requires_grad_(False)
 before=parameter_fingerprints(model);predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
 frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq11'];outputs=[]
 for old in prior['records']:
  i=old['target']
  if a.mode=='gate' and i not in (51,172):continue
  mask_path=a.reference/f'{i:06d}.png';old_box=a.prior/f'{i:06d}_box_only.png'
  if digest(frames[i].image_path)!=old['image_sha256'] or digest(mask_path)!=old['input_mask_sha256'] or digest(mask_path)!=rec['mask_sha256'][i] or digest(old_box)!=old['masks']['box_only']:raise ValueError('Frame, prediction, or prior box output changed')
  mask=np.asarray(Image.open(mask_path))>0;image=np.asarray(Image.open(frames[i].image_path).convert('RGB')).copy();predictor.set_image(image)
  box=old['box'];box_prediction=predict_mask(predictor,mask.shape,box)
  box_path=out/f'{i:06d}_box_only.png';Image.fromarray(box_prediction.astype(np.uint8)*255).save(box_path)
  if digest(box_path)!=old['masks']['box_only']:raise ValueError('Box-only control did not replay')
  keys=('random_points','foreground_points','contrast_points','rgb_contrast_points');points={arm:old['points'][key][1] for arm,key in zip(ARMS,keys)};masks={'box_only':digest(box_path)}
  for arm,point in points.items():
   if not mask[int(point[1]),int(point[0])]:raise ValueError('Negative point outside prediction')
   result,_,_=predictor.predict(box=np.asarray(box,np.float32),point_coords=np.asarray([point],np.float32),point_labels=np.array([0],np.int32),multimask_output=False)
   pred=np.asarray(result[0],bool)
   if pred.shape!=mask.shape:raise ValueError('Output geometry changed')
   path=out/f'{i:06d}_{arm}.png';Image.fromarray(pred.astype(np.uint8)*255).save(path);masks[arm]=digest(path)
  outputs.append({'source':old['source'],'target':i,'image_sha256':old['image_sha256'],'input_mask_sha256':old['input_mask_sha256'],'box':box,'points':points,'masks':masks})
 if a.mode=='gate' and [x['target'] for x in outputs]!=[51,172]:raise ValueError('Gate coverage differs')
 if a.mode=='pilot':
  for row in gates[0]['records']:
   if next(x for x in outputs if x['target']==row['target'])!=row:raise ValueError('Pilot differs from gate')
  if [x['target'] for x in outputs]!=[x['target'] for x in prior['records']]:raise ValueError('Pilot coverage differs')
 assert_parameter_integrity(model,before,out,'after');result={'identity':identity,'records':outputs}
 name='gate.json' if a.mode=='gate' else 'predictions.json';(out/name).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'mode':a.mode,'frames':len(outputs),'receipt_sha256':digest(out/name)}))
if __name__=='__main__':main()
