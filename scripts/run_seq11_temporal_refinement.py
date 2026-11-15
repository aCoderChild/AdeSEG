#!/usr/bin/env python3
"""Matched frozen-frame SAM2 point-prompt diagnostic; no propagation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.temporal_background_points import points_in_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_prompt_feature_refinement import predict_mask
from scripts.run_author_baseline import parameter_fingerprints,assert_parameter_integrity

PROTOCOL=ROOT/'protocols/seq11_temporal_refinement_v1.json'
PAIR_PROTOCOL=ROOT/'protocols/seq11_temporal_background_v1.json'
REPO=ROOT/'external/sam2_diagnostic_reference'
REVISION='2b90b9f5ceec907a1c18123530e92e794ad901a4'
ARMS=('box_only','random_points','foreground_points','contrast_points','rgb_contrast_points')
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','pilot'),required=True)
 for k in ('checkpoint','capture','reference','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--gate-a',type=Path);p.add_argument('--gate-b',type=Path);a=p.parse_args()
 receipt=read(a.capture/'capture.json');refdone=read(a.reference/'completion.json');ref=read(a.reference/'records.json')
 if not receipt['passed'] or receipt['protocol_sha256']!=digest(PAIR_PROTOCOL) or not refdone['passed'] or digest(a.reference/'records.json')!=refdone['records_sha256'] or receipt['reference_records_sha256']!=refdone['records_sha256'] or receipt['checkpoint_sha256']!=digest(a.checkpoint):raise ValueError('Source identity differs')
 if receipt['source_sha256']!=digest(ROOT/'scripts/seq11_temporal_background.py') or receipt['prototype_sha256']!=digest(ROOT/'modeling/prompt_feature_refinement.py'):raise ValueError('Capture source changed')
 rev=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
 dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True)
 if rev!=REVISION or dirty:raise ValueError('Native SAM2 reference changed')
 identity={'runner_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),'points_sha256':digest(ROOT/'modeling/temporal_background_points.py'),
           'capture_sha256':digest(a.capture/'capture.json'),'reference_sha256':refdone['records_sha256'],'checkpoint_sha256':digest(a.checkpoint),
           'revision':rev,'torch':torch.__version__,'python':sys.version}
 gates=[]
 if a.mode=='pilot':
  if not a.gate_a or not a.gate_b:raise ValueError('Two gates required')
  for gate in (a.gate_a,a.gate_b):
   g=read(gate/'gate.json')
   if g['identity']!=identity:raise ValueError('Gate identity changed')
   for row in g['records']:
    for arm,h in row['masks'].items():
     if digest(gate/f"{row['target']:06d}_{arm}.png")!=h:raise ValueError('Gate mask changed')
   gates.append(g)
  if gates[0]!=gates[1]:raise ValueError('Gates differ')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(REPO));from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 torch.set_num_threads(2);torch.manual_seed(20261114)
 model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(a.checkpoint),device='cpu',mode='eval',apply_postprocessing=False).eval().requires_grad_(False)
 before=parameter_fingerprints(model)
 predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos['seq11']
 outputs=[]
 for row in receipt['outputs']:
  src,dst=row['source'],row['target']
  if a.mode=='gate' and dst not in (51,172):continue
  if row['status']!='available':raise ValueError('Missing score map')
  map_path=a.capture/f'{src:06d}_{dst:06d}.npz';mask_path=a.reference/f'{dst:06d}.png'
  if digest(map_path)!=row['map_sha256'] or digest(mask_path)!=ref['mask_sha256'][dst] or digest(frames[dst].image_path)!=row['target_image_sha256']:raise ValueError('Input changed')
  pred=np.asarray(Image.open(mask_path))>0; yy,xx=np.nonzero(pred)
  if len(xx)==0:raise ValueError('Empty reference mask; pilot does not define a box')
  box=[float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)]
  seed=int.from_bytes(hashlib.sha256(f'20261114|seq11|{src}|{dst}'.encode()).digest()[:8],'little')
  with np.load(map_path,allow_pickle=False) as z:
   choices={k:points_in_mask(z[k],pred,seed) for k in ('foreground','contrast','rgb_contrast')}
  if any(v is None for v in choices.values()) or not choices['foreground']['random_points']==choices['contrast']['random_points']==choices['rgb_contrast']['random_points']:raise ValueError('Point geometry or random control changed')
  coords={'box_only':None,'random_points':choices['contrast']['random_points'],'foreground_points':choices['foreground']['feature_points'],
          'contrast_points':choices['contrast']['feature_points'],'rgb_contrast_points':choices['rgb_contrast']['feature_points']}
  for points in coords.values():
   if points is not None and any(not pred[int(y),int(x)] for x,y in points):raise ValueError('Point outside source mask')
  image=np.asarray(Image.open(frames[dst].image_path).convert('RGB')).copy();predictor.set_image(image)
  masks={}
  for arm in ARMS:
   result=predict_mask(predictor,pred.shape,box,coords[arm]);path=out/f'{dst:06d}_{arm}.png'
   Image.fromarray(result.astype(np.uint8)*255).save(path);masks[arm]=digest(path)
  outputs.append({'source':src,'target':dst,'image_sha256':row['target_image_sha256'],'input_mask_sha256':digest(mask_path),
                  'map_sha256':row['map_sha256'],'box':box,'points':coords,'masks':masks})
 if a.mode=='gate' and [x['target'] for x in outputs]!=[51,172]:raise ValueError('Gate coverage changed')
 if a.mode=='pilot':
  for row in gates[0]['records']:
   if next(x for x in outputs if x['target']==row['target'])!=row:raise ValueError('Pilot differs from gate')
  if [(x['source'],x['target']) for x in outputs]!=[tuple(x) for x in read(PAIR_PROTOCOL)['pairs']]:raise ValueError('Pilot coverage changed')
 assert_parameter_integrity(model,before,out,'after')
 result={'identity':identity,'records':outputs}
 filename='gate.json' if a.mode=='gate' else 'predictions.json'
 (out/filename).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'mode':a.mode,'frames':len(outputs),'receipt_sha256':digest(out/filename)}))
if __name__=='__main__':main()
