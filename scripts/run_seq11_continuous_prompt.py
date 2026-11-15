#!/usr/bin/env python3
"""Fixed-frame SAM2 dense-prompt pilot from archived temporal scores."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.continuous_mask_prompt import mask_and_score_prompts
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import parameter_fingerprints,assert_parameter_integrity

REPO=ROOT/'external/sam2_diagnostic_reference';REVISION='2b90b9f5ceec907a1c18123530e92e794ad901a4';PROTOCOL=ROOT/'protocols/seq11_continuous_mask_prompt_v1.json'
ARMS=('mask_only','foreground','contrast','rgb_contrast','shuffled_contrast')
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','pilot'),required=True)
 for k in ('checkpoint','capture','reference','prior','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--gate-a',type=Path);p.add_argument('--gate-b',type=Path);a=p.parse_args()
 cap=read(a.capture/'capture.json');prior=read(a.prior/'predictions.json');rd=read(a.reference/'completion.json');rr=read(a.reference/'records.json')
 if not cap['passed'] or not rd['passed'] or digest(a.reference/'records.json')!=rd['records_sha256'] or cap['reference_records_sha256']!=rd['records_sha256'] or prior['identity']['reference_sha256']!=rd['records_sha256'] or cap['checkpoint_sha256']!=digest(a.checkpoint) or prior['identity']['checkpoint_sha256']!=digest(a.checkpoint):raise ValueError('Inputs differ')
 if cap['source_sha256']!=digest(ROOT/'scripts/seq11_temporal_background.py') or prior['identity']['runner_sha256']!=digest(ROOT/'scripts/run_seq11_temporal_refinement.py'):raise ValueError('Input source changed')
 rev=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
 if rev!=REVISION or subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True):raise ValueError('SAM2 source changed')
 ident={'source_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),'prompt_sha256':digest(ROOT/'modeling/continuous_mask_prompt.py'),
        'capture_sha256':digest(a.capture/'capture.json'),'prior_sha256':digest(a.prior/'predictions.json'),'reference_sha256':rd['records_sha256'],
        'checkpoint_sha256':digest(a.checkpoint),'revision':rev,'torch':torch.__version__,'python':sys.version}
 gates=[]
 if a.mode=='pilot':
  for path in (a.gate_a,a.gate_b):
   if path is None:raise ValueError('Two gates required')
   data=read(path/'gate.json')
   if data['identity']!=ident or not read(path/'parameter_integrity_after.json')['unchanged']:raise ValueError('Gate differs')
   for item in data['records']:
    for arm,h in item['masks'].items():
     if digest(path/f"{item['target']:06d}_{arm}.png")!=h:raise ValueError('Gate mask changed')
   gates.append(data)
  if gates[0]!=gates[1]:raise ValueError('Gates differ')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(REPO));from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 torch.set_num_threads(2);torch.manual_seed(20261114)
 model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(a.checkpoint),device='cpu',mode='eval',apply_postprocessing=False).eval().requires_grad_(False)
 before=parameter_fingerprints(model);predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
 frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq11'];outputs=[]
 for item in cap['outputs']:
  src,i=item['source'],item['target']
  if a.mode=='gate' and i not in (51,172):continue
  old=next(x for x in prior['records'] if x['source']==src and x['target']==i);maskpath=a.reference/f'{i:06d}.png';scorepath=a.capture/f'{src:06d}_{i:06d}.npz'
  if item['status']!='available' or digest(scorepath)!=item['map_sha256'] or digest(maskpath)!=rr['mask_sha256'][i] or digest(maskpath)!=old['input_mask_sha256'] or digest(frames[i].image_path)!=item['target_image_sha256']:raise ValueError('Saved model input changed')
  mask=np.asarray(Image.open(maskpath))>0;seed=int.from_bytes(hashlib.sha256(f'20261114|seq11|{src}|{i}'.encode()).digest()[:8],'little')
  with np.load(scorepath,allow_pickle=False) as scores:
   fg=mask_and_score_prompts(mask,scores['foreground'],seed);contrast=mask_and_score_prompts(mask,scores['contrast'],seed);rgb=mask_and_score_prompts(mask,scores['rgb_contrast'],seed)
  if not np.array_equal(fg['mask_only'],contrast['mask_only']) or not np.array_equal(rgb['mask_only'],contrast['mask_only']):raise ValueError('Mask priors differ')
  prompts={'mask_only':contrast['mask_only'],'foreground':fg['score'],'contrast':contrast['score'],'rgb_contrast':rgb['score'],'shuffled_contrast':contrast['shuffled']}
  image=np.asarray(Image.open(frames[i].image_path).convert('RGB')).copy();predictor.set_image(image);masks={};prompt_hashes={}
  for arm,prompt in prompts.items():
   if prompt.shape!=(1,256,256) or not np.isfinite(prompt).all():raise ValueError('Invalid dense prompt')
   pred,_,_=predictor.predict(box=np.asarray(old['box'],np.float32),mask_input=prompt,multimask_output=False)
   mask_out=np.asarray(pred[0],bool)
   if mask_out.shape!=mask.shape:raise ValueError('Output shape changed')
   path=out/f'{i:06d}_{arm}.png';Image.fromarray(mask_out.astype(np.uint8)*255).save(path)
   masks[arm]=digest(path);prompt_hashes[arm]=hashlib.sha256(prompt.tobytes()).hexdigest()
  outputs.append({'source':src,'target':i,'image_sha256':item['target_image_sha256'],'input_mask_sha256':digest(maskpath),
                  'map_sha256':item['map_sha256'],'box':old['box'],'prompt_sha256':prompt_hashes,'masks':masks})
 if a.mode=='gate' and [x['target'] for x in outputs]!=[51,172]:raise ValueError('Gate coverage differs')
 if a.mode=='pilot':
  for row in gates[0]['records']:
   if next(x for x in outputs if x['target']==row['target'])!=row:raise ValueError('Pilot differs from gate')
  if [(x['source'],x['target']) for x in outputs]!=[(x['source'],x['target']) for x in prior['records']]:raise ValueError('Pilot coverage differs')
 assert_parameter_integrity(model,before,out,'after');data={'identity':ident,'records':outputs}
 file=out/('gate.json' if a.mode=='gate' else 'predictions.json');file.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'mode':a.mode,'frames':len(outputs),'receipt_sha256':digest(file)}))
if __name__=='__main__':main()
