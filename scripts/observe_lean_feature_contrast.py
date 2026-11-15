#!/usr/bin/env python3
"""Capture label-free FPN contrast scores during an exact indexed Lean replay."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import MethodType
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.prompt_feature_refinement import reference_prototypes
from scripts.memory_token_io import digest
import scripts.run_lean_indexed_spatial_rope as runner
PROTOCOL=ROOT/'protocols/lean_early_feature_contrast_diagnostic_v1.json'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint','merged','reference','indexed-reference','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--sequence',required=True);p.add_argument('--label-free-gate',action='store_true');a=p.parse_args()
 if a.label_free_gate and a.sequence!='seq3':raise ValueError('Engineering gate uses seq3')
 if a.sequence not in read(PROTOCOL)['cohort']+['seq3']:raise ValueError('Sequence out of scope')
 merge=read(a.merged/'completion.json');loc=read(a.merged/'locations.json');source=Path(loc[a.sequence]);folder=source/a.sequence
 done=read(source/'completion.json');old=read(source/'records.json')[0]
 if not merge['passed'] or digest(source/'records.json')!=done['records_sha256'] or digest(folder/'prompt.json')!=old['prompt_sha256']:
  raise ValueError('Source changed')
 prompt=read(folder/'prompt.json');first=prompt['first_detection_index'];frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}[a.sequence]
 if digest(folder/'prompt_refined.png')!=prompt['refined_mask_sha256']:raise ValueError('Automatic prompt changed')
 mask=np.asarray(Image.open(folder/'prompt_refined.png'))>0
 target=sorted(set(first+round((len(frames)-1-first)*i/4) for i in range(5)))
 reference=read(a.indexed_reference/'records.json');refdone=read(a.indexed_reference/'completion.json')
 if not refdone['passed'] or digest(a.indexed_reference/'records.json')!=refdone['records_sha256']:raise ValueError('Indexed reference invalid')
 build=runner.build_author_predictor;captured={};prototypes={};status={}
 def observed_build(args,repo):
  model=build(args,repo); original=model._get_image_feature
  def observed(self,inference_state,frame_idx,batch_size):
   out=original(inference_state,frame_idx,batch_size)
   if frame_idx not in target or frame_idx in captured:return out
   if batch_size!=1:raise ValueError('Unexpected batch')
   feature=out[1]['backbone_fpn'];base=a.output/'contrast';base.mkdir(exist_ok=True)
   if frame_idx==first:
    first_feature=feature[0].detach().float().cpu().contiguous()
    if hashlib.sha256(first_feature.numpy().tobytes()).hexdigest()!=prompt['early_feature_sha256']:raise ValueError('Reference FPN feature mismatch')
    for level,x in enumerate(feature):
     try:
      positive,negative=reference_prototypes(x[0].detach().float(),mask)
      prototypes[level]=(positive,negative);status[str(level)]='available'
     except ValueError:
      status[str(level)]='degenerate_automatic_prompt'
   if not status:raise ValueError('Target arrived before initial prompt')
   arrays={}
   with torch.inference_mode():
    for level,x in enumerate(feature):
     if level not in prototypes:continue
     positive,negative=prototypes[level];z=F.normalize(x[0].detach().float(),dim=0)
     fg=torch.einsum('chw,c->hw',z,positive);bg=torch.einsum('chw,c->hw',z,negative)
     arrays[f'level{level}_positive']=fg.cpu().contiguous().numpy();arrays[f'level{level}_contrast']=(fg-bg).cpu().contiguous().numpy()
   path=base/f'{frame_idx:06d}.npz';np.savez_compressed(path,**arrays)
   captured[frame_idx]={'frame':frame_idx,'map_sha256':digest(path),'arrays':{k:{'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in arrays.items()}}
   return out
  model._get_image_feature=MethodType(observed,model)
  return model
 runner.build_author_predictor=observed_build
 args=['run_lean_indexed_spatial_rope.py','--mode','gate' if a.label_free_gate else 'candidate','--sequence',a.sequence]
 for key in ('checkpoint','merged','reference','cpu_verification','observed_reference','output'):args.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
 args.extend(['--gate-a',str(a.indexed_gate_a),'--gate-b',str(a.indexed_gate_b)]);sys.argv=args
 runner.main()
 candidate=read(a.output/'records.json')
 keys=('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256')
 if not a.label_free_gate:keys=keys+('frames_sha256',)
 for key in keys:
  if candidate[key]!=reference[key]:raise ValueError(f'Observer changed reference: {key}')
 if sorted(captured)!=target:raise ValueError('Missing observed feature maps')
 result={'passed':True,'sequence':a.sequence,'label_free_gate':a.label_free_gate,'target_frames':target,'first_detection_index':first,'prototype_status':status,
  'reference_records_sha256':digest(a.indexed_reference/'records.json'),'replay_records_sha256':digest(a.output/'records.json'),
  'protocol_sha256':digest(PROTOCOL),'source_sha256':digest(Path(__file__)),'prototype_source_sha256':digest(ROOT/'modeling/prompt_feature_refinement.py'),
  'captures':[captured[x] for x in target]}
 (a.output/'contrast_observation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='captures'},indent=2))
if __name__=='__main__':main()
