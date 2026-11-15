#!/usr/bin/env python3
"""Indexed Lean engine for causal first-read conditioning-memory experiments.

The recorded identity pins the original indexed baseline; engine_variant_sha256
separately identifies this runner, whose only change is the causal prefix check.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest,stage_frames
from modeling.early_feature_prompt import early_feature_points
from modeling.lean_multiscale_anchor import install_multiscale_anchor
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import source_identity,build_author_predictor,parameter_fingerprints,assert_parameter_integrity

PROTOCOL=ROOT/'protocols/lean_indexed_spatial_rope_v1.json'
MANIFEST=ROOT/'data/polypgen_sequence.jsonl'

def read(path): return json.loads(path.read_text())

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('checkpoint','merged','reference','cpu-verification','observed-reference','output'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--sequence',required=True); p.add_argument('--mode',choices=('gate','candidate'),required=True)
 p.add_argument('--gate-a',type=Path); p.add_argument('--gate-b',type=Path)
 a=p.parse_args(); proto=read(PROTOCOL)
 if a.sequence not in proto['cohort'] or (a.mode=='gate' and a.sequence!='seq3'): raise ValueError('Out-of-scope sequence')
 verification=read(a.cpu_verification); observation=read(a.observed_reference/'position_observation.json')
 if (not verification['passed'] or not observation['passed'] or observation['sequence']!='seq3'
     or observation['source_sha256'] != digest(ROOT/'scripts/observe_lean_rope_positions.py')
     or observation['lookup_sha256'] != digest(ROOT/'modeling/lean_memory_position_lookup.py')
     or observation['reference_records_sha256'] != observation['observed_records_sha256']): raise ValueError('Position observer is invalid')
 merge=read(a.merged/'completion.json')
 if not merge['passed'] or digest(a.merged/'results.json') != verification['identity']['merged_results_sha256']: raise ValueError('Reference merge changed')
 ref=read(a.reference/'records.json'); ref_done=read(a.reference/'completion.json'); ref_setup=read(a.reference/'setup.json')
 if (not ref_done['passed'] or digest(a.reference/'records.json') != ref_done['records_sha256']
     or ref_setup['identity'] != verification['identity'] or ref_setup['sequence'] != a.sequence
     or ref_setup['mode'] != 'fixed' or digest(a.reference/'frames.json') != ref['frames_sha256']): raise ValueError('Corrected reference invalid')
 source=Path(read(a.merged/'locations.json')[a.sequence]); folder=source/a.sequence
 old_done=read(source/'completion.json'); old=read(source/'records.json')[0]; old_rows=read(folder/'frames.json')
 if digest(source/'records.json') != old_done['records_sha256'] or old_done['records_sha256'] != ref['original_records_sha256'] or digest(folder/'frames.json') != old['frames_sha256'] or digest(folder/'prompt.json') != old['prompt_sha256']:
  raise ValueError('Original prompt or reference changed')
 sig=verification['identity']['source_signature']
 pinned={'early_module_sha256':ROOT/'modeling/early_feature_prompt.py','prompt_module_sha256':ROOT/'modeling/prompt_feature_refinement.py',
  'memory_module_sha256':ROOT/'modeling/lean_multiscale_anchor.py','lean_runner_sha256':ROOT/'scripts/run_author_baseline.py',
  'runner_sha256':ROOT/'scripts/run_lean_auto_multiscale.py','manifest_sha256':MANIFEST,
  'bypass_sha256':ROOT/'external/Lean-SAM2/checkpoints/bypass_bottleneck_base.pth','lean_config_sha256':ROOT/'external/Lean-SAM2/configs/lean_sam2/lean_sam2.yaml'}
 if (digest(a.checkpoint) != sig['checkpoint_sha256'] or any(digest(path)!=sig[key] for key,path in pinned.items())
     or torch.__version__ != sig['torch'] or np.__version__ != sig['numpy'] or sys.version != sig['python']): raise ValueError('Model/runtime differs')
 identity={'runner_sha256':digest(ROOT/'scripts/run_lean_indexed_spatial_rope.py'),'protocol_sha256':digest(PROTOCOL),'adapter_sha256':digest(ROOT/'modeling/lean_indexed_spatial_rope.py'),
  'lookup_sha256':digest(ROOT/'modeling/lean_memory_position_lookup.py'),'cpu_verification_sha256':digest(a.cpu_verification),
  'observation_sha256':digest(a.observed_reference/'position_observation.json'),'cpu_identity':verification['identity']}
 if a.mode=='candidate':
  if not a.gate_a or not a.gate_b: raise ValueError('Two candidate engineering runs required')
  gates=[]
  for gate in (a.gate_a,a.gate_b):
   s,d=read(gate/'setup.json'),read(gate/'completion.json')
   if s['mode']!='gate' or s['identity']!=identity or not d['passed'] or digest(gate/'records.json')!=d['records_sha256']:
    raise ValueError('Candidate gate invalid')
   gates.append(read(gate/'records.json'))
  if gates[0]!=gates[1] or gates[0]['changed_masks']<=0: raise ValueError('Candidate gates differ or have no effect')
 prompt=read(folder/'prompt.json'); index=prompt['first_detection_index']
 videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')}; frames=videos[a.sequence]
 if index is None or len(frames)!=len(old_rows): raise ValueError('Missing prompt or frames')
 for frame,row in zip(frames,old_rows):
  if digest(frame.image_path)!=row['image_sha256']: raise ValueError('Input frame changed')
 out=validate_output(a.output); out.mkdir(parents=True,exist_ok=False)
 (out/'setup.json').write_text(json.dumps({'mode':a.mode,'sequence':a.sequence,'identity':identity,'engine_variant_sha256':digest(Path(__file__)),'engine_note':'identity pins original indexed baseline; this engine changes the causal prefix check','reference_records_sha256':ref_done['records_sha256']},indent=2)+'\n')
 torch.set_num_threads(2); repo,revision=source_identity('lean')
 if revision!=sig['revision']: raise ValueError('Lean revision changed')
 model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),repo)
 if model.mem_keep_ratio!=.25: raise ValueError('Memory budget changed')
 before=parameter_fingerprints(model); selections=[]; events=install_multiscale_anchor(model,.5,selection_events=selections); positions=[]
 for i in range(index):
  with Image.open(frames[i].image_path) as im: w,h=im.size
  Image.fromarray(np.zeros((h,w),np.uint8)).save(out/f'{i:06d}.png')
 with tempfile.TemporaryDirectory(prefix='staged_',dir=out) as tmp:
  stage_frames(frames,Path(tmp)); staged=sorted(Path(tmp).glob('*.jpg'),key=lambda q:int(q.stem))
  if len(staged)!=len(frames) or any(p.resolve()!=f.image_path.resolve() for p,f in zip(staged,frames)): raise ValueError('Frame order changed')
  state=model.init_state(video_path=tmp,offload_video_to_cpu=True,offload_state_to_cpu=False)
  try:
   with torch.inference_mode(), indexed_spatial_rope(model,positions):
    box=np.asarray(prompt['box'],np.float32); _,ids,logits=model.add_new_points_or_box(state,frame_idx=index,obj_id=1,box=box)
    if ids!=[1]: raise ValueError('Object changed')
    provisional=logits[0,0].cpu().numpy()>0
    if digest(folder/'prompt_provisional.png')!=prompt['provisional_mask_sha256'] or not np.array_equal(provisional,np.asarray(Image.open(folder/'prompt_provisional.png'))>0): raise ValueError('Box prompt changed')
    _,backbone,_,_,_=model._get_image_feature(state,index,batch_size=1); points,status=early_feature_points(backbone,provisional,box,seed=20261114)
    if status!=prompt['point_status'] or (None if points is None else points['feature_points'])!=prompt['points']: raise ValueError('Feature points changed')
    if points is not None:
     _,ids,logits=model.add_new_points_or_box(state,frame_idx=index,obj_id=1,points=np.asarray(points['feature_points'],np.float32),labels=np.array([1,0],np.int32),clear_old_points=False)
    initial=logits[0,0].cpu().numpy()>0
    if digest(folder/'prompt_refined.png')!=prompt['refined_mask_sha256'] or not np.array_equal(initial,np.asarray(Image.open(folder/'prompt_refined.png'))>0): raise ValueError('Refined prompt changed')
    if initial.any(): model.add_new_mask(state,frame_idx=index,obj_id=1,mask=initial)
    gen=model.propagate_in_video(state,start_frame_idx=index,reverse=False)
    for expected in range(index,len(frames)):
     actual,ids,output=next(gen)
     if actual!=expected or ids!=[1]: raise ValueError('Prediction order changed')
     Image.fromarray((output[0,0].cpu().numpy()>0).astype(np.uint8)*255).save(out/f'{actual:06d}.png')
    try: next(gen); raise ValueError('Unexpected extra frame')
    except StopIteration: pass
  finally: model.reset_state(state)
 assert_parameter_integrity(model,before,out,'after')
 masks=[digest(out/f'{i:06d}.png') for i in range(len(frames))]
 if a.sequence=='seq3' and masks[:2]!=ref['mask_sha256'][:2]: raise ValueError('Pre-intervention prefix changed')
 if not positions: raise ValueError('No rotary position events')
 (out/'positions.json').write_text(json.dumps(positions,indent=2)+'\n'); (out/'selections.json').write_text(json.dumps(selections,indent=2)+'\n'); (out/'memory_events.json').write_text(json.dumps(events,indent=2)+'\n')
 if a.mode=='candidate':
  scored=[]
  for i,frame in enumerate(frames):
   if digest(frame.mask_path)!=old_rows[i]['label_sha256']: raise ValueError('Label changed')
   truth=load_label_mask(frame.mask_path,[1])==1; pred=np.asarray(Image.open(out/f'{i:06d}.png'))>0
   if pred.shape!=truth.shape: raise ValueError('Shape changed')
   den=int(pred.sum()+truth.sum()); present=bool(truth.any())
   scored.append({'frame':i,'present':present,'dice':2*int((pred & truth).sum())/den if den else 1.,'false_alarm':bool(pred.any()) if not present else None,'mask_sha256':masks[i]})
  (out/'frames.json').write_text(json.dumps(scored,indent=2)+'\n')
 record={'sequence':a.sequence,'frames':len(frames),'mask_sha256':masks,'changed_masks':sum(x!=y for x,y in zip(masks,ref['mask_sha256'])),
  'positions_sha256':digest(out/'positions.json'),'selections_sha256':digest(out/'selections.json'),'memory_events_sha256':digest(out/'memory_events.json'),
  'reference_records_sha256':ref_done['records_sha256'],'frames_sha256':digest(out/'frames.json') if a.mode=='candidate' else None}
 (out/'records.json').write_text(json.dumps(record,indent=2)+'\n'); (out/'completion.json').write_text(json.dumps({'passed':True,'records_sha256':digest(out/'records.json')},indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k!='mask_sha256'},indent=2))
if __name__=='__main__':main()
