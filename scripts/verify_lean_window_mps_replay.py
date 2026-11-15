#!/usr/bin/env python3
"""Real seq4 replay of the pilot's token-window pipeline, CPU or MPS.

Engineering evidence only. Uses the unchanged archived automatic box and point
prompt, source images, multiscale anchor, token-window policy and indexed RoPE.
The MPS mode additionally uses the scoped pointer-transfer fix. No labels are
read, and no learned refiner is installed. Observers synchronize captured tensors.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest,stage_frames
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.lean_multiscale_anchor import install_multiscale_anchor
from modeling.lean_token_window_storage import TokenWindowRuntime
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,source_identity

FORK=ROOT/'external/Lean-SAM2-token-window'


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('checkpoint','capture-reference','preflight','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--device',choices=('cpu','mps'),required=True);p.add_argument('--end-frame',type=int,default=20)
 a=p.parse_args()
 if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
 if a.device=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise ValueError('Disable MPS fallback explicitly')
 torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
 pre=json.loads((a.preflight/'manifest.json').read_text());done=json.loads((a.preflight/'completion.json').read_text())
 if not done['passed'] or done['manifest_sha256']!=digest(a.preflight/'manifest.json'):raise ValueError('Preflight changed')
 source=pre['sequences']['seq4'];prompt_dir=Path(source['source'])/'seq4';prompt_path=prompt_dir/'prompt.json';prompt=json.loads(prompt_path.read_text());start=prompt['first_detection_index']
 if start!=9 or a.end_frame<19 or prompt['points'] is None or digest(prompt_path)!=source['prompt_sha256']:raise ValueError('Unexpected archived prompt')
 ref=json.loads((a.capture_reference/'channel_feature_receipt.json').read_text());raw=ref['pipeline']['raw_identity']
 if not ref['passed'] or ref['sequence']!='seq4' or raw['checkpoint_sha256']!=digest(a.checkpoint):raise ValueError('Reference differs')
 if any(digest(FORK/k)!=v for k,v in raw['patch_sha256'].items()) or digest(ROOT/'modeling/token_memory_window.py')!=raw['policy_sha256'] or digest(ROOT/'modeling/lean_token_window_storage.py')!=raw['runtime_sha256']:
  raise ValueError('Fork patch or token-window policy changed')
 if subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'],text=True).strip()!=raw['revision']:raise ValueError('Fork revision changed')
 _,original_rev=source_identity('lean')
 if original_rev!=raw['revision']:raise ValueError('Author revision changed')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos['seq4'];meta=json.loads((prompt_dir/'frames.json').read_text())
 if len(frames)!=source['frame_count'] or a.end_frame>=len(frames) or any(digest(f.image_path)!=meta[i]['image_sha256'] for i,f in enumerate(frames)):
  raise ValueError('Input video differs')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device=a.device),FORK)
 import sam2
 if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Wrong SAM2 fork')
 if model.num_maskmem!=7 or model.mem_keep_ratio!=.25 or model.insurance_max_frames!=3 or not model.enable_insurance_bank:raise ValueError('Wrong memory config')
 model.ema_history_replace_old_tail_in_attn=False;model.ema_history_resident_prune=False
 runtime=TokenWindowRuntime(16384,3072);model.token_window_runtime=runtime
 anchor_events=[];selection=[];anchor_events=install_multiscale_anchor(model,.5,selection_events=selection)
 torch.manual_seed(20261021)
 if a.device=='mps':torch.mps.manual_seed(20261021)
 old_track=model._track_step;features=[]
 def capture(self,*args,**kwargs):
  frame=args[0] if args else kwargs['frame_idx'];conditional=args[1] if args else kwargs['is_init_cond_frame']
  result=old_track(*args,**kwargs)
  if frame in (10,19) and not conditional:
   low=result[1][3];early=result[2][0];deep=result[3]
   path=out/f'features_{frame:06d}.pt';torch.save({'frame':frame,'early':early.detach().float().cpu(),
      'deep':deep.detach().float().cpu(),'low':low.detach().float().cpu()},path)
   features.append({'frame':frame,'sha256':digest(path)})
  return result
 model._track_step=MethodType(capture,model)
 positions=[];rows=[]
 with tempfile.TemporaryDirectory(prefix='lean_window_replay_') as temp:
  stage_frames(frames,Path(temp));state=model.init_state(video_path=temp,offload_video_to_cpu=True,offload_state_to_cpu=False)
  with ExitStack() as stack:
   stack.enter_context(torch.inference_mode())
   stack.enter_context(indexed_spatial_rope(model,positions) if a.device=='cpu' else indexed_spatial_rope_mps(model,positions))
   if a.device=='mps':stack.enter_context(blocking_mps_pointer_position(model))
   _,ids,logits=model.add_new_points_or_box(state,frame_idx=start,obj_id=1,box=np.asarray(prompt['box'],np.float32))
   if ids!=[1]:raise ValueError('Unexpected object id')
   provisional=logits[0,0].cpu().numpy()>0
   if a.device=='cpu' and not np.array_equal(provisional,np.asarray(Image.open(prompt_dir/'prompt_provisional.png'))>0):raise ValueError('CPU provisional prompt differs')
   _,ids,logits=model.add_new_points_or_box(state,frame_idx=start,obj_id=1,points=np.asarray(prompt['points'],np.float32),labels=np.array([1,0],np.int32),clear_old_points=False)
   initial=logits[0,0].cpu().numpy()>0
   if a.device=='cpu' and not np.array_equal(initial,np.asarray(Image.open(prompt_dir/'prompt_refined.png'))>0):raise ValueError('CPU refined prompt differs')
   if not initial.any():raise ValueError('Empty automatic prompt')
   model.add_new_mask(state,frame_idx=start,obj_id=1,mask=initial)
   for i,ids,logits in model.propagate_in_video(state,start_frame_idx=start,max_frame_num_to_track=a.end_frame-start,reverse=False):
    if ids!=[1] or i!=start+len(rows):raise ValueError('Frame order differs')
    path=out/f'{i:06d}.png';Image.fromarray((logits[0,0].cpu().numpy()>0).astype(np.uint8)*255).save(path)
    rows.append({'frame':i,'mask_sha256':digest(path)})
  model.reset_state(state)
 if len(rows)!=a.end_frame-start+1 or [x['frame'] for x in features]!=[10,19]:raise ValueError('Capture coverage differs')
 if not any(x['stage']=='admission_plan' and x['evict_frames'] for x in runtime.events):raise ValueError('Token-window eviction not exercised')
 result={'passed':True,'scope':'No-label automatic prompt engineering replay; no learned-refiner result',
   'device':a.device,'sequence':'seq4','prompt_frame':start,'end_frame':a.end_frame,
   'runner_sha256':digest(Path(__file__)),'capture_receipt_sha256':digest(a.capture_reference/'channel_feature_receipt.json'),
   'checkpoint_sha256':digest(a.checkpoint),'prompt_sha256':digest(prompt_path),'fork_patches':raw['patch_sha256'],
   'cpu_rope_sha256':digest(ROOT/'modeling/lean_indexed_spatial_rope.py'),
   'mps_rope_sha256':digest(ROOT/'modeling/lean_indexed_spatial_rope_mps.py'),
   'mps_blocking_sha256':digest(ROOT/'modeling/lean_mps_blocking_pointer_position.py'),
   'torch':torch.__version__,'fallback':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
   'features':features,'rows':rows,'positions':positions,'selection':selection,'anchor_events':anchor_events,'token_window_events':runtime.events}
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'device':a.device,'frames':len(rows),'features':len(features),'evictions':sum(bool(x['evict_frames']) for x in runtime.events if x['stage']=='admission_plan')}))
if __name__=='__main__':main()
