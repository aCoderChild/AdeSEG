#!/usr/bin/env python3
"""MPS indexed-RoPE and pointer-transfer repeatability engineering trial.

Run twice in separate processes and compare receipt hashes and saved masks. This
tests an automatic-prompt trajectory with author pruning; no labels are read.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import MethodType,SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest,stage_frames
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_contrast_refinement_integration import route_lean_contrast_refinement
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,source_identity


def thash(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

class Identity(torch.nn.Module):
 _trained_checkpoint_loaded=True  # Test-only identity, never a learned checkpoint.
 def forward(self,early,deep,logits):return logits.clone()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('checkpoint','prompt-dir','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--sequence',default='seq4');p.add_argument('--end-frame',type=int,default=13)
 p.add_argument('--blocking',action='store_true')
 p.add_argument('--arm',choices=('baseline','A','B','C','D'),required=True)
 a=p.parse_args()
 if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
 torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
 prompt_path=a.prompt_dir/a.sequence/'prompt.json';prompt=json.loads(prompt_path.read_text());start=prompt['first_detection_index']
 if not isinstance(start,int) or start<=0 or a.end_frame<=start:raise ValueError('Need nonzero prompt and subsequent frames')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
 frames=videos[a.sequence][:a.end_frame+1];meta=json.loads((a.prompt_dir/a.sequence/'frames.json').read_text())
 if len(frames)!=a.end_frame+1 or any(digest(f.image_path)!=meta[i]['image_sha256'] for i,f in enumerate(frames)):raise ValueError('Source changed')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 repo,revision=source_identity('lean');model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='mps'),repo)
 torch.manual_seed(20261021);torch.mps.manual_seed(20261021)
 events=[];old=model._compress_memory_entry
 def compress(self,entry):
  pending=entry.get('pending_keep_idx');sha=None if pending is None else thash(pending)
  result=old(entry)
  if sha is not None:events.append({'frame':entry.get('frame_idx'),'keep_sha256':sha,'count':entry.get('smr_pruned_token_count')})
  return result
 model._compress_memory_entry=MethodType(compress,model)
 rows=[];indexed_events=[];states=[]
 with tempfile.TemporaryDirectory(prefix='lean_mps_repeat_') as temp:
  stage_frames(frames,Path(temp));state=model.init_state(video_path=temp,offload_video_to_cpu=True,offload_state_to_cpu=False)
  torch.mps.synchronize();t0=time.perf_counter()
  transfer_context=blocking_mps_pointer_position(model) if a.blocking else nullcontext()
  route=nullcontext() if a.arm=='baseline' else route_lean_contrast_refinement(model,Identity().eval(),a.arm,test_only=True)
  with torch.inference_mode(),indexed_spatial_rope_mps(model,indexed_events),transfer_context,route:
   _,ids,logits=model.add_new_points_or_box(state,frame_idx=start,obj_id=1,box=np.asarray(prompt['box'],np.float32))
   if prompt['points'] is not None:
    _,ids,logits=model.add_new_points_or_box(state,frame_idx=start,obj_id=1,points=np.asarray(prompt['points'],np.float32),labels=np.array([1,0],np.int32),clear_old_points=False)
   if ids!=[1]:raise RuntimeError('Object differs')
   mask=logits[0,0].cpu().numpy()>0
   if not mask.any():raise RuntimeError('Empty automatic mask')
   model.add_new_mask(state,frame_idx=start,obj_id=1,mask=mask)
   for i,ids,logits in model.propagate_in_video(state,start_frame_idx=start,reverse=False):
    if ids!=[1]:raise RuntimeError('Object differs')
    mask=(logits[0,0].cpu().numpy()>0).astype(np.uint8)*255;path=out/f'{i:06d}.png';Image.fromarray(mask).save(path)
    rows.append({'frame':i,'logits_sha256':thash(logits),'mask_sha256':digest(path)})
    entry=state['output_dict_per_obj'][0]['non_cond_frame_outputs'].get(i)
    if entry is not None:
     states.append({'frame':i,**{key:thash(entry[key]) for key in ('pred_masks','maskmem_features','obj_ptr','object_score_logits')}})
  torch.mps.synchronize();elapsed=time.perf_counter()-t0
  model.reset_state(state)
 if not events:raise RuntimeError('No memory compression occurred')
 receipt={'passed':True,'scope':'MPS deterministic-algorithms repeatability trial, no labels; CPU-reference geometry caveats remain',
  'sequence':a.sequence,'prompt_frame':start,'end_frame':a.end_frame,'checkpoint_sha256':digest(a.checkpoint),
  'prompt_sha256':digest(prompt_path),'author_revision':revision,'runner_sha256':digest(Path(__file__)),
  'torch':torch.__version__,
  'pointer_transfer_patch_sha256':digest(ROOT/'modeling/lean_mps_blocking_pointer_position.py'),
  'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
  'random_seed':20261021,
  'blocking_pointer_transfer':a.blocking,'elapsed_seconds_including_prompt_and_mask_saving':elapsed,
  'arm':a.arm,
  'mps_environment':{k:v for k,v in os.environ.items() if k in ('PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','PYTORCH_MPS_PREFER_METAL')},
  'indexed_adapter_sha256':digest(ROOT/'modeling/lean_indexed_spatial_rope_mps.py'),
  'rows':rows,'states_at_write':states,'compressions':events,'indexed_events':indexed_events}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS frames',len(rows),'compressions',len(events))
if __name__=='__main__':main()
