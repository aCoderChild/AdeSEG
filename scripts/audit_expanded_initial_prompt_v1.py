#!/usr/bin/env python3
"""Replay and freeze the MPS initial automatic mask; no labels are read.

Compares the raw MPS mask supplied to add_new_mask with the archived CPU
automatic prompt, and checks the resulting first propagated mask against the
two existing MPS trajectories. It does not assert patient independence or
prove that an uncaptured earlier in-memory tensor was identical.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest,stage_frames
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.lean_multiscale_anchor import install_multiscale_anchor
from modeling.lean_token_window_storage import TokenWindowRuntime
from modeling.prompt_feature_refinement import reference_prototypes
from scripts.memory_token_io import digest,validate_output
from scripts.preflight_expanded_train_capture_v1 import TARGETS
from scripts.run_author_baseline import build_author_predictor
from scripts.run_expanded_train_capture_v1 import FORK,state_hash


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('preflight','reference','capture','checkpoint','output'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--sequence',choices=tuple(TARGETS),required=True);a=p.parse_args()
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0':
        raise RuntimeError('MPS with fallback and fast math disabled required')
    pre=read(a.preflight/'manifest.json');done=read(a.preflight/'completion.json')
    if (not done['passed'] or done['manifest_sha256']!=digest(a.preflight/'manifest.json')
            or pre['runner_sha256']!=digest(ROOT/'scripts/preflight_expanded_train_capture_v1.py')
            or pre['protocol_sha256']!=digest(ROOT/'protocols/EXPANDED_TRAIN_CAPTURE_SEQ6_SEQ13_V1.md')
            or pre['data_manifest_sha256']!=digest(ROOT/'data/polypgen_sequence.jsonl')):
        raise ValueError('Frozen plan changed')
    ref=read(a.reference/'receipt.json');cap=read(a.capture/'receipt.json');spec=pre['sequences'][a.sequence]
    if (ref['sequence']!=a.sequence or cap['sequence']!=a.sequence or ref['mode']!='reference' or cap['mode']!='capture'
            or not ref['passed'] or not cap['passed'] or ref['identity']!=cap['identity']
            or ref['identity']['preflight_sha256']!=digest(a.preflight/'manifest.json')
            or ref['identity']['runner_sha256']!=digest(ROOT/'scripts/run_expanded_train_capture_v1.py')
            or ref['identity']['mps_indexed_sha256']!=digest(ROOT/'modeling/lean_indexed_spatial_rope_mps.py')
            or ref['identity']['mps_blocking_sha256']!=digest(ROOT/'modeling/lean_mps_blocking_pointer_position.py')
            or ref['identity']['anchor_sha256']!=digest(ROOT/'modeling/lean_multiscale_anchor.py')
            or ref['identity']['token_runtime_sha256']!=digest(ROOT/'modeling/lean_token_window_storage.py')
            or ref['identity']['token_policy_sha256']!=digest(ROOT/'modeling/token_memory_window.py')
            or ref['identity']['checkpoint_sha256']!=digest(a.checkpoint)
            or ref['identity']['torch']!=str(torch.__version__) or ref['identity']['device']!='mps'
            or ref['identity']['fallback']!='0' or ref['identity']['fast_math']!='0'
            or ref['identity']['seed']!=20261021 or not ref['identity']['deterministic_algorithms']):
        raise ValueError('Frozen runtime identity differs')
    if any(digest(FORK/k)!=v for k,v in ref['identity']['fork_patches'].items()):raise ValueError('Fork patches changed')
    if subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'],text=True).strip()!=ref['identity']['fork_revision']:
        raise ValueError('Fork revision changed')
    source=Path(spec['source']);folder=source/a.sequence;prompt_file=folder/'prompt.json';prompt=read(prompt_file)
    if digest(prompt_file)!=spec['prompt_sha256'] or prompt['first_detection_index']!=spec['prompt_frame']:
        raise ValueError('Prompt source changed')
    archived=folder/'prompt_refined.png'
    if digest(archived)!=prompt['refined_mask_sha256']:raise ValueError('Archived CPU prompt mask changed')
    frames=load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev')[a.sequence]
    if len(frames)!=spec['frames'] or any(digest(frame.image_path)!=h for frame,h in zip(frames,spec['image_sha256'],strict=True)):
        raise ValueError('Source image changed')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(20261021);torch.mps.manual_seed(20261021);np.random.seed(20261021)
    sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='mps'),FORK)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Wrong predictor worktree')
    if state_hash(model)!=ref['identity']['model_state_sha256']:raise ValueError('Live predictor weights or buffers differ')
    model.ema_history_replace_old_tail_in_attn=False;model.ema_history_resident_prune=False
    model.token_window_runtime=TokenWindowRuntime(16384,3072);selection=[];install_multiscale_anchor(model,.5,selection_events=selection)
    positions=[];start=spec['prompt_frame']
    with tempfile.TemporaryDirectory(prefix='staged_',dir=out) as temp:
        stage_frames(frames,Path(temp));state=model.init_state(video_path=temp,offload_video_to_cpu=True,offload_state_to_cpu=False)
        try:
            with ExitStack() as stack:
                stack.enter_context(torch.inference_mode());stack.enter_context(indexed_spatial_rope_mps(model,positions));stack.enter_context(blocking_mps_pointer_position(model))
                _,ids,_=model.add_new_points_or_box(state,frame_idx=start,obj_id=1,box=np.asarray(prompt['box'],np.float32))
                if ids!=[1]:raise ValueError('Object ID differs')
                _,ids,logits=model.add_new_points_or_box(state,frame_idx=start,obj_id=1,points=np.asarray(prompt['points'],np.float32),labels=np.array([1,0],np.int32),clear_old_points=False)
                if ids!=[1]:raise ValueError('Object ID differs')
                initial=(logits[0,0].cpu().numpy()>0)
                if not initial.any():raise ValueError('Empty MPS prompt')
                _,backbone,_,_,_=model._get_image_feature(state,start,batch_size=1)
                reference_prototypes(backbone['backbone_fpn'][0][0].detach().float().cpu(),initial)
                model.add_new_mask(state,frame_idx=start,obj_id=1,mask=initial)
                frame,ids,output=next(model.propagate_in_video(state,start_frame_idx=start,max_frame_num_to_track=0,reverse=False))
                if frame!=start or ids!=[1]:raise ValueError('Prompt output frame differs')
                conditioned=(output[0,0].cpu().numpy()>0)
        finally:model.reset_state(state)
    if state_hash(model)!=ref['identity']['model_state_sha256']:raise ValueError('Live model state changed')
    original=np.asarray(Image.open(archived).convert('L'))>0
    if original.shape!=initial.shape or conditioned.shape!=initial.shape:raise ValueError('Prompt mask geometry differs')
    for folder,receipt in ((a.reference,ref),(a.capture,cap)):
        saved=folder/f'{start:06d}.png';item=next(x for x in receipt['rows'] if x['frame']==start)
        if digest(saved)!=item['mask_sha256'] or not np.array_equal(conditioned,np.asarray(Image.open(saved).convert('L'))>0):
            raise ValueError('Replayed conditioning output differs from MPS trajectory')
    out_mask=out/'mps_initial_before_memory.png';Image.fromarray(initial.astype(np.uint8)*255).save(out_mask)
    union=int((initial|original).sum());intersection=int((initial&original).sum())
    result={'passed':True,'scope':'Label-free fresh-process prompt replay. Exact same historical in-memory prompt cannot be independently proven because it was not saved.',
            'sequence':a.sequence,'prompt_frame':start,'archived_cpu_initial_sha256':digest(archived),
            'mps_initial_before_memory_sha256':digest(out_mask),'initial_disagree_pixels':int((initial!=original).sum()),
            'initial_iou':intersection/union if union else 1.,'mps_initial_pixels':int(initial.sum()),'archived_cpu_initial_pixels':int(original.sum()),
            'conditioning_output_equals_reference':True,'conditioning_output_equals_capture':True,
            'mps_initial_vs_conditioned_disagree_pixels':int((initial!=conditioned).sum()),
            'reference_receipt_sha256':digest(a.reference/'receipt.json'),'capture_receipt_sha256':digest(a.capture/'receipt.json'),
            'runner_sha256':digest(Path(__file__))}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
