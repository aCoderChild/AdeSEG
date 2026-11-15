#!/usr/bin/env python3
"""Frozen-prompt MPS project Lean token-window baseline; no learned refiner.

Uses exactly the archived SAMURAI-comparison mask and source frame/crop, while
retaining the project's existing multiscale anchor and token-window policy.
Predictions are saved before any scoring; no label pixels or detector are read.
This is not the unmodified author Lean baseline or a throughput benchmark.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
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
from modeling.seq23_viewport import restore_mask,stage_cropped_frames
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,source_identity
from scripts.run_samurai_automatic_priority import prepare_archived_prompt,model_state_hash,MANIFEST

FORK=ROOT/'external/Lean-SAM2-token-window'
POLICY=ROOT/'modeling/token_memory_window.py'
RUNTIME=ROOT/'modeling/lean_token_window_storage.py'
ROPE=ROOT/'modeling/lean_indexed_spatial_rope_mps.py'
BLOCKING=ROOT/'modeling/lean_mps_blocking_pointer_position.py'
ANCHOR=ROOT/'modeling/lean_multiscale_anchor.py'


def read(path):return json.loads(path.read_text())


def runtime_identity(checkpoint,capture_reference,mps_reference):
    """Check pinned author, actual fork patches, model asset and adapters."""
    receipt=read(capture_reference/'channel_feature_receipt.json')
    if not receipt.get('passed') or receipt.get('sequence')!='seq4':raise ValueError('Capture reference differs')
    nested=read(capture_reference/'utility_capture_receipt.json')
    if not nested.get('passed') or nested.get('sequence')!='seq4' or nested.get('feature_receipt_sha256')!=digest(capture_reference/'channel_feature_receipt.json'):
        raise ValueError('Nested capture receipt differs')
    raw=receipt['pipeline']['raw_identity']
    sig=receipt['pipeline']['cpu_identity']['source_signature']
    repo,revision=source_identity('lean')
    if subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'],text=True).strip()!=raw['revision'] or revision!=raw['revision']:
        raise ValueError('Lean source revision differs')
    changed=subprocess.check_output(['git','-C',str(FORK),'diff','--name-only','HEAD'],text=True).splitlines()
    if sorted(changed)!=sorted(raw['patch_sha256']):raise ValueError('Unexpected fork changes')
    for name,sha in raw['patch_sha256'].items():
        if digest(FORK/name)!=sha:raise ValueError('Fork patch changed')
    if digest(checkpoint)!=raw['checkpoint_sha256'] or digest(POLICY)!=raw['policy_sha256'] or digest(RUNTIME)!=raw['runtime_sha256']:
        raise ValueError('Checkpoint or token window policy differs')
    fixed={'memory_module_sha256':ANCHOR,'bypass_sha256':FORK/'checkpoints/bypass_bottleneck_base.pth',
           'lean_config_sha256':FORK/'configs/lean_sam2/lean_sam2.yaml','lean_runner_sha256':ROOT/'scripts/run_author_baseline.py',
           'manifest_sha256':MANIFEST,'detector_sha256':ROOT/'checkpoints/polypgen_yolov8n.pt'}
    if any(digest(path)!=sig[key] for key,path in fixed.items()) or sig['checkpoint_sha256']!=digest(checkpoint):
        raise ValueError('Archived prompt lineage or Lean configuration differs')
    if sig['torch']!=str(torch.__version__) or sig['numpy']!=str(np.__version__) or sig['python']!=sys.version:
        raise ValueError('Software version differs from recorded capture')
    mps=read(mps_reference);mps_id=mps['identities']
    if (not mps.get('passed') or mps.get('auditor_sha256')!=digest(ROOT/'scripts/audit_lean_window_mps_full.py')
        or mps_id.get('capture_receipt_sha256')!=digest(capture_reference/'channel_feature_receipt.json')
        or mps_id.get('checkpoint_sha256')!=digest(checkpoint) or mps_id.get('fork_patches')!=raw['patch_sha256']
        or mps_id.get('mps_rope_sha256')!=digest(ROPE) or mps_id.get('mps_blocking_sha256')!=digest(BLOCKING)
        or mps_id.get('torch')!=str(torch.__version__) or not mps.get('mps_repeat_events_equal')
        or not mps.get('mps_repeat_selection_equal') or any(row['mps_repeat_pixel_difference'] for row in mps['rows'])):
        raise ValueError('Prior full-sequence MPS engineering audit differs')
    return {'checkpoint_sha256':digest(checkpoint),'fork_revision':revision,'fork_patches':raw['patch_sha256'],
      'policy_sha256':digest(POLICY),'runtime_sha256':digest(RUNTIME),'rope_sha256':digest(ROPE),
      'blocking_sha256':digest(BLOCKING),'anchor_sha256':digest(ANCHOR),
      'bypass_sha256':digest(FORK/'checkpoints/bypass_bottleneck_base.pth'),
      'lean_config_sha256':digest(FORK/'configs/lean_sam2/lean_sam2.yaml'),
      'capture_receipt_sha256':digest(capture_reference/'channel_feature_receipt.json'),
      'mps_reference_sha256':digest(mps_reference),'detector_sha256':sig['detector_sha256'],
      'manifest_sha256':digest(MANIFEST),'runner_sha256':digest(Path(__file__)),
      'torch':str(torch.__version__),'platform':platform.platform(),'machine':platform.machine(),
      'device':'mps','fallback':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
      'fast_math':os.environ.get('PYTORCH_MPS_FAST_MATH'),'seed':20261021,'capacity':16384,'insurance_reserve':3072}


def verify_gate(path,sequence,prompt,identity):
    gate=read(path)
    if not gate.get('passed') or gate.get('sequence')!=sequence or gate.get('prompt')!=prompt or gate.get('identity')!=identity or gate.get('verifier_sha256')!=digest(ROOT/'scripts/verify_lean_window_baseline_mps_gate.py'):
        raise ValueError('Matching short MPS engineering check required')
    return gate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',choices=('seq11','seq20','seq23'),required=True)
    p.add_argument('--limit',type=int,help='Short post-prompt engineering replay only')
    p.add_argument('--gate',type=Path)
    for k in ('checkpoint','preflight','capture-reference','mps-reference','cropped-reference','cropped-backup','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0':
        raise ValueError('MPS must be available with fast math and fallback disabled')
    if a.limit is not None and a.limit<10:raise ValueError('Engineering replay must allow memory eviction')
    identity=runtime_identity(a.checkpoint,a.capture_reference,a.mps_reference)
    videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')};frames=videos[a.sequence]
    start,prompt,viewport,provenance=prepare_archived_prompt(a.sequence,frames,a.preflight,a.cropped_reference,a.cropped_backup)
    gate=None
    if a.limit is None:
        if a.gate is None:raise ValueError('Full video requires a matching engineering check')
        gate=verify_gate(a.gate,a.sequence,provenance,identity)
    end=min(len(frames)-1,start+a.limit-1) if a.limit is not None else len(frames)-1
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261021);torch.mps.manual_seed(20261021);torch.use_deterministic_algorithms(True)
    sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='mps'),FORK)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Imported wrong SAM2 fork')
    if model.num_maskmem!=7 or model.mem_keep_ratio!=.25 or model.insurance_max_frames!=3 or not model.enable_insurance_bank or model.fill_hole_area!=0:
        raise ValueError('Lean memory or mask configuration differs')
    model.ema_history_replace_old_tail_in_attn=False;model.ema_history_resident_prune=False
    window=TokenWindowRuntime(16384,3072);model.token_window_runtime=window
    selection=[];anchor_events=install_multiscale_anchor(model,.5,selection_events=selection)
    state_before=model_state_hash(model)
    if gate is not None and gate['state_sha256']!=state_before:raise ValueError('Loaded model parameters or buffers differ from gate')
    records=[];positions=[]
    # Pre-prompt frames are explicitly frozen as empty predictions.
    for i in range(start):
        with Image.open(frames[i].image_path) as image: size=image.size
        path=out/f'{i:06d}.png';Image.fromarray(np.zeros((size[1],size[0]),np.uint8)).save(path)
        records.append({'frame':i,'source_frame_index':frames[i].frame_index,'image_sha256':digest(frames[i].image_path),'mask_sha256':digest(path)})
    with tempfile.TemporaryDirectory(prefix='frames_',dir=out) as temporary:
        stage=Path(temporary)
        if a.sequence=='seq23':stage_cropped_frames(frames,stage)
        else:stage_frames(frames,stage)
        state=model.init_state(video_path=str(stage),offload_video_to_cpu=True,offload_state_to_cpu=False)
        if state['num_frames']!=len(frames):raise ValueError('Video frame count differs')
        try:
            with torch.inference_mode(),indexed_spatial_rope_mps(model,positions),blocking_mps_pointer_position(model):
                _,ids,_=model.add_new_mask(state,frame_idx=start,obj_id=1,mask=prompt)
                if ids!=[1]:raise ValueError('Object identity differs')
                for frame,ids,logits in model.propagate_in_video(state,start_frame_idx=start,max_frame_num_to_track=end-start,reverse=False):
                    if frame!=len(records) or ids!=[1]:raise ValueError('Frame order or object identity differs')
                    mask=logits[0,0].detach().cpu().numpy()>0
                    if a.sequence=='seq23':mask=restore_mask(mask)
                    if mask.shape!=(viewport[3],viewport[2]):raise ValueError('Output geometry differs')
                    path=out/f'{frame:06d}.png';Image.fromarray(mask.astype(np.uint8)*255).save(path)
                    records.append({'frame':frame,'source_frame_index':frames[frame].frame_index,'image_sha256':digest(frames[frame].image_path),'mask_sha256':digest(path)})
        finally:model.reset_state(state)
    if len(records)!=end+1 or state_before!=model_state_hash(model):raise ValueError('Coverage or model state differs')
    writes=[e for e in window.events if e['stage']=='after_write'];plans=[e for e in window.events if e['stage']=='admission_plan']
    if len(writes)!=end-start or len(plans)!=len(writes) or any(e['total_tokens']>16384 or e['insurance_tokens']>3072 for e in writes):
        raise ValueError('Memory accounting or write coverage differs')
    if not any(p['evict_frames'] for p in plans):raise ValueError('Engineering or full run did not exercise eviction')
    result={'passed':True,'scope':'Exploratory automatic-mask project Lean token window, no learned refiner or label use.',
      'sequence':a.sequence,'prompt_frame':start,'prompt':provenance,'end_frame':end,'full_video':a.limit is None,
      'identity':identity,'state_sha256':state_before,'gate_sha256':digest(a.gate) if a.gate else None,
      'positions':positions,'selection':selection,'anchor_events':anchor_events,'token_window_events':window.events,
      'records':records}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':a.sequence,'frames':len(records),'prompt_frame':start,
      'writes':len(writes),'evictions':sum(bool(p['evict_frames']) for p in plans)}))


if __name__=='__main__':main()
