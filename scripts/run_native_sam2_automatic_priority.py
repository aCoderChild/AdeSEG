#!/usr/bin/env python3
"""Run pinned native SAM2 on the existing automatic-mask comparison track.

Uses the clean official checkpoint/revision and the exact archived automatic
mask and fixed seq23 crop shared with SAMURAI. No labels or detector are read.
This is a new automatic-prompt track, not a rerun of historical GT baselines.
"""
import argparse
from contextlib import nullcontext
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest, stage_frames
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.seq23_viewport import restore_mask, stage_cropped_frames
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_samurai_automatic_priority import MANIFEST, model_state_hash, prepare_archived_prompt

REPO = ROOT / 'external/sam2_diagnostic_reference'
REVISION = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
CONFIG = REPO / 'sam2/configs/sam2.1/sam2.1_hiera_b+.yaml'
PROTOCOL = ROOT / 'protocols/native_sam2_automatic_priority_v1.json'
ADAPTER = ROOT / 'modeling/lean_mps_blocking_pointer_position.py'


def read(path): return json.loads(Path(path).read_text())


def native_source_identity():
    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != REVISION or dirty:
        raise ValueError('Native official reference revision or tracked files changed')
    return revision


def identity(checkpoint, device):
    protocol = read(PROTOCOL)
    if digest(checkpoint) != protocol['checkpoint_sha256'] or native_source_identity() != protocol['author_revision']:
        raise ValueError('Pinned native model or checkpoint differs')
    return {'checkpoint_sha256': digest(checkpoint), 'revision': REVISION, 'config_sha256': digest(CONFIG),
            'protocol_sha256': digest(PROTOCOL), 'manifest_sha256': digest(MANIFEST),
            'runner_sha256': digest(Path(__file__)), 'adapter_sha256': digest(ADAPTER),
            'prompt_helper_sha256': digest(ROOT / 'scripts/run_samurai_automatic_priority.py'),
            'video_loader_sha256': digest(ROOT / 'datasets/video.py'),
            'viewport_sha256': digest(ROOT / 'modeling/seq23_viewport.py'),
            'torch': str(torch.__version__), 'numpy': np.__version__, 'python': sys.version,
            'platform': platform.platform(), 'machine': platform.machine(), 'device': device,
            'dtype': 'float32', 'seed': 20261021,
            'fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
            'fast_math': os.environ.get('PYTORCH_MPS_FAST_MATH')}


def verify_gate(path, sequence, prompt, current):
    gate = read(path)
    if (not gate.get('passed') or gate.get('sequence') != sequence or gate.get('prompt') != prompt
            or gate.get('mps_identity') != current
            or gate.get('verifier_sha256') != digest(ROOT / 'scripts/verify_native_sam2_automatic_gate.py')):
        raise ValueError('A matching independently checked MPS replay is required')
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', choices=('seq11', 'seq20', 'seq23'), required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), required=True)
    parser.add_argument('--limit', type=int, help='Short post-prompt engineering replay')
    parser.add_argument('--gate', type=Path)
    for name in ('checkpoint', 'preflight', 'cropped-reference', 'cropped-backup', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 10:
        raise ValueError('Engineering replay must include at least ten frames')
    if args.device == 'mps' and (not torch.backends.mps.is_available()
            or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0'
            or os.environ.get('PYTORCH_MPS_FAST_MATH') != '0'):
        raise ValueError('MPS must be available with fallback and fast math disabled')
    current = identity(args.checkpoint, args.device)
    videos = {**load_manifest(MANIFEST,'dev'), **load_manifest(MANIFEST,'test')}
    frames = videos[args.sequence]
    start, prompt, viewport, provenance = prepare_archived_prompt(
        args.sequence, frames, args.preflight, args.cropped_reference, args.cropped_backup)
    gate = None
    if args.limit is None:
        if args.device != 'mps' or args.gate is None:
            raise ValueError('Full videos require the independently verified MPS gate')
        gate = verify_gate(args.gate, args.sequence, provenance, current)
    post = frames[start:][:args.limit] if args.limit is not None else frames[start:]
    out = validate_output(args.output); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(20261021); torch.use_deterministic_algorithms(True)
    if args.device == 'mps': torch.mps.manual_seed(20261021)
    sys.path.insert(0, str(REPO))
    model = build_author_predictor(SimpleNamespace(method='sam2',checkpoint=args.checkpoint,device=args.device), REPO)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(REPO.resolve()) or model.fill_hole_area != 0:
        raise ValueError('Wrong native package or mask postprocessing')
    before = model_state_hash(model)
    if gate is not None and gate['state_sha256'] != before:
        raise ValueError('Loaded native parameters or buffers differ from the gate')
    records = []
    # Save the same explicit empty pre-prompt frames used in the automatic track.
    for i in range(start):
        with Image.open(frames[i].image_path) as image: width,height=image.size
        path=out/f'{i:06d}.png';Image.fromarray(np.zeros((height,width),np.uint8)).save(path)
        records.append({'frame':i,'source_frame_index':frames[i].frame_index,
                        'image_sha256':digest(frames[i].image_path),'mask_sha256':digest(path),'post_prompt':False})
    with tempfile.TemporaryDirectory(prefix='native_auto_frames_', dir=out) as temp:
        stage=Path(temp)
        if args.sequence=='seq23':stage_cropped_frames(post,stage)
        else:stage_frames(post,stage)
        state=model.init_state(video_path=str(stage),offload_video_to_cpu=True,offload_state_to_cpu=False)
        if state['num_frames'] != len(post): raise ValueError('Input video coverage differs')
        adapter=blocking_mps_pointer_position(model) if args.device=='mps' else nullcontext()
        try:
            with torch.inference_mode(),adapter:
                _,ids,_=model.add_new_mask(state,frame_idx=0,obj_id=1,mask=prompt)
                if ids != [1]:raise ValueError('Unexpected object ID')
                for local,ids,logits in model.propagate_in_video(state,start_frame_idx=0,reverse=False):
                    if ids != [1] or local != len(records)-start:raise ValueError('Frame order or object ID differs')
                    mask=logits[0,0].detach().cpu().numpy()>0
                    if args.sequence=='seq23':mask=restore_mask(mask)
                    if mask.shape != (viewport[3],viewport[2]):raise ValueError('Output geometry differs')
                    i=start+local;path=out/f'{i:06d}.png';Image.fromarray(mask.astype(np.uint8)*255).save(path)
                    records.append({'frame':i,'source_frame_index':post[local].frame_index,
                                    'image_sha256':digest(post[local].image_path),'mask_sha256':digest(path),'post_prompt':True})
        finally:model.reset_state(state)
    if len(records)!=start+len(post) or model_state_hash(model)!=before:
        raise ValueError('Prediction coverage or model state differs')
    result={'passed':True,'scope':'Exploratory label-free automatic-mask native SAM2, no learned refiner.',
            'sequence':args.sequence,'prompt_frame':start,'prompt':provenance,'rebased_start':0,
            'full_video':args.limit is None,'device':args.device,'identity':current,
            'state_sha256':before,'deterministic':torch.are_deterministic_algorithms_enabled(),
            'blocking_pointer_transfer':args.device=='mps','gate_sha256':digest(args.gate) if args.gate else None,
            'frames':len(records),'post_prompt_frames':len(post),'records':records}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':args.sequence,'device':args.device,'prompt_frame':start,'frames':len(records)}))


if __name__=='__main__':main()
