#!/usr/bin/env python3
"""Pinned Lean with explicitly corrected indexed MPS RoPE, automatic prompts.

No project token window, multiscale anchor, refiner or detector call is added.
This is a compatibility-corrected Lean system, not byte-identical author code.
"""
import argparse
from pathlib import Path
import json
import os
import platform
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import stage_frames
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.seq23_viewport import restore_mask, stage_cropped_frames
from scripts.crossfit_observer_capture_shared_v1 import sequence_inputs, validate_plan
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v2 import (CHECKPOINT_SHA, DATA, DETECTOR_SHA,
    SEQUENCES as ADDITIONAL, input_data, new_output, sources as additional_sources)
from scripts.run_author_baseline import build_author_predictor, source_identity
from scripts.run_samurai_automatic_priority import model_state_hash
from scripts.source_group_label_guard_v1 import install_mask_open_guard

PRIORITY = ('seq11', 'seq20', 'seq23')
SEQUENCES = tuple(f'seq{i}' for i in range(1, 24) if i not in (1, 7))
SOURCES = ('scripts/run_author_lean_automatic_v1.py', 'scripts/verify_author_lean_automatic_gate_v1.py',
    'scripts/run_author_baseline.py',
    'scripts/run_additional_author_comparator_v2.py', 'scripts/crossfit_observer_capture_shared_v1.py',
    'scripts/source_group_label_guard_v1.py', 'modeling/lean_indexed_spatial_rope_mps.py',
    'modeling/lean_mps_blocking_pointer_position.py', 'modeling/seq23_viewport.py',
    'datasets/video.py', 'scripts/memory_token_io.py', 'protocols/AUTHOR_LEAN_AUTOMATIC_MPS_V1.md')


def inputs(sequence, extended, remaining, checkpoint, detector):
    if sequence not in SEQUENCES:
        raise ValueError('Sequence is outside the 21 initialized cases')
    if sequence in ADDITIONAL:
        return input_data(sequence, extended, remaining, checkpoint, detector)
    plan = validate_plan()
    frames, _, start, _, prompt, _, _, item = sequence_inputs(plan, sequence)
    return frames, start, prompt, item['prompt_provenance']


def identity(sequence, checkpoint, detector):
    if sequence not in SEQUENCES or digest(checkpoint) != CHECKPOINT_SHA or digest(detector) != DETECTOR_SHA:
        raise ValueError('Sequence, backbone or archived detector differs')
    repo, rev = source_identity('lean')
    assets = ('sam2/configs/sam2.1/sam2.1_hiera_b+.yaml', 'configs/runtime/lvos.yaml',
              'configs/lean_sam2/lean_sam2.yaml', 'checkpoints/bypass_bottleneck_base.pth')
    return {'sequence': sequence, 'author_revision': rev, 'author_repo': str(repo.resolve()),
            'checkpoint_sha256': digest(checkpoint), 'detector_sha256': digest(detector),
            'assets': {p: digest(repo / p) for p in assets}, 'data_sha256': digest(DATA),
            'sources': {p: digest(ROOT / p) for p in SOURCES}, 'torch': str(torch.__version__),
            'additional_sources': additional_sources(),
            'numpy': np.__version__, 'python': sys.version, 'platform': platform.platform(),
            'device': 'mps', 'dtype': 'float32', 'seed': 20261021,
            'hash_seed': os.environ.get('PYTHONHASHSEED'),
            'fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
            'fast_math': os.environ.get('PYTORCH_MPS_FAST_MATH')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence', choices=SEQUENCES, required=True)
    p.add_argument('--limit', type=int, help='Number of frames including initialization; short engineering run only')
    p.add_argument('--gate', type=Path)
    for name in ('checkpoint', 'detector', 'extended-preflight', 'remaining-preflight', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if (os.environ.get('PYTHONHASHSEED') != '0' or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0'
            or os.environ.get('PYTORCH_MPS_FAST_MATH') != '0' or not torch.backends.mps.is_available()):
        raise ValueError('MPS or required environment unavailable')
    if a.limit is not None and a.limit < 5:
        raise ValueError('A smoke needs at least five frames')
    if (a.limit is None) != (a.gate is not None):
        raise ValueError('Only full videos require a verified MPS smoke gate')
    guard = install_mask_open_guard(DATA)
    frames, start, prompt, provenance = inputs(a.sequence, a.extended_preflight, a.remaining_preflight,
                                               a.checkpoint, a.detector)
    ident = identity(a.sequence, a.checkpoint, a.detector)
    end = min(len(frames), start + a.limit) if a.limit is not None else len(frames)
    if end - start < 5:
        raise ValueError('Insufficient propagation frames')
    if a.gate is not None:
        from scripts.verify_author_lean_automatic_gate_v1 import verify_gate
        gate = verify_gate(a.gate, ident, a.checkpoint, a.detector, a.extended_preflight, a.remaining_preflight)
    else:
        gate = None
    out = new_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(20261021); torch.mps.manual_seed(20261021)
    torch.use_deterministic_algorithms(True)
    repo, _ = source_identity('lean'); sys.path.insert(0, str(repo))
    model = build_author_predictor(SimpleNamespace(method='lean', checkpoint=a.checkpoint, device='mps'), repo)
    import sam2
    if (not Path(sam2.__file__).resolve().is_relative_to(repo.resolve()) or model.num_maskmem != 7
            or model.mem_keep_ratio != .25 or model.insurance_max_frames != 3
            or not model.enable_insurance_bank or model.fill_hole_area != 0):
        raise ValueError('Author package or Lean configuration differs')
    before = model_state_hash(model)
    if gate is not None and before != gate['state_sha256']:
        raise ValueError('Model differs from MPS gate')
    records, positions = [], []
    for i in range(start):
        with Image.open(frames[i].image_path) as image: width, height = image.size
        path = out / f'{i:06d}.png'; Image.fromarray(np.zeros((height, width), np.uint8)).save(path)
        records.append({'frame': i, 'source_frame_index': frames[i].frame_index,
                        'image_sha256': digest(frames[i].image_path), 'mask_sha256': digest(path), 'post_prompt': False})
    with tempfile.TemporaryDirectory(prefix='frames_', dir=out) as tmp:
        if a.sequence == 'seq23': stage_cropped_frames(frames[:end], tmp)
        else: stage_frames(frames[:end], Path(tmp))
        state = model.init_state(video_path=tmp, offload_video_to_cpu=True, offload_state_to_cpu=False)
        if state['num_frames'] != end or (state['video_height'], state['video_width']) != prompt.shape:
            raise ValueError('Staged inputs or prompt geometry differ')
        try:
            with torch.inference_mode(), indexed_spatial_rope_mps(model, positions), blocking_mps_pointer_position(model):
                _, ids, _ = model.add_new_mask(state, frame_idx=start, obj_id=1, mask=prompt)
                if ids != [1]: raise ValueError('Unexpected object identity')
                for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=start, reverse=False):
                    if (frame != len(records) or ids != [1] or logits.ndim != 4 or
                            tuple(logits.shape[:2]) != (1, 1) or not bool(torch.isfinite(logits).all())):
                        raise ValueError('Output order, object or finite logits differ')
                    mask = logits[0, 0].detach().float().cpu().numpy() > 0
                    if a.sequence == 'seq23': mask = restore_mask(mask)
                    with Image.open(frames[frame].image_path) as image: shape = (image.height, image.width)
                    if mask.shape != shape: raise ValueError('Output geometry differs')
                    path = out / f'{frame:06d}.png'; Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                    records.append({'frame': frame, 'source_frame_index': frames[frame].frame_index,
                        'image_sha256': digest(frames[frame].image_path), 'mask_sha256': digest(path), 'post_prompt': True})
        finally:
            model.reset_state(state)
    if len(records) != end or model_state_hash(model) != before or guard['open_attempts']:
        raise ValueError('Coverage, state or label guard differs')
    receipt = {'passed': True, 'scope': 'Exploratory author Lean with indexed MPS RoPE compatibility correction.',
        'sequence': a.sequence, 'prompt_frame': start, 'prompt': provenance, 'identity': ident,
        'full_video': a.limit is None, 'state_sha256': before, 'positions': positions,
        'process_id': os.getpid(),
        'gate_sha256': digest(a.gate) if a.gate else None, 'label_guard': dict(guard), 'records': records}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'passed': True, 'sequence': a.sequence, 'frames': len(records)}))


if __name__ == '__main__': main()
