#!/usr/bin/env python3
"""Run pinned SAMURAI using archived automatic Lean masks, without labels.

This is an exploratory prompt-matched comparator, not original Lean or an
independent test. Use a fresh process per sequence and device.  Predictions
are frozen before any separate evaluation; no labels or detector are read.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest, stage_frames
from modeling.samurai_auto_prompt import prepare_prompt
from modeling.samurai_mps_kalman import portable_kalman_iou
from modeling.seq23_viewport import restore_mask, stage_cropped_frames
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, source_identity

MANIFEST = ROOT / 'data/polypgen_sequence.jsonl'
PROTOCOL = ROOT / 'protocols/samurai_automatic_priority_v1.json'
CONFIG = ROOT / 'external/SAMURAI/sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml'


def read(path):
    return json.loads(path.read_text())


def prepare_archived_prompt(sequence, frames, preflight, cropped_reference, cropped_backup):
    """Return archived prediction prompt, ROI and its immutable provenance."""
    pre = read(preflight / 'manifest.json')
    done = read(preflight / 'completion.json')
    if not done.get('passed') or done.get('manifest_sha256') != digest(preflight / 'manifest.json'):
        raise ValueError('Archived automatic split preflight differs')
    item = pre['sequences'][sequence]
    source = Path(item['source']) / sequence
    if len(frames) != item['frame_count'] or digest(source / 'frames.json') != item['frames_metadata_sha256']:
        raise ValueError('Archived sequence metadata differs')
    rows = read(source / 'frames.json')
    if len(rows) != len(frames) or any(digest(frame.image_path) != row['image_sha256'] for frame, row in zip(frames, rows, strict=True)):
        raise ValueError('Images do not match archived automatic rollout')
    if sequence == 'seq23':
        done = read(cropped_reference / 'completion.json')
        if not done.get('passed') or digest(cropped_reference / 'frames.json') != done['frames_sha256'] or digest(cropped_reference / 'prompt.json') != done['prompt_sha256']:
            raise ValueError('Cropped reference receipt differs')
        data = read(cropped_reference / 'prompt.json')
        src = cropped_reference / 'prompt.png'
        # The historical completion receipt does not bind prompt.png: require
        # the separately archived crop's matching prompt and record both hashes.
        backup = cropped_backup / 'prompt.png'
        if digest(src) != digest(backup):
            raise ValueError('Independent cropped prompt copies differ')
        cropped_rows = read(cropped_reference / 'frames.json')
        if len(cropped_rows) != len(frames) or any(digest(frame.image_path) != row['image_sha256'] for frame, row in zip(frames, cropped_rows, strict=True)):
            raise ValueError('Cropped reference image sequence differs')
        prompt_frame = data['first_detection_index']
        prompt_sha = digest(cropped_reference / 'prompt.json')
        reference_hash = digest(cropped_reference / 'completion.json')
    else:
        if digest(source / 'prompt.json') != item['prompt_sha256']:
            raise ValueError('Archived prompt metadata differs')
        data = read(source / 'prompt.json')
        src = source / 'prompt_refined.png'
        if digest(src) != item['prompt_mask_sha256'] or data['refined_mask_sha256'] != item['prompt_mask_sha256']:
            raise ValueError('Archived prompt mask differs')
        prompt_frame = data['first_detection_index']
        prompt_sha = digest(source / 'prompt.json')
        reference_hash = digest(preflight / 'completion.json')
    if type(prompt_frame) is not int or prompt_frame < 0 or prompt_frame >= len(frames):
        raise ValueError('Archived automatic prompt is unavailable')
    with Image.open(frames[prompt_frame].image_path) as image:
        width, height = image.size
    prompt, viewport = prepare_prompt(np.asarray(Image.open(src).convert('L')), sequence, (height, width))
    if not prompt.any():
        raise ValueError('Empty initial automatic mask is not supported by this comparison')
    return prompt_frame, prompt, viewport, {'prompt_file': str(src), 'prompt_mask_sha256': digest(src),
        'prompt_metadata_sha256': prompt_sha, 'reference_receipt_sha256': reference_hash,
        'preflight_sha256': digest(preflight / 'manifest.json'), 'roi_xyxy': list(viewport)}


def model_state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def verify_gate(path, *, sequence, device, checkpoint, prompt, revision):
    """Verify frozen short-replay evidence before any full-video inference."""
    gate = read(path)
    expected = {'sequence': sequence, 'device': device, 'checkpoint_sha256': digest(checkpoint),
                'config_sha256': digest(CONFIG), 'manifest_sha256': digest(MANIFEST),
                'protocol_sha256': digest(PROTOCOL), 'runner_sha256': digest(Path(__file__)),
                'kalman_adapter_sha256': digest(ROOT / 'modeling/samurai_mps_kalman.py'),
                'verifier_sha256': digest(ROOT / 'scripts/verify_samurai_automatic_gate.py'),
                'source_revision': revision, 'torch': str(torch.__version__), 'prompt': prompt}
    if not gate.get('passed') or any(gate.get(key) != value for key, value in expected.items()):
        raise ValueError('Full video requires matching independently checked engineering replays')
    return gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence', choices=('seq11', 'seq20', 'seq23'), required=True)
    p.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    p.add_argument('--limit', type=int, help='Optional engineering-only post-prompt frame count')
    p.add_argument('--gate', type=Path, help='Verified same-sequence/device engineering receipt for full videos')
    for k in ('checkpoint', 'preflight', 'cropped-reference', 'cropped-backup', 'output'):
        p.add_argument('--' + k, type=Path, required=True)
    args = p.parse_args()
    if args.limit is not None and args.limit < 2:
        raise ValueError('Engineering check needs at least two frames')
    if args.device == 'mps' and (not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0' or os.environ.get('PYTORCH_MPS_FAST_MATH') != '0'):
        raise ValueError('MPS must be available, with fallback and fast math explicitly disabled')
    repo, rev = source_identity('samurai')
    expected_ckpt = read(ROOT / 'protocols/memory_token_regions.json')['model']['checkpoint_sha256']
    if digest(args.checkpoint) != expected_ckpt:
        raise ValueError('SAM2.1 B+ checkpoint differs')
    videos = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    frames = videos[args.sequence]
    start, prompt, viewport, provenance = prepare_archived_prompt(args.sequence, frames, args.preflight, args.cropped_reference, args.cropped_backup)
    if args.limit is None:
        if args.gate is None: raise ValueError('Full video requires a verified gate')
        gate = verify_gate(args.gate, sequence=args.sequence, device=args.device, checkpoint=args.checkpoint, prompt=provenance, revision=rev)
    else:
        gate = None
    post_frames = frames[start:]
    if args.limit is not None:
        post_frames = post_frames[:args.limit]
    out = validate_output(args.output); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(20261021); torch.use_deterministic_algorithms(True)
    if args.device == 'mps': torch.mps.manual_seed(20261021)
    sys.path.insert(0, str(repo / 'sam2'))
    model = build_author_predictor(SimpleNamespace(method='samurai', checkpoint=args.checkpoint, device=args.device), repo)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(repo.resolve()) or model.fill_hole_area != 0:
        raise ValueError('Wrong author package or mask postprocessing')
    state_before = model_state_hash(model)
    if gate is not None and gate['state_sha256'] != state_before:
        raise ValueError('Actual loaded model state differs from engineering replay')
    records = []; kalman_events = []
    with tempfile.TemporaryDirectory(prefix='frames_', dir=out) as temp:
        stage = Path(temp)
        if args.sequence == 'seq23': stage_cropped_frames(post_frames, stage)
        else: stage_frames(post_frames, stage)
        state = model.init_state(video_path=str(stage), offload_video_to_cpu=True, offload_state_to_cpu=False)
        if state['num_frames'] != len(post_frames): raise ValueError('Video frame count differs')
        try:
            with torch.inference_mode(), portable_kalman_iou(model, args.device, kalman_events):
                _, ids, _ = model.add_new_mask(state, frame_idx=0, obj_id=1, mask=prompt)
                if ids != [1]: raise ValueError('Object identity differs')
                for local, ids, logits in model.propagate_in_video(state, start_frame_idx=0, reverse=False):
                    if local != len(records) or ids != [1]: raise ValueError('Rebased video output order differs')
                    mask = logits[0, 0].detach().cpu().numpy() > 0
                    if args.sequence == 'seq23': mask = restore_mask(mask)
                    if mask.shape != (viewport[3], viewport[2]): raise ValueError('Output geometry differs')
                    file = out / f'{start + local:06d}.png'
                    Image.fromarray(mask.astype(np.uint8) * 255).save(file)
                    records.append({'local_index': local, 'frame': start + local, 'source_frame_index': post_frames[local].frame_index,
                                    'image_sha256': digest(post_frames[local].image_path), 'mask_sha256': digest(file)})
        finally:
            model.reset_state(state)
    if len(records) != len(post_frames) or model_state_hash(model) != state_before:
        raise ValueError('Output coverage or model state differs')
    receipt = {'passed': True, 'scope': 'Exploratory label-free automatic-prompt SAMURAI; not a clinical or independent evaluation.',
        'sequence': args.sequence, 'prompt_frame': start, 'prompt': provenance, 'rebased_start': 0,
        'full_video': args.limit is None, 'device': args.device, 'dtype': 'float32', 'torch': str(torch.__version__),
        'seed': 20261021, 'deterministic': torch.are_deterministic_algorithms_enabled(),
        'fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'), 'checkpoint_sha256': digest(args.checkpoint),
        'fast_math': os.environ.get('PYTORCH_MPS_FAST_MATH'), 'gate_sha256': digest(args.gate) if args.gate else None,
        'kalman_adapter_sha256': digest(ROOT / 'modeling/samurai_mps_kalman.py'), 'kalman_iou_events': kalman_events,
        'config_sha256': digest(CONFIG), 'manifest_sha256': digest(MANIFEST), 'protocol_sha256': digest(PROTOCOL),
        'runner_sha256': digest(Path(__file__)), 'source_revision': rev, 'state_sha256': state_before,
        'frames': len(records), 'records': records}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'passed': True, 'sequence': args.sequence, 'device': args.device, 'frames': len(records), 'prompt_frame': start}))


if __name__ == '__main__': main()
