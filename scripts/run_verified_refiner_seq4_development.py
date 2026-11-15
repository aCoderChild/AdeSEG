#!/usr/bin/env python3
"""Freeze one seq4 development factorial arm under an owned MPS runtime.

The development screen opens validation labels; video propagation does not
open target-sequence masks. The factory must verify assets, model/buffers,
hooks and same-model baseline replays before any candidate propagation.
This is restricted to the archived seq4 automatic prompt; it is not a full
cohort or independent-evaluation runner.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest, stage_frames
from modeling.lean_contrast_refinement_integration import route_lean_contrast_refinement
from modeling.lean_verified_window_runtime import verified_mps_window_refiner
from scripts.memory_token_io import digest, validate_output
from scripts.verify_token_window_refiner_routing_mps import tensor_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ('preflight', 'split-preflight', 'static-capture', 'seq4-capture',
             'checkpoint', 'detector', 'baseline-reference', 'candidate-run',
             'control-run', 'data-preflight', 'runner-source', 'output')
    for name in names:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=('A', 'B', 'C', 'D'), required=True)
    parser.add_argument('--global-run', type=Path)
    parser.add_argument('--shuffled-run', type=Path)
    args = parser.parse_args()
    out = validate_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    options = vars(args).copy()
    options.pop('arm')
    options['output'] = out / 'verification'
    with verified_mps_window_refiner(**options) as (model, refiner, lease):
        pre = json.loads((args.split_preflight / 'manifest.json').read_text())
        source = pre['sequences']['seq4']
        folder = Path(source['source']) / 'seq4'
        prompt_path = folder / 'prompt.json'
        if digest(prompt_path) != source['prompt_sha256']:
            raise ValueError('Prompt changed after runtime verification')
        prompt = json.loads(prompt_path.read_text())
        videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
                  **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
        frames = videos['seq4']
        metadata = json.loads((folder / 'frames.json').read_text())
        if len(frames) != len(metadata) or any(
                digest(frame.image_path) != metadata[i]['image_sha256'] for i, frame in enumerate(frames)):
            raise ValueError('Sequence images changed after runtime verification')
        start = prompt['first_detection_index']
        for i in range(start):
            with Image.open(frames[i].image_path) as image:
                width, height = image.size
            Image.fromarray(np.zeros((height, width), dtype=np.uint8)).save(out / f'{i:06d}.png')
        rows = []
        with tempfile.TemporaryDirectory(prefix='seq4_frames_', dir=out) as temporary:
            stage_frames(frames, Path(temporary))
            state = model.init_state(video_path=temporary, offload_video_to_cpu=True,
                                     offload_state_to_cpu=False)
            try:
                with torch.inference_mode(), route_lean_contrast_refinement(
                        model, refiner, args.arm, runtime_lease=lease):
                    _, ids, _ = model.add_new_points_or_box(
                        state, frame_idx=start, obj_id=1, box=np.asarray(prompt['box'], np.float32))
                    _, ids, logits = model.add_new_points_or_box(
                        state, frame_idx=start, obj_id=1,
                        points=np.asarray(prompt['points'], np.float32),
                        labels=np.array([1, 0], np.int32), clear_old_points=False)
                    initial = logits[0, 0].cpu().numpy() > 0
                    if ids != [1] or not initial.any():
                        raise ValueError('Unexpected automatic prompt')
                    model.add_new_mask(state, frame_idx=start, obj_id=1, mask=initial)
                    for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=start, reverse=False):
                        if ids != [1] or frame != start + len(rows):
                            raise ValueError('Unexpected frame order')
                        path = out / f'{frame:06d}.png'
                        Image.fromarray((logits[0, 0].cpu().numpy() > 0).astype(np.uint8) * 255).save(path)
                        row = {'frame': frame, 'mask_sha256': digest(path)}
                        if frame > start:
                            entry = state['output_dict_per_obj'][0]['non_cond_frame_outputs'][frame]
                            for key in ('pred_masks', 'maskmem_features', 'obj_ptr', 'object_score_logits'):
                                row[key + '_sha256'] = tensor_hash(entry[key])
                        rows.append(row)
            finally:
                model.reset_state(state)
        if len(rows) != len(frames)-start:
            raise ValueError('Incomplete video output')
        runtime = model.token_window_runtime
        writes = [x for x in runtime.events if x['stage'] == 'after_write']
        if any(x['total_tokens'] > 16384 or x['insurance_tokens'] > 3072 for x in writes):
            raise ValueError('Memory budget exceeded')
        (out / 'memory_events.json').write_text(json.dumps(runtime.events, indent=2) + '\n')
        # Target-sequence labels are not opened during video propagation.
        receipt = {'passed': True, 'arm': args.arm, 'sequence': 'seq4',
                   'scope': 'Frozen development predictions only; not scored, not independent evaluation.',
                   'prompt_frame': start, 'prompt_sha256': digest(prompt_path),
                   'candidate_checkpoint_sha256': digest(args.candidate_run / 'checkpoint.pt'),
                   'runtime_verification_sha256': digest(out / 'verification/runtime_verification.json'),
                   'memory_events_sha256': digest(out / 'memory_events.json'),
                   'masks': [{'frame': i, 'sha256': digest(out / f'{i:06d}.png')} for i in range(len(frames))],
                   'post_prompt_state': rows, 'runner_source_sha256': digest(Path(__file__))}
    # Write completion only after the owned runtime has verified tensor state on exit.
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'passed': True, 'arm': args.arm, 'frames': len(frames), 'prompt_frame': start}))


if __name__ == '__main__':
    main()
