#!/usr/bin/env python3
"""Compare two automatic-mask initialization paths on a real MPS video.

Both arms inject the exact same frozen, archived automatic mask. One first
executes historical box/point calls, the other injects the mask directly.
This is a no-label engineering test, not learned-refiner efficacy.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.lean_live_state_audit import module_tensor_fingerprints
from modeling.seq23_viewport import stage_cropped_frames
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_verified_refiner_priority_development import priority_inputs
from scripts.verify_refiner_capture_static import verify as verify_capture
from scripts.verify_token_window_refiner_routing_mps import FORK, run_one


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight', 'split-preflight', 'static-capture', 'checkpoint',
                 'detector', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sequence', choices=('seq4', 'seq23'), default='seq4')
    p.add_argument('--seq23-reference', type=Path)
    p.add_argument('--seq23-backup', type=Path)
    p.add_argument('--end-frame', type=int, default=13)
    a = p.parse_args()
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0' or os.environ.get('PYTORCH_MPS_FAST_MATH') != '0':
        raise RuntimeError('MPS with explicit no-fallback and no-fast-math is required')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    static = verify_capture(a.preflight, a.split_preflight, a.static_capture, a.checkpoint, a.detector)
    frames, prompt, prompt_sha, view, mask, provenance = priority_inputs(
        a.sequence, a.split_preflight, a.seq23_reference, a.seq23_backup, a.checkpoint, a.detector)
    if a.end_frame <= prompt['first_detection_index'] or a.end_frame >= len(frames):
        raise ValueError('Invalid engineering frame range')
    out = validate_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(FORK))
    reference = build_author_predictor(SimpleNamespace(method='lean', checkpoint=a.checkpoint, device='cpu'), FORK)
    expected = module_tensor_fingerprints(reference)
    del reference
    with tempfile.TemporaryDirectory(prefix='prompt_viewport_', dir=out) as temporary:
        if a.sequence == 'seq23':
            stage_cropped_frames(frames, temporary)
            inputs = [SimpleNamespace(image_path=Path(temporary) / f'{i:06d}.jpg') for i in range(len(frames))]
        else:
            inputs = frames
        warmed = run_one('box_points_then_frozen_mask', None, None, a.checkpoint, expected,
                         inputs, prompt, a.end_frame, out, frozen_mask=mask, warm_with_points=True)
        direct = run_one('direct_frozen_mask', None, None, a.checkpoint, expected,
                         inputs, prompt, a.end_frame, out, frozen_mask=mask, warm_with_points=False)
    rows = []
    for x, y in zip(warmed['rows'], direct['rows'], strict=True):
        if x['frame'] != y['frame']:
            raise ValueError('Frame order differs')
        rows.append({'frame': x['frame'], 'equal_fields': {k: x.get(k) == y.get(k)
                     for k in ('mask_sha256', 'logits_sha256', 'pred_masks_sha256',
                               'maskmem_features_sha256', 'obj_ptr_sha256', 'object_score_logits_sha256')}})
    result = {'passed': all(all(row['equal_fields'].values()) for row in rows)
              and warmed['token_window_events'] == direct['token_window_events']
              and warmed['selection_events'] == direct['selection_events']
              and warmed['positions'] == direct['positions'],
              'scope': 'One MPS initialization engineering check; no labels or learned refiner.',
              'sequence': a.sequence, 'view': view, 'prompt_metadata_sha256': prompt_sha,
              'prompt_mask_sha256': provenance['prompt_mask_sha256'],
              'checkpoint_sha256': digest(a.checkpoint), 'static_manifest_sha256': static['manifest_sha256'],
              'rows': rows,
              'memory_events_equal': warmed['token_window_events'] == direct['token_window_events'],
              'selection_equal': warmed['selection_events'] == direct['selection_events'],
              'positions_equal': warmed['positions'] == direct['positions'],
              'warm_receipt_sha256': digest(out / 'box_points_then_frozen_mask/receipt.json'),
              'direct_receipt_sha256': digest(out / 'direct_frozen_mask/receipt.json'),
              'runner_sha256': digest(Path(__file__))}
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
