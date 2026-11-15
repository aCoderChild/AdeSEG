#!/usr/bin/env python3
"""Check recorded identities and compare two full MPS replays to the CPU archive.

This audits saved engineering replays only. It does not verify a live model,
establish numerical equivalence, or authorize learned-refiner inference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output

SOURCE_FILES = {
    'runner_sha256': 'scripts/verify_lean_window_mps_replay.py',
    'cpu_rope_sha256': 'modeling/lean_indexed_spatial_rope.py',
    'mps_rope_sha256': 'modeling/lean_indexed_spatial_rope_mps.py',
    'mps_blocking_sha256': 'modeling/lean_mps_blocking_pointer_position.py',
}


def verify_identities(receipts, expected):
    """Check values actually recorded, without asserting unrecorded settings."""
    for receipt in receipts:
        if not receipt.get('passed') or receipt.get('device') != 'mps' or receipt.get('fallback') != '0':
            raise ValueError('Invalid MPS receipt or fallback setting')
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(f'Recorded runtime identity differs: {key}')


def read_json(path):
    return json.loads(path.read_text())


def audit(reference, preflight, checkpoint, mps_a, mps_b, output):
    ref = read_json(reference / 'channel_feature_receipt.json')
    pre = read_json(preflight / 'manifest.json')
    pre_done = read_json(preflight / 'completion.json')
    if not pre_done.get('passed') or pre_done.get('manifest_sha256') != digest(preflight / 'manifest.json'):
        raise ValueError('Preflight changed')
    if not ref.get('passed') or ref.get('sequence') != 'seq4':
        raise ValueError('Unexpected capture reference')
    raw = ref['pipeline']['raw_identity']
    source = pre['sequences']['seq4']
    prompt = Path(source['source']) / 'seq4' / 'prompt.json'
    if digest(prompt) != source['prompt_sha256'] or digest(checkpoint) != raw['checkpoint_sha256']:
        raise ValueError('Prompt or model checkpoint changed')
    for path, expected in raw['patch_sha256'].items():
        if digest(ROOT / 'external/Lean-SAM2-token-window' / path) != expected:
            raise ValueError('Token-window fork patch changed')
    for key, path in [('policy_sha256', 'modeling/token_memory_window.py'),
                      ('runtime_sha256', 'modeling/lean_token_window_storage.py')]:
        if digest(ROOT / path) != raw[key]:
            raise ValueError('Token-window policy or runtime changed')
    expected = {key: digest(ROOT / path) for key, path in SOURCE_FILES.items()}
    expected.update({'sequence': 'seq4', 'prompt_frame': 9, 'end_frame': source['frame_count'] - 1,
                     'capture_receipt_sha256': digest(reference / 'channel_feature_receipt.json'),
                     'checkpoint_sha256': digest(checkpoint), 'prompt_sha256': digest(prompt),
                     'fork_patches': raw['patch_sha256'], 'torch': ref['pipeline']['torch']})
    paths = (mps_a, mps_b)
    receipts = [read_json(path / 'receipt.json') for path in paths]
    verify_identities(receipts, expected)
    features = []
    for frame in (10, 19):
        actual = []
        for path, receipt in zip(paths, receipts, strict=True):
            entries = [x for x in receipt['features'] if x['frame'] == frame]
            feature = path / f'features_{frame:06d}.pt'
            if len(entries) != 1 or digest(feature) != entries[0]['sha256']:
                raise ValueError('Saved feature missing or changed')
            actual.append(torch.load(feature, map_location='cpu', weights_only=True))
        early = torch.load(reference / 'features' / f'{frame:06d}.pt', map_location='cpu', weights_only=True)['feature'].unsqueeze(0)
        snapshot = torch.load(reference / 'snapshots' / f'{frame:06d}.pt', map_location='cpu', weights_only=True)
        for name, cpu_tensor in [('early', early), ('deep', snapshot['decoder_inputs']['backbone_features']),
                                 ('low', snapshot['decoder_outputs'][3])]:
            if any(x[name].shape != cpu_tensor.shape or not torch.isfinite(x[name]).all() for x in actual):
                raise ValueError('Saved feature geometry or values differ')
            difference = (actual[0][name] - cpu_tensor).abs()
            features.append({'frame': frame, 'kind': name, 'cpu_mps_mean_abs': float(difference.mean()),
                             'cpu_mps_max_abs': float(difference.max()),
                             'mps_repeat_equal': torch.equal(actual[0][name], actual[1][name])})
    old = read_json(reference / 'records.json')
    old_events = read_json(reference / 'token_window_events.json')
    rows = []
    for frame in range(expected['prompt_frame'], expected['end_frame'] + 1):
        images = []
        for path, receipt in zip(paths, receipts, strict=True):
            entries = [x for x in receipt['rows'] if x['frame'] == frame]
            image = path / f'{frame:06d}.png'
            if len(entries) != 1 or digest(image) != entries[0]['mask_sha256']:
                raise ValueError('Saved mask missing or changed')
            images.append(np.asarray(Image.open(image)) > 0)
        archive = np.asarray(Image.open(reference / f'{frame:06d}.png')) > 0
        if digest(reference / f'{frame:06d}.png') != old['mask_sha256'][frame]:
            raise ValueError('Archived CPU mask changed')
        if any(x.shape != archive.shape for x in images):
            raise ValueError('Mask geometry differs')
        ious = []
        for mask in images:
            union = int((mask | archive).sum())
            ious.append(int((mask & archive).sum()) / union if union else 1.0)
        rows.append({'frame': frame, 'cpu_mps_a_iou': ious[0], 'cpu_mps_b_iou': ious[1],
                     'cpu_mps_a_pixel_difference': int((images[0] != archive).sum()),
                     'mps_repeat_pixel_difference': int((images[0] != images[1]).sum())})
    budget = []
    for receipt in receipts:
        writes = [x for x in receipt['token_window_events'] if x['stage'] == 'after_write']
        if len(writes) != len(rows) - 1 or any(x['total_tokens'] > 16384 or x['insurance_tokens'] > 3072 for x in writes):
            raise ValueError('Memory budget or write coverage differs')
        budget.append({'writes': len(writes), 'max_total_tokens': max(x['total_tokens'] for x in writes),
                       'cpu_events_equal': receipt['token_window_events'] == old_events,
                       'evictions': sum(bool(x['evict_frames']) for x in receipt['token_window_events'] if x['stage'] == 'admission_plan')})
    result = {'passed': True, 'scope': 'Saved seq4 full-video engineering replay, not a learned-refiner or efficacy evaluation.',
              'runtime_recording_limit': 'Seed, fast-math flag, hardware and full active model state were not recorded in source receipts; they are not independently certified here.',
              'mps_repeat_selection_equal': receipts[0]['selection'] == receipts[1]['selection'],
              'mps_repeat_positions_equal': receipts[0]['positions'] == receipts[1]['positions'],
              'mps_repeat_events_equal': receipts[0]['token_window_events'] == receipts[1]['token_window_events'],
              'identities': expected, 'budget': budget, 'rows': rows, 'features': features,
              'mps_input_receipt_sha256': [digest(path / 'receipt.json') for path in paths],
              'auditor_sha256': digest(Path(__file__))}
    out = validate_output(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'frames': len(rows), 'minimum_iou': min(x['cpu_mps_a_iou'] for x in rows),
                      'cpu_mps_differing_pixels': sum(x['cpu_mps_a_pixel_difference'] for x in rows),
                      'mps_repeat_differing_pixels': sum(x['mps_repeat_pixel_difference'] for x in rows),
                      'budget': budget}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'preflight', 'checkpoint', 'mps-a', 'mps-b', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    audit(args.reference, args.preflight, args.checkpoint, args.mps_a, args.mps_b, args.output)
