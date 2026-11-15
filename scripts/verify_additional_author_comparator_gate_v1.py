#!/usr/bin/env python3
"""Verify CPU and two fresh MPS short clips without opening annotations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v1 import DATA, identity, new_output
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def audit(cpu, mps_a, mps_b, checkpoint, detector, extended_preflight, remaining_preflight):
    paths = [Path(cpu).resolve(), Path(mps_a).resolve(), Path(mps_b).resolve()]
    if len(set(paths)) != 3:
        raise ValueError('Replay directories must be distinct')
    runs = [json.loads((path / 'receipt.json').read_text()) for path in paths]
    first = runs[0]
    if len({run.get('process_id') for run in runs}) != 3 or len({run.get('run_nonce') for run in runs}) != 3:
        raise ValueError('Short runs are not distinct processes')
    expected = [identity(first['method'], first['sequence'], checkpoint, detector,
                         extended_preflight, remaining_preflight, device) for device in ('cpu', 'mps', 'mps')]
    shared = ('method', 'sequence', 'prompt_frame', 'prompt', 'state_sha256', 'noncausal')
    for path, run, want in zip(paths, runs, expected, strict=True):
        if (not run.get('passed') or run.get('full_video') is not False or not run.get('deterministic')
                or run.get('identity') != want or run.get('label_guard', {}).get('open_attempts') != 0
                or any(run.get(key) != first.get(key) for key in shared)):
            raise ValueError('Short replay identity or label guard differs')
        records = run.get('records', [])
        if (len(records) < run['prompt_frame'] + 5 or
                [r['frame'] for r in records] != list(range(len(records))) or
                any(digest(path / f"{r['frame']:06d}.png") != r['mask_sha256'] for r in records)):
            raise ValueError('Short replay output coverage or hashes differ')
    for run in runs[1:]:
        if len(run['records']) != len(first['records']):
            raise ValueError('Short replay lengths differ')
        for a, b in zip(first['records'], run['records'], strict=True):
            if any(a[k] != b[k] for k in ('frame', 'source_frame_index', 'image_sha256', 'post_prompt')):
                raise ValueError('Short replay input frames differ')
    rows = []
    for item in first['records']:
        frame = item['frame']
        masks = []
        for path in paths:
            with Image.open(path / f'{frame:06d}.png') as image:
                masks.append(np.asarray(image.convert('L')) > 0)
        if len({mask.shape for mask in masks}) != 1:
            raise ValueError('Short replay output geometry differs')
        repeat = int((masks[1] != masks[2]).sum())
        if repeat > max(1, int(masks[1].size * 0.0001)):
            raise ValueError('MPS repeat differs by more than 0.01% of image area')
        union = int((masks[0] | masks[1]).sum())
        rows.append({'frame': frame, 'cpu_mps_pixels': int((masks[0] != masks[1]).sum()),
                     'mps_repeat_pixels': repeat,
                     'cpu_mps_iou': int((masks[0] & masks[1]).sum()) / union if union else 1.0})
    return {'passed': True, 'scope': 'Short label-free engineering check, not video efficacy.',
            'identity': runs[1]['identity'], 'prompt': first['prompt'], 'state_sha256': first['state_sha256'],
            'repeat_pixel_fraction_limit': 0.0001, 'rows': rows,
            'receipt_paths': [{'path': str(path / 'receipt.json'), 'sha256': digest(path / 'receipt.json')}
                              for path in paths], 'verifier_sha256': digest(Path(__file__))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cpu', 'mps-a', 'mps-b', 'checkpoint', 'detector', 'extended-preflight', 'remaining-preflight', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    guard = install_mask_open_guard(DATA)
    result = audit(a.cpu, a.mps_a, a.mps_b, a.checkpoint, a.detector, a.extended_preflight, a.remaining_preflight)
    if guard['open_attempts']:
        raise ValueError('Annotation mask was opened')
    result['label_guard'] = dict(guard)
    out = new_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'gate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'method': result['identity']['method'],
                      'sequence': result['identity']['sequence'], 'frames': len(result['rows'])}))


if __name__ == '__main__':
    main()
