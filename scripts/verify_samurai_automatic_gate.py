#!/usr/bin/env python3
"""Audit SAMURAI automatic-prompt short replays; never opens annotations.

Two MPS replays and one CPU replay establish the tested configuration and
quantify disagreements. This is not an efficacy or full-video guarantee.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output
from scripts.run_samurai_automatic_priority import CONFIG, MANIFEST, PROTOCOL
from scripts.run_author_baseline import source_identity


def read(path): return json.loads(path.read_text())


def audit(cpu, mps_a, mps_b):
    paths = [Path(cpu), Path(mps_a), Path(mps_b)]
    runs = [read(path / 'receipt.json') for path in paths]
    _, revision = source_identity('samurai')
    required = {'source_revision': revision, 'torch': str(torch.__version__),
        'config_sha256': digest(CONFIG), 'manifest_sha256': digest(MANIFEST),
        'protocol_sha256': digest(PROTOCOL), 'runner_sha256': digest(ROOT / 'scripts/run_samurai_automatic_priority.py'),
        'kalman_adapter_sha256': digest(ROOT / 'modeling/samurai_mps_kalman.py')}
    for index, (path, run) in enumerate(zip(paths, runs, strict=True)):
        if not run.get('passed') or run['full_video'] or run['device'] != ('cpu' if index == 0 else 'mps'):
            raise ValueError('Unexpected short-replay receipt')
        if any(run.get(k) != v for k, v in required.items()): raise ValueError('Recorded runtime differs')
        if index and (run.get('fallback') != '0' or run.get('fast_math') != '0'):
            raise ValueError('MPS execution settings differ')
        records = run['records']
        if len(records) < 4 or len(records) != run['frames'] or [r['local_index'] for r in records] != list(range(len(records))):
            raise ValueError('Short-replay coverage differs')
        if any(digest(path / f"{r['frame']:06d}.png") != r['mask_sha256'] for r in records):
            raise ValueError('Short-replay mask changed')
    shared = ('sequence', 'prompt_frame', 'prompt', 'rebased_start', 'checkpoint_sha256',
              'state_sha256', 'seed', 'dtype', 'deterministic', 'frames')
    if any(any(run.get(k) != runs[0].get(k) for k in shared) for run in runs[1:]):
        raise ValueError('Prompt or actual model identity differs')
    if not runs[0]['deterministic']:
        raise ValueError('Deterministic algorithms must be enabled')
    if runs[0]['sequence']=='seq20' and any(not run['kalman_iou_events'] for run in runs):
        raise ValueError('Seq20 gate must exercise the Kalman stable-scoring branch')
    for run in runs[1:]:
        for a, b in zip(runs[0]['records'], run['records'], strict=True):
            if any(a[k] != b[k] for k in ('local_index', 'frame', 'source_frame_index', 'image_sha256')):
                raise ValueError('Frame ordering or input identity differs')
    rows = []
    for record in runs[0]['records']:
        frame = record['frame']
        masks = [np.asarray(Image.open(path / f'{frame:06d}.png').convert('L')) > 0 for path in paths]
        if len({x.shape for x in masks}) != 1: raise ValueError('Output geometry differs')
        union = int((masks[0] | masks[1]).sum())
        rows.append({'frame': frame, 'cpu_mps_pixels': int((masks[0] != masks[1]).sum()),
                     'mps_repeat_pixels': int((masks[1] != masks[2]).sum()),
                     'cpu_mps_iou': int((masks[0] & masks[1]).sum()) / union if union else 1.0})
    return {'passed': True, 'scope': 'Short label-free prompt/ROI engineering check; no efficacy, full-video, or general equivalence claim.',
            **{k: runs[0][k] for k in shared}, **required, 'device': 'mps',
            'rows': rows, 'kalman_iou_calls': [len(run['kalman_iou_events']) for run in runs],
            'kalman_max_rounding_error': [max((event['max_rounding_error'] for event in run['kalman_iou_events']),default=0.) for run in runs],
            'receipt_sha256': [digest(path / 'receipt.json') for path in paths],
            'verifier_sha256': digest(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu', 'mps-a', 'mps-b', 'output'): p.add_argument('--' + name, type=Path, required=True)
    a=p.parse_args(); result=audit(a.cpu, a.mps_a, a.mps_b)
    out=validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    (out / 'gate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'sequence': result['sequence'], 'frames': result['frames'],
                      'cpu_mps_pixels': sum(x['cpu_mps_pixels'] for x in result['rows']),
                      'mps_repeat_pixels': sum(x['mps_repeat_pixels'] for x in result['rows'])}))


if __name__ == '__main__': main()
