#!/usr/bin/env python3
"""Rescore saved SAMRefiner masks and compare them with their SAM2 inputs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from evaluation.statistics import sequence_bootstrap_interval  # noqa: E402
from scripts.memory_token_io import digest  # noqa: E402


def score(pred, truth):
    denom = int(pred.sum() + truth.sum())
    return 2 * int(np.count_nonzero(pred & truth)) / denom if denom else 1.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--input-run', required=True, type=Path)
    p.add_argument('--gate', required=True, type=Path)
    a = p.parse_args()
    setup = json.loads((a.run / 'setup.json').read_text())
    done = json.loads((a.run / 'completion.json').read_text())
    if not done['passed'] or digest(a.run / 'records.json') != done['records_sha256']:
        raise ValueError('Refiner run incomplete or changed')
    if digest(a.input_run / 'records.json') != setup['input_records_sha256']:
        raise ValueError('Input run changed')
    rows = json.loads((a.run / 'records.json').read_text())
    original = json.loads((a.input_run / 'records.json').read_text())
    if [(r['sequence'], r['index']) for r in rows] != [(r['sequence'], r['index']) for r in original]:
        raise ValueError('Frame plan differs')
    gate = json.loads((a.gate / 'records.json').read_text())
    if gate != [r for r in rows if r['sequence'] == 'seq11']:
        raise ValueError('Seq11 gate does not reproduce')
    manifest = ROOT / 'data/polypgen_sequence.jsonl'
    if digest(manifest) != setup['manifest_sha256']:
        raise ValueError('Manifest changed')
    data = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
    per = {}
    alarms = [0, 0]
    frame_details = []
    for row, old in zip(rows, original):
        name, idx = row['sequence'], row['index']
        frame = data[name][idx]
        if digest(frame.image_path) != old['image_sha256'] or digest(frame.mask_path) != old['target_mask_sha256']:
            raise ValueError('Dataset inputs changed')
        gt = load_label_mask(frame.mask_path, [1]) == 1
        src = a.input_run / name / f'{idx:06d}_yolo_box.png'
        dst = a.run / name / f'{idx:06d}.png'
        if digest(src) != row['input_sha256'] or digest(dst) != row['mask_sha256']:
            raise ValueError('Mask changed')
        before = np.asarray(Image.open(src)) > 0
        after = np.asarray(Image.open(dst)) > 0
        if gt.shape != before.shape or gt.shape != after.shape:
            raise ValueError('Dimension mismatch')
        if row['present'] != bool(gt.any()) or not np.isclose(score(after, gt), row['dice'], atol=1e-12, rtol=0):
            raise ValueError('Score mismatch')
        if not np.isclose(score(before, gt), old['conditions']['yolo_box']['dice'], atol=1e-12, rtol=0):
            raise ValueError('Input score mismatch')
        if row['empty_input_fallback'] != (not before.any()) or (not before.any() and after.any()):
            raise ValueError('Empty-input fallback mismatch')
        def counts(mask):
            return {'tp': int(np.count_nonzero(mask & gt)), 'fp': int(np.count_nonzero(mask & ~gt)),
                    'fn': int(np.count_nonzero(~mask & gt))}
        frame_details.append({'sequence': name, 'index': idx, 'present': bool(gt.any()),
                              'baseline': counts(before), 'refined': counts(after)})
        if gt.any():
            per.setdefault(name, [[], []])
            per[name][0].append(score(before, gt))
            per[name][1].append(score(after, gt))
        else:
            alarms[0] += bool(before.any())
            alarms[1] += bool(after.any())
    means = {n: [float(np.mean(v[0])), float(np.mean(v[1]))] for n, v in per.items()}
    differences = {n: x[1] - x[0] for n, x in means.items()}
    result = {'frames': len(rows), 'present': sum(r['present'] for r in rows), 'empty': sum(not r['present'] for r in rows),
              'empty_input_fallbacks': sum(r['empty_input_fallback'] for r in rows),
              'base_equal_sequence_dice': float(np.mean([v[0] for v in means.values()])),
              'refined_equal_sequence_dice': float(np.mean([v[1] for v in means.values()])),
              'false_alarms_before_after': alarms, 'paired_difference': float(np.mean(list(differences.values()))),
              'sequence_bootstrap_interval': sequence_bootstrap_interval(differences), 'per_sequence': means,
              'seq10_excluded_difference': float(np.mean([v for k, v in differences.items() if k != 'seq10']))}
    (a.run / 'audit_results.json').write_text(json.dumps(result, indent=2) + '\n')
    (a.run / 'audit_frame_details.json').write_text(json.dumps(frame_details, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
