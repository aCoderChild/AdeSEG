#!/usr/bin/env python3
"""Rescore a completed SASVi/CMSA adaptation and its overseer masks."""
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
from scripts.memory_token_io import digest  # noqa: E402


def score(pred, truth):
    denom = int(pred.sum() + truth.sum())
    return 2 * int(np.count_nonzero(pred & truth)) / denom if denom else 1.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--cmsa-run', required=True, type=Path)
    args = p.parse_args()
    if (args.run / 'superseded_invalid.json').exists():
        raise ValueError('Run was marked invalid; inspect its superseded_invalid.json')
    setup = json.loads((args.run / 'setup.json').read_text())
    done = json.loads((args.run / 'completion.json').read_text())
    if not done['passed'] or digest(args.run / 'records.json') != done['records_sha256']:
        raise ValueError('Incomplete or changed run')
    if digest(args.cmsa_run / 'setup.json') != setup['cmsa_setup_sha256'] or digest(args.cmsa_run / 'frames.json') != setup['cmsa_frames_sha256']:
        raise ValueError('CMSA provenance mismatch')
    manifest = ROOT / 'data/polypgen_sequence.jsonl'
    if digest(manifest) != setup['manifest_sha256']:
        raise ValueError('Manifest mismatch')
    data = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
    name = setup['sequence']
    frames = data[name][:setup['frames']]
    rows = json.loads((args.run / 'records.json').read_text())
    if len(rows) != len(frames) or [r['index'] for r in rows] != list(range(len(frames))):
        raise ValueError('Frame accounting mismatch')
    expected = {(r['sequence'], r['frame']): r for r in json.loads((args.cmsa_run / 'frames.json').read_text())}
    details = []
    for i, (row, frame) in enumerate(zip(rows, frames)):
        ref = frame.image_path.resolve().relative_to((ROOT / 'data').resolve()).with_suffix('.png')
        cmsa_path = args.cmsa_run / 'predictions' / ref
        pred_path = args.run / 'predictions' / name / f'{i:06d}.png'
        if digest(frame.image_path) != row['image_sha256'] or digest(pred_path) != row['mask_sha256'] or digest(cmsa_path) != row['cmsa_mask_sha256']:
            raise ValueError('Image or mask changed')
        truth = load_label_mask(frame.mask_path, [1]) == 1
        before = np.asarray(Image.open(cmsa_path).convert('L')) >= 128
        after = np.asarray(Image.open(pred_path)) > 0
        if before.shape != truth.shape or after.shape != truth.shape:
            raise ValueError('Shape mismatch')
        if bool(truth.any()) != row['present'] or not np.isclose(score(after, truth), row['dice'], atol=1e-12, rtol=0):
            raise ValueError('SASVi score mismatch')
        if not np.isclose(score(before, truth), expected[(name, i)]['dice'], atol=1e-12, rtol=0):
            raise ValueError('CMSA score mismatch')
        details.append({'index': i, 'present': bool(truth.any()), 'cmsa_dice': score(before, truth),
                        'sasvi_dice': score(after, truth), 'cmsa_positive': bool(before.any()), 'sasvi_positive': bool(after.any())})
    prompt = next(i for i, r in enumerate(details) if r['present'])
    selected = details[prompt + 1:]
    positive = [r for r in selected if r['present']]
    empty = [r for r in selected if not r['present']]
    report = {'sequence': name, 'prompt_index_for_evaluation_only': prompt, 'targets': len(selected), 'present': len(positive), 'empty': len(empty),
              'cmsa_present_dice': float(np.mean([r['cmsa_dice'] for r in positive])) if positive else None,
              'sasvi_present_dice': float(np.mean([r['sasvi_dice'] for r in positive])) if positive else None,
              'cmsa_false_alarms': sum(r['cmsa_positive'] for r in empty),
              'sasvi_false_alarms': sum(r['sasvi_positive'] for r in empty), 'frames': details}
    (args.run / 'audit_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
