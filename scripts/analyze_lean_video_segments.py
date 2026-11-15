#!/usr/bin/env python3
"""Analyze saved automatic Lean errors across labeled presence intervals."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merged', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', action='append', required=True)
    args = parser.parse_args()
    completion = read(args.merged / 'completion.json')
    if not completion['passed'] or digest(args.merged / 'results.json') != completion['results_sha256']:
        raise ValueError('Merged reference changed')
    locations = read(args.merged / 'locations.json')
    if len(set(args.sequence)) != len(args.sequence) or not set(args.sequence).issubset(locations):
        raise ValueError('Invalid or duplicate sequence')
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    cases = {}
    for name in args.sequence:
        run = Path(locations[name]); folder = run / name
        done = read(run / 'completion.json'); audit = read(run / 'independent_audit.json')
        if (not done['passed'] or not audit['passed'] or digest(run / 'records.json') != done['records_sha256']
            or audit['records_sha256'] != done['records_sha256']):
            raise ValueError(f'{name}: source run incomplete')
        record = read(run / 'records.json')[0]
        if digest(folder / 'frames.json') != record['frames_sha256']:
            raise ValueError(f'{name}: frame metadata changed')
        saved = read(folder / 'frames.json'); frames = videos[name]
        if len(saved) != len(frames):
            raise ValueError(f'{name}: incomplete frames')
        rows = []
        for i, (row, frame) in enumerate(zip(saved, frames)):
            path = folder / f'{i:06d}.png'
            if (row['index'] != i or row['source_frame_index'] != frame.frame_index or
                digest(path) != row['mask_sha256'] or digest(frame.mask_path) != row['label_sha256'] or
                digest(frame.image_path) != row['image_sha256']):
                raise ValueError(f'{name}/{i}: input or output changed')
            pred = np.asarray(Image.open(path)) > 0
            truth = load_label_mask(frame.mask_path, [1]) == 1
            if pred.shape != truth.shape or bool(truth.any()) != row['present']:
                raise ValueError(f'{name}/{i}: label mismatch')
            tp = int(np.count_nonzero(pred & truth)); fp = int(np.count_nonzero(pred & ~truth))
            fn = int(np.count_nonzero(~pred & truth)); denominator = int(pred.sum() + truth.sum())
            dice = 2 * tp / denominator if denominator else 1.
            if not np.isclose(dice, row['dice'], rtol=0, atol=1e-12):
                raise ValueError(f'{name}/{i}: Dice mismatch')
            rows.append({'index': i, 'source_frame_index': frame.frame_index, 'present': row['present'],
                         'dice': dice, 'tp': tp, 'fp': fp, 'fn': fn,
                         'predicted_pixels': int(pred.sum()), 'image_pixels': int(pred.size)})
        segments = []
        for present, group in itertools.groupby(rows, key=lambda row: row['present']):
            members = list(group); tp = sum(r['tp'] for r in members)
            fp = sum(r['fp'] for r in members); fn = sum(r['fn'] for r in members)
            segments.append({'first_index': members[0]['index'], 'last_index': members[-1]['index'],
                             'first_source_frame': members[0]['source_frame_index'],
                             'last_source_frame': members[-1]['source_frame_index'],
                             'present': present, 'frames': len(members),
                             'mean_dice_present': float(np.mean([r['dice'] for r in members])) if present else None,
                             'predicted_present_frames': sum(r['predicted_pixels'] > 0 for r in members),
                             'mean_prediction_area_fraction': float(np.mean([r['predicted_pixels']/r['image_pixels'] for r in members])),
                             'pooled_pixel_precision': tp/(tp+fp) if present and tp+fp else None,
                             'pooled_pixel_recall': tp/(tp+fn) if present and tp+fn else None})
        cases[name] = {'first_detection_index': record['first_detection_index'], 'frames': len(rows),
                       'segments': segments, 'frame_records': rows}
        print(f'{name}: {len(rows)} frames, {len(segments)} presence intervals', flush=True)
    out = validate_output(args.output); out.mkdir(parents=True, exist_ok=False)
    result = {'scope': 'Post-hoc description of previously saved automatic outputs. Presence intervals use evaluation labels, not model inputs; frame indices refer to manifest samples, not elapsed seconds.',
              'reference_results_sha256': completion['results_sha256'], 'cases': cases}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Automatic Lean errors by labeled interval', '', result['scope'], '',
             'Precision and recall pool pixels in each present interval and can be dominated by large masks. On empty intervals, predicted-present frames count any foreground pixel. Empty annotations may be incomplete.', '']
    for name, case in cases.items():
        lines.extend([f'## {name}', '', f"First automatic detection: manifest index {case['first_detection_index']}.", '',
                      '| Manifest indices | Labeled status | Frames | Mean Dice | Foreground output frames | Mean predicted area | Pooled precision | Pooled recall |',
                      '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for x in case['segments']:
            fmt = lambda v: '—' if v is None else f'{v:.3f}'
            lines.append(f"| {x['first_index']}–{x['last_index']} | {'present' if x['present'] else 'empty'} | {x['frames']} | "
                         f"{fmt(x['mean_dice_present'])} | {x['predicted_present_frames']} | {100*x['mean_prediction_area_fraction']:.1f}% | "
                         f"{fmt(x['pooled_pixel_precision'])} | {fmt(x['pooled_pixel_recall'])} |")
        lines.append('')
    (out / 'report.md').write_text('\n'.join(lines))
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'results_sha256': digest(out / 'results.json'),
                                                    'source_sha256': digest(Path(__file__))}, indent=2) + '\n')


if __name__ == '__main__':
    main()
