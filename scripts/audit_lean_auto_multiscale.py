#!/usr/bin/env python3
"""Independently rescore the automatic Lean video outputs and prompt receipts."""
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

MANIFEST = ROOT / 'data/polypgen_sequence.jsonl'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--historical', required=True, type=Path, help='Prior Lean full-comparison parent')
    args = p.parse_args()
    setup = json.loads((args.run / 'setup.json').read_text())
    done = json.loads((args.run / 'completion.json').read_text())
    if not done['passed'] or digest(args.run / 'records.json') != done['records_sha256']:
        raise ValueError('Incomplete or changed experiment')
    if setup['signature']['manifest_sha256'] != digest(MANIFEST):
        raise ValueError('Manifest changed')
    old_audit = json.loads((args.historical / 'independent_mask_audit.json').read_text())
    if not old_audit['passed'] or old_audit['comparison_sha256'] != digest(args.historical / 'comparison.json'):
        raise ValueError('Historical baseline audit is invalid')
    old = json.loads((args.historical / 'comparison.json').read_text())
    old_by_name = {r['sequence']: r for r in old['sequences']}
    videos = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    records = json.loads((args.run / 'records.json').read_text())
    if [r['sequence'] for r in records] != setup['sequences']:
        raise ValueError('Sequence coverage mismatch')
    summaries = {}
    for record in records:
        name = record['sequence']
        folder = args.run / name
        for key, file in [('prompt_sha256', 'prompt.json'), ('frames_sha256', 'frames.json'),
                          ('events_sha256', 'memory_events.json'), ('selections_sha256', 'selections.json')]:
            if digest(folder / file) != record[key]:
                raise ValueError(f'Changed receipt: {name}/{file}')
        prompt = json.loads((folder / 'prompt.json').read_text())
        rows = json.loads((folder / 'frames.json').read_text())
        frames = videos[name]
        if len(rows) != len(frames) or record['frames'] != len(frames):
            raise ValueError('Incomplete frame coverage')
        first = prompt['first_detection_index']
        if first != record['first_detection_index']:
            raise ValueError('Detection index mismatch')
        if first is not None:
            if not 0 <= first < len(frames) or prompt['box'] is None:
                raise ValueError('Invalid automatic detection')
            with Image.open(frames[first].image_path) as image:
                w, h = image.size
            x0, y0, x1, y1 = prompt['box']
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError('Invalid prompt coordinates')
            for typ in ('provisional', 'refined'):
                path = folder / f'prompt_{typ}.png'
                if digest(path) != prompt[f'{typ}_mask_sha256'] or np.asarray(Image.open(path)).shape != (h, w):
                    raise ValueError('Prompt mask changed or has wrong shape')
            points = prompt['points']
            if prompt['point_status'] == 'available':
                if points is None or len(points) != 2 or points[0] == points[1] or any(not (x0 <= x < x1 and y0 <= y < y1) for x, y in points):
                    raise ValueError('Invalid earlier-stage points')
            elif points is not None:
                raise ValueError('Points should be absent in fallback')
            integrity = json.loads((folder / 'parameter_integrity_after.json').read_text())
            if not integrity['unchanged']:
                raise ValueError('Lean weights changed')
        elif prompt['box'] is not None or prompt['points'] is not None:
            raise ValueError('A prompt exists without a detection')
        selections = json.loads((folder / 'selections.json').read_text())
        events = json.loads((folder / 'memory_events.json').read_text())
        if len(events) != record['memory_events'] or len(selections) != record['selection_events']:
            raise ValueError('Memory event count differs')
        for event in selections:
            if event['kept'] != max(1, int(event['candidates'] * .25)):
                raise ValueError('Lean token budget differs')
        all_present, all_empty, values = [], [], {}
        for i, (row, frame) in enumerate(zip(rows, frames)):
            path = folder / f'{i:06d}.png'
            if (row['index'] != i or row['source_frame_index'] != frame.frame_index or digest(path) != row['mask_sha256']
                or digest(frame.image_path) != row['image_sha256'] or digest(frame.mask_path) != row['label_sha256']):
                raise ValueError(f'Frame provenance mismatch: {name}/{i}')
            pred = np.asarray(Image.open(path)) > 0
            truth = load_label_mask(frame.mask_path, [1]) == 1
            if pred.shape != truth.shape or row['present'] != bool(truth.any()):
                raise ValueError('Mask dimensions or labels differ')
            if (first is None or i < first) and pred.any():
                raise ValueError('Foreground before first detection')
            denom = int(pred.sum() + truth.sum())
            dice = 2 * int(np.count_nonzero(pred & truth)) / denom if denom else 1.
            if not np.isclose(dice, row['dice'], rtol=0, atol=1e-12):
                raise ValueError('Dice mismatch')
            if not truth.any() and row['false_alarm'] != bool(pred.any()):
                raise ValueError('False-alarm mismatch')
            (all_present if truth.any() else all_empty).append(dice if truth.any() else bool(pred.any()))
            values[i] = (dice, bool(truth.any()), bool(pred.any()))
        baseline_inputs = json.loads((args.historical / 'full' / name / 'inputs.json').read_text())
        if len(baseline_inputs) != len(rows):
            raise ValueError('Historical input length mismatch')
        for row, old_input in zip(rows, baseline_inputs):
            if (row['source_frame_index'], row['image_sha256'], row['label_sha256']) != (old_input['frame'], old_input['image_sha256'], old_input['mask_sha256']):
                raise ValueError('Historical inputs differ')
        historical_rows = [json.loads(line) for line in (args.historical / 'full' / name / 'frames.jsonl').read_text().splitlines()]
        if len(historical_rows) != old_by_name[name]['targets']:
            raise ValueError('Historical target count mismatch')
        selected = [values[r['frame']] for r in historical_rows]
        if any(x[1] != r['present'] for x, r in zip(selected, historical_rows)):
            raise ValueError('Historical target labels differ')
        historical_present = [v[0] for v in selected if v[1]]
        historical_empty = [v[2] for v in selected if not v[1]]
        summaries[name] = {'frames': len(rows), 'present': len(all_present), 'empty': len(all_empty),
            'all_present_dice': float(np.mean(all_present)) if all_present else None,
            'all_false_alarms': sum(all_empty), 'first_detection_index': first,
            'first_detection_labeled_present': None if first is None else values[first][1],
            'present_before_first_detection': sum(values[i][1] for i in range(first if first is not None else len(rows))),
            'point_status': prompt['point_status'], 'lean_mask_status': prompt['lean_mask_status'],
            'memory_events': len(events), 'selection_events': len(selections),
            'historical_present_dice': float(np.mean(historical_present)),
            'historical_false_alarms': sum(historical_empty),
            'historical_present': len(historical_present), 'historical_empty': len(historical_empty)}
    result = {'passed': True, 'scope': 'single automatic combined system; historical baselines use GT initialization and are not matched prompts',
              'records_sha256': done['records_sha256'], 'sequences': summaries,
              'all_present_dice_equal_sequence': float(np.mean([x['all_present_dice'] for x in summaries.values() if x['all_present_dice'] is not None])),
              'all_present_frames': sum(x['present'] for x in summaries.values()),
              'all_empty_frames': sum(x['empty'] for x in summaries.values()),
              'all_false_alarms': sum(x['all_false_alarms'] for x in summaries.values()),
              'historical_present_dice_equal_sequence': float(np.mean([x['historical_present_dice'] for x in summaries.values()])),
              'historical_present_frames': sum(x['historical_present'] for x in summaries.values()),
              'historical_empty_frames': sum(x['historical_empty'] for x in summaries.values()),
              'historical_false_alarms': sum(x['historical_false_alarms'] for x in summaries.values())}
    (args.run / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sequences'}, indent=2))


if __name__ == '__main__':
    main()
