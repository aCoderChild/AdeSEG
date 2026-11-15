#!/usr/bin/env python3
"""Describe audited automatic Lean failure cases from saved predictions."""
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
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--merged', type=Path, required=True)
    p.add_argument('--sequence', action='append', required=True)
    p.add_argument('--output', type=Path, help='Output JSON on the approved experiment drive')
    a = p.parse_args()
    merged = validate_output(a.merged)
    completion = json.loads((merged / 'completion.json').read_text())
    if not completion['passed'] or digest(merged / 'results.json') != completion['results_sha256']:
        raise ValueError('Merged results changed')
    result = json.loads((merged / 'results.json').read_text())
    locations = json.loads((merged / 'locations.json').read_text())
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    summary = {}
    for name in a.sequence:
        run = Path(locations[name]); folder = run / name
        receipt = json.loads((run / 'completion.json').read_text())
        audit = json.loads((run / 'independent_audit.json').read_text())
        if not audit['passed'] or digest(run / 'records.json') != receipt['records_sha256'] or audit['records_sha256'] != receipt['records_sha256']:
            raise ValueError(f'{name} audit changed')
        record = json.loads((run / 'records.json').read_text())[0]
        if digest(folder / 'frames.json') != record['frames_sha256']:
            raise ValueError(f'{name} frame receipt changed')
        rows = json.loads((folder / 'frames.json').read_text())
        frames = videos[name]
        if len(rows) != len(frames):
            raise ValueError('Frame coverage differs')
        tp = fp = fn = missed = 0
        for i, (row, frame) in enumerate(zip(rows, frames)):
            path = folder / f'{i:06d}.png'
            if digest(path) != row['mask_sha256'] or digest(frame.mask_path) != row['label_sha256']:
                raise ValueError(f'{name}/{i} mask changed')
            pred = np.asarray(Image.open(path)) > 0
            truth = load_label_mask(frame.mask_path, [1]) == 1
            if truth.any():
                tp += int(np.count_nonzero(pred & truth))
                fp += int(np.count_nonzero(pred & ~truth))
                fn += int(np.count_nonzero(~pred & truth))
                missed += not pred.any()
        r = result['sequences'][name]
        summary[name] = {'first_detection_index': r['first_detection_index'],
            'first_detection_labeled_present': r['first_detection_labeled_present'],
            'present_frames_before_detection': r['present_before_first_detection'],
            'present_frames_with_empty_prediction': int(missed), 'present_frames': r['present'],
            'present_frame_dice': r['all_present_dice'], 'empty_false_alarms': r['all_false_alarms'],
            'empty_frames': r['empty'], 'present_pixel_true_positive': tp, 'present_pixel_false_positive': fp,
            'present_pixel_false_negative': fn, 'present_pixel_precision': tp / (tp + fp) if tp + fp else None,
            'present_pixel_recall': tp / (tp + fn) if tp + fn else None}
    target = validate_output(a.output) if a.output else merged / 'case_analysis.json'
    if target.exists():
        raise ValueError('Case analysis already exists')
    target.write_text(json.dumps({'scope': 'Descriptive pooled-pixel counts on labeled-present frames, plus labeled-empty frame false alarms; no causal attribution.', 'cases': summary}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
