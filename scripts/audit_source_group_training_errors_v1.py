#!/usr/bin/env python3
"""Audit only frozen training samples for baseline foreground coverage.

This post-hoc descriptive audit is for planning a training-only loss.  It does
not open validation or protected test masks and must not select samples from
them.  The source-group checkpoint and observer receipts remain unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from scripts.memory_token_io import digest, validate_output
from scripts.source_group_boundary_shared_v1 import load_prepared, read
from scripts.source_group_training_label_guard_v1 import install_training_mask_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    guard = install_training_mask_guard(ROOT/'data/polypgen_sequence.jsonl')
    manifest, manifest_sha = load_prepared(args.manifest)
    rows = []
    for item in manifest['samples']:
        if item['split'] != 'train':
            continue
        seq, frame = item['sequence'], item['frame']
        receipt = read(manifest['sequences'][seq]['capture_receipt_path'])
        entries = [r for r in receipt['rows'] if r['frame'] == frame]
        if len(entries) != 1:
            raise ValueError('Capture receipt frame is missing or duplicated')
        prediction_path = Path(manifest['sequences'][seq]['capture_receipt_path']).parent/f'{frame:06d}.png'
        if digest(prediction_path) != entries[0]['mask_sha256'] or digest(item['label']) != item['label_sha256']:
            raise ValueError('Frozen prediction or training label changed')
        with Image.open(prediction_path) as image:
            pred = np.asarray(image.convert('L')) > 0
        target = load_label_mask(Path(item['label']), [1]) == 1
        if pred.shape != target.shape:
            raise ValueError('Prediction/label geometry differs')
        pp, gt = int(pred.sum()), int(target.sum())
        intersection = int((pred & target).sum())
        rows.append({'sequence': seq, 'frame': frame, 'prediction_sha256': entries[0]['mask_sha256'],
                     'label_sha256': item['label_sha256'], 'prediction_pixels': pp, 'target_pixels': gt,
                     'intersection_pixels': intersection, 'recall': intersection/gt if gt else None,
                     'precision': intersection/pp if pp else None,
                     'dice': 2*intersection/(pp+gt) if pp+gt else 1.0})
    if len(rows) != 164 or guard['blocked_open_attempts']:
        raise ValueError('Training sample coverage or label guard differs')
    present = [x for x in rows if x['target_pixels']]
    empty = [x for x in rows if not x['target_pixels']]
    def count(predicate):
        return sum(bool(predicate(x)) for x in present)
    summary = {'training_reads': len(rows), 'present': len(present), 'labeled_empty': len(empty),
               'labeled_empty_false_alarms': sum(x['prediction_pixels'] > 0 for x in empty),
               'present_zero_overlap': count(lambda x: x['intersection_pixels'] == 0),
               'present_recall_gte_0_8_precision_lt_0_2': count(lambda x: x['recall'] >= .8 and x['precision'] is not None and x['precision'] < .2),
               'present_recall_gte_0_8_precision_lt_0_5': count(lambda x: x['recall'] >= .8 and x['precision'] is not None and x['precision'] < .5),
               'present_recall_gte_0_8': count(lambda x: x['recall'] >= .8),
               'per_sequence': {s: sum(x['sequence'] == s for x in rows) for s in sorted({x['sequence'] for x in rows})}}
    out = validate_output(args.output); out.mkdir(parents=True, exist_ok=False)
    result = {'passed': True, 'scope': 'Retrospective training-only fixed-read coverage; not validation or independent evidence.',
              'manifest_sha256': manifest_sha, 'source_sha256': digest(Path(__file__)), 'label_guard': dict(guard),
              'definitions': 'Thresholds describe coverage only and do not affect sample selection or inference.',
              'summary': summary, 'rows': rows}
    (out/'training_error_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
