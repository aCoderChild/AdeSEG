#!/usr/bin/env python3
"""Rescore CMSA masks and compare exactly matched archived native SAM2 frames."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from scripts.run_cmsa_polypgen import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True, help='native_results.json')
    args = parser.parse_args()
    setup = json.loads((args.run / 'setup.json').read_text())
    rows = json.loads((args.run / 'frames.json').read_text())
    result = json.loads((args.run / 'results.json').read_text())
    manifest = ROOT / 'data/polypgen_sequence.jsonl'
    if setup['manifest_sha256'] != sha(manifest):
        raise RuntimeError('Manifest changed since inference')
    videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
    native = json.loads(args.native.read_text())
    data_root = (ROOT / 'data').resolve()
    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row['sequence'], []).append(row)
    if set(by_sequence) != set(setup['sequences']):
        raise RuntimeError('CMSA sequence coverage differs')
    differences = []
    table = {}
    for name in setup['sequences']:
        frames = videos[name]
        cmsa = by_sequence[name]
        if [r['frame'] for r in cmsa] != list(range(len(frames))):
            raise RuntimeError(f'CMSA coverage mismatch: {name}')
        evidence = native['frame_evidence'][name]
        native_path = Path(evidence['path'])
        if sha(native_path) != evidence['sha256']:
            raise RuntimeError(f'Native evidence digest mismatch: {name}')
        reference = json.loads(native_path.read_text())
        prompt = result['per_sequence'][name]['prompt_index']
        if [r['frame'] for r in reference] != list(range(prompt + 1, len(frames))):
            raise RuntimeError(f'Frame matching mismatch: {name}')
        for row, frame in zip(cmsa, frames):
            image = frame.image_path.resolve().relative_to(data_root).with_suffix('.png')
            pred_path = args.run / 'predictions' / image
            if sha(pred_path) != row['prediction_sha256']:
                raise RuntimeError(f'Prediction hash mismatch: {pred_path}')
            mask = np.asarray(Image.open(pred_path).convert('L')) >= 128
            gt = load_label_mask(frame.mask_path, [1]) == 1
            scores = calculate_scores(mask, gt)
            if scores['dice'] != row['dice'] or scores['iou'] != row['iou'] or bool(gt.any()) != row['present']:
                raise RuntimeError(f'Rescoring mismatch: {name} frame {row["frame"]}')
        targets = cmsa[prompt + 1:]
        if any(a['present'] != b['present'] for a, b in zip(targets, reference)):
            raise RuntimeError(f'Presence mismatch: {name}')
        present = [r['dice'] for r in targets if r['present']]
        avg = float(np.mean(present))
        native_avg = float(np.mean([r['scores']['dice'] for r in reference if r['present']]))
        if not np.isclose(avg, result['per_sequence'][name]['mean_present_dice'], rtol=0, atol=1e-14):
            raise RuntimeError(f'CMSA summary mismatch: {name}')
        if not np.isclose(native_avg, native['sequences'][name]['present_dice'], rtol=0, atol=1e-14):
            raise RuntimeError(f'Native summary mismatch: {name}')
        differences.append(avg - native_avg)
        empty = [r for r in targets if not r['present']]
        native_empty = [r for r in reference if not r['present']]
        table[name] = {'targets': len(targets), 'present': len(present), 'cmsa_dice': avg, 'native_dice': native_avg, 'difference': avg - native_avg,
                       'empty_targets': len(empty), 'cmsa_false_alarms': sum(r['dice'] == 0 for r in empty),
                       'native_false_alarms': sum(bool(r['absent_fp']) for r in native_empty)}
    report = {'status': 'passed', 'cmsa_prediction_rows_rescored': len(rows),
              'native_results_sha256': sha(args.native), 'matched_targets': sum(x['targets'] for x in table.values()),
              'cmsa_equal_sequence_present_dice': float(np.mean([r['cmsa_dice'] for r in table.values()])),
              'native_equal_sequence_present_dice': float(np.mean([r['native_dice'] for r in table.values()])),
              'mean_difference': float(np.mean(differences)),
              'empty_targets': sum(r['empty_targets'] for r in table.values()),
              'cmsa_false_alarms': sum(r['cmsa_false_alarms'] for r in table.values()),
              'native_false_alarms': sum(r['native_false_alarms'] for r in table.values()), 'per_sequence': table,
              'limitations': 'CMSA is unprompted; SAM2 receives a GT mask prompt. Frame 0 may use frame 1 as an anchor and is excluded from the matched targets. Different training and source domains; cohort is exploratory.'}
    destination = args.run / 'matched_native_audit.json'
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
