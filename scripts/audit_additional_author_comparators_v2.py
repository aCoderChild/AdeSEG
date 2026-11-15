#!/usr/bin/env python3
"""Independently rescore completed author-comparator predictions on CPU.

Uses independent PIL/NumPy/SciPy metric implementations. Structural gate
validation intentionally reuses the reviewed scorer's artifact verifier; this
audit is not an independent patient-level or clinical evaluation.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_boundary_video_factorial_v1 import label, metrics, summarize, paired, same
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v2 import DATA, METHODS, input_data, new_output
from scripts.run_additional_author_comparator_study_v2 import CHECKPOINT, DETECTOR, EXTENDED, REMAINING
from scripts.score_additional_author_comparators_v2 import install_phase_mask_guard, validate_frozen


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def compare(errors, prefix, actual, expected):
    if set(actual) != set(expected):
        errors.append([*prefix, 'keys', sorted(actual), sorted(expected)])
    for key in actual:
        if key in expected and isinstance(actual[key], dict) and isinstance(expected[key], dict):
            compare(errors, [*prefix, key], actual[key], expected[key])
        elif key not in expected or not same(actual[key], expected[key]):
            errors.append([*prefix, key, actual[key], expected.get(key)])


def audit(run_map, verification, scores):
    run_map, verification, scores = (Path(p).resolve(strict=True) for p in (run_map, verification, scores))
    proof, primary = read(verification), read(scores/'results.json')
    guard = install_phase_mask_guard(DATA)
    current = validate_frozen(run_map, allow_completed_score=True)
    require(guard['open_attempts'] == 0 and proof.get('passed') is True and
            primary.get('passed') is True and primary.get('verification_sha256') == digest(verification) and
            primary.get('run_map_sha256') == digest(run_map) and
            primary.get('scorer_sha256') == proof.get('scorer_sha256') and
            primary.get('report_sha256') == digest(scores/'report.md'), 'Primary scores or verification differ')
    for key in ('selection', 'run_map_sha256', 'launch_manifest_sha256', 'receipts', 'references', 'inputs'):
        require(current[key] == proof[key], 'Frozen inputs changed after verification')
    mapping, errors = read(run_map), []
    guard['active'] = False
    checked_labels = checked_predictions = 0
    for seq in proof['selection']:
        meta_file = Path(proof['inputs'][seq]['frame_metadata']['path'])
        require(digest(meta_file) == proof['inputs'][seq]['frame_metadata']['sha256'], 'Metadata changed')
        metadata = read(meta_file)
        require(digest(meta_file) == proof['inputs'][seq]['frame_metadata']['sha256'], 'Metadata changed during read')
        frames, start, _, _ = input_data(seq, EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
        references = {arm: Path(item['path']).parent for arm, item in proof['references'][seq].items()}
        folders = {arm: Path(mapping[seq][arm]) for arm in METHODS} | references
        hashes = {}
        for arm, root in folders.items():
            receipt_path = root/('completion.json' if arm in ('A', 'D') else 'receipt.json')
            expected = (proof['references'][seq][arm] if arm in ('A', 'D') else proof['receipts'][seq][arm])
            require(digest(receipt_path) == expected['sha256'], 'Receipt changed')
            receipt = read(receipt_path)
            require(digest(receipt_path) == expected['sha256'], 'Receipt changed during read')
            records = receipt['masks'] if arm in ('A', 'D') else receipt['records']
            hashes[arm] = {row['frame']: row['sha256' if arm in ('A', 'D') else 'mask_sha256'] for row in records}
        rows = []
        require(len(frames) == len(metadata[seq]) == len(primary['rows'][seq]), 'Scored frame coverage differs')
        for i, frame in enumerate(frames):
            expected_label = metadata[seq][i]['label_sha256']
            require(digest(frame.image_path) == proof['inputs'][seq]['images'][i] and
                    digest(frame.mask_path) == expected_label, 'Input image or label changed')
            truth = label(frame.mask_path)
            require(digest(frame.mask_path) == expected_label, 'Label changed during audit')
            checked_labels += 1
            row = {'frame': i, 'source_frame_index': frame.frame_index,
                   'post_prompt': i > start, 'initialization_frame': i == start}
            for arm, root in folders.items():
                path = root/f'{i:06d}.png'
                require(digest(path) == hashes[arm][i], 'Prediction changed')
                with Image.open(path) as image:
                    pred = np.asarray(image.convert('L')) > 0
                require(digest(path) == hashes[arm][i], 'Prediction changed during audit')
                checked_predictions += 1
                row[arm] = metrics(pred, truth)
                compare(errors, [seq, i, arm], row[arm], primary['rows'][seq][i][arm])
            require(digest(frame.image_path) == proof['inputs'][seq]['images'][i], 'Image changed during audit')
            rows.append(row)
        for arm in folders:
            compare(errors, [seq, arm, 'post'], summarize(rows, arm, True), primary['summaries'][seq][arm])
            compare(errors, [seq, arm, 'all'], summarize(rows, arm, False), primary['all_frame_summaries'][seq][arm])
        if 'A' in folders:
            for arm in folders:
                if arm != 'A':
                    compare(errors, [seq, arm, 'paired'], paired(rows, arm, 'A'), primary['paired_descriptive'][seq][arm])
        label_arm = 'A' if 'A' in folders else 'native'
        expected_pre = {'frames': start, 'labeled_present': sum(r[label_arm]['present'] for r in rows[:start]),
                        'present_misses': {arm: sum(r[arm]['present'] and r[arm]['dice'] == 0 for r in rows[:start])
                                           for arm in folders}}
        compare(errors, [seq, 'preprompt'], expected_pre, primary['preprompt'][seq])
    return {'passed': not errors, 'scope': 'Independent CPU metric audit; shared structural gate verifier.',
            'run_map_sha256': digest(run_map), 'verification_sha256': digest(verification),
            'score_sha256': digest(scores/'results.json'), 'auditor_sha256': digest(Path(__file__)),
            'label_files_checked': checked_labels, 'prediction_files_checked': checked_predictions,
            'prelabel_guard': dict(guard), 'error_count': len(errors), 'errors': errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-map', 'verification', 'scores', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = new_output(a.output)
    result = audit(a.run_map, a.verification, a.scores)
    out.mkdir(parents=True, exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise ValueError(f"Independent audit found {result['error_count']} discrepancies")
    print(json.dumps({'passed': True, 'labels': result['label_files_checked'],
                      'predictions': result['prediction_files_checked']}))


if __name__ == '__main__':
    main()
