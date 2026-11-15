#!/usr/bin/env python3
"""Independently audit saved source-group fixed-read masks and scoring arithmetic.

This CPU-only, post-scoring audit opens previously scored validation labels and
does not modify the study or establish independent patient/video validation.
It directly decodes PNG/JPEG files and computes its own metrics, rather than
importing the experiment's scorer or dataset mask loader.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import validate_output

ARMS = ('region', 'global', 'no_contrast', 'mask_only')
NAMES = ('baseline', *ARMS)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def same(left, right):
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def metrics(prediction, truth):
    if prediction.shape != truth.shape:
        raise ValueError('Prediction and label geometries differ')
    tp = int(np.count_nonzero(prediction & truth))
    predicted = int(np.count_nonzero(prediction))
    target = int(np.count_nonzero(truth))
    return {'dice': 2*tp/(predicted+target) if predicted+target else 1.0,
            'precision': tp/predicted if predicted else None,
            'recall': tp/target if target else None,
            'predicted_pixels': predicted, 'false_alarm': bool(predicted) if not target else None}


def mean(values):
    return math.fsum(values)/len(values) if values else None


def summarize(rows, name):
    present = [row[name] for row in rows if row['present']]
    empty = [row[name] for row in rows if not row['present']]
    precision = [x['precision'] for x in present if x['precision'] is not None]
    return {'present_frames': len(present), 'mean_present_dice': mean([x['dice'] for x in present]),
            'mean_present_precision_when_defined': mean(precision), 'precision_defined_frames': len(precision),
            'mean_present_recall': mean([x['recall'] for x in present]), 'empty_frames': len(empty),
            'empty_false_alarms': sum(x['false_alarm'] for x in empty),
            'mean_empty_predicted_pixels': mean([x['predicted_pixels'] for x in empty])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study = validate_output(args.study)
    manifest_path = study/'manifest/manifest.json'
    score_path = study/'scores/scores.json'
    manifest, saved = read(manifest_path), read(score_path)
    actual_manifest_sha = digest(manifest_path)
    last_sequence = list(manifest['sequences'])[-1]
    shadowed_path = Path(manifest['sequences'][last_sequence]['capture_receipt_path'])
    shadowed_sha = digest(shadowed_path)
    # The v1 helper shadows its manifest-path variable and returns this last
    # capture-receipt hash. Preserve and identify the bug; do not normalize the
    # old evidence into a claim that its plan/checkpoints bound the manifest.
    if (not saved.get('passed') or read(study/'manifest/completion.json').get('manifest_sha256') != actual_manifest_sha
            or saved.get('manifest_sha256') != shadowed_sha or actual_manifest_sha == shadowed_sha
            or saved.get('plan_sha256') != digest(study/'plan/plan.json')
            or saved.get('verification_sha256') != digest(study/'verification/verification.json')
            or saved.get('report_sha256') != digest(study/'scores/report.md')
            or read(study/'scores/completion.json').get('scores_sha256') != digest(score_path)):
        raise ValueError('Study score provenance differs')
    items = {(x['sequence'], x['frame']): x for x in manifest['samples'] if x['split'] == 'validation'}
    recorded = {(r['sequence'], r['frame']): r for r in saved['rows']}
    if len(items) != 50 or len(recorded) != 50 or len(saved['rows']) != 50 or set(items) != set(recorded):
        raise ValueError('Validation coverage differs')
    arms = {}
    for arm in ARMS:
        payload = read(study/'runs'/arm/'frozen_predictions.json')
        rows = payload['rows']
        arms[arm] = {(x['sequence'], x['frame']): x for x in rows}
        if len(rows) != 50 or len(arms[arm]) != 50 or set(arms[arm]) != set(items):
            raise ValueError(f'{arm}: frozen prediction coverage differs')
    errors, rows, labels, masks = [], [], [], []
    for key, item in sorted(items.items()):
        label_path = Path(item['label'])
        label_sha = digest(label_path)
        if label_sha != item['label_sha256']:
            raise ValueError(f'{key}: label hash differs')
        with Image.open(label_path) as image:
            raw = np.asarray(image)
            gray = np.asarray(image.convert('L'))
            truth = gray > 127 if raw.ndim == 3 or (raw.max() > 1 and image.format == 'JPEG') else raw == 1
        labels.append({'sequence': key[0], 'frame': key[1], 'path': str(label_path), 'sha256': label_sha})
        row = {'sequence': key[0], 'frame': key[1], 'present': bool(truth.any())}
        if row['present'] != recorded[key]['present']:
            errors.append([*key, 'presence'])
        baseline = None
        for arm in ARMS:
            info = arms[arm][key]
            for kind in ('baseline', 'refined'):
                path = Path(info[kind]).resolve()
                if not path.is_relative_to((study/'runs'/arm/'frozen_predictions').resolve()):
                    raise ValueError(f'{key}/{arm}: mask path differs')
                mask_sha = digest(path)
                if mask_sha != info[kind+'_sha256']:
                    raise ValueError(f'{key}/{arm}/{kind}: mask hash differs')
                with Image.open(path) as image:
                    prediction = np.asarray(image.convert('L')) > 0
                if kind == 'baseline':
                    if baseline is None:
                        baseline = prediction
                    elif not np.array_equal(baseline, prediction):
                        errors.append([*key, arm, 'baseline pixels differ'])
                    name = 'baseline'
                else:
                    name = arm
                value = metrics(prediction, truth)
                for field, actual in value.items():
                    if not same(actual, recorded[key][name][field]):
                        errors.append([*key, arm, kind, field, actual, recorded[key][name][field]])
                row[name] = value
                masks.append({'sequence': key[0], 'frame': key[1], 'arm': arm, 'kind': kind,
                              'path': str(path), 'sha256': mask_sha})
        rows.append(row)
    computed = {}
    sequences = sorted({x['sequence'] for x in rows})
    for name in NAMES:
        pooled = summarize(rows, name)
        per_video = {seq: summarize([x for x in rows if x['sequence'] == seq], name) for seq in sequences}
        means = [entry['mean_present_dice'] for entry in per_video.values()
                 if entry['mean_present_dice'] is not None]
        equal = mean(means)
        computed[name] = {'pooled_frames': pooled, 'per_video': per_video, 'equal_video_mean_present_dice': equal}
        original = saved['summary'][name]
        for seq, stats in [(None, pooled), *per_video.items()]:
            expected = original['pooled_frames'] if seq is None else original['per_video'][seq]
            for field, value in stats.items():
                if not same(value, expected[field]):
                    errors.append([name, seq, field, value, expected[field]])
        if not same(equal, original['equal_video_mean_present_dice']):
            errors.append([name, 'equal_video_mean_present_dice', equal,
                           original['equal_video_mean_present_dice']])
    if len(labels) != 50 or len(masks) != 400:
        raise ValueError('Unexpected label or output mask count')
    result = {'passed': not errors, 'scope': 'Post-scoring arithmetic and frozen-mask audit on previously inspected development videos; not independent validation.',
              'study_path': str(study), 'audit_script_sha256': digest(Path(__file__)),
              'actual_manifest_sha256': actual_manifest_sha,
              'recorded_manifest_sha256': saved['manifest_sha256'],
              'shadowed_capture_receipt_path': str(shadowed_path),
              'shadowed_capture_receipt_sha256': shadowed_sha,
              'provenance_limitation': 'The original v1 load_prepared returns the last capture-receipt hash as manifest_sha256 due to variable shadowing. Its separate manifest completion hash is correct, but plan/checkpoint/score fields are mislabeled and do not directly bind the manifest.',
              'scores_sha256': digest(score_path),
              'plan_sha256': digest(study/'plan/plan.json'),
              'verification_sha256': digest(study/'verification/verification.json'),
              'validation_frames': len(rows), 'present_frames': sum(x['present'] for x in rows),
              'empty_frames': sum(not x['present'] for x in rows), 'mask_files_checked': len(masks),
              'label_sha256': labels, 'mask_sha256': masks, 'independent_summary': computed,
              'error_count': len(errors), 'errors': errors}
    output = validate_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = output/'audit.json'
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    (output/'completion.json').write_text(json.dumps({'passed': not errors, 'audit_sha256': digest(receipt)}, indent=2)+'\n')
    print(json.dumps({'passed': not errors, 'validation_frames': len(rows),
                      'mask_files_checked': len(masks), 'error_count': len(errors)}))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
