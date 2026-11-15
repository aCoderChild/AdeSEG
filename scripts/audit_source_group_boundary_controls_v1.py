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
import cv2
from scipy import ndimage
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import validate_output

ARMS = tuple(f'{m}_boundary' for m in ('region', 'global', 'descriptor_roll', 'spatial_shuffle'))
NAMES = ('baseline', *ARMS)
LOSS = {arm: 'CMSA corrected weighted BCE + aggregate weighted IoU + project soft Dice' for arm in ARMS}


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


def boundary_five_pixel(prediction, truth):
    structure = np.ones((3, 3), dtype=bool)
    pred_boundary = prediction & ~ndimage.binary_erosion(prediction, structure=structure, border_value=0)
    true_boundary = truth & ~ndimage.binary_erosion(truth, structure=structure, border_value=0)
    p, t = int(pred_boundary.sum()), int(true_boundary.sum())
    if not p or not t:
        return 1.0 if not p and not t else 0.0
    # Same specified ellipse, independently applied with SciPy morphology.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)).astype(bool)
    near_true = ndimage.binary_dilation(true_boundary, structure=kernel)
    near_pred = ndimage.binary_dilation(pred_boundary, structure=kernel)
    precision = int((pred_boundary & near_true).sum()) / p
    recall = int((true_boundary & near_pred).sum()) / t
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def metrics(prediction, truth):
    if prediction.shape != truth.shape:
        raise ValueError('Prediction and label geometries differ')
    tp = int(np.count_nonzero(prediction & truth))
    predicted = int(np.count_nonzero(prediction))
    target = int(np.count_nonzero(truth))
    return {'dice': 2*tp/(predicted+target) if predicted+target else 1.0,
            'precision': tp/predicted if predicted else None,
            'recall': tp/target if target else None,
            'predicted_pixels': predicted, 'zero_overlap': bool(tp == 0) if target else None,
            'boundary_f_5px': boundary_five_pixel(prediction, truth),
            'false_alarm': bool(predicted) if not target else None}


def mean(values):
    return math.fsum(values)/len(values) if values else None


def summarize(rows, name):
    present = [row[name] for row in rows if row['present']]
    empty = [row[name] for row in rows if not row['present']]
    precision = [x['precision'] for x in present if x['precision'] is not None]
    return {'present_frames': len(present), 'mean_present_dice': mean([x['dice'] for x in present]),
            'mean_present_boundary_f_5px': mean([x['boundary_f_5px'] for x in present]),
            'present_zero_overlap_frames': sum(x['zero_overlap'] for x in present),
            'mean_present_precision_when_defined': mean(precision), 'precision_defined_frames': len(precision),
            'mean_present_recall': mean([x['recall'] for x in present]), 'empty_frames': len(empty),
            'empty_false_alarms': sum(x['false_alarm'] for x in empty),
            'mean_empty_predicted_pixels': mean([x['predicted_pixels'] for x in empty])}


def matched_contrasts(computed):
    """Report post-hoc control differences relative to the repeated reference."""
    result = {}
    reference = computed['region_boundary']
    for arm in ARMS:
        other = computed[arm]
        delta = lambda a, b: None if a is None or b is None else a-b
        result[arm] = {
            'pooled_present_dice_delta': delta(other['pooled_frames']['mean_present_dice'], reference['pooled_frames']['mean_present_dice']),
            'equal_video_present_dice_delta': delta(other['equal_video_mean_present_dice'], reference['equal_video_mean_present_dice']),
            'pooled_boundary_f_delta': delta(other['pooled_frames']['mean_present_boundary_f_5px'], reference['pooled_frames']['mean_present_boundary_f_5px']),
            'empty_false_alarm_delta': other['pooled_frames']['empty_false_alarms'] - reference['pooled_frames']['empty_false_alarms'],
        }
    return result


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
    plan_path = study/'plan/plan.json'
    parity_path = study/'parity/parity.json'
    verification_path = study/'verification/verification.json'
    plan, verified = read(plan_path), read(verification_path)
    verify_completion = read(study/'verification/completion.json')
    if (not saved.get('passed') or not plan.get('passed') or plan.get('loss') != LOSS
            or plan.get('arms') != list(ARMS) or plan.get('parameter_count') != 48005
            or plan.get('epochs') != 10 or plan.get('seed') != 20261115
            or read(study/'manifest/completion.json').get('manifest_sha256') != actual_manifest_sha
            or saved.get('manifest_sha256') != actual_manifest_sha
            or plan.get('manifest_sha256') != actual_manifest_sha
            or verified.get('manifest_sha256') != actual_manifest_sha
            or not verified.get('passed') or not verify_completion.get('passed')
            or verify_completion.get('verification_sha256') != digest(verification_path)
            or saved.get('plan_sha256') != digest(plan_path)
            or verified.get('plan_sha256') != digest(plan_path)
            or verified.get('parity_sha256') != digest(parity_path)
            or not read(parity_path).get('passed')
            or saved.get('verification_sha256') != digest(verification_path)
            or saved.get('report_sha256') != digest(study/'scores/report.md')
            or saved.get('scorer_sha256') != digest(ROOT/'scripts/score_source_group_boundary_controls_v1.py')
            or verified.get('runner_sha256') != digest(ROOT/'scripts/verify_source_group_boundary_controls_v1.py')
            or read(study/'scores/completion.json').get('scores_sha256') != digest(score_path)):
        raise ValueError('Study score provenance differs')
    items = {(x['sequence'], x['frame']): x for x in manifest['samples'] if x['split'] == 'validation'}
    recorded = {(r['sequence'], r['frame']): r for r in saved['rows']}
    if len(items) != 50 or len(recorded) != 50 or len(saved['rows']) != 50 or set(items) != set(recorded):
        raise ValueError('Validation coverage differs')
    arms = {}
    provenance = {}
    for arm in ARMS:
        folder = study/'runs'/arm
        frozen_path, completion_path, checkpoint_path = (folder/'frozen_predictions.json', folder/'completion.json',
                                                         folder/'checkpoint.pt')
        payload = read(frozen_path)
        completion = read(completion_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        evidence = verified['runs'][arm]
        if (not completion.get('passed') or completion.get('arm') != arm
                or completion.get('manifest_sha256') != actual_manifest_sha
                or checkpoint.get('study_arm') != arm or checkpoint.get('loss') != LOSS[arm]
                or checkpoint.get('initial_state_sha256') != plan.get('initial_state_sha256')
                or checkpoint.get('training_steps') != 1640 or checkpoint.get('seed') != 20261115
                or checkpoint.get('train_manifest_sha256') != actual_manifest_sha
                or checkpoint.get('plan_sha256') != digest(plan_path)
                or checkpoint.get('parity_sha256') != digest(parity_path)
                or completion.get('plan_sha256') != digest(plan_path)
                or completion.get('parity_sha256') != digest(parity_path)
                or completion.get('checkpoint_sha256') != digest(checkpoint_path)
                or payload.get('checkpoint_sha256') != digest(checkpoint_path)
                or completion.get('frozen_predictions_sha256') != digest(frozen_path)
                or completion.get('training_order') != plan.get('training_order')
                or completion.get('label_guard', {}).get('blocked_open_attempts') != 0
                or evidence.get('completion_sha256') != digest(completion_path)
                or evidence.get('checkpoint_sha256') != digest(checkpoint_path)
                or evidence.get('frozen_sha256') != digest(frozen_path)
                or Path(evidence['completion_path']).resolve() != completion_path.resolve()):
            raise ValueError(f'{arm}: checkpoint, completion or replay provenance differs')
        provenance[arm] = {'checkpoint_sha256': digest(checkpoint_path),
                           'completion_sha256': digest(completion_path), 'frozen_sha256': digest(frozen_path)}
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
              'provenance_scope': 'Manifest hash matches plan, replay receipt, four checkpoints, completions and score. This audit does not rerun training or establish patient independence.',
              'arm_provenance': provenance,
              'scores_sha256': digest(score_path),
              'plan_sha256': digest(study/'plan/plan.json'),
              'parity_sha256': digest(parity_path),
              'verification_sha256': digest(study/'verification/verification.json'),
              'validation_frames': len(rows), 'present_frames': sum(x['present'] for x in rows),
              'empty_frames': sum(not x['present'] for x in rows), 'mask_files_checked': len(masks),
              'label_sha256': labels, 'mask_sha256': masks, 'independent_summary': computed,
              'matched_control_minus_region_boundary': matched_contrasts(computed),
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
