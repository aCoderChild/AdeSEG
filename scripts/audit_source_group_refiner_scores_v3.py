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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import validate_output

ARMS = ('reference', 'early_only', 'deep_only', 'foreground_only', 'rgb_contrast', 'spatial_shuffle')
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
    parser.add_argument('--v2-study', type=Path, required=True)
    parser.add_argument('--v2-audit', type=Path, required=True)
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
    if (not saved.get('passed') or read(study/'manifest/completion.json').get('manifest_sha256') != actual_manifest_sha
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
            or saved.get('scorer_sha256') != digest(ROOT/'scripts/score_source_group_refiner_v3.py')
            or verified.get('runner_sha256') != digest(ROOT/'scripts/verify_source_group_refiner_v3.py')
            or read(study/'scores/completion.json').get('scores_sha256') != digest(score_path)):
        raise ValueError('Study score provenance differs')
    dependencies = plan.get('dependency_sha256', {})
    if (not dependencies or any(digest(ROOT/path) != sha for path, sha in dependencies.items())
            or verified.get('dependency_sha256') != dependencies
            or verified.get('environment') != plan.get('environment')
            or verified.get('label_guard', {}).get('open_attempts') != 0
            or set(plan.get('arms', [])) != set(ARMS) or set(verified.get('runs', {})) != set(ARMS)):
        raise ValueError('Source, environment, guard, or control identities differ')
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
                or checkpoint.get('study_arm') != arm or checkpoint.get('control') != arm
                or checkpoint.get('dependency_sha256') != dependencies
                or checkpoint.get('initial_state_sha256') != plan.get('initial_state_sha256')
                or checkpoint.get('train_manifest_sha256') != actual_manifest_sha
                or checkpoint.get('plan_sha256') != digest(plan_path)
                or checkpoint.get('parity_sha256') != digest(parity_path)
                or completion.get('plan_sha256') != digest(plan_path)
                or completion.get('parity_sha256') != digest(parity_path)
                or completion.get('checkpoint_sha256') != digest(checkpoint_path)
                or payload.get('checkpoint_sha256') != digest(checkpoint_path)
                or completion.get('frozen_predictions_sha256') != digest(frozen_path)
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
    if len(labels) != 50 or len(masks) != 600:
        raise ValueError('Unexpected label or output mask count')
    # Compare with the separately audited V2 output, without asserting
    # bitwise identity: both runs use the same data and seed.
    v2 = validate_output(args.v2_study)
    v2_audit = read(args.v2_audit/'audit.json')
    if (not v2_audit.get('passed') or v2_audit.get('actual_manifest_sha256') != digest(v2/'manifest/manifest.json')
            or v2_audit.get('scores_sha256') != digest(v2/'scores/scores.json')
            or read(args.v2_audit/'completion.json').get('audit_sha256') != digest(args.v2_audit/'audit.json')):
        raise ValueError('Independent V2 audit identity differs')
    v2_manifest = read(v2/'manifest/manifest.json')
    v2_items = {(x['sequence'], x['frame']): x for x in v2_manifest['samples'] if x['split'] == 'validation'}
    if set(v2_items) != set(items) or any(v2_items[k]['label_sha256'] != items[k]['label_sha256']
                                         or v2_items[k]['image_sha256'] != items[k]['image_sha256'] for k in items):
        raise ValueError('V2 and V3 validation image/label identities differ')
    v2_pred = read(v2/'runs/region/frozen_predictions.json')
    v2_rows = {(x['sequence'], x['frame']): x for x in v2_pred['rows']}
    if len(v2_rows) != 50 or len(v2_pred['rows']) != 50 or set(v2_rows) != set(items):
        raise ValueError('V2 reference coverage differs')
    comparisons = {}
    for kind in ('baseline', 'refined'):
        byte_equal, pixel_differences, changed = 0, 0, []
        for key in sorted(items):
            v2_path = Path(v2_rows[key][kind]); v3_path = Path(arms['reference'][key][kind])
            if digest(v2_path) != v2_rows[key][kind+'_sha256']:
                raise ValueError('V2 saved mask differs from recorded identity')
            byte_equal += digest(v2_path) == digest(v3_path)
            with Image.open(v2_path) as im:
                a = np.asarray(im.convert('L')) > 0
            with Image.open(v3_path) as im:
                b = np.asarray(im.convert('L')) > 0
            if a.shape != b.shape:
                raise ValueError('V2 and V3 mask geometries differ')
            n = int(np.count_nonzero(a != b)); pixel_differences += n
            if n:
                changed.append({'sequence': key[0], 'frame': key[1], 'pixels': n})
        comparisons[kind] = {'count': 50, 'byte_identical': byte_equal,
                             'different_pixels': pixel_differences, 'changed_frames': changed}
    result = {'passed': not errors, 'scope': 'Post-scoring arithmetic and frozen-mask audit on previously inspected development videos; not independent validation.',
              'study_path': str(study), 'audit_script_sha256': digest(Path(__file__)),
              'actual_manifest_sha256': actual_manifest_sha,
              'recorded_manifest_sha256': saved['manifest_sha256'],
              'provenance_scope': 'V3 manifest hash matches the plan, replay receipt, six checkpoints, completions and score. This independent audit does not rerun training or establish patient independence.',
              'arm_provenance': provenance, 'comparison_v2_same_seed_same_data': comparisons,
              'v2_audit_sha256': digest(args.v2_audit/'audit.json'),
              'scores_sha256': digest(score_path),
              'plan_sha256': digest(study/'plan/plan.json'),
              'parity_sha256': digest(parity_path),
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
