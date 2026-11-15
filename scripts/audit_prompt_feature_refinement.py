#!/usr/bin/env python3
"""Independently check the saved matched-frame prompt pilot and make a report."""
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
from evaluation.statistics import sequence_bootstrap_interval  # noqa: E402
from scripts.memory_token_io import digest  # noqa: E402
from scripts.run_prompt_feature_refinement import (ARMS, MANIFEST, box_iou, choose_cases, gt_box)  # noqa: E402


def audit_mask(path, truth, expected):
    if digest(path) != expected['mask_sha256']:
        raise ValueError(f'Prediction changed: {path}')
    pred = np.asarray(Image.open(path)) > 0
    if pred.shape != truth.shape:
        raise ValueError(f'Dimensions differ: {path}')
    intersection = int(np.count_nonzero(pred & truth))
    denominator = int(pred.sum() + truth.sum())
    dice = 2 * intersection / denominator if denominator else 1.0
    if not np.isclose(dice, expected['dice'], atol=1e-12, rtol=0):
        raise ValueError(f'Dice mismatch: {path}')
    if not truth.any() and bool(pred.any()) != expected['false_alarm']:
        raise ValueError(f'False-alarm mismatch: {path}')
    return dice


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    run = args.run
    setup = json.loads((run / 'setup.json').read_text())
    done = json.loads((run / 'completion.json').read_text())
    if setup['mode'] != 'pilot' or not done['passed']:
        raise ValueError('Pilot is incomplete')
    if done['records_sha256'] != digest(run / 'records.json') or done['results_sha256'] != digest(run / 'results.json'):
        raise ValueError('Pilot results changed')
    if setup['signature']['manifest_sha256'] != digest(MANIFEST):
        raise ValueError('Manifest changed')
    records = json.loads((run / 'records.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    data = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    names = [f'seq{i}' for i in range(2, 24) if i != 7]
    expected_cases = []
    for name in names:
        prompt, indices = choose_cases(data[name])
        expected_cases.extend((name, i, prompt) for i in indices)
    if [(r['sequence'], r['index'], r['prompt_index']) for r in records] != expected_cases:
        raise ValueError('Frame selection differs from frozen policy')
    values = {arm: {name: [] for name in names} for arm in ARMS}
    alarms = {arm: 0 for arm in ARMS}
    image_count = 0
    for row in records:
        name, i, prompt = row['sequence'], row['index'], row['prompt_index']
        frame = data[name][i]
        if frame.frame_index != row['source_frame_index'] or digest(frame.image_path) != row['image_sha256'] or digest(frame.mask_path) != row['target_mask_sha256']:
            raise ValueError(f'Target provenance mismatch: {name}/{i}')
        if row['reference_status'] == 'available':
            ref_index = row['reference_index']
            if ref_index is None or not 0 <= ref_index < i:
                raise ValueError(f'Reference is not strictly earlier: {name}/{i}')
            reference = data[name][ref_index]
            if reference.frame_index != row['reference_frame'] or digest(reference.image_path) != row['reference_image_sha256']:
                raise ValueError(f'Reference provenance mismatch: {name}/{i}')
            reference_prediction = run / name / 'reference_prediction.png'
            if digest(reference_prediction) != row['reference_prediction_sha256'] or row['reference_box'] is None or not np.asarray(Image.open(reference_prediction)).any():
                raise ValueError(f'Invalid automatic reference: {name}/{i}')
        elif row['reference_status'] != 'unavailable_prior' or row['reference_index'] is not None or row['reference_prediction_sha256'] is not None:
            raise ValueError(f'Invalid reference fallback: {name}/{i}')
        truth = load_label_mask(frame.mask_path, [1]) == 1
        oracle = gt_box(truth)
        if row['oracle_box'] != oracle or row['present'] != bool(truth.any()):
            raise ValueError(f'Oracle/presence mismatch: {name}/{i}')
        if truth.any() and not np.isclose(row['detector_box_iou'], box_iou(row['box'], oracle)):
            raise ValueError(f'Detector IoU mismatch: {name}/{i}')
        if row['box'] is None and row['point_selection'] is not None:
            raise ValueError('Points exist without a box')
        if row['point_selection']:
            x0, y0, x1, y1 = row['box']
            for typ in ('feature_points', 'random_points'):
                pts = row['point_selection'][typ]
                if len(pts) != 2 or pts[0] == pts[1] or any(not (x0 <= x < x1 and y0 <= y < y1) for x, y in pts):
                    raise ValueError(f'Invalid {typ}: {name}/{i}')
        expected_arms = set(ARMS if truth.any() else ARMS[:-1])
        if set(row['conditions']) != expected_arms:
            raise ValueError('Incorrect oracle availability')
        for arm, expected in row['conditions'].items():
            path = run / name / f'{i:06d}_{arm}.png'
            dice = audit_mask(path, truth, expected)
            if row['box'] is None and arm != 'oracle_gt_box' and np.asarray(Image.open(path)).any():
                raise ValueError('No-detection fallback produced foreground')
            if truth.any():
                values[arm][name].append(dice)
            else:
                alarms[arm] += int(expected['false_alarm'])
            image_count += 1
    if len(records) != results['targets'] or sum(r['present'] for r in records) != results['present_targets'] or sum(not r['present'] for r in records) != results['empty_targets']:
        raise ValueError('Denominator mismatch')
    if sum(r['box'] is None for r in records) != results['detector_no_box'] or sum(r['box'] is not None for r in records if not r['present']) != results['empty_frame_detections']:
        raise ValueError('Detector accounting mismatch')
    if sum(r['point_fallback'] for r in records) != results['point_fallbacks'] or len({r['sequence'] for r in records if r['reference_status'] == 'available'}) != results['reference_available_sequences']:
        raise ValueError('Reference/fallback accounting mismatch')
    present = [r for r in records if r['present']]
    if not np.isclose(np.mean([r['detector_box_iou'] >= .5 for r in present]), results['detector_recall_iou_0_5']):
        raise ValueError('Detector recall mismatch')
    if not np.isclose(np.mean([r['detector_box_iou'] for r in present]), results['mean_detector_box_iou_present']):
        raise ValueError('Detector IoU aggregate mismatch')
    means = {}
    for arm in ARMS:
        means[arm] = {name: float(np.mean(v)) for name, v in values[arm].items() if v}
        if means[arm] != results['arms'][arm]['per_sequence_dice']:
            raise ValueError(f'Per-sequence mismatch: {arm}')
        if not np.isclose(np.mean(list(means[arm].values())), results['arms'][arm]['mean_present_dice_equal_sequence']):
            raise ValueError(f'Aggregate Dice mismatch: {arm}')
        if arm != 'oracle_gt_box' and alarms[arm] != results['arms'][arm]['empty_false_alarms']:
            raise ValueError(f'Aggregate false alarms mismatch: {arm}')
    differences = {name: means['box_feature_points'][name] - means['yolo_box'][name] for name in means['yolo_box']}
    random_diff = {name: means['box_feature_points'][name] - means['box_random_points'][name] for name in means['yolo_box']}
    lines = ['# Feature-guided prompt pilot', '',
        f"{len(names)} sequences, {results['present_targets']} present and {results['empty_targets']} labeled-empty selected targets. "
        'The first usable prior YOLO–SAM2 prediction supplies the automatic feature reference; GT is used for retrospective frame sampling, scoring and the oracle only. No video propagation.', '',
        '| Prompt condition | Equal-sequence present Dice | False alarms / empty targets |', '|---|---:|---:|']
    for arm in ARMS:
        alarm = 'N/A: no oracle on empty frames' if arm == 'oracle_gt_box' else f"{alarms[arm]} / {results['empty_targets']}"
        lines.append(f"| {arm} | {results['arms'][arm]['mean_present_dice_equal_sequence']:.5f} | {alarm} |")
    lines += ['', f"Top YOLO box IoU ≥ 0.5 on {sum(r['detector_box_iou'] >= .5 for r in present)} / {len(present)} present frames; "
              f"mean top-box IoU {results['mean_detector_box_iou_present']:.5f}. "
              f"No detector box on {results['detector_no_box']} / {len(records)} targets, including "
              f"{sum(r['box'] is None for r in present)} present; detections on {results['empty_frame_detections']} / {results['empty_targets']} empty frames.",
              f"Automatic reference available in {results['reference_available_sequences']} / {len(names)} sequences; feature/random prompts fell back to box-only on {results['point_fallbacks']} detected targets.",
              'The feature-point method retains the same detector box, so its box IoU and detection recall are unchanged.',
              '', f"Feature points − YOLO box: {np.mean(list(differences.values())):+.5f}; exploratory sequence-bootstrap interval {sequence_bootstrap_interval(differences)}.",
              f"Feature points − random points: {np.mean(list(random_diff.values())):+.5f}; exploratory sequence-bootstrap interval {sequence_bootstrap_interval(random_diff)}.", '',
              '| Sequence | YOLO box | Random points | Feature points | GT box oracle |', '|---|---:|---:|---:|---:|']
    for name in names:
        lines.append(f"| {name} | " + ' | '.join(f'{means[arm][name]:.5f}' for arm in ARMS) + ' |')
    lines += ['', 'Detector checkpoint metadata names polypgen.yaml, but training split and overlap are unknown. Frames were selected using GT presence from an already inspected dataset. '
              'The feature condition uses an extra earlier image and model calls; its automatically predicted reference can be wrong. Comparisons are exploratory. '
              'An oracle GT box is a reference condition, not a mathematical upper bound on mask Dice. '
              'No-box frames use empty output for all detector conditions; no re-detection or fallback is evaluated.']
    (run / 'report.md').write_text('\n'.join(lines) + '\n')
    (run / 'independent_audit.json').write_text(json.dumps({'passed': True, 'targets': len(records),
        'saved_masks_checked': image_count, 'records_sha256': digest(run / 'records.json'),
        'results_sha256': digest(run / 'results.json'), 'audit_source_sha256': digest(Path(__file__))}, indent=2) + '\n')
    print(f'Audited {len(records)} targets and {image_count} saved masks')


if __name__ == '__main__':
    main()
