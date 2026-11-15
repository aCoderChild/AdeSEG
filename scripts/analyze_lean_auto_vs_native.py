#!/usr/bin/env python3
"""Audit matched-frame trajectories of existing automatic Lean and native runs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def read_json(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('merged', 'historical', 'native', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    done = read_json(a.merged / 'completion.json')
    if not done['passed'] or digest(a.merged / 'results.json') != done['results_sha256'] or digest(a.historical / 'comparison.json') != done['historical_comparison_sha256']:
        raise ValueError('Merged or historical receipt invalid')
    result = read_json(a.merged / 'results.json')
    locations = read_json(a.merged / 'locations.json')
    history = read_json(a.historical / 'comparison.json')
    prior = read_json(a.historical / 'preflight.json')
    if digest(a.native / 'setup.json') != prior['native_setup_sha256']:
        raise ValueError('Native setup differs from historical comparison')
    old = {row['sequence']: row for row in history['sequences']}
    summary = {}
    for name, aggregate in result['sequences'].items():
        run = Path(locations[name]); folder = run / name
        receipt = read_json(run / 'completion.json'); audit = read_json(run / 'independent_audit.json')
        if not receipt['passed'] or not audit['passed'] or digest(run / 'records.json') != receipt['records_sha256'] or audit['records_sha256'] != receipt['records_sha256']:
            raise ValueError(f'{name} incomplete')
        record = read_json(run / 'records.json')[0]
        if digest(folder / 'frames.json') != record['frames_sha256']:
            raise ValueError(f'{name} frames receipt changed')
        native_file = a.native / name / 'frames.jsonl'
        if digest(native_file) != prior['native_frames_sha256'][name]:
            raise ValueError(f'{name} native frame record changed')
        auto = read_json(folder / 'frames.json')
        native = [json.loads(line) for line in native_file.read_text().splitlines()]
        selected = [(row, auto[row['frame']]) for row in native]
        if any(n['frame'] != x['index'] or n['present'] != x['present'] for n, x in selected):
            raise ValueError(f'{name} frame alignment differs')
        present = [(n, x) for n, x in selected if n['present']]
        auto_mean = float(np.mean([x['dice'] for n, x in present]))
        native_mean = float(np.mean([n['scores']['dice'] for n, x in present]))
        if not np.isclose(auto_mean, aggregate['historical_present_dice'], atol=1e-12) or not np.isclose(native_mean, old[name]['native_dice'], atol=1e-12):
            raise ValueError(f'{name} aggregation differs')
        first = aggregate['first_detection_index']
        pre = [(n, x) for n, x in present if first is None or n['frame'] < first]
        post = [(n, x) for n, x in present if first is not None and n['frame'] >= first]
        contribution = lambda pairs: sum(x['dice'] - n['scores']['dice'] for n, x in pairs) / len(present)
        lean_mean = old[name]['lean_dice']
        summary[name] = {'native_dice': native_mean, 'lean_dice': lean_mean,
            'better_historical_dice': max(native_mean, lean_mean),
            'better_historical_method': 'native' if native_mean >= lean_mean else 'lean',
            'delta_auto_minus_better_historical': auto_mean - max(native_mean, lean_mean),
            'automatic_dice': auto_mean,
            'delta_auto_minus_native': auto_mean-native_mean, 'present_targets': len(present),
            'pre_detection_present_targets': len(pre), 'pre_detection_delta_contribution': contribution(pre),
            'post_detection_delta_contribution': contribution(post),
            'native_gt_prompt': True, 'first_detection_index': first,
            'first_detection_labeled_present': aggregate['first_detection_labeled_present'],
            'auto_zero_dice_present_frames': sum(x['dice'] == 0 for _, x in present),
            'native_zero_dice_present_frames': sum(n['scores']['dice'] == 0 for n, _ in present),
            'auto_empty_false_alarms': aggregate['historical_false_alarms'],
            'native_empty_false_alarms': old[name]['native_false_alarms'],
            'empty_targets': aggregate['historical_empty']}
    out = validate_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    data = {'scope': 'Exploratory descriptive matched-frame trajectories. Native uses a GT initial mask, combined uses an automatic prompt; differences do not isolate components or establish causal mechanisms.',
            'merged_results_sha256': digest(a.merged / 'results.json'), 'historical_comparison_sha256': digest(a.historical / 'comparison.json'),
            'sequences': summary}
    (out / 'results.json').write_text(json.dumps(data, indent=2) + '\n')
    lines = ['# Automatic Lean versus historical native: per-sequence diagnostic', '', data['scope'], '',
             'Delta is automatic minus native Dice, decomposed into each frame group’s contribution to the sequence mean. An automatic value of zero before detection is not a controlled comparison of segmenters.', '',
             'The better historical score is selected after evaluation for each sequence; it is a retrospective reference, not an automatic inference-time selector.', '',
             '| Sequence | Native | Lean | Automatic | Delta vs better | Delta vs native | Present targets before detection | Pre-detection contribution vs native | Post-detection contribution vs native | Automatic zero-Dice frames |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, r in summary.items():
        lines.append(f"| {name} | {r['native_dice']:.3f} | {r['lean_dice']:.3f} | {r['automatic_dice']:.3f} | {r['delta_auto_minus_better_historical']:+.3f} | {r['delta_auto_minus_native']:+.3f} | {r['pre_detection_present_targets']} | {r['pre_detection_delta_contribution']:+.3f} | {r['post_detection_delta_contribution']:+.3f} | {r['auto_zero_dice_present_frames']} / {r['present_targets']} |")
    lines.extend(['', 'First detections, empty-frame false alarms, and machine-readable values are in results.json. The current outputs cannot independently distinguish prompt, memory, and propagation effects.', ''])
    (out / 'report.md').write_text('\n'.join(lines))
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'results_sha256': digest(out / 'results.json'), 'source_sha256': digest(Path(__file__))}, indent=2) + '\n')
    print(json.dumps({name: row for name, row in summary.items() if row['delta_auto_minus_native'] < -0.01}, indent=2))


if __name__ == '__main__':
    main()
