"""Frozen sequence-level analysis of the broader immediate-read diagnostic."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.statistics import holm_adjust, paired_wilcoxon, sequence_bootstrap_interval
from scripts.memory_token_io import digest, validate_output


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def validate_target_record(row, target, methods):
    """Reject corrupt strata and scores before any averaging or screening."""
    if not all(row.get(key) is True for key in
               ('native_parity', 'bank_unchanged', 'native_restored')):
        raise ValueError('Integrity check failed')
    if row['frame'] != target['frame'] or set(row['rows']) != set(methods):
        raise ValueError('Frozen target or condition coverage differs')
    for method in methods:
        value = row['rows'][method]
        if type(value['present']) is not bool or value['present'] != target['present']:
            raise ValueError('Frozen target strata differs across conditions')
        if target['present']:
            for metric in ('dice', 'iou', 'boundary_f'):
                score = value['scores'][metric]
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f'Invalid present-target {metric} for {method}')
            if value['absent_fp'] is not None:
                raise ValueError('Present target has GT-empty occupancy evidence')
        elif type(value['absent_fp']) is not bool:
            raise ValueError('GT-empty target lacks boolean occupancy evidence')
        if 'seconds' in value:
            seconds = value['seconds']
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('Invalid measured replay/scoring time')


def summarize(folder, plan_path):
    folder = validate_output(folder)
    plan = json.loads(plan_path.read_text())
    protocol = plan['protocol']
    completion = json.loads((folder / 'completion.json').read_text())
    if not completion.get('passed') or completion['scope'] != 'broader_causal_diagnostic':
        raise ValueError('Completed broader causal diagnostic required')
    if digest(plan_path) != completion['plan_sha256'] or digest(folder / 'results.json') != completion['results_sha256']:
        raise ValueError('Plan or results changed')
    records = json.loads((folder / 'results.json').read_text())
    if set(records) != set(protocol['sequences']):
        raise ValueError('Incomplete sequence coverage')
    methods = ['native'] + protocol['conditions']
    by_method = {method: {} for method in methods}
    totals = {method: {'false_alarms': 0, 'gt_empty_targets': 0, 'return_events': 0,
                      'returns_with_overlap': 0, 'replay_seconds': []} for method in methods}
    returns = {method: {} for method in methods}
    for sequence in protocol['sequences']:
        specification = plan['sequences'][sequence]
        schedule = specification['schedule']
        rows = {row['frame']: row for row in records[sequence]}
        if len(rows) != len(records[sequence]) or set(rows) != {row['frame'] for row in schedule['targets']}:
            raise ValueError('Duplicate or incomplete target records')
        for target in schedule['targets']:
            row = rows[target['frame']]
            validate_target_record(row, target, methods)
        for method in methods:
            present = [row['rows'][method] for row in rows.values() if row['rows'][method]['present']]
            if not present:
                raise ValueError(f'No evaluable present target for {sequence}')
            by_method[method][sequence] = {
                metric: statistics.mean(row['scores'][metric] for row in present)
                for metric in ('dice', 'iou', 'boundary_f')}
            events = []
            for event in schedule['return_events']:
                first = next((offset for offset, frame in enumerate(event['evaluated_followup'])
                              if rows[frame]['rows'][method]['scores']['iou'] > 0), None)
                events.append({**event, 'detected_in_window': first is not None,
                               'first_overlap_offset': first})
            returns[method][sequence] = events
            total = totals[method]
            total['return_events'] += len(events)
            total['returns_with_overlap'] += sum(event['detected_in_window'] for event in events)
            absent = [row['rows'][method] for row in rows.values() if not row['rows'][method]['present']]
            total['false_alarms'] += sum(bool(row['absent_fp']) for row in absent)
            total['gt_empty_targets'] += len(absent)
            total['replay_seconds'].extend(row['rows'][method]['seconds'] for row in rows.values()
                                          if 'seconds' in row['rows'][method])
    from datasets.video import load_manifest
    videos = {**load_manifest(ROOT / protocol['manifest'], 'dev'), **load_manifest(ROOT / protocol['manifest'], 'test')}
    for sequence, specification in plan['sequences'].items():
        for index, sha in specification['annotation_sha256'].items():
            if digest(videos[sequence][int(index)].mask_path) != sha:
                raise ValueError('Evaluation annotations changed after coverage was frozen')
    dice = {method: {name: values['dice'] for name, values in rows.items()} for method, rows in by_method.items()}
    contrasts = {}
    for candidate, controls in protocol['candidate_screen']['comparators'].items():
        for control in controls:
            key = candidate + '_minus_' + control
            delta = {name: dice[candidate][name]-dice[control][name] for name in protocol['sequences']}
            contrasts[key] = {'mean': statistics.mean(delta.values()),
                'ci95': sequence_bootstrap_interval(delta, seed=20261111),
                'wilcoxon': paired_wilcoxon(dice[candidate], dice[control]),
                'wins': sum(value > 1e-12 for value in delta.values()),
                'without_seq8': statistics.mean(value for name, value in delta.items() if name != 'seq8')}
    adjusted = holm_adjust({key: value['wilcoxon']['p'] for key, value in contrasts.items()})
    for key, value in contrasts.items():
        value['holm_p'] = adjusted[key]
    decisions = {}
    screen = protocol['candidate_screen']
    for candidate, controls in screen['comparators'].items():
        failed = []
        for control in controls:
            value = contrasts[candidate+'_minus_'+control]
            threshold = screen['native_practical_delta'] if control == 'native' else screen['matched_control_practical_delta']
            if value['mean'] < threshold:
                failed.append(f'mean_gain_vs_{control}')
            if value['ci95'][0] <= 0 or value['holm_p'] >= .05:
                failed.append(f'uncertainty_vs_{control}')
        if contrasts[candidate+'_minus_native']['wins'] < screen['minimum_sequence_wins_vs_native']:
            failed.append('sequence_win_count')
        if totals[candidate]['false_alarms'] > totals['native']['false_alarms']:
            failed.append('false_alarm_safety')
        if totals[candidate]['returns_with_overlap'] < totals['native']['returns_with_overlap']:
            failed.append('return_window_safety')
        decisions[candidate] = {'passes_diagnostic_screen': not failed, 'failed_criteria': failed,
            'full_propagation_result': None}
    aggregate = {}
    for method in methods:
        aggregate[method] = {metric: statistics.mean(v[metric] for v in by_method[method].values())
                             for metric in ('dice', 'iou', 'boundary_f')}
        aggregate[method].update({k: v for k, v in totals[method].items() if k != 'replay_seconds'})
        times = totals[method]['replay_seconds']
        aggregate[method]['median_treatment_replay_seconds'] = statistics.median(times) if times else None
    result = {'scope': 'Fixed-native-history immediate diagnostic, not full propagation',
              'coverage': plan['counts'], 'aggregate': aggregate, 'by_sequence': by_method,
              'contrasts': contrasts, 'decisions': decisions, 'return_windows': returns}
    (folder / 'summary.json').write_text(json.dumps(clean_json(result), indent=2)+'\n')
    lines = ['# ' + protocol.get('name', 'Broader token-matching causal diagnostic'), '',
             'All 21 eligible PolypGen sequences; fixed targets and archived native histories. These means describe the frozen diagnostic sample, not complete-video Dice.', '',
             '| Method | Dice | Boundary F | False alarms / GT-empty | Return-window overlap / events |',
             '|---|---:|---:|---:|---:|']
    for method, value in aggregate.items():
        lines.append(f"| {method} | {value['dice']:.6f} | {value['boundary_f']:.6f} | {value['false_alarms']}/{value['gt_empty_targets']} | {value['returns_with_overlap']}/{value['return_events']} |")
    lines += ['', '| Planned contrast | Δ Dice | 95% sequence CI | Holm p | Wins / 21 | Δ without seq8 |',
              '|---|---:|---|---:|---:|---:|']
    for name, value in contrasts.items():
        lines.append(f"| {name} | {value['mean']:+.6f} | [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}] | {value['holm_p']:.4f} | {value['wins']} | {value['without_seq8']:+.6f} |")
    labels = {'native': 'Native', 'tome_regional': 'ToMe regional', 'tome_global': 'ToMe generic',
              'atc_regional': 'ATC regional', 'atc_global': 'ATC generic',
              'pitome_regional': 'PiToMe regional', 'pitome_global': 'PiToMe generic'}
    lines += ['', '| Sequence | ' + ' | '.join(labels.get(m, m) for m in methods) + ' |',
              '|---|' + '---:|' * len(methods)]
    for sequence in protocol['sequences']:
        lines.append('| '+sequence+' | '+' | '.join(f'{dice[method][sequence]:.6f}' for method in methods)+' |')
    lines += ['', '## Frozen screening decision', '']
    for candidate, decision in decisions.items():
        lines.append(f"- {candidate}: {'passes diagnostic screen' if decision['passes_diagnostic_screen'] else 'stopped at diagnostic screen'}; failed criteria: {', '.join(decision['failed_criteria']) or 'none'}.")
    lines += ['', 'Return-window overlap is evaluated during the first three available GT-present frames after a return, with each read using the fixed native history. It is not a full closed-loop re-detection latency. Frames are correlated; uncertainty is computed over paired sequence means. Patient independence and external generalization are not established.', '']
    (folder / 'report.md').write_text('\n'.join(lines))
    return folder / 'report.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.folder, args.plan))
