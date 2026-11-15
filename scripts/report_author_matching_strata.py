"""Supplemental descriptive tables; never change the frozen eligibility screen."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output

GROUPS = ('all_present', 'all_empty', 'uniform', 'present_midpoints', 'empty_midpoints',
          'any_reappearance', 'reappearance_t0', 'reappearance_t1', 'reappearance_t2')


def belongs(target, group):
    strata = target['strata']
    return {
        'all_present': target['present'], 'all_empty': not target['present'],
        'uniform': any(value.startswith('uniform_') for value in strata),
        'present_midpoints': 'gt_present_midpoint' in strata,
        'empty_midpoints': 'gt_empty_midpoint' in strata,
        'any_reappearance': any(value.startswith('reappearance_t') for value in strata),
    }.get(group, group in strata)


def describe(records, plan):
    methods = ['native'] + plan['protocol']['conditions']
    output = {}
    for group in GROUPS:
        output[group] = {}
        for method in methods:
            sequence_dice, sequence_boundary, sequence_delta = [], [], []
            frame_delta, absent, elapsed = [], [], []
            n_targets = n_present = 0
            for sequence in plan['protocol']['sequences']:
                rows = {row['frame']: row for row in records[sequence]}
                targets = [target for target in plan['sequences'][sequence]['schedule']['targets'] if belongs(target, group)]
                if any(rows[target['frame']]['rows'][method]['present'] != target['present'] for target in targets):
                    raise ValueError('Saved method presence labels differ from frozen evaluation strata')
                positive = [target for target in targets if target['present']]
                if positive:
                    values = [rows[target['frame']]['rows'][method]['scores']['dice'] for target in positive]
                    base = [rows[target['frame']]['rows']['native']['scores']['dice'] for target in positive]
                    sequence_dice.append(statistics.mean(values))
                    sequence_boundary.append(statistics.mean(rows[target['frame']]['rows'][method]['scores']['boundary_f'] for target in positive))
                    sequence_delta.append(statistics.mean(values)-statistics.mean(base))
                    frame_delta.extend(a-b for a, b in zip(values, base))
                for target in targets:
                    value = rows[target['frame']]['rows'][method]
                    if not target['present']:
                        if type(value['absent_fp']) is not bool:
                            raise ValueError('GT-empty frame lacks boolean occupancy evidence')
                        absent.append(value['absent_fp'])
                    if method != 'native':
                        seconds = value['seconds']
                        if not math.isfinite(seconds) or seconds <= 0:
                            raise ValueError('Invalid measured replay/scoring time')
                        elapsed.append(seconds)
                n_targets += len(targets)
                n_present += len(positive)
            counts = lambda values: [sum(x > 1e-12 for x in values), sum(abs(x) <= 1e-12 for x in values), sum(x < -1e-12 for x in values)]
            output[group][method] = {'targets': n_targets, 'present_targets': n_present,
                'sequences_with_present_targets': len(sequence_dice),
                'equal_sequence_present_dice': statistics.mean(sequence_dice) if sequence_dice else None,
                'equal_sequence_present_boundary_f': statistics.mean(sequence_boundary) if sequence_boundary else None,
                'equal_sequence_delta_native': statistics.mean(sequence_delta) if sequence_delta else None,
                'present_frame_wins_ties_losses': counts(frame_delta),
                'sequence_wins_ties_losses': counts(sequence_delta),
                'false_alarms': sum(absent), 'gt_empty_targets': len(absent),
                'pooled_false_alarm_rate': sum(absent)/len(absent) if absent else None,
                'median_replay_check_scoring_seconds': statistics.median(elapsed) if elapsed else None}
    return output


def report(folder, plan_path):
    folder = validate_output(folder)
    completion = json.loads((folder / 'completion.json').read_text())
    queue = json.loads((folder.parent / 'completion.json').read_text())
    if not completion.get('passed') or completion['scope'] != 'broader_causal_diagnostic' or not queue.get('complete'):
        raise ValueError('Completed primary experiment and frozen analysis required')
    if digest(plan_path) != completion['plan_sha256'] or digest(folder / 'results.json') != completion['results_sha256']:
        raise ValueError('Plan or results changed')
    primary_sha = digest(folder / 'summary.json')
    if primary_sha != queue['summary_sha256']:
        raise ValueError('Frozen screening summary changed')
    plan = json.loads(plan_path.read_text())
    records = json.loads((folder / 'results.json').read_text())
    values = describe(records, plan)
    payload = {'scope': 'Descriptive planned strata; overlapping groups; no added tests or screening changes',
               'primary_summary_sha256': primary_sha, 'reporter_sha256': digest(Path(__file__)),
               'groups': values}
    (folder / 'descriptive_tables.json').write_text(json.dumps(payload, indent=2)+'\n')
    lines = ['# Descriptive strata and runtime', '',
             'These tables supplement the frozen primary analysis. Strata overlap; counts must not be summed across groups. Dice and Boundary F use present targets only, with equal weight per represented sequence. No additional significance tests or eligibility rules are applied.', '',
             '| Stratum | Method | Targets present/empty | Sequences with positive GT | Dice | Δ native | Frame W/T/L | FA / empty |',
             '|---|---|---:|---:|---:|---:|---|---:|']
    number = lambda value: 'N/A' if value is None else f'{value:.6f}'
    for group, methods in values.items():
        for method, value in methods.items():
            lines.append(f"| {group} | {method} | {value['present_targets']}/{value['gt_empty_targets']} | {value['sequences_with_present_targets']} | {number(value['equal_sequence_present_dice'])} | {number(value['equal_sequence_delta_native'])} | {'/'.join(map(str,value['present_frame_wins_ties_losses']))} | {value['false_alarms']}/{value['gt_empty_targets']} |")
    lines += ['', '| Method | Median replay/check/scoring seconds, present targets | Median, empty targets |', '|---|---:|---:|']
    for method in values['all_present']:
        lines.append(f"| {method} | {number(values['all_present'][method]['median_replay_check_scoring_seconds'])} | {number(values['all_empty'][method]['median_replay_check_scoring_seconds'])} |")
    lines += ['', 'The timer includes treatment replay, integrity checks and scoring. Native replay was not timed by the same measurement, so this table does not establish attention speedup or end-to-end throughput. Frame wins are correlated observations. Empty-target Dice is intentionally omitted.', '']
    (folder / 'descriptive_tables.md').write_text('\n'.join(lines))
    if digest(folder / 'summary.json') != primary_sha:
        raise RuntimeError('Primary screening summary changed during descriptive reporting')
    return folder / 'descriptive_tables.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(report(args.folder, args.plan))
