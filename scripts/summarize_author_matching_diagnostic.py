"""Describe the fixed-history comparison without promoting it to efficacy."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output


def summarize(folder):
    folder = validate_output(folder)
    complete = json.loads((folder / 'completion.json').read_text())
    if not complete.get('engineering_passed') or digest(folder / 'results.json') != complete['results_sha256']:
        raise ValueError('Missing or modified completion evidence')
    setup = json.loads((folder / 'setup.json').read_text())
    records = json.loads((folder / 'results.json').read_text())
    methods = ['native'] + setup['protocol']['conditions']
    summary = {}
    for method in methods:
        by_sequence = {}
        deltas, seconds, absent = [], [], []
        for name, targets in records.items():
            present = [r for r in targets if r['rows']['native']['present']]
            if not present:
                raise ValueError('A sequence has no GT-present diagnostic targets')
            by_sequence[name] = {metric: statistics.mean(r['rows'][method]['scores'][metric] for r in present)
                                 for metric in ('dice', 'boundary_f')}
            for row in targets:
                if not all(row[k] for k in ('native_parity', 'bank_unchanged', 'native_restored')):
                    raise ValueError('Read integrity failed')
                value = row['rows'][method]
                if value['present']:
                    deltas.append(value['scores']['dice'] - row['rows']['native']['scores']['dice'])
                else:
                    absent.append(bool(value['absent_fp']))
                if 'seconds' in value:
                    seconds.append(value['seconds'])
        summary[method] = {'by_sequence': by_sequence,
            'equal_sequence_dice': statistics.mean(v['dice'] for v in by_sequence.values()),
            'equal_sequence_boundary_f': statistics.mean(v['boundary_f'] for v in by_sequence.values()),
            'target_wins_ties_losses': [sum(d > 1e-12 for d in deltas), sum(abs(d) <= 1e-12 for d in deltas), sum(d < -1e-12 for d in deltas)],
            'false_alarms': sum(absent), 'gt_empty_targets': len(absent),
            'median_treatment_replay_seconds': statistics.median(seconds) if seconds else None}
    baseline = summary['native']['equal_sequence_dice']
    lines = ['# ToMe / ATC matched-budget diagnostic', '',
             '**Scope:** immediate interventions on fixed native histories; selected seq2/seq3 targets. These are real segmentation Dice scores on those targets, not full-sequence or full-PolypGen results.', '',
             '| Method | seq2 Dice | seq3 Dice | Equal-sequence Dice | Δ native | Target W/T/L |',
             '|---|---:|---:|---:|---:|---|']
    for method, result in summary.items():
        values = result['by_sequence']
        lines.append(f"| {method} | {values['seq2']['dice']:.6f} | {values['seq3']['dice']:.6f} | {result['equal_sequence_dice']:.6f} | {result['equal_sequence_dice']-baseline:+.6f} | {'/'.join(map(str,result['target_wins_ties_losses']))} |")
    lines += ['', 'All four conditions use equal per-frame/per-window token budgets, identical native histories and the same pretrained SAM2 weights. Native replay and post-intervention restoration passed; the native memory bank was unchanged.',
              '', 'Native object-presence handling, mask selection and direct low-resolution-to-video interpolation are retained. ATC changes matching to upstream average-linkage clustering; the weighted aggregation and insertion point are otherwise shared.',
              '', 'Frame wins are correlated observations, not independent replications. These selected targets cannot establish significance, full-video temporal stability, improvement over archived Run155, or generalization. The GT-empty denominator is recorded in summary.json; zero absent targets provides no false-alarm safety evidence.', '']
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (folder / 'report.md').write_text('\n'.join(lines))
    return folder / 'report.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    print(summarize(parser.parse_args().folder))
