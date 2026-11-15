#!/usr/bin/env python3
"""Combine fully audited disjoint automatic Lean cohort partitions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part', action='append', required=True, type=Path)
    p.add_argument('--historical', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if len(args.part) != 2:
        raise ValueError('Expected exactly two partitions')
    configs = [json.loads((part / 'setup.json').read_text()) for part in args.part]
    if {k: v for k, v in configs[0].items() if k != 'sequences'} != {k: v for k, v in configs[1].items() if k != 'sequences'}:
        raise ValueError('Partition configurations differ')
    historical_audit = json.loads((args.historical / 'independent_mask_audit.json').read_text())
    if not historical_audit['passed'] or historical_audit['comparison_sha256'] != digest(args.historical / 'comparison.json'):
        raise ValueError('Historical reference not independently audited')
    history = json.loads((args.historical / 'comparison.json').read_text())
    names = [f'seq{i}' for i in range(2, 24) if i != 7]
    summaries, locations = {}, {}
    for part, config in zip(args.part, configs):
        done = json.loads((part / 'completion.json').read_text())
        if not done['passed'] or digest(part / 'results.json') != done['results_sha256']:
            raise ValueError('Partition incomplete or changed')
        result = json.loads((part / 'results.json').read_text())['sequences']
        if set(result) != set(config['sequences']) or set(result) & set(summaries):
            raise ValueError('Partition coverage mismatch or overlap')
        for name, item in result.items():
            run = part / name
            setup = json.loads((run / 'setup.json').read_text())
            receipt = json.loads((run / 'completion.json').read_text())
            audit = json.loads((run / 'independent_audit.json').read_text())
            if (setup['signature'] != config['signature'] or not receipt['passed'] or not audit['passed']
                or digest(run / 'records.json') != receipt['records_sha256']
                or audit['records_sha256'] != receipt['records_sha256'] or audit['sequences'][name] != item):
                raise ValueError(f'Sequence provenance mismatch: {name}')
            summaries[name] = item
            locations[name] = str(run.resolve())
    if set(summaries) != set(names):
        raise ValueError('Cohort coverage incomplete')
    rows = [summaries[name] for name in names]
    if sum(x['historical_present'] for x in rows) != history['target_frames'] - history['empty_frames'] or sum(x['historical_empty'] for x in rows) != history['empty_frames']:
        raise ValueError('Historical frame denominators differ')
    result = {
        'scope': 'Automatic combined Lean on all frames; historical comparison uses same post-prompt frame IDs but a GT mask initializes the previous native and Lean runs.',
        'sequences': {name: summaries[name] for name in names},
        'all_frames': sum(x['frames'] for x in rows),
        'all_present_frames': sum(x['present'] for x in rows),
        'all_empty_frames': sum(x['empty'] for x in rows),
        'all_present_dice_equal_sequence': float(np.mean([x['all_present_dice'] for x in rows if x['all_present_dice'] is not None])),
        'all_false_alarms': sum(x['all_false_alarms'] for x in rows),
        'labeled_present_before_detection': sum(x['present_before_first_detection'] for x in rows),
        'first_detection_labeled_empty_sequences': sum(x['first_detection_labeled_present'] is False for x in rows),
        'no_detection_sequences': [name for name in names if summaries[name]['first_detection_index'] is None],
        'point_status_counts': dict(Counter(x['point_status'] for x in rows)),
        'mask_status_counts': dict(Counter(x['lean_mask_status'] for x in rows)),
        'sequences_with_memory_events': sum(x['memory_events'] > 0 for x in rows),
        'memory_events': sum(x['memory_events'] for x in rows),
        'selection_events': sum(x['selection_events'] for x in rows),
        'historical_matched_frames': history['target_frames'],
        'historical_matched_present': sum(x['historical_present'] for x in rows),
        'historical_matched_empty': history['empty_frames'],
        'historical_native_dice': history['native_dice'],
        'historical_lean_dice': history['lean_dice'],
        'automatic_combined_historical_dice': float(np.mean([x['historical_present_dice'] for x in rows])),
        'historical_native_false_alarms': history['native_false_alarms'],
        'historical_lean_false_alarms': history['lean_false_alarms'],
        'automatic_combined_historical_false_alarms': sum(x['historical_false_alarms'] for x in rows)
    }
    out = validate_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'locations.json').write_text(json.dumps(locations, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'results_sha256': digest(out / 'results.json'),
        'part_results_sha256': [digest(part / 'results.json') for part in args.part],
        'historical_comparison_sha256': digest(args.historical / 'comparison.json')}, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sequences'}, indent=2))


if __name__ == '__main__':
    main()
