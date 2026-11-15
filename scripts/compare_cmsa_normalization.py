#!/usr/bin/env python3
"""Compare the same CMSA checkpoint with released and channel normalization."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.statistics import sequence_bootstrap_interval
from scripts.run_cmsa_polypgen import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--released', type=Path, required=True)
    p.add_argument('--channel', type=Path, required=True)
    args = p.parse_args()
    folders = [args.released, args.channel]
    setups = [json.loads((path / 'setup.json').read_text()) for path in folders]
    results = [json.loads((path / 'results.json').read_text()) for path in folders]
    audits = [json.loads((path / 'matched_native_audit.json').read_text()) for path in folders]
    frames = [json.loads((path / 'frames.json').read_text()) for path in folders]
    for key in ('author_revision', 'checkpoint_sha256', 'manifest_sha256', 'sequences', 'clip_length', 'probability_threshold', 'device'):
        if setups[0][key] != setups[1][key]:
            raise ValueError(f'Comparison setup mismatch: {key}')
    if setups[0].get('normalization', 'released') != 'released' or setups[1].get('normalization') != 'channel':
        raise ValueError('Wrong normalization conditions')
    if any(a['status'] != 'passed' for a in audits) or audits[0]['native_results_sha256'] != audits[1]['native_results_sha256']:
        raise ValueError('Independent native audits do not match')
    if len(frames[0]) != len(frames[1]):
        raise ValueError('Frame coverage mismatch')
    for old, new in zip(*frames):
        if any(old[k] != new[k] for k in ('sequence', 'frame', 'present')):
            raise ValueError('Frame identity mismatch')
    differences = {}
    table = {}
    for name in setups[0]['sequences']:
        before = results[0]['per_sequence'][name]
        after = results[1]['per_sequence'][name]
        if any(before[key] != after[key] for key in ('prompt_index', 'matched_targets', 'present_targets', 'gt_empty_targets')):
            raise ValueError(f'Target coverage mismatch for {name}')
        delta = after['mean_present_dice'] - before['mean_present_dice']
        differences[name] = delta
        table[name] = {'released_dice': before['mean_present_dice'], 'channel_dice': after['mean_present_dice'],
                       'difference': delta, 'released_false_alarms': before['false_alarms'], 'channel_false_alarms': after['false_alarms']}
    report = {'status': 'passed', 'setup_hashes': [sha(path / 'setup.json') for path in folders],
              'matched_targets': audits[0]['matched_targets'], 'empty_targets': audits[0]['empty_targets'],
              'released_dice': results[0]['equal_sequence_mean_present_dice'],
              'channel_dice': results[1]['equal_sequence_mean_present_dice'],
              'mean_difference': float(np.mean(list(differences.values()))),
              'sequence_bootstrap_95_ci': sequence_bootstrap_interval(differences),
              'sequence_wins': sum(delta > 0 for delta in differences.values()),
              'sequence_ties': sum(delta == 0 for delta in differences.values()),
              'sequence_losses': sum(delta < 0 for delta in differences.values()),
              'released_false_alarms': audits[0]['cmsa_false_alarms'], 'channel_false_alarms': audits[1]['cmsa_false_alarms'],
              'per_sequence': table,
              'interpretation': 'Exploratory matched comparison. The checkpoint may have been trained using the released column-wise normalization; corrected inference is a distribution change, not a guaranteed model correction.'}
    output = args.channel / 'normalization_comparison.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
