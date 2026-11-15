#!/usr/bin/env python3
"""Merge two completed, disjoint, identically configured SASVi cohorts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part', type=Path, required=True, action='append')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.part) != 2:
        raise ValueError('Expected two cohort partitions')
    setups = [json.loads((part / 'setup.json').read_text()) for part in args.part]
    signatures = [{k: v for k, v in setup.items() if k != 'sequences'} for setup in setups]
    if signatures[0] != signatures[1]:
        raise ValueError('Partition configurations differ')
    values, locations = {}, {}
    for part, setup in zip(args.part, setups):
        receipt = json.loads((part / 'completion.json').read_text())
        if not receipt['passed'] or digest(part / 'results.json') != receipt['results_sha256']:
            raise ValueError('Partition incomplete or changed')
        result = json.loads((part / 'results.json').read_text())
        if set(values) & set(result['per_sequence']):
            raise ValueError('Overlapping sequences')
        for name, row in result['per_sequence'].items():
            run = part / name
            run_receipt = json.loads((run / 'completion.json').read_text())
            audit = json.loads((run / 'audit_results.json').read_text())
            run_setup = json.loads((run / 'setup.json').read_text())
            if (run_setup['sam2_checkpoint_sha256'] != setup['checkpoint_sha256'] or
                run_setup['runner_sha256'] != setup['runner_sha256'] or
                run_setup['cmsa_setup_sha256'] != setup['cmsa_setup_sha256'] or
                run_setup['cmsa_frames_sha256'] != setup['cmsa_frames_sha256'] or
                not setup['device'].startswith(run_setup['device'])):
                raise ValueError(f'Inconsistent sequence configuration: {name}')
            if not run_receipt['passed'] or digest(run / 'records.json') != run_receipt['records_sha256'] or {k: v for k, v in audit.items() if k != 'frames'} != row:
                raise ValueError(f'Sequence receipt mismatch: {name}')
            values[name], locations[name] = row, str(run.resolve())
    names = [f'seq{i}' for i in range(2, 24) if i != 7]
    if set(values) != set(names):
        raise ValueError('Partitions do not cover the full cohort')
    result = {'sequences': len(values), 'cmsa_equal_sequence_dice': float(np.mean([values[n]['cmsa_present_dice'] for n in names])),
              'sasvi_equal_sequence_dice': float(np.mean([values[n]['sasvi_present_dice'] for n in names])),
              'empty_targets': sum(values[n]['empty'] for n in names),
              'cmsa_false_alarms': sum(values[n]['cmsa_false_alarms'] for n in names),
              'sasvi_false_alarms': sum(values[n]['sasvi_false_alarms'] for n in names),
              'per_sequence': {name: values[name] for name in names}}
    args.output.mkdir(parents=True)
    (args.output / 'setup.json').write_text(json.dumps({**signatures[0], 'sequences': names}, indent=2) + '\n')
    (args.output / 'locations.json').write_text(json.dumps(locations, indent=2) + '\n')
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'completion.json').write_text(json.dumps({'passed': True, 'results_sha256': digest(args.output / 'results.json'),
        'partition_results_sha256': [digest(part / 'results.json') for part in args.part]}, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_sequence'}, indent=2))


if __name__ == '__main__':
    main()
