"""Verify exact fresh-process replay of the canonical-timing matching diagnostic."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output


def verify(first, second, output, protocol_path=None):
    protocol_path = protocol_path or ROOT / 'protocols/author_matching_diagnostic_v2.json'
    setups = []
    records = []
    for folder in (first, second):
        done = json.loads((folder / 'completion.json').read_text())
        if not done.get('engineering_passed') or digest(folder / 'results.json') != done['results_sha256']:
            raise ValueError('Missing or changed completion evidence')
        setup = json.loads((folder / 'setup.json').read_text())
        if setup['protocol_sha256'] != digest(protocol_path):
            raise ValueError('Replay protocol differs from the requested frozen protocol')
        setups.append(setup)
        records.append(json.loads((folder / 'results.json').read_text()))
    if setups[0] != setups[1]:
        raise ValueError('Replay source or protocol differs')
    first_records, second_records = records
    if set(first_records) != set(second_records):
        raise ValueError('Replay sequence coverage differs')
    tensors, targets = 0, 0
    for sequence, rows in first_records.items():
        other_rows = second_records[sequence]
        if len(rows) != len(other_rows):
            raise ValueError('Replay target coverage differs')
        for row, other in zip(rows, other_rows):
            if row['frame'] != other['frame'] or row['attention_audit'] != other['attention_audit']:
                raise ValueError('Replay frame or attention budget differs')
            if set(row['rows']) != set(other['rows']):
                raise ValueError('Replay condition coverage differs')
            for condition, values in row['rows'].items():
                current = {k: v for k, v in values.items() if k != 'seconds'}
                expected = {k: v for k, v in other['rows'][condition].items() if k != 'seconds'}
                if current != expected:
                    raise ValueError(f'Replay metric/hash mismatch: {sequence}/{row["frame"]}/{condition}')
                if condition == 'native':
                    continue
                filename = f'{row["frame"]}_{condition}.pt'
                left = torch.load(first / sequence / filename, map_location='cpu', weights_only=True)
                right = torch.load(second / sequence / filename, map_location='cpu', weights_only=True)
                if set(left) != set(right):
                    raise ValueError('Replay artifact keys differ')
                for key in left:
                    if not torch.equal(left[key], right[key]):
                        raise ValueError(f'Replay tensor differs: {sequence}/{filename}/{key}')
                    tensors += 1
            targets += 1
    receipt = {'passed': True, 'canonical_full_video_timing': True,
               'targets': targets, 'treatment_tensors_bit_identical': tensors,
               'native_prediction_hashes_metrics_and_budgets_identical': True,
               'first': str(first), 'fresh_process': str(second),
               'protocol_sha256': setups[0]['protocol_sha256'], 'sources': setups[0]['sources']}
    validate_output(output).write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.first, args.second, args.output, args.protocol), indent=2))
