#!/usr/bin/env python3
"""Verify independent Lean 25%-versus-100% engineering receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def read(path): return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('noop-a', 'noop-b', 'dense-a', 'dense-b', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    groups = []
    for mode, paths in [('replay', (a.noop_a, a.noop_b)), ('ablation', (a.dense_a, a.dense_b))]:
        entries = []
        for path in paths:
            setup, done, record = read(path / 'setup.json'), read(path / 'completion.json'), read(path / 'records.json')
            integrity = read(path / 'parameter_integrity_after.json')
            if (setup['mode'] != mode or setup['sequence'] != 'seq3' or not done['passed'] or
                digest(path / 'records.json') != done['records_sha256'] or not integrity['unchanged'] or
                digest(path / 'memory_reads.json') != record['memory_reads_sha256'] or
                read(path / 'selections.json') != record['selections']):
                raise ValueError(f'{path}: engineering receipt invalid')
            if mode == 'ablation' and (not setup['engineering'] or record['keep_ratio'] != 1.0):
                raise ValueError('Dense run is not the designated fixed-100% check')
            if mode == 'replay' and record['keep_ratio'] != 0.25:
                raise ValueError('No-op ratio differs')
            entries.append((setup, record, read(path / 'memory_reads.json')))
        if entries[0] != entries[1]: raise ValueError(f'{mode}: independent processes differ')
        groups.append(entries[0])
    baseline, dense = groups
    if baseline[0]['gate_identity'] != dense[0]['gate_identity']:
        raise ValueError('No-op and dense source identities differ')
    for event in dense[1]['selections']:
        count = event['candidates']
        expected_hash = hashlib.sha256(np.arange(count, dtype=np.int64).tobytes()).hexdigest()
        if event['kept'] != count or event['indices_sha256'] != expected_hash:
            raise ValueError('Dense token indices do not contain every spatial token')
    old_reads, new_reads = baseline[2], dense[2]
    if (len(old_reads) != len(new_reads) or old_reads[:2] != new_reads[:2] or
        [r['pointer_tokens'] for r in old_reads] != [r['pointer_tokens'] for r in new_reads] or
        max(r['memory_tokens'] for r in new_reads) <= max(r['memory_tokens'] for r in old_reads)):
        raise ValueError('Attention input or pointer counts do not satisfy the control design')
    out = validate_output(a.output)
    if out.exists(): raise FileExistsError(out)
    receipt = {'passed': True, 'scope': 'seq3 engineering verification; fixed retention change, no efficacy score',
               'noop_records_sha256': digest(a.noop_a / 'records.json'), 'dense_records_sha256': digest(a.dense_a / 'records.json'),
               'noop_memory_reads_sha256': digest(a.noop_a / 'memory_reads.json'),
               'dense_memory_reads_sha256': digest(a.dense_a / 'memory_reads.json'),
               'selection_events': len(dense[1]['selections']), 'all_candidate_indices_present': True,
               'order_note': 'The receipt hashes sorted indices, so order is not independently measured; the inspected author full-budget branch constructs arange indices.',
               'memory_and_position_shapes_checked': True, 'pointer_counts_match_seq3': True,
               'noop_peak_attention_tokens': max(r['memory_tokens'] for r in old_reads),
               'dense_peak_attention_tokens': max(r['memory_tokens'] for r in new_reads),
               'source_sha256': digest(Path(__file__))}
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
