#!/usr/bin/env python3
"""Independently check cohort records against frozen data and gate results."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.multiscale_memory_utility import evaluate_scores
from scripts.memory_token_io import digest
from scripts.run_multiscale_memory_utility_cohort import PROTOCOL, PILOT, summarize, verify_setup


def audit(out, gate):
    protocol = json.loads(PROTOCOL.read_text())
    verify_setup(protocol)
    completion = json.loads((out / 'completion.json').read_text())
    if not completion['complete'] or digest(out / 'setup.json') != completion['setup_sha256'] or digest(out / 'results.json') != completion['results_sha256']:
        raise RuntimeError('Cohort completion hashes differ')
    setup = json.loads((out / 'setup.json').read_text())
    for path, expected in setup['sources'].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f'Frozen cohort source changed: {path}')
    if setup['protocol_sha256'] != digest(PROTOCOL) or setup['sequences'] != list(protocol['plan']):
        raise RuntimeError('Full frozen cohort or protocol differs')
    report = json.loads((out / 'results.json').read_text())
    rows = report['rows']
    if len(rows) != protocol['target_count'] or report['efficacy_evaluated']:
        raise RuntimeError('Incomplete or mislabelled cohort')
    by_key = {(r['sequence'], r['target_frame']): r for r in rows}
    if len(by_key) != len(rows):
        raise RuntimeError('Duplicate sequence/target')
    for name, plan in protocol['plan'].items():
        path = Path(protocol['utility_root']) / 'sequences' / name / 'results.json'
        if digest(path) != plan['utility_results_sha256']:
            raise RuntimeError(f'Archived utility changed: {name}')
        original = json.loads(path.read_text())['main']
        for item in plan['targets']:
            row = by_key[(name, item['target_frame'])]
            if any(row[key] != value for key, value in item.items()):
                raise RuntimeError('Native target identity/hash/source IDs differ')
            if row['pilot_pair'] != ((name, item['source_frame'], item['target_frame']) in PILOT):
                raise RuntimeError('Pilot separation changed')
            target = next(r for r in original if r['anchor_target'] == item['target_frame'] and r['source_frame'] == item['source_frame'])
            native = target['arms']['native']
            if not native['present'] or native['prediction_sha256'] != item['native_prediction_sha256']:
                raise RuntimeError('Archived target mismatch')
            utility = [float(native['scores']['dice']) - float(target['arms'][f'tile_{i:02d}']['scores']['dice']) for i in range(16)]
            if not np.array_equal(row['tile_removal_utility'], utility):
                raise RuntimeError('Tile utility differs from pinned receipts')
            for method in ('scale_1', 'multiscale', 'native_attention'):
                if row['metrics'][method] != evaluate_scores(row['scores'][method], utility):
                    raise RuntimeError('Tile selection or correlation arithmetic mismatch')
    if summarize(rows) != report['summary']:
        raise RuntimeError('Summary statistics differ')
    gate_completion = json.loads((gate / 'completion.json').read_text())
    if digest(gate / 'results.json') != gate_completion['results_sha256'] or digest(gate / 'setup.json') != gate_completion['setup_sha256']:
        raise RuntimeError('Prior gate results changed')
    gate_setup = json.loads((gate / 'setup.json').read_text())
    if gate_setup['sources'] != setup['sources'] or gate_setup['sequences'] != ['seq2', 'seq3']:
        raise RuntimeError('Gate did not exercise the same code')
    gate_rows = json.loads((gate / 'results.json').read_text())['rows']
    if any(by_key[(r['sequence'], r['target_frame'])] != r for r in gate_rows):
        raise RuntimeError('Cohort outputs differ from independent real-data gate')
    record = {'passed': True, 'scope': 'Independent arithmetic/provenance audit; no independent inference on all 42 targets.',
              'sequences': len(protocol['plan']), 'present_target_pairs': len(rows), 'tile_values_checked': len(rows) * 16,
              'exact_gate_replay_pairs': len(gate_rows), 'pilot_pairs': sum(r['pilot_pair'] for r in rows),
              'results_sha256': digest(out / 'results.json'), 'protocol_sha256': digest(PROTOCOL)}
    dst = out / 'independent_audit.json'
    if dst.exists():
        raise FileExistsError(dst)
    dst.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gate', type=Path, required=True)
    args = p.parse_args()
    audit(args.output_dir, args.gate)


if __name__ == '__main__':
    main()
