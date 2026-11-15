#!/usr/bin/env python3
"""Independently audit repeat runs and archived utility alignment."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.multiscale_memory_utility import evaluate_scores
from scripts.memory_token_io import digest
from scripts.run_multiscale_memory_utility import PROTOCOL, checked_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--repeat', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    _, cases = checked_inputs(protocol)
    all_results = []
    for folder in (args.first, args.repeat):
        completion = json.loads((folder / 'completion.json').read_text())
        if not completion['complete'] or digest(folder / 'results.json') != completion['results_sha256'] or digest(folder / 'setup.json') != completion['setup_sha256']:
            raise RuntimeError(f'Incomplete or modified result: {folder}')
        setup = json.loads((folder / 'setup.json').read_text())
        if setup['protocol_sha256'] != digest(PROTOCOL) or setup['code_sha256'] != digest(ROOT / 'scripts/run_multiscale_memory_utility.py') or setup['descriptor_sha256'] != digest(ROOT / 'modeling/multiscale_memory_utility.py'):
            raise RuntimeError('Frozen protocol or code differs')
        report = json.loads((folder / 'results.json').read_text())
        if len(report['pairs']) != len(cases) or report['efficacy_evaluated']:
            raise RuntimeError('Unexpected pair coverage or claim')
        for row, (pair, receipt, utility) in zip(report['pairs'], cases, strict=True):
            if (row['sequence'], row['source_frame'], row['target_frame']) != (pair['sequence'], pair['source_frame'], pair['target_frame']):
                raise RuntimeError('Pair identity changed')
            if row['native_prediction_sha256'] != pair['expected_native_prediction_sha256'] or not np.array_equal(row['tile_removal_utility'], utility):
                raise RuntimeError('Native output or utility differs')
            for name in ('scale_1', 'multiscale', 'native_attention'):
                if evaluate_scores(row['scores'][name], utility) != row['metrics'][name]:
                    raise RuntimeError('Descriptor scoring arithmetic differs')
        all_results.append(report)
    first, repeat = all_results
    for a, b in zip(first['pairs'], repeat['pairs'], strict=True):
        for key in ('sequence', 'source_frame', 'target_frame', 'native_prediction_sha256', 'source_span', 'scores', 'tile_removal_utility', 'metrics'):
            if a[key] != b[key]:
                raise RuntimeError(f'Independent replay mismatch: {a["sequence"]}, {key}')
    if first['equal_pair_summary'] != repeat['equal_pair_summary']:
        raise RuntimeError('Repeat summary differs')
    audit = {'passed': True, 'pairs': len(cases), 'independent_process_scores_identical': True,
             'scope': 'Four post-hoc selected reads only. Fixed-history diagnostic; no efficacy or generalization result.',
             'input_receipts_reverified': True, 'metrics_recomputed': True,
             'first_results_sha256': digest(args.first / 'results.json'),
             'repeat_results_sha256': digest(args.repeat / 'results.json'),
             'note': 'Elapsed wall times differ and are excluded from equality.'}
    dest = args.first / 'independent_audit.json'
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
