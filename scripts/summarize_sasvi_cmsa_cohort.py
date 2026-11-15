#!/usr/bin/env python3
"""Summarize completed independently audited SASVi/CMSA sequence runs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.statistics import sequence_bootstrap_interval  # noqa: E402
from scripts.memory_token_io import digest  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    root = args.run
    setup = json.loads((root / 'setup.json').read_text())
    done = json.loads((root / 'completion.json').read_text())
    if not done['passed'] or digest(root / 'results.json') != done['results_sha256']:
        raise ValueError('Cohort is incomplete or changed')
    result = json.loads((root / 'results.json').read_text())
    locations = json.loads((root / 'locations.json').read_text()) if (root / 'locations.json').exists() else {}
    values = {}
    for name in setup['sequences']:
        run = Path(locations[name]) if locations else root / name
        receipt = json.loads((run / 'completion.json').read_text())
        if not receipt['passed'] or digest(run / 'records.json') != receipt['records_sha256']:
            raise ValueError(f'Changed sequence {name}')
        audit = json.loads((run / 'audit_results.json').read_text())
        compact = {k: v for k, v in audit.items() if k != 'frames'}
        if compact != result['per_sequence'][name]:
            raise ValueError(f'Changed audit {name}')
        values[name] = compact
    delta = {name: v['sasvi_present_dice'] - v['cmsa_present_dice'] for name, v in values.items()}
    analysis = {'method': 'SASVi/CMSA adaptation minus its CMSA overseer; exploratory sequence-level analysis',
                'results_sha256': done['results_sha256'], 'paired_mean_difference': float(np.mean(list(delta.values()))),
                'sequence_bootstrap_interval': sequence_bootstrap_interval(delta),
                'wins': sum(v > 1e-12 for v in delta.values()), 'losses': sum(v < -1e-12 for v in delta.values()),
                'ties': sum(abs(v) <= 1e-12 for v in delta.values()), 'sequence_differences': delta}
    (root / 'cohort_analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
