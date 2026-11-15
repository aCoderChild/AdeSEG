#!/usr/bin/env python3
"""Recheck the development gate and, if eligible, the owned MPS runtime.

No segmentation quality experiment is run. An ineligible checkpoint must be
reported as such; this test cannot promote inspected validation to a final test.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.lean_verified_window_runtime import verified_mps_window_refiner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ('preflight', 'split-preflight', 'static-capture', 'seq4-capture',
             'checkpoint', 'detector', 'baseline-reference', 'candidate-run',
             'control-run', 'data-preflight', 'runner-source', 'output')
    for name in names:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--global-run', type=Path)
    parser.add_argument('--shuffled-run', type=Path)
    args = parser.parse_args()
    try:
        with verified_mps_window_refiner(**vars(args)) as (model, refiner, lease):
            lease.validate_live(model, refiner, tensors=True)
            print(json.dumps({'eligible': True, 'runtime_verified': True,
                              'scope': 'Development engineering only; no video efficacy result.'}))
    except ValueError as exc:
        if str(exc) != 'Candidate did not pass the matched development screen':
            raise
        print(json.dumps({'eligible': False, 'runtime_verified': False,
                          'reason': 'Candidate failed the independently recomputed development screen.'}))


if __name__ == '__main__':
    main()
