#!/usr/bin/env python3
"""Fresh fold worker for a separately supervised mixed-source study.

All-label verification must already have run in another process. This
worker installs a fold-scoped guard, rechecks the technical authorization,
and remains hard-disabled in the shared worker pending independent review.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import bounded_crossfit_shared_worker_v1 as worker
from scripts.preflight_bounded_crossfit_training_v1 import read_bound_json, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization', required=True, type=Path)
    parser.add_argument('--batch-root', required=True, type=Path)
    parser.add_argument('--fold', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache-mib', type=int, default=1024)
    args = parser.parse_args()
    path = worker.path_under_drive(args.authorization)
    saved, sha = read_bound_json(path)
    require(saved.get('label_guard', {}).get('open_attempts') == 0, 'Authorization label guard differs')
    authorization = {k: v for k, v in saved.items() if k != 'label_guard'}
    require(sha == worker.digest(path), 'Authorization changed')
    worker.train_fold(authorization, args.fold, args.output, batch_root=args.batch_root, cache_mib=args.cache_mib)
    print(json.dumps({'passed': True, 'fold': args.fold, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
