#!/usr/bin/env python3
"""Generate technical training evidence in a label-blocked process.

No training is performed; actual execution remains disabled until the
separate worker and supervisor are reviewed and real MPS proofs exist.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import bounded_crossfit_shared_worker_3fold_v1 as worker
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA
from scripts.preflight_bounded_crossfit_training_v1 import require
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registry', 'verification', 'probe', 'probe-audit', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(); out = args.output.resolve()
    require(out.is_relative_to(BASE.resolve(strict=True)) and not out.exists(), 'Invalid output')
    guard = install_mask_open_guard(DATA); frozen = worker.sources()
    result = worker.authorize(args.registry, args.verification, args.probe, args.probe_audit)
    require(result['source_sha256'] == frozen == worker.sources() and guard['open_attempts'] == 0,
            'Sources changed or target label opened')
    result['label_guard'] = dict(guard)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'authorization.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'scope': result['scope'], 'output': str(out)}))


if __name__ == '__main__':
    main()
