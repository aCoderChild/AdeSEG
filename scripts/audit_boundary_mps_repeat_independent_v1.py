#!/usr/bin/env python3
"""Independently recheck saved, instrumented six-step MPS repeat artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import validate_output
from scripts.source_group_label_guard_v1 import install_mask_open_guard

STAGES = ('pre_model', 'prediction', 'loss', 'gradients', 'post_model', 'optimizer')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def difference(a, b):
    """Calculate exact equality and largest absolute delta without probe code."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Tensor metadata or finiteness differs')
        return bool(torch.equal(a, b)), float((a.double()-b.double()).abs().max()) if a.numel() else 0.0
    if type(a) is not type(b):
        raise ValueError('Snapshot types differ')
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('Snapshot keys differ')
        values = [difference(a[k], b[k]) for k in a]
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise ValueError('Snapshot lengths differ')
        values = [difference(x, y) for x, y in zip(a, b, strict=True)]
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Nonfinite scalar')
        return a == b, float(abs(a-b))
    else:
        if a != b:
            raise ValueError('Nonnumeric snapshot values differ')
        return True, 0.0
    return all(x[0] for x in values), max((x[1] for x in values), default=0.0)


def audit(plan_path, a_path, b_path, comparison_path):
    plan, a, b, saved = map(read, (plan_path, a_path/'receipt.json', b_path/'receipt.json', comparison_path))
    if (not plan.get('passed') or len(plan.get('sequence', [])) != 6
            or plan.get('source_sha256') != sha(ROOT/'scripts/audit_boundary_mps_repeatability_v1.py')
            or any(sha(ROOT/p) != value for p, value in plan['dependencies'].items())
            or not saved.get('passed') or saved.get('plan_sha256') != sha(plan_path)
            or saved.get('run_a_receipt_sha256') != sha(a_path/'receipt.json')
            or saved.get('run_b_receipt_sha256') != sha(b_path/'receipt.json')
            or len(saved.get('rows', [])) != 6
            or a.get('process_id') == b.get('process_id') or a.get('process_token') == b.get('process_token')):
        raise ValueError('Plan, source, comparison or process identity differs')
    for receipt in (a, b):
        if (not receipt.get('passed') or receipt.get('plan_sha256') != sha(plan_path)
                or receipt.get('source_sha256') != plan['source_sha256']
                or receipt.get('pythonhashseed') != '0' or receipt.get('environment') != plan['environment']
                or receipt.get('initial_state_sha256') != plan['initial_state_sha256']
                or receipt.get('label_guard', {}).get('blocked_open_attempts') != 0
                or len(receipt.get('steps', [])) != 6):
            raise ValueError('Run receipt identity differs')
    if [x['input_fingerprints'] for x in a['steps']] != [x['input_fingerprints'] for x in b['steps']]:
        raise ValueError('Recorded training inputs differ')
    first, rows = None, []
    for i, target in enumerate(plan['sequence']):
        snapshots = []
        for directory, receipt in ((a_path, a), (b_path, b)):
            step = receipt['steps'][i]
            if (step['index'], step['sequence'], step['frame']) != (i, target['sequence'], target['frame']):
                raise ValueError('Training step identity differs')
            path = directory/f'step_{i:02d}.pt'
            if sha(path) != step['snapshot_sha256']:
                raise ValueError('Saved tensor artifact changed')
            snapshot = torch.load(path, map_location='cpu', weights_only=True)
            if set(snapshot) != set(STAGES):
                raise ValueError('Snapshot stages differ')
            snapshots.append(snapshot)
        stages = {}
        for stage in STAGES:
            exact, delta = difference(snapshots[0][stage], snapshots[1][stage])
            expected = saved['rows'][i]['stages'][stage]
            if exact != expected['exact'] or not math.isclose(delta, expected['max_abs_difference'], rel_tol=1e-12, abs_tol=1e-20):
                raise ValueError(f'Comparison discrepancy: {i}/{stage}')
            stages[stage] = {'exact': exact, 'max_abs_difference': delta}
            if not exact and first is None:
                first = {'step': i, 'stage': stage, 'sequence': target['sequence'], 'frame': target['frame']}
        rows.append({'index': i, 'sequence': target['sequence'], 'frame': target['frame'], 'stages': stages})
    if first != saved.get('first_difference') or (first is None) != saved.get('all_stages_exact'):
        raise ValueError('First-difference summary differs')
    return {'passed': True, 'plan_sha256': sha(plan_path), 'comparison_sha256': sha(comparison_path),
            'run_a_receipt_sha256': sha(a_path/'receipt.json'), 'run_b_receipt_sha256': sha(b_path/'receipt.json'),
            'first_difference': first, 'rows': rows,
            'scope': 'Independent artifact and numeric comparison only; actual input fingerprints are compared between receipts, not independently reconstructed.',
            'limitations': ['Per-step CPU copies synchronize MPS; this is not a full ten-epoch training replay.',
                            'A difference does not identify the responsible operator.',
                            'CPython mask-open guards do not intercept arbitrary native-code I/O.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'run-a', 'run-b', 'comparison', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    guard = install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl')
    result = audit(args.plan, args.run_a, args.run_b, args.comparison)
    if guard['open_attempts']:
        raise ValueError('Protected label open attempted')
    result.update({'audit_source_sha256': sha(Path(__file__)), 'label_guard': dict(guard)})
    out = validate_output(args.output); out.mkdir(parents=True, exist_ok=False)
    receipt = out/'audit.json'; receipt.write_text(json.dumps(result, indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed': True, 'audit_sha256': sha(receipt)}, indent=2)+'\n')
    print(json.dumps({'passed': True, 'first_difference': result['first_difference']}))


if __name__ == '__main__':
    main()
