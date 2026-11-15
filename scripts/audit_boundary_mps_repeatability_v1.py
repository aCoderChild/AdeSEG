#!/usr/bin/env python3
"""Six-update same-implementation MPS repeat probe on training reads only.

Run ``plan`` once, ``run`` twice in separate fresh processes, then ``compare``.
All outputs are new Drive artifacts. This is a short diagnostic, not a replay
of the ten-epoch study or evidence of reproducibility on every input.
"""
import argparse
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss
from scripts.memory_token_io import digest, validate_output
from scripts.source_group_boundary_shared_v1 import dependencies, ensure_target, initial_hash, load_prepared, model_for
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_training_label_guard_v1 import install_training_mask_guard
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture, to_video
from scripts.train_region_mixed_loss_v1 import environment_identity

SEED = 20261115
KEYS = (('seq2', 4), ('seq4', 18), ('seq4', 24)) * 2
STAGES = ('pre_model', 'prediction', 'loss', 'gradients', 'post_model', 'optimizer')


def fingerprint(value):
    """Hash type, structure, tensor geometry and raw bytes, not zip metadata."""
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(b'tensor' + json.dumps([str(t.dtype), list(t.shape)]).encode() + b'\0')
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            h.update(b'dict\0')
            for k in sorted(x, key=lambda v: (type(v).__name__, str(v))):
                visit(k); visit(x[k])
            h.update(b'end\0')
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode() + b'\0')
            for v in x:
                visit(v)
            h.update(b'end\0')
        elif isinstance(x, float):
            if not math.isfinite(x):
                raise ValueError('Nonfinite scalar')
            h.update(b'float' + struct.pack('!d', x))
        elif x is None or isinstance(x, (bool, int, str)):
            h.update(type(x).__name__.encode() + b':' + str(x).encode() + b'\0')
        else:
            raise TypeError(f'Unsupported snapshot type: {type(x)}')
    visit(value)
    return h.hexdigest()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def max_difference(a, b):
    """Largest absolute numerical delta; structure mismatches fail closed."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('Tensor metadata differs')
        if a.numel() == 0:
            return 0.0
        return float((a.double() - b.double()).abs().max())
    if type(a) is not type(b):
        raise ValueError('Snapshot type differs')
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('Snapshot keys differ')
        return max((max_difference(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError('Snapshot length differs')
        return max((max_difference(x, y) for x, y in zip(a, b, strict=True)), default=0.0)
    if isinstance(a, (float, int)) and not isinstance(a, bool):
        return abs(a - b)
    if a != b:
        raise ValueError('Snapshot nonnumeric value differs')
    return 0.0


def selected_rows(manifest):
    rows = []
    for seq, frame in KEYS:
        matching = [r for r in manifest['samples'] if r['split'] == 'train' and r['sequence'] == seq and r['frame'] == frame]
        if len(matching) != 1:
            raise ValueError('Training probe missing or duplicated')
        rows.append(matching[0])
    return rows


def build_plan(manifest, manifest_sha):
    torch.manual_seed(SEED)
    initial = initial_hash(model_for('region_boundary'))
    return {'passed': True, 'scope': 'Short same-implementation train-only repeat, not a full-study reproducibility claim.',
            'manifest_sha256': manifest_sha, 'feature_pipeline_sha256': manifest['feature_pipeline_sha256'],
            'environment': manifest['feature_pipeline']['runtime_environment'], 'source_sha256': digest(Path(__file__)),
            'dependencies': dependencies(), 'pythonhashseed': '0', 'seed': SEED, 'initial_state_sha256': initial,
            'sequence': [{'sequence': r['sequence'], 'frame': r['frame'], 'image_sha256': r['image_sha256'],
                          'feature_sha256': r['feature_sha256'], 'snapshot_sha256': r['snapshot_sha256'],
                          'label_sha256': r['label_sha256']} for r in selected_rows(manifest)],
            'optimizer': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001,
            'objective': 'corrected CMSA boundary loss + project soft Dice'}


def require_plan(path, manifest, manifest_sha):
    saved = json.loads(Path(path).read_text())
    if saved != build_plan(manifest, manifest_sha):
        raise ValueError('Frozen plan, training inputs or sources changed')
    return saved


def run(manifest, manifest_sha, plan_path, output, guard):
    plan = require_plan(plan_path, manifest, manifest_sha)
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Fresh process must start with PYTHONHASHSEED=0')
    if environment_identity() != plan['environment']:
        raise ValueError('Live MPS environment differs from capture')
    out = validate_output(output); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED); torch.mps.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    model = model_for('region_boundary').train()
    if initial_hash(model) != plan['initial_state_sha256']:
        raise ValueError('Initial parameters differ')
    model.to('mps')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
    steps = []
    for index, row in enumerate(selected_rows(manifest)):
        early, deep, logits, shape = read_capture(row)
        target = ensure_target(row, shape)
        input_fingerprints = {name: fingerprint(value) for name, value in
                              (('early', early), ('deep', deep), ('logits', logits), ('target', target))}
        target = target.to('mps')
        args = (early.to('mps'), deep.to('mps'), logits.to('mps'))
        optimizer.zero_grad(set_to_none=True)
        pre = cpu_copy(model.state_dict())
        prediction = model(*args)
        loss = cmsa_boundary_loss(to_video(prediction, shape), target)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        grads = {name: cpu_copy(p.grad) for name, p in model.named_parameters()}
        if any(g is None or not torch.isfinite(g).all() for g in grads.values()):
            raise ValueError('Missing or nonfinite gradients')
        snapshot = {'pre_model': pre, 'prediction': cpu_copy(prediction), 'loss': float(loss.detach().cpu()),
                    'gradients': grads}
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('Nonfinite parameter')
        snapshot['post_model'] = cpu_copy(model.state_dict())
        snapshot['optimizer'] = cpu_copy(optimizer.state_dict())
        snapshot_file = out/f'step_{index:02d}.pt'
        torch.save(snapshot, snapshot_file)
        steps.append({'index': index, 'sequence': row['sequence'], 'frame': row['frame'],
                      'input_fingerprints': input_fingerprints,
                      'snapshot_sha256': digest(snapshot_file),
                      'fingerprints': {k: fingerprint(snapshot[k]) for k in STAGES}})
    if guard['blocked_open_attempts']:
        raise ValueError('Protected mask open attempted')
    receipt = {'passed': True, 'scope': plan['scope'], 'plan_sha256': digest(plan_path),
               'source_sha256': digest(Path(__file__)), 'pythonhashseed': os.environ['PYTHONHASHSEED'],
               'environment': environment_identity(), 'initial_state_sha256': plan['initial_state_sha256'],
               'process_id': os.getpid(), 'process_token': uuid.uuid4().hex, 'finished_unix_ns': time.time_ns(),
               'label_guard': dict(guard), 'steps': steps}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


def checked_run(folder, plan, plan_path):
    folder = Path(folder); receipt = json.loads((folder/'receipt.json').read_text())
    if (not receipt.get('passed') or receipt.get('plan_sha256') != digest(plan_path)
            or receipt.get('source_sha256') != plan['source_sha256'] or receipt.get('pythonhashseed') != '0'
            or receipt.get('environment') != plan['environment'] or receipt.get('initial_state_sha256') != plan['initial_state_sha256']
            or type(receipt.get('process_id')) is not int or receipt['process_id'] <= 0
            or not isinstance(receipt.get('process_token'), str) or len(receipt['process_token']) != 32
            or receipt.get('label_guard', {}).get('blocked_open_attempts') != 0 or len(receipt.get('steps', [])) != len(KEYS)):
        raise ValueError('Run receipt differs from plan')
    snapshots = []
    for i, step in enumerate(receipt['steps']):
        if (step['index'], step['sequence'], step['frame']) != (i, *KEYS[i]):
            raise ValueError('Run input schedule differs')
        file = folder/f'step_{i:02d}.pt'
        if digest(file) != step['snapshot_sha256']:
            raise ValueError('Snapshot file changed')
        snap = torch.load(file, map_location='cpu', weights_only=True)
        if set(snap) != set(STAGES) or {k: fingerprint(snap[k]) for k in STAGES} != step['fingerprints']:
            raise ValueError('Snapshot content differs')
        snapshots.append(snap)
    return receipt, snapshots


def compare(plan, plan_path, run_a, run_b):
    if Path(run_a).resolve() == Path(run_b).resolve():
        raise ValueError('Two distinct process outputs required')
    a, xs = checked_run(run_a, plan, plan_path); b, ys = checked_run(run_b, plan, plan_path)
    if a['process_id'] == b['process_id'] or a['process_token'] == b['process_token']:
        raise ValueError('Independent fresh process identities required')
    if [x['input_fingerprints'] for x in a['steps']] != [x['input_fingerprints'] for x in b['steps']]:
        raise ValueError('Actual training inputs differ')
    first = None; rows = []
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        stages = {}
        for stage in STAGES:
            exact = fingerprint(x[stage]) == fingerprint(y[stage])
            stages[stage] = {'exact': exact, 'max_abs_difference': max_difference(x[stage], y[stage])}
            if not exact and first is None:
                first = {'step': i, 'stage': stage, 'sequence': KEYS[i][0], 'frame': KEYS[i][1]}
        rows.append({'index': i, 'sequence': KEYS[i][0], 'frame': KEYS[i][1], 'stages': stages})
    return {'passed': True, 'scope': plan['scope'], 'plan_sha256': digest(plan_path),
            'run_a_receipt_sha256': digest(Path(run_a)/'receipt.json'), 'run_b_receipt_sha256': digest(Path(run_b)/'receipt.json'),
            'all_stages_exact': first is None, 'first_difference': first, 'rows': rows,
            'limitations': ['Six ordered training steps are not a full ten-epoch repeat.',
                            'CPU snapshots synchronize each step, unlike the original training loop.',
                            'A matching probe does not prove all MPS operations deterministic.',
                            'CPython open guards do not intercept arbitrary native-code I/O.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'run', 'compare'), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--run-a', type=Path); parser.add_argument('--run-b', type=Path)
    args = parser.parse_args()
    guard = (install_training_mask_guard(ROOT/'data/polypgen_sequence.jsonl') if args.mode == 'run'
             else install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl'))
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    manifest, manifest_sha = load_prepared(args.manifest)
    if args.mode == 'plan':
        plan = build_plan(manifest, manifest_sha)
        out = validate_output(args.output); out.mkdir(parents=True, exist_ok=False)
        (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    elif args.mode == 'run':
        if not args.plan:
            raise ValueError('A frozen plan is required')
        run(manifest, manifest_sha, args.plan, args.output, guard)
    else:
        if not args.plan or not args.run_a or not args.run_b:
            raise ValueError('Plan and two runs are required')
        plan = require_plan(args.plan, manifest, manifest_sha)
        result = compare(plan, args.plan, args.run_a, args.run_b)
        if guard['open_attempts']:
            raise ValueError('Mask open attempted during comparison')
        out = validate_output(args.output); out.mkdir(parents=True, exist_ok=False)
        (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
