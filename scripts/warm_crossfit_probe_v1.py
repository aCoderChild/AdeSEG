#!/usr/bin/env python3
"""Focused label-free MPS shape, initial-identity and gradient probe."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.channel_value_fixed_read_v1 import model_for, require_environment_flags
from scripts.crossfit_observer_capture_shared_v1 import DATA
from scripts.crossfit_training_contract_v1 import restore_view_mask
from scripts.preflight_warm_crossfit_training_v1 import ARMS, SEED
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v2 import initial_hash
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.warm_crossfit_fold_training_v1 import inputs, load_preflight, new_output, require, samples_for


def probe(preflight, preflight_sha, rows, output, guard, *, device='mps'):
    keys = []
    for seq in ('seq6', 'seq23'):
        candidates = sorted(key for key in rows if key[0] == seq)
        require(candidates, f'Missing probe source: {seq}')
        keys.append(candidates[0])
    checks = []
    for arm in ARMS:
        torch.manual_seed(SEED)
        if device == 'mps': torch.mps.manual_seed(SEED)
        model = model_for(arm).train().to(device)
        require(initial_hash(model) == preflight['initial_state_sha256'][arm], 'Probe initial state differs')
        for key in keys:
            model.zero_grad(set_to_none=True)
            values, shape = inputs(rows[key], device)
            require((key[0] != 'seq23' or shape == (720, 834)), 'Probe viewport differs')
            prediction = model(*values)
            require(torch.isfinite(prediction).all() and torch.equal(prediction, values[2]),
                    'Initial residual is not a finite no-op')
            prediction.float().mean().backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
                    'Probe gradient is nonfinite')
            bias_gradient = model.head[-1].bias.grad
            require(bias_gradient is not None and torch.isfinite(bias_gradient).all() and
                    bool((bias_gradient.abs() > 0).any()), 'Probe residual-head gradient is missing or zero')
            original_shape = (720, 1280) if key[0] == 'seq23' else shape
            view = np.zeros(shape, dtype=bool)
            restored = restore_view_mask(view, key[0], original_shape)
            require(restored.shape == original_shape, 'Probe restored geometry differs')
            checks.append({'arm': arm, 'sequence': key[0], 'frame': key[1], 'video_shape': list(shape),
                           'head_bias_gradient_abs': float(bias_gradient.abs().max().detach().cpu())})
    require(guard['open_attempts'] == 0, 'Annotation open attempted during MPS probe')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'completion.json').write_text(json.dumps({'passed': True,
        'scope': 'Targeted label-free same-device MPS probe; not a cross-device comparison or efficacy test.',
        'preflight_sha256': preflight_sha, 'device': device, 'environment': preflight.get('environment'),
        'checks': checks, 'label_guard': dict(guard)}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); out = new_output(args.output)
    guard = install_mask_open_guard(DATA)
    require_environment_flags(); require(torch.backends.mps.is_available(), 'MPS unavailable')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(2)
    preflight, sha = load_preflight(args.preflight)
    require(environment_identity() == preflight['environment'], 'MPS/runtime environment differs from capture')
    _, rows = samples_for(preflight)
    probe(preflight, sha, rows, out, guard)
    print(json.dumps({'passed': True, 'output': str(out)}))


if __name__ == '__main__':
    main()
