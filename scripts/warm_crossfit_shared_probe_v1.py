#!/usr/bin/env python3
"""Label-free MPS probe of the actual six-arm initializer and update path."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.channel_value_fixed_read_v1 import require_environment_flags
from scripts.crossfit_observer_capture_shared_v1 import DATA
from scripts.crossfit_training_contract_v1 import restore_view_mask
from scripts.memory_token_io import digest
from scripts.preflight_warm_crossfit_training_v1 import ARMS
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.train_multilevel_refiner_seq4_empty_v2 import to_video
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.warm_crossfit_fold_training_v1 import inputs, load_preflight, new_output, require, samples_for, save_payload, trained_model
from scripts.warm_crossfit_shared_fold_v1 import make_arms, step_arm


def probe(preflight, preflight_sha, rows, output, guard, *, device='mps'):
    keys = []
    for seq in ('seq6', 'seq23'):
        candidates = sorted(key for key in rows if key[0] == seq)
        require(candidates, f'Missing probe source: {seq}')
        keys.append(candidates[0])
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    for key in keys:
        values, shape = inputs(rows[key], device)
        require(key[0] != 'seq23' or shape == (720, 834), 'Probe viewport differs')
        original_shape = (720, 1280) if key[0] == 'seq23' else shape
        require(restore_view_mask(np.zeros(shape, dtype=bool), key[0], original_shape).shape == original_shape,
                'Probe restored geometry differs')
        # Deliberately disagree with native logits: synthetic self-derived target,
        # never an annotation, so a nonzero correction gradient is exercised.
        target = (to_video(values[2], shape) <= 0).float()
        models, optimizers, rng, initial = make_arms(preflight, device)
        for arm in ARMS:
            model = models[arm]
            prediction = model(*values)
            require(torch.isfinite(prediction).all() and torch.equal(prediction, values[2]),
                    'Initial residual is not a finite no-op')
            del prediction
            loss, norm, rng[arm] = step_arm(model, optimizers[arm], rng[arm], values, target, shape, device)
            grad = model.head[-1].bias.grad
            require(grad is not None and torch.isfinite(grad).all() and bool((grad.abs() > 0).any()) and
                    norm > 0 and np.isfinite(loss), 'Shared probe gradient is missing or zero')
            require(bool((model.head[-1].bias.detach().abs() > 0).any()), 'Shared optimizer did not update head')
            before_reload = model(*values).detach()
            require(torch.isfinite(before_reload).all(), 'Updated shared model is nonfinite')
            checkpoint = output / f'{key[0]}_{arm}.pt'
            torch.save(save_payload(model, arm=arm, fold=1, preflight=preflight,
                                    preflight_sha=preflight_sha, steps=1), checkpoint)
            checkpoint_sha = digest(checkpoint)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(digest(checkpoint) == checkpoint_sha, 'Probe checkpoint changed during loading')
            restored = trained_model(payload, arm=arm, fold=1, preflight=preflight, preflight_sha=preflight_sha)
            restored.to(device).eval().requires_grad_(False)
            with torch.no_grad():
                after_reload = restored(*values)
                require(torch.isfinite(after_reload).all() and
                        torch.allclose(before_reload, after_reload, rtol=1e-5, atol=1e-6),
                        'Reloaded shared checkpoint output differs')
            checks.append({'arm': arm, 'sequence': key[0], 'frame': key[1], 'video_shape': list(shape),
                'head_bias_gradient_abs': float(grad.abs().max().detach().cpu()), 'gradient_l2': norm,
                'synthetic_loss': loss, 'checkpoint_path': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
                'initial_state_sha256': initial[arm], 'optimizer_updated': True})
            del restored, before_reload, after_reload
        del models, optimizers, rng, values, target
    require(guard['open_attempts'] == 0, 'Annotation open attempted during shared MPS probe')
    (output / 'completion.json').write_text(json.dumps({'passed': True, 'kind': 'shared_six_arm_probe_v1',
        'scope': 'Label-free shared MPS update and checkpoint probe with synthetic targets; no efficacy or cross-device claim.',
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


if __name__ == '__main__': main()
