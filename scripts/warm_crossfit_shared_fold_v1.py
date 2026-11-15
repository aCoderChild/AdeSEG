#!/usr/bin/env python3
"""Train six isolated fold controls while sharing each validated input read.

Each arm has independent model, Adam optimizer and Python/NumPy/Torch RNG
state. Arm updates follow exactly the same per-arm schedule as the separate
worker. Inputs/targets are read once per scheduled step, and all output
checkpoints are independently saved and bound by a shared group receipt.
"""
import argparse
import json
from pathlib import Path
import random
import resource
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss
from scripts.channel_value_fixed_read_v1 import model_for, require_environment_flags
from scripts.crossfit_observer_capture_shared_v1 import DATA
from scripts.crossfit_training_contract_v1 import install_fold_training_guard, training_label_hashes
from scripts.memory_token_io import digest
from scripts.preflight_warm_crossfit_training_v1 import ARMS, SEED
from scripts.source_group_refiner_shared_v2 import initial_hash
from scripts.train_multilevel_refiner_seq4_empty_v2 import to_video
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.verified_capture_cache_v1 import VerifiedCaptureCache
from scripts.warm_crossfit_fold_training_v1 import (checked_train_target, cpu_inputs, fold_definition,
    load_preflight, new_output, require, samples_for, save_payload)


def capture_rng(device):
    return {'torch': torch.get_rng_state().clone(),
            'mps': torch.mps.get_rng_state().clone() if device == 'mps' else None,
            'python': random.getstate(), 'numpy': np.random.get_state()}


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    if device == 'mps': torch.mps.set_rng_state(state['mps'])
    random.setstate(state['python']); np.random.set_state(state['numpy'])


def seed_arm(device):
    torch.manual_seed(SEED)
    if device == 'mps': torch.mps.manual_seed(SEED)
    random.seed(SEED); np.random.seed(SEED)


def make_arms(preflight, device):
    """Initialize isolated trainable states, identically to separate workers."""
    models = {}; optimizers = {}; rng = {}; initial = {}
    for arm in ARMS:
        seed_arm(device)
        model = model_for(arm).train().to(device)
        initial[arm] = initial_hash(model)
        require(initial[arm] == preflight['initial_state_sha256'][arm], 'Initial model state differs')
        models[arm] = model
        optimizers[arm] = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
        rng[arm] = capture_rng(device)
    require(len({id(p) for m in models.values() for p in m.parameters()}) ==
            sum(1 for m in models.values() for _ in m.parameters()), 'Arm parameters are shared')
    return models, optimizers, rng, initial


def step_arm(model, optimizer, rng_state, values, target, shape, device):
    """One shared-schedule update; also used verbatim by the MPS probe."""
    restore_rng(rng_state, device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(*values)
    loss = cmsa_boundary_loss(to_video(prediction, shape), target)
    require(torch.isfinite(prediction).all() and torch.isfinite(loss), 'Nonfinite training values')
    loss.backward()
    require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
            'Nonfinite training gradients')
    norm2 = sum((p.grad.float().square().sum() for p in model.parameters() if p.grad is not None),
                torch.zeros((), device=device))
    norm = float(torch.sqrt(norm2).detach().cpu())
    require(np.isfinite(norm), 'Nonfinite gradient norm')
    optimizer.step()
    require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite model parameters')
    return float(loss.detach().cpu()), norm, capture_rng(device)


def train_shared(preflight, preflight_sha, fold, output, guard, *, device='mps', cache_mib=1024):
    index, rows = samples_for(preflight)
    train_groups, _, _, order = fold_definition(preflight, fold, rows)
    warm_train = [x for x in rows.values() if x['origin'] == 'warm' and x['sequence'] in train_groups]
    hashes = training_label_hashes(fold, warm_train, index['source_frames'])
    models, optimizers, rng, initial = make_arms(preflight, device)
    history = {a: [] for a in ARMS}
    cache = VerifiedCaptureCache(cache_mib * 1024**2, cpu_inputs)
    output.mkdir(parents=True, exist_ok=False)
    steps = 0
    for epoch, epoch_order in enumerate(order):
        losses = {a: [] for a in ARMS}; peaks = {a: 0.0 for a in ARMS}
        for seq, frame in epoch_order:
            require(seq in train_groups, 'Non-training patient in schedule')
            item = rows[(seq, frame)]
            label_sha = item['label_sha256'] if item['origin'] == 'historical' else hashes[(seq, frame)]
            values, shape = cache.get(item, device)
            target = checked_train_target(item, shape, label_sha).to(device)
            for arm in ARMS:
                model, optimizer = models[arm], optimizers[arm]
                loss, norm, rng[arm] = step_arm(model, optimizer, rng[arm], values, target, shape, device)
                peaks[arm] = max(peaks[arm], norm)
                losses[arm].append(loss)
            steps += 1
        cache.verify_boundary()
        for arm in ARMS:
            history[arm].append({'epoch': epoch + 1, 'steps': len(losses[arm]),
                'mean_loss': float(np.mean(losses[arm])), 'peak_gradient_l2': peaks[arm]})
        (output / 'progress.json').write_text(json.dumps({'history': history}, indent=2) + '\n')
    require(guard['blocked_open_attempts'] == 0, 'Protected annotation open attempted')
    cache.verify_boundary(final=True); stats = cache.stats(); arm_receipts = {}
    for arm in ARMS:
        folder = output / arm; folder.mkdir()
        checkpoint = folder / 'checkpoint.pt'
        torch.save(save_payload(models[arm], arm=arm, fold=fold, preflight=preflight,
                                preflight_sha=preflight_sha, steps=steps), checkpoint)
        record = {'passed': True, 'fold': fold, 'arm': arm, 'preflight_sha256': preflight_sha,
            'checkpoint_sha256': digest(checkpoint), 'initial_state_sha256': initial[arm],
            'training_steps': steps, 'history': history[arm], 'device': device,
            'environment': preflight.get('environment'), 'cache': stats, 'label_guard': dict(guard),
            'training_mode': 'shared_inputs_separate_models_optimizers_and_rng'}
        path = folder / 'completion.json'; path.write_text(json.dumps(record, indent=2) + '\n')
        arm_receipts[arm] = {'path': str(path), 'sha256': digest(path), 'checkpoint_sha256': digest(checkpoint)}
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
    (output / 'group.json').write_text(json.dumps({'passed': True, 'fold': fold, 'preflight_sha256': preflight_sha,
        'scope': 'Shared validated inputs; separate models, optimizers and RNG states.',
        'arms': arm_receipts, 'cache': stats, 'peak_process_rss_bytes': rss_bytes,
        'label_guard': dict(guard)}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', required=True, type=Path)
    parser.add_argument('--fold', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache-mib', type=int, default=1024)
    args = parser.parse_args(); output = new_output(args.output)
    require(0 <= args.cache_mib <= 1024, 'CPU cache must be between 0 and 1024 MiB')
    guard = install_fold_training_guard(DATA, args.fold)
    require_environment_flags(); require(torch.backends.mps.is_available(), 'MPS unavailable')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(2)
    preflight, sha = load_preflight(args.preflight)
    require(environment_identity() == preflight['environment'], 'MPS/runtime environment differs from capture')
    train_shared(preflight, sha, args.fold, output, guard, cache_mib=args.cache_mib)
    print(json.dumps({'passed': True, 'fold': args.fold, 'output': str(output)}))


if __name__ == '__main__':
    main()
