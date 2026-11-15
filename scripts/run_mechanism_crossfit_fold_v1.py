#!/usr/bin/env python3
"""Train or predict four separate control arms on one certified fold."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from scripts.channel_value_fixed_read_v1 import require_environment_flags
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, phase_artifact
from scripts.crossfit_training_contract_v1 import install_fold_training_guard, restore_view_mask, training_label_hashes
from scripts.load_aligned_rgb_control_v1 import load_aligned_rgb
from scripts.mechanism_crossfit_control_core_v1 import ARMS, forward, make_arms, save_payload, step, trained_model
from scripts.memory_token_io import digest
from scripts.preflight_mechanism_crossfit_v1 import sources
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.train_multilevel_refiner_seq4_empty_v2 import to_video
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.verified_capture_cache_v1 import VerifiedCaptureCache
from scripts.warm_crossfit_fold_training_v1 import (checked_train_target, cpu_inputs, fold_definition, inputs,
    load_preflight, new_output, require, samples_for)


def read_bound(path, expected):
    require(digest(path) == expected, 'Study input changed')
    value = json.loads(Path(path).read_text())
    require(digest(path) == expected, 'Study input changed during reading')
    return value


def study_inputs(folder):
    folder = Path(folder).resolve(strict=True)
    require(folder.is_relative_to(BASE.resolve(strict=True)), 'Control plan outside Drive')
    plan_path = folder / 'plan.json'; completion = json.loads((folder / 'completion.json').read_text())
    sha = digest(plan_path); plan = read_bound(plan_path, sha)
    require(completion.get('passed') is True and completion.get('plan_sha256') == sha and
            completion.get('label_guard', {}).get('open_attempts') == 0 and plan.get('passed') is True and
            plan.get('arms') == list(ARMS) and plan.get('source_sha256') == sources(), 'Control plan differs')
    warm, warm_sha = load_preflight(plan['warm_preflight_path'])
    require(plan['warm_preflight_sha256'] == warm_sha and plan['folds'] == warm['folds'] and
            plan['aggregate_sha256'] == warm['aggregate_sha256'] and
            plan['initial_state_sha256'] == warm['initial_state_sha256']['foreground_background_asinh'],
            'Control and warm plans differ')
    index, rows = samples_for(warm)
    require(len(rows) == plan['samples'] == 357, 'Control capture coverage differs')
    return plan, sha, warm, warm_sha, index, rows


def rgb_input(index, videos, sequence, frame, shape, device):
    prompt = index['prompts'][sequence]
    provenance = prompt.get('prompt_provenance', prompt)
    return load_aligned_rgb(videos[sequence][frame], index['source_frames'][sequence][frame],
        sequence, frame, provenance['roi_xyxy'], shape).to(device)


def probe(plan, plan_sha, warm, warm_sha, index, rows, out, guard, *, device='mps'):
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}
    out.mkdir(parents=True, exist_ok=False); checks = []
    for seq in ('seq6', 'seq23'):
        keys = sorted(key for key in rows if key[0] == seq)
        require(keys, 'Probe sequence has no captured reads')
        _, frame = keys[0]; values, shape = inputs(rows[(seq, frame)], device)
        require(seq != 'seq23' or shape == (720, 834), 'Probe viewport differs')
        rgb = rgb_input(index, videos, seq, frame, shape, device)
        target = (to_video(values[2], shape) <= 0).float()
        models, optimizers, states = make_arms(plan['initial_state_sha256'], device)
        for arm in ARMS:
            model = models[arm]
            require(torch.equal(forward(model, arm, values, rgb), values[2]), 'Control initial no-op differs')
            loss, norm, states[arm] = step(model, arm, optimizers[arm], states[arm],
                                           values, rgb, target, shape, device)
            grad = model.head[-1].bias.grad
            require(grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0 and norm > 0,
                    'Control probe gradient is missing or zero')
            # Training mode permits the live, just-updated model to run without
            # bypassing the evaluation checkpoint guard.
            before = forward(model, arm, values, rgb).detach()
            path = out / f'{seq}_{arm}.pt'
            torch.save(save_payload(model, arm=arm, fold=1, warm_plan=warm, warm_plan_sha=warm_sha,
                       study_plan_sha=plan_sha, steps=1, sources=plan['source_sha256']), path)
            checkpoint_sha = digest(path); payload = torch.load(path, map_location='cpu', weights_only=True)
            require(digest(path) == checkpoint_sha, 'Probe checkpoint changed')
            reloaded = trained_model(payload, arm=arm, fold=1, warm_plan=warm, warm_plan_sha=warm_sha,
                study_plan_sha=plan_sha, sources=plan['source_sha256']).to(device).eval().requires_grad_(False)
            with torch.no_grad(): after = forward(reloaded, arm, values, rgb)
            require(torch.isfinite(after).all() and torch.allclose(before, after, rtol=1e-5, atol=1e-6),
                    'Reloaded control output differs')
            checks.append({'sequence': seq, 'frame': frame, 'arm': arm, 'video_shape': list(shape),
                'loss': loss, 'gradient_l2': norm, 'head_bias_gradient_abs': float(grad.abs().max().cpu()),
                'checkpoint_path': str(path), 'checkpoint_sha256': checkpoint_sha})
    require(guard['open_attempts'] == 0 and sources() == plan['source_sha256'], 'Probe guard or sources differ')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'plan_sha256': plan_sha,
        'warm_preflight_sha256': warm_sha, 'device': device, 'environment': warm['environment'],
        'checks': checks, 'label_guard': dict(guard), 'scope': 'Synthetic targets; engineering only.'}, indent=2) + '\n')


def train(plan, plan_sha, warm, warm_sha, index, rows, fold, out, guard, *, device='mps', cache_mib=1024):
    train_groups, _, _, order = fold_definition(warm, fold, rows)
    warm_train = [x for x in rows.values() if x['origin'] == 'warm' and x['sequence'] in train_groups]
    hashes = training_label_hashes(fold, warm_train, index['source_frames'])
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}
    models, optimizers, states = make_arms(plan['initial_state_sha256'], device)
    cache = VerifiedCaptureCache(cache_mib * 1024**2, cpu_inputs)
    out.mkdir(parents=True, exist_ok=False); history = {arm: [] for arm in ARMS}; steps = 0
    for epoch, epoch_order in enumerate(order):
        losses = {arm: [] for arm in ARMS}; peaks = {arm: 0. for arm in ARMS}
        for seq, frame in epoch_order:
            require(seq in train_groups, 'Non-training patient in control schedule')
            item = rows[(seq, frame)]; values, shape = cache.get(item, device)
            label_sha = item['label_sha256'] if item['origin'] == 'historical' else hashes[(seq, frame)]
            target = checked_train_target(item, shape, label_sha).to(device)
            rgb = rgb_input(index, videos, seq, frame, shape, device)
            for arm in ARMS:
                loss, norm, states[arm] = step(models[arm], arm, optimizers[arm], states[arm],
                                               values, rgb, target, shape, device)
                losses[arm].append(loss); peaks[arm] = max(peaks[arm], norm)
            steps += 1
        cache.verify_boundary()
        for arm in ARMS:
            history[arm].append({'epoch': epoch + 1, 'steps': len(losses[arm]),
                                 'mean_loss': float(np.mean(losses[arm])), 'peak_gradient_l2': peaks[arm]})
        (out / 'progress.json').write_text(json.dumps(history, indent=2) + '\n')
    require(guard['blocked_open_attempts'] == 0 and sources() == plan['source_sha256'], 'Training guard or source differs')
    cache.verify_boundary(final=True); records = {}
    for arm in ARMS:
        folder = out / arm; folder.mkdir(); checkpoint = folder / 'checkpoint.pt'
        torch.save(save_payload(models[arm], arm=arm, fold=fold, warm_plan=warm, warm_plan_sha=warm_sha,
                   study_plan_sha=plan_sha, steps=steps, sources=plan['source_sha256']), checkpoint)
        records[arm] = {'checkpoint_path': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                        'history': history[arm]}
    (out / 'group.json').write_text(json.dumps({'passed': True, 'fold': fold, 'arms': records,
        'plan_sha256': plan_sha, 'warm_preflight_sha256': warm_sha, 'training_steps': steps,
        'initial_state_sha256': plan['initial_state_sha256'],
        'device': device, 'environment': warm['environment'], 'cache': cache.stats(),
        'label_guard': dict(guard)}, indent=2) + '\n')


def predict(plan, plan_sha, warm, warm_sha, index, rows, fold, training, out, guard, *, device='mps'):
    _, _, test, order = fold_definition(warm, fold, rows)
    training = Path(training).resolve(strict=True); root = training.parent.parent
    group_path = training / 'group.json'; group_sha = phase_artifact(root, f'train_f{fold}', group_path)
    group = read_bound(group_path, group_sha)
    require(group.get('passed') is True and group.get('fold') == fold and group.get('plan_sha256') == plan_sha and
            group.get('warm_preflight_sha256') == warm_sha and group.get('training_steps') == sum(map(len, order)) and
            group.get('device') == device and group.get('environment') == warm['environment'] and
            group.get('label_guard', {}).get('blocked_open_attempts') == 0 and set(group['arms']) == set(ARMS),
            'Control training receipt differs')
    models = {}
    for arm in ARMS:
        item = group['arms'][arm]; path = Path(item['checkpoint_path']).resolve(strict=True)
        require(path == training / arm / 'checkpoint.pt' and digest(path) == item['checkpoint_sha256'],
                'Control checkpoint changed')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        require(digest(path) == item['checkpoint_sha256'] and payload.get('training_steps') == group['training_steps'],
                'Control checkpoint changed during loading')
        models[arm] = trained_model(payload, arm=arm, fold=fold, warm_plan=warm, warm_plan_sha=warm_sha,
            study_plan_sha=plan_sha, sources=plan['source_sha256']).to(device).eval().requires_grad_(False)
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}
    out.mkdir(parents=True, exist_ok=False); saved = []
    with torch.no_grad():
        for seq, frame in sorted(rows):
            if seq not in test:
                continue
            values, shape = inputs(rows[(seq, frame)], device)
            rgb = rgb_input(index, videos, seq, frame, shape, device)
            original_shape = (720, 1280) if seq == 'seq23' else shape
            row = {'sequence': seq, 'frame': frame, 'source_frame_index':
                index['source_frames'][seq][frame]['source_frame_index'], 'video_shape': list(shape),
                'source_shape': list(original_shape), 'masks': {}}
            predictions = {'baseline': values[2], **{arm: forward(models[arm], arm, values, rgb) for arm in ARMS}}
            for arm, logits in predictions.items():
                require(torch.isfinite(logits).all(), 'Nonfinite held-out control prediction')
                view = (to_video(logits, shape) > 0).cpu().numpy()[0, 0]
                mask = restore_view_mask(view, seq, original_shape)
                path = out / f'{seq}_{frame:06d}_{arm}.png'
                Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                row['masks'][arm] = {'path': str(path), 'sha256': digest(path)}
            saved.append(row)
    require(guard['open_attempts'] == 0 and digest(group_path) == group_sha and
            all(digest(group['arms'][arm]['checkpoint_path']) == group['arms'][arm]['checkpoint_sha256'] for arm in ARMS)
            and sources() == plan['source_sha256'], 'Prediction inputs or label guard differ')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'fold': fold, 'plan_sha256': plan_sha,
        'warm_preflight_sha256': warm_sha, 'training_group_sha256': group_sha, 'rows': saved,
        'device': device, 'environment': warm['environment'], 'label_guard': dict(guard)}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('probe', 'train', 'predict'), required=True)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--fold', type=int)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--training', type=Path)
    parser.add_argument('--cache-mib', type=int, default=1024)
    args = parser.parse_args(); out = new_output(args.output)
    require((args.mode == 'predict') == (args.training is not None) and
            ((args.mode == 'probe' and args.fold is None) or
             (args.mode != 'probe' and type(args.fold) is int and 1 <= args.fold <= 5)) and 0 <= args.cache_mib <= 1024,
            'Invalid control mode or cache size')
    guard = install_fold_training_guard(DATA, args.fold) if args.mode == 'train' else install_mask_open_guard(DATA)
    require_environment_flags(); require(torch.backends.mps.is_available(), 'MPS unavailable')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(2)
    plan, sha, warm, warm_sha, index, rows = study_inputs(args.plan)
    require(environment_identity() == warm['environment'], 'MPS/runtime environment differs')
    if args.mode != 'probe':
        from scripts.run_mechanism_crossfit_study_v1 import check_probe, phases, validate_resume
        root = out.parent.parent
        require(out == root / args.mode / f'fold{args.fold}', 'Control output is outside its assigned batch phase')
        _, _, completed = validate_resume(root, args.plan.resolve(strict=True), sha, plan['source_sha256'], args.cache_mib)
        require(phases(root, args.plan.resolve(strict=True), args.cache_mib)[completed][0] ==
                f'{args.mode}_f{args.fold}', 'Control worker phase is not next in frozen schedule')
        check_probe(root, sha, warm_sha, warm, rows)
    if args.mode == 'probe':
        probe(plan, sha, warm, warm_sha, index, rows, out, guard)
    elif args.mode == 'train':
        train(plan, sha, warm, warm_sha, index, rows, args.fold, out, guard, cache_mib=args.cache_mib)
    else:
        predict(plan, sha, warm, warm_sha, index, rows, args.fold, args.training, out, guard)
    print(json.dumps({'passed': True, 'mode': args.mode, 'fold': args.fold, 'output': str(out)}))


if __name__ == '__main__':
    main()
