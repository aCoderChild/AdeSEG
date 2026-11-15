#!/usr/bin/env python3
"""One fold-scoped warm fixed-read training or prediction worker.

No live video propagation, global score, model selection or independent-test
claim is made here. Prediction must be run in a separate label-blocked process.
The orchestration and all-fold verification/scoring are deliberately separate.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss
from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1
from scripts.channel_value_fixed_read_v1 import CONFIG, model_for, require_environment_flags
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, EXISTING, PLAN_SHA, phase_artifact, validate_plan
from scripts.crossfit_training_contract_v1 import (checked_aligned_training_target, fold_split,
    install_fold_training_guard, patient_balanced_schedule, restore_view_mask,
    training_label_hashes, validate_warm_index_structure)
from scripts.memory_token_io import digest
from scripts.preflight_warm_crossfit_training_v1 import ARMS, EPOCHS, SEED, UPDATES_PER_PATIENT, sources
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v2 import initial_hash
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture, to_video
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.verified_capture_cache_v1 import VerifiedCaptureCache

def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def new_output(path):
    out = Path(path).resolve()
    require(out.is_relative_to(BASE.resolve(strict=True)) and not out.exists(), 'Invalid output directory')
    return out


def load_preflight(folder):
    folder = Path(folder).resolve(strict=True)
    require(folder.is_relative_to(BASE.resolve(strict=True)), 'Preflight outside Drive experiment root')
    path = folder / 'plan.json'; done = read(folder / 'completion.json'); sha = digest(path)
    plan = read(path)
    require(digest(path) == sha and done.get('passed') is True and done.get('plan_sha256') == sha and
            done.get('label_guard', {}).get('open_attempts') == 0 and plan.get('passed') is True and
            plan.get('source_sha256') == sources() and plan.get('index_plan_sha256') == PLAN_SHA and
            plan.get('seed') == SEED and plan.get('epochs') == EPOCHS and
            plan.get('updates_per_patient_per_epoch') == UPDATES_PER_PATIENT and
            plan.get('arms') == list(ARMS) and set(plan.get('initial_state_sha256', {})) == set(ARMS) and
            len(set(plan['initial_state_sha256'].values())) == 1 and
            plan.get('optimizer') == {'name': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001} and
            plan.get('loss') == 'corrected CMSA boundary weighted BCE and IoU plus soft Dice',
            'Warm cross-fit preflight or sources changed')
    return plan, sha


def samples_for(preflight):
    """Load index metadata, not target mask bytes; preflight validated receipts."""
    index = validate_plan()
    require(preflight['backbone_sha256'] == index['checkpoint_sha256'], 'Backbone checkpoint differs')
    manifest_path = EXISTING / 'manifest.json'
    require(digest(manifest_path) == preflight['old_manifest_sha256'] == index['existing_manifest_sha256'],
            'Historical manifest changed')
    old = read(manifest_path)
    require(digest(manifest_path) == preflight['old_manifest_sha256'], 'Historical manifest changed while reading')
    folder = Path(preflight['aggregate_path']).resolve(strict=True); aggregate_path = folder / 'captures.json'
    require(digest(aggregate_path) == preflight['aggregate_sha256'], 'Warm aggregate changed')
    aggregate = read(aggregate_path)
    require(digest(aggregate_path) == preflight['aggregate_sha256'], 'Warm aggregate changed while reading')
    certificate, certificate_sha = verify_certificate(aggregate['certificate_path'])
    require(certificate_sha == preflight['certificate_sha256'] == aggregate['certificate_sha256'] and
            certificate['warm_gate_sha256'] == preflight['warm_gate_sha256'] == aggregate['warm_gate_sha256'] and
            certificate['warm_gate_path'] == aggregate['warm_gate_path'] and
            certificate['prompt_policy'] == aggregate['prompt_policy'] == 'warm_box_points_then_commit_archived_mask',
            'Warm certificate or gate changed')
    validate_warm_index_structure(index, old, aggregate, index_sha256=PLAN_SHA,
        warm_preflight_sha256=aggregate['warm_preflight_sha256'], warm_gate_path=aggregate['warm_gate_path'],
        warm_gate_sha256=preflight['warm_gate_sha256'], certificate_path=aggregate['certificate_path'],
        certificate_sha256=preflight['certificate_sha256'])
    # The dataset manifest is source-bound in the frozen V2 plan.  Loading it
    # resolves paths and stats files but does not decode any annotation.
    require(digest(DATA) == index['data_manifest_sha256'], 'Dataset manifest changed')
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}
    rows = {}
    for item in old['samples']:
        key = (item['sequence'], item['frame'])
        row = dict(item); row['origin'] = 'historical'
        rows[key] = row
    for item in aggregate['observations']:
        key = (item['sequence'], item['frame'])
        frame = videos[key[0]][key[1]]
        require(frame.frame_index == item['source_frame_index'], 'Dataset source frame differs')
        row = dict(item); row.update(origin='warm', feature=item['feature_file'], snapshot=item['snapshot_file'],
                                     label=str(frame.mask_path))
        rows[key] = row
    for (sequence, ordinal), row in rows.items():
        frame = videos[sequence][ordinal]
        require(frame.frame_index == index['source_frames'][sequence][ordinal]['source_frame_index'] and
                os.path.realpath(frame.mask_path) == os.path.realpath(row['label']),
                'Label path or source-frame index differs')
    require(len(rows) == 357 and digest(DATA) == index['data_manifest_sha256'], 'Warm sample coverage differs')
    return index, rows


def fold_definition(preflight, fold, rows):
    require(type(fold) is int and 1 <= fold <= 5, 'Invalid fold')
    train, validation, test = fold_split(fold)
    expected = {'fold': fold, 'train': sorted(train), 'validation': sorted(validation), 'test': sorted(test),
                'training_order': patient_balanced_schedule(list(rows.values()), fold, SEED, EPOCHS,
                                                            UPDATES_PER_PATIENT)}
    require(len(preflight['folds']) == 5 and preflight['folds'][fold - 1] == expected,
            'Fold identities or training schedule differ')
    return train, validation, test, expected['training_order']


def cpu_inputs(item):
    early, deep, logits, shape = read_capture(item)
    return (early, deep, logits), shape


def inputs(item, device):
    tensors, shape = cpu_inputs(item)
    require(all(digest(item[k]) == item[k + '_sha256'] for k in ('feature', 'snapshot')),
            'Capture changed during decoding')
    return tuple(t.to(device) for t in tensors), shape


def checked_train_target(item, shape, expected_sha):
    original_shape = (720, 1280) if item['sequence'] == 'seq23' else shape
    return checked_aligned_training_target(item['label'], expected_sha, item['sequence'], shape, original_shape)


def save_payload(model, *, arm, fold, preflight, preflight_sha, steps):
    return {'format': 'multiscale_channel_value_refiner_v1', 'research_only': True,
        'training_steps': steps, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'source_sha256': digest(ROOT / 'modeling/multiscale_channel_value_refiner_v1.py'),
        'mode': arm, 'config': CONFIG, 'backbone_sha256': preflight['backbone_sha256'],
        'feature_pipeline_sha256': preflight['feature_pipeline_sha256'], 'train_manifest_sha256': preflight_sha,
        'crossfit_fold': fold, 'crossfit_arm': arm, 'preflight_sha256': preflight_sha,
        'aggregate_sha256': preflight['aggregate_sha256'], 'source_sha256_map': sources()}


def trained_model(payload, *, arm, fold, preflight, preflight_sha):
    require(payload.get('crossfit_fold') == fold and payload.get('crossfit_arm') == arm and
            payload.get('preflight_sha256') == preflight_sha and
            payload.get('aggregate_sha256') == preflight['aggregate_sha256'] and
            payload.get('source_sha256_map') == sources(), 'Fold checkpoint provenance differs')
    model = model_for(arm)
    require(isinstance(model, MultiscaleChannelValueRefinerV1), 'Unsupported refiner')
    model.load_trained_state(payload, backbone_sha256=preflight['backbone_sha256'],
        feature_pipeline_sha256=preflight['feature_pipeline_sha256'], train_manifest_sha256=preflight_sha)
    return model


def train(preflight, preflight_sha, fold, arm, output, guard, *, device='mps', cache_mib=1024):
    index, rows = samples_for(preflight)
    train_groups, _, _, order = fold_definition(preflight, fold, rows)
    warm_train = [x for x in rows.values() if x['origin'] == 'warm' and x['sequence'] in train_groups]
    hashes = training_label_hashes(fold, warm_train, index['source_frames'])
    torch.manual_seed(SEED)
    if device == 'mps':
        torch.mps.manual_seed(SEED)
    random.seed(SEED); np.random.seed(SEED)
    model = model_for(arm).train().to(device)
    starting_sha = initial_hash(model)
    require(starting_sha == preflight['initial_state_sha256'][arm], 'Initial model state differs')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
    cache = VerifiedCaptureCache(cache_mib * 1024**2, cpu_inputs)
    output.mkdir(parents=True, exist_ok=False)
    history = []; steps = 0
    for epoch, epoch_order in enumerate(order):
        losses = []; peak_gradient = 0.0
        for seq, frame in epoch_order:
            item = rows[(seq, frame)]
            require(seq in train_groups, 'Non-training patient in schedule')
            label_sha = item['label_sha256'] if item['origin'] == 'historical' else hashes[(seq, frame)]
            values, shape = cache.get(item, device)
            target = checked_train_target(item, shape, label_sha).to(device)
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
            peak_gradient = max(peak_gradient, norm)
            optimizer.step()
            require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite model parameters')
            losses.append(float(loss.detach().cpu())); steps += 1
        history.append({'epoch': epoch + 1, 'steps': len(losses), 'mean_loss': float(np.mean(losses)),
                        'peak_gradient_l2': peak_gradient})
        cache.verify_boundary()
        (output / 'progress.json').write_text(json.dumps({'history': history}, indent=2) + '\n')
    require(guard['blocked_open_attempts'] == 0, 'Protected annotation open attempted')
    cache.verify_boundary(final=True)
    checkpoint = output / 'checkpoint.pt'
    torch.save(save_payload(model, arm=arm, fold=fold, preflight=preflight,
                            preflight_sha=preflight_sha, steps=steps), checkpoint)
    result = {'passed': True, 'fold': fold, 'arm': arm, 'preflight_sha256': preflight_sha,
        'checkpoint_sha256': digest(checkpoint), 'initial_state_sha256': starting_sha,
        'training_steps': steps, 'history': history, 'device': device,
        'environment': preflight.get('environment'), 'cache': cache.stats(), 'label_guard': dict(guard)}
    (output / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')


def predict(preflight, preflight_sha, fold, arm, training, output, guard, *, device='mps'):
    index, rows = samples_for(preflight)
    _, _, test, order = fold_definition(preflight, fold, rows)
    training = Path(training).resolve(strict=True)
    require(training.is_relative_to(BASE.resolve(strict=True)), 'Training output outside Drive')
    batch = training.parent.parent.parent
    training_path = training / 'completion.json'
    training_sha = phase_artifact(batch, f'train_f{fold}_{arm}', training_path)
    rec = read(training_path); checkpoint = training / 'checkpoint.pt'
    require(rec.get('passed') is True and rec.get('fold') == fold and rec.get('arm') == arm and
            digest(training_path) == training_sha and rec.get('preflight_sha256') == preflight_sha and
            rec.get('checkpoint_sha256') == digest(checkpoint) and
            rec.get('device') == device and rec.get('environment') == preflight.get('environment') and
            rec.get('training_steps') == sum(map(len, order)) and
            rec.get('label_guard', {}).get('blocked_open_attempts') == 0, 'Fold training receipt differs')
    checkpoint_sha = digest(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(digest(checkpoint) == checkpoint_sha and payload.get('training_steps') == rec['training_steps'],
            'Checkpoint changed during loading')
    model = trained_model(payload, arm=arm, fold=fold, preflight=preflight, preflight_sha=preflight_sha)
    model.to(device).eval().requires_grad_(False)
    output.mkdir(parents=True, exist_ok=False)
    saved = []
    with torch.no_grad():
        for key in sorted(rows):
            if key[0] not in test:
                continue
            item = rows[key]; values, shape = inputs(item, device)
            refined = model(*values)
            require(torch.isfinite(refined).all(), 'Nonfinite prediction')
            original_shape = (720, 1280) if key[0] == 'seq23' else shape
            result = {'sequence': key[0], 'frame': key[1], 'source_frame_index':
                      index['source_frames'][key[0]][key[1]]['source_frame_index'],
                      'video_shape': list(shape), 'source_shape': list(original_shape)}
            for name, logits in (('baseline', values[2]), ('refined', refined)):
                view = (to_video(logits, shape) > 0).detach().cpu().numpy()[0, 0]
                mask = restore_view_mask(view, key[0], original_shape)
                path = output / f'{key[0]}_{key[1]:06d}_{name}.png'
                Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                result[name + '_path'] = str(path); result[name + '_sha256'] = digest(path)
            saved.append(result)
    require(guard['open_attempts'] == 0, 'Annotation open attempted during prediction')
    require(digest(training_path) == training_sha and digest(checkpoint) == checkpoint_sha,
            'Fold training artifacts changed during prediction')
    result = {'passed': True, 'fold': fold, 'arm': arm, 'preflight_sha256': preflight_sha,
        'training_completion_sha256': training_sha, 'checkpoint_sha256': checkpoint_sha,
        'rows': saved, 'device': device, 'environment': preflight.get('environment'), 'label_guard': dict(guard)}
    (output / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('train', 'predict'))
    parser.add_argument('--preflight', required=True, type=Path)
    parser.add_argument('--fold', required=True, type=int)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--training', type=Path)
    parser.add_argument('--cache-mib', type=int, default=1024)
    args = parser.parse_args()
    out = new_output(args.output)
    require((args.mode == 'predict') == (args.training is not None), 'Prediction requires a training receipt')
    require(0 <= args.cache_mib <= 1024, 'CPU cache must be between 0 and 1024 MiB')
    guard = install_fold_training_guard(DATA, args.fold) if args.mode == 'train' else install_mask_open_guard(DATA)
    require_environment_flags()
    require(torch.backends.mps.is_available(), 'MPS unavailable')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(2)
    preflight, sha = load_preflight(args.preflight)
    require(environment_identity() == preflight['environment'], 'MPS/runtime environment differs from capture')
    if args.mode == 'train':
        train(preflight, sha, args.fold, args.arm, out, guard, cache_mib=args.cache_mib)
    else:
        predict(preflight, sha, args.fold, args.arm, args.training, out, guard)
    print(json.dumps({'passed': True, 'mode': args.mode, 'fold': args.fold, 'arm': args.arm, 'output': str(out)}))


if __name__ == '__main__':
    main()
