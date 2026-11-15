#!/usr/bin/env python3
"""Freeze a matched plan, check parity, or train one fixed-read arm.

This exploratory study never runs video propagation. Every arm is initialized
from scratch, uses the same training order, and freezes all validation masks
without opening validation labels. No GPU work is launched by the plan mode.
"""
import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from datasets.io import save_binary_mask
from modeling.lean_channel_attention import author_sha256
from scripts.memory_token_io import digest, validate_output
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_training_label_guard_v1 import install_training_mask_guard
from scripts.source_group_boundary_shared_v1 import (ARMS, EPOCHS, LOSS, PARAMETERS, SEED, dependencies,
    ensure_target, initial_hash, load_prepared, model_for, read, schedule)
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture, to_video
from scripts.train_region_mixed_loss_v1 import environment_identity, pixel_loss
from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss


def matched_loss(arm, prediction, target):
    if arm not in ARMS:
        raise ValueError('Unknown matched objective')
    return cmsa_boundary_loss(prediction, target) if arm.endswith('_boundary') else pixel_loss(prediction, target, 'bce')

PARITY_KEYS = (('seq2', 4), ('seq4', 18), ('seq4', 24))


def inputs(item, device):
    early, deep, logits, shape = read_capture(item)
    return (early.to(device), deep.to(device), logits.to(device)), shape


def build_plan(manifest, manifest_sha):
    hashes = []
    for arm in ARMS:
        torch.manual_seed(SEED)
        hashes.append(initial_hash(model_for(arm)))
    if len(set(hashes)) != 1:
        raise ValueError('Matched arms have different initial parameter bytes')
    return {'passed': True, 'scope': 'Exploratory, previously inspected source-group fixed reads.',
            'manifest_sha256': manifest_sha, 'feature_pipeline_sha256': manifest['feature_pipeline_sha256'],
            'dependency_sha256': dependencies(), 'author_sha256': author_sha256(),
            'environment': manifest['feature_pipeline']['runtime_environment'],
            'arms': list(ARMS), 'seed': SEED, 'epochs': EPOCHS, 'parameter_count': PARAMETERS,
            'optimizer': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001, 'loss': LOSS,
            'initial_state_sha256': hashes[0], 'training_order': schedule(manifest['samples']),
            'parity_keys': [list(x) for x in PARITY_KEYS],
            'evaluation': 'Freeze every arm and replay all validation predictions before reading validation labels.',
            'limitations': ['A single seed and previously inspected validation videos are exploratory.',
                            'This fixed-read study does not establish safe or effective video propagation.']}


def require_plan(path, manifest, manifest_sha):
    saved = read(path); expected = build_plan(manifest, manifest_sha)
    if saved != expected:
        raise ValueError('Frozen matched plan or source identity changed')
    return saved


def parity(manifest, manifest_sha, plan_path, output):
    plan = require_plan(plan_path, manifest, manifest_sha)
    if environment_identity() != plan['environment']:
        raise ValueError('Live environment differs from capture')
    checks = []
    for arm in ARMS:
        torch.manual_seed(SEED); cpu = model_for(arm).train()
        # Exercise gradients beyond the zero-initialized output projection.
        torch.nn.init.normal_(cpu.head[-1].weight, mean=0, std=0.01)
        gpu = model_for(arm).train(); gpu.load_state_dict(cpu.state_dict()); gpu.to('mps')
        for seq, frame in PARITY_KEYS:
            item = next(x for x in manifest['samples'] if x['sequence']==seq and x['frame']==frame and x['split']=='train')
            target = None; values = []
            for model, device in ((cpu, 'cpu'), (gpu, 'mps')):
                model.zero_grad(set_to_none=True); args, shape = inputs(item, device)
                if target is None: target = ensure_target(item, shape)
                pred = model(*args); loss = matched_loss(arm, to_video(pred, shape), target.to(device))
                if not torch.isfinite(pred).all() or not torch.isfinite(loss): raise ValueError('Nonfinite parity output')
                loss.backward(); grads = [p.grad.detach().cpu().clone() for p in model.parameters()]
                if not all(torch.isfinite(g).all() for g in grads): raise ValueError('Nonfinite parity gradient')
                values.append((pred.detach().cpu(), loss.detach().cpu(), grads, (to_video(pred.detach(), shape)>0).cpu()))
            torch.testing.assert_close(values[0][0], values[1][0], rtol=0, atol=0.001)
            torch.testing.assert_close(values[0][1], values[1][1], rtol=0.001, atol=0.001)
            logit_difference = float((values[0][0]-values[1][0]).abs().max())
            loss_difference = float((values[0][1]-values[1][1]).abs())
            loss_ratio = loss_difference/(0.001+0.001*abs(float(values[1][1])))
            gradient_difference = gradient_ratio = 0.0
            for left, right in zip(values[0][2], values[1][2], strict=True):
                torch.testing.assert_close(left, right, rtol=0.001, atol=0.001)
                delta = (left.double()-right.double()).abs()
                gradient_difference = max(gradient_difference, float(delta.max()))
                gradient_ratio = max(gradient_ratio, float((delta/(0.001+0.001*right.double().abs())).max()))
            changed = int((values[0][3]!=values[1][3]).sum()); pixels = values[0][3].numel()
            if changed>20 or changed/pixels>1e-5: raise ValueError('CPU/MPS native-size masks differ')
            checks.append({'arm': arm, 'sequence': seq, 'frame': frame, 'mask_difference': changed, 'pixels': pixels,
                           'max_logit_difference': logit_difference, 'cpu_loss': float(values[0][1]),
                           'mps_loss': float(values[1][1]), 'loss_difference': loss_difference,
                           'loss_tolerance_ratio': loss_ratio, 'max_gradient_difference': gradient_difference,
                           'max_gradient_tolerance_ratio': gradient_ratio})
        del cpu, gpu
    out = validate_output(output); out.mkdir(parents=True, exist_ok=False)
    (out/'parity.json').write_text(json.dumps({'passed': True, 'scope': 'Training reads only; CPU/MPS numerical check, not video inference.',
        'manifest_sha256': manifest_sha, 'plan_sha256': digest(plan_path), 'environment': environment_identity(),
        'checks': checks, 'logit_atol': 0.001, 'gradient_atol_rtol': 0.001,
        'mask_max_pixels': 20, 'mask_max_fraction': 1e-5}, indent=2)+'\n')


def require_parity(path, plan, plan_path, manifest_sha):
    r = read(path)
    if (not r.get('passed') or r.get('manifest_sha256') != manifest_sha or r.get('plan_sha256') != digest(plan_path)
            or r.get('environment') != plan['environment'] or r.get('logit_atol') != 0.001
            or r.get('gradient_atol_rtol') != 0.001 or r.get('mask_max_pixels') != 20
            or r.get('mask_max_fraction') != 1e-5
            or {(x['arm'], x['sequence'], x['frame']) for x in r.get('checks', [])} != {(a, s, f) for a in ARMS for s, f in PARITY_KEYS}
            or len(r.get('checks', [])) != len(ARMS)*len(PARITY_KEYS)
            or any(x['pixels'] <= 0 or x['mask_difference'] > 20 or x['mask_difference']/x['pixels'] > 1e-5
                   or not all(np.isfinite(x[k]) for k in ('max_logit_difference', 'cpu_loss', 'mps_loss', 'loss_difference',
                                                           'loss_tolerance_ratio', 'max_gradient_difference', 'max_gradient_tolerance_ratio'))
                   or x['max_logit_difference'] > 0.001 or x['loss_tolerance_ratio'] > 1.00001
                   or x['max_gradient_tolerance_ratio'] > 1.00001 for x in r['checks'])):
        raise ValueError('Training-read parity receipt differs')


def train(manifest, manifest_sha, arm, plan_path, parity_path, output, guard):
    plan = require_plan(plan_path, manifest, manifest_sha); require_parity(parity_path, plan, plan_path, manifest_sha)
    if environment_identity() != plan['environment']: raise ValueError('Live environment differs from capture')
    samples = manifest['samples']; trainrows = {(x['sequence'], x['frame']): x for x in samples if x['split']=='train'}
    valrows = sorted((x for x in samples if x['split']=='validation'), key=lambda x: (x['sequence'], x['frame']))
    out = validate_output(output); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED); torch.mps.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    model = model_for(arm).train()
    initial = initial_hash(model)
    if initial != plan['initial_state_sha256']: raise ValueError('Arm initial parameters differ')
    model.to('mps'); optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
    history = []
    for epoch, order in enumerate(plan['training_order']):
        values = []; actual_order = []
        for seq, frame in order:
            item = trainrows[(seq, frame)]; args, shape = inputs(item, 'mps'); target = ensure_target(item, shape).to('mps')
            optimizer.zero_grad(set_to_none=True); loss = matched_loss(arm, to_video(model(*args), shape), target)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite training gradient')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()): raise ValueError('Nonfinite parameter')
            values.append(float(loss.detach().cpu())); actual_order.append([item['sequence'], item['frame']])
        history.append({'epoch': epoch+1, 'samples': len(values), 'sample_order': actual_order,
                        'mean_training_loss': float(np.mean(values))})
        (out/'progress.json').write_text(json.dumps({'history': history}, indent=2)+'\n')
        print(f'{arm}: epoch {epoch+1}/{EPOCHS}', flush=True)
    payload = {'format': 'foreground_conditioned_refiner_v1', 'research_only': True,
        'training_steps': len(trainrows)*EPOCHS, 'state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()},
        'backbone_sha256': manifest['checkpoint_sha256'], 'feature_pipeline_sha256': manifest['feature_pipeline_sha256'],
        'train_manifest_sha256': manifest_sha, 'author_sha256': author_sha256(),
        'conditioned_source_sha256': digest(ROOT/'modeling/foreground_conditioned_refiner.py'),
        'base_source_sha256': digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
        'gate_mode': model.gate_mode, 'contrast_head': model.contrast_head,
        'arm': 'mask_only' if arm.rsplit('_', 1)[0]=='mask_only' else 'multilevel', 'study_arm': arm,
        'plan_sha256': digest(plan_path), 'parity_sha256': digest(parity_path), 'initial_state_sha256': initial,
        'dependency_sha256': dependencies(), 'environment': environment_identity(), 'seed': SEED,
        'epochs': EPOCHS, 'parameter_count': PARAMETERS, 'loss': LOSS[arm],
        'optimizer': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001}
    checkpoint = out/'checkpoint.pt'; torch.save(payload, checkpoint)
    fitted = model_for(arm); fitted.load_trained_state(torch.load(checkpoint, map_location='cpu', weights_only=True),
        backbone_sha256=manifest['checkpoint_sha256'], feature_pipeline_sha256=manifest['feature_pipeline_sha256'], train_manifest_sha256=manifest_sha)
    fitted.to('mps').eval(); folder = out/'frozen_predictions'; folder.mkdir(); rows = []
    with torch.no_grad():
        for item in valrows:
            args, shape = inputs(item, 'mps')
            masks = {'baseline': (to_video(args[2], shape)>0).cpu().numpy()[0,0],
                     'refined': (to_video(fitted(*args), shape)>0).cpu().numpy()[0,0]}
            row = {'sequence': item['sequence'], 'frame': item['frame'], 'shape': list(shape)}
            for name, mask in masks.items():
                file = folder/f"{item['sequence']}_{item['frame']:06d}_{name}.png"; save_binary_mask(mask, file)
                row[name] = str(file); row[name+'_sha256'] = digest(file)
            rows.append(row)
    if guard['blocked_open_attempts'] != 0: raise ValueError('Non-training mask open attempted')
    frozen = out/'frozen_predictions.json'; frozen.write_text(json.dumps({'checkpoint_sha256': digest(checkpoint), 'rows': rows}, indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed': True, 'arm': arm, 'manifest_sha256': manifest_sha,
        'plan_sha256': digest(plan_path), 'parity_sha256': digest(parity_path), 'checkpoint_sha256': digest(checkpoint),
        'frozen_predictions_sha256': digest(frozen), 'initial_state_sha256': initial,
        'training_order': [x['sample_order'] for x in history], 'history': history, 'label_guard': dict(guard)}, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('plan', 'parity', 'train'), required=True)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path); p.add_argument('--parity', type=Path); p.add_argument('--arm', choices=ARMS)
    a = p.parse_args()
    guard = (install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl') if a.mode=='plan'
             else install_training_mask_guard(ROOT/'data/polypgen_sequence.jsonl'))
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    manifest, manifest_sha = load_prepared(a.manifest)
    if a.mode == 'plan':
        plan = build_plan(manifest, manifest_sha); out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
        (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    elif a.mode == 'parity':
        if not a.plan: raise ValueError('Parity requires the frozen plan')
        parity(manifest, manifest_sha, a.plan, a.output)
    else:
        if not a.plan or not a.parity or not a.arm: raise ValueError('Training requires arm, plan and parity')
        train(manifest, manifest_sha, a.arm, a.plan, a.parity, a.output, guard)


if __name__ == '__main__':
    main()
