#!/usr/bin/env python3
"""Predict held-out fixed reads using a source-bound bounded fold checkpoint.

No target label is opened. This separate path is disabled pending independent
review and a completed, audited mixed-source training study.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import bounded_crossfit_shared_worker_3fold_v1 as worker
from scripts.bounded_crossfit_supervision_3fold_v1 import assert_supervised
from scripts.channel_value_fixed_read_v1 import model_for, require_environment_flags
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, phase_artifact, validate_plan
from scripts.crossfit_training_contract_v1 import restore_view_mask
from scripts.memory_token_io import digest
from scripts.preflight_bounded_crossfit_training_v1 import read_bound_json, require
from scripts.preflight_warm_crossfit_training_v1 import ARMS
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.train_multilevel_refiner_seq4_empty_v2 import to_video
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.warm_crossfit_fold_training_v1 import inputs
from scripts.bounded_crossfit_fold_3fold_v1 import fold_definition


ENABLE_REVIEWED_PREDICTION = True


def checked_fold(batch, fold, arm, authorization, registry):
    batch = Path(batch).resolve(strict=True)
    require(batch.is_relative_to(BASE.resolve(strict=True)) and type(fold) is int and 1 <= fold <= 3 and arm in ARMS,
            'Invalid bounded fold or batch')
    folder = batch / f'fold{fold}'
    group_path = folder / 'group.json'
    group_sha = phase_artifact(batch, f'train_f{fold}', group_path)
    group, read_sha = read_bound_json(group_path)
    require(group_sha == read_sha and group.get('passed') is True and group.get('fold') == fold and
            group.get('authorization') == authorization and group.get('label_guard', {}).get('blocked_open_attempts') == 0 and
            set(group.get('arms', {})) == set(ARMS), 'Bounded training group differs')
    item = group['arms'][arm]; record_path = folder / arm / 'completion.json'; checkpoint = folder / arm / 'checkpoint.pt'
    record, record_sha = read_bound_json(record_path)
    manifest, _ = read_bound_json(batch / 'run_manifest.json')
    require(type(manifest.get('cache_mib')) is int and 0 <= manifest['cache_mib'] <= 1024,
            'Bounded cache limit differs')
    rows = {(x['sequence'], x['frame']): x for x in registry['samples']}
    require(len(rows) == 357, 'Mixed sample coverage differs')
    _, _, test, order = fold_definition(registry, fold, rows)
    require(item.get('path') == str(record_path) and item.get('sha256') == record_sha and
            record.get('passed') is True and record.get('fold') == fold and record.get('arm') == arm and
            record.get('authorization') == authorization and record.get('device') == 'mps' and
            record.get('environment') == registry['environment'] and
            record.get('initial_state_sha256') == registry['initial_state_sha256'][arm] and
            record.get('training_steps') == sum(map(len, order)) and
            record.get('label_guard', {}).get('blocked_open_attempts') == 0 and
            record.get('cache') == group.get('cache') and record.get('label_guard') == group.get('label_guard') and
            record.get('cache', {}).get('limit_bytes') == manifest['cache_mib'] * 1024**2 and
            type(record['cache'].get('peak_bytes')) is int and
            0 <= record['cache']['peak_bytes'] <= manifest['cache_mib'] * 1024**2 and
            type(record['cache'].get('hits')) is int and type(record['cache'].get('misses')) is int and
            record['cache']['hits'] >= 0 and record['cache']['misses'] >= 0 and
            record['cache']['hits'] + record['cache']['misses'] == record['training_steps'] and
            type(record['cache'].get('unique_reads')) is int and
            0 < record['cache']['unique_reads'] <= record['cache']['misses'] and
            record['cache'].get('final_sha256_verified') is True and
            digest(checkpoint) == record.get('checkpoint_sha256') == item.get('checkpoint_sha256'),
            'Bounded training receipt or checkpoint differs')
    checkpoint_sha = digest(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(digest(checkpoint) == checkpoint_sha and payload.get('crossfit_fold') == fold and
            payload.get('crossfit_arm') == arm and payload.get('training_steps') == record['training_steps'] and
            payload.get('bounded_registry_sha256') == authorization['registry_sha256'] and
            payload.get('aggregate_sha256') == authorization['handoff_sha256'] and
            payload.get('source_sha256_map') == authorization['source_sha256'], 'Bounded checkpoint provenance differs')
    model = model_for(arm)
    model.load_trained_state(payload, backbone_sha256=registry['backbone_sha256'],
        feature_pipeline_sha256=registry['feature_pipeline_sha256'],
        train_manifest_sha256=authorization['preflight_sha256'])
    return model, rows, test, group_sha, record_sha, checkpoint_sha


def predict(batch, fold, arm, authorization, registry, output, guard):
    require(ENABLE_REVIEWED_PREDICTION, 'Bounded prediction awaits independent review and real trained folds')
    assert_supervised(batch, f'predict_f{fold}_{arm}', output)
    model, rows, test, group_sha, record_sha, checkpoint_sha = checked_fold(batch, fold, arm, authorization, registry)
    index = validate_plan(); model.to('mps').eval().requires_grad_(False)
    output.mkdir(parents=True, exist_ok=False); saved = []
    with torch.no_grad():
        for key in sorted(rows):
            if key[0] not in test:
                continue
            values, shape = inputs(rows[key], 'mps')
            refined = model(*values)
            require(torch.isfinite(refined).all(), 'Nonfinite bounded prediction')
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
    require(guard['open_attempts'] == 0 and worker.sources() == authorization['source_sha256'] and
            phase_artifact(batch, f'train_f{fold}', Path(batch) / f'fold{fold}' / 'group.json') == group_sha and
            digest(Path(batch) / f'fold{fold}' / arm / 'completion.json') == record_sha and
            digest(Path(batch) / f'fold{fold}' / arm / 'checkpoint.pt') == checkpoint_sha,
            'Target opened or bounded prediction sources changed')
    result = {'passed': True, 'format': 'bounded_crossfit_prediction_3fold_v1', 'fold': fold, 'arm': arm,
        'authorization': authorization, 'group_sha256': group_sha, 'training_completion_sha256': record_sha,
        'checkpoint_sha256': checkpoint_sha, 'rows': saved, 'device': 'mps', 'environment': registry['environment'],
        'label_guard': dict(guard)}
    (output / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--authorization', required=True, type=Path)
    parser.add_argument('--fold', required=True, type=int)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); output = args.output.resolve()
    require(output.is_relative_to(BASE.resolve(strict=True)) and not output.exists(), 'Invalid output')
    require(ENABLE_REVIEWED_PREDICTION, 'Bounded prediction awaits independent review and real trained folds')
    assert_supervised(args.batch, f'predict_f{args.fold}_{args.arm}', output)
    guard = install_mask_open_guard(DATA)
    saved, authorization_sha = read_bound_json(worker.path_under_drive(args.authorization))
    require(saved.get('label_guard', {}).get('open_attempts') == 0, 'Authorization label guard differs')
    authorization = {k: v for k, v in saved.items() if k != 'label_guard'}
    current = worker.authorize(authorization['registry_path'], authorization['verification_path'],
                               authorization['probe_path'], authorization['probe_audit_path'])
    require(current == authorization and digest(args.authorization) == authorization_sha, 'Authorization changed')
    registry, registry_sha = read_bound_json(Path(authorization['registry_path']) / 'registry.json')
    require(registry_sha == authorization['registry_sha256'], 'Mixed registry changed')
    require_environment_flags(); require(torch.backends.mps.is_available(), 'MPS unavailable')
    require(environment_identity() == registry['environment'], 'Runtime environment differs from capture')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(2)
    predict(args.batch, args.fold, args.arm, authorization, registry, output, guard)
    print(json.dumps({'passed': True, 'fold': args.fold, 'arm': args.arm, 'output': str(output)}))


if __name__ == '__main__':
    main()
