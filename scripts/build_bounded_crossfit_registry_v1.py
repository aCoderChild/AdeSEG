#!/usr/bin/env python3
"""Build a mixed-source read-only sample registry; never launch training.

The bounded-aware preflight and its entire handoff proof chain are
reconstructed first. This registry provides path/ordinal mappings for a
future separately reviewed training worker, but it neither grants training
authorization nor opens target masks.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, PLAN_SHA
from scripts.crossfit_training_contract_v1 import fold_split, patient_balanced_schedule
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.preflight_bounded_crossfit_training_v1 import read_bound_json, require, verified_inputs
from scripts.preflight_bounded_crossfit_training_v1 import sources as preflight_sources
from scripts.preflight_warm_crossfit_training_v1 import ARMS, EPOCHS, SEED, UPDATES_PER_PATIENT
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def sources():
    result = preflight_sources()
    for name in local_import_closure(ROOT, ('scripts/build_bounded_crossfit_registry_v1.py',)):
        value = digest(ROOT / name)
        require(name not in result or result[name] == value, f'Conflicting source identity: {name}')
        result[name] = value
    return result


def attach_rows(index, old, handoff, videos):
    """Map audited captures to dataset label paths without decoding labels."""
    rows = {}
    for item in old['samples']:
        key = (item['sequence'], item['frame'])
        require(key not in rows, 'Duplicate historical capture')
        row = dict(item); row['origin'] = 'historical'
        rows[key] = row
    require(len(rows) == 214, 'Historical sample count differs')
    for item in handoff['observations']:
        key = (item['sequence'], item['frame'])
        require(key not in rows, 'Duplicate or overlapping fresh capture')
        frame = videos[key[0]][key[1]]
        require(frame.frame_index == item['source_frame_index'], 'Dataset source frame differs')
        row = dict(item)
        row.update(origin='warm', feature=item['feature_file'], snapshot=item['snapshot_file'],
                   label=str(frame.mask_path))
        rows[key] = row
    require(len(rows) == 357, 'Mixed sample count differs')
    expected = {(x['sequence'], x['frame']) for x in index['existing_reads'] + index['fresh_reads']}
    require(len(expected) == 357 and set(rows) == expected, 'Mixed sample grid differs')
    for (sequence, ordinal), row in rows.items():
        frame = videos[sequence][ordinal]
        source = index['source_frames'][sequence][ordinal]
        require(frame.frame_index == source['source_frame_index'] == row['source_frame_index'] and
                os.path.realpath(frame.mask_path) == os.path.realpath(row['label']),
                'Label path or source-frame index differs')
    return rows


def build(preflight_folder):
    folder = Path(preflight_folder).resolve(strict=True)
    require(folder.is_relative_to(BASE.resolve(strict=True)), 'Preflight outside Drive experiment root')
    plan_path, completion_path = folder / 'plan.json', folder / 'completion.json'
    preflight, preflight_sha = read_bound_json(plan_path)
    completion, completion_sha = read_bound_json(completion_path)
    require(preflight.get('passed') is True and preflight.get('training_authorized') is False and
            preflight.get('format') == 'bounded_warm_crossfit_training_preflight_v1' and
            preflight.get('label_guard', {}).get('open_attempts') == 0 and
            completion.get('passed') is True and completion.get('training_authorized') is False and
            completion.get('label_guard', {}).get('open_attempts') == 0 and
            completion.get('plan_sha256') == preflight_sha and
            preflight.get('source_sha256') == preflight_sources() and
            preflight.get('index_plan_sha256') == PLAN_SHA and
            preflight.get('seed') == SEED and preflight.get('epochs') == EPOCHS and
            preflight.get('updates_per_patient_per_epoch') == UPDATES_PER_PATIENT and
            preflight.get('arms') == list(ARMS), 'Mixed preflight or source identity differs')
    index, old, handoff, handoff_sha, certificate_sha, warm_gate_sha = verified_inputs(preflight['handoff_path'])
    require(preflight['handoff_sha256'] == handoff_sha and preflight['certificate_sha256'] == certificate_sha and
            preflight['warm_gate_sha256'] == warm_gate_sha and
            preflight['short_gate_sha256'] == handoff['short_gate_sha256'] and
            preflight['seq6_proof_sha256'] == handoff['seq6_proof_sha256'] and
            preflight['seq11_proof_sha256'] == handoff['seq11_proof_sha256'] and
            preflight['old_manifest_sha256'] == index['existing_manifest_sha256'] and
            preflight['feature_pipeline_sha256'] == index['feature_pipeline_sha256'] and
            preflight['backbone_sha256'] == index['checkpoint_sha256'], 'Mixed preflight inputs changed')
    initial = preflight.get('initial_state_sha256', {})
    require(isinstance(initial, dict) and set(initial) == set(ARMS) and len(set(initial.values())) == 1 and
            all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in initial.values()) and
            preflight.get('optimizer') == {'name': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001} and
            preflight.get('loss') == 'corrected CMSA boundary weighted BCE and IoU plus soft Dice' and
            preflight.get('environment') == old['feature_pipeline']['runtime_environment'],
            'Initialization, optimizer, loss or environment metadata differs')
    require(digest(DATA) == index['data_manifest_sha256'], 'Dataset manifest changed')
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}
    rows = attach_rows(index, old, handoff, videos)
    folds = []
    for fold in range(1, 6):
        train, validation, test = fold_split(fold)
        order = patient_balanced_schedule(list(rows.values()), fold, SEED, EPOCHS, UPDATES_PER_PATIENT)
        folds.append({'fold': fold, 'train': sorted(train), 'validation': sorted(validation),
                      'test': sorted(test), 'training_order': order})
    require(folds == preflight['folds'], 'Fold partition or update schedule differs')
    require(digest(DATA) == index['data_manifest_sha256'] and digest(plan_path) == preflight_sha and
            digest(completion_path) == completion_sha, 'Input changed during registry construction')
    return {'passed': True, 'training_authorized': False, 'format': 'bounded_crossfit_registry_v1',
        'scope': 'Path and fold registry only; no training, predictions or target-label access.',
        'preflight_path': str(folder), 'preflight_sha256': preflight_sha,
        'preflight_completion_sha256': completion_sha,
        'handoff_path': preflight['handoff_path'], 'handoff_sha256': handoff_sha,
        'certificate_sha256': certificate_sha, 'warm_gate_sha256': warm_gate_sha,
        'short_gate_sha256': preflight['short_gate_sha256'], 'seq6_proof_sha256': preflight['seq6_proof_sha256'],
        'seq11_proof_sha256': preflight['seq11_proof_sha256'], 'index_plan_sha256': PLAN_SHA,
        'data_manifest_sha256': index['data_manifest_sha256'], 'source_sha256': sources(),
        'backbone_sha256': preflight['backbone_sha256'],
        'feature_pipeline_sha256': preflight['feature_pipeline_sha256'],
        'initial_state_sha256': initial, 'optimizer': preflight['optimizer'], 'loss': preflight['loss'],
        'environment': preflight['environment'],
        'historical_samples': 214, 'fresh_samples': 143,
        'samples': [rows[key] for key in sorted(rows, key=lambda x: (int(x[0][3:]), x[1]))],
        'folds': folds,
        'limitations': ['Previously inspected patients; not blind evaluation.',
            'Initial model hashes are checked for format and equality only, not recomputed by this registry.',
            'A separately reviewed bounded-aware trainer and scoring pipeline are still required.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(BASE.resolve(strict=True)) and not out.exists(), 'Invalid output')
    guard = install_mask_open_guard(DATA)
    frozen_sources = sources()
    result = build(args.preflight)
    require(guard['open_attempts'] == 0 and result['source_sha256'] == frozen_sources == sources(),
            'Sources changed or target label opened')
    result['label_guard'] = dict(guard)
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'registry.json'; path.write_text(json.dumps(result, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'training_authorized': False,
        'registry_sha256': digest(path), 'label_guard': dict(guard)}, indent=2) + '\n')
    print(json.dumps({'passed': True, 'training_authorized': False, 'output': str(out)}))


if __name__ == '__main__':
    main()
