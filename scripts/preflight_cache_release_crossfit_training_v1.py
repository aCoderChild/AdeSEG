#!/usr/bin/env python3
"""Validate mixed eager/cache-release cross-fit inputs without authorizing training.

This deliberately has a distinct output format from the frozen eager
preflight. The bounded handoff, its full seq6/seq11 audits and all eager
captures are reconstructed under a target-label-open guard. This is a
preflight only: an independently reviewed bounded-aware trainer, predictor
and evaluator are still required before this evidence may be used to train.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.certify_crossfit_warm_gate_v2 import reconstruct_gate, verify_certificate
from scripts.channel_value_fixed_read_v1 import model_for
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, EXISTING, FRESH, PLAN_SHA, validate_plan
from scripts.crossfit_training_contract_v1 import fold_split, patient_balanced_schedule, validate_warm_index_structure
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.preflight_warm_crossfit_training_v1 import ARMS, EPOCHS, SEED, UPDATES_PER_PATIENT
from scripts.preflight_warm_crossfit_training_v1 import sources as eager_training_sources
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v2 import initial_hash, load_prepared
from scripts.verify_cache_release_crossfit_handoff_v1 import CACHE_RELEASE, EAGER, build as rebuild_handoff


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bound_json(path):
    path = Path(path).resolve(strict=True)
    before = digest(path)
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict) and digest(path) == before, f'Changed or invalid JSON: {path}')
    return value, before


def sources():
    result = eager_training_sources()
    for name in local_import_closure(ROOT, ('scripts/preflight_cache_release_crossfit_training_v1.py',)):
        value = digest(ROOT / name)
        require(name not in result or result[name] == value, f'Conflicting source identity: {name}')
        result[name] = value
    return result


def structural_check(plan, old, handoff, certificate, certificate_path, certificate_sha, gate_path, gate_sha):
    """Reuse the frozen 214+143 checker on an explicitly in-memory projection.

    The projection is never written as an aggregate and cannot make the
    original eager trainer accept cache-release data. Each cache-release proof
    must have been independently reconstructed before calling this function.
    """
    require(handoff.get('index_plan_sha256') == PLAN_SHA and
            handoff.get('existing_manifest_sha256') == plan['existing_manifest_sha256'] and
            handoff.get('existing_feature_pipeline_sha256') == plan['feature_pipeline_sha256'],
            'Mixed handoff plan or historical feature identity differs')
    projected = []
    for original in handoff['observations']:
        row = dict(original)
        sequence = row['sequence']
        expected_kind = ('historical_warm_reference' if sequence == 'seq6' else
                         'matched_cache_release_reference' if sequence in CACHE_RELEASE else
                         'matched_warm_reference')
        require(row.get('reference_kind') == expected_kind, 'Mixed capture reference kind differs')
        if sequence in CACHE_RELEASE:
            require(all(row.get(f'{mode}_sidecar_{part}') for mode in ('reference', 'capture') for part in ('path', 'sha256')) and
                    all(row.get(f'{mode}_{kind}_trace_{part}') for mode in ('reference', 'capture')
                        for kind in ('setup', 'frame') for part in ('path', 'sha256')) and
                    row.get('seq6_proof_path') == handoff['seq6_proof_path'] and
                    row.get('seq6_proof_sha256') == handoff['seq6_proof_sha256'] and
                    row.get('memory_gate_path') == handoff['memory_gate_path'] and
                    row.get('memory_gate_sha256') == handoff['memory_gate_sha256'],
                    'Cache-release proof, sidecar or trace binding differs')
            row['reference_kind'] = 'matched_warm_reference'
        else:
            require(not any(k in row for k in ('reference_sidecar_path', 'capture_sidecar_path',
                                               'seq6_proof_path', 'memory_gate_path')),
                    'Unexpected cache-release proof on eager row')
        row.update(warm_gate_path=str(gate_path), warm_gate_sha256=gate_sha,
                   certificate_path=str(certificate_path), certificate_sha256=certificate_sha)
        projected.append(row)
    projection = {'passed': True, 'plan_sha256': PLAN_SHA, 'index_plan_sha256': PLAN_SHA,
                  'warm_preflight_sha256': certificate['warm_plan_sha256'],
                  'warm_gate_path': str(gate_path), 'warm_gate_sha256': gate_sha,
                  'certificate_path': str(certificate_path), 'certificate_sha256': certificate_sha,
                  'existing_manifest_sha256': handoff['existing_manifest_sha256'],
                  'existing_feature_pipeline_sha256': handoff['existing_feature_pipeline_sha256'],
                  'observations': projected}
    return validate_warm_index_structure(plan, old, projection, index_sha256=PLAN_SHA,
        warm_preflight_sha256=certificate['warm_plan_sha256'], warm_gate_path=gate_path,
        warm_gate_sha256=gate_sha, certificate_path=certificate_path, certificate_sha256=certificate_sha)


def verified_inputs(folder):
    folder = Path(folder).resolve(strict=True)
    require(folder.is_relative_to(BASE.resolve(strict=True)), 'Handoff outside Drive experiment root')
    handoff_path = folder / 'handoff.json'
    completion, completion_sha = read_bound_json(folder / 'completion.json')
    handoff, handoff_sha = read_bound_json(handoff_path)
    require(completion.get('passed') is True and completion.get('training_authorized') is False and
            completion.get('handoff_sha256') == handoff_sha and
            completion.get('label_guard', {}).get('open_attempts') == 0 and
            handoff.get('passed') is True and handoff.get('training_authorized') is False and
            handoff.get('format') == 'cache_release_crossfit_handoff_v1' and
            handoff.get('label_guard', {}).get('open_attempts') == 0,
            'Mixed handoff or non-training completion differs')
    plan = validate_plan()
    require(set(handoff['audits']) == set(FRESH), 'Mixed handoff audit coverage differs')
    eager_audits = [Path(handoff['audits'][sequence]['path']) for sequence in FRESH if sequence in EAGER]
    cache_audits = [Path(handoff['audits'][sequence]['path']) for sequence in FRESH if sequence in CACHE_RELEASE]
    certificate_path = Path(handoff['warm_capture_certificate_path']).resolve(strict=True)
    gate_path = Path(handoff['short_gate_path']).resolve(strict=True)
    memory_path = Path(handoff['memory_gate_path']).resolve(strict=True)
    seq6_path = Path(handoff['seq6_proof_path']).resolve(strict=True)
    reconstructed = rebuild_handoff(certificate_path, gate_path, memory_path, seq6_path, eager_audits, cache_audits)
    require({k: v for k, v in handoff.items() if k != 'label_guard'} == reconstructed,
            'Mixed handoff failed independent reconstruction')
    certificate, certificate_sha = verify_certificate(certificate_path)
    require(certificate_sha == handoff['warm_capture_certificate_sha256'], 'Warm certificate changed')
    warm_gate_path = Path(certificate['warm_gate_path']).resolve(strict=True)
    _, warm_gate_sha = reconstruct_gate(warm_gate_path)
    require(warm_gate_sha == certificate['warm_gate_sha256'], 'Warm gate changed')
    old, old_sha = load_prepared(EXISTING)
    require(old_sha == plan['existing_manifest_sha256'], 'Historical 214-read manifest changed')
    structural_check(plan, old, handoff, certificate, certificate_path, certificate_sha,
                     warm_gate_path, warm_gate_sha)
    require(digest(handoff_path) == handoff_sha and digest(folder / 'completion.json') == completion_sha,
            'Mixed handoff changed during preflight')
    return plan, old, handoff, handoff_sha, certificate_sha, warm_gate_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(BASE.resolve(strict=True)) and not out.exists(), 'Invalid output')
    guard = install_mask_open_guard(DATA)
    source_identity = sources()
    plan, old, handoff, handoff_sha, certificate_sha, warm_gate_sha = verified_inputs(args.handoff)
    items = plan['existing_reads'] + plan['fresh_reads']
    folds = []
    for fold in range(1, 6):
        train, validation, test = fold_split(fold)
        order = patient_balanced_schedule(items, fold, SEED, EPOCHS, UPDATES_PER_PATIENT)
        folds.append({'fold': fold, 'train': sorted(train), 'validation': sorted(validation),
                      'test': sorted(test), 'training_order': order})
    initial = {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        model = model_for(arm)
        require(sum(p.numel() for p in model.parameters()) == 63809, 'Control parameter count differs')
        initial[arm] = initial_hash(model)
    require(len(set(initial.values())) == 1, 'Matched control initialization differs')
    require(source_identity == sources() and guard['open_attempts'] == 0, 'Sources changed or target label opened')
    result = {'passed': True, 'training_authorized': False,
        'format': 'cache_release_warm_crossfit_training_preflight_v1',
        'scope': 'Mixed-source input preparation only; no training, prediction or target-label access.',
        'handoff_path': str(Path(args.handoff).resolve()), 'handoff_sha256': handoff_sha,
        'certificate_sha256': certificate_sha, 'warm_gate_sha256': warm_gate_sha,
        'short_gate_sha256': handoff['short_gate_sha256'], 'seq6_proof_sha256': handoff['seq6_proof_sha256'],
        'memory_gate_sha256': handoff['memory_gate_sha256'], 'index_plan_sha256': PLAN_SHA,
        'old_manifest_sha256': plan['existing_manifest_sha256'], 'backbone_sha256': plan['checkpoint_sha256'],
        'feature_pipeline_sha256': plan['feature_pipeline_sha256'], 'environment': old['feature_pipeline']['runtime_environment'],
        'source_sha256': source_identity, 'seed': SEED, 'epochs': EPOCHS,
        'updates_per_patient_per_epoch': UPDATES_PER_PATIENT, 'arms': list(ARMS),
        'initial_state_sha256': initial, 'optimizer': {'name': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001},
        'loss': 'corrected CMSA boundary weighted BCE and IoU plus soft Dice', 'folds': folds,
        'label_guard': dict(guard), 'limitations': [
            'Previously inspected patients; not blind evaluation.',
            'Seven sequences use a cache-release loader; full eager equivalence is unavailable.',
            'A separate bounded-aware trainer, prediction, verification and audit are still required.']}
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'plan.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'training_authorized': False,
        'plan_sha256': digest(path), 'label_guard': dict(guard)}, indent=2) + '\n')
    print(json.dumps({'passed': True, 'training_authorized': False, 'output': str(out)}))


if __name__ == '__main__':
    main()
