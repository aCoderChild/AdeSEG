#!/usr/bin/env python3
"""Validate a completed unified warm capture set without opening labels.

This preflight cannot train or authorize a live video run. It only accepts a
finished phase-bound capture aggregate and independently reconstructs the
warm compatibility gate. Outputs are written under the Drive experiment root.
"""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.crossfit_observer_capture_shared_v1 import (BASE, DATA, EXISTING,
    PLAN_SHA, archived_frame_metadata, read, require, validate_plan)
from scripts.crossfit_training_contract_v1 import (fold_split, patient_balanced_schedule,
    validate_warm_index_structure)
from scripts.certify_crossfit_warm_gate_v2 import reconstruct_gate, verify_certificate
from scripts.memory_token_io import digest
from scripts.local_import_closure_v1 import local_import_closure
from scripts.run_crossfit_warm_video_study_v2 import checked_audits, certified_seq6
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v2 import load_prepared
from scripts.source_group_refiner_shared_v2 import initial_hash
from scripts.channel_value_fixed_read_v1 import deps as channel_dependencies, model_for
from scripts.certify_crossfit_warm_gate_v2 import sources as warm_capture_sources

SEED = 20261115
EPOCHS = 10
UPDATES_PER_PATIENT = 12
ARMS = ('mask_only', 'global', 'global_contrast', 'foreground_background_no_contrast',
        'foreground_background', 'foreground_background_asinh')
SOURCES = ('scripts/preflight_warm_crossfit_training_v1.py',
           'scripts/crossfit_training_contract_v1.py',
           'scripts/warm_crossfit_fold_training_v1.py',
           'scripts/verified_capture_cache_v1.py',
           'scripts/warm_crossfit_probe_v1.py',
           'scripts/run_warm_crossfit_study_v1.py',
           'scripts/run_warm_crossfit_shared_v1.py',
           'scripts/warm_crossfit_shared_fold_v1.py',
           'scripts/warm_crossfit_shared_probe_v1.py',
           'scripts/register_warm_crossfit_arm_v1.py',
           'scripts/score_warm_crossfit_v1.py',
           'scripts/project_archived_label_hashes_v1.py',
           'scripts/certify_crossfit_warm_gate_v2.py',
           'scripts/run_crossfit_warm_video_study_v2.py',
           'scripts/crossfit_observer_capture_shared_v1.py',
           'scripts/source_group_label_guard_v1.py',
           'scripts/source_group_training_label_guard_v1.py',
           'scripts/source_group_refiner_shared_v2.py',
           'scripts/preflight_exploratory_crossfit_capture_v2.py',
           'scripts/memory_token_io.py',
           'scripts/local_import_closure_v1.py',
           'scripts/resume_source_group_capture_v1.py',
           'scripts/run_boundary_video_factorial_v1.py',
           'datasets/video.py', 'datasets/io.py', 'modeling/seq23_viewport.py',
           'scripts/channel_value_fixed_read_v1.py',
           'scripts/train_multilevel_refiner_seq4_empty_v2.py',
           'modeling/multiscale_channel_value_refiner_v1.py',
           'modeling/cmsa_boundary_training_loss.py',
           'scripts/score_boundary_video_factorial_v1.py',
           'scripts/audit_boundary_video_factorial_v1.py',
           'protocols/EXPLORATORY_WARM_UNIFIED_CROSSFIT_V1.md')


def sources():
    result = {name: digest(ROOT / name) for name in SOURCES}
    dependencies = channel_dependencies()
    for mapping in (dependencies['current'], dependencies['prior'], warm_capture_sources()):
        for name, value in mapping.items():
            require(name not in result or result[name] == value, f'Conflicting source identity: {name}')
            result[name] = value
    for name in local_import_closure(ROOT, result):
        value = digest(ROOT / name)
        require(name not in result or result[name] == value, f'Conflicting local import: {name}')
        result[name] = value
    return result


def verified_aggregate(folder):
    """Validate independent old/new evidence; never decode an annotation."""
    folder = Path(folder).resolve(strict=True)
    require(folder.is_relative_to(BASE.resolve(strict=True)), 'Aggregate outside Drive experiment root')
    capture_path, completion_path = folder / 'captures.json', folder / 'completion.json'
    completion = read(completion_path)
    aggregate_sha = digest(capture_path)
    require(completion.get('passed') is True and completion.get('captures_sha256') == aggregate_sha and
            completion.get('label_guard', {}).get('open_attempts') == 0, 'Aggregate completion differs')
    aggregate = read(capture_path)
    require(digest(capture_path) == aggregate_sha and
            aggregate.get('prompt_policy') == 'warm_box_points_then_commit_archived_mask',
            'Aggregate or uniform warm prompt policy differs')
    plan = validate_plan()
    old_manifest, old_sha = load_prepared(EXISTING)
    require(old_sha == plan['existing_manifest_sha256'], 'Historical manifest differs')
    certificate_path = Path(aggregate['certificate_path']).resolve(strict=True)
    certificate, certificate_sha = verify_certificate(certificate_path)
    require(certificate_sha == aggregate['certificate_sha256'], 'Aggregate certificate differs')
    gate_path = Path(certificate['warm_gate_path']).resolve(strict=True)
    gate, gate_sha = reconstruct_gate(gate_path)
    require(gate_sha == certificate['warm_gate_sha256'] and gate_sha == aggregate['warm_gate_sha256'],
            'Reconstructed warm gate differs')
    validate_warm_index_structure(plan, old_manifest, aggregate, index_sha256=PLAN_SHA,
        warm_preflight_sha256=certificate['warm_plan_sha256'], warm_gate_path=gate_path,
        warm_gate_sha256=gate_sha, certificate_path=certificate_path, certificate_sha256=certificate_sha)
    require(aggregate.get('sources') == certificate['source_sha256'], 'Capture source identity differs')
    audits = aggregate['audits']
    require(set(audits) == {x['sequence'] for x in plan['fresh_reads']}, 'Capture audit coverage differs')
    require(audits['seq6'] == {'path': str(gate_path), 'sha256': gate_sha}, 'Seq6 gate audit differs')
    for sequence, binding in audits.items():
        file = Path(binding['path']).resolve(strict=True)
        require(digest(file) == binding['sha256'], f'{sequence}: audit changed')
    audit_paths = [Path(v['path']) for k, v in audits.items() if k != 'seq6']
    accepted = checked_audits(audit_paths, plan, certificate['source_sha256'], certificate_path)
    accepted['seq6'] = certified_seq6(certificate, plan)
    require(set(accepted) == set(audits), 'Accepted warm batches differ')
    by_sequence = {}
    for row in aggregate['observations']:
        by_sequence.setdefault(row['sequence'], []).append(row)
    for sequence, rows in by_sequence.items():
        if sequence == 'seq6':
            record = certificate['seq6_gate_capture']
            capture_path_for_sequence = Path(record['receipt_path']).resolve(strict=True)
            capture_sha = record['receipt_sha256']
            reference_path = Path(certificate['seq6_historical_reference']['reference_path']).resolve(strict=True)
            reference_sha = certificate['seq6_historical_reference']['reference_sha256']
        else:
            batch = accepted[sequence]
            capture_path_for_sequence = (batch / sequence / 'capture/receipt.json').resolve(strict=True)
            reference_path = (batch / sequence / 'reference/receipt.json').resolve(strict=True)
            capture_sha, reference_sha = digest(capture_path_for_sequence), digest(reference_path)
        capture = read(capture_path_for_sequence)
        require(digest(capture_path_for_sequence) == capture_sha and digest(reference_path) == reference_sha,
                f'{sequence}: receipt changed')
        observed = {item['frame']: item for item in capture['observations']}
        require(len(observed) == len(rows) and set(observed) == {item['frame'] for item in rows},
                f'{sequence}: observation coverage differs')
        metadata = archived_frame_metadata(plan, sequence)
        for item in rows:
            original = observed[item['frame']]
            require(item['capture_receipt_path'] == str(capture_path_for_sequence) and
                    item['capture_receipt_sha256'] == capture_sha and
                    item['reference_receipt_path'] == str(reference_path) and
                    item['reference_receipt_sha256'] == reference_sha and
                    item['sequence_audit_path'] == audits[sequence]['path'] and
                    item['sequence_audit_sha256'] == audits[sequence]['sha256'] and
                    item['label_metadata_path'] == metadata['path'] and
                    item['label_metadata_sha256'] == metadata['sha256'] and
                    all(item[key] == original[key] for key in
                        ('feature_file', 'feature_sha256', 'snapshot_file', 'snapshot_sha256')),
                    f'{sequence}: aggregate row differs from bound evidence')
    require(digest(capture_path) == aggregate_sha and digest(certificate_path) == certificate_sha,
            'Warm aggregate or certificate changed during verification')
    return plan, old_manifest, aggregate, aggregate_sha, certificate_sha, gate_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(BASE.resolve(strict=True)) and not out.exists(), 'Invalid output')
    guard = install_mask_open_guard(DATA)
    source_identity = sources()
    plan, old, aggregate, aggregate_sha, certificate_sha, gate_sha = verified_aggregate(args.aggregate)
    items = plan['existing_reads'] + plan['fresh_reads']
    folds = []
    for fold in range(1, 6):
        train, validation, test = fold_split(fold)
        order = patient_balanced_schedule(items, fold, SEED, EPOCHS, UPDATES_PER_PATIENT)
        folds.append({'fold': fold, 'train': sorted(train), 'validation': sorted(validation), 'test': sorted(test),
                      'training_order': order})
    initial = {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        model = model_for(arm)
        require(sum(p.numel() for p in model.parameters()) == 63809, 'Control parameter count differs')
        initial[arm] = initial_hash(model)
    require(len(set(initial.values())) == 1, 'Matched control initialization differs')
    require(source_identity == sources() and guard['open_attempts'] == 0, 'Sources changed or label opened')
    result = {'passed': True, 'scope': 'Exploratory warm cross-fit training input preflight; no training or labels.',
        'aggregate_path': str(Path(args.aggregate).resolve()), 'aggregate_sha256': aggregate_sha,
        'certificate_sha256': certificate_sha, 'warm_gate_sha256': gate_sha, 'index_plan_sha256': PLAN_SHA,
        'old_manifest_sha256': plan['existing_manifest_sha256'], 'backbone_sha256': plan['checkpoint_sha256'],
        'feature_pipeline_sha256': plan['feature_pipeline_sha256'],
        'environment': old['feature_pipeline']['runtime_environment'],
        'source_sha256': source_identity, 'seed': SEED, 'epochs': EPOCHS,
        'updates_per_patient_per_epoch': UPDATES_PER_PATIENT, 'arms': list(ARMS),
        'initial_state_sha256': initial,
        'optimizer': {'name': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001},
        'loss': 'corrected CMSA boundary weighted BCE and IoU plus soft Dice',
        'folds': folds, 'label_guard': dict(guard),
        'limitations': ['Previously inspected patients; not blind evaluation.',
                        'Fixed-read observations do not establish live-video efficacy.',
                        'No training or target-label access is performed by this preflight.']}
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'plan.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    (out / 'completion.json').write_text(json.dumps({'passed': True, 'plan_sha256': digest(path),
                                                     'label_guard': dict(guard)}, indent=2) + '\n')
    print(json.dumps({'passed': True, 'output': str(out), 'folds': len(folds), 'label_open_attempts': 0}))


if __name__ == '__main__':
    main()
