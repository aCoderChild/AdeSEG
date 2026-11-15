#!/usr/bin/env python3
"""Freeze, score and audit mixed-source held-out fixed-read predictions.

All 30 held-out prediction sets must be frozen and verified without labels
before target masks are opened. Uses the project's existing primary and
independent metric implementations. This separate path remains disabled until
review and real training/prediction artifacts exist.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling.seq23_viewport import VIEWPORT
from scripts import bounded_crossfit_shared_worker_v1 as worker
from scripts import verify_cache_release_crossfit_registry_v1 as verifier
from scripts.bounded_crossfit_supervision_v1 import assert_supervised, check_source_snapshot
from scripts.channel_value_fixed_read_v1 import install_phase_mask_guard
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, phase_artifact, validate_plan
from scripts.memory_token_io import digest
from scripts.predict_bounded_crossfit_fold_v1 import checked_fold
from scripts.preflight_bounded_crossfit_training_v1 import read_bound_json, require
from scripts.preflight_warm_crossfit_training_v1 import ARMS
from scripts.score_warm_crossfit_v1 import compare, compute
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_fold_training_v1 import fold_definition


ENABLE_REVIEWED_SCORING = True


def check_phase_order(phase_rows, tasks, completed=()):
    expected = [name for name, _, _ in tasks if name not in ('verify', 'score', 'audit')] + list(completed)
    require([row.get('phase') for row in phase_rows] == expected, 'All bounded training and prediction phases required')


def validate_all(root, completed=()):
    from scripts.run_bounded_crossfit_shared_study_v1 import phases
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(BASE.resolve(strict=True)), 'Batch outside Drive experiment root')
    manifest_path = root / 'run_manifest.json'; manifest, manifest_sha = read_bound_json(manifest_path)
    registry_folder = Path(manifest['registry_path']).resolve(strict=True)
    registry, registry_sha = read_bound_json(registry_folder / 'registry.json')
    require(manifest.get('format') == 'bounded_shared_crossfit_training_study_v1' and
            manifest.get('registry_sha256') == registry_sha and manifest.get('source_sha256') == worker.sources() and
            digest(registry_folder / 'completion.json') == manifest['registry_completion_sha256'] and
            type(manifest.get('cache_mib')) is int and 0 <= manifest['cache_mib'] <= 1024,
            'Bounded study manifest or registry differs')
    check_source_snapshot(root, manifest)
    require(type(manifest.get('max_process_rss_bytes')) is int and
            0 < manifest['max_process_rss_bytes'] <= 6 * 1024**3, 'Invalid recorded RSS cap')
    tasks = phases(root, registry_folder, manifest['cache_mib'], manifest['max_process_rss_bytes'])
    require(manifest.get('phases') == [name for name, _, _ in tasks], 'Bounded phase plan differs')
    phase_rows = [json.loads(line) for line in (root / 'phases.jsonl').read_text().splitlines() if line.strip()]
    check_phase_order(phase_rows, tasks, completed)
    phase_shas = {}
    for row, (_, _, artifact) in zip(phase_rows, tasks, strict=False):
        phase_shas[row['phase']] = phase_artifact(root, row['phase'], artifact)
    verification_path = root / 'verification/verification.json'
    verification, verification_sha = read_bound_json(verification_path)
    require(verification_sha == phase_shas['verification'] and
            verification.get('label_guard', {}).get('open_attempts') == 0 and
            {k: v for k, v in verification.items() if k != 'label_guard'} == verifier.verify(registry_folder),
            'Bounded registry verification differs')
    authorization_path = root / 'authorization/authorization.json'
    authorization_record, authorization_sha = read_bound_json(authorization_path)
    require(authorization_sha == phase_shas['authorization'] and
            authorization_record.get('label_guard', {}).get('open_attempts') == 0, 'Bounded authorization differs')
    authorization = {k: v for k, v in authorization_record.items() if k != 'label_guard'}
    current = worker.authorize(registry_folder, verification_path, root / 'probe/completion.json',
                               root / 'probe_audit/audit.json')
    require(authorization == current, 'Bounded authorization evidence changed')
    probe, _ = read_bound_json(root / 'probe/completion.json')
    require(probe.get('max_process_rss_bytes') == manifest['max_process_rss_bytes'], 'Probe RSS cap differs')
    samples = {(row['sequence'], row['frame']): row for row in registry['samples']}
    require(len(samples) == 357, 'Mixed capture coverage differs')
    index = validate_plan(); records = {}; owners = {}
    for fold in range(1, 6):
        _, _, test, _ = fold_definition(registry, fold, samples)
        for sequence in test:
            require(sequence not in owners, 'Patient appears in multiple test folds')
            owners[sequence] = fold
        expected_keys = {key for key in samples if key[0] in test}
        for arm in ARMS:
            _, rows, actual_test, group_sha, train_sha, checkpoint_sha = checked_fold(root, fold, arm,
                                                                                   authorization, registry)
            require(actual_test == test and set(rows) == set(samples), 'Bounded fold ownership differs')
            prediction = root / 'predict' / f'fold{fold}' / arm
            pred_path = prediction / 'completion.json'; pred_sha = phase_shas[f'predict_f{fold}_{arm}']
            pred, read_sha = read_bound_json(pred_path)
            require(pred_sha == read_sha and pred.get('passed') is True and
                    pred.get('format') == 'bounded_crossfit_prediction_v1' and pred.get('fold') == fold and
                    pred.get('arm') == arm and pred.get('authorization') == authorization and
                    pred.get('group_sha256') == group_sha and pred.get('training_completion_sha256') == train_sha and
                    pred.get('checkpoint_sha256') == checkpoint_sha and pred.get('device') == 'mps' and
                    pred.get('environment') == registry['environment'] and
                    pred.get('label_guard', {}).get('open_attempts') == 0, 'Frozen bounded prediction differs')
            mapping = {(row['sequence'], row['frame']): row for row in pred['rows']}
            require(len(mapping) == len(pred['rows']) and set(mapping) == expected_keys, 'Held-out read coverage differs')
            for key, row in mapping.items():
                require(row['source_frame_index'] == index['source_frames'][key[0]][key[1]]['source_frame_index'],
                        'Prediction source frame differs')
                if key[0] == 'seq23':
                    require(row['video_shape'] == [720, 834] and row['source_shape'] == [720, 1280],
                            'Seq23 view differs')
                else:
                    require(row['video_shape'] == row['source_shape'], 'Full-view geometry differs')
                for name in ('baseline', 'refined'):
                    path = Path(row[name + '_path']).resolve(strict=True)
                    require(path.is_relative_to(prediction.resolve()) and digest(path) == row[name + '_sha256'],
                            'Frozen prediction file differs')
                    with Image.open(path) as image:
                        array = np.asarray(image.convert('L'))
                        require(image.format == 'PNG' and list(array.shape) == row['source_shape'] and
                                np.isin(array, (0, 255)).all(), 'Prediction encoding differs')
                        if key[0] == 'seq23':
                            require(not array[:, :VIEWPORT[0]].any(), 'Seq23 outside-view prediction differs')
                    require(digest(path) == row[name + '_sha256'], 'Prediction changed during decode')
            records[(fold, arm)] = {'training_sha256': train_sha, 'checkpoint_sha256': checkpoint_sha,
                                    'prediction_sha256': pred_sha, 'mapping': mapping}
        first = records[(fold, ARMS[0])]['mapping']
        for arm in ARMS[1:]:
            require(all(first[key]['baseline_sha256'] == records[(fold, arm)]['mapping'][key]['baseline_sha256']
                        for key in expected_keys), 'Baseline differs between arms')
    require(set(owners) == {f'seq{i}' for i in range(1, 24)} and digest(manifest_path) == manifest_sha and
            digest(registry_folder / 'registry.json') == registry_sha and worker.sources() == manifest['source_sha256'],
            'Study ownership or source identity changed')
    proof = {'manifest_sha256': manifest_sha, 'registry_sha256': registry_sha,
             'authorization_sha256': authorization_sha,
             'records': {f'{fold}:{arm}': {k: v for k, v in value.items() if k != 'mapping'}
                         for (fold, arm), value in records.items()}, 'patient_test_folds': owners}
    return index, samples, records, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('verify', 'score', 'audit'), required=True)
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.batch_root.resolve(strict=True); output = args.output.resolve()
    require(output.is_relative_to(root) and not output.exists(), 'Invalid output')
    require(ENABLE_REVIEWED_SCORING, 'Bounded scoring awaits independent review and complete frozen predictions')
    assert_supervised(root, args.mode, output)
    guard = install_mask_open_guard(DATA) if args.mode == 'verify' else install_phase_mask_guard(DATA)
    completed = () if args.mode == 'verify' else ('verify',) if args.mode == 'score' else ('verify', 'score')
    index, samples, records, proof = validate_all(root, completed=completed)
    require(guard['open_attempts'] == 0, 'Target label opened before verification')
    if args.mode == 'verify':
        proof.update(passed=True, label_guard=dict(guard), source_sha256=digest(Path(__file__)))
        output.mkdir(parents=True); (output / 'verification.json').write_text(json.dumps(proof, indent=2) + '\n')
        return
    verification_path = root / 'verify/verification.json'; verification_sha = phase_artifact(root, 'verify', verification_path)
    saved_proof, _ = read_bound_json(verification_path)
    require(saved_proof.get('passed') is True and saved_proof.get('label_guard', {}).get('open_attempts') == 0 and
            saved_proof.get('source_sha256') == digest(Path(__file__)) and
            all(saved_proof.get(k) == value for k, value in proof.items()), 'All-fold verification differs')
    if args.mode == 'audit':
        score_path = root / 'score/results.json'; score_sha = phase_artifact(root, 'score', score_path)
        saved_score, _ = read_bound_json(score_path)
        require(saved_score.get('passed') is True and saved_score.get('verification_sha256') == verification_sha and
                saved_score.get('report_sha256') == digest(root / 'score/report.md'), 'Frozen score differs')
    require(guard['open_attempts'] == 0, 'Target label opened before score release')
    prelabel_guard = dict(guard)
    guard['active'] = False
    result = compute(index, samples, records, independent=args.mode == 'audit')
    manifest, manifest_sha = read_bound_json(root / 'run_manifest.json')
    require(manifest_sha == proof['manifest_sha256'] and manifest.get('source_sha256') == worker.sources() and
            digest(Path(manifest['registry_path']) / 'registry.json') == proof['registry_sha256'] and
            digest(verification_path) == verification_sha and
            (args.mode != 'audit' or digest(score_path) == score_sha),
            'Frozen sources or scoring evidence changed during metric computation')
    output.mkdir(parents=True)
    if args.mode == 'score':
        lines = ['# Exploratory bounded cross-fit fixed-read results', '',
            'Previously inspected patients; held-out sampled reads, not live-video propagation.', '',
            '| Arm | Patient-macro Dice | Patients | Frame-weighted Dice | Present reads | Empty false alarms |',
            '|---|---:|---:|---:|---:|---:|']
        for name, summary in result['summaries'].items():
            pooled = summary['pooled']; macro = summary['patient_macro_present_dice']; dice = pooled['mean_present_dice']
            lines.append(f"| {name} | {'N/A' if macro is None else f'{macro:.4f}'} | "
                         f"{summary['patients_with_present_reads']} | {'N/A' if dice is None else f'{dice:.4f}'} | "
                         f"{pooled['present_frames']} | {pooled['false_alarms']}/{pooled['empty_frames']} |")
        lines += ['', 'Seq23 viewport-only results, patient-level results, paired changes and no-initialization outcomes are in results.json.', '']
        (output / 'report.md').write_text('\n'.join(lines))
        result.update(passed=True, verification_sha256=verification_sha, report_sha256=digest(output / 'report.md'),
                      scorer_sha256=digest(Path(__file__)), label_guard=prelabel_guard)
        (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        errors = []; compare(result, {k: saved_score[k] for k in result}, [], errors)
        audit = {'passed': not errors, 'error_count': len(errors), 'errors': errors,
                 'verification_sha256': verification_sha, 'score_sha256': score_sha,
                 'auditor_sha256': digest(Path(__file__)), 'label_guard': prelabel_guard,
                 'scope': 'Independent CPU metric recomputation.'}
        (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        require(not errors, 'Independent bounded scoring audit found discrepancies')


if __name__ == '__main__':
    main()
