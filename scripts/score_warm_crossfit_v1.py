#!/usr/bin/env python3
"""Verify, score or independently audit frozen five-fold fixed-read outputs.

Held-out masks are not opened until every fold/arm checkpoint and prediction
has been frozen and a separate no-label verification exists.
This study reuses inspected patients and does not measure live propagation.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.channel_value_fixed_read_v1 import install_phase_mask_guard
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, phase_artifact
from scripts.crossfit_training_contract_v1 import fold_split
from scripts.memory_token_io import digest
from scripts.preflight_warm_crossfit_training_v1 import ARMS, sources
from scripts.project_archived_label_hashes_v1 import project_label_hashes
from scripts.score_boundary_video_factorial_v1 import metric as primary_metric
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_fold_training_v1 import fold_definition, load_preflight, read, require, samples_for, trained_model


def check_output(path, root):
    out = Path(path).resolve()
    require(out.is_relative_to(root) and not out.exists(), 'Invalid or existing output')
    return out


def read_bound(path, expected_sha):
    """Read a phase-bound receipt without accepting an intervening rewrite."""
    require(digest(path) == expected_sha, 'Phase-bound receipt changed before parsing')
    value = read(path)
    require(digest(path) == expected_sha, 'Phase-bound receipt changed during parsing')
    return value


def validate_all(root, preflight_path, preflight, preflight_sha, *, completed=()):
    """Verify all fold artifacts and test ownership before any target read."""
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(BASE.resolve(strict=True)), 'Batch outside Drive')
    manifest_path = root / 'run_manifest.json'; manifest_sha = digest(manifest_path); launch = read(manifest_path)
    if launch.get('trainer') == 'shared_six_arm_v1':
        from scripts.run_warm_crossfit_shared_v1 import phases
        from scripts.register_warm_crossfit_arm_v1 import verify_arm
    else:
        require(launch.get('trainer') is None, 'Unsupported trainer')
        from scripts.run_warm_crossfit_study_v1 import phases
        verify_arm = None
    cache_mib = launch.get('cache_mib')
    require(type(cache_mib) is int and 0 <= cache_mib <= 1024, 'Invalid frozen cache size')
    tasks = phases(root, preflight_path, cache_mib)
    require(digest(manifest_path) == manifest_sha and launch.get('preflight_path') == str(preflight_path) and
            launch.get('preflight_sha256') == preflight_sha and launch.get('source_sha256') == sources() and
            launch.get('phases') == [t[0] for t in tasks], 'Study manifest differs')
    rows = [json.loads(x) for x in (root / 'phases.jsonl').read_text().splitlines() if x.strip()]
    expected = [name for name, _, _ in tasks if name not in ('verify', 'score', 'audit')] + list(completed)
    require([r.get('phase') for r in rows] == expected, 'Probe and all train/prediction phases are required')
    phase_shas = {}
    for row, (_, _, artifact) in zip(rows, tasks, strict=False):
        expected_sha = phase_artifact(root, row['phase'], artifact)
        require(expected_sha == row.get('artifact_sha256'),
                'Study phase artifact changed')
        phase_shas[row['phase']] = expected_sha
    index, samples = samples_for(preflight)
    probe = read_bound(root / 'probe/completion.json', phase_shas['probe'])
    checks = probe.get('checks', [])
    require(probe.get('passed') is True and probe.get('preflight_sha256') == preflight_sha and
            probe.get('device') == 'mps' and probe.get('environment') == preflight['environment'] and
            probe.get('label_guard', {}).get('open_attempts') == 0 and len(checks) == 2 * len(ARMS) and
            {(x['arm'], x['sequence']) for x in checks} == {(arm, seq) for arm in ARMS for seq in ('seq6', 'seq23')} and
            all((x['sequence'], x['frame']) in samples and
                (x['sequence'] != 'seq23' or x['video_shape'] == [720, 834]) and
                isinstance(x.get('head_bias_gradient_abs'), (int, float)) and
                math.isfinite(x['head_bias_gradient_abs']) and x['head_bias_gradient_abs'] > 0 for x in checks),
            'Targeted MPS probe differs')
    if verify_arm is not None:
        require(probe.get('kind') == 'shared_six_arm_probe_v1', 'Shared-path probe was not run')
        for check in checks:
            checkpoint = Path(check.get('checkpoint_path', '')).resolve(strict=True)
            require(checkpoint.is_relative_to((root / 'probe').resolve()) and
                    digest(checkpoint) == check.get('checkpoint_sha256') and
                    check.get('initial_state_sha256') == preflight['initial_state_sha256'][check['arm']] and
                    check.get('optimizer_updated') is True and
                    isinstance(check.get('gradient_l2'), (int, float)) and
                    math.isfinite(check['gradient_l2']) and check['gradient_l2'] > 0 and
                    isinstance(check.get('synthetic_loss'), (int, float)) and
                    math.isfinite(check['synthetic_loss']), 'Shared probe checkpoint or update differs')
    records = {}; owners = {}
    for fold in range(1, 6):
        _, _, test, order = fold_definition(preflight, fold, samples)
        for sequence in test:
            require(sequence not in owners, 'Patient appears in more than one held-out fold')
            owners[sequence] = fold
        expected_keys = {key for key in samples if key[0] in test}
        for arm in ARMS:
            if verify_arm is not None:
                require(verify_arm(root, preflight_sha, fold, arm) == phase_shas[f'train_f{fold}_{arm}'],
                        'Shared arm phase binding differs')
            train = root / 'train' / f'fold{fold}' / arm
            prediction = root / 'predict' / f'fold{fold}' / arm
            checkpoint = train / 'checkpoint.pt'; train_path = train / 'completion.json'
            train_sha = phase_shas[f'train_f{fold}_{arm}']
            train_rec = read_bound(train_path, train_sha); checkpoint_sha = digest(checkpoint)
            require(digest(train_path) == train_sha and train_rec.get('passed') is True and
                train_rec.get('fold') == fold and train_rec.get('arm') == arm and
                train_rec.get('preflight_sha256') == preflight_sha and
                train_rec.get('device') == 'mps' and train_rec.get('environment') == preflight['environment'] and
                train_rec.get('checkpoint_sha256') == checkpoint_sha and
                train_rec.get('training_steps') == sum(map(len, order)) and
                train_rec.get('initial_state_sha256') == preflight['initial_state_sha256'][arm] and
                train_rec.get('cache', {}).get('limit_bytes') == cache_mib * 1024**2 and
                type(train_rec['cache'].get('peak_bytes')) is int and
                0 <= train_rec['cache']['peak_bytes'] <= cache_mib * 1024**2 and
                type(train_rec['cache'].get('hits')) is int and type(train_rec['cache'].get('misses')) is int and
                train_rec['cache']['hits'] >= 0 and train_rec['cache']['misses'] >= 0 and
                train_rec['cache']['hits'] + train_rec['cache']['misses'] == train_rec['training_steps'] and
                type(train_rec['cache'].get('unique_reads')) is int and
                0 < train_rec['cache']['unique_reads'] <= train_rec['cache']['misses'] and
                train_rec['cache'].get('final_sha256_verified') is True and
                train_rec.get('label_guard', {}).get('blocked_open_attempts') == 0,
                'Fold training receipt differs')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(digest(checkpoint) == checkpoint_sha and payload.get('training_steps') == train_rec['training_steps'],
                    'Fold checkpoint changed')
            trained_model(payload, arm=arm, fold=fold, preflight=preflight, preflight_sha=preflight_sha)
            pred_path = prediction / 'completion.json'; pred_sha = phase_shas[f'predict_f{fold}_{arm}']
            pred = read_bound(pred_path, pred_sha)
            require(pred.get('passed') is True and pred.get('fold') == fold and pred.get('arm') == arm and
                    pred.get('device') == 'mps' and pred.get('environment') == preflight['environment'] and
                    pred.get('preflight_sha256') == preflight_sha and pred.get('checkpoint_sha256') == checkpoint_sha and
                    pred.get('training_completion_sha256') == train_sha and
                    pred.get('label_guard', {}).get('open_attempts') == 0, 'Frozen prediction differs')
            mapping = {(r['sequence'], r['frame']): r for r in pred['rows']}
            require(len(mapping) == len(pred['rows']) and set(mapping) == expected_keys,
                    'Fold held-out read coverage differs')
            for key, row in mapping.items():
                require(row['source_frame_index'] == index['source_frames'][key[0]][key[1]]['source_frame_index'],
                        'Prediction source frame differs')
                if key[0] == 'seq23':
                    require(row['video_shape'] == [720, 834] and row['source_shape'] == [720, 1280],
                            'Seq23 prediction view differs')
                else:
                    require(row['video_shape'] == row['source_shape'], 'Full-view prediction geometry differs')
                for name in ('baseline', 'refined'):
                    path = Path(row[name + '_path']).resolve(strict=True)
                    require(path.is_relative_to(prediction.resolve()) and digest(path) == row[name + '_sha256'],
                            'Fold prediction file changed')
                    with Image.open(path) as image:
                        array = np.asarray(image.convert('L'))
                        require(image.format == 'PNG' and list(array.shape) == row['source_shape'] and
                                np.isin(array, (0, 255)).all(), 'Fold prediction encoding differs')
                        if key[0] == 'seq23':
                            require(not array[:, :VIEWPORT[0]].any(), 'Seq23 outside-view prediction differs')
                    require(digest(path) == row[name + '_sha256'], 'Fold prediction changed during decoding')
            records[(fold, arm)] = {'training_sha256': train_sha, 'checkpoint_sha256': checkpoint_sha,
                'prediction_sha256': pred_sha, 'mapping': mapping}
        first = records[(fold, ARMS[0])]['mapping']
        for arm in ARMS[1:]:
            require(all(first[k]['baseline_sha256'] == records[(fold, arm)]['mapping'][k]['baseline_sha256']
                        for k in expected_keys), 'Baseline differs between arms')
    require(set(owners) == {f'seq{i}' for i in range(1, 24)} and digest(manifest_path) == manifest_sha,
            'Patient ownership or study manifest differs')
    return index, samples, records, {'manifest_sha256': manifest_sha, 'preflight_sha256': preflight_sha,
        'records': {f'{f}:{a}': {k: v for k, v in rec.items() if k != 'mapping'}
                    for (f, a), rec in records.items()}, 'patient_test_folds': owners}


def label_hashes(index, samples):
    """Project historical hashes only after the all-fold verifier is accepted."""
    groups = defaultdict(list); output = {}
    for key, item in samples.items():
        if item['origin'] == 'historical':
            output[key] = item['label_sha256']
        else:
            groups[key[0]].append(item)
    for seq, items in groups.items():
        binding = {'path': items[0]['label_metadata_path'], 'sha256': items[0]['label_metadata_sha256']}
        require(all(x['label_metadata_path'] == binding['path'] and x['label_metadata_sha256'] == binding['sha256']
                    and x['label_metadata_locator'] == x['frame'] for x in items), 'Label metadata binding differs')
        selected = project_label_hashes(binding, len(index['source_frames'][seq]), [x['frame'] for x in items])
        output.update({(seq, frame): sha for frame, sha in selected.items()})
    require(set(output) == set(samples), 'Label hash coverage differs')
    return output


def summarize(rows, name):
    present = [r[name] for r in rows if r[name]['present']]
    empty = [r[name] for r in rows if not r[name]['present']]
    def average(values):
        values = list(values)
        return math.fsum(values) / len(values) if values else None
    return {'frames': len(rows), 'present_frames': len(present),
            'mean_present_dice': average(x['dice'] for x in present),
            'mean_present_iou': average(x['iou'] for x in present),
            'mean_present_boundary_f_5px': average(x['boundary_f_5px'] for x in present),
            'empty_frames': len(empty), 'false_alarms': sum(x['false_alarm'] for x in empty)}


def compute(index, samples, records, *, independent=False):
    if independent:
        from scripts.audit_boundary_video_factorial_v1 import label, metrics
    else:
        label = lambda path: load_label_mask(path, [1]) == 1
        metrics = primary_metric
    hashes = label_hashes(index, samples)
    by_sequence = defaultdict(list)
    for key in sorted(samples):
        item = samples[key]; fold = next(i for i in range(1, 6) if key[0] in fold_split(i)[2])
        path = Path(item['label']); require(digest(path) == hashes[key], 'Label changed before decoding')
        truth = label(path); require(digest(path) == hashes[key], 'Label changed during decoding')
        row = {'sequence': key[0], 'frame': key[1], 'fold': fold}
        if key[0] == 'seq23':
            require(truth.shape == (720, 1280), 'Seq23 source-label geometry differs')
            row['viewport'] = {}
        for name in ('baseline', *ARMS):
            arm = ARMS[0] if name == 'baseline' else name
            kind = 'baseline' if name == 'baseline' else 'refined'
            saved = records[(fold, arm)]['mapping'][key]
            file = Path(saved[kind + '_path'])
            require(digest(file) == saved[kind + '_sha256'], 'Prediction changed before scoring')
            with Image.open(file) as image:
                mask = np.asarray(image.convert('L')) > 0
            require(digest(file) == saved[kind + '_sha256'] and mask.shape == truth.shape,
                    'Prediction changed or target geometry differs')
            row[name] = metrics(mask, truth)
            if key[0] == 'seq23':
                x0, y0, x1, y1 = VIEWPORT
                row['viewport'][name] = metrics(mask[y0:y1, x0:x1], truth[y0:y1, x0:x1])
        by_sequence[key[0]].append(row)
    names = ('baseline', *ARMS); flat = [row for sequence in sorted(by_sequence) for row in by_sequence[sequence]]
    summaries = {}
    for name in names:
        per = {seq: summarize(rows, name) for seq, rows in sorted(by_sequence.items())}
        means = [x['mean_present_dice'] for x in per.values() if x['mean_present_dice'] is not None]
        summaries[name] = {'pooled': summarize(flat, name), 'per_sequence': per,
                           'patient_macro_present_dice': math.fsum(means) / len(means) if means else None,
                           'patients_with_present_reads': len(means)}
    def paired(items, arm):
        delta = [r[arm]['dice'] - r['baseline']['dice'] for r in items if r['baseline']['present']]
        return {'present_reads': len(delta), 'mean_present_dice_difference':
                math.fsum(delta) / len(delta) if delta else None,
                'improved': sum(x > 0 for x in delta), 'worsened': sum(x < 0 for x in delta),
                'unchanged': sum(x == 0 for x in delta)}
    comparisons = {arm: {'pooled': paired(flat, arm),
                         'per_sequence': {seq: paired(items, arm) for seq, items in sorted(by_sequence.items())}}
                   for arm in ARMS}
    viewport = {name: summarize([r['viewport'] for r in by_sequence.get('seq23', [])], name) for name in names}
    return {'rows': dict(by_sequence), 'summaries': summaries, 'paired_vs_baseline': comparisons,
            'seq23_viewport_summaries': viewport,
            'no_initialization': ['seq1', 'seq7'], 'no_initialization_dice': None}


def compare(actual, expected, path, errors):
    from scripts.audit_boundary_video_factorial_v1 import same
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected): errors.append([*path, 'keys'])
        for key in actual.keys() & expected.keys(): compare(actual[key], expected[key], [*path, key], errors)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected): errors.append([*path, 'length'])
        for i, (a, b) in enumerate(zip(actual, expected, strict=False)): compare(a, b, [*path, i], errors)
    elif isinstance(actual, str) or isinstance(expected, str):
        if actual != expected: errors.append([*path, actual, expected])
    elif not same(actual, expected): errors.append([*path, actual, expected])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('verify', 'score', 'audit'), required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.batch_root.resolve(strict=True); out = check_output(args.output, root)
    guard = install_mask_open_guard(DATA) if args.mode == 'verify' else install_phase_mask_guard(DATA)
    preflight_path = args.preflight.resolve(strict=True); preflight, preflight_sha = load_preflight(preflight_path)
    completed = () if args.mode == 'verify' else ('verify',) if args.mode == 'score' else ('verify', 'score')
    index, samples, records, proof = validate_all(root, preflight_path, preflight, preflight_sha, completed=completed)
    require(guard['open_attempts'] == 0, 'Target label opened before verification')
    if args.mode == 'verify':
        proof.update(passed=True, label_guard=dict(guard), source_sha256=digest(Path(__file__)))
        out.mkdir(parents=True); (out / 'verification.json').write_text(json.dumps(proof, indent=2) + '\n')
        return
    verification = root / 'verify/verification.json'; expected = phase_artifact(root, 'verify', verification)
    saved_proof = read(verification)
    require(digest(verification) == expected and saved_proof.get('passed') is True and
            saved_proof.get('label_guard', {}).get('open_attempts') == 0 and
            saved_proof.get('source_sha256') == digest(Path(__file__)) and
            all(saved_proof.get(k) == v for k, v in proof.items()), 'All-fold verification differs')
    score_sha = None
    if args.mode == 'audit':
        score_path = root / 'score/results.json'; score_sha = phase_artifact(root, 'score', score_path)
        saved = read(score_path)
        require(digest(score_path) == score_sha and saved.get('passed') is True and
                saved.get('verification_sha256') == expected and
                saved.get('report_sha256') == digest(root / 'score/report.md'), 'Frozen score differs')
    require(guard['open_attempts'] == 0, 'Target label opened before score release')
    guard['active'] = False
    result = compute(index, samples, records, independent=args.mode == 'audit')
    out.mkdir(parents=True)
    if args.mode == 'score':
        lines = ['# Exploratory warm cross-fit fixed-read results', '',
            'Previously inspected patients; sampled post-prompt reads, not live-video propagation.',
            'Seq1 and seq7 have no automatic initialization and are reported separately as N/A.', '',
            '| Arm | Patient-macro Dice | Patients | Frame-weighted Dice | Present reads | Empty false alarms |',
            '|---|---:|---:|---:|---:|---:|']
        for name, summary in result['summaries'].items():
            pooled = summary['pooled']; macro = summary['patient_macro_present_dice']; dice = pooled['mean_present_dice']
            lines.append(f"| {name} | {'N/A' if macro is None else f'{macro:.4f}'} | "
                         f"{summary['patients_with_present_reads']} | {'N/A' if dice is None else f'{dice:.4f}'} | "
                         f"{pooled['present_frames']} | {pooled['false_alarms']}/{pooled['empty_frames']} |")
        lines += ['', 'Seq23 viewport-restricted metrics are reported separately from the full-frame table.', '',
                  '| Arm | Viewport Dice | Present reads | Empty false alarms |', '|---|---:|---:|---:|']
        for name, summary in result['seq23_viewport_summaries'].items():
            dice = summary['mean_present_dice']
            lines.append(f"| {name} | {'N/A' if dice is None else f'{dice:.4f}'} | {summary['present_frames']} | "
                         f"{summary['false_alarms']}/{summary['empty_frames']} |")
        names = ('baseline', *ARMS)
        lines += ['', 'Per-sequence present-read Dice (full-frame evaluation):', '',
                  '| Sequence | Present reads | ' + ' | '.join(names) + ' |',
                  '|---|---:|' + '---:|' * len(names)]
        for number in range(1, 24):
            sequence = f'seq{number}'
            base = result['summaries']['baseline']['per_sequence'].get(sequence)
            count = 'N/A' if base is None else str(base['present_frames'])
            values = []
            for name in names:
                item = result['summaries'][name]['per_sequence'].get(sequence)
                dice = None if item is None else item['mean_present_dice']
                values.append('N/A' if dice is None else f'{dice:.4f}')
            lines.append(f"| {sequence} | {count} | " + ' | '.join(values) + ' |')
        report = out / 'report.md'; report.write_text('\n'.join(lines) + '\n')
        result.update(passed=True, verification_sha256=expected, prelabel_guard=dict(guard),
                      report_sha256=digest(report),
                      scope='Exploratory patient-cross-fit fixed reads; not live video or blinded data.')
        (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        errors = []
        for key in ('rows', 'summaries', 'paired_vs_baseline', 'seq23_viewport_summaries',
                    'no_initialization', 'no_initialization_dice'):
            compare(result[key], saved[key], [key], errors)
        (out / 'audit.json').write_text(json.dumps({'passed': not errors, 'score_sha256': score_sha,
            'prelabel_guard': dict(guard), 'scope': 'Independent CPU metric calculation; shared structural verifier.',
            'error_count': len(errors), 'errors': errors}, indent=2) + '\n')
        require(not errors, 'Independent metric audit differs')


if __name__ == '__main__':
    main()
