#!/usr/bin/env python3
"""Freeze, score and independently audit all four control arms on five folds."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.channel_value_fixed_read_v1 import install_phase_mask_guard
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, phase_artifact
from scripts.crossfit_training_contract_v1 import fold_split
from scripts.mechanism_crossfit_control_core_v1 import ARMS, trained_model
from scripts.memory_token_io import digest
from scripts.preflight_mechanism_crossfit_v1 import reference_study, sources
from scripts.run_mechanism_crossfit_fold_v1 import read_bound, study_inputs
from scripts.score_boundary_video_factorial_v1 import metric as primary_metric
from scripts.score_warm_crossfit_v1 import compare, label_hashes, summarize
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_fold_training_v1 import fold_definition, require


def validate_all(root, plan_path, plan, plan_sha, warm, warm_sha, index, samples, completed=()):
    from scripts.run_mechanism_crossfit_study_v1 import phases
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(BASE.resolve(strict=True)), 'Study outside Drive')
    path = root/'run_manifest.json'; manifest_sha = digest(path)
    manifest = read_bound(path, manifest_sha); cache = manifest['cache_mib']
    require(type(cache) is int and 0 <= cache <= 1024, 'Invalid frozen cache')
    tasks = phases(root, plan_path, cache)
    require(manifest.get('plan_path') == str(plan_path) and manifest.get('plan_sha256') == plan_sha and
            manifest.get('source_sha256') == sources() == plan['source_sha256'] and
            manifest.get('phases') == [x[0] for x in tasks], 'Study manifest differs')
    reference, reference_proof = reference_study(plan['reference_study']['root'],
        Path(plan['warm_preflight_path']), warm, warm_sha)
    require(reference_proof == plan['reference_study'], 'Matched asinh reference changed')
    log = root/'phases.jsonl'; rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    expected = [name for name, _, _ in tasks if name not in ('verify', 'score', 'audit')] + list(completed)
    require([r['phase'] for r in rows] == expected, 'All control fits and predictions must precede scoring')
    phase_shas = {row['phase']: phase_artifact(root, row['phase'], artifact)
                  for row, (_, _, artifact) in zip(rows, tasks, strict=False)}
    probe = read_bound(root/'probe/completion.json', phase_shas['probe'])
    checks = probe.get('checks', [])
    require(probe.get('passed') is True and probe.get('plan_sha256') == plan_sha and
            probe.get('warm_preflight_sha256') == warm_sha and probe.get('device') == 'mps' and
            probe.get('environment') == warm['environment'] and probe.get('label_guard', {}).get('open_attempts') == 0 and
            len(checks) == 2*len(ARMS) and {(x['arm'], x['sequence']) for x in checks} ==
            {(arm, seq) for arm in ARMS for seq in ('seq6', 'seq23')}, 'Control probe differs')
    for check in checks:
        file = Path(check['checkpoint_path']).resolve(strict=True)
        require(file.is_relative_to((root/'probe').resolve()) and digest(file) == check['checkpoint_sha256'] and
                (check['sequence'], check['frame']) in samples and
                (check['sequence'] != 'seq23' or check['video_shape'] == [720, 834]) and
                math.isfinite(check['head_bias_gradient_abs']) and check['head_bias_gradient_abs'] > 0 and
                math.isfinite(check['gradient_l2']) and check['gradient_l2'] > 0, 'Control probe evidence differs')
    owners, mapping, summary = {}, {}, {}
    for fold in range(1, 6):
        _, _, test, order = fold_definition(warm, fold, samples)
        for seq in test:
            require(seq not in owners, 'Patient has multiple test folds'); owners[seq] = fold
        train = root/'train'/f'fold{fold}'; group_sha = phase_shas[f'train_f{fold}']
        group = read_bound(train/'group.json', group_sha)
        require(group.get('passed') is True and group.get('fold') == fold and group.get('plan_sha256') == plan_sha and
                group.get('warm_preflight_sha256') == warm_sha and group.get('training_steps') == sum(map(len, order)) and
                group.get('initial_state_sha256') == plan['initial_state_sha256'] and group.get('device') == 'mps' and
                group.get('environment') == warm['environment'] and group.get('label_guard', {}).get('blocked_open_attempts') == 0 and
                group.get('cache', {}).get('limit_bytes') == cache*1024**2 and
                group.get('cache', {}).get('final_sha256_verified') is True and set(group['arms']) == set(ARMS),
                'Control training differs')
        for arm in ARMS:
            item = group['arms'][arm]; file = Path(item['checkpoint_path']).resolve(strict=True)
            require(file == train/arm/'checkpoint.pt' and digest(file) == item['checkpoint_sha256'], 'Checkpoint differs')
            payload = torch.load(file, map_location='cpu', weights_only=True)
            require(digest(file) == item['checkpoint_sha256'] and payload.get('training_steps') == group['training_steps'],
                    'Checkpoint changed while reading')
            trained_model(payload, arm=arm, fold=fold, warm_plan=warm, warm_plan_sha=warm_sha,
                          study_plan_sha=plan_sha, sources=plan['source_sha256'])
        prediction = root/'predict'/f'fold{fold}'; pred_sha = phase_shas[f'predict_f{fold}']
        rec = read_bound(prediction/'completion.json', pred_sha)
        require(rec.get('passed') is True and rec.get('fold') == fold and rec.get('plan_sha256') == plan_sha and
                rec.get('warm_preflight_sha256') == warm_sha and rec.get('training_group_sha256') == group_sha and
                rec.get('device') == 'mps' and rec.get('environment') == warm['environment'] and
                rec.get('label_guard', {}).get('open_attempts') == 0, 'Control prediction differs')
        got = {(x['sequence'], x['frame']): x for x in rec['rows']}
        require(len(got) == len(rec['rows']) and set(got) == {key for key in samples if key[0] in test},
                'Control held-out read coverage differs')
        for key, row in got.items():
            require(row['source_frame_index'] == index['source_frames'][key[0]][key[1]]['source_frame_index'] and
                    set(row['masks']) == {'baseline', *ARMS}, 'Control source frame or mask set differs')
            require((row['video_shape'] == [720, 834] and row['source_shape'] == [720, 1280]) if key[0] == 'seq23'
                    else row['video_shape'] == row['source_shape'], 'Control viewport differs')
            for item in row['masks'].values():
                file = Path(item['path']).resolve(strict=True)
                require(file.is_relative_to(prediction.resolve()) and digest(file) == item['sha256'], 'Control mask differs')
                with Image.open(file) as image:
                    mask = np.asarray(image.convert('L'))
                    require(image.format == 'PNG' and list(mask.shape) == row['source_shape'] and
                            np.isin(mask, (0, 255)).all(), 'Control mask encoding differs')
                    if key[0] == 'seq23': require(not mask[:, :VIEWPORT[0]].any(), 'Mask escaped seq23 viewport')
                require(digest(file) == item['sha256'], 'Control mask changed while reading')
            matched = reference[(fold, 'foreground_background_asinh')]['mapping'][key]
            require(row['masks']['baseline']['sha256'] == matched['baseline_sha256'],
                    'Control and reference baseline predictions differ')
            row['masks']['asinh_reference'] = {'path': matched['refined_path'], 'sha256': matched['refined_sha256']}
        mapping.update(got)
        summary[str(fold)] = {'training_sha256': group_sha, 'prediction_sha256': pred_sha,
                              'checkpoint_sha256': {arm: group['arms'][arm]['checkpoint_sha256'] for arm in ARMS}}
    require(set(owners) == {f'seq{i}' for i in range(1, 24)} and set(mapping) == set(samples) and
            digest(path) == manifest_sha, 'Control patient or read coverage differs')
    return mapping, {'manifest_sha256': manifest_sha, 'plan_sha256': plan_sha,
                     'folds': summary, 'patient_test_folds': owners}


def compute(index, samples, mapping, *, independent=False):
    if independent:
        from scripts.audit_boundary_video_factorial_v1 import label, metrics
    else:
        label = lambda p: load_label_mask(p, [1]) == 1
        metrics = primary_metric
    hashes = label_hashes(index, samples); by_sequence = defaultdict(list)
    names = ('baseline', 'asinh_reference', *ARMS)
    for key in sorted(samples):
        path = Path(samples[key]['label']); require(digest(path) == hashes[key], 'Label changed')
        truth = label(path); require(digest(path) == hashes[key], 'Label changed during decoding')
        row = {'sequence': key[0], 'frame': key[1],
               'fold': next(f for f in range(1, 6) if key[0] in fold_split(f)[2])}
        if key[0] == 'seq23': row['viewport'] = {}
        for arm in names:
            item = mapping[key]['masks'][arm]; file = Path(item['path'])
            require(digest(file) == item['sha256'], 'Prediction changed')
            with Image.open(file) as image: mask = np.asarray(image.convert('L')) > 0
            require(digest(file) == item['sha256'] and mask.shape == truth.shape, 'Prediction or target shape differs')
            row[arm] = metrics(mask, truth)
            if key[0] == 'seq23':
                x0, y0, x1, y1 = VIEWPORT
                row['viewport'][arm] = metrics(mask[y0:y1, x0:x1], truth[y0:y1, x0:x1])
        by_sequence[key[0]].append(row)
    flat = [r for seq in sorted(by_sequence) for r in by_sequence[seq]]; summaries = {}
    for arm in names:
        per = {seq: summarize(rows, arm) for seq, rows in sorted(by_sequence.items())}
        means = [x['mean_present_dice'] for x in per.values() if x['mean_present_dice'] is not None]
        summaries[arm] = {'pooled': summarize(flat, arm), 'per_sequence': per,
            'patient_macro_present_dice': math.fsum(means)/len(means) if means else None,
            'patients_with_present_reads': len(means)}
    def paired(items, arm, reference):
        delta = [r[arm]['dice']-r[reference]['dice'] for r in items if r[reference]['present']]
        return {'present_reads': len(delta), 'mean_present_dice_difference': math.fsum(delta)/len(delta) if delta else None,
                'improved': sum(x > 0 for x in delta), 'worsened': sum(x < 0 for x in delta),
                'unchanged': sum(x == 0 for x in delta)}
    return {'rows': dict(by_sequence), 'summaries': summaries,
        'paired_vs_baseline': {a: {'pooled': paired(flat, a, 'baseline'), 'per_sequence':
            {s: paired(rows, a, 'baseline') for s, rows in sorted(by_sequence.items())}} for a in ARMS},
        'paired_vs_asinh_reference': {a: {'pooled': paired(flat, a, 'asinh_reference'), 'per_sequence':
            {s: paired(rows, a, 'asinh_reference') for s, rows in sorted(by_sequence.items())}} for a in ARMS},
        'seq23_viewport_summaries': {a: summarize([r['viewport'] for r in by_sequence.get('seq23', [])], a) for a in names},
        'no_initialization': ['seq1', 'seq7'], 'no_initialization_dice': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('verify', 'score', 'audit'), required=True)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.batch_root.resolve(strict=True); out = args.output.resolve()
    require(root.is_relative_to(BASE.resolve(strict=True)) and out.is_relative_to(root) and not out.exists(), 'Invalid output')
    guard = install_mask_open_guard(DATA) if args.mode == 'verify' else install_phase_mask_guard(DATA)
    plan_path = args.plan.resolve(strict=True); plan, sha, warm, warm_sha, index, samples = study_inputs(plan_path)
    completed = () if args.mode == 'verify' else ('verify',) if args.mode == 'score' else ('verify', 'score')
    mapping, proof = validate_all(root, plan_path, plan, sha, warm, warm_sha, index, samples, completed=completed)
    require(guard['open_attempts'] == 0, 'Label opened before control verification')
    if args.mode == 'verify':
        proof.update(passed=True, label_guard=dict(guard), source_sha256=digest(Path(__file__)))
        out.mkdir(parents=True); (out/'verification.json').write_text(json.dumps(proof, indent=2) + '\n'); return
    verification = root/'verify/verification.json'; verification_sha = phase_artifact(root, 'verify', verification)
    saved = read_bound(verification, verification_sha)
    require(saved.get('passed') is True and saved.get('label_guard', {}).get('open_attempts') == 0 and
            saved.get('source_sha256') == digest(Path(__file__)) and all(saved.get(k) == v for k, v in proof.items()),
            'Control verification differs')
    if args.mode == 'audit':
        score_path = root/'score/results.json'; score_sha = phase_artifact(root, 'score', score_path)
        score = read_bound(score_path, score_sha)
        require(score.get('passed') is True and score.get('verification_sha256') == verification_sha and
                score.get('report_sha256') == digest(root/'score/report.md'), 'Score differs')
    require(guard['open_attempts'] == 0, 'Label opened before score release'); guard['active'] = False
    result = compute(index, samples, mapping, independent=args.mode == 'audit'); out.mkdir(parents=True)
    if args.mode == 'score':
        lines = ['# Exploratory four-control cross-fit results', '',
            'Previously inspected patients; sampled post-prompt reads, not live video. '
            'Seq1 and seq7 have no automatic initialization and are N/A.', '',
            '| Arm | Patient-macro Dice | Patients | Frame-weighted Dice | Present reads | Empty false alarms |',
            '|---|---:|---:|---:|---:|---:|']
        for name, summary in result['summaries'].items():
            pooled = summary['pooled']; macro = summary['patient_macro_present_dice']; dice = pooled['mean_present_dice']
            lines.append(f"| {name} | {'N/A' if macro is None else f'{macro:.4f}'} | "
                         f"{summary['patients_with_present_reads']} | {'N/A' if dice is None else f'{dice:.4f}'} | "
                         f"{pooled['present_frames']} | {pooled['false_alarms']}/{pooled['empty_frames']} |")
        lines += ['', 'Paired results against the separately audited asinh reference and seq23 viewport results '
                  'are available in results.json.', '']
        report = out/'report.md'; report.write_text('\n'.join(lines))
        result.update(passed=True, verification_sha256=verification_sha, prelabel_guard=dict(guard),
                      report_sha256=digest(report),
                      scope='Previously inspected patients; fixed reads, not live-video propagation.')
        (out/'results.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        errors = []
        for key in ('rows', 'summaries', 'paired_vs_baseline', 'paired_vs_asinh_reference',
                    'seq23_viewport_summaries', 'no_initialization', 'no_initialization_dice'):
            compare(result[key], score[key], [key], errors)
        (out/'audit.json').write_text(json.dumps({'passed': not errors, 'score_sha256': score_sha,
            'prelabel_guard': dict(guard), 'error_count': len(errors), 'errors': errors,
            'scope': 'Independent metric calculation; shared structural verifier.'}, indent=2) + '\n')
        require(not errors, 'Independent control metric audit differs')


if __name__ == '__main__':
    main()
