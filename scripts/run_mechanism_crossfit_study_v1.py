#!/usr/bin/env python3
"""Resumable four-control fixed-read study; default is a no-GPU dry run."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.crossfit_observer_capture_shared_v1 import DATA, OUT, phase_artifact
from scripts.memory_token_io import digest
from scripts.mechanism_crossfit_control_core_v1 import ARMS
from scripts.preflight_mechanism_crossfit_v1 import sources
from scripts.resume_source_group_capture_v1 import exclusive_lock
from scripts.run_mechanism_crossfit_fold_v1 import read_bound, study_inputs
from scripts.run_warm_crossfit_shared_v1 import headroom, run_phase
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_fold_training_v1 import require


def phases(root, plan, cache_mib):
    root = Path(root); py = sys.executable
    worker = str(ROOT / 'scripts/run_mechanism_crossfit_fold_v1.py')
    tasks = [('probe', [py, worker, '--mode', 'probe', '--plan', str(plan),
                       '--output', str(root / 'probe')], root / 'probe/completion.json')]
    for fold in range(1, 6):
        out = root / 'train' / f'fold{fold}'
        tasks.append((f'train_f{fold}', [py, worker, '--mode', 'train', '--plan', str(plan), '--fold', str(fold),
                      '--cache-mib', str(cache_mib), '--output', str(out)], out / 'group.json'))
    for fold in range(1, 6):
        out = root / 'predict' / f'fold{fold}'
        tasks.append((f'predict_f{fold}', [py, worker, '--mode', 'predict', '--plan', str(plan), '--fold', str(fold),
                      '--training', str(root / 'train' / f'fold{fold}'), '--cache-mib', str(cache_mib),
                      '--output', str(out)], out / 'completion.json'))
    for mode in ('verify', 'score', 'audit'):
        out = root / mode
        artifact = {'verify': 'verification.json', 'score': 'results.json', 'audit': 'audit.json'}[mode]
        tasks.append((mode, [py, str(ROOT / 'scripts/score_mechanism_crossfit_v1.py'), '--mode', mode,
                      '--plan', str(plan), '--batch-root', str(root), '--output', str(out)], out / artifact))
    return tasks


def validate_resume(root, plan_path, plan_sha, source_sha, cache_mib):
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(OUT.resolve(strict=True)), 'Control study outside Drive')
    path = root / 'run_manifest.json'; sha = digest(path); manifest = json.loads(path.read_text())
    tasks = phases(root, plan_path, cache_mib)
    require(digest(path) == sha and manifest.get('plan_path') == str(plan_path) and
            manifest.get('plan_sha256') == plan_sha and manifest.get('source_sha256') == source_sha and
            manifest.get('cache_mib') == cache_mib and manifest.get('phases') == [x[0] for x in tasks],
            'Control study manifest differs')
    log = root / 'phases.jsonl'
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    require(len(rows) < len(tasks) and [x['phase'] for x in rows] == [x[0] for x in tasks[:len(rows)]],
            'Control phases are not an incomplete prefix')
    for row, (_, _, artifact) in zip(rows, tasks, strict=False):
        require(phase_artifact(root, row['phase'], artifact) == row['artifact_sha256'], 'Completed control phase changed')
    return root, sha, len(rows)


def preserve_partial(root, phase, artifact):
    folder = artifact.parent; log = root / (phase + '.log')
    if not folder.exists() and not log.exists():
        return
    destination = root / 'partial_attempts' / f'{phase}_{uuid.uuid4().hex}'
    destination.mkdir(parents=True, exist_ok=False)
    if folder.exists():
        folder.rename(destination / 'output')
    if log.exists():
        log.rename(destination / 'phase.log')


def check_probe(root, plan_sha, warm_sha, warm, samples):
    file = root/'probe/completion.json'; sha = phase_artifact(root, 'probe', file)
    record = read_bound(file, sha); checks = record.get('checks', [])
    require(record.get('passed') is True and record.get('plan_sha256') == plan_sha and
            record.get('warm_preflight_sha256') == warm_sha and record.get('device') == 'mps' and
            record.get('environment') == warm['environment'] and record.get('label_guard', {}).get('open_attempts') == 0 and
            len(checks) == 2*len(ARMS) and {(x['arm'], x['sequence']) for x in checks} ==
            {(arm, seq) for arm in ARMS for seq in ('seq6', 'seq23')}, 'Control probe differs')
    for check in checks:
        path = Path(check['checkpoint_path']).resolve(strict=True)
        require(path.is_relative_to((root/'probe').resolve()) and digest(path) == check['checkpoint_sha256'] and
                (check['sequence'], check['frame']) in samples and
                (check['sequence'] != 'seq23' or check['video_shape'] == [720, 834]) and
                math.isfinite(check['head_bias_gradient_abs']) and check['head_bias_gradient_abs'] > 0 and
                math.isfinite(check['gradient_l2']) and check['gradient_l2'] > 0, 'Control probe evidence differs')
    return sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--resume', type=Path)
    parser.add_argument('--cache-mib', type=int, default=1024); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(); require(0 <= args.cache_mib <= 1024, 'Invalid CPU cache size')
    guard = install_mask_open_guard(DATA); plan_path = args.plan.resolve(strict=True)
    plan, plan_sha, warm, warm_sha, _, samples = study_inputs(plan_path)
    source_sha = sources(); free = headroom(OUT)
    if args.resume:
        validate_resume(args.resume, plan_path, plan_sha, source_sha, args.cache_mib)
    require(guard['open_attempts'] == 0, 'Label opened during control launch')
    if not args.execute:
        print(json.dumps({'passed': True, 'scope': 'Dry run; no MPS or labels.', 'free_bytes': free,
                          'phases': [x[0] for x in phases(OUT/'DRY_RUN', plan_path, args.cache_mib)]}))
        return
    import torch
    require(torch.backends.mps.is_available(), 'MPS unavailable')
    with exclusive_lock(OUT / '.source_group_capture_resume_v1.lock'):
        require(study_inputs(plan_path)[1] == plan_sha and sources() == source_sha, 'Control inputs changed')
        if args.resume:
            root, manifest_sha, completed = validate_resume(args.resume, plan_path, plan_sha, source_sha, args.cache_mib)
            name, _, artifact = phases(root, plan_path, args.cache_mib)[completed]
            preserve_partial(root, name, artifact)
        else:
            root = Path(tempfile.mkdtemp(prefix='mechanism_crossfit_v1_', dir=OUT))
            path = root / 'run_manifest.json'
            path.write_text(json.dumps({'plan_path': str(plan_path), 'plan_sha256': plan_sha,
                'source_sha256': source_sha, 'cache_mib': args.cache_mib,
                'phases': [x[0] for x in phases(root, plan_path, args.cache_mib)],
                'scope': 'Exploratory, previously inspected patients; fixed reads only.'}, indent=2) + '\n')
            manifest_sha = digest(path); completed = 0
        for name, command, artifact in phases(root, plan_path, args.cache_mib)[completed:]:
            require(digest(root/'run_manifest.json') == manifest_sha and sources() == source_sha and
                    study_inputs(plan_path)[1] == plan_sha and guard['open_attempts'] == 0,
                    'Frozen control inputs differ')
            if name.startswith(('train_', 'predict_')):
                check_probe(root, plan_sha, warm_sha, warm, samples)
            free = headroom(root); started = time.monotonic()
            print(json.dumps({'phase': name, 'status': 'starting', 'free_bytes': free, 'output': str(root)}), flush=True)
            run_phase(command, root / (name + '.log'), root)
            require(artifact.is_file() and sources() == source_sha, 'Control phase artifact or sources differ')
            with (root/'phases.jsonl').open('a') as stream:
                stream.write(json.dumps({'phase': name, 'passed': True, 'artifact_path': str(artifact),
                    'artifact_sha256': digest(artifact), 'log_sha256': digest(root/(name+'.log')),
                    'wall_seconds': time.monotonic() - started}) + '\n')
            print(json.dumps({'phase': name, 'status': 'completed'}), flush=True)
    print(json.dumps({'passed': True, 'output': str(root)}))


if __name__ == '__main__':
    main()
