#!/usr/bin/env python3
"""Resumable fold-scoped warm fixed-read study; default is a label-free dry run.

All model fits finish before any held-out predictions; all predictions and
artifact verification finish before the separate scorer can open evaluation
labels. This launcher does not claim a blind study or live-video efficacy.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.crossfit_observer_capture_shared_v1 import DATA, OUT, phase_artifact
from scripts.memory_token_io import digest
from scripts.preflight_warm_crossfit_training_v1 import ARMS, sources
from scripts.resume_source_group_capture_v1 import exclusive_lock
from scripts.run_boundary_video_factorial_v1 import phase_environment
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_fold_training_v1 import load_preflight, require

GUARD = 22 * 1024**3


def phases(root, preflight, cache_mib=1024):
    root = Path(root); py = sys.executable
    worker = ROOT / 'scripts/warm_crossfit_fold_training_v1.py'
    scorer = ROOT / 'scripts/score_warm_crossfit_v1.py'
    probe = root / 'probe'
    tasks = [('probe', [py, str(ROOT / 'scripts/warm_crossfit_probe_v1.py'), '--preflight', str(preflight),
                        '--output', str(probe)], probe / 'completion.json')]
    for mode in ('train', 'predict'):
        for fold in range(1, 6):
            for arm in ARMS:
                name = f'{mode}_f{fold}_{arm}'
                out = root / mode / f'fold{fold}' / arm
                cmd = [py, str(worker), '--mode', mode, '--preflight', str(preflight), '--fold', str(fold),
                       '--arm', arm, '--output', str(out)]
                if mode != 'train':
                    cmd += ['--training', str(root / 'train' / f'fold{fold}' / arm)]
                else:
                    cmd += ['--cache-mib', str(cache_mib)]
                tasks.append((name, cmd, out / 'completion.json'))
    for mode in ('verify', 'score', 'audit'):
        out = root / mode
        file = out / ('verification.json' if mode == 'verify' else 'results.json' if mode == 'score' else 'audit.json')
        tasks.append((mode, [py, str(scorer), '--mode', mode, '--preflight', str(preflight),
                             '--batch-root', str(root), '--output', str(out)], file))
    return tasks


def headroom(root):
    free = shutil.disk_usage(root).free
    require(free >= GUARD, f'Free disk {free} is below reserve {GUARD}')
    return free


def run_phase(command, log, root):
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=phase_environment(), stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                code = child.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                if shutil.disk_usage(root).free < GUARD:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                    raise RuntimeError('Disk reserve crossed; partial phase preserved')
        require(code == 0 and shutil.disk_usage(root).free >= GUARD, f'Phase failed or disk reserve crossed: {log}')


def validate_resume(root, preflight, source_sha, preflight_sha, cache_mib):
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(OUT.resolve(strict=True)), 'Study outside Drive')
    manifest_path = root / 'run_manifest.json'; manifest = json.loads(manifest_path.read_text())
    tasks = phases(root, preflight, cache_mib)
    require(manifest.get('preflight_path') == str(preflight) and manifest.get('preflight_sha256') == preflight_sha and
            manifest.get('source_sha256') == source_sha and manifest.get('cache_mib') == cache_mib and
            manifest.get('phases') == [x[0] for x in tasks],
            'Study manifest changed')
    phase_path = root / 'phases.jsonl'
    rows = [json.loads(line) for line in phase_path.read_text().splitlines() if line.strip()] if phase_path.exists() else []
    require(len(rows) < len(tasks) and [x['phase'] for x in rows] == [x[0] for x in tasks[:len(rows)]],
            'Completed phases are not an incomplete prefix')
    for row, (_, _, artifact) in zip(rows, tasks, strict=False):
        require(phase_artifact(root, row['phase'], artifact) == row['artifact_sha256'], 'Completed phase changed')
    return root, digest(manifest_path), len(rows)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', required=True, type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--cache-mib', type=int, default=1024)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(0 <= args.cache_mib <= 1024, 'CPU cache must be between 0 and 1024 MiB')
    guard = install_mask_open_guard(DATA)
    preflight_path = args.preflight.resolve(strict=True)
    _, preflight_sha = load_preflight(preflight_path)
    source_sha = sources(); free = headroom(OUT)
    if args.resume:
        validate_resume(args.resume, preflight_path, source_sha, preflight_sha, args.cache_mib)
    require(guard['open_attempts'] == 0, 'Label opened in launcher')
    if not args.execute:
        print(json.dumps({'passed': True, 'scope': 'Dry run only; no GPU or labels.', 'free_bytes': free,
            'cache_mib': args.cache_mib,
            'phases': [x[0] for x in phases(OUT / 'DRY_RUN', preflight_path, args.cache_mib)]}))
        return
    import torch
    require(torch.backends.mps.is_available(), 'MPS unavailable')
    with exclusive_lock(OUT / '.source_group_capture_resume_v1.lock'):
        _, actual_sha = load_preflight(preflight_path)
        require(actual_sha == preflight_sha and sources() == source_sha, 'Inputs changed before launch')
        if args.resume:
            root, manifest_sha, completed = validate_resume(args.resume, preflight_path, source_sha, preflight_sha, args.cache_mib)
            name, _, artifact = phases(root, preflight_path, args.cache_mib)[completed]
            preserve_partial(root, name, artifact)
        else:
            root = Path(tempfile.mkdtemp(prefix='warm_crossfit_training_v1_', dir=OUT))
            manifest = root / 'run_manifest.json'
            manifest.write_text(json.dumps({'preflight_path': str(preflight_path), 'preflight_sha256': preflight_sha,
                'source_sha256': source_sha, 'cache_mib': args.cache_mib,
                'phases': [x[0] for x in phases(root, preflight_path, args.cache_mib)],
                'scope': 'Previously inspected patients; exploratory fixed-read only.'}, indent=2) + '\n')
            manifest_sha = digest(manifest); completed = 0
        tasks = phases(root, preflight_path, args.cache_mib)
        for name, command, artifact in tasks[completed:]:
            require(digest(root / 'run_manifest.json') == manifest_sha and sources() == source_sha and
                    load_preflight(preflight_path)[1] == preflight_sha and guard['open_attempts'] == 0,
                    'Frozen study or label guard changed')
            free = headroom(root); started = time.monotonic()
            print(json.dumps({'phase': name, 'status': 'starting', 'free_bytes': free, 'output': str(root)}), flush=True)
            run_phase(command, root / (name + '.log'), root)
            require(artifact.is_file() and sources() == source_sha, 'Phase artifact or source differs')
            with (root / 'phases.jsonl').open('a') as log:
                log.write(json.dumps({'phase': name, 'passed': True, 'artifact_path': str(artifact),
                    'artifact_sha256': digest(artifact), 'log_sha256': digest(root / (name + '.log')),
                    'wall_seconds': time.monotonic() - started}) + '\n')
            print(json.dumps({'phase': name, 'status': 'completed'}), flush=True)
    print(json.dumps({'passed': True, 'output': str(root)}))


if __name__ == '__main__':
    main()
