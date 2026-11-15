#!/usr/bin/env python3
"""Supervise bounded-source three-fold fixed-read training and evaluation.

This runner is disabled pending independent review and completed real capture
proofs. It does not measure live-video propagation or support confirmatory
generalization claims.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import bounded_crossfit_shared_worker_3fold_v1 as worker
from scripts import probe_bounded_crossfit_shared_3fold_v1 as probe_module
from scripts import predict_bounded_crossfit_fold_3fold_v1 as prediction_module
from scripts import score_bounded_crossfit_3fold_v1 as scoring_module
from scripts.channel_value_fixed_read_v1 import require_environment_flags
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA, OUT
from scripts.memory_token_io import digest
from scripts.preflight_bounded_crossfit_training_v1 import read_bound_json, require
from scripts.preflight_warm_crossfit_training_v1 import ARMS
from scripts.resume_source_group_capture_v1 import exclusive_lock
from scripts.run_boundary_video_factorial_v1 import phase_environment
from scripts.source_group_label_guard_v1 import install_mask_open_guard


GIB = 1024 ** 3
ENABLE_REVIEWED_STUDY = True


def phases(root, registry, cache_mib, max_rss=6 * GIB):
    root = Path(root); scripts = ROOT / 'scripts'; python = sys.executable
    verification = root / 'verification' / 'verification.json'
    probe = root / 'probe' / 'completion.json'
    audit = root / 'probe_audit' / 'audit.json'
    authorization = root / 'authorization' / 'authorization.json'
    commands = [
        ('verification', [python, str(scripts / 'verify_cache_release_crossfit_registry_3fold_v1.py'),
                          '--registry', str(registry), '--output', str(verification.parent)], verification),
        ('probe', [python, str(scripts / 'probe_bounded_crossfit_shared_3fold_v1.py'),
                   '--batch-root', str(root), '--registry', str(registry), '--verification', str(verification), '--output', str(probe.parent),
                   '--max-process-rss-bytes', str(max_rss)], probe),
        ('probe_audit', [python, str(scripts / 'audit_bounded_crossfit_shared_probe_3fold_v1.py'),
                         '--registry', str(registry), '--verification', str(verification), '--probe', str(probe),
                         '--output', str(audit.parent)], audit),
        ('authorization', [python, str(scripts / 'authorize_bounded_crossfit_shared_3fold_v1.py'),
                           '--registry', str(registry), '--verification', str(verification), '--probe', str(probe),
                           '--probe-audit', str(audit), '--output', str(authorization.parent)], authorization),
    ]
    for fold in range(1, 4):
        folder = root / f'fold{fold}'
        commands.append((f'train_f{fold}', [python, str(scripts / 'run_bounded_crossfit_fold_3fold_v1.py'),
            '--batch-root', str(root), '--authorization', str(authorization), '--fold', str(fold), '--output', str(folder),
            '--cache-mib', str(cache_mib)], folder / 'group.json'))
    for fold in range(1, 4):
        for arm in ARMS:
            folder = root / 'predict' / f'fold{fold}' / arm
            commands.append((f'predict_f{fold}_{arm}', [python, str(scripts / 'predict_bounded_crossfit_fold_3fold_v1.py'),
                '--batch', str(root), '--authorization', str(authorization), '--fold', str(fold),
                '--arm', arm, '--output', str(folder)], folder / 'completion.json'))
    for mode, artifact in (('verify', 'verification.json'), ('score', 'results.json'), ('audit', 'audit.json')):
        folder = root / mode
        commands.append((mode, [python, str(scripts / 'score_bounded_crossfit_3fold_v1.py'), '--mode', mode,
                                '--batch-root', str(root), '--output', str(folder)], folder / artifact))
    return commands


def run_monitored(command, log_path, root, *, min_disk, min_ram, max_rss):
    minimum_disk = minimum_ram = None; peak = 0
    def sample(process=None):
        nonlocal minimum_disk, minimum_ram, peak
        disk = shutil.disk_usage(root).free; ram = psutil.virtual_memory().available; rss = 0
        if process is not None:
            try:
                rss = process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        minimum_disk = disk if minimum_disk is None else min(minimum_disk, disk)
        minimum_ram = ram if minimum_ram is None else min(minimum_ram, ram)
        peak = max(peak, rss)
        require(disk >= min_disk and ram >= min_ram and rss <= max_rss,
                f'Resource guard crossed: disk={disk}, RAM={ram}, process RSS={rss}')
    sample()
    with Path(log_path).open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=phase_environment(), stdout=stream, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        try:
            while True:
                try:
                    code = child.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    sample(process)
                else:
                    require(code == 0, f'Study phase failed: {log_path}')
                    sample()
                    return {'minimum_sampled_disk_free_bytes': minimum_disk,
                            'minimum_sampled_system_available_ram_bytes': minimum_ram,
                            'peak_sampled_process_rss_bytes': peak}
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
            raise


def snapshot_sources(destination, source_map):
    """Keep the exact reviewed source bytes with the study's phase receipts."""
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    for name, sha in source_map.items():
        source = (ROOT / name).resolve(strict=True)
        require(source.is_relative_to(ROOT.resolve(strict=True)) and digest(source) == sha,
                'Source changed before snapshot')
        output = destination / name; output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        require(digest(output) == sha and digest(source) == sha, 'Source changed during snapshot')
    path = destination / 'snapshot_manifest.json'
    path.write_text(json.dumps({'source_sha256': source_map}, indent=2) + '\n')
    return digest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--cache-mib', type=int, default=1024)
    parser.add_argument('--min-disk-gib', type=int, default=26)
    parser.add_argument('--min-available-ram-gib', type=int, default=4)
    parser.add_argument('--max-process-rss-gib', type=int, default=6)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(0 <= args.cache_mib <= 1024 and args.min_disk_gib >= 26 and
            args.min_available_ram_gib >= 4 and 1 <= args.max_process_rss_gib <= 6, 'Invalid resource limits')
    registry = args.registry.resolve(strict=True)
    require(registry.is_relative_to(BASE.resolve(strict=True)), 'Registry outside Drive experiment root')
    guard = install_mask_open_guard(DATA)
    identity = worker.sources()
    record, registry_sha = read_bound_json(registry / 'registry.json')
    done, completion_sha = read_bound_json(registry / 'completion.json')
    require(record.get('passed') is True and record.get('training_authorized') is False and
            done.get('passed') is True and done.get('training_authorized') is False and
            done.get('registry_sha256') == registry_sha and record.get('label_guard', {}).get('open_attempts') == 0 and
            done.get('label_guard', {}).get('open_attempts') == 0, 'Mixed registry receipt differs')
    free = shutil.disk_usage(BASE).free; ram = psutil.virtual_memory().available
    require(free >= args.min_disk_gib * GIB and ram >= args.min_available_ram_gib * GIB and
            guard['open_attempts'] == 0, 'Insufficient resources or target label opened')
    if not args.execute:
        print(json.dumps({'passed': True, 'scope': 'Source/receipt resource check only; no proof reconstruction, MPS or training.',
                          'disk_free_bytes': free, 'available_ram_bytes': ram})); return
    require(ENABLE_REVIEWED_STUDY and worker.ENABLE_REVIEWED_TRAINING and probe_module.ENABLE_REVIEWED_PROBE and
            prediction_module.ENABLE_REVIEWED_PREDICTION and scoring_module.ENABLE_REVIEWED_SCORING,
            'Mixed-source study execution awaits independent review and real proofs')
    require_environment_flags(); require(torch.backends.mps.is_available(), 'MPS unavailable')
    with exclusive_lock(OUT / '.source_group_capture_resume_v1.lock'):
        require(worker.sources() == identity and digest(registry / 'registry.json') == registry_sha and
                digest(registry / 'completion.json') == completion_sha, 'Frozen source or registry changed')
        root = Path(tempfile.mkdtemp(prefix='bounded_shared_crossfit_3fold_v1_', dir=OUT))
        tasks = phases(root, registry, args.cache_mib, args.max_process_rss_gib * GIB)
        snapshot_sha = snapshot_sources(root / 'source_snapshot', identity)
        manifest = {'format': 'bounded_shared_crossfit_training_study_3fold_v1', 'registry_path': str(registry),
            'registry_sha256': registry_sha, 'registry_completion_sha256': completion_sha,
            'source_sha256': identity, 'phases': [name for name, _, _ in tasks],
            'source_snapshot_sha256': snapshot_sha,
            'supervisor_pid': psutil.Process().pid, 'supervisor_create_time': psutil.Process().create_time(),
            'min_disk_bytes': args.min_disk_gib * GIB, 'min_available_ram_bytes': args.min_available_ram_gib * GIB,
            'max_process_rss_bytes': args.max_process_rss_gib * GIB, 'cache_mib': args.cache_mib,
            'scope': 'Exploratory mixed-source three-fold fixed-read study; no live-video efficacy.'}
        manifest_path = root / 'run_manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n'); manifest_sha = digest(manifest_path)
        for name, command, artifact in tasks:
            require(digest(manifest_path) == manifest_sha and worker.sources() == identity and
                    digest(registry / 'registry.json') == registry_sha and guard['open_attempts'] == 0,
                    'Frozen inputs changed during study')
            print(json.dumps({'phase': name, 'status': 'starting', 'output': str(root)}), flush=True)
            started = time.monotonic()
            try:
                monitor = run_monitored(command, root / f'{name}.log', root,
                    min_disk=args.min_disk_gib * GIB, min_ram=args.min_available_ram_gib * GIB,
                    max_rss=args.max_process_rss_gib * GIB)
                record, artifact_sha = read_bound_json(artifact)
                require(record.get('passed') is True and worker.sources() == identity, 'Phase artifact or source differs')
                if name.startswith('train_'):
                    require(record.get('label_guard', {}).get('blocked_open_attempts') == 0 and
                            0 < record.get('peak_process_rss_bytes', 0) <= args.max_process_rss_gib * GIB,
                            'Fold guard or recorded high-water RSS differs')
                else:
                    require(record.get('label_guard', {}).get('open_attempts') == 0, 'Pretraining target opened')
                    if name == 'probe':
                        require(type(record.get('peak_process_rss_bytes')) is int and
                                0 < record['peak_process_rss_bytes'] <= args.max_process_rss_gib * GIB and
                                record.get('max_process_rss_bytes') == args.max_process_rss_gib * GIB,
                                'Probe recorded high-water RSS differs')
            except BaseException as error:
                (root / 'failure.json').write_text(json.dumps({'phase': name, 'error': str(error),
                    'scope': 'Partial outputs and logs retained.'}, indent=2) + '\n')
                raise
            with (root / 'phases.jsonl').open('a') as stream:
                stream.write(json.dumps({'phase': name, 'passed': True, 'artifact_path': str(artifact),
                    'artifact_sha256': artifact_sha, 'log_sha256': digest(root / f'{name}.log'),
                    'wall_seconds': time.monotonic() - started, 'resource_monitor': monitor}) + '\n')
            print(json.dumps({'phase': name, 'status': 'completed'}), flush=True)
    print(json.dumps({'passed': True, 'output': str(root), 'scope': 'Exploratory fixed-read study; no live-video claim.'}))


if __name__ == '__main__':
    main()
