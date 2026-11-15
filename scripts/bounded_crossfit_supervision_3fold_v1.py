"""Fail-closed validation of a source-bound bounded-study child process.

This check is an operational invariant, not a security sandbox. It requires
the recorded live supervisor parent, its held shared lock, frozen phase order,
resource limits, and the expected child artifact location. The parent remains
responsible for live sampling and preserving partial artifacts on failure.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.crossfit_observer_capture_shared_v1 import BASE, OUT, phase_artifact
from scripts.memory_token_io import digest
from scripts.preflight_bounded_crossfit_training_v1 import read_bound_json, require


def check_source_snapshot(root, manifest):
    root = Path(root).resolve(strict=True)
    snapshot_root = root / 'source_snapshot'
    snapshot, snapshot_sha = read_bound_json(snapshot_root / 'snapshot_manifest.json')
    require(snapshot_sha == manifest.get('source_snapshot_sha256') and
            snapshot.get('source_sha256') == manifest.get('source_sha256'), 'Source snapshot manifest differs')
    for name, expected in snapshot['source_sha256'].items():
        path = (snapshot_root / name).resolve(strict=True)
        require(path.is_relative_to(snapshot_root.resolve(strict=True)) and digest(path) == expected,
                'Source snapshot content differs')


def assert_supervised(root, phase, output):
    from scripts import bounded_crossfit_shared_worker_3fold_v1 as worker
    from scripts.run_bounded_crossfit_shared_study_3fold_v1 import GIB, phases

    root = Path(root).resolve(strict=True); output = Path(output).resolve()
    require(root.is_relative_to(OUT.resolve(strict=True)) and output.is_relative_to(root), 'Invalid study root')
    manifest_path = root / 'run_manifest.json'; manifest, manifest_sha = read_bound_json(manifest_path)
    registry = Path(manifest['registry_path']).resolve(strict=True)
    require(registry.is_relative_to(BASE.resolve(strict=True)) and manifest.get('format') ==
            'bounded_shared_crossfit_training_study_3fold_v1' and manifest.get('source_sha256') == worker.sources() and
            manifest.get('registry_sha256') == digest(registry / 'registry.json') and
            manifest.get('registry_completion_sha256') == digest(registry / 'completion.json') and
            type(manifest.get('cache_mib')) is int and 0 <= manifest['cache_mib'] <= 1024 and
            type(manifest.get('min_disk_bytes')) is int and manifest['min_disk_bytes'] >= 26 * GIB and
            type(manifest.get('min_available_ram_bytes')) is int and manifest['min_available_ram_bytes'] >= 4 * GIB and
            type(manifest.get('max_process_rss_bytes')) is int and 0 < manifest['max_process_rss_bytes'] <= 6 * GIB,
            'Supervised manifest, registry or resource policy differs')
    require(type(manifest.get('supervisor_pid')) is int and manifest['supervisor_pid'] == os.getppid() and
            isinstance(manifest.get('supervisor_create_time'), (int, float)), 'Supervisor parent differs')
    parent = psutil.Process(os.getppid())
    cmd = parent.cmdline()
    require(parent.is_running() and parent.create_time() == manifest['supervisor_create_time'] and len(cmd) >= 2 and
            Path(cmd[1] if Path(cmd[1]).is_absolute() else Path(parent.cwd()) / cmd[1]).resolve() ==
            (ROOT / 'scripts/run_bounded_crossfit_shared_study_3fold_v1.py').resolve() and
            os.path.realpath(parent.exe()) == os.path.realpath(psutil.Process().exe()), 'Supervisor process identity differs')
    lock_path = OUT / '.source_group_capture_resume_v1.lock'
    require(lock_path.is_file(), 'Shared study lock is missing')
    with lock_path.open('r') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise ValueError('Shared study lock is not held')
    check_source_snapshot(root, manifest)
    tasks = phases(root, registry, manifest['cache_mib'], manifest['max_process_rss_bytes'])
    require(manifest.get('phases') == [name for name, _, _ in tasks], 'Supervised phase plan differs')
    ledger_path = root / 'phases.jsonl'
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()] if ledger_path.exists() else []
    require(len(rows) < len(tasks) and [row.get('phase') for row in rows] ==
            [name for name, _, _ in tasks[:len(rows)]] and tasks[len(rows)][0] == phase and
            tasks[len(rows)][2].parent == output, 'Child phase or output differs')
    for row, (_, _, artifact) in zip(rows, tasks, strict=False):
        require(phase_artifact(root, row['phase'], artifact) == row.get('artifact_sha256'),
                'Earlier supervised phase changed')
    require(shutil.disk_usage(root).free >= manifest['min_disk_bytes'] and
            psutil.virtual_memory().available >= manifest['min_available_ram_bytes'] and
            digest(manifest_path) == manifest_sha and worker.sources() == manifest['source_sha256'],
            'Supervised resources or source changed')
    return manifest
