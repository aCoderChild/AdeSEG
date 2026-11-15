#!/usr/bin/env python3
"""Guard and record two fresh-process short MPS loader diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import psutil
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import DATA, OUT
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import exclusive_lock
from scripts.run_boundary_video_factorial_v1 import phase_environment
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_video_contract_v1 import require

GIB = 1024**3
ENTRYPOINTS = ('scripts/run_mps_memory_diagnostic_study_v1.py','scripts/run_mps_memory_diagnostic_v1.py',
               'scripts/audit_mps_memory_diagnostic_v1.py','modeling/bounded_sam2_video_loader_v2.py',
               'modeling/bounded_loader_feature_observer_v1.py')
PROTOCOL = 'protocols/MPS_MEMORY_DIAGNOSTIC_V1.md'
AUTHOR_FILES = ('external/Lean-SAM2-token-window/sam2/sam2_video_predictor_swr_wrapper.py',
                'external/Lean-SAM2-token-window/sam2/sam2_video_predictor_topk_insurance.py',
                'external/Lean-SAM2-token-window/sam2/utils/misc.py')


def sources():
    names = set(local_import_closure(ROOT, ENTRYPOINTS)); names.add(PROTOCOL)
    return {name: digest(ROOT/name) for name in sorted(names)}



def snapshot_sources(destination, project_root, source_map):
    destination=Path(destination); project_root=Path(project_root).resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=False)
    for name, sha in source_map.items():
        source=(project_root/name).resolve(strict=True)
        require(source.is_relative_to(project_root) and digest(source)==sha, 'Source changed before snapshot')
        target=destination/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source,target)
        require(digest(target)==sha and digest(source)==sha, 'Source changed during snapshot')
    manifest=destination/'snapshot_manifest.json'
    manifest.write_text(json.dumps({'source_sha256':source_map},indent=2)+'\n')
    return digest(manifest)


def resources(root, min_disk, min_ram):
    disk = shutil.disk_usage(root).free; memory = psutil.virtual_memory().available
    require(disk >= min_disk and memory >= min_ram,
            f'Insufficient resources: disk={disk}, RAM_available={memory}, required={min_disk}/{min_ram}')
    return disk, memory


def run_monitored(command, log, root, min_disk, min_ram, max_rss):
    peak = 0; minimum_disk, minimum_ram = resources(root, min_disk, min_ram)
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=phase_environment(), stdout=stream, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        try:
            while True:
                try:
                    code = child.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    try: rss = process.memory_info().rss
                    except psutil.NoSuchProcess: rss = 0
                    disk = shutil.disk_usage(root).free; ram = psutil.virtual_memory().available
                    peak = max(peak, rss); minimum_disk = min(minimum_disk, disk); minimum_ram = min(minimum_ram, ram)
                    if disk < min_disk or ram < min_ram or rss > max_rss:
                        raise RuntimeError(f'Resource threshold crossed: disk={disk}, RAM_available={ram}, process_rss={rss}')
                else:
                    require(code == 0, f'Diagnostic phase failed: {log}')
                    disk, ram = resources(root, min_disk, min_ram)
                    minimum_disk = min(minimum_disk, disk); minimum_ram = min(minimum_ram, ram)
                    return {'peak_sampled_process_rss_bytes': peak, 'minimum_sampled_disk_free_bytes': minimum_disk,
                            'minimum_sampled_system_available_ram_bytes': minimum_ram}
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--certificate', type=Path, required=True); p.add_argument('--execute', action='store_true')
    p.add_argument('--min-disk-gib', type=int, default=26); p.add_argument('--min-available-ram-gib', type=int, default=4)
    p.add_argument('--max-process-rss-gib', type=int, default=6); p.add_argument('--cache-frames', type=int, default=4)
    a = p.parse_args(); require(a.min_disk_gib >= 26 and a.min_available_ram_gib >= 4 and
        1 <= a.max_process_rss_gib <= 6 and 1 <= a.cache_frames <= 4, 'Unsafe diagnostic resource settings')
    guard = install_mask_open_guard(DATA); certificate, cert_sha = verify_certificate(a.certificate)
    source_map = sources(); disk, ram = resources(OUT, a.min_disk_gib*GIB, a.min_available_ram_gib*GIB)
    require(guard['open_attempts'] == 0, 'Annotation opened in diagnostic preflight')
    if not a.execute:
        print(json.dumps({'passed': True, 'scope': 'CPU-only dry run; no diagnostic inference.',
            'disk_free_bytes': disk, 'system_available_ram_bytes': ram, 'certificate_sha256': cert_sha,
            'min_disk_bytes':a.min_disk_gib*GIB, 'min_ram_bytes':a.min_available_ram_gib*GIB,
            'max_process_rss_bytes':a.max_process_rss_gib*GIB})); return
    import torch
    require(torch.backends.mps.is_available(), 'MPS unavailable')
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        current, current_sha = verify_certificate(a.certificate)
        require(current == certificate and current_sha == cert_sha and sources() == source_map, 'Preflight changed')
        root = Path(tempfile.mkdtemp(prefix='mps_memory_diagnostic_v1_', dir=OUT))
        snapshot_map={**source_map, **{name:digest(ROOT/name) for name in AUTHOR_FILES}}
        snapshot_sha=snapshot_sources(root/'source_snapshot', ROOT, snapshot_map)
        manifest = {'scope':'Short same-MPS cache-memory diagnostic; no efficacy or full-video claim.',
            'source_snapshot_manifest_sha256':snapshot_sha,
            'certificate_path':str(a.certificate.resolve()),'certificate_sha256':cert_sha,'source_sha256':source_map,
            'frames_staged':7,'cache_frames':a.cache_frames,'min_disk_bytes':a.min_disk_gib*GIB,
            'min_available_ram_bytes':a.min_available_ram_gib*GIB,'max_process_rss_bytes':a.max_process_rss_gib*GIB,
            'psutil_version':psutil.__version__,'phases':['observe','clear_idle_cache','audit']}
        manifest_file = root/'run_manifest.json'; manifest_file.write_text(json.dumps(manifest,indent=2)+'\n')
        manifest_sha = digest(manifest_file)
        for phase in manifest['phases']:
            require(digest(manifest_file) == manifest_sha and sources() == source_map and
                    verify_certificate(a.certificate)[1] == cert_sha and guard['open_attempts'] == 0,
                    'Frozen diagnostic inputs changed')
            out = root/phase
            if phase == 'audit':
                command = [sys.executable,str(ROOT/'scripts/audit_mps_memory_diagnostic_v1.py'),
                           '--root',str(root),'--output',str(out)]; artifact = out/'audit.json'
            else:
                command = [sys.executable,str(ROOT/'scripts/run_mps_memory_diagnostic_v1.py'),
                    '--mode',phase,'--certificate',str(a.certificate.resolve()),'--frames','7',
                    '--cache-frames',str(a.cache_frames),'--min-available-ram-bytes',str(a.min_available_ram_gib*GIB),
                    '--max-process-rss-bytes',str(a.max_process_rss_gib*GIB),'--output',str(out)]; artifact = out/'receipt.json'
            started = time.monotonic()
            print(json.dumps({'phase':phase,'status':'starting','output':str(root)}),flush=True)
            try:
                monitor = run_monitored(command,root/f'{phase}.log',root,a.min_disk_gib*GIB,
                                        a.min_available_ram_gib*GIB,a.max_process_rss_gib*GIB)
                require(artifact.is_file() and sources() == source_map, 'Diagnostic artifact or source differs')
                if phase in ('observe', 'clear_idle_cache'):
                    receipt = json.loads(artifact.read_text())
                    require(receipt.get('passed') is True and type(receipt.get('peak_process_rss_bytes')) is int and
                            receipt['peak_process_rss_bytes'] <= a.max_process_rss_gib*GIB,
                            'Diagnostic process high-water RSS exceeded its configured cap')
            except BaseException as error:
                (root/'failure.json').write_text(json.dumps({'phase':phase,'error':str(error),
                    'scope':'Partial outputs and logs retained.'},indent=2)+'\n'); raise
            with (root/'phases.jsonl').open('a') as log:
                log.write(json.dumps({'phase':phase,'passed':True,'artifact_path':str(artifact),
                    'artifact_sha256':digest(artifact),'log_sha256':digest(root/f'{phase}.log'),
                    'wall_seconds':time.monotonic()-started,'resource_monitor':monitor})+'\n')
            print(json.dumps({'phase':phase,'status':'completed','resource_monitor':monitor}),flush=True)
    print(json.dumps({'passed':True,'output':str(root)}))


if __name__ == '__main__': main()
