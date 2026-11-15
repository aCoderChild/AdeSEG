#!/usr/bin/env python3
"""Guard a separately versioned bounded seq6/seq11 warm full capture."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import DATA, OUT, PER_READ, validate_plan
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import exclusive_lock
from scripts.run_bounded_loader_gate_study_v2 import GIB, resources
from scripts.run_boundary_video_factorial_v1 import phase_environment
from scripts.run_bounded_warm_capture_v1 import sources
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_bounded_loader_gate_v2 import verify_gate
from scripts.warm_crossfit_video_contract_v1 import read_bound, require

ENABLE_REVIEWED_EXECUTION = True  # Reviewed for the seq6 compatibility attempt only.


def snapshot_sources(destination, project_root, source_map):
    """Retain the exact code bytes used by a bounded full-capture attempt."""
    destination=Path(destination);project_root=Path(project_root).resolve(strict=True)
    destination.mkdir(parents=True,exist_ok=False)
    for name,sha in source_map.items():
        source=(project_root/name).resolve(strict=True)
        require(source.is_relative_to(project_root) and digest(source)==sha,'Bounded source changed before snapshot')
        output=destination/name;output.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,output)
        require(digest(output)==sha and digest(source)==sha,'Bounded source changed during snapshot')
    manifest=destination/'snapshot_manifest.json';manifest.write_text(json.dumps({'source_sha256':source_map},indent=2)+'\n')
    return digest(manifest)


def tensor_bytes(folder):
    folder=Path(folder)
    return sum(p.stat().st_size for name in ('features','snapshots')
               for p in (folder/name).glob('*.pt') if p.is_file())


def required_disk(base, budget, written):
    require(type(base) is int and type(budget) is int and type(written) is int and
            base>=26*GIB and budget>=0 and written>=0,'Invalid disk reservation')
    return base + max(0,budget-min(written,budget))


def run_monitored(command,log,root,capture_folder,min_disk,min_ram,max_rss,budget):
    minimum_disk=None;minimum_ram=None;minimum_margin=None;peak=0
    def sample(process=None):
        nonlocal minimum_disk,minimum_ram,minimum_margin,peak
        written=tensor_bytes(capture_folder);need=required_disk(min_disk,budget,written)
        disk=shutil.disk_usage(root).free;ram=psutil.virtual_memory().available
        rss=0
        if process is not None:
            try:rss=process.memory_info().rss
            except psutil.NoSuchProcess:pass
        minimum_disk=disk if minimum_disk is None else min(minimum_disk,disk)
        minimum_ram=ram if minimum_ram is None else min(minimum_ram,ram)
        margin=disk-need;minimum_margin=margin if minimum_margin is None else min(minimum_margin,margin)
        peak=max(peak,rss)
        require(disk>=need and ram>=min_ram and rss<=max_rss,
                f'Bounded capture resources below guard: disk={disk}, required={need}, RAM={ram}, RSS={rss}')
    sample()
    with Path(log).open('x') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=phase_environment(),stdout=stream,stderr=subprocess.STDOUT)
        process=psutil.Process(child.pid)
        try:
            while True:
                try: code=child.wait(timeout=.5)
                except subprocess.TimeoutExpired: sample(process)
                else:
                    require(code==0, f'Capture phase failed: {log}')
                    sample()
                    return {'peak_sampled_process_rss_bytes':peak,'minimum_sampled_disk_free_bytes':minimum_disk,
                        'minimum_sampled_system_available_ram_bytes':minimum_ram,
                        'minimum_sampled_payload_margin_bytes':minimum_margin,'planned_payload_bytes':budget,
                        'written_payload_bytes':min(tensor_bytes(capture_folder),budget)}
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            raise


def phases(root, sequence, certificate, short_gate, cache_frames):
    root=Path(root); worker=ROOT/'scripts/run_bounded_warm_capture_v1.py'
    tasks=[]
    for mode in ('reference','capture'):
        out=root/sequence/mode
        command=[sys.executable,str(worker),'--sequence',sequence,'--mode',mode,'--output',str(out),
            '--certificate',str(certificate),'--short-gate',str(short_gate),'--batch-root',str(root),
            '--cache-frames',str(cache_frames)]
        if mode=='capture': command+=['--reference',str(root/sequence/'reference')]
        tasks.append((f'{sequence}_{mode}',command,out/'receipt.json'))
    out=root/sequence/'audit'
    tasks.append((f'{sequence}_audit',[sys.executable,str(ROOT/'scripts/audit_bounded_warm_capture_v1.py'),
        '--root',str(root),'--sequence',sequence,'--certificate',str(certificate),'--short-gate',str(short_gate),
        '--output',str(out)],out/'audit.json'))
    return tasks


def verify_seq6_proof(path, certificate_sha, gate_sha):
    from scripts.audit_bounded_warm_capture_v1 import audit
    from scripts.crossfit_observer_capture_shared_v1 import phase_artifact
    path=Path(path).resolve(strict=True)
    require(path.is_relative_to(OUT.resolve(strict=True)) and path.name=='audit.json' and
            path.parent.parent.name=='seq6', 'Invalid seq6 proof')
    root=path.parent.parent.parent;sha=phase_artifact(root,'seq6_audit',path);saved=read_bound(path,sha)
    require(saved.get('passed') is True and saved.get('sequence')=='seq6' and
            saved.get('certificate_sha256')==certificate_sha and saved.get('short_gate_sha256')==gate_sha and
            saved.get('label_guard',{}).get('open_attempts')==0,'Seq6 proof differs')
    # Reconstruct with the exact paths bound in the originating manifest.
    manifest_path=root/'run_manifest.json';manifest_sha=digest(manifest_path);manifest=read_bound(manifest_path,manifest_sha)
    rebuilt=audit(root,'seq6',Path(manifest['certificate_path']),Path(manifest['short_gate_path']))
    require({k:v for k,v in saved.items() if k!='label_guard'}==rebuilt and digest(path)==sha and
            digest(manifest_path)==manifest_sha, 'Seq6 proof reconstruction differs')
    return saved,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sequence',choices=('seq6','seq11'),required=True)
    p.add_argument('--certificate',type=Path,required=True);p.add_argument('--short-gate',type=Path,required=True)
    p.add_argument('--seq6-proof',type=Path);p.add_argument('--cache-frames',type=int,default=4)
    p.add_argument('--min-disk-gib',type=int,default=26);p.add_argument('--min-available-ram-gib',type=int,default=4)
    p.add_argument('--max-process-rss-gib',type=int,default=6);p.add_argument('--execute',action='store_true');a=p.parse_args()
    require((a.sequence=='seq11')==(a.seq6_proof is not None) and 1<=a.cache_frames<=4 and
            a.min_disk_gib>=26 and a.min_available_ram_gib>=4 and 1<=a.max_process_rss_gib<=6,'Invalid resource or proof configuration')
    guard=install_mask_open_guard(DATA);cert,cert_sha=verify_certificate(a.certificate);gate,gate_sha=verify_gate(a.short_gate)
    require(gate['certificate_sha256']==cert_sha,'Short gate certificate differs')
    seq6_sha=None
    if a.seq6_proof: _,seq6_sha=verify_seq6_proof(a.seq6_proof,cert_sha,gate_sha)
    source_map=sources();disk,ram=resources(OUT,a.min_disk_gib*GIB,a.min_available_ram_gib*GIB)
    index=validate_plan();reads=sum(x['sequence']==a.sequence for x in index['fresh_reads'])
    require(reads>0 and disk>=required_disk(a.min_disk_gib*GIB,reads*PER_READ,0),
            'Insufficient disk for the current sequence payload')
    require(guard['open_attempts']==0,'Annotation opened before launch')
    if not a.execute:
        print(json.dumps({'passed':True,'scope':'CPU-only dry run; no full capture.','sequence':a.sequence,
            'disk_free_bytes':disk,'system_available_ram_bytes':ram,'seq6_proof_sha256':seq6_sha}));return
    require(ENABLE_REVIEWED_EXECUTION, 'Full bounded capture execution awaits independent review')
    require(a.sequence=='seq6','Only reviewed seq6 execution is enabled')
    import torch
    require(torch.backends.mps.is_available(),'MPS unavailable')
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        live,live_sha=verify_certificate(a.certificate); live_gate,live_gate_sha=verify_gate(a.short_gate)
        require(live==cert and live_sha==cert_sha and live_gate==gate and live_gate_sha==gate_sha and
                sources()==source_map,'Bounded capture preflight changed')
        if a.seq6_proof:
            require(verify_seq6_proof(a.seq6_proof,cert_sha,gate_sha)[1]==seq6_sha,'Seq6 proof changed')
        root=Path(tempfile.mkdtemp(prefix='bounded_warm_full_v1_',dir=OUT))
        tasks=phases(root,a.sequence,a.certificate.resolve(),a.short_gate.resolve(),a.cache_frames)
        author_files=('external/Lean-SAM2-token-window/sam2/sam2_video_predictor_swr_wrapper.py',
                      'external/Lean-SAM2-token-window/sam2/sam2_video_predictor_topk_insurance.py',
                      'external/Lean-SAM2-token-window/sam2/utils/misc.py')
        snapshot_map={**source_map,**{name:digest(ROOT/name) for name in author_files}}
        snapshot_sha=snapshot_sources(root/'source_snapshot',ROOT,snapshot_map)
        manifest={'scope':'Separate bounded full warm capture; no labels or efficacy.',
            'sequence':a.sequence,'certificate_path':str(a.certificate.resolve()),'certificate_sha256':cert_sha,
            'short_gate_path':str(a.short_gate.resolve()),'short_gate_sha256':gate_sha,
            'seq6_proof_path':str(a.seq6_proof.resolve()) if a.seq6_proof else None,'seq6_proof_sha256':seq6_sha,
            'cache_frames':a.cache_frames,'source_sha256':source_map,'phases':[x[0] for x in tasks],
            'min_disk_bytes':a.min_disk_gib*GIB,'min_available_ram_bytes':a.min_available_ram_gib*GIB,
            'max_process_rss_bytes':a.max_process_rss_gib*GIB,'psutil_version':psutil.__version__,
            'planned_payload_bytes':reads*PER_READ,'planned_capture_reads':reads,
            'source_snapshot_manifest_sha256':snapshot_sha}
        manifest_path=root/'run_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');manifest_sha=digest(manifest_path)
        for name,command,artifact in tasks:
            require(digest(manifest_path)==manifest_sha and sources()==source_map and
                    verify_certificate(a.certificate)[1]==cert_sha and verify_gate(a.short_gate)[1]==gate_sha and
                    guard['open_attempts']==0,'Bounded capture frozen inputs changed')
            started=time.monotonic(); print(json.dumps({'phase':name,'status':'starting','output':str(root)}),flush=True)
            try:
                monitor=run_monitored(command,root/f'{name}.log',root,root/a.sequence/'capture',a.min_disk_gib*GIB,
                    a.min_available_ram_gib*GIB,a.max_process_rss_gib*GIB,reads*PER_READ)
                require(artifact.is_file() and sources()==source_map,'Capture phase artifact or source differs')
                extra={}
                if not name.endswith('_audit'):
                    side=artifact.parent/'bounded_receipt.json';side_sha=digest(side);rec=read_bound(side,side_sha)
                    require(rec.get('passed') is True and rec.get('original_receipt_sha256')==digest(artifact) and
                            rec.get('source_sha256')==source_map and type(rec.get('peak_process_rss_bytes')) is int and
                            0<rec['peak_process_rss_bytes']<=a.max_process_rss_gib*GIB,
                            'Capture sidecar or high-water RSS differs')
                    extra={'bounded_receipt_sha256':side_sha}
            except BaseException as error:
                (root/'failure.json').write_text(json.dumps({'phase':name,'error':str(error),
                    'scope':'Partial outputs and logs retained.'},indent=2)+'\n');raise
            with (root/'phases.jsonl').open('a') as log:
                log.write(json.dumps({'phase':name,'passed':True,'artifact_path':str(artifact),
                    'artifact_sha256':digest(artifact),'log_sha256':digest(root/f'{name}.log'),
                    'wall_seconds':time.monotonic()-started,'resource_monitor':monitor,**extra})+'\n')
            print(json.dumps({'phase':name,'status':'completed','resource_monitor':monitor}),flush=True)
    print(json.dumps({'passed':True,'output':str(root)}))


if __name__=='__main__':main()
