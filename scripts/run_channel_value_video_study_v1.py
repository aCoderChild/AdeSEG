#!/usr/bin/env python3
"""MPS-only asinh video batch; default is a no-GPU dry run.

The first invocation certifies historical inputs and runs five MPS smokes.
An explicit resume rechecks that certificate and smoke before twelve B/C/D
videos. All outputs freeze before scoring opens labels.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_value_window_runtime_v1 import BASE,runtime_sources,require
from scripts.crossfit_observer_capture_shared_v1 import DATA,sequence_inputs,validate_plan
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import OUT,exclusive_lock
from scripts.run_boundary_video_factorial_v1 import CHECKPOINT,phase_environment
from scripts.source_group_label_guard_v1 import install_mask_open_guard

SEQUENCES=('seq10','seq11','seq20','seq23')
GUARD=22*1024**3


def source_hashes():
    paths=('scripts/run_channel_value_video_study_v1.py','scripts/score_channel_value_video_v1.py',
           'scripts/score_boundary_video_factorial_v1.py','scripts/score_source_group_boundary_v1.py',
           'scripts/audit_boundary_video_factorial_v1.py','scripts/source_group_label_guard_v1.py',
           'scripts/source_group_training_label_guard_v1.py','scripts/channel_value_fixed_read_v1.py',
           'scripts/resume_source_group_capture_v1.py','scripts/run_boundary_video_factorial_v1.py',
           'protocols/CHANNEL_VALUE_ASINH_VIDEO_V1.md')
    return {'batch':{p:digest(ROOT/p) for p in paths},'runtime':runtime_sources()}


def available(path):
    free=shutil.disk_usage(path).free
    require(free>=GUARD,f'Free disk {free} is below reserve {GUARD}')
    return free


def phases(root):
    root=Path(root);py=sys.executable;cert=root/'certificate/certificate.json'
    cert_sha=digest(cert) if cert.exists() else '0'*64
    runner=ROOT/'scripts/run_channel_value_video_v1.py';scorer=ROOT/'scripts/score_channel_value_video_v1.py'
    common=['--batch-root',str(root),'--certificate',str(cert),'--certificate-sha256',cert_sha,
            '--checkpoint',str(CHECKPOINT)]
    result=[('certificate',[py,str(ROOT/'scripts/verify_channel_value_video_study_v1.py'),
                            '--output',str(root/'certificate')])]
    for seq,arm in [('seq20','A'),('seq20','B'),('seq20','C'),('seq20','D'),('seq23','A')]:
        result.append((f'smoke_{seq}_{arm}',[py,str(runner),*common,'--sequence',seq,'--arm',arm,
                       '--limit','4','--output',str(root/'smoke'/seq/arm)]))
    audit=root/'smoke_audit/audit.json'
    result.append(('smoke_audit',[py,str(ROOT/'scripts/verify_channel_value_video_smoke_v1.py'),
        '--batch-root',str(root),'--certificate-sha256',cert_sha,'--output',str(root/'smoke_audit')]))
    for seq in SEQUENCES:
        for arm in ('B','C','D'):
            result.append((f'full_{seq}_{arm}',[py,str(runner),*common,'--sequence',seq,'--arm',arm,
                '--limit','0','--smoke-audit',str(audit),'--output',str(root/'full'/seq/arm)]))
    for name,dest in (('verify','verification'),('score','scores'),('audit','independent_audit')):
        result.append((name,[py,str(scorer),'--mode',name,'--batch-root',str(root),'--output',str(root/dest)]))
    return result


def artifact(root,name):
    if name=='certificate':return root/'certificate/certificate.json'
    if name=='smoke_audit':return root/'smoke_audit/audit.json'
    if name in ('verify','score','audit'):
        return root/({'verify':'verification/verification.json','score':'scores/results.json',
                      'audit':'independent_audit/audit.json'}[name])
    category,seq,arm=name.split('_',2)
    return root/category/seq/arm/'receipt.json'


def monitor(root,stop,ready,errors,monitor_path):
    try:
        with monitor_path.open('x') as out:
            while True:
                free=shutil.disk_usage(root).free
                out.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'free_bytes':free,
                                      'below_guard':free<GUARD})+'\n');out.flush();ready.set()
                if stop.wait(5):break
    except Exception as exc:errors.append(repr(exc));ready.set()


def run_phase(command,log,root,errors):
    """Stop a phase if shared free space falls below the reserved floor."""
    with log.open('x') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=phase_environment(),stdout=stream,stderr=subprocess.STDOUT)
        while True:
            try:
                code=process.wait(timeout=2);break
            except subprocess.TimeoutExpired:
                if errors or shutil.disk_usage(root).free<GUARD:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise RuntimeError('Storage monitor failed or reserve crossed during phase; partial outputs retained')
        require(code==0,f'Phase failed with exit code {code}; inspect {log}')
        require(not errors and shutil.disk_usage(root).free>=GUARD,
                'Storage monitor failed or reserve crossed at phase completion')


def validate_resume(root,frozen_sources):
    from scripts.crossfit_observer_capture_shared_v1 import phase_artifact
    from scripts.verify_channel_value_video_smoke_v1 import verify_smoke_audit
    from modeling.lean_channel_value_window_runtime_v1 import verify_certificate
    root=Path(root).resolve(strict=True);require(root.is_relative_to(OUT.resolve(strict=True)), 'Resume outside batch root')
    manifest_path=root/'run_manifest.json';manifest=json.loads(manifest_path.read_text())
    require(manifest.get('sources')==frozen_sources and manifest.get('runtime_sources')==runtime_sources() and
            manifest.get('phases')==[name for name,_ in phases(root)] and manifest.get('label_guard',{}).get('open_attempts')==0,
            'Frozen short-batch configuration differs')
    rows=[json.loads(x) for x in (root/'phases.jsonl').read_text().splitlines() if x.strip()]
    names=[name for name,_ in phases(root)]
    require(7<=len(rows)<len(names) and [x.get('phase') for x in rows]==names[:len(rows)],
            'Resume requires a complete smoke and an incomplete valid phase prefix')
    for row in rows:
        require(phase_artifact(root,row['phase'],artifact(root,row['phase']))==row.get('artifact_sha256'),
                'Previously completed phase changed')
    cert=root/'certificate/certificate.json';cert_sha=phase_artifact(root,'certificate',cert)
    verify_certificate(cert,cert_sha)
    verify_smoke_audit(root/'smoke_audit/audit.json',root,cert_sha,reconstruct=True)
    return root,digest(manifest_path),len(rows)


def preserve_partial(root,name):
    """Move an incomplete next phase aside without deleting research artifacts."""
    folder=artifact(root,name).parent;log=root/(name+'.log')
    if not folder.exists() and not log.exists():return None
    destination=root/'partial_attempts'/f'{name}_{uuid.uuid4().hex}'
    destination.mkdir(parents=True,exist_ok=False)
    if folder.exists():folder.rename(destination/'output')
    if log.exists():log.rename(destination/'phase.log')
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',type=Path,help='Resume a completed reviewed smoke batch for full videos');a=p.parse_args()
    guard=install_mask_open_guard(DATA);plan=validate_plan()
    for seq in SEQUENCES:sequence_inputs(plan,seq)
    require(guard['open_attempts']==0,'Label opened during batch preflight')
    frozen_sources=source_hashes();free=available(OUT)
    if a.resume:validate_resume(a.resume,frozen_sources)
    if not a.execute:
        print(json.dumps({'preflight_passed':True,'scope':'No GPU inference or labels opened; the full historical certificate has not run.',
            'free_bytes':free,'stage':'full' if a.resume else 'short',
            'phases':[x[0] for x in phases(OUT/'DRY_RUN')][7:] if a.resume else
                     [x[0] for x in phases(OUT/'DRY_RUN')][:7]}));return
    import torch
    require(torch.backends.mps.is_available(),'MPS unavailable')
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        require(source_hashes()==frozen_sources and validate_plan()['passed'] is True,'Preflight source changed')
        available(OUT)
        if a.resume:
            root,frozen,completed=validate_resume(a.resume,frozen_sources);indices=range(completed,len(phases(root)))
            preserve_partial(root,phases(root)[completed][0])
        else:
            root=Path(tempfile.mkdtemp(prefix='channel_value_asinh_video_v1_',dir=OUT));manifest=root/'run_manifest.json'
            manifest.write_text(json.dumps({'scope':'Exploratory previously inspected four-video test.',
                'sequences':list(SEQUENCES),'full_arms':['B','C','D'],'runtime_sources':runtime_sources(),
                'sources':frozen_sources,'phases':[name for name,_ in phases(root)],
                'python_hash_seed':'0','mps_fallback':'0','mps_fast_math':'0','label_guard':dict(guard)},indent=2)+'\n')
            frozen=digest(manifest);indices=range(7)
        manifest=root/'run_manifest.json';stop=threading.Event();ready=threading.Event();errors=[]
        monitor_path=root/(f'free_space_resume_{uuid.uuid4().hex}.jsonl' if a.resume else 'free_space_short.jsonl')
        thread=threading.Thread(target=monitor,args=(root,stop,ready,errors,monitor_path),daemon=True);thread.start();ready.wait(10)
        try:
            require(ready.is_set() and not errors,'Storage monitor failed')
            for index in indices:
                name,command=phases(root)[index]
                require(digest(manifest)==frozen and source_hashes()==frozen_sources and
                        guard['open_attempts']==0 and not errors,'Frozen sources or monitor changed')
                free=available(root);started=time.monotonic()
                print(json.dumps({'phase':name,'status':'starting','output':str(root),'free_bytes':free}),flush=True)
                run_phase(command,root/(name+'.log'),root,errors)
                require(not errors,'Storage monitor failed')
                file=artifact(root,name);require(file.is_file(),f'Missing phase artifact: {name}')
                with (root/'phases.jsonl').open('a') as log:
                    log.write(json.dumps({'phase':name,'passed':True,'artifact_path':str(file),
                        'artifact_sha256':digest(file),'log_sha256':digest(root/(name+'.log')),
                        'wall_seconds':time.monotonic()-started,'free_bytes_before':free})+'\n')
                print(json.dumps({'phase':name,'status':'completed'}),flush=True)
        finally:
            stop.set();thread.join(10)
            if thread.is_alive() or errors:raise RuntimeError(f'Storage monitor failed: {errors}')
    print(json.dumps({'passed':True,'stage':'full' if a.resume else 'short','output':str(root)}),flush=True)


if __name__=='__main__':main()
