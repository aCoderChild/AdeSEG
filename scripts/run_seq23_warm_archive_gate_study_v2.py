#!/usr/bin/env python3
"""Queue-only-by-default, MPS warm-compatibility gate with disk monitoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.resume_source_group_capture_v1 import exclusive_lock
from scripts.run_boundary_video_factorial_v1 import phase_environment
from scripts.crossfit_observer_capture_shared_v1 import BASE,DATA,GUARD,MARGIN,OUT,PER_READ,require,require_headroom
from scripts.run_seq23_warm_archive_gate_v2 import SOURCES,WARM_SHA,warm_plan
from scripts.audit_seq23_warm_archive_gate_v2 import check_partial


def phase_tasks(root):
    root=Path(root);tasks=[]
    for seq,mode,count in (('seq23','reference',0),('seq23','capture',1)):
        folder=root/seq/mode
        cmd=[sys.executable,str(ROOT/'scripts/run_seq23_warm_archive_gate_v2.py'),'--sequence',seq,'--mode',mode,'--output',str(folder)]
        tasks.append((f'{seq}_{mode}',cmd,folder,count))
    tasks.append(('audit',[sys.executable,str(ROOT/'scripts/audit_seq23_warm_archive_gate_v2.py'),'--root',str(root),
                           '--output',str(root/'audit')],None,0))
    return tasks


def output_bytes(folder):
    if folder is None:return 0
    return sum(p.stat().st_size for sub in ('features','snapshots') for p in (folder/sub).glob('*.pt') if p.is_file())


def run_phase(command,log,folder,count):
    import shutil
    start_bytes=output_bytes(folder);budget=count*PER_READ
    with log.open('x') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,env=phase_environment(),stdout=stream,stderr=subprocess.STDOUT)
        while True:
            try:code=proc.wait(timeout=2);break
            except subprocess.TimeoutExpired:
                written=min(max(0,output_bytes(folder)-start_bytes),budget)
                free=shutil.disk_usage(BASE).free;need=GUARD+MARGIN+max(0,budget-written)
                if free<need:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    raise RuntimeError(f'Storage headroom lost: {free} < {need}')
        if code!=0:raise RuntimeError(f'Gate subprocess exited {code}; inspect {log}')
        written=min(max(0,output_bytes(folder)-start_bytes),budget)
        free=shutil.disk_usage(BASE).free;need=GUARD+MARGIN+max(0,budget-written)
        require(free>=need,f'Storage headroom lost after child: {free} < {need}')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');a=parser.parse_args()
    guard=install_mask_open_guard(DATA);plan=warm_plan();check_partial();sources={p:digest(ROOT/p) for p in SOURCES}
    require(guard['open_attempts']==0,'Annotation open attempted')
    headroom=require_headroom(BASE,1)
    if not a.execute:
        print(json.dumps({'ready_for_requested_stage':True,'scope':'Dry run; no GPU inference or labels; per-phase space only.',
            'phases':[x[0] for x in phase_tasks(OUT/'DRY_RUN')],'headroom':headroom,'whole_study_capacity_reserved':False,
            'label_open_attempts':0}));return
    import torch
    require(torch.backends.mps.is_available(),'MPS unavailable')
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        require(warm_plan()==plan and sources=={p:digest(ROOT/p) for p in SOURCES},'Warm plan or source changed')
        require_headroom(BASE,1)
        root=Path(tempfile.mkdtemp(prefix='seq23_warm_archive_gate_v2_',dir=OUT));tasks=phase_tasks(root)
        manifest=root/'run_manifest.json';manifest.write_text(json.dumps({'warm_plan_sha256':WARM_SHA,
            'source_sha256':sources,'phases':[x[0] for x in tasks],'scope':'Seq23 uniform archived-mask warm compatibility only; no labels.',
            'reserve_bytes':GUARD,'margin_bytes':MARGIN,'bytes_per_read':PER_READ,
            'capacity_scope':'Per phase; full study not reserved.'},indent=2)+'\n');frozen=digest(manifest)
        for name,cmd,folder,count in tasks:
            require(digest(manifest)==frozen and warm_plan()==plan and sources=={p:digest(ROOT/p) for p in SOURCES},
                    'Warm plan or source changed')
            space=require_headroom(BASE,count);start=time.monotonic()
            print(json.dumps({'phase':name,'status':'starting','output':str(root),**space}),flush=True)
            run_phase(cmd,root/(name+'.log'),folder,count)
            require(digest(manifest)==frozen and warm_plan()==plan and sources=={p:digest(ROOT/p) for p in SOURCES},
                    'Warm plan or source changed during phase')
            artifact=folder/'receipt.json' if folder else root/'audit/audit.json'
            require(artifact.is_file(),'Missing gate artifact')
            with (root/'phases.jsonl').open('a') as log:
                log.write(json.dumps({'phase':name,'passed':True,'artifact_path':str(artifact),
                    'artifact_sha256':digest(artifact),'log_sha256':digest(root/(name+'.log')),
                    'wall_seconds':time.monotonic()-start})+'\n')
            print(json.dumps({'phase':name,'status':'completed','output':str(root)}),flush=True)
        require(guard['open_attempts']==0,'Annotation open attempted')
        print(json.dumps({'passed':True,'output':str(root),'label_open_attempts':0}))

if __name__=='__main__':main()
