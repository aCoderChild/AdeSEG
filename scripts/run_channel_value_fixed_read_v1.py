#!/usr/bin/env python3
"""Queue-only-by-default launcher for the nine-arm fixed-read development study."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import OUT, exclusive_lock
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v2 import load_prepared
from scripts.channel_value_fixed_read_v1 import ARMS, deps, build_plan

MANIFEST=OUT/'source_group_boundary_factorial_v1_bvqxd475/manifest'
SCRIPT=ROOT/'scripts/channel_value_fixed_read_v1.py'
GUARD=22*1024**3


def available(path):
    free=shutil.disk_usage(path).free
    if free<GUARD:raise RuntimeError('Insufficient free space for another phase')
    return free


def phases(folder):
    folder=Path(folder);py=sys.executable
    def cmd(mode,output,extra=()):
        return [py,str(SCRIPT),'--mode',mode,'--manifest',str(MANIFEST),'--output',str(folder/output),*extra]
    plan=folder/'plan/plan.json';parity=folder/'parity/parity.json'
    common=['--plan',str(plan),'--parity',str(parity)]
    result=[('plan',cmd('plan','plan')),('parity',cmd('parity','parity',['--plan',str(plan)]))]
    result += [('train_'+a,cmd('train','runs/'+a,[*common,'--arm',a])) for a in ARMS]
    result += [('replay',cmd('replay','verification',[*common,'--run',str(folder)])),
               ('score',cmd('score','scores',[*common,'--run',str(folder),
                '--verification',str(folder/'verification/verification.json')])),
               ('audit',cmd('audit','independent_audit',[*common,'--run',str(folder),
                '--verification',str(folder/'verification/verification.json')]))]
    return result


def monitor(folder,stop,ready,errors):
    try:
        with (folder/'free_space.jsonl').open('x') as out:
            while True:
                free=shutil.disk_usage(folder).free
                out.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'free_bytes':free,
                                      'below_guard':free<GUARD})+'\n');out.flush();ready.set()
                if stop.wait(5):break
    except Exception as exc:errors.append(repr(exc));ready.set()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');a=p.parse_args()
    guard=install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl')
    manifest,sha=load_prepared(MANIFEST);plan=build_plan(manifest,sha)
    if guard['open_attempts']:raise RuntimeError('Launcher attempted label access')
    free=available(OUT)
    if not a.execute:
        print(json.dumps({'ready':True,'manifest_sha256':sha,'arms':list(ARMS),
                          'parameter_counts':plan['parameter_count'],'free_bytes':free,
                          'phases':[x[0] for x in phases(OUT/'DRY_RUN')]}));return
    import torch
    if not torch.backends.mps.is_available():raise RuntimeError('MPS is unavailable')
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        manifest,again=load_prepared(MANIFEST)
        if again!=sha or guard['open_attempts']:raise RuntimeError('Manifest or label guard changed')
        available(OUT);folder=Path(tempfile.mkdtemp(prefix='channel_value_fixed_read_v1_',dir=OUT))
        source_hashes=deps();config=folder/'run_manifest.json'
        config.write_text(json.dumps({'scope':'Reused development fixed reads; not a video result.',
            'manifest_path':str(MANIFEST),'manifest_sha256':sha,'dependencies':source_hashes,
            'arms':list(ARMS),'phases':[n for n,_ in phases(folder)],'mps_fallback':'0',
            'mps_fast_math':'0','python_hash_seed':'0','launcher_label_guard':dict(guard)},indent=2)+'\n')
        frozen=digest(config);stop=threading.Event();ready=threading.Event();errors=[]
        thread=threading.Thread(target=monitor,args=(folder,stop,ready,errors),daemon=True);thread.start();ready.wait(10)
        env=os.environ.copy();env.update(PYTHONHASHSEED='0',PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0')
        try:
            if not ready.is_set() or errors:raise RuntimeError('Storage monitor failed')
            for name,command in phases(folder):
                if digest(config)!=frozen or deps()!=source_hashes or digest(MANIFEST/'manifest.json')!=sha or errors or guard['open_attempts']:
                    raise RuntimeError('Frozen launcher inputs or monitor changed')
                free=available(folder);start=time.monotonic()
                print(json.dumps({'phase':name,'status':'starting','output':str(folder),'free_bytes':free}),flush=True)
                with (folder/(name+'.log')).open('x') as log:
                    subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if errors:raise RuntimeError('Storage monitor failed')
                receipt=None
                if name.startswith('train_'):receipt=digest(folder/'runs'/name[6:]/'completion.json')
                verification=digest(folder/'verification/verification.json') if name=='replay' else None
                artifact=(digest(folder/'plan/plan.json') if name=='plan' else
                          digest(folder/'parity/parity.json') if name=='parity' else
                          digest(folder/'scores/scores.json') if name=='score' else None)
                with (folder/'phases.jsonl').open('a') as log:
                    log.write(json.dumps({'phase':name,'passed':True,'wall_seconds':time.monotonic()-start,
                        'log_sha256':digest(folder/(name+'.log')),'completion_sha256':receipt,
                        'verification_sha256':verification,'artifact_sha256':artifact,'free_bytes_before':free})+'\n')
                print(json.dumps({'phase':name,'status':'completed'}),flush=True)
        finally:
            stop.set();thread.join(10)
            if thread.is_alive() or errors:raise RuntimeError(f'Storage monitor failed: {errors}')
    print(json.dumps({'passed':True,'output':str(folder)}),flush=True)


if __name__=='__main__':main()
