#!/usr/bin/env python3
"""Run and independently audit two extended boundary-refiner short clips.

The ten MPS runs exercise a learned checkpoint, not efficacy. Original and
extension identities are frozen; no ground-truth mask is opened. Full videos
must separately require the resulting gate and independent audit.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.preflight_extended_boundary_video_v1 import load_extended_preflight
from scripts.resume_source_group_capture_v1 import BASE, OUT, REF, exclusive_lock
from scripts.run_boundary_video_factorial_v1 import (AUDIT, BASELINE, CHECKPOINT, DETECTOR, PRE, STUDY,
                                                     free_bytes, monitor_space, phase_environment, verify_gate_audit)
from scripts.verify_boundary_video_short_gate_v1 import verify_full_gate

RUNNER=ROOT/'scripts/run_extended_boundary_refiner_v1.py'
VERIFIER=ROOT/'scripts/verify_extended_boundary_video_short_gate_v1.py'
AUDITOR=ROOT/'scripts/audit_extended_boundary_video_short_v1.py'
SEQUENCES=('seq8','seq17')
ARMS=('A','A_repeat','B','C','D')


def commands(run,preflight,original_gate):
    """A then independent repeat then B/C/D; audit after all ten complete."""
    run=Path(run)
    common=['--study',str(STUDY),'--audit',str(AUDIT),'--checkpoint',str(CHECKPOINT),
            '--detector',str(DETECTOR),'--split-preflight',str(PRE),'--extended-preflight',str(preflight),
            '--baseline-reference',str(BASELINE),'--seq4-capture',str(REF),'--engineering-gate',str(original_gate)]
    phases=[];mapping={}
    for seq in SEQUENCES:
        mapping[seq]={}
        for arm in ARMS:
            folder=run/seq/arm; mapping[seq][arm]=str(folder)
            cmd=[sys.executable,str(RUNNER),*common,'--sequence',seq,'--arm','A' if arm=='A_repeat' else arm,
                 '--limit','4','--output',str(folder)]
            if arm in ('B','C','D'):cmd += ['--matched-a',str(run/seq/'A')]
            phases.append((f'{seq}_{arm}',cmd))
    gate=run/'gate/verification.json'
    phases.append(('gate',[sys.executable,str(VERIFIER),'--mapping',str(run/'run_map.json'),
                           '--extended-preflight',str(preflight),'--engineering-gate',str(original_gate),
                           '--output',str(run/'gate')]))
    phases.append(('independent_audit',[sys.executable,str(AUDITOR),'--gate',str(gate),
                           '--extended-preflight',str(preflight),'--engineering-gate',str(original_gate),
                           '--checkpoint',str(CHECKPOINT),'--detector',str(DETECTOR),
                           '--output',str(run/'independent_audit_attempt01')]))
    return phases,mapping


def check_inputs(preflight,gate,audit):
    if not all(Path(p).resolve().is_relative_to(BASE.resolve()) for p in (preflight,gate,audit)):
        raise ValueError('Extended preflight and short-gate inputs must be under the experiment root')
    load_extended_preflight(preflight,CHECKPOINT,DETECTOR)
    verify_full_gate(gate,study=STUDY,audit=AUDIT,checkpoint=CHECKPOINT,
                     split_preflight=PRE,baseline_reference=BASELINE,seq4_capture=REF)
    verify_gate_audit(gate,audit)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended-preflight',type=Path,required=True)
    parser.add_argument('--engineering-gate',type=Path,required=True)
    parser.add_argument('--engineering-audit',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    a=parser.parse_args()
    preflight,gate,audit=(p.resolve(strict=True) for p in (a.extended_preflight,a.engineering_gate,a.engineering_audit))
    check_inputs(preflight,gate,audit)
    if not a.execute:
        phases,_=commands(OUT/'DRY_RUN',preflight,gate)
        print(json.dumps({'ready':True,'phases':[x[0] for x in phases],'free_bytes':free_bytes(OUT)}));return
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        check_inputs(preflight,gate,audit);free_bytes(OUT)
        run=Path(tempfile.mkdtemp(prefix='extended_boundary_video_short_v1_',dir=OUT))
        phases,mapping=commands(run,preflight,gate)
        source_paths={'launcher':Path(__file__),'runner':RUNNER,'verifier':VERIFIER,'auditor':AUDITOR,
                      'preflight_source':ROOT/'scripts/preflight_extended_boundary_video_v1.py'}
        inputs={'preflight_manifest':preflight/'manifest.json','preflight_completion':preflight/'completion.json',
                'original_gate':gate,'original_audit':audit,'checkpoint':CHECKPOINT,'detector':DETECTOR}
        manifest={'scope':'Short learned-checkpoint engineering only; no labels or efficacy.',
                  'sequences':list(SEQUENCES),'arms':list(ARMS),'source_sha256':{k:digest(v) for k,v in source_paths.items()},
                  'input_sha256':{k:digest(v) for k,v in inputs.items()},'python_hash_seed':'0','mps_fallback':'0','mps_fast_math':'0'}
        (run/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (run/'run_map.json').write_text(json.dumps(mapping,indent=2)+'\n')
        frozen_manifest,frozen_map=digest(run/'run_manifest.json'),digest(run/'run_map.json')
        stop,ready,errors=threading.Event(),threading.Event(),[]
        monitor=threading.Thread(target=monitor_space,args=(run,stop,ready,errors),daemon=True)
        monitor.start();ready.wait(10)
        try:
            if not ready.is_set() or errors:raise RuntimeError(f'Storage monitor failed: {errors}')
            for name,cmd in phases:
                if (digest(run/'run_manifest.json') != frozen_manifest or digest(run/'run_map.json') != frozen_map or
                        any(digest(path) != manifest['source_sha256'][key] for key,path in source_paths.items()) or
                        any(digest(path) != manifest['input_sha256'][key] for key,path in inputs.items())):
                    raise RuntimeError('Frozen short-run inputs changed')
                if errors:raise RuntimeError(f'Storage monitor failed: {errors}')
                available=free_bytes(run);started=time.monotonic()
                print(json.dumps({'phase':name,'status':'starting','output':str(run),'free_bytes':available}),flush=True)
                with (run/f'{name}.log').open('x') as log:
                    subprocess.run(cmd,cwd=ROOT,env=phase_environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
                if errors:raise RuntimeError(f'Storage monitor failed: {errors}')
                with (run/'phases.jsonl').open('a') as log:
                    log.write(json.dumps({'phase':name,'passed':True,'wall_seconds':time.monotonic()-started,
                                          'free_bytes_before':available,'log_sha256':digest(run/f'{name}.log')})+'\n')
                print(json.dumps({'phase':name,'status':'completed'}),flush=True)
        finally:
            stop.set();monitor.join(10)
            if monitor.is_alive() or errors:raise RuntimeError(f'Storage monitor did not close cleanly: {errors}')
    print(json.dumps({'passed':True,'output':str(run)}),flush=True)


if __name__=='__main__':main()
