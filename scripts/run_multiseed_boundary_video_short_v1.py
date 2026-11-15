#!/usr/bin/env python3
"""Run two seeded, label-free short routing checks before video inference.

Each seed gets fresh A/A-repeat/B/C/D processes on seq4 and cropped seq23,
then a source-bound gate and an independent CPU artifact audit. These are
engineering checks, not efficacy or selection tests.
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
from scripts.resume_source_group_capture_v1 import OUT, REF, exclusive_lock
from scripts.run_boundary_video_factorial_v1 import (BASELINE, CHECKPOINT, DETECTOR,
    DETECTOR_SHA256, PRE, SEQ23, SEQ23_BACKUP, free_bytes, monitor_space, phase_environment)

STUDY=OUT/'source_group_boundary_multiseed_v1_4j22w0xh'
SEQ8_PRE=OUT/'extended_boundary_video_preflight_attempt01'
SEEDS=(20261125,20261135)
SEQUENCES=('seq4','seq23')
ARMS=('A','A_repeat','B','C','D')
RUNNER=ROOT/'scripts/run_multiseed_boundary_refiner_v1.py'
VERIFIER=ROOT/'scripts/verify_multiseed_boundary_video_short_gate_v1.py'
AUDITOR=ROOT/'scripts/audit_multiseed_boundary_video_short_v1.py'


def commands(run):
    run=Path(run); phases=[]; mapping={}
    for seed in SEEDS:
        mapping[str(seed)]={}
        common=['--multiseed-root',str(STUDY),'--seed',str(seed),'--checkpoint',str(CHECKPOINT),
                '--detector',str(DETECTOR),'--split-preflight',str(PRE),'--baseline-reference',str(BASELINE),
                '--seq8-preflight',str(SEQ8_PRE),
                '--seq4-capture',str(REF),'--seq23-reference',str(SEQ23),'--seq23-backup',str(SEQ23_BACKUP)]
        for seq in SEQUENCES:
            mapping[str(seed)][seq]={}
            for arm in ARMS:
                folder=run/str(seed)/seq/arm; mapping[str(seed)][seq][arm]=str(folder)
                cmd=[sys.executable,str(RUNNER),*common,'--sequence',seq,
                     '--arm','A' if arm=='A_repeat' else arm,'--limit','4','--output',str(folder)]
                if arm in ('B','C','D'): cmd += ['--matched-a',str(run/str(seed)/seq/'A')]
                phases.append((f'{seed}_{seq}_{arm}',seed,cmd))
        phases.append((f'{seed}_gate',seed,[sys.executable,str(VERIFIER),'--mapping',str(run/f'{seed}_map.json'),
                     '--seed',str(seed),'--output',str(run/str(seed)/'gate')]))
        phases.append((f'{seed}_independent_audit',seed,[sys.executable,str(AUDITOR),
                     '--gate',str(run/str(seed)/'gate/verification.json'),
                     '--detector',str(DETECTOR),'--checkpoint',str(CHECKPOINT),'--seq8-preflight',str(SEQ8_PRE),
                     '--output',str(run/str(seed)/'independent_audit_attempt01')]))
    return phases,mapping


def source_paths():
    return {'launcher':Path(__file__),'runner':RUNNER,'gate':VERIFIER,'auditor':AUDITOR,
            'runtime':ROOT/'modeling/lean_boundary_multiseed_window_runtime_v1.py',
            'routing':ROOT/'modeling/lean_boundary_refinement_routing_v1.py',
            'seq8_preflight_source':ROOT/'scripts/preflight_extended_boundary_video_v1.py'}


def input_paths():
    result={'manifest':STUDY/'manifest/manifest.json','preflight':PRE/'manifest.json',
            'seq8_preflight':SEQ8_PRE/'manifest.json','seq8_preflight_completion':SEQ8_PRE/'completion.json',
            'baseline':BASELINE/'receipt.json','seq4_capture':REF/'channel_feature_receipt.json',
            'checkpoint':CHECKPOINT,'detector':DETECTOR}
    for seed in SEEDS:
        folder=STUDY/str(seed)
        for key, rel in (('plan','plan/plan.json'),('parity','parity/parity.json'),
                         ('verification','verification/verification.json'),('score','scores/scores.json'),
                         ('audit','independent_audit/audit.json'),
                         ('candidate','runs/region_boundary/checkpoint.pt'),
                         ('control','runs/mask_only_boundary/checkpoint.pt')):
            result[f'{seed}_{key}']=folder/rel
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args();sources=source_paths();inputs=input_paths()
    if digest(DETECTOR)!=DETECTOR_SHA256: raise ValueError('Archived detector differs')
    if not a.execute:
        phases,_=commands(OUT/'DRY_RUN')
        print(json.dumps({'ready':True,'phases':[p[0] for p in phases],
                          'source_sha256':{k:digest(v) for k,v in sources.items()},
                          'input_sha256':{k:digest(v) for k,v in inputs.items()},
                          'free_bytes':free_bytes(OUT)}));return
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        free_bytes(OUT)
        run=Path(tempfile.mkdtemp(prefix='multiseed_boundary_video_short_v1_',dir=OUT))
        phases,mapping=commands(run)
        manifest={'scope':'Two-seed real-checkpoint short routing only; no efficacy.',
                  'seeds':list(SEEDS),'sequences':list(SEQUENCES),'arms':list(ARMS),
                  'source_sha256':{k:digest(v) for k,v in sources.items()},
                  'input_sha256':{k:digest(v) for k,v in inputs.items()},
                  'python_hash_seed':'0','mps_fallback':'0','mps_fast_math':'0'}
        mp=run/'run_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
        for seed in SEEDS: (run/f'{seed}_map.json').write_text(json.dumps(mapping[str(seed)],indent=2)+'\n')
        frozen_manifest=digest(mp); frozen_maps={s:digest(run/f'{s}_map.json') for s in SEEDS}
        stop,ready,errors=threading.Event(),threading.Event(),[]
        monitor=threading.Thread(target=monitor_space,args=(run,stop,ready,errors),daemon=True)
        monitor.start();ready.wait(10)
        try:
            if not ready.is_set() or errors:raise RuntimeError(f'Storage monitor failed: {errors}')
            for name,seed,cmd in phases:
                if (digest(mp)!=frozen_manifest or any(digest(run/f'{s}_map.json')!=frozen_maps[s] for s in SEEDS)
                        or any(digest(path)!=manifest['source_sha256'][key] for key,path in sources.items())
                        or any(digest(path)!=manifest['input_sha256'][key] for key,path in inputs.items())):
                    raise RuntimeError('Frozen short-run inputs changed')
                if errors:raise RuntimeError(f'Storage monitor failed: {errors}')
                free=free_bytes(run); started=time.monotonic()
                print(json.dumps({'phase':name,'status':'starting','output':str(run),'free_bytes':free}),flush=True)
                env=phase_environment();env['ADESEG_BOUNDARY_MULTISEED']=str(seed)
                with (run/f'{name}.log').open('x') as log:
                    subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if errors:raise RuntimeError(f'Storage monitor failed: {errors}')
                with (run/'phases.jsonl').open('a') as log:
                    log.write(json.dumps({'phase':name,'seed':seed,'passed':True,
                        'wall_seconds':time.monotonic()-started,'free_bytes_before':free,
                        'log_sha256':digest(run/f'{name}.log')})+'\n')
                print(json.dumps({'phase':name,'status':'completed'}),flush=True)
        finally:
            stop.set();monitor.join(10)
            if monitor.is_alive() or errors:raise RuntimeError(f'Storage monitor did not close cleanly: {errors}')
    print(json.dumps({'passed':True,'output':str(run)}),flush=True)


if __name__=='__main__':main()
