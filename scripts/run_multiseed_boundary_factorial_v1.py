#!/usr/bin/env python3
"""Run two-seed exploratory A/B/C/D videos after audited short gates.

Each seed has its own matched A baseline and source-bound checkpoint. All 32
predictions must freeze before the scorer opens a target mask. These videos
have been inspected previously; this is not an independent or blind test.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import BASE,OUT,REF,exclusive_lock
from scripts.run_boundary_video_factorial_v1 import (BASELINE,CHECKPOINT,DETECTOR,DETECTOR_SHA256,
    PRE,SEQ23,SEQ23_BACKUP,free_bytes,monitor_space,phase_environment)
from scripts.run_multiseed_boundary_video_short_v1 import STUDY,SEQ8_PRE,SEEDS
from scripts.verify_multiseed_boundary_video_short_gate_v1 import verify_full_gate

SEQUENCES=('seq8','seq11','seq20','seq23')
ARMS=('A','B','C','D')
RUNNER=ROOT/'scripts/run_multiseed_boundary_refiner_v1.py'
SCORER=ROOT/'scripts/score_multiseed_boundary_factorial_v1.py'
AUDITOR=ROOT/'scripts/audit_multiseed_boundary_factorial_v1.py'
SHORT_AUDITOR=ROOT/'scripts/audit_multiseed_boundary_video_short_v1.py'


def read(path):return json.loads(Path(path).read_text())


def sources():
    return {'launcher':Path(__file__),'runner':RUNNER,'scorer':SCORER,'auditor':AUDITOR,
        'runtime':ROOT/'modeling/lean_boundary_multiseed_window_runtime_v1.py',
        'routing':ROOT/'modeling/lean_boundary_refinement_routing_v1.py',
        'short_verifier':ROOT/'scripts/verify_multiseed_boundary_video_short_gate_v1.py',
        'short_auditor':SHORT_AUDITOR,'seq8_preflight_source':ROOT/'scripts/preflight_extended_boundary_video_v1.py',
        'primary_metric':ROOT/'scripts/score_boundary_video_factorial_v1.py',
        'primary_boundary_metric':ROOT/'scripts/score_source_group_boundary_v1.py',
        'independent_metric':ROOT/'scripts/audit_boundary_video_factorial_v1.py',
        'mask_loader':ROOT/'datasets/io.py','video_loader':ROOT/'datasets/video.py'}


def verify_audit(seed,gate,audit):
    checked=read(audit); short=read(gate); candidate=STUDY/str(seed)/'runs/region_boundary/checkpoint.pt'
    if (checked.get('passed') is not True or checked.get('seed')!=seed or
            checked.get('gate_sha256')!=digest(gate) or checked.get('audit_source_sha256')!=digest(SHORT_AUDITOR) or
            checked.get('candidate_checkpoint_sha256')!=digest(candidate) or digest(DETECTOR)!=DETECTOR_SHA256):
        raise ValueError('Independent short-gate audit or detector differs')
    for seq in ('seq4','seq23'):
        for arm in ('A','A_repeat','B','C','D'):
            run=short['runs'][seq][arm];path=Path(run['folder'])/'completion.json';rec=read(path)
            if (run['completion_sha256']!=digest(path) or
                    checked['sequences'][seq][arm]['completion_sha256']!=digest(path) or
                    rec.get('detector_sha256')!=DETECTOR_SHA256 or rec.get('seed')!=seed):
                raise ValueError('Short-gate audit does not bind this seed/detector/run')
    return checked


def checked_gates(short_root):
    short_root=Path(short_root).resolve(strict=True)
    if not short_root.is_relative_to(BASE.resolve(strict=True)):raise ValueError('Gate outside experiment root')
    gates={}
    for seed in SEEDS:
        gate=short_root/str(seed)/'gate/verification.json'
        audit=short_root/str(seed)/'independent_audit_attempt01/audit.json'
        verify_full_gate(gate,multiseed_root=STUDY,seed=seed,checkpoint=CHECKPOINT,detector=DETECTOR,
            seq8_preflight=SEQ8_PRE,split_preflight=PRE,baseline_reference=BASELINE,seq4_capture=REF)
        verify_audit(seed,gate,audit);gates[seed]=(gate,audit)
    return gates


def commands(run,gates):
    run=Path(run);phases=[];mapping={}
    for seed in SEEDS:
        mapping[str(seed)]={};gate,audit=gates[seed]
        common=['--multiseed-root',str(STUDY),'--seed',str(seed),'--checkpoint',str(CHECKPOINT),
            '--detector',str(DETECTOR),'--split-preflight',str(PRE),'--seq8-preflight',str(SEQ8_PRE),
            '--baseline-reference',str(BASELINE),'--seq4-capture',str(REF),
            '--seq23-reference',str(SEQ23),'--seq23-backup',str(SEQ23_BACKUP),
            '--engineering-gate',str(gate),'--engineering-audit',str(audit)]
        for seq in SEQUENCES:
            mapping[str(seed)][seq]={}
            for arm in ARMS:
                folder=run/str(seed)/seq/arm;mapping[str(seed)][seq][arm]=str(folder)
                cmd=[sys.executable,str(RUNNER),*common,'--sequence',seq,'--arm',arm,'--output',str(folder)]
                if arm!='A':cmd+=['--matched-a',str(run/str(seed)/seq/'A')]
                phases.append((f'{seed}_{seq}_{arm}',seed,cmd))
    phases.append(('score',SEEDS[0],[sys.executable,str(SCORER),'--run',str(run),'--output',str(run/'scores')]))
    return phases,mapping


def inputs(short_root,gates):
    result={'short_root_manifest':Path(short_root)/'run_manifest.json','seq8_preflight':SEQ8_PRE/'manifest.json',
        'seq8_preflight_completion':SEQ8_PRE/'completion.json','split_preflight':PRE/'manifest.json',
        'baseline':BASELINE/'receipt.json','capture':REF/'channel_feature_receipt.json',
        'checkpoint':CHECKPOINT,'detector':DETECTOR,'multiseed_manifest':STUDY/'manifest/manifest.json',
        'seq23_prompt':SEQ23/'prompt.json','seq23_completion':SEQ23/'completion.json'}
    for seed in SEEDS:
        gate,audit=gates[seed];study=STUDY/str(seed)
        result[f'{seed}_gate']=gate;result[f'{seed}_audit']=audit
        result[f'{seed}_candidate']=study/'runs/region_boundary/checkpoint.pt'
        result[f'{seed}_control']=study/'runs/mask_only_boundary/checkpoint.pt'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--short-root',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    short_root=a.short_root.resolve(strict=True);gates=checked_gates(short_root);src=sources();inp=inputs(short_root,gates)
    if not a.execute:
        phases,_=commands(OUT/'DRY_RUN',gates)
        print(json.dumps({'ready':True,'phases':[n for n,_,_ in phases],
            'source_sha256':{k:digest(v) for k,v in src.items()},'input_sha256':{k:digest(v) for k,v in inp.items()},
            'free_bytes':free_bytes(OUT)}));return
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        gates=checked_gates(short_root);free_bytes(OUT)
        run=Path(tempfile.mkdtemp(prefix='multiseed_boundary_factorial_v1_',dir=OUT))
        phases,mapping=commands(run,gates);manifest=run/'run_manifest.json';map_path=run/'run_map.json'
        manifest.write_text(json.dumps({'scope':'Previously inspected patients; exploratory two-seed video replication.',
            'seeds':list(SEEDS),'sequences':list(SEQUENCES),'arms':list(ARMS),'run_map':mapping,
            'sources':{k:{'path':str(v),'sha256':digest(v)} for k,v in src.items()},
            'inputs':{k:{'path':str(v),'sha256':digest(v)} for k,v in inp.items()},
            'short_root':str(short_root),'python_hash_seed':'0','mps_fallback':'0','mps_fast_math':'0'},indent=2)+'\n')
        map_path.write_text(json.dumps(mapping,indent=2)+'\n');mf,mh=digest(manifest),digest(map_path)
        frozen_sources={k:digest(v) for k,v in src.items()};frozen_inputs={k:digest(v) for k,v in inp.items()}
        stop,ready,errors=threading.Event(),threading.Event(),[]
        monitor=threading.Thread(target=monitor_space,args=(run,stop,ready,errors),daemon=True);monitor.start();ready.wait(10)
        try:
            if not ready.is_set() or errors:raise RuntimeError(f'Storage monitor failed: {errors}')
            for name,seed,cmd in phases:
                if (digest(manifest)!=mf or digest(map_path)!=mh or
                        any(digest(v)!=frozen_sources[k] for k,v in src.items()) or
                        any(digest(v)!=frozen_inputs[k] for k,v in inp.items())):
                    raise RuntimeError('Frozen full-run sources or inputs changed')
                if errors:raise RuntimeError(f'Storage monitor failed: {errors}')
                free=free_bytes(run);started=time.monotonic();env=phase_environment();env['ADESEG_BOUNDARY_MULTISEED']=str(seed)
                print(json.dumps({'phase':name,'status':'starting','output':str(run),'free_bytes':free}),flush=True)
                with (run/f'{name}.log').open('x') as log:
                    subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if errors:raise RuntimeError(f'Storage monitor failed: {errors}')
                with (run/'phases.jsonl').open('a') as log:
                    log.write(json.dumps({'phase':name,'seed':seed,'passed':True,'wall_seconds':time.monotonic()-started,
                        'free_bytes_before':free,'log_sha256':digest(run/f'{name}.log')})+'\n')
                print(json.dumps({'phase':name,'status':'completed'}),flush=True)
        finally:
            stop.set();monitor.join(10)
            if monitor.is_alive() or errors:raise RuntimeError(f'Storage monitor did not close cleanly: {errors}')
    print(json.dumps({'passed':True,'output':str(run)}),flush=True)


if __name__=='__main__':main()
