#!/usr/bin/env python3
"""Source-bound four-method warm-prompt study; predictions freeze before labels."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.local_import_closure_v1 import local_import_closure
from scripts.resume_source_group_capture_v1 import OUT,exclusive_lock
from scripts.run_additional_author_comparator_study_v2 import available,monitor,run_phase
from scripts.run_warm_author_comparator_v1 import DATA,METHODS,SEQUENCES,frozen_preflight,identity,input_data,source_hashes as runner_sources
from scripts.source_group_label_guard_v1 import install_mask_open_guard

RUNNER=ROOT/'scripts/run_warm_author_comparator_v1.py';VERIFIER=ROOT/'scripts/verify_warm_author_comparator_gate_v1.py'
CONTROL_AUDITOR=ROOT/'scripts/audit_warm_author_feature_access_v1.py'
SCORER=ROOT/'scripts/score_warm_author_study_v1.py';AUDITOR=ROOT/'scripts/audit_warm_author_study_v1.py'
SOURCES=(Path(__file__),SCORER,AUDITOR,ROOT/'scripts/run_additional_author_comparator_study_v2.py',
         ROOT/'scripts/resume_source_group_capture_v1.py',ROOT/'scripts/score_boundary_video_factorial_v1.py',
         ROOT/'scripts/score_source_group_boundary_v1.py',ROOT/'scripts/audit_boundary_video_factorial_v1.py',
         ROOT/'scripts/source_group_training_label_guard_v1.py',ROOT/'datasets/io.py',
         ROOT/'protocols/WARM_AUTHOR_STUDY_EXECUTION_V1.md')


def source_hashes():
    entries={str(p.relative_to(ROOT)) for p in SOURCES}|{'scripts/local_import_closure_v1.py'}
    entries|=set(local_import_closure(ROOT,entries))
    return {'study':{name:digest(ROOT/name) for name in sorted(entries)},'runner':runner_sources()}


def commands(folder):
    folder=Path(folder);py=sys.executable;phases=[]
    for method in METHODS:
        for seq in ('seq6','seq23'):
            phases.append((f'smoke_{method}_{seq}',[py,str(RUNNER),'--method',method,'--sequence',seq,
                '--limit','5','--output',str(folder/'smoke'/method/seq)]))
        phases.append((f'gate_{method}',[py,str(VERIFIER),'--method',method,
            '--seq6',str(folder/'smoke'/method/'seq6'),'--seq23',str(folder/'smoke'/method/'seq23'),
            '--output',str(folder/'gate'/method)]))
        phases.append((f'control_{method}',[py,str(RUNNER),'--method',method,'--sequence','seq6','--limit','5',
            '--control-no-callback','--output',str(folder/'control'/method)]))
        phases.append((f'featureaudit_{method}',[py,str(CONTROL_AUDITOR),'--method',method,
            '--gate',str(folder/'gate'/method/'gate.json'),'--control',str(folder/'control'/method),
            '--output',str(folder/'feature_audit'/method)]))
    for seq in SEQUENCES:
        for method in METHODS:
            phases.append((f'full_{seq}_{method}',[py,str(RUNNER),'--method',method,'--sequence',seq,
                '--gate',str(folder/'gate'/method/'gate.json'),'--output',str(folder/'full'/seq/method)]))
    phases += [('verify',[py,str(SCORER),'--mode','verify','--run',str(folder)]),
               ('score',[py,str(SCORER),'--mode','score','--run',str(folder)]),
               ('audit',[py,str(AUDITOR),'--run',str(folder)])]
    return phases


def artifact(folder,name):
    parts=name.split('_')
    if parts[0]=='smoke':return folder/'smoke'/parts[1]/parts[2]/'receipt.json'
    if parts[0]=='gate':return folder/'gate'/parts[1]/'gate.json'
    if parts[0]=='control':return folder/'control'/parts[1]/'receipt.json'
    if parts[0]=='featureaudit':return folder/'feature_audit'/parts[1]/'audit.json'
    if parts[0]=='full':return folder/'full'/parts[1]/parts[2]/'receipt.json'
    return folder/({'verify':'verification/verification.json','score':'scores/results.json',
                    'audit':'independent_audit/audit.json'}[name])


def validate_resume(folder,plan):
    folder=Path(folder).resolve(strict=True)
    if not folder.is_relative_to(OUT.resolve(strict=True)):raise ValueError('Batch outside experiment folder')
    phases=commands(folder);path=folder/'run_manifest.json';sha=digest(path);manifest=json.loads(path.read_text())
    if (digest(path)!=sha or manifest.get('sequences')!=list(SEQUENCES) or manifest.get('methods')!=list(METHODS) or
            manifest.get('phases')!=[n for n,_ in phases] or manifest.get('sources')!=source_hashes() or
            manifest.get('identities')!={m:identity(m,'seq6',plan) for m in METHODS} or
            manifest.get('label_guard',{}).get('open_attempts')!=0):raise ValueError('Frozen warm study differs')
    phase_file=folder/'phases.jsonl';rows=[json.loads(s) for s in phase_file.read_text().splitlines()] if phase_file.exists() else []
    if len(rows)>=len(phases) or [r.get('phase') for r in rows]!=[n for n,_ in phases[:len(rows)]]:
        raise ValueError('Completed phases are not an incomplete prefix')
    for row in rows:
        name=row['phase']
        if (row.get('passed') is not True or row.get('artifact_sha256')!=digest(artifact(folder,name)) or
                row.get('log_sha256')!=digest(folder/(name+'.log'))):raise ValueError('Completed phase changed')
    from scripts.verify_warm_author_comparator_gate_v1 import verify_gate
    from scripts.audit_warm_author_feature_access_v1 import audit as audit_feature
    for method in METHODS:
        if f'gate_{method}' in {r['phase'] for r in rows}:
            verify_gate(folder/'gate'/method/'gate.json',identity(method,'seq6',plan),plan)
        if f'featureaudit_{method}' in {r['phase'] for r in rows}:
            path=folder/'feature_audit'/method/'audit.json';saved=json.loads(path.read_text())
            current=audit_feature(method,folder/'gate'/method/'gate.json',folder/'control'/method,plan)
            if ({k:v for k,v in saved.items() if k!='label_guard'}!=current or
                    saved.get('label_guard',{}).get('open_attempts')!=0):raise ValueError('Feature-access audit differs')
    return folder,phases,sha,len(rows)


def preserve_partial(folder,name):
    target=artifact(folder,name).parent;log=folder/(name+'.log')
    if target.exists() or log.exists():
        dest=folder/'partial_attempts'/f'{name}_{uuid.uuid4().hex}';dest.mkdir(parents=True)
        if target.exists():target.rename(dest/'output')
        if log.exists():log.rename(dest/'phase.log')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',type=Path);a=p.parse_args()
    if any(os.environ.get(k)!='0' for k in ('PYTHONHASHSEED','PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH')):
        raise ValueError('Required MPS environment differs')
    guard=install_mask_open_guard(DATA);plan=frozen_preflight()
    for seq in SEQUENCES:input_data(seq,plan)
    sources=source_hashes();identities={m:identity(m,'seq6',plan) for m in METHODS}
    if guard['open_attempts']:raise ValueError('Annotation open during preflight')
    done=validate_resume(a.resume,plan)[3] if a.resume else 0
    if not a.execute:
        print(json.dumps({'preflight_passed':True,'full_videos':len(SEQUENCES)*len(METHODS),
                          'smoke_trajectories':len(METHODS)*2,'feature_control_trajectories':len(METHODS),'completed_phases':done,
                          'free_bytes':available(OUT),'annotation_opens':0}));return
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        available(OUT)
        if a.resume:
            folder,phases,manifest_sha,done=validate_resume(a.resume,plan);preserve_partial(folder,phases[done][0])
        else:
            folder=Path(tempfile.mkdtemp(prefix='warm_author_comparator_v1_',dir=OUT));phases=commands(folder)
            manifest={'scope':'Previously inspected automatic warm-prompt cases; not independent validation.',
                'sequences':list(SEQUENCES),'methods':list(METHODS),'phases':[n for n,_ in phases],
                'sources':sources,'identities':identities,'label_guard':dict(guard)}
            (folder/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');manifest_sha=digest(folder/'run_manifest.json')
        stop,ready,errors=threading.Event(),threading.Event(),[]
        thread=threading.Thread(target=monitor,args=(folder,stop,ready,errors,folder/f'free_space_{uuid.uuid4().hex}.jsonl'),daemon=True)
        thread.start();ready.wait(10)
        try:
            if not ready.is_set() or errors:raise RuntimeError('Storage monitor failed')
            for name,cmd in phases[done:]:
                if (digest(folder/'run_manifest.json')!=manifest_sha or source_hashes()!=sources or
                        frozen_preflight()!=plan or guard['open_attempts'] or errors):raise RuntimeError('Frozen inputs changed')
                free=available(folder);start=time.monotonic()
                print(json.dumps({'phase':name,'status':'starting','output':str(folder),'free_bytes':free}),flush=True)
                run_phase(cmd,folder/(name+'.log'),folder,errors)
                with (folder/'phases.jsonl').open('a') as log:
                    log.write(json.dumps({'phase':name,'passed':True,'wall_seconds':time.monotonic()-start,
                        'artifact_sha256':digest(artifact(folder,name)),'log_sha256':digest(folder/(name+'.log'))})+'\n')
                print(json.dumps({'phase':name,'status':'completed'}),flush=True)
        finally:
            stop.set();thread.join(10)
            if thread.is_alive() or errors:raise RuntimeError('Storage monitor did not close cleanly')
    print(json.dumps({'passed':True,'output':str(folder)}),flush=True)


if __name__=='__main__':main()
