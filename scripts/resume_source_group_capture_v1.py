#!/usr/bin/env python3
"""Safely resume the remaining source-group captures without overwriting runs.

Default is CPU-only dry-run. --execute requires the original free-space guard,
creates a unique Drive directory and stops on the first failed phase. Completed
sequences are skipped only after their full saved audit/trajectory/observations
are reverified. Incomplete runs remain preserved and are not treated as passes.
Space is sampled during each phase; the 22 GiB guard is enforced at phase start,
and monitoring does not interrupt a subprocess already writing. The lock only
prevents simultaneous invocations of this launcher; coordinate other GPU work.
"""
import argparse,fcntl,json,os,shutil,subprocess,sys,tempfile,threading
from contextlib import contextmanager
import numpy as np
from PIL import Image
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.freeze_source_group_runtime_v1 import current_identity,validate_source_freeze
from scripts.run_source_group_capture_v1 import verify_short_gate
from scripts.audit_source_group_capture_v1 import verify_trajectory,verify_observation_coverage,verify_sample_file

BASE=Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment')
OUT=BASE/'contrast_refinement_v1'
PRE=OUT/'source_group_capture_preflight_attempt03'
FREEZE=OUT/'source_group_runtime_freeze_attempt01'
GATE=OUT/'source_group_seq3_short_audit_attempt01/verification.json'
REF=BASE/'lean_utility_training_capture_v1/seq4_capture_attempt01'
FULLAUDIT=OUT/'lean_window_full_seq4_audit_attempt02/comparison.json'
CHECKPOINT=BASE/'author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt'
CONFIG=ROOT/'protocols/source_group_capture_v1.json'
PRIOR=('seq3','seq2','seq4','seq5')
REMAINING=('seq12','seq13','seq14','seq9','seq15')
ALL_SEQUENCES=PRIOR+REMAINING

def read(path):return json.loads(Path(path).read_text())

def validate_gate_binding(audit,ref,cap,gate,gate_sha):
    identity=gate.get('identity')
    if (audit.get('identity')!=identity or ref.get('identity')!=identity or cap.get('identity')!=identity
        or ref.get('short_gate_sha256')!=gate_sha or cap.get('short_gate_sha256')!=gate_sha):
        raise ValueError('Saved capture is not bound to the current short gate')


def validate_archived_conditioning(spec,seq,*paths):
    archive=Path(spec['source'])/seq/'prompt_refined.png'
    if digest(archive)!=spec['archived_prompt_mask_sha256']:raise ValueError('Archived prompt mask changed')
    original=np.asarray(Image.open(archive).convert('L'))>0
    for path in paths:
        saved=np.asarray(Image.open(path.parent/'conditioning_mask.png').convert('L'))>0
        if saved.shape!=original.shape or not np.array_equal(saved,original):raise ValueError('Conditioning differs from archived prompt')


@contextmanager
def exclusive_lock(path):
    with Path(path).open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another source-group resume is already running')
        try:yield
        finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def validate_reference(run,seq,pre,freeze_sha):
    ref_path=run/f'{seq}_reference/receipt.json'
    if not ref_path.is_file():return None
    if not ref_path.resolve().is_relative_to(OUT.resolve()):raise ValueError('Reference path escaped experiment storage')
    ref=read(ref_path);spec=pre['sequences'][seq];gate=read(GATE);identity=ref.get('identity',{})
    if (not ref.get('passed') or ref.get('mode')!='reference' or ref.get('sequence')!=seq
        or ref.get('full_sequence') is not True or ref.get('limit')!=0 or ref.get('targets')!=spec['targets']
        or ref.get('prompt_frame')!=spec['prompt_frame'] or ref.get('short_gate_sha256')!=digest(GATE)
        or ref.get('prompt_sha256')!=spec['archived_prompt_sha256'] or ref.get('archived_prompt_mask_sha256')!=spec['archived_prompt_mask_sha256']
        or ref.get('observations')!=[]
        or identity!=gate.get('identity') or identity.get('preflight_sha256')!=digest(PRE/'manifest.json')
        or identity.get('source_freeze_sha256')!=freeze_sha or identity.get('runner_sha256')!=digest(ROOT/'scripts/run_source_group_capture_v1.py')
        or ref.get('label_guard',{}).get('open_attempts')!=0):
        raise ValueError(f'{seq}: reference identity differs: {run}')
    _,targets=verify_trajectory(ref,ref,spec)
    writes=[x for x in ref['token_window_events'] if x['stage']=='after_write']
    if len(writes)!=len(ref['rows'])-1 or any(x['total_tokens']>16384 or x['insurance_tokens']>3072 for x in writes):
        raise ValueError(f'{seq}: reference token budget differs')
    for row in ref['rows']:
        if digest(ref_path.parent/f"{row['frame']:06d}.png")!=row['mask_sha256']:raise ValueError(f'{seq}: reference mask changed')
    if digest(ref_path.parent/'conditioning_mask.png')!=ref.get('conditioning_sha256'):raise ValueError(f'{seq}: reference conditioning changed')
    validate_archived_conditioning(spec,seq,ref_path)
    return {'run':str(run),'receipt_path':str(ref_path),'receipt_sha256':digest(ref_path),'targets':len(targets)}


def validate_existing(run,seq,pre,freeze_sha):
    audit_path=run/f'{seq}_audit/verification.json';cap_path=run/f'{seq}_capture/receipt.json'
    if not audit_path.is_file():return None
    audit=read(audit_path);ref_path=Path(audit['reference_receipt_path']).resolve()
    if not ref_path.is_relative_to(OUT.resolve()) or ref_path.parent.name!=f'{seq}_reference':raise ValueError('Audit reference path is invalid')
    if not ref_path.is_file() or not cap_path.is_file():raise ValueError(f'{seq}: audit without source receipts: {run}')
    if validate_reference(ref_path.parent.parent,seq,pre,freeze_sha) is None:raise ValueError('Validated reference missing')
    ref=read(ref_path);cap=read(cap_path);spec=pre['sequences'][seq]
    identity=ref.get('identity',{})
    if (not audit.get('passed') or audit.get('sequence')!=seq or audit.get('full_sequence') is not True or audit.get('limit')!=0
        or audit.get('targets')!=spec['targets'] or audit.get('frames')!=spec['frames']-spec['prompt_frame']
        or audit.get('auditor_sha256')!=digest(ROOT/'scripts/audit_source_group_capture_v1.py')
        or Path(audit.get('reference_receipt_path','')).resolve()!=ref_path.resolve() or Path(audit.get('capture_receipt_path','')).resolve()!=cap_path.resolve()
        or audit.get('reference_receipt_sha256')!=digest(ref_path) or audit.get('capture_receipt_sha256')!=digest(cap_path)
        or ref.get('mode')!='reference' or cap.get('mode')!='capture' or ref.get('sequence')!=seq or cap.get('sequence')!=seq
        or audit.get('identity')!=identity or cap.get('identity')!=identity or identity.get('preflight_sha256')!=digest(PRE/'manifest.json')
        or identity.get('source_freeze_sha256')!=freeze_sha or identity.get('runner_sha256')!=digest(ROOT/'scripts/run_source_group_capture_v1.py')
        or any(x.get('label_guard',{}).get('open_attempts')!=0 for x in (audit,ref,cap))):
        raise ValueError(f'{seq}: saved audit identity differs: {run}')
    validate_gate_binding(audit,ref,cap,read(GATE),digest(GATE));validate_archived_conditioning(spec,seq,ref_path,cap_path)
    _,targets=verify_trajectory(ref,cap,spec);verify_observation_coverage(cap['observations'],targets)
    for row in ref['rows']:
        for folder in (ref_path.parent,cap_path.parent):
            if digest(folder/f"{row['frame']:06d}.png")!=row['mask_sha256']:raise ValueError(f'{seq}: trajectory file changed')
    for path,receipt in ((ref_path,ref),(cap_path,cap)):
        if digest(path.parent/'conditioning_mask.png')!=receipt.get('conditioning_sha256') or receipt.get('conditioning_sha256')!=audit.get('conditioning_sha256'):
            raise ValueError(f'{seq}: conditioning changed')
    for row in cap['observations']:
        verify_sample_file(Path(row['feature_file']),cap_path.parent,row['feature_sha256'])
        verify_sample_file(Path(row['snapshot_file']),cap_path.parent,row['snapshot_sha256'])
    return {'run':str(run),'audit_path':str(audit_path),'audit_sha256':digest(audit_path),'samples':len(targets)}

def discover_completed(root,pre,freeze_sha):
    completed={}
    for run in sorted(p for p in root.iterdir() if p.is_dir() and (p.name.startswith('source_group_full_capture_') or p.name.startswith('source_group_capture_resume_v1_'))):
        for seq in ALL_SEQUENCES:
            record=validate_existing(run,seq,pre,freeze_sha)
            if record is not None:completed.setdefault(seq,[]).append(record)
    return completed


def discover_references(root,pre,freeze_sha):
    references={}
    for run in sorted(p for p in root.iterdir() if p.is_dir() and (p.name.startswith('source_group_full_capture_') or p.name.startswith('source_group_capture_resume_v1_'))):
        for seq in REMAINING:
            record=validate_reference(run,seq,pre,freeze_sha)
            if record is not None:references.setdefault(seq,[]).append(record)
    return references

def environment_setup():
    pre=read(PRE/'manifest.json');freeze=read(FREEZE/'source_freeze.json');done=read(FREEZE/'completion.json')
    identity=current_identity(PRE,REF,CHECKPOINT);validate_source_freeze(freeze,done,digest(FREEZE/'source_freeze.json'),identity)
    gate=read(GATE);verify_short_gate(gate,digest(PRE/'manifest.json'),digest(FREEZE/'source_freeze.json'),digest(ROOT/'scripts/run_source_group_capture_v1.py'),digest(ROOT/'scripts/audit_source_group_capture_v1.py'),gate['identity']['runtime_environment'])
    return pre,read(CONFIG),digest(FREEZE/'source_freeze.json')

def run_phase(command,log,env):
    with log.open('x') as f:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Phase failed (exit {result.returncode}); see {log}')


def verify_launcher_files(launcher_path,launcher_sha,manifest_path,manifest_sha):
    if digest(launcher_path)!=launcher_sha or digest(manifest_path)!=manifest_sha:
        raise RuntimeError('Launcher or its recorded run manifest changed during execution')

def monitor_space(run,guard,stop,ready,errors):
    previous=None
    try:
        with (run/'free_space_monitor.jsonl').open('x') as f:
            while True:
                free=shutil.disk_usage(run).free;below=free<guard
                row={'utc':datetime.now(timezone.utc).isoformat(),'free_bytes':free,'guard_bytes':guard,
                     'below_guard':below,'transition':previous is not None and below!=previous}
                f.write(json.dumps(row)+'\n');f.flush();ready.set()
                if previous is None or below!=previous:print('space',json.dumps(row),flush=True)
                previous=below
                if stop.wait(5):break
    except Exception as error:
        errors.append(repr(error));ready.set()
        print('Space monitor failed:',repr(error),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');a=p.parse_args()
    pre,config,freeze_sha=environment_setup();completed=discover_completed(OUT,pre,freeze_sha)
    pending=[s for s in REMAINING if s not in completed];references=discover_references(OUT,pre,freeze_sha)
    free=shutil.disk_usage(OUT).free;guard=config['min_free_bytes']
    missing_prior=[s for s in PRIOR if s not in completed]
    plan={'completed':completed,'missing_prior':missing_prior,'pending':pending,'reusable_references':references,'free_bytes':free,'guard_bytes':guard,'guard_pass':free>=guard,'execute':a.execute,'scope':'Exploratory; patient independence unknown.'}
    print(json.dumps(plan,indent=2),flush=True)
    if not a.execute:return
    if missing_prior:raise RuntimeError('Previously completed audits missing; verify before resuming: '+','.join(missing_prior))
    if not pending:return
    if free<guard:raise RuntimeError('Free space below original 22 GiB guard; no output directory created')
    with exclusive_lock(OUT/'.source_group_capture_resume_v1.lock'):
        # Recheck after acquiring the lock to prevent duplicate concurrent work.
        pre,config,freeze_sha=environment_setup();completed=discover_completed(OUT,pre,freeze_sha)
        pending=[s for s in REMAINING if s not in completed];references=discover_references(OUT,pre,freeze_sha)
        free=shutil.disk_usage(OUT).free;guard=config['min_free_bytes']
        if any(s not in completed for s in PRIOR):raise RuntimeError('Previously completed audits missing after locking')
        if not pending:
            print('All remaining videos already verified.',flush=True);return
        if free<guard:raise RuntimeError('Free space below original 22 GiB guard after locking; no output directory created')
        execute_pending(pre,config,freeze_sha,pending,guard,references,completed)


def execute_pending(pre,config,freeze_sha,pending,guard,references,completed):
    run=Path(tempfile.mkdtemp(prefix='source_group_capture_resume_v1_',dir=validate_output(OUT)))
    launcher_sha=digest(Path(__file__))
    run_manifest={'launcher_sha256':launcher_sha,'preflight_sha256':digest(PRE/'manifest.json'),
                  'source_freeze_sha256':freeze_sha,'short_gate_sha256':digest(GATE),'guard_bytes':guard,
                  'pending':pending,'reusable_references':references,'previously_completed':completed,
                  'scope':'Exploratory capture; no patient-independence or efficacy claim.'}
    manifest_path=run/'run_manifest.json'
    with manifest_path.open('x') as f:json.dump(run_manifest,f,indent=2);f.write('\n')
    manifest_sha=digest(manifest_path)
    events=run/'status_events.jsonl';env=os.environ.copy();env['PYTORCH_ENABLE_MPS_FALLBACK']='0';env['PYTORCH_MPS_FAST_MATH']='0'
    def event(seq,phase,status):
        record={'utc':datetime.now(timezone.utc).isoformat(),'sequence':seq,'phase':phase,'status':status,'free_bytes':shutil.disk_usage(run).free,
                'launcher_sha256':launcher_sha,'run_manifest_sha256':manifest_sha}
        with events.open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
        print(json.dumps(record),flush=True)
    print('PID',os.getpid(),'output',run,flush=True)
    stop=threading.Event();ready=threading.Event();monitor_errors=[]
    monitor=threading.Thread(target=monitor_space,args=(run,guard,stop,ready,monitor_errors),daemon=True);monitor.start()
    if not ready.wait(10) or monitor_errors:
        stop.set();monitor.join(timeout=5);raise RuntimeError('Free-space monitor did not start')
    common=['--preflight',str(PRE),'--reference-capture',str(REF),'--checkpoint',str(CHECKPOINT)]
    try:
        for seq in pending:
            verify_launcher_files(Path(__file__),launcher_sha,manifest_path,manifest_sha)
            reference_dir=Path(references[seq][0]['receipt_path']).parent if seq in references else run/f'{seq}_reference'
            for mode in ('reference','capture'):
                verify_launcher_files(Path(__file__),launcher_sha,manifest_path,manifest_sha)
                if mode=='reference' and seq in references:
                    event(seq,mode,'reused_verified');continue
                phase=mode;event(seq,phase,'start')
                command=[sys.executable,str(ROOT/'scripts/run_source_group_capture_v1.py'),*common,'--mps-full-audit',str(FULLAUDIT),'--source-freeze',str(FREEZE),'--short-gate',str(GATE),'--sequence',seq,'--mode',mode,'--output',str(run/f'{seq}_{mode}')]
                run_phase(command,run/f'{seq}_{mode}.log',env)
                if monitor_errors:raise RuntimeError('Free-space monitor failed; stopping before next phase')
                event(seq,mode,'passed')
            phase='audit';event(seq,phase,'start')
            verify_launcher_files(Path(__file__),launcher_sha,manifest_path,manifest_sha)
            command=[sys.executable,str(ROOT/'scripts/audit_source_group_capture_v1.py'),*common,'--source-freeze',str(FREEZE),'--reference',str(reference_dir),'--capture',str(run/f'{seq}_capture'),'--output',str(run/f'{seq}_audit')]
            run_phase(command,run/f'{seq}_audit.log',env)
            if monitor_errors:raise RuntimeError('Free-space monitor failed; stopping before next phase')
            record=validate_existing(run,seq,pre,freeze_sha)
            if record is None:raise ValueError('Expected complete audit missing')
            event(seq,'audit','passed')
    except Exception:
        event(seq,phase if 'phase' in locals() else 'setup','stopped');raise
    finally:
        stop.set();monitor.join()
    print('All pending full captures and independent audits passed.',flush=True)
if __name__=='__main__':main()
