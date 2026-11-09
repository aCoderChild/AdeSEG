"""Wait for one identified training process, then run fixed validation only.

Reuses the project's process observer and stage runner. No training restart,
checkpoint selection, replication seed or historical-test entry point.
"""
import argparse
import fcntl
import json
import re
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_rde_value_repair import evaluation_sources
from scripts.memory_token_io import digest,validate_output
from scripts.mavos_value_utils import atomic_json
from scripts.queue_davis_region_vs_temporal_selection import process_command,run_stage,update_status
from scripts.run_rde_value_training import verify_receipt
from scripts.run_token_conditioned_key_adapter import _guard_sources


def check_training_process(command,training_root):
    if command is None:return False
    output_argument=r'--output-dir\s+'+re.escape(str(training_root.resolve()))+r'(?=\s+--|$)'
    if 'scripts/run_rde_value_training.py' not in command or re.search(output_argument,command) is None:
        raise RuntimeError('Observed PID no longer belongs to the frozen training run')
    return True


def ready_training(training_root,identity_sha):
    if digest(training_root/'identity.json')!=identity_sha:raise RuntimeError('Training identity changed while waiting')
    result,_=verify_receipt(training_root)
    if result.get('engineering') is not False:raise RuntimeError('Engineering run cannot release candidate validation')
    for arm in ('segmentation_only','segmentation_plus_consistency'):
        receipt,_=verify_receipt(training_root/'training'/arm)
        if result['arms'][arm]!=receipt or digest(training_root/'training'/arm/'adapter_final.pt')!=receipt['final_adapter_file_sha256']:
            raise RuntimeError('Completed training arm/checkpoint differs')


def run(args,ownership):
    out=validate_output(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    ownership['created']=True
    with (out/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources=evaluation_sources()
        sources[str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
        for name in ['scripts/queue_davis_region_vs_temporal_selection.py']:
            sources[name]=digest(ROOT/name)
        identity_sha=digest(args.training_root/'identity.json')
        gate,_=verify_receipt(args.evaluation_gate)
        if not gate.get('evaluation_safety_passed') or gate['evaluation_sources']!=evaluation_sources():
            raise RuntimeError('Evaluation safety receipt is missing or stale')
        atomic_json(out/'identity.json',{'sources':sources,'training_identity_sha256':identity_sha,'training_pid':args.training_pid,
            'training_root':str(args.training_root),'validation_output':str(out/'validation')})
        for name in sources:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
        status={};path=out/'queue.json'
        while check_training_process(process_command(args.training_pid),args.training_root):
            _guard_sources(sources)
            if digest(args.training_root/'identity.json')!=identity_sha:raise RuntimeError('Training identity changed')
            update_status(path,status,state='waiting_for_fixed_training',training_pid=args.training_pid)
            time.sleep(30)
        _guard_sources(sources);ready_training(args.training_root,identity_sha)
        command=[sys.executable,str(ROOT/'scripts/evaluate_rde_value_repair.py'),'--training-root',str(args.training_root),
            '--native-reference',str(args.native_reference),'--evaluation-gate',str(args.evaluation_gate),'--output-dir',str(out/'validation')]
        code=run_stage(command,out/'validation.log',path,status,'fixed_validation',out/'validation/progress.json')
        if code:raise RuntimeError(f'Validation process failed with exit code {code}')
        _guard_sources(sources);result,sha=verify_receipt(out/'validation')
        update_status(path,status,state='complete',validation_results_sha256=sha,
            primary_screen_passed=result['validation_screen']['passed'],historical_test_started=False,replication_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['training-root','native-reference','evaluation-gate','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-pid',type=int,required=True);a=p.parse_args();ownership={}
    try:run(a,ownership)
    except Exception as e:
        if ownership.get('created'):atomic_json(a.output_dir/'failure.json',{'type':type(e).__name__,'error':str(e)})
        raise
