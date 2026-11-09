"""Run the real training executor uninterrupted and across a planned crash gap."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.mavos_value_utils import atomic_json
from scripts.memory_token_io import digest,validate_output
from scripts.rde_training_state import load_checkpoint,verify_journal
from scripts.run_rde_value_training import frozen_sources
from scripts.run_token_conditioned_key_adapter import _guard_sources,_tree_digest
from scripts.verify_token_conditioned_key_adapter import require_tree_equal


def child(command,log,out,phase):
    with log.open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        atomic_json(out/'progress.json',{'phase':phase,'child_pid':process.pid,'status':'running','efficacy_started':False})
        code=process.wait()
    return code


def run(out,gates,rde):
    out=validate_output(out);out.mkdir(parents=True,exist_ok=False);sources=frozen_sources()
    atomic_json(out/'identity.json',{'training_sources':sources,'script_sha256':digest(Path(__file__))})
    base=[sys.executable,'-u',str(ROOT/'scripts/run_rde_value_training.py'),'--gates-root',str(gates),'--rde-repository',str(rde),'--engineering']
    reference=out/'uninterrupted';resumed=out/'resumed'
    # Exercise the lock fence against an actually running writer. It must fail
    # without creating a failure receipt in the owner's attempt.
    with (out/'uninterrupted.log').open('w') as stream:
        process=subprocess.Popen(base+['--output-dir',str(reference)],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        atomic_json(out/'progress.json',{'phase':'uninterrupted','child_pid':process.pid,'status':'running'})
        locked=False
        while process.poll() is None and not locked:
            path=reference/'writer.lock'
            if path.exists():
                with path.open('a+') as f:
                    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
                    except BlockingIOError:locked=True
            if not locked:time.sleep(.05)
        if not locked:raise RuntimeError('Reference writer terminated before lock ownership was observed')
        with (out/'intruder.log').open('w') as log:
            intruder=subprocess.run(base+['--output-dir',str(reference)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        if intruder.returncode==0 or list(reference.glob('failure_*.json')):raise RuntimeError('Concurrent writer was not fenced without mutating owner error state')
        if process.wait()!=0:raise RuntimeError('Uninterrupted training fixture failed')
    code=child(base+['--output-dir',str(resumed),'--interrupt-before-checkpoint','3'],out/'interrupted.log',out,'controlled_interruption')
    if code!=75:raise RuntimeError('Controlled interruption did not occur at the declared commit gap')
    identity_sha=digest(resumed/'identity.json');state=load_checkpoint(resumed/'training/segmentation_only',identity_sha)
    if state['frames_seen']!=2 or verify_journal(resumed/'training/segmentation_only/frames',identity_sha,2,state['journal_tip'])!=1:
        raise RuntimeError('Interruption did not leave exactly the planned orphan frame')
    if child(base+['--output-dir',str(resumed)],out/'resume.log',out,'fresh_process_resume')!=0:raise RuntimeError('Fresh runner resume failed')
    if digest(reference/'identity.json')!=digest(resumed/'identity.json'):raise RuntimeError('Fixtures used different training identities')
    rows={}
    for arm in ['segmentation_only','segmentation_plus_consistency']:
        left=reference/'training'/arm;right=resumed/'training'/arm
        a=load_checkpoint(left,identity_sha);b=load_checkpoint(right,identity_sha);require_tree_equal(a,b)
        files=sorted((left/'frames').glob('*.json'));others=sorted((right/'frames').glob('*.json'))
        if [p.name for p in files]!=[p.name for p in others] or any(digest(p)!=digest(q) for p,q in zip(files,others)):
            raise RuntimeError('Scientific frame receipts differ after interrupted runner resume')
        if a['frames_seen']!=8 or a['cursor']!={'epoch':1,'sequence_index':0,'next_frame':None}:raise RuntimeError('Fixture sequence/epoch coverage differs')
        rows[arm]={'frames':8,'updates':a['updates'],'journal_tip':a['journal_tip'],'final_state_sha256':_tree_digest(a),'exact':True}
    _guard_sources(sources)
    result={'passed':True,'runner_safety_passed':True,'training_sources':sources,'arms':rows,'exclusive_writer_fence_passed':True,
        'orphan_replay_exact':True,'fresh_process_full_state_exact':True,'scope':'Same executor on fixed one-epoch seq2/seq3 TRAIN prefixes; no candidate efficacy result',
        'full_training_authorized_only_with_matched_native_reference':True}
    atomic_json(out/'results.json',result);atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),'candidate_efficacy_started':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--gates-root',type=Path,required=True);p.add_argument('--rde-repository',type=Path,required=True);a=p.parse_args()
    try:run(a.output_dir,a.gates_root,a.rde_repository)
    except Exception as e:
        if a.output_dir.exists():atomic_json(a.output_dir/'failure.json',{'error':str(e),'type':type(e).__name__})
        raise
