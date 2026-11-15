#!/usr/bin/env python3
"""Verify a completed short-only Lean batch before a later separate resume.

This does not authorize or start the full study; the caller must enforce this
check immediately before launching it under the shared lock.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import OUT
from scripts.run_author_lean_automatic_study_v1 import artifact, validate_resume
from scripts.run_author_lean_automatic_v1 import DATA
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def read_bound(path, expected):
    if digest(path) != expected: raise ValueError('Bound artifact changed')
    value = json.loads(path.read_text())
    if digest(path) != expected: raise ValueError('Bound artifact changed during read')
    return value


def verify(folder):
    folder = Path(folder).resolve(strict=True)
    if not folder.is_relative_to(OUT.resolve(strict=True)): raise ValueError('Batch outside experiment root')
    handoff_path = folder/'short_handoff.json'; handoff_sha = digest(handoff_path)
    handoff = read_bound(handoff_path, handoff_sha)
    manifest = read_bound(folder/'run_manifest.json', handoff['manifest_sha256'])
    snapshot = folder/'short_only_launcher.py'
    if (handoff.get('full_resume_requires_independent_handoff_validation') is not True or
        Path(handoff.get('launcher_snapshot', '')).resolve() != snapshot.resolve() or
        handoff['launcher_sha256'] != digest(snapshot) or
        manifest.get('short_only_launcher_sha256') != handoff['launcher_sha256'] or
        handoff['phases_sha256'] != digest(folder/'phases.jsonl') or
        handoff['gate_sha256'] != digest(artifact(folder, 'gate'))):
        raise ValueError('Short handoff binding differs')
    verified_folder, phases, manifest_sha, done = validate_resume(folder)
    if (verified_folder != folder or done != 3 or manifest_sha != handoff['manifest_sha256'] or
        [name for name, _ in phases[:3]] != ['smoke_seq6', 'smoke_seq23', 'gate'] or
        digest(handoff_path) != handoff_sha or digest(folder/'phases.jsonl') != handoff['phases_sha256'] or
        digest(snapshot) != handoff['launcher_sha256'] or digest(artifact(folder, 'gate')) != handoff['gate_sha256']):
        raise ValueError('Short phase proof changed or is incomplete')
    return {'passed': True, 'scope': 'Short MPS smoke handoff only; does not start or authorize full study.',
        'batch': str(folder), 'handoff_sha256': handoff_sha, 'manifest_sha256': manifest_sha,
        'launcher_sha256': handoff['launcher_sha256'], 'phases_sha256': handoff['phases_sha256'],
        'gate_sha256': handoff['gate_sha256'], 'verifier_sha256': digest(Path(__file__)),
        'full_resume_must_reverify_under_shared_lock': True}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to(OUT.resolve(strict=True)) or out.exists(): raise ValueError('Invalid output path')
    guard=install_mask_open_guard(DATA); result=verify(a.run)
    if guard['open_attempts']: raise ValueError('Label open attempted')
    result['label_guard']=dict(guard); out.mkdir(parents=True,exist_ok=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out)}))

if __name__=='__main__': main()
