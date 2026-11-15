#!/usr/bin/env python3
"""Audit one complete observer trajectory and any shared historical reads."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.crossfit_observer_capture_shared_v1 import (BASE,DATA,FRESH,PLAN_SHA,require,read,
    phase_artifact,sequence_inputs,source_hashes,validate_plan)
from scripts.audit_crossfit_observer_capture_v1 import checked_run,historical_seq6_comparison,verify_engineering_audit


def audit_sequence(sequence,reference,capture,engineering,plan,sources):
    gate=verify_engineering_audit(engineering,plan,sources)
    frames,records,start,grid,_,baseline,_,prompt=sequence_inputs(plan,sequence)
    args=dict(sequence=sequence,plan=plan,frames=frames,records=records,start=start,
              baseline=baseline,prompt=prompt,sources=sources,limit=0,grid=grid)
    ref=checked_run(reference,mode='reference',**args)
    cap=checked_run(capture,mode='capture',**args)
    root=Path(reference).resolve(strict=True).parent.parent
    require(Path(capture).resolve(strict=True).parent.parent==root,'Reference and capture batches differ')
    require(phase_artifact(root,f'{sequence}_reference',Path(reference)/'receipt.json')==ref['receipt_sha256'] and
            phase_artifact(root,f'{sequence}_capture',Path(capture)/'receipt.json')==cap['receipt_sha256'],
            'Full receipts differ from launch phases')
    require(ref['process_id']!=cap['process_id'] and ref['run_uuid']!=cap['run_uuid'],
            'Reference and observer were not fresh processes')
    require(cap['reference_receipt_sha256']==ref['receipt_sha256'] and cap['rows']==ref['rows'] and
            cap['trace']==ref['trace'],'Full observer differs from reference')
    for folder,expected in ((Path(reference),ref['receipt_sha256']),(Path(capture),cap['receipt_sha256'])):
        file=folder/'receipt.json'
        require(digest(file)==expected,'Full receipt changed before parsing')
        receipt=read(file)
        require(digest(file)==expected and receipt.get('engineering_audit_sha256')==digest(engineering),
                'Full receipt or engineering gate differs')
    historical=[]
    if sequence=='seq6':
        for frame in (24,90):
            require(frame in grid,'Required shared historical frame is absent')
            historical.append(historical_seq6_comparison(cap['tensors'][grid.index(frame)],frame))
    return {'passed':True,'scope':'Independent full observer artifact/no-op audit; no labels or efficacy.',
            'sequence':sequence,'plan_sha256':PLAN_SHA,'source_sha256':sources,
            'engineering_audit_sha256':digest(engineering),
            'reference_receipt_sha256':ref['receipt_sha256'],'capture_receipt_sha256':cap['receipt_sha256'],
            'target_frames':grid,'historical_seq6_observations':historical,
            'auditor_sha256':digest(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sequence',choices=FRESH,required=True)
    for name in ('reference','capture','engineering-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Invalid output')
    guard=install_mask_open_guard(DATA);plan=validate_plan()
    result=audit_sequence(a.sequence,a.reference,a.capture,a.engineering_audit,plan,source_hashes())
    require(guard['open_attempts']==0,'Annotation-mask open attempted');result['label_guard']=dict(guard)
    out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':a.sequence,'targets':len(result['target_frames'])}))


if __name__=='__main__':main()
