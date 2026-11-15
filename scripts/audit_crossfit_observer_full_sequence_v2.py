#!/usr/bin/env python3
"""Audit one complete observer trajectory and any shared historical reads."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.crossfit_observer_capture_shared_v2 import (BASE,DATA,FRESH,PLAN_SHA,require,read,
    phase_artifact,sequence_inputs,source_hashes,v1_source_hashes,validate_plan)
from scripts.audit_crossfit_observer_capture_v1 import checked_run,historical_seq6_comparison,verify_engineering_audit


def audit_sequence(sequence,capture,engineering,plan,sources):
    gate=verify_engineering_audit(engineering,plan,v1_source_hashes())
    frames,records,start,grid,_,baseline,_,prompt=sequence_inputs(plan,sequence)
    args=dict(sequence=sequence,plan=plan,frames=frames,records=records,start=start,
              baseline=baseline,prompt=prompt,sources=sources,limit=0,grid=grid)
    cap=checked_run(capture,mode='capture',**args)
    root=Path(capture).resolve(strict=True).parent.parent
    require(phase_artifact(root,f'{sequence}_capture',Path(capture)/'receipt.json')==cap['receipt_sha256'],
            'Full observer receipt differs from launch phase')
    file=Path(capture)/'receipt.json';receipt=read(file)
    require(digest(file)==cap['receipt_sha256'] and receipt.get('engineering_audit_sha256')==digest(engineering) and
            receipt.get('reference_kind')=='source_bound_archived_A' and
            receipt.get('reference_receipt_sha256')==prompt['baseline_completion_sha256'],
            'Observer receipt, archived A reference or engineering gate differs')
    historical=[]
    if sequence=='seq6':
        for frame in (24,90):
            require(frame in grid,'Required shared historical frame is absent')
            historical.append(historical_seq6_comparison(cap['tensors'][grid.index(frame)],frame))
    return {'passed':True,'scope':'Independent full observer artifact/no-op audit; no labels or efficacy.',
            'sequence':sequence,'plan_sha256':PLAN_SHA,'source_sha256':sources,
            'engineering_audit_sha256':digest(engineering),
            'reference_kind':'source_bound_archived_A','reference_receipt_path':str(baseline),
            'reference_receipt_sha256':prompt['baseline_completion_sha256'],'capture_receipt_sha256':cap['receipt_sha256'],
            'target_frames':grid,'historical_seq6_observations':historical,
            'auditor_sha256':digest(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sequence',choices=FRESH,required=True)
    for name in ('capture','engineering-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Invalid output')
    guard=install_mask_open_guard(DATA);plan=validate_plan()
    result=audit_sequence(a.sequence,a.capture,a.engineering_audit,plan,source_hashes())
    require(guard['open_attempts']==0,'Annotation-mask open attempted');result['label_guard']=dict(guard)
    out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':a.sequence,'targets':len(result['target_frames'])}))


if __name__=='__main__':main()
