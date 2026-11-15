#!/usr/bin/env python3
"""Audit completed seq3/seq6 warm gate phases, even if seq23 failed."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v2 import load_prepared
from scripts.crossfit_observer_capture_shared_v1 import BASE,DATA,EXISTING,read,require,validate_plan
from scripts.run_crossfit_warm_gate_v1 import WARM_SHA,warm_plan,verify_old_mps_hooks
from scripts.audit_crossfit_warm_gate_v1 import checked_capture,compare_tensors


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve(strict=True);out=a.output.resolve()
    require(root.is_relative_to(BASE.resolve()) and out.is_relative_to(BASE.resolve()) and not out.exists(),'Invalid paths')
    guard=install_mask_open_guard(DATA);warm=warm_plan();verify_old_mps_hooks(warm['old_identity'])
    index=validate_plan();existing,_=load_prepared(EXISTING);results={}
    for seq,frames in (('seq3',list(range(1,15))),('seq6',[24,90])):
        rec,sha=checked_capture(root,seq,'capture',index,existing,warm)
        if seq=='seq3':
            evidence=existing['sequences'][seq];refpath=Path(evidence['reference_receipt_path']);cappath=Path(evidence['capture_receipt_path'])
            refsha=evidence['reference_receipt_sha256'];capsha=evidence['capture_receipt_sha256']
        else:
            evidence=warm['seq6_historical'];refpath=Path(evidence['reference_path']);cappath=Path(evidence['capture_path'])
            refsha=evidence['reference_sha256'];capsha=evidence['capture_sha256']
        old_ref=read(refpath);old_cap=read(cappath)
        require(digest(refpath)==refsha and digest(cappath)==capsha,'Historic receipt changed')
        trajectory={key:{'old_reference_matches_capture':old_ref[key]==old_cap[key],
                         'new_matches_old_reference':rec[key]==old_ref[key]}
                    for key in ('rows','positions','selection','anchor_events','token_window_events')}
        tensors={}
        for frame in frames:
            try:compare_tensors(rec,old_cap,[frame]);tensors[str(frame)]={'exact':True}
            except (ValueError,KeyError,StopIteration) as exc:tensors[str(frame)]={'exact':False,'error':str(exc)}
        results[seq]={'current_receipt_path':str(root/seq/'capture/receipt.json'),'current_receipt_sha256':sha,
            'historical_reference_sha256':refsha,'historical_capture_sha256':capsha,
            'trajectory_comparisons':trajectory,'tensor_comparisons':tensors,
            'exact_compatibility':all(x['old_reference_matches_capture'] and x['new_matches_old_reference'] for x in trajectory.values())
                and all(x['exact'] for x in tensors.values())}
    require(guard['open_attempts']==0,'Annotation open attempted')
    result={'passed':all(x['exact_compatibility'] for x in results.values()),
        'scope':'Partial same-MPS compatibility audit for seq3/6 only; seq23 gate did not pass.',
        'warm_plan_sha256':WARM_SHA,'root':str(root),'sequences':results,
        'auditor_sha256':digest(Path(__file__)),'label_guard':dict(guard)}
    out.mkdir(parents=True,exist_ok=False);file=out/'audit.json';file.write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':result['passed'],'audit_sha256':digest(file),'label_guard':dict(guard)},indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'output':str(out),'results':{k:v['exact_compatibility'] for k,v in results.items()}}))

if __name__=='__main__':main()
