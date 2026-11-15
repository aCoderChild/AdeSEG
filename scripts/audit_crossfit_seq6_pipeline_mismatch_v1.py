#!/usr/bin/env python3
"""Audit saved seq6 direct-mask vs historical observer outputs without labels/GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.crossfit_observer_capture_shared_v1 import DATA,OUT,BASE,require
from scripts.memory_token_io import digest

NEW=OUT/'crossfit_observer_full_v3_54r_l_xj/seq6/capture/receipt.json'
OLD=OUT/'expanded_train_seq6_capture_attempt01/receipt.json'
REFERENCE=OUT/'expanded_train_seq6_reference_attempt01/receipt.json'
VERIFICATION=OUT/'expanded_train_seq6_verification_attempt01/verification.json'


def read_bound(path):
    sha=digest(path);value=json.loads(path.read_text());require(digest(path)==sha,'Input changed while parsing')
    return value,sha


def compare(a,b,path,report):
    if torch.is_tensor(a) and torch.is_tensor(b):
        if not torch.equal(a,b):
            item={'path':path,'new_shape':list(a.shape),'old_shape':list(b.shape),
                  'new_dtype':str(a.dtype),'old_dtype':str(b.dtype)}
            if a.shape==b.shape and a.is_floating_point() and b.is_floating_point():
                delta=(a.float()-b.float()).abs()
                require(bool(torch.isfinite(delta).all()),'Nonfinite diagnostic delta')
                item.update(maxabs=float(delta.max()) if delta.numel() else 0,
                    meanabs=float(delta.mean()) if delta.numel() else 0,
                    nonzero=int((delta!=0).sum()),numel=delta.numel())
            report.append(item)
    elif isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys():report.append({'path':path,'new_keys':sorted(a),'old_keys':sorted(b)})
        for key in sorted(a.keys() & b.keys()):compare(a[key],b[key],path+'/'+str(key),report)
    elif isinstance(a,(list,tuple)) and isinstance(b,type(a)):
        if len(a)!=len(b):report.append({'path':path,'new_length':len(a),'old_length':len(b)})
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),report)
    elif type(a)!=type(b) or a!=b:report.append({'path':path,'new':str(a)[:100],'old':str(b)[:100]})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve();require(output.is_relative_to(BASE.resolve()) and not output.exists(),'Invalid output')
    guard=install_mask_open_guard(DATA);new,new_sha=read_bound(NEW);old,old_sha=read_bound(OLD)
    reference,reference_sha=read_bound(REFERENCE);verified,verified_sha=read_bound(VERIFICATION)
    require(new['passed'] and old['passed'] and verified['passed'] and verified['capture_receipt_sha256']==old_sha and
            verified['reference_receipt_sha256']==reference_sha,'Receipt identities differ')
    frames={}
    for frame in (24,90):
        n=next(row for row in new['observations'] if row['frame']==frame)
        o=next(row for row in old['observations'] if row['frame']==frame);details=[];files={}
        for kind in ('feature','snapshot'):
            values=[]
            for record in (n,o):
                path=Path(record[kind+'_file']);expected=record[kind+'_sha256']
                require(digest(path)==expected,'Tensor bytes changed')
                values.append(torch.load(path,map_location='cpu',weights_only=True))
                require(digest(path)==expected,'Tensor changed while loading')
            files[kind]={'new_sha256':n[kind+'_sha256'],'old_sha256':o[kind+'_sha256']}
            compare(values[0],values[1],kind,details)
        frames[str(frame)]={'files':files,'differences':details}
    comparisons={}
    for name,oldkey in (('memory_events','token_window_events'),('positions','positions'),('selections','selection')):
        trace=NEW.parent/(name+'.json');rows,sha=read_bound(trace);require(sha==new['trace_sha256'][name],'New trace changed')
        previous=old[oldkey];pairs=list(zip(rows,previous));first=next((i for i,(a,b) in enumerate(pairs) if a!=b),None)
        comparisons[name]={'new_count':len(rows),'old_count':len(previous),
            'same_at_index':sum(a==b for a,b in pairs),'first_difference_index':first,'new_sha256':sha}
    nrows=new['rows'];orows=old['rows'];require([r['frame'] for r in nrows]==[r['frame'] for r in orows],'Frame order differs')
    equal_frames=[a['frame'] for a,b in zip(nrows,orows) if a['mask_sha256']==b['mask_sha256']]
    require(guard['open_attempts']==0,'Annotation open attempted')
    result={'passed':True,'scope':'Saved artifact comparison only; no labels, GPU, efficacy or causal attribution.',
        'source_sha256':digest(Path(__file__)),'new_receipt_sha256':new_sha,'old_receipt_sha256':old_sha,
        'old_reference_sha256':reference_sha,'old_verification_sha256':verified_sha,
        'new_baseline_completion_sha256':new['baseline_completion_sha256'],
        'new_runtime_verification_sha256':new['runtime_verification_sha256'],
        'old_identity':old['identity'],'new_environment_flags':new['environment_flags'],
        'old_prompt_sha256':old['prompt_sha256'],'new_prompt_sha256':new['prompt_sha256'],
        'frames':frames,'trajectory':comparisons,'mask_equal_frames':equal_frames,
        'mask_frame_count':len(nrows),'old_row_fields':sorted(orows[0]),'new_row_fields':sorted(nrows[0]),
        'label_guard':dict(guard)}
    output.mkdir(parents=True,exist_ok=False);file=output/'audit.json';file.write_text(json.dumps(result,indent=2)+'\n')
    (output/'completion.json').write_text(json.dumps({'passed':True,'audit_sha256':digest(file),'label_guard':dict(guard)},indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(output),'label_open_attempts':guard['open_attempts']}))

if __name__=='__main__':main()
