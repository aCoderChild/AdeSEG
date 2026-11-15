#!/usr/bin/env python3
"""CPU-only audit of short observer/reference tensor artifacts.

This checks saved model outputs and observation tensors, never annotations or
GPU inference. It does not establish segmentation efficacy or independence.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.crossfit_observer_capture_shared_v1 import (BASE,OUT,DATA,PLAN_SHA,SHORT,read,require,
    archived_frame_metadata,phase_artifact,sequence_inputs,source_hashes,validate_plan,verify_frames)


def tensor_tree_equal(a,b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and bool(torch.equal(a,b))
    if isinstance(a,dict) and isinstance(b,dict):
        return set(a)==set(b) and all(tensor_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and type(a) is type(b):
        return len(a)==len(b) and all(tensor_tree_equal(x,y) for x,y in zip(a,b,strict=True))
    return type(a) is type(b) and a==b


def tensor_tree_finite(value):
    if torch.is_tensor(value):return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value,dict):return all(tensor_tree_finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(tensor_tree_finite(v) for v in value)
    return True


def historical_seq6_comparison(observed,frame):
    """Require actual value equality to the previously audited warm-prompt observer."""
    verify=OUT/'expanded_train_seq6_verification_attempt01/verification.json'
    capture=OUT/'expanded_train_seq6_capture_attempt01/receipt.json'
    reference=OUT/'expanded_train_seq6_reference_attempt01/receipt.json'
    verification=read(verify);record=read(capture)
    require(verification.get('passed') is True and verification.get('sequence')=='seq6' and
            verification.get('capture_receipt_sha256')==digest(capture) and
            verification.get('reference_receipt_sha256')==digest(reference) and
            verification.get('auditor_sha256')==digest(ROOT/'scripts/audit_expanded_train_capture_v1.py') and
            record.get('passed') is True and record.get('sequence')=='seq6',
            'Historical seq6 observer audit differs')
    matches=[x for x in record['observations'] if x['frame']==frame]
    require(len(matches)==1,'Historical seq6 shared observation missing')
    values=[]
    for key in ('feature','snapshot'):
        file=Path(matches[0][key+'_file']).resolve(strict=True)
        require(file.is_relative_to(capture.parent.resolve()) and digest(file)==matches[0][key+'_sha256'],
                'Historical observer tensor changed')
        value=torch.load(file,map_location='cpu',weights_only=True)
        require(digest(file)==matches[0][key+'_sha256'] and tensor_tree_finite(value),
                'Historical observer tensor changed while loading')
        values.append(value)
    require(tensor_tree_equal(observed,values),
            'Direct-mask observer differs from historical warmed-prompt features')
    return {'frame':frame,'verification_sha256':digest(verify),'capture_receipt_sha256':digest(capture),
            'reference_receipt_sha256':digest(reference),
            'feature_sha256':matches[0]['feature_sha256'],'snapshot_sha256':matches[0]['snapshot_sha256']}


def checked_run(folder,*,sequence,mode,plan,frames,records,start,baseline,prompt,sources,limit=4,grid=None):
    folder=Path(folder).resolve(strict=True);path=folder/'receipt.json'
    require(folder.is_relative_to(BASE.resolve()),'Engineering output outside experiment root')
    receipt_sha=digest(path);rec=read(path);require(digest(path)==receipt_sha,'Receipt changed while reading')
    require(rec.get('passed') is True and rec.get('sequence')==sequence and rec.get('mode')==mode and
        rec.get('limit')==limit and rec.get('plan_sha256')==PLAN_SHA and rec.get('source_sha256')==sources and
        rec.get('baseline_completion_sha256')==prompt['baseline_completion_sha256'] and
        rec.get('prompt_frame')==start and rec.get('view')==prompt['view'] and
        rec.get('roi_xyxy')==prompt['prompt_provenance']['roi_xyxy'] and rec.get('input_images')==records and
        rec.get('archived_frame_metadata')==archived_frame_metadata(plan,sequence) and
        rec.get('environment_flags')=={'PYTHONHASHSEED':'0','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'} and
        rec.get('label_guard',{}).get('open_attempts')==0,'Engineering run identity differs')
    end=start+5 if limit else len(frames)
    require([r['frame'] for r in rec['rows']]==list(range(start,end)),'State coverage differs')
    original=read(baseline);require(digest(baseline)==prompt['baseline_completion_sha256'],'Original A receipt changed')
    require(rec['rows']==original['post_prompt_state'][:end-start] and
            (limit or len(rec['rows'])==len(original['post_prompt_state'])),
            'Native trajectory differs from original A')
    for row in rec['rows']:
        file=folder/f"{row['frame']:06d}.png";require(digest(file)==row['mask_sha256'],'Saved engineering mask changed')
        with Image.open(frames[row['frame']].image_path) as source,Image.open(file) as image:
            require(image.format=='PNG' and image.size==source.size and
                    set(np.unique(np.asarray(image.convert('L')))).issubset({0,255}),'Engineering output mask geometry differs')
        require(digest(file)==row['mask_sha256'],'Engineering mask changed while parsing')
    trace={}
    for key in ('memory_events','positions','selections'):
        file=folder/(key+'.json');original_file=baseline.parent/(key+'.json')
        require(digest(file)==rec['trace_sha256'][key] and digest(original_file)==original[key+'_sha256'],'Trace file changed')
        trace[key]=read(file);expected=read(original_file)
        require(digest(file)==rec['trace_sha256'][key] and digest(original_file)==original[key+'_sha256'] and
                trace[key]==expected[:len(trace[key])] and (limit or len(trace[key])==len(expected)),
                f'{key} differs from A baseline')
    runtime_path=folder/'verification/runtime_verification.json'
    require(digest(runtime_path)==rec['runtime_verification_sha256'],
            'Runtime verification changed')
    original_runtime=baseline.parent/'verification/runtime_verification.json'
    require(digest(original_runtime)==original['runtime_verification_sha256'] and
            read(runtime_path)==read(original_runtime) and
            digest(runtime_path)==rec['runtime_verification_sha256'] and
            digest(original_runtime)==original['runtime_verification_sha256'],
            'Live runtime differs from original A')
    observations=rec['observations']
    expected_targets=[start+1] if limit else grid
    require((mode=='reference' and not observations) or
            (mode=='capture' and [x['frame'] for x in observations]==expected_targets),'Observer target coverage differs')
    tensors=[]
    for row in observations:
        files=[]
        for key in ('feature','snapshot'):
            file=Path(row[key+'_file']).resolve(strict=True)
            require(file.is_relative_to(folder) and digest(file)==row[key+'_sha256'],'Observed tensor file changed')
            value=torch.load(file,map_location='cpu',weights_only=True)
            require(digest(file)==row[key+'_sha256'] and tensor_tree_finite(value),'Observed tensor changed or nonfinite')
            files.append(value)
        feature,snapshot=files
        roi=prompt['prompt_provenance']['roi_xyxy']
        require(feature['frame']==row['frame']==snapshot['frame'] and feature['feature'].shape==(32,256,256) and
                feature['foreground'].shape==feature['background'].shape==(32,) and
                snapshot['video_height']==roi[3]-roi[1] and snapshot['video_width']==roi[2]-roi[0] and
                snapshot['decoder_inputs']['backbone_features'].shape==(1,256,64,64) and
                snapshot['decoder_outputs'][3].shape==(1,1,256,256),'Observed tensor geometry differs')
        tensors.append(files)
    verify_frames(frames,records)
    return {'folder':str(folder),'receipt_sha256':receipt_sha,'process_id':rec['process_id'],
            'run_uuid':rec['run_uuid'],'reference_receipt_sha256':rec['reference_receipt_sha256'],
            'trace':trace,'rows':rec['rows'],'tensors':tensors}


def audit_engineering(root,plan,sources):
    root=Path(root).resolve(strict=True);require(root.is_relative_to(BASE.resolve()),'Short gate outside experiment root')
    result={};pids=set();uuids=set();historical=None
    for sequence in SHORT:
        frames,records,start,_,_,baseline,_,prompt=sequence_inputs(plan,sequence)
        runs={name:checked_run(root/sequence/name,sequence=sequence,mode=name,
            plan=plan,frames=frames,records=records,start=start,baseline=baseline,prompt=prompt,sources=sources)
              for name in ('reference','capture')}
        ref,cap=(runs[k] for k in ('reference','capture'))
        for name,item in runs.items():
            require(phase_artifact(root,f'{sequence}_{name}',Path(item['folder'])/'receipt.json')==item['receipt_sha256'],
                    'Short receipt differs from launch phase')
            require(item['process_id'] not in pids and item['run_uuid'] not in uuids,'Short runs are not fresh processes')
            pids.add(item['process_id']);uuids.add(item['run_uuid'])
        require(cap['reference_receipt_sha256']==ref['receipt_sha256'],
                'Observed runs do not bind the reference')
        require(cap['rows']==ref['rows'] and cap['trace']==ref['trace'],
                'Observer changed trajectory or trace')
        if sequence=='seq6':historical=historical_seq6_comparison(cap['tensors'][0],start+1)
        result[sequence]={k:{f:v for f,v in value.items() if f in ('folder','receipt_sha256','process_id','run_uuid')}
                          for k,value in runs.items()}
    return {'passed':True,'scope':'Short no-label observer/no-op audit; seq23 tensor repeatability not separately tested; no efficacy.',
            'plan_sha256':PLAN_SHA,'source_sha256':sources,'short_root':str(root),'runs':result,
            'historical_seq6_first_observation':historical,
            'auditor_sha256':digest(Path(__file__))}


def verify_engineering_audit(path,plan,sources):
    path=Path(path).resolve(strict=True);require(path.is_relative_to(BASE.resolve()),'Audit outside experiment root')
    expected=phase_artifact(path.parent.parent,'engineering_audit',path)
    evidence=read(path);require(digest(path)==expected,'Engineering audit changed while parsing')
    require(Path(evidence['short_root']).resolve(strict=True)==path.parent.parent,
            'Engineering audit belongs to a different short batch')
    actual=audit_engineering(evidence['short_root'],plan,sources)
    require({k:v for k,v in evidence.items() if k!='label_guard'}==actual and
            evidence.get('label_guard',{}).get('open_attempts')==0,'Short engineering audit changed')
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--short-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Invalid audit output')
    guard=install_mask_open_guard(DATA);plan=validate_plan();result=audit_engineering(a.short_root,plan,source_hashes())
    require(guard['open_attempts']==0,'Annotation-mask open attempted');result['label_guard']=dict(guard)
    out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out),'sequences':list(SHORT)}))


if __name__=='__main__':main()
