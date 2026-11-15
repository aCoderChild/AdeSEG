#!/usr/bin/env python3
"""Independently audit saved warm-compatibility artifacts; no labels or GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v2 import load_prepared
from scripts.crossfit_observer_capture_shared_v1 import BASE,DATA,EXISTING,phase_artifact,read,require,validate_plan,verify_frames
from scripts.audit_crossfit_observer_capture_v1 import tensor_tree_equal,tensor_tree_finite
from scripts.audit_warm_observation_v1 import validate_observation
from scripts.run_crossfit_warm_gate_v1 import (ROOT as PROJECT,SOURCES,WARM_SHA,gate_inputs,verify_old_mps_hooks,warm_plan)


def checked_capture(root,seq,mode,index,existing,warm):
    folder=Path(root)/seq/mode;path=folder/'receipt.json';expected=phase_artifact(root,f'{seq}_{mode}',path)
    rec=read(path);require(digest(path)==expected,'Gate receipt changed while parsing')
    frames,records,start,targets,mask,_,_,limit,roi=gate_inputs(seq,index,existing,warm)
    require(rec.get('passed') is True and rec.get('sequence')==seq and rec.get('mode')==mode and
        rec.get('warm_plan_sha256')==WARM_SHA and rec.get('source_sha256')=={p:digest(PROJECT/p) for p in SOURCES} and
        rec.get('environment_flags')=={'PYTHONHASHSEED':'0','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'} and
        rec.get('label_guard',{}).get('open_attempts')==0 and rec.get('prompt_frame')==start and
        rec.get('roi_xyxy')==roi and rec.get('target_frames')==targets and rec.get('input_images')==records and
        rec.get('model_state_sha256')==warm['old_identity']['model_state_sha256'], 'Warm gate identity differs')
    conditioning=folder/'conditioning_mask.png';require(digest(conditioning)==rec['conditioning_sha256'],'Conditioning changed')
    with Image.open(conditioning) as image:generated=np.asarray(image.convert('L'))>0
    require(digest(conditioning)==rec['conditioning_sha256'] and np.array_equal(generated,mask),'Archived prompt differs')
    end=len(frames) if limit==0 else start+limit+1
    require([r['frame'] for r in rec['rows']]==list(range(start,end)),'Warm frame coverage differs')
    for row in rec['rows']:
        image=folder/f"{row['frame']:06d}.png"
        require(digest(image)==row['mask_sha256'],'Saved gate mask changed')
        with Image.open(image) as saved,Image.open(frames[row['frame']].image_path) as source:
            require(saved.format=='PNG' and saved.size==source.size and
                set(np.unique(np.asarray(saved.convert('L')))).issubset({0,255}),'Saved gate geometry differs')
        require(digest(image)==row['mask_sha256'],'Saved gate mask changed during decode')
    observations=rec['observations'];expected_targets=targets if mode=='capture' else []
    require([r['frame'] for r in observations]==expected_targets,'Warm observations differ')
    for row in observations:validate_observation(row,folder,roi)
    verify_frames(frames,records);return rec,expected


def compare_tensors(current,previous,frames):
    for frame in frames:
        a=next(x for x in current['observations'] if x['frame']==frame)
        b=next(x for x in previous['observations'] if x['frame']==frame)
        for key in ('feature','snapshot'):
            vals=[]
            for row in (a,b):
                path=Path(row[key+'_file']);require(digest(path)==row[key+'_sha256'],'Compatibility tensor changed')
                vals.append(torch.load(path,map_location='cpu',weights_only=True))
                require(digest(path)==row[key+'_sha256'] and tensor_tree_finite(vals[-1]),'Compatibility tensor changed or nonfinite')
            require(tensor_tree_equal(vals[0],vals[1]),f'Warm historical tensor differs: frame {frame} {key}')


def audit(root):
    root=Path(root).resolve(strict=True);require(root.is_relative_to(BASE.resolve()),'Gate root outside experiment')
    warm=warm_plan();verify_old_mps_hooks(warm['old_identity']);index=validate_plan();existing,_=load_prepared(EXISTING)
    results={};pids=set();uuids=set()
    for seq,mode in (('seq3','capture'),('seq6','capture'),('seq23','reference'),('seq23','capture')):
        rec,sha=checked_capture(root,seq,mode,index,existing,warm)
        require(rec['process_id'] not in pids and rec['run_uuid'] not in uuids,'Gate runs are not fresh processes')
        pids.add(rec['process_id']);uuids.add(rec['run_uuid']);results[(seq,mode)]=(rec,sha)
    for seq,frames in (('seq3',list(range(1,15))),('seq6',[24,90])):
        rec=results[(seq,'capture')][0]
        if seq=='seq3':
            evidence=existing['sequences'][seq];old_ref=read(evidence['reference_receipt_path']);old_cap=read(evidence['capture_receipt_path'])
            require(digest(evidence['reference_receipt_path'])==evidence['reference_receipt_sha256'] and
                    digest(evidence['capture_receipt_path'])==evidence['capture_receipt_sha256'],'Historical seq3 receipt changed')
        else:
            evidence=warm['seq6_historical'];old_ref=read(evidence['reference_path']);old_cap=read(evidence['capture_path'])
            require(digest(evidence['reference_path'])==evidence['reference_sha256'] and
                    digest(evidence['capture_path'])==evidence['capture_sha256'],'Historical seq6 receipt changed')
        require(all(rec[key]==old_ref[key]==old_cap[key] for key in ('rows','positions','selection','anchor_events','token_window_events')),
                f'{seq} warm historical trajectory differs')
        compare_tensors(rec,old_cap,frames)
    ref=results[('seq23','reference')][0];cap=results[('seq23','capture')][0]
    require(all(ref[key]==cap[key] for key in ('rows','positions','selection','anchor_events','token_window_events')) and
            ref['conditioning_sha256']==cap['conditioning_sha256'],'Seq23 warm observer changed reference')
    return {'passed':True,'scope':'Same-MPS warm compatibility; not efficacy or CPU/MPS parity; seq23 tensors not independently repeated.',
        'warm_plan_sha256':WARM_SHA,'seq3_shared_frames':list(range(1,15)),'seq6_shared_frames':[24,90],
        'runs':{seq+'_'+mode:{'receipt_path':str(root/seq/mode/'receipt.json'),'receipt_sha256':sha,
            'process_id':rec['process_id'],'run_uuid':rec['run_uuid']} for (seq,mode),(rec,sha) in results.items()},
        'auditor_sha256':digest(Path(__file__)),'source_sha256':{p:digest(PROJECT/p) for p in SOURCES}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();output=a.output.resolve();require(output.is_relative_to(BASE.resolve()) and not output.exists(),'Invalid output')
    guard=install_mask_open_guard(DATA);result=audit(a.root);require(guard['open_attempts']==0,'Annotation open attempted')
    result['label_guard']=dict(guard);output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(output),'label_open_attempts':0}))

if __name__=='__main__':main()
