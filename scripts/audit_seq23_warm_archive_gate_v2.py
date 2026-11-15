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
from scripts.audit_warm_prompt_prototypes_v1 import validate_prompt_prototypes
from scripts.audit_crossfit_warm_gate_v1 import compare_tensors
from scripts.run_seq23_warm_archive_gate_v2 import (ROOT as PROJECT,SOURCES,WARM_SHA,gate_inputs,verify_old_mps_hooks,warm_plan)


def checked_capture(root,seq,mode,index,existing,warm):
    folder=Path(root)/seq/mode;path=folder/'receipt.json';expected=phase_artifact(root,f'{seq}_{mode}',path)
    rec=read(path);require(digest(path)==expected,'Gate receipt changed while parsing')
    frames,records,start,targets,mask,_,_,limit,roi=gate_inputs(seq,index,existing,warm)
    require(rec.get('passed') is True and rec.get('sequence')==seq and rec.get('mode')==mode and
        rec.get('warm_plan_sha256')==WARM_SHA and rec.get('source_sha256')=={p:digest(PROJECT/p) for p in SOURCES} and
        rec.get('environment_flags')=={'PYTHONHASHSEED':'0','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'} and
        rec.get('label_guard',{}).get('open_attempts')==0 and rec.get('prompt_frame')==start and
        rec.get('roi_xyxy')==roi and rec.get('target_frames')==targets and rec.get('input_images')==records and
        rec.get('model_state_sha256')==warm['old_identity']['model_state_sha256'] and
        rec.get('prompt_policy')=='warm_box_points_then_commit_archived_mask', 'Warm gate identity differs')
    conditioning=folder/'conditioning_mask.png';require(digest(conditioning)==rec['conditioning_sha256'],'Conditioning changed')
    with Image.open(conditioning) as image:generated=np.asarray(image.convert('L'))>0
    require(digest(conditioning)==rec['conditioning_sha256'] and np.array_equal(generated,mask),'Archived prompt differs')
    generated_path=folder/'generated_mask.png';require(digest(generated_path)==rec['generated_sha256'],'Generated mask changed')
    with Image.open(generated_path) as image:point_generated=np.asarray(image.convert('L'))>0
    require(digest(generated_path)==rec['generated_sha256'] and point_generated.shape==mask.shape and
        rec['generated_equals_archived']==bool(np.array_equal(point_generated,mask)) and
        rec['generated_difference_pixels']==int(np.count_nonzero(point_generated!=mask)),
        'Generated/archived diagnostic differs')
    prototypes=validate_prompt_prototypes(folder,rec,mask)
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
    for row in observations:
        feature,_=validate_observation(row,folder,roi)
        require(torch.equal(feature['foreground'],prototypes['foreground']) and
                torch.equal(feature['background'],prototypes['background']),
                'Observed feature prototypes differ from archived-mask prototypes')
    verify_frames(frames,records);return rec,expected


PARTIAL=BASE/'contrast_refinement_v1/crossfit_warm_partial_audit_v1_attempt01/audit.json'
PARTIAL_SHA='73332a8fc45bfb6e4a3bc0cbd3f8fb284fb9ca2d4a27182d0a07a3db3aa42843'


def check_partial():
    done=read(PARTIAL.parent/'completion.json');require(digest(PARTIAL)==PARTIAL_SHA and done.get('passed') is True and
        done.get('audit_sha256')==PARTIAL_SHA and done.get('label_guard',{}).get('open_attempts')==0,
        'Prior warm partial evidence changed')
    part=read(PARTIAL);require(digest(PARTIAL)==PARTIAL_SHA and part.get('passed') is True and
        part.get('warm_plan_sha256')==WARM_SHA and part.get('label_guard',{}).get('open_attempts')==0 and
        part.get('auditor_sha256')==digest(ROOT/'scripts/audit_crossfit_warm_partial_v1.py'),
        'Partial audit identity differs')
    warm=warm_plan();manifest_path=EXISTING/'manifest.json'
    require(digest(manifest_path)==warm['existing_manifest_sha256'],'Historical manifest changed')
    manifest=read(manifest_path);require(digest(manifest_path)==warm['existing_manifest_sha256'],'Manifest changed while reading')
    for seq,frames in (('seq3',list(range(1,15))),('seq6',[24,90])):
        item=part['sequences'][seq];path=Path(item['current_receipt_path']).resolve(strict=True)
        root=Path(part['root']).resolve(strict=True)
        require(path.is_relative_to(root) and phase_artifact(root,f'{seq}_capture',path)==item['current_receipt_sha256'] and
            item['exact_compatibility'] is True and set(item['tensor_comparisons'])=={str(x) for x in frames} and
            all(value.get('exact') is True for value in item['tensor_comparisons'].values()) and
            all(value.get('old_reference_matches_capture') and value.get('new_matches_old_reference')
                for value in item['trajectory_comparisons'].values()), 'Prior warm compatibility differs')
        if seq=='seq3':
            evidence=manifest['sequences'][seq]
            refpath=Path(evidence['reference_receipt_path']);cappath=Path(evidence['capture_receipt_path'])
            refsha=evidence['reference_receipt_sha256'];capsha=evidence['capture_receipt_sha256']
        else:
            evidence=warm['seq6_historical'];refpath=Path(evidence['reference_path']);cappath=Path(evidence['capture_path'])
            refsha=evidence['reference_sha256'];capsha=evidence['capture_sha256']
        require(digest(refpath)==refsha==item['historical_reference_sha256'] and
                digest(cappath)==capsha==item['historical_capture_sha256'],'Historical receipts changed')
        previous_ref=read(refpath);previous_cap=read(cappath);current=read(path)
        require(digest(refpath)==refsha and digest(cappath)==capsha and digest(path)==item['current_receipt_sha256'] and
            all(current[k]==previous_ref[k]==previous_cap[k] for k in
                ('rows','positions','selection','anchor_events','token_window_events')),
            'Historical warm trajectory changed')
        for row in current['rows']:
            for folder in (path.parent,refpath.parent,cappath.parent):
                require(digest(folder/f"{row['frame']:06d}.png")==row['mask_sha256'], 'Historical warm mask changed')
        for row in current['observations']:
            for key in ('feature','snapshot'):
                tensor=Path(row[key+'_file']).resolve(strict=True)
                require(tensor.is_relative_to(path.parent) and digest(tensor)==row[key+'_sha256'],
                        'Warm partial tensor changed')
        compare_tensors(current,previous_cap,frames)
    return part


def audit(root):
    root=Path(root).resolve(strict=True);require(root.is_relative_to(BASE.resolve()),'Gate root outside experiment')
    part=check_partial();warm=warm_plan();verify_old_mps_hooks(warm['old_identity']);index=validate_plan()
    existing,_=load_prepared(EXISTING)
    ref,refsha=checked_capture(root,'seq23','reference',index,existing,warm)
    cap,capsha=checked_capture(root,'seq23','capture',index,existing,warm)
    require(ref['process_id']!=cap['process_id'] and ref['run_uuid']!=cap['run_uuid'] and
        all(ref[k]==cap[k] for k in ('rows','positions','selection','anchor_events','token_window_events',
             'conditioning_sha256','generated_sha256','prototypes_sha256','prompt_early_sha256','generated_equals_archived','generated_difference_pixels')),
        'Seq23 archived-mask warm observer differs from reference')
    return {'passed':True,'scope':'Uniform archived-mask warm initialization; seq23 same-MPS no-op, prior seq3/6 exact warm compatibility; no efficacy.',
        'warm_plan_sha256':WARM_SHA,'prompt_policy':'warm_box_points_then_commit_archived_mask',
        'partial_audit_path':str(PARTIAL),'partial_audit_sha256':PARTIAL_SHA,
        'seq3_shared_frames':list(range(1,15)),'seq6_shared_frames':[24,90],
        'seq23_generated_equals_archived':cap['generated_equals_archived'],
        'seq23_generated_difference_pixels':cap['generated_difference_pixels'],
        'runs':{'seq23_reference':{'receipt_path':str(root/'seq23/reference/receipt.json'),'receipt_sha256':refsha},
                'seq23_capture':{'receipt_path':str(root/'seq23/capture/receipt.json'),'receipt_sha256':capsha}},
        'auditor_sha256':digest(Path(__file__)),'source_sha256':{p:digest(PROJECT/p) for p in SOURCES}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();output=a.output.resolve();require(output.is_relative_to(BASE.resolve()) and not output.exists(),'Invalid output')
    guard=install_mask_open_guard(DATA);result=audit(a.root);require(guard['open_attempts']==0,'Annotation open attempted')
    result['label_guard']=dict(guard);output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(output),'label_open_attempts':0}))

if __name__=='__main__':main()
