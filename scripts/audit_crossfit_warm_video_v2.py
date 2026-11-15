#!/usr/bin/env python3
"""Audit a full warm reference and observer for one video, without labels/GPU."""
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
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import (BASE,DATA,EXISTING,archived_frame_metadata,
    phase_artifact,read,require,validate_plan,verify_frames)
from scripts.audit_warm_observation_v1 import validate_observation
from scripts.audit_warm_prompt_prototypes_v1 import validate_prompt_prototypes
from scripts.run_crossfit_warm_video_v2 import SOURCES,WARM_SHA,video_inputs,verify_old_mps_hooks,warm_plan


def checked_run(root,seq,mode,index,warm,gate_sha,cert_sha):
    folder=Path(root)/seq/mode;path=folder/'receipt.json';sha=phase_artifact(root,f'{seq}_{mode}',path)
    rec=read(path);require(digest(path)==sha,'Warm video receipt changed')
    frames,records,start,targets,mask,_,_,roi=video_inputs(seq,index,warm)
    require(rec.get('passed') is True and rec.get('sequence')==seq and rec.get('mode')==mode and
        rec.get('warm_plan_sha256')==WARM_SHA and rec.get('warm_gate_sha256')==gate_sha and
        rec.get('certificate_sha256')==cert_sha and
        rec.get('source_sha256')=={p:digest(ROOT/p) for p in SOURCES} and
        rec.get('environment_flags')=={'PYTHONHASHSEED':'0','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'} and
        rec.get('label_guard',{}).get('open_attempts')==0 and rec.get('prompt_frame')==start and
        rec.get('view')==warm['fresh_prompts'][seq]['view'] and rec.get('roi_xyxy')==roi and
        rec.get('target_frames')==targets and rec.get('input_images')==records and
        rec.get('archived_frame_metadata')==archived_frame_metadata(index,seq) and
        rec.get('model_state_sha256')==warm['old_identity']['model_state_sha256'] and
        rec.get('prompt_policy')=='warm_box_points_then_commit_archived_mask', 'Warm video identity differs')
    cond=folder/'conditioning_mask.png';require(digest(cond)==rec['conditioning_sha256'],'Conditioning changed')
    with Image.open(cond) as image:generated=np.asarray(image.convert('L'))>0
    require(digest(cond)==rec['conditioning_sha256'] and np.array_equal(generated,mask),'Archived prompt differs')
    generated_path=folder/'generated_mask.png';require(digest(generated_path)==rec['generated_sha256'],'Generated mask changed')
    with Image.open(generated_path) as image:point_generated=np.asarray(image.convert('L'))>0
    require(digest(generated_path)==rec['generated_sha256'] and point_generated.shape==mask.shape and
        rec['generated_equals_archived']==bool(np.array_equal(point_generated,mask)) and
        rec['generated_difference_pixels']==int(np.count_nonzero(point_generated!=mask)),'Generated-mask diagnostic differs')
    prototypes=validate_prompt_prototypes(folder,rec,mask)
    require([r['frame'] for r in rec['rows']]==list(range(start,len(frames))),'Warm coverage differs')
    for row in rec['rows']:
        image=folder/f"{row['frame']:06d}.png";require(digest(image)==row['mask_sha256'],'Prediction changed')
        with Image.open(image) as saved,Image.open(frames[row['frame']].image_path) as source:
            require(saved.format=='PNG' and saved.size==source.size and
                set(np.unique(np.asarray(saved.convert('L')))).issubset({0,255}),'Warm prediction geometry differs')
        require(digest(image)==row['mask_sha256'],'Prediction changed while decoding')
    expected=targets if mode=='capture' else []
    require([r['frame'] for r in rec['observations']]==expected,'Warm tensor coverage differs')
    for item in rec['observations']:
        feature,_=validate_observation(item,folder,roi)
        require(torch.equal(feature['foreground'],prototypes['foreground']) and
                torch.equal(feature['background'],prototypes['background']), 'Warm feature prototypes differ')
    verify_frames(frames,records);return rec,sha


def audit(root,sequence,certificate):
    root=Path(root).resolve(strict=True);require(root.is_relative_to(BASE.resolve()),'Invalid batch root')
    warm=warm_plan();verify_old_mps_hooks(warm['old_identity']);index=validate_plan()
    cert,cert_sha=verify_certificate(certificate);gate_sha=cert['warm_gate_sha256']
    ref,refsha=checked_run(root,sequence,'reference',index,warm,gate_sha,cert_sha)
    cap,capsha=checked_run(root,sequence,'capture',index,warm,gate_sha,cert_sha)
    require(ref['process_id']!=cap['process_id'] and ref['run_uuid']!=cap['run_uuid'] and
        cap['reference_receipt_sha256']==refsha and all(ref[k]==cap[k] for k in
        ('rows','positions','selection','anchor_events','token_window_events','conditioning_sha256',
         'generated_sha256','prototypes_sha256','prompt_early_sha256','generated_equals_archived','generated_difference_pixels')),
        'Warm observer differs from its reference')
    if sequence=='seq6':
        old=warm['seq6_historical'];refpath=Path(old['reference_path']);previous=read(refpath)
        require(digest(refpath)==old['reference_sha256'] and all(ref[k]==previous[k] for k in
            ('rows','positions','selection','anchor_events','token_window_events')),
            'Seq6 warm reference differs from historic trajectory')
    return {'passed':True,'scope':'Full warm reference/observer audit; no labels or efficacy.',
        'sequence':sequence,'warm_plan_sha256':WARM_SHA,'warm_gate_path':cert['warm_gate_path'],
        'warm_gate_sha256':gate_sha,'certificate_path':str(Path(certificate).resolve()),'certificate_sha256':cert_sha,'reference_receipt_sha256':refsha,'capture_receipt_sha256':capsha,
        'target_frames':cap['target_frames'],'auditor_sha256':digest(Path(__file__)),
        'source_sha256':{p:digest(ROOT/p) for p in SOURCES}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','certificate','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--sequence',required=True);a=p.parse_args()
    out=a.output.resolve();require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Invalid output')
    guard=install_mask_open_guard(DATA);result=audit(a.root,a.sequence,a.certificate)
    require(guard['open_attempts']==0,'Annotation open attempted');result['label_guard']=dict(guard)
    out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':a.sequence,'targets':len(result['target_frames'])}))

if __name__=='__main__':main()
