#!/usr/bin/env python3
"""One label-free reference/observer pass for the frozen cross-fit grid.

No detector or refiner is run. The observer is existing read-only project code;
each output is checked against the previously frozen automatic-prompt A run.
Full capture requires the separately audited short-engineering gate.
"""
import argparse
from contextlib import ExitStack,nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import stage_frames
from modeling.lean_boundary_refinement_routing_v1 import route_boundary_refinement
from modeling.lean_boundary_window_runtime_v1 import verified_boundary_mps_window
from modeling.lean_mps_decoder_observer import observe_decoder_reads
from modeling.prompt_feature_refinement import reference_prototypes
from modeling.seq23_viewport import VIEWPORT,restore_mask,stage_cropped_frames
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_token_window_refiner_routing_mps import tensor_hash
from scripts.run_boundary_video_factorial_v1 import AUDIT,BASELINE,CHECKPOINT,PRE,STUDY
from scripts.resume_source_group_capture_v1 import BASE,REF
from scripts.crossfit_observer_capture_shared_v1 import (DATA,FRESH,SHORT,PLAN_SHA,read,require,
    archived_frame_metadata,phase_artifact,require_headroom,sequence_inputs,source_hashes,validate_plan,verify_frames)


def checked_trace(folder,rec):
    mapping={'memory_events':'memory_events.json','positions':'positions.json','selections':'selections.json'}
    result={}
    for key,file in mapping.items():
        path=folder/file;require(digest(path)==rec[key+'_sha256'],f'Frozen {file} changed')
        result[key]=read(path);require(digest(path)==rec[key+'_sha256'],f'Frozen {file} changed during parsing')
    return result


def check_baseline_rows(rows,trace,baseline,baseline_rec,*,short):
    expected=baseline_rec['post_prompt_state']
    require(len(rows)<=len(expected) and (short or len(rows)==len(expected)),'Baseline state coverage differs')
    for actual,old in zip(rows,expected,strict=False):
        require(actual==old,'Reference output or native state differs from frozen A baseline')
    previous=checked_trace(baseline.parent,baseline_rec)
    for key,actual in trace.items():
        old=previous[key]
        require(actual==old[:len(actual)] and (short or len(actual)==len(old)),f'{key} differs from frozen A')


def check_reference(folder, *, sequence, limit, plan_sha, baseline_sha, sources):
    folder=Path(folder).resolve(strict=True);path=folder/'receipt.json'
    expected=phase_artifact(folder.parent.parent,f'{sequence}_reference',path)
    rec=read(path);require(digest(path)==expected,'Reference changed while parsing')
    require(folder.is_relative_to(BASE.resolve()) and rec.get('passed') is True and
        rec.get('mode')=='reference' and rec.get('sequence')==sequence and rec.get('limit')==limit and
        rec.get('plan_sha256')==plan_sha and rec.get('baseline_completion_sha256')==baseline_sha and
        rec.get('source_sha256')==sources and rec.get('label_guard',{}).get('open_attempts')==0,
        'Matched reference identity differs')
    for row in rec['rows']:
        require(digest(folder/f"{row['frame']:06d}.png")==row['mask_sha256'],'Reference mask changed')
    for key in ('memory_events','positions','selections'):
        require(digest(folder/(key+'.json'))==rec['trace_sha256'][key],f'Reference {key} changed')
    return rec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',choices=FRESH,required=True);p.add_argument('--mode',choices=('reference','capture'),required=True)
    p.add_argument('--limit',type=int,choices=(0,4),default=0);p.add_argument('--reference',type=Path)
    p.add_argument('--engineering-audit',type=Path);p.add_argument('--remaining-reads',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    flags={'PYTHONHASHSEED':'0','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
    require(all(os.environ.get(k)==v for k,v in flags.items()),'Required hash seed or MPS flags differ')
    require(a.limit==0 or a.sequence in SHORT,'Only declared short-gate sequences may use a limit')
    require((a.mode=='capture')==(a.reference is not None),'Capture requires exactly one frozen reference')
    require((a.limit==0)==(a.engineering_audit is not None),'Full pass requires independently audited short gate')
    out=a.output.resolve();require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Invalid or existing output')
    guard=install_mask_open_guard(DATA);plan=validate_plan();sources=source_hashes()
    if a.engineering_audit:
        from scripts.audit_crossfit_observer_capture_v1 import verify_engineering_audit
        verify_engineering_audit(a.engineering_audit,plan,sources)
    frames,frame_records,start,grid,mask,baseline,baseline_rec,prompt=sequence_inputs(plan,a.sequence)
    label_metadata=archived_frame_metadata(plan,a.sequence)
    targets=grid if a.limit==0 else [start+1]
    require(a.remaining_reads>=len(targets) if a.mode=='capture' else a.remaining_reads>=0,'Insufficient reserved reads')
    headroom=require_headroom(BASE,a.remaining_reads)
    ref=check_reference(a.reference,sequence=a.sequence,limit=a.limit,plan_sha=PLAN_SHA,
          baseline_sha=prompt['baseline_completion_sha256'],sources=sources) if a.reference else None
    out.mkdir(parents=True,exist_ok=False);rows=[];observations=[];telemetry=[]
    with verified_boundary_mps_window(study=STUDY,audit=AUDIT,split_preflight=PRE,checkpoint=CHECKPOINT,
            baseline_reference=BASELINE,seq4_capture=REF,output=out/'verification') as (model,refiner,lease):
        old_runtime=read(baseline.parent/'verification/runtime_verification.json')
        require(read(out/'verification/runtime_verification.json')==old_runtime,'Live predictor differs from original A runtime')
        with tempfile.TemporaryDirectory(prefix='staged_',dir=out) as temporary:
            if a.sequence=='seq23':stage_cropped_frames(frames,temporary)
            else:stage_frames(frames,Path(temporary))
            state=model.init_state(video_path=temporary,offload_video_to_cpu=True,offload_state_to_cpu=False)
            require((state['video_height'],state['video_width'])==mask.shape,'Staged video/prompt geometry differs')
            try:
                with ExitStack() as stack:
                    stack.enter_context(torch.inference_mode())
                    stack.enter_context(route_boundary_refinement(model,refiner,'A',runtime_lease=lease,telemetry=telemetry))
                    _,backbone,_,_,_=model._get_image_feature(state,start,batch_size=1)
                    early=backbone['backbone_fpn'][0][0].detach().float().cpu()
                    prototypes=reference_prototypes(early,mask)
                    _,ids,_=model.add_new_mask(state,frame_idx=start,obj_id=1,mask=mask)
                    require(ids==[1],'Automatic prompt object differs')
                    stack.enter_context(observe_decoder_reads(model,set(targets),out,prototypes,observations)
                                        if a.mode=='capture' else nullcontext())
                    kwargs={} if a.limit==0 else {'max_frame_num_to_track':a.limit}
                    for frame,ids,logits in model.propagate_in_video(state,start_frame_idx=start,reverse=False,**kwargs):
                        require(ids==[1] and frame==start+len(rows) and logits.ndim==4 and
                                tuple(logits.shape[:2])==(1,1) and bool(torch.isfinite(logits).all()),
                                'Frame order, object or video logits differ')
                        prediction=logits[0,0].cpu().numpy()>0
                        if a.sequence=='seq23':prediction=restore_mask(prediction)
                        file=out/f'{frame:06d}.png';Image.fromarray(prediction.astype(np.uint8)*255).save(file)
                        row={'frame':frame,'mask_sha256':digest(file),'output_logits_sha256':tensor_hash(logits)}
                        group='cond_frame_outputs' if frame==start else 'non_cond_frame_outputs'
                        entry=state['output_dict_per_obj'][0][group][frame]
                        for key in ('pred_masks','maskmem_features','obj_ptr','object_score_logits'):
                            row[key+'_sha256']=tensor_hash(entry[key])
                        rows.append(row)
            finally:model.reset_state(state)
        torch.mps.synchronize()
        expected_count=len(frames)-start if a.limit==0 else min(a.limit+1,len(frames)-start)
        require(len(rows)==expected_count,'Propagation coverage differs')
        positions,selections=lease.observations()
        trace={'memory_events':model.token_window_runtime.events,'positions':positions,'selections':selections}
        check_baseline_rows(rows,trace,baseline,baseline_rec,short=bool(a.limit))
        if ref:
            require(rows==ref['rows'],'Observer output or native state differs from reference')
            for key,value in trace.items():require(value==read(a.reference/(key+'.json')),'Observer trace differs from reference')
        if a.mode=='capture':
            require({x['frame'] for x in observations}==set(targets),'Observer target coverage differs')
            for item in observations:
                for key in ('feature','snapshot'):
                    file=Path(item[key+'_file']).resolve(strict=True)
                    require(file.is_relative_to(out), 'Observed tensor path escapes output')
                    item[key+'_sha256']=digest(file)
        verify_frames(frames,frame_records)
        require(guard['open_attempts']==0,'Annotation-mask open attempted')
        trace_sha={}
        for key,value in trace.items():
            file=out/(key+'.json');file.write_text(json.dumps(value,indent=2)+'\n');trace_sha[key]=digest(file)
        receipt={'passed':True,'scope':'Exploratory label-free capture; not an independent test or complete attention capture.',
            'sequence':a.sequence,'mode':a.mode,'limit':a.limit,'plan_sha256':PLAN_SHA,'source_sha256':sources,
            'process_id':os.getpid(),'run_uuid':uuid.uuid4().hex,
            'environment_flags':flags,
            'baseline_completion_sha256':prompt['baseline_completion_sha256'],'prompt_sha256':prompt['prompt_provenance']['prompt_mask_sha256'],
            'view':prompt['view'],'roi_xyxy':prompt['prompt_provenance']['roi_xyxy'],'prompt_frame':start,'targets':targets,
            'reference_receipt_sha256':digest(a.reference/'receipt.json') if a.reference else None,
            'engineering_audit_sha256':digest(a.engineering_audit) if a.engineering_audit else None,
            'runtime_verification_sha256':digest(out/'verification/runtime_verification.json'),
            'archived_frame_metadata':label_metadata,
            'headroom':headroom,'input_images':frame_records,'rows':rows,'trace_sha256':trace_sha,
            'observations':observations,'label_guard':dict(guard)}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':a.sequence,'mode':a.mode,'frames':len(rows),'targets':len(observations)}))


if __name__=='__main__':main()
