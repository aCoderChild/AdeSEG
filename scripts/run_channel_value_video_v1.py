#!/usr/bin/env python3
"""One MPS video arm using the frozen asinh channel-value checkpoint.

Archived automatic prompts and audited Lean A predictions are reused. C/D
replace only the native memory encoder's mask input; native object score,
pointer and other stored state are not directly replaced. No re-detection,
new threshold or memory-token selection rule is introduced.
"""
import argparse
from pathlib import Path
import json
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import stage_frames
from modeling.lean_channel_value_routing_v1 import route_channel_value_refinement
from modeling.lean_channel_value_window_runtime_v1 import BASE, verified_channel_value_window, require
from modeling.seq23_viewport import restore_mask,stage_cropped_frames
from scripts.crossfit_observer_capture_shared_v1 import DATA,phase_artifact,sequence_inputs,validate_plan,verify_frames
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_token_window_refiner_routing_mps import tensor_hash

SEQUENCES=('seq10','seq11','seq20','seq23')
ARMS=('A','B','C','D')


def read(path):return json.loads(Path(path).read_text())


def compare_archived(rows,trace,baseline,old,arm,short):
    """Protect native A/B state and conditioning; C/D may diverge later."""
    expected=old['post_prompt_state']
    require(len(rows)<=len(expected) and (short or len(rows)==len(expected)), 'Baseline state coverage differs')
    for i,(row,original) in enumerate(zip(rows,expected,strict=False)):
        if i==0:
            require(row['mask_sha256']==original['mask_sha256'] and
                    row['output_logits_sha256']==original['output_logits_sha256'],
                    'Initial conditioning output differs')
        if arm=='A':
            require(row==original,'A differs from archived matched Lean baseline')
        elif arm=='B' or i==0:
            for key in ('pred_masks_sha256','maskmem_features_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                require(row[key]==original[key],f'{arm} native state differs for {key}')
        elif i==1:
            for key in ('pred_masks_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                require(row[key]==original[key],f'{arm} initial native decoder differs for {key}')
    if arm in ('A','B'):
        for key,value in trace.items():
            path=baseline.parent/(key+'.json');expected_sha=old[key+'_sha256']
            require(digest(path)==expected_sha,'Archived baseline trace changed')
            original=read(path);require(digest(path)==expected_sha,'Archived trace changed during read')
            require(value==original[:len(value)] and (short or len(value)==len(original)),f'{arm} memory trace differs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch-root','certificate','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--certificate-sha256',required=True)
    p.add_argument('--sequence',choices=SEQUENCES,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--limit',type=int,choices=(0,4),required=True);p.add_argument('--smoke-audit',type=Path)
    a=p.parse_args();batch=a.batch_root.resolve(strict=True);cert=a.certificate.resolve(strict=True);out=a.output.resolve()
    require(batch.is_relative_to(BASE.resolve(strict=True)) and out.is_relative_to(batch) and not out.exists() and
            cert==batch/'certificate/certificate.json' and digest(cert)==a.certificate_sha256 and
            phase_artifact(batch,'certificate',cert)==a.certificate_sha256,'Batch output/certificate differs')
    require((a.limit==0)==(a.smoke_audit is not None),'Full video requires a verified MPS smoke audit')
    guard=install_mask_open_guard(DATA)
    if a.smoke_audit:
        from scripts.verify_channel_value_video_smoke_v1 import verify_smoke_audit
        verify_smoke_audit(a.smoke_audit,batch,a.certificate_sha256,reconstruct=False)
    plan=validate_plan();frames,records,start,_,mask,baseline,old,prompt=sequence_inputs(plan,a.sequence)
    end=len(frames) if a.limit==0 else min(len(frames),start+a.limit+1)
    require(end>start+1,'Insufficient post-prompt frames')
    out.mkdir(parents=True,exist_ok=False)
    with verified_channel_value_window(checkpoint=a.checkpoint,certificate=cert,
            certificate_sha256=a.certificate_sha256,output=out/'verification') as (model,refiner,lease):
        for i in range(start):
            with Image.open(frames[i].image_path) as image:width,height=image.size
            Image.fromarray(np.zeros((height,width),np.uint8)).save(out/f'{i:06d}.png')
        rows=[];telemetry=[];start_time=time.monotonic()
        with tempfile.TemporaryDirectory(prefix='frames_',dir=out) as staged:
            if a.sequence=='seq23':stage_cropped_frames(frames,staged)
            else:stage_frames(frames,Path(staged))
            state=model.init_state(video_path=staged,offload_video_to_cpu=True,offload_state_to_cpu=False)
            require((state['video_height'],state['video_width'])==mask.shape,'Prompt or staged view differs')
            try:
                with torch.inference_mode(),route_channel_value_refinement(model,refiner,a.arm,
                        runtime_lease=lease,telemetry=telemetry):
                    _,ids,_=model.add_new_mask(state,frame_idx=start,obj_id=1,mask=mask)
                    require(ids==[1],'Unexpected prompted object')
                    kwargs={} if a.limit==0 else {'max_frame_num_to_track':a.limit}
                    for frame,ids,logits in model.propagate_in_video(state,start_frame_idx=start,reverse=False,**kwargs):
                        require(frame==start+len(rows) and ids==[1] and logits.ndim==4 and
                                tuple(logits.shape[:2])==(1,1) and bool(torch.isfinite(logits).all()),
                                'Propagation frame/object or logits differ')
                        mask_out=logits[0,0].cpu().numpy()>0
                        if a.sequence=='seq23':mask_out=restore_mask(mask_out)
                        with Image.open(frames[frame].image_path) as image: expected_shape=(image.height,image.width)
                        require(mask_out.shape==expected_shape,'Video output geometry differs')
                        file=out/f'{frame:06d}.png';Image.fromarray(mask_out.astype(np.uint8)*255).save(file)
                        row={'frame':frame,'mask_sha256':digest(file),'output_logits_sha256':tensor_hash(logits)}
                        group='cond_frame_outputs' if frame==start else 'non_cond_frame_outputs'
                        entry=state['output_dict_per_obj'][0][group][frame]
                        for key in ('pred_masks','maskmem_features','obj_ptr','object_score_logits'):
                            require(torch.isfinite(entry[key]).all(),f'Nonfinite native state: {key}')
                            row[key+'_sha256']=tensor_hash(entry[key])
                        rows.append(row)
            finally:model.reset_state(state)
        torch.mps.synchronize()
        wall_seconds=time.monotonic()-start_time
        require(len(rows)==end-start,'Video coverage differs')
        positions,selections=lease.observations()
        trace={'memory_events':model.token_window_runtime.events,'positions':positions,'selections':selections}
        writes=[x for x in trace['memory_events'] if x['stage']=='after_write']
        require(writes and all(x['total_tokens']<=16384 and x['insurance_tokens']<=3072 for x in writes),
                'Verified memory budget differs')
        compare_archived(rows,trace,baseline,old,a.arm,short=bool(a.limit))
        require([x['frame'] for x in telemetry]==list(range(start,end)), 'Per-frame telemetry differs')
        trace_sha={}
        for key,value in (*trace.items(),('telemetry',telemetry)):
            file=out/(key+'.json');file.write_text(json.dumps(value,indent=2)+'\n');trace_sha[key]=digest(file)
        verify_frames(frames,records)
        require(guard['open_attempts']==0,'Annotation mask open attempted')
        receipt={'passed':True,'sequence':a.sequence,'arm':a.arm,'limit':a.limit,'full_video':a.limit==0,
            'scope':'Exploratory matched-prompt video; C/D change only the memory encoder mask input.',
            'certificate_sha256':a.certificate_sha256,'smoke_audit_sha256':digest(a.smoke_audit) if a.smoke_audit else None,
            'checkpoint_sha256':digest(a.checkpoint),'prompt':prompt,'prompt_frame':start,'input_images':records,
            'baseline_completion_path':str(baseline),'baseline_completion_sha256':digest(baseline),
            'prediction_wall_seconds':wall_seconds,
            'timing_scope':'Includes staging, state setup, inference, synchronization, telemetry and mask writes.',
            'runtime_verification_sha256':digest(out/'verification/runtime_verification.json'),
            'rows':rows,'masks':[{'frame':i,'sha256':digest(out/f'{i:06d}.png')} for i in range(end)],
            'trace_sha256':trace_sha,'label_guard':dict(guard),'runner_sha256':digest(Path(__file__))}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':a.sequence,'arm':a.arm,'frames':len(rows)}))


if __name__=='__main__':main()
