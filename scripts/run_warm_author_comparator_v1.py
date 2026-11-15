#!/usr/bin/env python3
"""Run a single author predictor with the same archived warm automatic prompt.

This is a different initialization track from earlier direct-mask comparisons.
SAM2Long remains an offline final-path method. No target label is read.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import stage_frames
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.samurai_mps_kalman import portable_kalman_iou
from modeling.seq23_viewport import restore_mask,stage_cropped_frames
from modeling.prompt_feature_refinement import reference_prototypes
from modeling.warm_author_initialization_v1 import archived_warm_components
from modeling.warm_automatic_prompt_v2 import apply_archived_warm_prompt_v2
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.preflight_warm_author_inputs_v1 import api_evidence,source_hashes as preflight_sources
from scripts.run_additional_author_comparator_study_v2 import CHECKPOINT,DETECTOR,EXTENDED,REMAINING
from scripts.run_additional_author_comparator_v2 import CONFIGS,model_repo,new_output
from scripts.run_author_baseline import build_author_predictor,source_identity
from scripts.run_author_lean_automatic_v1 import DATA,SEQUENCES,inputs
from scripts.run_sam2long_automatic_priority import PATHWAY_CONFIG,audit_pathway_state
from scripts.run_samurai_automatic_priority import model_state_hash
from scripts.source_group_label_guard_v1 import install_mask_open_guard

METHODS=('native','lean','samurai','sam2long')
PREFLIGHT=Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/contrast_refinement_v1/warm_author_comparator_inputs_v1_attempt04/preflight.json')
PREFLIGHT_SHA='248b29e2a8c2b2bc253f158dff429f6892920d68a71a15aca8ef91fad3b237b6'
SOURCES=('scripts/run_warm_author_comparator_v1.py','scripts/verify_warm_author_comparator_gate_v1.py',
         'scripts/audit_warm_author_feature_access_v1.py',
         'scripts/preflight_warm_author_inputs_v1.py','protocols/WARM_MATCHED_AUTHOR_COMPARATORS_V1.md')


def require(value,message):
    if not value:raise ValueError(message)


def tensor_hash(tensor):
    tensor=tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def warm_feature_callback(model,state,frame,mask,record):
    """Mirror project warm feature access; prototype values do not alter logits."""
    before=sorted(state['cached_features'])
    features=model._get_image_feature(state,frame,batch_size=1)
    require(isinstance(features,tuple) and len(features)==5 and isinstance(features[1],dict),
            'Warm image-feature API differs')
    early=features[1]['backbone_fpn'][0][0].detach().float().cpu()
    positive,negative=reference_prototypes(early,mask)
    require(bool(torch.isfinite(early).all()) and bool(torch.isfinite(positive).all()) and
            bool(torch.isfinite(negative).all()),'Nonfinite warm feature/prototype')
    record.update(cache_before=before,cache_after=sorted(state['cached_features']),
                  early_shape=list(early.shape),early_sha256=tensor_hash(early),
                  positive_sha256=tensor_hash(positive),negative_sha256=tensor_hash(negative))


def source_hashes():
    names=set(SOURCES)|{'scripts/local_import_closure_v1.py'}|set(local_import_closure(ROOT,SOURCES))
    return {name:digest(ROOT/name) for name in sorted(names)}


def frozen_preflight():
    require(PREFLIGHT_SHA!='TO_BE_FROZEN' and digest(PREFLIGHT)==PREFLIGHT_SHA,'Warm input preflight not frozen or changed')
    data=json.loads(PREFLIGHT.read_text())
    require(digest(PREFLIGHT)==PREFLIGHT_SHA and data.get('passed') is True and
            data.get('label_guard',{}).get('open_attempts')==0 and data.get('source_sha256')==preflight_sources() and
            data.get('api_evidence')==api_evidence() and data.get('checkpoint_sha256')==digest(CHECKPOINT) and
            data.get('detector_sha256')==digest(DETECTOR) and set(data.get('sequences',{}))==set(SEQUENCES),
            'Warm input preflight sources or coverage differ')
    return data


def identity(method,seq,plan):
    require(method in METHODS and seq in SEQUENCES,'Unsupported method or sequence')
    repo,revision=source_identity('lean') if method=='lean' else model_repo(method)
    config=(repo/'configs/lean_sam2/lean_sam2.yaml') if method=='lean' else CONFIGS[method]
    assets={'config':digest(config)}
    if method=='lean':
        assets.update(bypass=digest(repo/'checkpoints/bypass_bottleneck_base.pth'),
                      runtime=digest(repo/'configs/runtime/lvos.yaml'),
                      backbone_config=digest(repo/'sam2/configs/sam2.1/sam2.1_hiera_b+.yaml'))
    return {'method':method,'sequence':seq,'repo':str(repo.resolve()),'revision':revision,
            'checkpoint_sha256':digest(CHECKPOINT),'detector_sha256':digest(DETECTOR),
            'preflight_sha256':digest(PREFLIGHT),'assets':assets,'sources':source_hashes(),
            'torch':str(torch.__version__),'numpy':np.__version__,'python':sys.version,
            'platform':platform.platform(),'device':'mps','dtype':'float32','seed':20261021,
            'hash_seed':os.environ.get('PYTHONHASHSEED'),'fallback':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
            'fast_math':os.environ.get('PYTORCH_MPS_FAST_MATH')}


def input_data(seq,plan):
    frames,start,mask,provenance=inputs(seq,EXTENDED,REMAINING,CHECKPOINT,DETECTOR)
    item=plan['sequences'][seq];warm=archived_warm_components(seq,start,mask,provenance)
    require(item['prompt_frame']==start and item['prompt']==provenance and item['box']==warm['box'].tolist() and
            item['points']==warm['points'].tolist() and item['point_labels']==[1,0] and
            item['view_shape']==list(mask.shape) and item['image_sha256']==[digest(f.image_path) for f in frames],
            'Warm automatic prompt or image identity differs')
    return frames,start,provenance,warm


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--sequence',choices=SEQUENCES,required=True);p.add_argument('--limit',type=int)
    p.add_argument('--control-no-callback',action='store_true',help='Short feature-access mechanism control only')
    p.add_argument('--gate',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(all(os.environ.get(k)=='0' for k in ('PYTHONHASHSEED','PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH'))
            and torch.backends.mps.is_available(),'Required MPS environment differs')
    require((a.limit is None)==(a.gate is not None) and (a.limit is None or a.limit>=5),
            'Full run requires a matching gate; smoke needs at least five frames')
    require(not a.control_no_callback or (a.limit is not None and a.sequence=='seq6'),
            'No-callback control is restricted to the short seq6 mechanism check')
    out=new_output(a.output);guard=install_mask_open_guard(DATA);plan=frozen_preflight()
    frames,start,provenance,warm=input_data(a.sequence,plan);ident=identity(a.method,a.sequence,plan)
    end=len(frames) if a.limit is None else min(len(frames),start+a.limit)
    require(end-start>=5,'Insufficient smoke coverage')
    if a.gate:
        from scripts.verify_warm_author_comparator_gate_v1 import verify_gate
        gate=verify_gate(a.gate,ident,plan)
    else:gate=None
    out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(20261021);torch.mps.manual_seed(20261021)
    torch.use_deterministic_algorithms(True)
    repo,_=source_identity('lean') if a.method=='lean' else model_repo(a.method)
    sys.path.insert(0,str(repo/'sam2') if a.method=='samurai' else str(repo))
    model=build_author_predictor(SimpleNamespace(method='sam2' if a.method=='native' else a.method,
        checkpoint=CHECKPOINT,device='mps'),repo)
    import sam2
    require(Path(sam2.__file__).resolve().is_relative_to(repo.resolve()) and model.fill_hole_area==0,'Wrong predictor')
    if a.method=='lean':
        require(model.num_maskmem==7 and model.mem_keep_ratio==.25 and model.insurance_max_frames==3 and
                model.enable_insurance_bank,'Author Lean config differs')
    if a.method=='sam2long':
        require(model.num_maskmem==7 and model.memory_temporal_stride_for_eval==1 and model.iou_prediction_use_sigmoid,
                'SAM2Long author configuration differs')
    state_before=model_state_hash(model)
    if gate is not None:require(gate['state_sha256']==state_before,'Live model differs from smoke')
    rows=[];positions=[];kalman=[];warm_feature={};start_time=time.monotonic()
    for i in range(start):
        with Image.open(frames[i].image_path) as image:width,height=image.size
        file=out/f'{i:06d}.png';Image.fromarray(np.zeros((height,width),np.uint8)).save(file)
        rows.append({'frame':i,'source_frame_index':frames[i].frame_index,'image_sha256':digest(frames[i].image_path),
                     'mask_sha256':digest(file),'post_prompt':False})
    with tempfile.TemporaryDirectory(prefix='frames_',dir=out) as tmp:
        if a.sequence=='seq23':stage_cropped_frames(frames[:end],tmp)
        else:stage_frames(frames[:end],Path(tmp))
        state=model.init_state(video_path=tmp,offload_video_to_cpu=True,offload_state_to_cpu=False)
        require(state['num_frames']==end and (state['video_height'],state['video_width'])==warm['mask'].shape,
                'Staged frames or warm prompt geometry differ')
        if a.method=='sam2long':state.update(PATHWAY_CONFIG)
        try:
            with ExitStack() as stack:
                stack.enter_context(torch.inference_mode())
                if a.method=='lean':stack.enter_context(indexed_spatial_rope_mps(model,positions))
                if a.method in ('native','lean'):stack.enter_context(blocking_mps_pointer_position(model))
                if a.method=='samurai':stack.enter_context(portable_kalman_iou(model,'mps',kalman))
                callback=None if a.control_no_callback else lambda mask:warm_feature_callback(model,state,start,mask,warm_feature)
                decision=apply_archived_warm_prompt_v2(model,state,start,warm['box'],warm['points'],warm['mask'],
                    before_mask=callback)
                if a.method=='sam2long':
                    ids,logits_list=model.propagate_in_video(state,start_frame_idx=start,reverse=False)
                    require(ids==[1] and len(logits_list)==end-start,'SAM2Long coverage differs')
                    pathway=audit_pathway_state(state,start,end-1)
                    outputs=((start+i,ids,logits) for i,logits in enumerate(logits_list))
                else:
                    pathway=None;outputs=model.propagate_in_video(state,start_frame_idx=start,reverse=False)
                for frame,ids,logits in outputs:
                    require(frame==len(rows) and ids==[1] and logits.ndim==4 and tuple(logits.shape[:2])==(1,1)
                            and bool(torch.isfinite(logits).all()),'Output order, identity or finiteness differs')
                    mask=logits[0,0].float().cpu().numpy()>0
                    if a.sequence=='seq23':mask=restore_mask(mask)
                    with Image.open(frames[frame].image_path) as image:shape=(image.height,image.width)
                    require(mask.shape==shape,'Output geometry differs')
                    file=out/f'{frame:06d}.png';Image.fromarray(mask.astype(np.uint8)*255).save(file)
                    rows.append({'frame':frame,'source_frame_index':frames[frame].frame_index,
                                 'image_sha256':digest(frames[frame].image_path),'mask_sha256':digest(file),
                                 'logits_sha256':tensor_hash(logits),'post_prompt':True})
        finally:model.reset_state(state)
    torch.mps.synchronize();wall_seconds=time.monotonic()-start_time
    require(len(rows)==end and model_state_hash(model)==state_before and guard['open_attempts']==0,
            'Output coverage, weights or annotation guard differs')
    result={'passed':True,'scope':'Warm automatic prompt; source-level plan, MPS behavior must be separately validated.',
            'method':a.method,'sequence':a.sequence,'identity':ident,'prompt':provenance,'prompt_frame':start,
            'full_video':a.limit is None,'state_sha256':state_before,'gate_sha256':digest(a.gate) if a.gate else None,
            'noncausal':a.method=='sam2long','process_id':os.getpid(),'generated_equals_archived':decision['generated_equals_archived'],
            'generated_difference_pixels':decision['different_pixels'],'warm_feature_callback':warm_feature,
            'feature_callback_enabled':not a.control_no_callback,
            'positions':positions,'kalman_events':kalman,'prediction_wall_seconds':wall_seconds,
            'timing_scope':'Includes staging, warm prompt, feature telemetry, propagation, synchronization and mask writes; excludes model loading.',
            'pathway_audit':pathway,'label_guard':dict(guard),'records':rows}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'method':a.method,'sequence':a.sequence,'frames':len(rows)}))


if __name__=='__main__':main()
