#!/usr/bin/env python3
"""Run one frozen-prompt warm-compatibility engineering clip on MPS, no labels."""
import argparse
from contextlib import ExitStack, nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from datasets.video import load_manifest,stage_frames
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.lean_mps_decoder_observer import observe_decoder_reads
from modeling.lean_multiscale_anchor import install_multiscale_anchor
from modeling.lean_token_window_storage import TokenWindowRuntime
from modeling.prompt_feature_refinement import reference_prototypes
from modeling.seq23_viewport import restore_mask,stage_cropped_frames
from modeling.warm_automatic_prompt_v2 import apply_archived_warm_prompt_v2
from scripts.crossfit_observer_capture_shared_v1 import (BASE,DATA,EXISTING,OUT,PLAN_SHA,read,require,
    require_headroom,sequence_inputs,validate_plan,verify_frames)
from scripts.memory_token_io import digest
from scripts.local_import_closure_v1 import local_import_closure
from scripts.run_author_baseline import build_author_predictor
from scripts.run_source_group_capture_v1 import state_hash
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v2 import load_prepared
from scripts.train_region_mixed_loss_v1 import environment_identity

FORK=ROOT/'external/Lean-SAM2-token-window'
WARM=OUT/'crossfit_warm_observer_preflight_v3_attempt01'
WARM_SHA='c77833f050dfafceab78273deb3fce3cd06474232c5c75bee2ee7c2597804f9d'
SOURCES=('scripts/run_seq23_warm_archive_gate_v2.py','modeling/warm_automatic_prompt_v2.py',
         'scripts/run_crossfit_warm_gate_v1.py','modeling/warm_automatic_prompt_v1.py',
         'scripts/preflight_crossfit_warm_observer_v3.py','protocols/EXPLORATORY_CROSSFIT_WARM_OBSERVER_V1.md',
         'scripts/run_source_group_capture_v1.py','scripts/audit_source_group_capture_v1.py',
         'scripts/resume_source_group_capture_v1.py',
         'protocols/EXPLORATORY_CROSSFIT_WARM_GATE_V1.md',
         'scripts/audit_crossfit_warm_gate_v1.py','scripts/run_crossfit_warm_gate_study_v1.py',
         'scripts/audit_seq23_warm_archive_gate_v2.py','scripts/run_seq23_warm_archive_gate_study_v2.py',
         'scripts/audit_crossfit_warm_partial_v1.py',
         'modeling/lean_indexed_spatial_rope_mps.py','modeling/lean_mps_blocking_pointer_position.py',
         'scripts/audit_warm_observation_v1.py','scripts/train_multilevel_refiner_seq4_empty_v2.py',
         'modeling/prompt_feature_refinement.py','scripts/audit_warm_prompt_prototypes_v1.py')
SOURCES=tuple(sorted(set(SOURCES)|{'scripts/local_import_closure_v1.py'}|local_import_closure(ROOT,SOURCES)))


def warm_plan():
    path=WARM/'plan.json';done=read(WARM/'completion.json')
    require(digest(path)==WARM_SHA and done.get('passed') is True and done.get('plan_sha256')==WARM_SHA,
            'Warm preflight changed')
    plan=read(path);require(digest(path)==WARM_SHA and plan.get('passed') is True and
        plan.get('capture_authorized') is False and plan.get('label_guard',{}).get('open_attempts')==0 and
        plan.get('frozen_index_plan_sha256')==PLAN_SHA and
        all(digest(ROOT/p)==sha for p,sha in plan['source_sha256'].items()),'Warm plan sources changed')
    return plan


def verify_old_mps_hooks(identity):
    for key,path in (('mps_indexed_sha256','modeling/lean_indexed_spatial_rope_mps.py'),
                     ('mps_blocking_sha256','modeling/lean_mps_blocking_pointer_position.py')):
        require(identity[key]==digest(ROOT/path),f'Historical MPS adapter differs: {key}')


def gate_inputs(sequence,index,existing,warm):
    if sequence in ('seq6','seq23'):
        frames,records,start,grid,mask,_,_,prompt=sequence_inputs(index,sequence)
        spec=warm['fresh_prompts'][sequence];file=Path(spec['prompt_metadata_path'])
        require(digest(file)==spec['prompt_metadata_sha256'],'Prompt changed');p=read(file)
        require(digest(file)==spec['prompt_metadata_sha256'],'Prompt changed while reading')
        box=p[spec['box_field']];points=p[spec['points_field']]
        targets=grid if sequence=='seq6' else [start+1]
        limit=0 if sequence=='seq6' else 4
        roi=spec['roi_xyxy']
    else:
        require(sequence=='seq3','Undeclared warm gate sequence')
        frames={**load_manifest(DATA,'dev'),**load_manifest(DATA,'test')}[sequence]
        records=index['source_frames'][sequence];verify_frames(frames,records)
        pre=read(existing['preflight_path']);spec=pre['sequences'][sequence]
        folder=Path(spec['source'])/sequence;file=folder/'prompt.json';maskfile=folder/'prompt_refined.png'
        require(digest(file)==spec['archived_prompt_sha256'] and digest(maskfile)==spec['archived_prompt_mask_sha256'],
                'Historical seq3 prompt changed')
        p=read(file);start=p['first_detection_index'];box=p['box'];points=p['points']
        with Image.open(maskfile) as im:mask=np.asarray(im.convert('L'))>0
        require(digest(file)==spec['archived_prompt_sha256'] and digest(maskfile)==spec['archived_prompt_mask_sha256'],
                'Historical seq3 prompt changed while decoding')
        targets=list(range(1,15));limit=0;roi=[0,0,mask.shape[1],mask.shape[0]]
    require(start>=0 and (not limit or start+limit<len(frames)), 'Invalid gate frame range')
    return frames,records,start,targets,mask,box,points,limit,roi


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sequence',choices=('seq23',),required=True)
    parser.add_argument('--mode',choices=('reference','capture'),required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Invalid output')
    guard=install_mask_open_guard(DATA);warm=warm_plan();verify_old_mps_hooks(warm['old_identity'])
    index=validate_plan();existing,manifest_sha=load_prepared(EXISTING)
    require(manifest_sha==warm['existing_manifest_sha256'],'Existing manifest differs')
    flags={'PYTHONHASHSEED':'0','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
    require(all(os.environ.get(k)==v for k,v in flags.items()) and torch.backends.mps.is_available(), 'Required MPS flags differ')
    frames,records,start,targets,mask,box,points,limit,roi=gate_inputs(args.sequence,index,existing,warm)
    require_headroom(BASE,len(targets) if args.mode=='capture' else 0)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    require(environment_identity()==warm['old_identity']['runtime_environment'],'Runtime environment differs from old source-group')
    torch.manual_seed(20261021);torch.mps.manual_seed(20261021);np.random.seed(20261021)
    sys.path.insert(0,str(FORK));checkpoint=Path(existing['checkpoint_path'])
    model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=checkpoint,device='mps'),FORK)
    import sam2
    require(Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()),'Wrong SAM2 worktree')
    require(model.num_maskmem==7 and model.mem_keep_ratio==.25 and model.insurance_max_frames==3 and
            model.enable_insurance_bank and model.fill_hole_area==0,'Model config differs')
    model.ema_history_replace_old_tail_in_attn=False;model.ema_history_resident_prune=False
    model.token_window_runtime=TokenWindowRuntime(16384,3072)
    selection=[];anchor_events=install_multiscale_anchor(model,.5,selection_events=selection)
    before=state_hash(model);require(before==warm['old_identity']['model_state_sha256'],'Model fingerprint differs')
    sources={p:digest(ROOT/p) for p in SOURCES};out.mkdir(parents=True,exist_ok=False)
    positions=[];observations=[];rows=[]
    with tempfile.TemporaryDirectory(prefix='staged_',dir=out) as temp:
        if args.sequence=='seq23':stage_cropped_frames(frames,temp)
        else:stage_frames(frames,Path(temp))
        state=model.init_state(video_path=temp,offload_video_to_cpu=True,offload_state_to_cpu=False)
        require((state['video_height'],state['video_width'])==mask.shape,'Staged prompt geometry differs')
        try:
            with ExitStack() as stack:
                stack.enter_context(torch.inference_mode());stack.enter_context(indexed_spatial_rope_mps(model,positions))
                stack.enter_context(blocking_mps_pointer_position(model));holder={}
                def before_mask(initial):
                    _,backbone,_,_,_=model._get_image_feature(state,start,batch_size=1)
                    early=backbone['backbone_fpn'][0][0].detach().float().cpu()
                    holder['early']=early.detach().clone()
                    holder['prototypes']=reference_prototypes(early,initial)
                decision=apply_archived_warm_prompt_v2(model,state,start,box,points,mask,before_mask=before_mask)
                conditioning=out/'conditioning_mask.png';generated=out/'generated_mask.png';prototypes=out/'prototypes.pt'
                Image.fromarray(decision['conditioning_mask'].astype(np.uint8)*255).save(conditioning)
                Image.fromarray(decision['generated_mask'].astype(np.uint8)*255).save(generated)
                torch.save({'foreground':holder['prototypes'][0],'background':holder['prototypes'][1]},prototypes)
                prompt_early=out/'prompt_early.pt';torch.save({'frame':start,'early':holder['early']},prompt_early)
                conditioning_sha=digest(conditioning);generated_sha=digest(generated);prototypes_sha=digest(prototypes)
                prompt_early_sha=digest(prompt_early)
                stack.enter_context(observe_decoder_reads(model,set(targets),out,holder['prototypes'],observations)
                                    if args.mode=='capture' else nullcontext())
                kwargs={} if not limit else {'max_frame_num_to_track':limit}
                for frame,ids,logits in model.propagate_in_video(state,start_frame_idx=start,reverse=False,**kwargs):
                    require(ids==[1] and frame==start+len(rows) and logits.ndim==4 and
                            tuple(logits.shape[:2])==(1,1) and bool(torch.isfinite(logits).all()),'Invalid output')
                    prediction=logits[0,0].cpu().numpy()>0
                    if args.sequence=='seq23':prediction=restore_mask(prediction)
                    file=out/f'{frame:06d}.png';Image.fromarray(prediction.astype(np.uint8)*255).save(file)
                    rows.append({'frame':frame,'mask_sha256':digest(file)})
        finally:model.reset_state(state)
    torch.mps.synchronize();expected=len(frames)-start if not limit else limit+1
    require(len(rows)==expected and state_hash(model)==before,'Model/trajectory coverage differs')
    require(args.mode!='capture' or [x['frame'] for x in observations]==targets,'Observation grid differs')
    for item in observations:
        for key in ('feature','snapshot'):
            file=Path(item[key+'_file']).resolve(strict=True);require(file.is_relative_to(out),'Tensor path escaped')
            item[key+'_sha256']=digest(file)
    verify_frames(frames,records);require(guard['open_attempts']==0,'Annotation open attempted')
    require(sources=={p:digest(ROOT/p) for p in SOURCES} and digest(WARM/'plan.json')==WARM_SHA,'Source changed')
    receipt={'passed':True,'scope':'Uniform archived-mask warm seq23 MPS no-op gate; no labels or efficacy.','sequence':args.sequence,
        'mode':args.mode,'warm_plan_sha256':WARM_SHA,'source_sha256':sources,'environment_flags':flags,
        'model_state_sha256':before,'process_id':os.getpid(),'run_uuid':uuid.uuid4().hex,
        'prompt_frame':start,'target_frames':targets,'roi_xyxy':roi,'input_images':records,
        'conditioning_sha256':conditioning_sha,'generated_sha256':generated_sha,
        'prototypes_sha256':prototypes_sha,'prompt_early_sha256':prompt_early_sha,
        'generated_equals_archived':decision['generated_equals_archived'],
        'generated_difference_pixels':decision['different_pixels'],
        'prompt_policy':'warm_box_points_then_commit_archived_mask',
        'rows':rows,'positions':positions,'selection':selection,'anchor_events':anchor_events,
        'token_window_events':model.token_window_runtime.events,'observations':observations,'label_guard':dict(guard)}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':args.sequence,'mode':args.mode,'frames':len(rows),'targets':len(observations)}))

if __name__=='__main__':main()
