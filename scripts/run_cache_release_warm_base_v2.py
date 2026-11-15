#!/usr/bin/env python3
"""Versioned warm capture derivative: mask agreement is assessed by the separate bounded audit.

Derived from run_crossfit_warm_video_v2.py; source, prompt, shape, trace and
conditioning checks are retained. Prediction mask hashes are not required to
be byte-identical between fresh same-MPS processes.
"""
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
    require_headroom,sequence_inputs,validate_plan,verify_frames,phase_artifact,archived_frame_metadata)
from scripts.memory_token_io import digest
from scripts.run_mps_memory_diagnostic_v3 import record_setup_cache, record_memory
from scripts.run_author_baseline import build_author_predictor
from scripts.run_source_group_capture_v1 import state_hash
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.certify_crossfit_warm_gate_v2 import GATE_SOURCES,VIDEO_SOURCES,verify_certificate
from scripts.train_region_mixed_loss_v1 import environment_identity

FORK=ROOT/'external/Lean-SAM2-token-window'
WARM=OUT/'crossfit_warm_observer_preflight_v3_attempt01'
WARM_SHA='c77833f050dfafceab78273deb3fce3cd06474232c5c75bee2ee7c2597804f9d'
SOURCES=(*GATE_SOURCES,*VIDEO_SOURCES)


def bounded_base_source_hashes():
    result={p:digest(ROOT/p) for p in SOURCES}
    result['scripts/run_cache_release_warm_base_v2.py']=digest(Path(__file__))
    return result


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


def video_inputs(sequence,index,warm):
    from scripts.crossfit_observer_capture_shared_v1 import FRESH
    require(sequence in FRESH,'Sequence is not part of the 143-read plan')
    frames,records,start,targets,mask,_,_,_=sequence_inputs(index,sequence)
    spec=warm['fresh_prompts'][sequence];file=Path(spec['prompt_metadata_path'])
    require(digest(file)==spec['prompt_metadata_sha256'],'Prompt changed');prompt=read(file)
    require(digest(file)==spec['prompt_metadata_sha256'] and prompt['first_detection_index']==start,
            'Prompt changed during parsing')
    return frames,records,start,targets,mask,prompt[spec['box_field']],prompt[spec['points_field']],spec['roi_xyxy']



def reference_receipt(folder,sequence,warm_sha,certificate_sha,sources):
    folder=Path(folder).resolve(strict=True);path=folder/'receipt.json'
    expected=phase_artifact(folder.parent.parent,f'{sequence}_reference',path);rec=read(path)
    require(digest(path)==expected and rec.get('passed') is True and rec.get('mode')=='reference' and
            rec.get('sequence')==sequence and rec.get('warm_gate_sha256')==warm_sha and rec.get('certificate_sha256')==certificate_sha and
            rec.get('source_sha256')==sources and rec.get('label_guard',{}).get('open_attempts')==0,
            'Warm reference receipt differs')
    return rec,expected


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sequence',required=True)
    parser.add_argument('--mode',choices=('reference','capture'),required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--certificate',type=Path,required=True);parser.add_argument('--reference',type=Path)
    parser.add_argument('--batch-root',type=Path,required=True)
    parser.add_argument('--short-gate',type=Path,required=True);parser.add_argument('--cache-frames',type=int,required=True)
    args=parser.parse_args();require((args.mode=='capture')==(args.reference is not None),'Observer requires one reference')
    from scripts.run_cache_release_full_v2 import validate_supervision, assert_live_supervisor
    validate_supervision(sequence=args.sequence,mode=args.mode,output=args.output,certificate_path=args.certificate,
                         short_gate_path=args.short_gate,reference=args.reference,cache_frames=args.cache_frames)
    out=args.output.resolve();require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Invalid output')
    batch=args.batch_root.resolve(strict=True); manifest_path=batch/'run_manifest.json'
    manifest_sha=digest(manifest_path); manifest=read(manifest_path)
    require(args.sequence in ('seq8','seq11','seq16','seq17','seq18','seq19','seq21') and out==batch/args.sequence/args.mode and
            digest(manifest_path)==manifest_sha and manifest.get('sequence')==args.sequence, 'Invalid cache-release batch')
    min_ram=manifest['min_available_ram_bytes']; max_rss=manifest['max_process_rss_bytes']
    require(type(min_ram) is int and min_ram>=4*1024**3 and type(max_rss) is int and 0<max_rss<=6*1024**3,
            'Invalid cache-release resources')
    out.parent.mkdir(parents=True,exist_ok=True)
    setup_file=out.parent/(args.mode+'_setup_cache.jsonl'); frame_file=out.parent/(args.mode+'_frame_cache.jsonl')
    require(not setup_file.exists() and not frame_file.exists(), 'Cache-release telemetry already exists')
    def setup(name):
        assert_live_supervisor(manifest)
        record_setup_cache(setup_file,name,'clear_idle_cache',min_ram,max_rss)
        assert_live_supervisor(manifest)
    guard=install_mask_open_guard(DATA);warm=warm_plan();verify_old_mps_hooks(warm['old_identity'])
    index=validate_plan();certificate,certificate_sha=verify_certificate(args.certificate)
    gate_sha=certificate['warm_gate_sha256']
    flags={'PYTHONHASHSEED':'0','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
    require(all(os.environ.get(k)==v for k,v in flags.items()) and torch.backends.mps.is_available(), 'Required MPS flags differ')
    frames,records,start,targets,mask,box,points,roi=video_inputs(args.sequence,index,warm)
    require_headroom(BASE,len(targets) if args.mode=='capture' else 0)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    require(environment_identity()==warm['old_identity']['runtime_environment'],'Runtime environment differs from old source-group')
    torch.manual_seed(20261021);torch.mps.manual_seed(20261021);np.random.seed(20261021)
    sys.path.insert(0,str(FORK));checkpoint=Path(certificate['checkpoint_path'])
    model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=checkpoint,device='mps'),FORK)
    setup('after_model_load')
    import sam2
    require(Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()),'Wrong SAM2 worktree')
    require(model.num_maskmem==7 and model.mem_keep_ratio==.25 and model.insurance_max_frames==3 and
            model.enable_insurance_bank and model.fill_hole_area==0,'Model config differs')
    model.ema_history_replace_old_tail_in_attn=False;model.ema_history_resident_prune=False
    model.token_window_runtime=TokenWindowRuntime(16384,3072)
    selection=[];anchor_events=install_multiscale_anchor(model,.5,selection_events=selection)
    before=state_hash(model);require(before==warm['old_identity']['model_state_sha256'],'Model fingerprint differs')
    sources=bounded_base_source_hashes()
    ref,ref_sha=reference_receipt(args.reference,args.sequence,gate_sha,certificate_sha,sources) if args.reference else (None,None)
    out.mkdir(parents=True,exist_ok=False)
    positions=[];observations=[];rows=[]
    with tempfile.TemporaryDirectory(prefix='staged_',dir=out) as temp:
        if args.sequence=='seq23':stage_cropped_frames(frames,temp)
        else:stage_frames(frames,Path(temp))
        state=model.init_state(video_path=temp,offload_video_to_cpu=True,offload_state_to_cpu=False)
        setup('after_state_init')
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
                setup('after_prompt')
                conditioning=out/'conditioning_mask.png';generated=out/'generated_mask.png';prototypes=out/'prototypes.pt'
                Image.fromarray(decision['conditioning_mask'].astype(np.uint8)*255).save(conditioning)
                Image.fromarray(decision['generated_mask'].astype(np.uint8)*255).save(generated)
                torch.save({'foreground':holder['prototypes'][0],'background':holder['prototypes'][1]},prototypes)
                prompt_early=out/'prompt_early.pt';torch.save({'frame':start,'early':holder['early']},prompt_early)
                conditioning_sha=digest(conditioning);generated_sha=digest(generated);prototypes_sha=digest(prototypes)
                prompt_early_sha=digest(prompt_early)
                stack.enter_context(observe_decoder_reads(model,set(targets),out,holder['prototypes'],observations)
                                    if args.mode=='capture' else nullcontext())
                for frame,ids,logits in model.propagate_in_video(state,start_frame_idx=start,reverse=False):
                    require(ids==[1] and frame==start+len(rows) and logits.ndim==4 and
                            tuple(logits.shape[:2])==(1,1) and bool(torch.isfinite(logits).all()),'Invalid output')
                    prediction=logits[0,0].cpu().numpy()>0
                    if args.sequence=='seq23':prediction=restore_mask(prediction)
                    file=out/f'{frame:06d}.png';Image.fromarray(prediction.astype(np.uint8)*255).save(file)
                    rows.append({'frame':frame,'mask_sha256':digest(file)})
                    assert_live_supervisor(manifest)
                    record_memory(frame_file,frame,'clear_idle_cache',min_ram,max_rss)
                    assert_live_supervisor(manifest)
        finally:model.reset_state(state)
    torch.mps.synchronize();expected=len(frames)-start
    require(len(rows)==expected and state_hash(model)==before,'Model/trajectory coverage differs')
    require(args.mode!='capture' or [x['frame'] for x in observations]==targets,'Observation grid differs')
    for item in observations:
        for key in ('feature','snapshot'):
            file=Path(item[key+'_file']).resolve(strict=True);require(file.is_relative_to(out),'Tensor path escaped')
            item[key+'_sha256']=digest(file)
    if ref is not None:
        require([r['frame'] for r in rows]==[r['frame'] for r in ref['rows']] and
                positions==ref['positions'] and selection==ref['selection'] and
                anchor_events==ref['anchor_events'] and model.token_window_runtime.events==ref['token_window_events'] and
                conditioning_sha==ref['conditioning_sha256'], 'Warm observer changed reference trajectory')
        require(digest(Path(args.reference)/'receipt.json')==ref_sha,'Warm reference changed')
    verify_frames(frames,records);require(guard['open_attempts']==0 and digest(manifest_path)==manifest_sha,'Annotation open or manifest changed')
    require(sources==bounded_base_source_hashes() and digest(WARM/'plan.json')==WARM_SHA,'Source changed')
    receipt={'passed':True,'scope':'Full warm reference or observer; no labels or efficacy.','sequence':args.sequence,
        'mode':args.mode,'bounded_base_source_sha256':digest(Path(__file__)),'warm_plan_sha256':WARM_SHA,'warm_gate_sha256':gate_sha,
        'certificate_path':str(args.certificate.resolve()),'certificate_sha256':certificate_sha,
        'reference_receipt_sha256':ref_sha,'source_sha256':sources,'environment_flags':flags,
        'model_state_sha256':before,'process_id':os.getpid(),'run_uuid':uuid.uuid4().hex,
        'prompt_frame':start,'target_frames':targets,'view':warm['fresh_prompts'][args.sequence]['view'],
        'roi_xyxy':roi,'input_images':records,'archived_frame_metadata':archived_frame_metadata(index,args.sequence),
        'conditioning_sha256':conditioning_sha,'generated_sha256':generated_sha,'prototypes_sha256':prototypes_sha,'prompt_early_sha256':prompt_early_sha,
        'generated_equals_archived':decision['generated_equals_archived'],
        'generated_difference_pixels':decision['different_pixels'],
        'prompt_policy':'warm_box_points_then_commit_archived_mask',
        'rows':rows,'cache_setup_sha256':digest(setup_file),'cache_frame_sha256':digest(frame_file),
        'positions':positions,'selection':selection,'anchor_events':anchor_events,
        'token_window_events':model.token_window_runtime.events,'observations':observations,'label_guard':dict(guard)}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':args.sequence,'mode':args.mode,'frames':len(rows),'targets':len(observations)}))

if __name__=='__main__':main()
