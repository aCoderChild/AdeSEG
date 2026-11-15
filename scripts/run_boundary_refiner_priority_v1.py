#!/usr/bin/env python3
"""Freeze one user-requested exploratory boundary-refiner video arm on MPS.

The prior fixed-read promotion screen was not passed; the user explicitly
requested this exploratory A/B/C/D video comparison. No target label is an
inference input. Short runs are engineering checks, not efficacy measurements.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import stage_frames
from modeling.lean_boundary_refinement_routing_v1 import route_boundary_refinement
from modeling.lean_boundary_window_runtime_v1 import verified_boundary_mps_window
from modeling.seq23_viewport import VIEWPORT, restore_mask, stage_cropped_frames
from scripts.memory_token_io import digest, validate_output
from scripts.run_verified_refiner_priority_development import priority_inputs
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_token_window_refiner_routing_mps import tensor_hash
from scripts.verify_boundary_video_short_gate_v1 import verify_full_gate


def read(path):
    return json.loads(Path(path).read_text())


def matching_a(folder, *, sequence, prompt_sha, prompt_mask_sha, checkpoint, detector,
               study, view, count, start, limit):
    folder = Path(folder); rec = read(folder/'completion.json')
    expected = {'passed': True, 'arm': 'A', 'sequence': sequence,
                'prompt_sha256': prompt_sha, 'prompt_mask_sha256': prompt_mask_sha,
                'checkpoint_sha256': digest(checkpoint), 'detector_sha256': digest(detector),
                'candidate_checkpoint_sha256': digest(Path(study)/'runs/region_boundary/checkpoint.pt'),
                'control_checkpoint_sha256': digest(Path(study)/'runs/mask_only_boundary/checkpoint.pt'),
                'runner_source_sha256': digest(Path(__file__)), 'view': view,
                'prompt_frame': start, 'full_video': limit == 0, 'limit': limit}
    if any(rec.get(k) != v for k,v in expected.items()):
        raise ValueError('Matched baseline identity differs')
    if ([r['frame'] for r in rec['masks']] != list(range(count)) or
            [r['frame'] for r in rec['post_prompt_state']] != list(range(start, count))):
        raise ValueError('Matched baseline frame coverage differs')
    for r in rec['masks']:
        if digest(folder/f"{r['frame']:06d}.png") != r['sha256']:
            raise ValueError('Matched baseline mask changed')
    for key, file in (('runtime_verification_sha256','verification/runtime_verification.json'),
                      ('memory_events_sha256','memory_events.json'), ('positions_sha256','positions.json'),
                      ('selections_sha256','selections.json'), ('telemetry_sha256','telemetry.json')):
        if digest(folder/file) != rec[key]:
            raise ValueError(f'Matched baseline {file} changed')
    ver = read(folder/'verification/runtime_verification.json')
    if (not ver.get('passed') or ver.get('candidate_checkpoint_sha256') != expected['candidate_checkpoint_sha256']
            or ver.get('control_checkpoint_sha256') != expected['control_checkpoint_sha256']
            or ver.get('source_sha256',{}).get('scripts/run_boundary_refiner_priority_v1.py') != digest(Path(__file__))
            or ver.get('device') != 'mps float32' or ver.get('memory_budget') != 16384
            or ver.get('insurance_reserve') != 3072):
        raise ValueError('Matched baseline runtime differs')
    return rec


def compare_runtime(a, b):
    a,b = Path(a),Path(b)
    old,new = read(a/'runtime_verification.json'),read(b/'runtime_verification.json')
    if old != new or not old.get('passed'):
        raise ValueError('Factorial arm runtime identity differs')
    for folder, rec in ((a,old),(b,new)):
        for name in ('model','refiner'):
            if digest(folder/f'{name}_tensor_fingerprints.json') != rec[name+'_tensor_fingerprints_sha256']:
                raise ValueError('Factorial tensor fingerprint differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('study','audit','split-preflight','checkpoint','detector','baseline-reference','seq4-capture','output'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--sequence', choices=('seq4','seq11','seq20','seq23'), required=True)
    p.add_argument('--arm', choices=('A','B','C','D'), required=True)
    p.add_argument('--seq23-reference', type=Path); p.add_argument('--seq23-backup', type=Path)
    p.add_argument('--matched-a', type=Path)
    p.add_argument('--engineering-gate', type=Path)
    p.add_argument('--limit', type=int, default=0, help='0=full video; positive=post-prompt engineering steps')
    a=p.parse_args()
    if a.limit < 0 or a.limit > 4:
        raise ValueError('Engineering limit must be 1-4 steps; 0 means full video')
    if a.limit == 0 and a.engineering_gate is None:
        raise ValueError('Full video requires the verified short-factorial gate')
    if a.limit != 0 and a.engineering_gate is not None:
        raise ValueError('A short run must not reuse a full-video gate')
    gate = verify_full_gate(a.engineering_gate, study=a.study, audit=a.audit, checkpoint=a.checkpoint,
                            split_preflight=a.split_preflight, baseline_reference=a.baseline_reference,
                            seq4_capture=a.seq4_capture) if a.engineering_gate else None
    guard = install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl')
    frames,prompt,prompt_sha,view,prompt_mask,prompt_provenance = priority_inputs(
        a.sequence,a.split_preflight,a.seq23_reference,a.seq23_backup,a.checkpoint,a.detector)
    start=prompt['first_detection_index']; count = len(frames) if a.limit == 0 else min(len(frames),start+a.limit+1)
    if count <= start + 1:
        raise ValueError('At least one post-prompt frame is required')
    if a.arm != 'A' and a.matched_a is None:
        raise ValueError('B/C/D require a matching frozen A baseline')
    base = matching_a(a.matched_a, sequence=a.sequence,prompt_sha=prompt_sha,
                      prompt_mask_sha=prompt_provenance['prompt_mask_sha256'],checkpoint=a.checkpoint,
                      detector=a.detector,study=a.study,view=view,count=count,start=start,limit=a.limit) if a.arm != 'A' else None
    out=validate_output(a.output); out.mkdir(parents=True,exist_ok=False)
    with verified_boundary_mps_window(study=a.study,audit=a.audit,split_preflight=a.split_preflight,
            checkpoint=a.checkpoint,baseline_reference=a.baseline_reference,seq4_capture=a.seq4_capture,
            output=out/'verification') as (model,refiner,lease):
        if gate is not None and read(out/'verification/runtime_verification.json') != gate['runtime_identity']:
            raise ValueError('Full-run runtime differs from short-factorial gate')
        if base is not None:
            compare_runtime(a.matched_a/'verification',out/'verification')
        for i in range(start):
            with Image.open(frames[i].image_path) as image: width,height=image.size
            Image.fromarray(np.zeros((height,width),dtype=np.uint8)).save(out/f'{i:06d}.png')
        rows=[]; telemetry=[]; inference_start=time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='priority_frames_',dir=out) as temporary:
            if a.sequence == 'seq23': stage_cropped_frames(frames,temporary)
            else: stage_frames(frames,Path(temporary))
            state=model.init_state(video_path=temporary,offload_video_to_cpu=True,offload_state_to_cpu=False)
            if a.sequence == 'seq23' and (state['video_width'],state['video_height']) != (VIEWPORT[2]-VIEWPORT[0],VIEWPORT[3]-VIEWPORT[1]):
                raise ValueError('Seq23 crop geometry changed')
            try:
                with torch.inference_mode(), route_boundary_refinement(model,refiner,a.arm,runtime_lease=lease,telemetry=telemetry):
                    if not prompt_mask.any(): raise ValueError('Automatic prompt is empty')
                    _,ids,_=model.add_new_mask(state,frame_idx=start,obj_id=1,mask=prompt_mask)
                    if ids != [1]: raise ValueError('Unexpected prompt object')
                    kwargs = {} if a.limit == 0 else {'max_frame_num_to_track': a.limit}
                    for frame,ids,logits in model.propagate_in_video(state,start_frame_idx=start,reverse=False,**kwargs):
                        if ids != [1] or frame != start+len(rows) or frame >= count:
                            raise ValueError('Unexpected propagation frame order')
                        mask=logits[0,0].cpu().numpy()>0
                        if a.sequence == 'seq23': mask=restore_mask(mask)
                        file=out/f'{frame:06d}.png'; Image.fromarray(mask.astype(np.uint8)*255).save(file)
                        row={'frame':frame,'mask_sha256':digest(file),'output_logits_sha256':tensor_hash(logits)}
                        group='cond_frame_outputs' if frame==start else 'non_cond_frame_outputs'
                        entry=state['output_dict_per_obj'][0][group][frame]
                        for key in ('pred_masks','maskmem_features','obj_ptr','object_score_logits'):
                            row[key+'_sha256']=tensor_hash(entry[key])
                        rows.append(row)
            finally:
                model.reset_state(state)
        torch.mps.synchronize()
        elapsed_seconds=time.perf_counter()-inference_start
        if len(rows) != count-start or len(telemetry) < len(rows)-1:
            raise ValueError('Incomplete video or telemetry')
        runtime=model.token_window_runtime; writes=[x for x in runtime.events if x['stage']=='after_write']
        if not writes or any(x['total_tokens']>16384 or x['insurance_tokens']>3072 for x in writes):
            raise ValueError('Memory budget exceeded')
        if base is not None:
            for key in ('mask_sha256','pred_masks_sha256','maskmem_features_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                if rows[0][key] != base['post_prompt_state'][0][key]:
                    raise ValueError('Conditioning output or compact state differs from baseline')
            for key in ('pred_masks_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                if rows[1][key] != base['post_prompt_state'][1][key]:
                    raise ValueError('First post-prompt native state differs')
            if a.arm == 'B':
                for row,original in zip(rows,base['post_prompt_state'],strict=True):
                    for key in ('pred_masks_sha256','maskmem_features_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                        if row[key] != original[key]: raise ValueError('Output-only route changed stored state')
                if runtime.events != read(a.matched_a/'memory_events.json'):
                    raise ValueError('Output-only route changed memory events')
        positions,selections=lease.observations()
        if base is not None and a.arm=='B' and (positions != read(a.matched_a/'positions.json') or selections != read(a.matched_a/'selections.json')):
            raise ValueError('Output-only route changed position or selection traces')
        for filename,obj in (('memory_events.json',runtime.events),('positions.json',positions),
                              ('selections.json',selections),('telemetry.json',telemetry)):
            (out/filename).write_text(json.dumps(obj,indent=2)+'\n')
        if guard['open_attempts'] != 0: raise ValueError('A label-mask open was attempted')
        receipt={'passed':True,'arm':a.arm,'sequence':a.sequence,'view':view,'full_video':a.limit==0,'limit':a.limit,
          'scope':'Explicit user-requested exploratory video run; not an independently validated result.',
          'engineering_gate_sha256':digest(a.engineering_gate) if a.engineering_gate else None,
          'prediction_wall_seconds':elapsed_seconds,
          'prediction_timing_scope':'Includes frame staging, model state initialization, inference, telemetry, mask saves and reset; not model-only latency.',
          'prompt_frame':start,'prompt_sha256':prompt_sha,'prompt_mask_sha256':prompt_provenance['prompt_mask_sha256'],
          'prompt_provenance':prompt_provenance,'automatic_box':prompt['box'],'automatic_points':prompt['points'],
          'checkpoint_sha256':digest(a.checkpoint),'detector_sha256':digest(a.detector),
          'candidate_checkpoint_sha256':digest(a.study/'runs/region_boundary/checkpoint.pt'),
          'control_checkpoint_sha256':digest(a.study/'runs/mask_only_boundary/checkpoint.pt'),
          'runtime_verification_sha256':digest(out/'verification/runtime_verification.json'),
          'memory_events_sha256':digest(out/'memory_events.json'),'positions_sha256':digest(out/'positions.json'),
          'selections_sha256':digest(out/'selections.json'),'telemetry_sha256':digest(out/'telemetry.json'),
          'input_images':[{'frame':i,'sha256':digest(f.image_path)} for i,f in enumerate(frames)],
          'masks':[{'frame':i,'sha256':digest(out/f'{i:06d}.png')} for i in range(count)],
          'post_prompt_state':rows,'label_guard':dict(guard),'runner_source_sha256':digest(Path(__file__))}
    (out/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':True,'arm':a.arm,'sequence':a.sequence,'frames':count,'full_video':a.limit==0}))


if __name__ == '__main__': main()
