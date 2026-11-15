#!/usr/bin/env python3
"""Audit source-group observer against its matched full reference trajectory."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.train_multilevel_refiner_seq4_empty_v2 import to_video
from scripts.freeze_source_group_runtime_v1 import current_identity,validate_source_freeze
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def read(path):return json.loads(Path(path).read_text())


def verify_trajectory(ref,cap,spec):
    total=spec['frames']-spec['prompt_frame'];limit=ref.get('limit')
    if not isinstance(limit,int) or cap.get('limit')!=limit or (limit!=0 and (ref.get('sequence')!='seq3' or limit!=5 or total<5)):raise ValueError('Engineering limit differs')
    count=limit or total;targets=[x for x in spec['targets'] if spec['prompt_frame']<x<spec['prompt_frame']+count]
    if not targets or ref.get('full_sequence')!=(limit==0) or cap.get('full_sequence')!=(limit==0) or ref['prompt_frame']!=spec['prompt_frame'] or ref['targets']!=targets or cap['targets']!=targets:
        raise ValueError('Capture target plan differs')
    if len(ref['rows'])!=count or len(cap['rows'])!=count or [x['frame'] for x in ref['rows']]!=list(range(spec['prompt_frame'],spec['prompt_frame']+count)):
        raise ValueError('Trajectory coverage differs')
    if any(ref[key]!=cap[key] for key in ('rows','positions','selection','anchor_events','token_window_events')):
        raise ValueError('Observer changed masks or memory decisions')
    return limit,targets


def verify_observation_coverage(obs,targets):
    if sorted(x['frame'] for x in obs)!=targets:raise ValueError('Observed target coverage differs')


def verify_sample_file(path,root,expected_sha):
    if not path.resolve().is_relative_to(root.resolve()) or digest(path)!=expected_sha:
        raise ValueError('Observed sample changed')


def exact_nested(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact_nested(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(exact_nested(x,y) for x,y in zip(a,b,strict=True))
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    return type(a) is type(b) and a==b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('preflight','reference','capture','source-freeze','reference-capture','checkpoint','output'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--repeat-capture',type=Path,help='Second independently observed pass; required for the short gate.')
    a=p.parse_args();label_guard=install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl');torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0':raise RuntimeError('MPS fallback and fast math must be disabled')
    pre=read(a.preflight/'manifest.json');done=read(a.preflight/'completion.json')
    if not done['passed'] or done['manifest_sha256']!=digest(a.preflight/'manifest.json'):raise ValueError('Preflight changed')
    freeze=read(a.source_freeze/'source_freeze.json');freeze_done=read(a.source_freeze/'completion.json')
    validate_source_freeze(freeze,freeze_done,digest(a.source_freeze/'source_freeze.json'),current_identity(a.preflight,a.reference_capture,a.checkpoint))
    ref=read(a.reference/'receipt.json');cap=read(a.capture/'receipt.json')
    if (not ref['passed'] or not cap['passed'] or ref['mode']!='reference' or cap['mode']!='capture'
            or ref['sequence']!=cap['sequence'] or ref['identity']!=cap['identity']
            or ref['identity']['preflight_sha256']!=digest(a.preflight/'manifest.json')
            or ref['identity']['runner_sha256']!=digest(ROOT/'scripts/run_source_group_capture_v1.py')
            or ref['identity']['observer_sha256']!=digest(ROOT/'modeling/lean_mps_decoder_observer.py')
            or ref['identity']['source_freeze_sha256']!=digest(a.source_freeze/'source_freeze.json')
            or ref['identity']['runtime_environment']!=environment_identity()):
        raise ValueError('Reference/capture runtime identities differ')
    if (ref.get('conditioning_sha256')!=cap.get('conditioning_sha256') or ref.get('archived_prompt_mask_sha256')!=cap.get('archived_prompt_mask_sha256') or digest(a.reference/'conditioning_mask.png')!=ref.get('conditioning_sha256') or digest(a.capture/'conditioning_mask.png')!=cap.get('conditioning_sha256')):
        raise ValueError('Initial conditioning mask differs')
    if any(x.get('label_guard',{}).get('open_attempts')!=0 or x.get('label_guard',{}).get('blocked_paths')!=label_guard['blocked_paths'] for x in (ref,cap)):
        raise ValueError('Capture label-open guard differs')
    spec=pre['sequences'][ref['sequence']]
    archived=Path(spec['source'])/ref['sequence']/'prompt_refined.png'
    if digest(archived)!=spec['archived_prompt_mask_sha256'] or not np.array_equal(np.asarray(Image.open(archived).convert('L'))>0,np.asarray(Image.open(a.capture/'conditioning_mask.png').convert('L'))>0):raise ValueError('Conditioning mask differs from archived prompt')
    limit,targets=verify_trajectory(ref,cap,spec)
    repeat=None
    if limit and a.repeat_capture is None:raise ValueError('Short engineering gate requires a second observed pass')
    if a.repeat_capture is not None:
        if len({a.reference.resolve(),a.capture.resolve(),a.repeat_capture.resolve()})!=3:raise ValueError('Reference and observed pass paths must be distinct')
        repeat=read(a.repeat_capture/'receipt.json')
        if not repeat.get('passed') or repeat.get('mode')!='capture' or repeat.get('sequence')!=ref['sequence'] or repeat.get('identity')!=ref['identity'] or repeat.get('conditioning_sha256')!=ref['conditioning_sha256'] or repeat.get('label_guard',{}).get('open_attempts')!=0 or repeat.get('label_guard',{}).get('blocked_paths')!=label_guard['blocked_paths'] or digest(a.repeat_capture/'conditioning_mask.png')!=repeat.get('conditioning_sha256'):
            raise ValueError('Repeated observed pass identity differs')
        verify_trajectory(ref,repeat,spec);verify_observation_coverage(repeat['observations'],targets)
    for row in ref['rows']:
        for folder in ((a.reference,a.capture,a.repeat_capture) if repeat is not None else (a.reference,a.capture)):
            if digest(folder/f"{row['frame']:06d}.png")!=row['mask_sha256']:raise ValueError('Saved trajectory mask changed')
    writes=[x for x in ref['token_window_events'] if x['stage']=='after_write']
    if len(writes)!=len(ref['rows'])-1 or any(x['total_tokens']>16384 or x['insurance_tokens']>3072 for x in writes):
        raise ValueError('Window budget or coverage differs')
    obs=cap['observations']
    verify_observation_coverage(obs,targets)
    for item in obs:
        feature=Path(item['feature_file']);snapshot=Path(item['snapshot_file'])
        verify_sample_file(feature,a.capture,item['feature_sha256']);verify_sample_file(snapshot,a.capture,item['snapshot_sha256'])
        f=torch.load(feature,map_location='cpu',weights_only=True);s=torch.load(snapshot,map_location='cpu',weights_only=True)
        if repeat is not None:
            other=next(x for x in repeat['observations'] if x['frame']==item['frame'])
            rf=Path(other['feature_file']);rs=Path(other['snapshot_file'])
            verify_sample_file(rf,a.repeat_capture,other['feature_sha256']);verify_sample_file(rs,a.repeat_capture,other['snapshot_sha256'])
            if not exact_nested(f,torch.load(rf,map_location='cpu',weights_only=True)) or not exact_nested(s,torch.load(rs,map_location='cpu',weights_only=True)):
                raise ValueError('Independently captured feature/head tensors differ')
        if (f['frame']!=item['frame'] or s['frame']!=item['frame'] or f['feature'].shape!=(32,256,256)
                or f['foreground'].shape!=(32,) or f['background'].shape!=(32,)
                or not torch.equal(f['feature'].unsqueeze(0),s['decoder_inputs']['high_res_features'][0])
                or s['decoder_outputs'][3].shape!=(1,1,256,256)
                or f['feature'].dtype!=torch.float32 or s['decoder_inputs']['backbone_features'].dtype!=torch.float32 or s['decoder_outputs'][3].dtype!=torch.float32
                or any(not torch.isfinite(x).all() for x in (f['feature'],s['decoder_inputs']['backbone_features'],s['decoder_outputs'][3]))):
            raise ValueError('Observed decoder sample geometry differs')
        with torch.inference_mode():
            video=to_video(s['decoder_outputs'][3].float().to('mps'),(s['video_height'],s['video_width']))
            recomputed=(video>0).cpu().numpy()[0,0]
        original=np.asarray(Image.open(a.capture/f"{item['frame']:06d}.png").convert('L'))>0
        if not np.array_equal(recomputed,original):raise ValueError('Observed decoder output does not reproduce video mask')
    result={'passed':True,'scope':('Full trajectory' if limit==0 else 'Bounded short engineering clip')+' reference/observer mask and trajectory parity plus captured-head reconstruction; unobserved reference feature tensors were not captured or compared. No patient independence or complete attention capture claim.',
            'sequence':ref['sequence'],'prompt_frame':ref['prompt_frame'],'targets':targets,'limit':limit,'full_sequence':limit==0,'frames':len(ref['rows']),
            'max_total_tokens':max(x['total_tokens'] for x in writes),'evictions':sum(bool(x['evict_frames']) for x in ref['token_window_events'] if x['stage']=='admission_plan'),
            'identity':ref['identity'],'reference_receipt_path':str((a.reference/'receipt.json').resolve()),'capture_receipt_path':str((a.capture/'receipt.json').resolve()),'reference_receipt_sha256':digest(a.reference/'receipt.json'),'capture_receipt_sha256':digest(a.capture/'receipt.json'),
            'repeat_capture_receipt_path':str((a.repeat_capture/'receipt.json').resolve()) if repeat is not None else None,'repeat_capture_receipt_sha256':digest(a.repeat_capture/'receipt.json') if repeat is not None else None,'observed_tensor_repeat_equal':repeat is not None,
            'auditor_sha256':digest(Path(__file__)),'label_guard':dict(label_guard),'conditioning_sha256':ref['conditioning_sha256']}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','sequence','frames','targets','max_total_tokens','evictions')}),flush=True)


if __name__=='__main__':main()
