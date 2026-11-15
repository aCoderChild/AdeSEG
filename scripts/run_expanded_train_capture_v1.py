#!/usr/bin/env python3
"""Full label-free MPS reference or observed seq6/13 token-window trajectory."""
import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest,stage_frames
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.lean_mps_decoder_observer import observe_decoder_reads
from modeling.lean_multiscale_anchor import install_multiscale_anchor
from modeling.lean_token_window_storage import TokenWindowRuntime
from modeling.prompt_feature_refinement import reference_prototypes
from scripts.memory_token_io import digest,validate_output
from scripts.preflight_expanded_train_capture_v1 import TARGETS
from scripts.run_author_baseline import build_author_predictor,source_identity

FORK=ROOT/'external/Lean-SAM2-token-window'


def read(p):return json.loads(Path(p).read_text())


def state_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('preflight','reference-capture','mps-full-audit','checkpoint','output'):
        p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--sequence',choices=tuple(TARGETS),required=True)
    p.add_argument('--mode',choices=('reference','capture'),required=True)
    a=p.parse_args()
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0':
        raise RuntimeError('MPS must be available, fallback disabled and fast math disabled')
    pre=read(a.preflight/'manifest.json');done=read(a.preflight/'completion.json')
    if (not done['passed'] or done['manifest_sha256']!=digest(a.preflight/'manifest.json')
            or pre['runner_sha256']!=digest(ROOT/'scripts/preflight_expanded_train_capture_v1.py')
            or pre['protocol_sha256']!=digest(ROOT/'protocols/EXPANDED_TRAIN_CAPTURE_SEQ6_SEQ13_V1.md')
            or pre['data_manifest_sha256']!=digest(ROOT/'data/polypgen_sequence.jsonl')
            or pre['checkpoint_sha256']!=digest(a.checkpoint)):
        raise ValueError('Frozen capture plan changed')
    ref=read(a.reference_capture/'channel_feature_receipt.json');raw=ref['pipeline']['raw_identity']
    if (not ref['passed'] or ref['sequence']!='seq4' or digest(a.reference_capture/'channel_feature_receipt.json')!=pre['reference_capture_sha256']
            or ref['feature_pipeline_sha256']!=pre['cpu_feature_pipeline_sha256']):raise ValueError('CPU training reference changed')
    audit=read(a.mps_full_audit)
    if not audit['passed'] or digest(a.mps_full_audit)!=pre['mps_full_audit_sha256']:
        raise ValueError('MPS engineering audit changed')
    for key,path in (('mps_rope_sha256','modeling/lean_indexed_spatial_rope_mps.py'),
                     ('mps_blocking_sha256','modeling/lean_mps_blocking_pointer_position.py')):
        if audit['identities'][key]!=digest(ROOT/path):raise ValueError('MPS adapter source changed')
    if any(digest(FORK/k)!=v for k,v in raw['patch_sha256'].items()):raise ValueError('Fork patches changed')
    if subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'],text=True).strip()!=raw['revision']:
        raise ValueError('Fork revision changed')
    if any(digest(ROOT/path)!=raw[key] for key,path in (('policy_sha256','modeling/token_memory_window.py'),('runtime_sha256','modeling/lean_token_window_storage.py'))):
        raise ValueError('Token-window code changed')
    _,rev=source_identity('lean')
    if rev!=raw['revision']:raise ValueError('Author revision changed')
    sig=ref['pipeline']['cpu_identity']['source_signature']
    for key,path in (('bypass_sha256',FORK/'checkpoints/bypass_bottleneck_base.pth'),
                     ('lean_config_sha256',FORK/'configs/lean_sam2/lean_sam2.yaml'),
                     ('memory_module_sha256',ROOT/'modeling/lean_multiscale_anchor.py')):
        if digest(path)!=sig[key]:raise ValueError('Runtime asset differs: '+key)
    spec=pre['sequences'][a.sequence];source=Path(spec['source']);folder=source/a.sequence
    if (digest(source/'records.json')!=spec['source_records_sha256'] or digest(folder/'frames.json')!=spec['source_frames_sha256']
            or digest(folder/'prompt.json')!=spec['prompt_sha256'] or spec['targets']!=TARGETS[a.sequence]):
        raise ValueError('Archived prompt or target plan changed')
    prompt=read(folder/'prompt.json');start=prompt['first_detection_index']
    frames=load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev')[a.sequence]
    if start!=spec['prompt_frame'] or len(frames)!=spec['frames'] or any(digest(f.image_path)!=h for f,h in zip(frames,spec['image_sha256'],strict=True)):
        raise ValueError('Input frame source changed')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261021);torch.mps.manual_seed(20261021);np.random.seed(20261021)
    sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='mps'),FORK)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Wrong model worktree')
    if model.num_maskmem!=7 or model.mem_keep_ratio!=.25 or model.insurance_max_frames!=3 or not model.enable_insurance_bank or model.fill_hole_area!=0:
        raise ValueError('Model config differs')
    model.ema_history_replace_old_tail_in_attn=False;model.ema_history_resident_prune=False
    model.token_window_runtime=TokenWindowRuntime(16384,3072)
    selections=[];anchor_events=install_multiscale_anchor(model,.5,selection_events=selections)
    before=state_hash(model);positions=[];observations=[];rows=[]
    with tempfile.TemporaryDirectory(prefix='staged_',dir=out) as temp:
        stage_frames(frames,Path(temp));state=model.init_state(video_path=temp,offload_video_to_cpu=True,offload_state_to_cpu=False)
        try:
            with ExitStack() as stack:
                stack.enter_context(torch.inference_mode());stack.enter_context(indexed_spatial_rope_mps(model,positions))
                stack.enter_context(blocking_mps_pointer_position(model))
                _,ids,logits=model.add_new_points_or_box(state,frame_idx=start,obj_id=1,box=np.asarray(prompt['box'],np.float32))
                if ids!=[1]:raise ValueError('Unexpected object ID')
                _,ids,logits=model.add_new_points_or_box(state,frame_idx=start,obj_id=1,points=np.asarray(prompt['points'],np.float32),labels=np.array([1,0],np.int32),clear_old_points=False)
                if ids!=[1]:raise ValueError('Unexpected object ID')
                initial=logits[0,0].cpu().numpy()>0
                if not initial.any():raise ValueError('Empty automatic conditioning mask')
                _,backbone,_,_,_=model._get_image_feature(state,start,batch_size=1)
                prompt_feature=backbone['backbone_fpn'][0][0].detach().float().cpu()
                prototypes=reference_prototypes(prompt_feature,initial)
                model.add_new_mask(state,frame_idx=start,obj_id=1,mask=initial)
                context=observe_decoder_reads(model,set(spec['targets']),out,prototypes,observations) if a.mode=='capture' else nullcontext()
                stack.enter_context(context)
                for i,ids,output in model.propagate_in_video(state,start_frame_idx=start,reverse=False):
                    if ids!=[1] or i!=start+len(rows):raise ValueError('Video frame order differs')
                    path=out/f'{i:06d}.png';Image.fromarray((output[0,0].cpu().numpy()>0).astype(np.uint8)*255).save(path)
                    rows.append({'frame':i,'mask_sha256':digest(path)})
        finally:model.reset_state(state)
    if len(rows)!=len(frames)-start or state_hash(model)!=before:raise ValueError('Output coverage or model state differs')
    writes=[e for e in model.token_window_runtime.events if e['stage']=='after_write']
    if len(writes)!=len(rows)-1 or any(x['total_tokens']>16384 or x['insurance_tokens']>3072 for x in writes):
        raise ValueError('Window budget or frame coverage differs')
    for item in observations:
        item['feature_sha256']=digest(item['feature_file']);item['snapshot_sha256']=digest(item['snapshot_file'])
    if a.mode=='capture' and {x['frame'] for x in observations}!=set(spec['targets']):raise ValueError('Missing target read')
    identity={'checkpoint_sha256':digest(a.checkpoint),'preflight_sha256':digest(a.preflight/'manifest.json'),
              'cpu_feature_pipeline_sha256':pre['cpu_feature_pipeline_sha256'],'fork_revision':raw['revision'],'fork_patches':raw['patch_sha256'],
              'token_policy_sha256':raw['policy_sha256'],'token_runtime_sha256':raw['runtime_sha256'],
              'mps_indexed_sha256':digest(ROOT/'modeling/lean_indexed_spatial_rope_mps.py'),
              'mps_blocking_sha256':digest(ROOT/'modeling/lean_mps_blocking_pointer_position.py'),
              'observer_sha256':digest(ROOT/'modeling/lean_mps_decoder_observer.py'),
              'anchor_sha256':digest(ROOT/'modeling/lean_multiscale_anchor.py'),
              'runner_sha256':digest(Path(__file__)),'torch':str(torch.__version__),'python':sys.version,
              'platform':platform.platform(),'seed':20261021,'device':'mps','fallback':os.environ['PYTORCH_ENABLE_MPS_FALLBACK'],
              'fast_math':os.environ['PYTORCH_MPS_FAST_MATH'],'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
              'model_state_sha256':before,'capacity':16384,'insurance_reserve':3072,'anchor_strength':.5}
    result={'passed':True,'scope':'Label-free MPS development capture; not full memory-read snapshots or independent evaluation.',
            'mode':a.mode,'sequence':a.sequence,'prompt_frame':start,'targets':spec['targets'],
            'prompt_sha256':spec['prompt_sha256'],'identity':identity,'rows':rows,'positions':positions,'selection':selections,
            'anchor_events':anchor_events,'token_window_events':model.token_window_runtime.events,'observations':observations}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'sequence':a.sequence,'mode':a.mode,'frames':len(rows),'observations':len(observations),
                      'evictions':sum(bool(x['evict_frames']) for x in model.token_window_runtime.events if x['stage']=='admission_plan')}),flush=True)


if __name__=='__main__':main()
