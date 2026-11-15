#!/usr/bin/env python3
"""Source-group exploratory MPS reference or observed trajectory; not live promotion."""
import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
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
CONFIG=ROOT/'protocols/source_group_capture_v1.json'
from scripts.run_author_baseline import build_author_predictor,source_identity
from scripts.freeze_source_group_runtime_v1 import current_identity,validate_source_freeze
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.source_group_label_guard_v1 import install_mask_open_guard

FORK=ROOT/'external/Lean-SAM2-token-window'


def read(p):return json.loads(Path(p).read_text())


def digest_pipeline(pipeline):
    return hashlib.sha256(json.dumps(pipeline,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def ensure_storage_headroom(free_bytes,minimum_bytes):
    if free_bytes<minimum_bytes:raise ValueError('Free space below conservative 22 GiB guard')


def verify_conditioning_mask(initial,archived):
    if archived.shape!=initial.shape or not np.array_equal(archived,initial):
        raise ValueError('MPS automatic conditioning mask differs from frozen archive')


def verify_limit(sequence,limit,total):
    if limit==0:return total
    if sequence!='seq3' or limit!=5 or total<5:
        raise ValueError('Only the declared five-frame seq3 engineering gate is permitted without a full-run gate')
    return limit


def verify_short_gate(gate,preflight_sha,source_freeze_sha,runner_sha,auditor_sha,environment):
    identity=gate.get('identity',{})
    if (not gate.get('passed') or gate.get('sequence')!='seq3' or gate.get('limit')!=5 or gate.get('full_sequence') is not False
        or gate.get('frames')!=5 or gate.get('targets')!=[1,2,3,4] or gate.get('auditor_sha256')!=auditor_sha
        or identity.get('preflight_sha256')!=preflight_sha or identity.get('source_freeze_sha256')!=source_freeze_sha
        or identity.get('runner_sha256')!=runner_sha or identity.get('runtime_environment')!=environment
        or gate.get('label_guard',{}).get('open_attempts')!=0 or gate.get('observed_tensor_repeat_equal') is not True):
        raise ValueError('A source-bound passing five-frame seq3 engineering gate is required before full capture')
    paths=[Path(gate['reference_receipt_path']),Path(gate['capture_receipt_path']),Path(gate['repeat_capture_receipt_path'])]
    if len({p.resolve() for p in paths})!=3 or any(digest(p)!=gate[k] for p,k in zip(paths,('reference_receipt_sha256','capture_receipt_sha256','repeat_capture_receipt_sha256'),strict=True)):
        raise ValueError('Short gate receipt files changed or alias')
    ref,cap,repeat=(read(p) for p in paths)
    if ref.get('mode')!='reference' or cap.get('mode')!='capture' or repeat.get('mode')!='capture' or any(x.get('sequence')!='seq3' or x.get('identity')!=identity for x in (ref,cap,repeat)):
        raise ValueError('Short gate replay identity differs')
    if (ref.get('limit')!=5 or cap.get('limit')!=5 or ref.get('targets')!=[1,2,3,4] or cap.get('targets')!=[1,2,3,4]
        or repeat.get('limit')!=5 or repeat.get('targets')!=[1,2,3,4] or [x['frame'] for x in ref.get('rows',[])]!=[0,1,2,3,4] or any(sorted(x['frame'] for x in v.get('observations',[]))!=[1,2,3,4] for v in (cap,repeat))):raise ValueError('Short gate coverage differs')
    if any(ref.get('rows')!=v.get('rows') or any(ref.get(k)!=v.get(k) for k in ('positions','selection','anchor_events','token_window_events')) for v in (cap,repeat)):
        raise ValueError('Short gate trajectory changed')
    for path,entry in zip(paths,(ref,cap,repeat),strict=True):
        if digest(path.parent/'conditioning_mask.png')!=entry['conditioning_sha256'] or entry['conditioning_sha256']!=gate['conditioning_sha256']:
            raise ValueError('Short gate conditioning changed')
    for row in ref['rows']:
        for path in paths:
            if digest(path.parent/f"{row['frame']:06d}.png")!=row['mask_sha256']:raise ValueError('Short gate saved mask changed')
    for passpath,entry in zip(paths[1:],(cap,repeat),strict=True):
        for item in entry['observations']:
            for key in ('feature','snapshot'):
                path=Path(item[key+'_file'])
                if not path.resolve().is_relative_to(passpath.parent.resolve()) or digest(path)!=item[key+'_sha256']:
                    raise ValueError('Short gate saved observation changed')


def state_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('preflight','reference-capture','mps-full-audit','source-freeze','checkpoint','output'):
        p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--sequence',required=True)
    p.add_argument('--mode',choices=('reference','capture'),required=True)
    p.add_argument('--limit',type=int,default=0,help='First N frames for a short engineering gate; 0 means full video.')
    p.add_argument('--short-gate',type=Path,help='Required audited seq3 five-frame gate for full capture.')
    a=p.parse_args();label_guard=install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl')
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0':
        raise RuntimeError('MPS must be available, fallback disabled and fast math disabled')
    cfg=read(CONFIG);pre=read(a.preflight/'manifest.json');done=read(a.preflight/'completion.json')
    if a.sequence not in cfg['sequences']:raise ValueError('Sequence absent from frozen plan')
    if (not done['passed'] or done['manifest_sha256']!=digest(a.preflight/'manifest.json')
            or pre['preflight_sha256']!=digest(ROOT/'scripts/preflight_source_group_capture_v1.py')
            or pre['config_sha256']!=digest(CONFIG)
            or pre['manifest_sha256']!=digest(ROOT/'data/polypgen_sequence.jsonl')
            or any(digest(ROOT/path)!=sha for path,sha in pre['source_sha256'].items())
            or pre['checkpoint_sha256']!=digest(a.checkpoint)):
        raise ValueError('Frozen capture plan changed')
    ref=read(a.reference_capture/'channel_feature_receipt.json');raw=ref['pipeline']['raw_identity']
    if (not ref['passed'] or ref['sequence']!='seq4' or digest(a.reference_capture/'channel_feature_receipt.json')!=pre['reference_capture_sha256']
            or ref['feature_pipeline_sha256']!=digest_pipeline(ref['pipeline'])):raise ValueError('CPU training reference changed')
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
    freeze=read(a.source_freeze/'source_freeze.json');freeze_done=read(a.source_freeze/'completion.json')
    validate_source_freeze(freeze,freeze_done,digest(a.source_freeze/'source_freeze.json'),current_identity(a.preflight,a.reference_capture,a.checkpoint))
    _,rev=source_identity('lean')
    if rev!=raw['revision']:raise ValueError('Author revision changed')
    sig=ref['pipeline']['cpu_identity']['source_signature']
    for key,path in (('bypass_sha256',FORK/'checkpoints/bypass_bottleneck_base.pth'),
                     ('lean_config_sha256',FORK/'configs/lean_sam2/lean_sam2.yaml'),
                     ('memory_module_sha256',ROOT/'modeling/lean_multiscale_anchor.py')):
        if digest(path)!=sig[key]:raise ValueError('Runtime asset differs: '+key)
    spec=pre['sequences'][a.sequence];source=Path(spec['source']);folder=source/a.sequence
    if (digest(source/'records.json')!=spec['archived_records_sha256'] or digest(folder/'frames.json')!=spec['archived_frames_sha256']
            or digest(folder/'prompt.json')!=spec['archived_prompt_sha256'] or digest(folder/'prompt_refined.png')!=spec['archived_prompt_mask_sha256'] or spec['targets']!=cfg['sequences'][a.sequence]['targets']):
        raise ValueError('Archived prompt or target plan changed')
    prompt=read(folder/'prompt.json');start=prompt['first_detection_index']
    frames=load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev')[a.sequence]
    if start!=spec['prompt_frame'] or len(frames)!=spec['frames'] or any(digest(f.image_path)!=h for f,h in zip(frames,spec['image_sha256'],strict=True)):
        raise ValueError('Input frame source changed')
    total=len(frames)-start
    expected_frames=verify_limit(a.sequence,a.limit,total)
    run_targets=[f for f in spec['targets'] if start<f<start+expected_frames]
    if not run_targets:raise ValueError('Engineering run has no capture targets')
    out=validate_output(a.output)
    if out.exists():raise FileExistsError(out)
    # Conservative per-run gate; full-batch headroom still requires review.
    ensure_storage_headroom(shutil.disk_usage(out.parent).free,cfg['min_free_bytes'])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);runtime_environment=environment_identity()
    short_gate=None
    if a.limit==0:
        if a.short_gate is None:raise ValueError('Full capture requires short gate')
        short_gate=read(a.short_gate)
        verify_short_gate(short_gate,digest(a.preflight/'manifest.json'),digest(a.source_freeze/'source_freeze.json'),digest(Path(__file__)),digest(ROOT/'scripts/audit_source_group_capture_v1.py'),runtime_environment)
    out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(20261021);torch.mps.manual_seed(20261021);np.random.seed(20261021)
    sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='mps'),FORK)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Wrong model worktree')
    if model.num_maskmem!=7 or model.mem_keep_ratio!=.25 or model.insurance_max_frames!=3 or not model.enable_insurance_bank or model.fill_hole_area!=0:
        raise ValueError('Model config differs')
    model.ema_history_replace_old_tail_in_attn=False;model.ema_history_resident_prune=False
    model.token_window_runtime=TokenWindowRuntime(16384,3072)
    selections=[];anchor_events=install_multiscale_anchor(model,.5,selection_events=selections)
    before=state_hash(model)
    if short_gate is not None and short_gate['identity'].get('model_state_sha256')!=before:raise ValueError('Loaded model state differs from short gate')
    positions=[];observations=[];rows=[]
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
                archived=np.asarray(Image.open(folder/'prompt_refined.png').convert('L'))>0
                verify_conditioning_mask(initial,archived)
                Image.fromarray(initial.astype(np.uint8)*255).save(out/'conditioning_mask.png')
                conditioning_sha256=digest(out/'conditioning_mask.png')
                _,backbone,_,_,_=model._get_image_feature(state,start,batch_size=1)
                prompt_feature=backbone['backbone_fpn'][0][0].detach().float().cpu()
                prototypes=reference_prototypes(prompt_feature,initial)
                model.add_new_mask(state,frame_idx=start,obj_id=1,mask=initial)
                context=observe_decoder_reads(model,set(run_targets),out,prototypes,observations) if a.mode=='capture' else nullcontext()
                stack.enter_context(context)
                for i,ids,output in model.propagate_in_video(state,start_frame_idx=start,reverse=False):
                    if ids!=[1] or i!=start+len(rows):raise ValueError('Video frame order differs')
                    path=out/f'{i:06d}.png';Image.fromarray((output[0,0].cpu().numpy()>0).astype(np.uint8)*255).save(path)
                    rows.append({'frame':i,'mask_sha256':digest(path)})
                    if len(rows)==expected_frames:break
        finally:model.reset_state(state)
    if len(rows)!=expected_frames or state_hash(model)!=before:raise ValueError('Output coverage or model state differs')
    writes=[e for e in model.token_window_runtime.events if e['stage']=='after_write']
    if len(writes)!=len(rows)-1 or any(x['total_tokens']>16384 or x['insurance_tokens']>3072 for x in writes):
        raise ValueError('Window budget or frame coverage differs')
    for item in observations:
        item['feature_sha256']=digest(item['feature_file']);item['snapshot_sha256']=digest(item['snapshot_file'])
    if a.mode=='capture' and {x['frame'] for x in observations}!=set(run_targets):raise ValueError('Missing target read')
    identity={'checkpoint_sha256':digest(a.checkpoint),'preflight_sha256':digest(a.preflight/'manifest.json'),
              'cpu_feature_pipeline_sha256':ref['feature_pipeline_sha256'],'fork_revision':raw['revision'],'fork_patches':raw['patch_sha256'],
              'token_policy_sha256':raw['policy_sha256'],'token_runtime_sha256':raw['runtime_sha256'],
              'mps_indexed_sha256':digest(ROOT/'modeling/lean_indexed_spatial_rope_mps.py'),
              'mps_blocking_sha256':digest(ROOT/'modeling/lean_mps_blocking_pointer_position.py'),
              'observer_sha256':digest(ROOT/'modeling/lean_mps_decoder_observer.py'),
              'anchor_sha256':digest(ROOT/'modeling/lean_multiscale_anchor.py'),
              'runner_sha256':digest(Path(__file__)),'source_freeze_sha256':digest(a.source_freeze/'source_freeze.json'),'runtime_environment':runtime_environment,'torch':str(torch.__version__),'python':sys.version,
              'platform':platform.platform(),'seed':20261021,'device':'mps','fallback':os.environ['PYTORCH_ENABLE_MPS_FALLBACK'],
              'fast_math':os.environ['PYTORCH_MPS_FAST_MATH'],'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
              'model_state_sha256':before,'capacity':16384,'insurance_reserve':3072,'anchor_strength':.5}
    result={'passed':True,'scope':'Label-free MPS development capture; not full memory-read snapshots or independent evaluation.',
            'mode':a.mode,'sequence':a.sequence,'prompt_frame':start,'targets':run_targets,'limit':a.limit,'full_sequence':a.limit==0,
            'label_guard':dict(label_guard),'short_gate_sha256':digest(a.short_gate) if a.short_gate else None,'prompt_sha256':spec['archived_prompt_sha256'],'archived_prompt_mask_sha256':spec['archived_prompt_mask_sha256'],'conditioning_sha256':conditioning_sha256,'identity':identity,'rows':rows,'positions':positions,'selection':selections,
            'anchor_events':anchor_events,'token_window_events':model.token_window_runtime.events,'observations':observations}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'sequence':a.sequence,'mode':a.mode,'frames':len(rows),'observations':len(observations),
                      'evictions':sum(bool(x['evict_frames']) for x in model.token_window_runtime.events if x['stage']=='admission_plan')}),flush=True)


if __name__=='__main__':main()
