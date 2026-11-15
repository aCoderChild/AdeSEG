#!/usr/bin/env python3
"""Freeze native raw decoder candidates from saved fixed-read head inputs.

No dataset labels are opened; native post-suppression head output, pointer,
object score and author IoU ranking are reproduced on each capture device.
This performs an offline decoder replay, not a video or changed-inference run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_memory_capture import clone_data
from scripts.diagnose_lean_raw_decoder_train_reads import select_raw_mask,state_hash
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,source_identity
from scripts.run_lean_offline_memory_ablation import equal
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture
from scripts.train_region_mixed_loss_v1 import mixed_manifest

FORK=ROOT/'external/Lean-SAM2-token-window'


def move(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    if value is None or isinstance(value,(bool,int,float,str)):return value
    raise ValueError('Unexpected nested decoder input')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data-preflight','reference-capture','checkpoint','output'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--device',choices=('cpu','mps'),required=True);a=p.parse_args()
    if a.device=='mps' and (not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0'):
        raise RuntimeError('MPS with fallback and fast math disabled is required')
    manifest,mhash=mixed_manifest(a.data_preflight)
    refpath=a.reference_capture/'channel_feature_receipt.json';ref=json.loads(refpath.read_text());rawid=ref['pipeline']['raw_identity']
    if not ref['passed'] or ref['sequence']!='seq4' or ref['feature_pipeline_sha256']!=manifest['pipeline_identity']['cpu_feature_pipeline_sha256'] or digest(a.checkpoint)!=manifest['checkpoint_sha256']:
        raise ValueError('Capture or model reference differs')
    if subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'],text=True).strip()!=rawid['revision']:
        raise ValueError('Fork revision changed')
    changed=subprocess.check_output(['git','-C',str(FORK),'diff','--name-only','HEAD'],text=True).splitlines()
    if sorted(changed)!=sorted(rawid['patch_sha256']) or any(digest(FORK/k)!=v for k,v in rawid['patch_sha256'].items()):
        raise ValueError('Fork tracked changes differ')
    _,rev=source_identity('lean')
    if rev!=rawid['revision']:raise ValueError('Author revision differs')
    sig=ref['pipeline']['cpu_identity']['source_signature']
    for key,path in (('bypass_sha256',FORK/'checkpoints/bypass_bottleneck_base.pth'),('lean_config_sha256',FORK/'configs/lean_sam2/lean_sam2.yaml')):
        if digest(path)!=sig[key]:raise ValueError('Author model asset changed')
    samples=sorted((x for x in manifest['samples'] if x['capture_device']==a.device),key=lambda x:(x['split'],x['sequence'],x['frame']))
    if len(samples)!=(29 if a.device=='cpu' else 20):raise ValueError('Per-device sample coverage differs')
    for item in samples:
        if digest(item['image'])!=item['image_sha256']:raise ValueError('Input image changed')
        read_capture(item)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if a.device=='mps':torch.mps.manual_seed(20261021)
    sys.path.insert(0,str(FORK));model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device=a.device),FORK)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()) or not model.pred_obj_scores:
        raise ValueError('Model worktree or native presence policy differs')
    before=state_hash(model)
    if a.device=='mps':
        source=manifest['expanded_sources']['seq6']['capture_path'];identity=json.loads(Path(source).read_text())['identity']
        if before!=identity['model_state_sha256']:raise ValueError('Live MPS model state differs from capture')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);folder=out/'raw';folder.mkdir();rows=[]
    with torch.inference_mode():
        for item in samples:
            snap=torch.load(item['snapshot'],map_location='cpu',weights_only=True)
            if snap['frame']!=item['frame'] or snap['decoder_inputs']['multimask_output'] is not True:
                raise ValueError('Original saved head did not use three candidates')
            captured=[];hook=model.sam_mask_decoder.register_forward_hook(lambda _m,_i,value:captured.append(clone_data(value)))
            try:post=clone_data(model._forward_sam_heads(**move(snap['decoder_inputs'],a.device)))
            finally:hook.remove()
            if len(captured)!=1 or not equal(post,snap['decoder_outputs']):
                raise ValueError(f"Original post-suppression head output differs: {item['sequence']}/{item['frame']}")
            raw_masks,raw_ious,_,raw_score=captured[0]
            if raw_masks.shape!=(1,3,256,256) or raw_ious.shape!=(1,3) or not torch.equal(raw_score,post[6]) or not torch.equal(raw_ious,post[2]):
                raise ValueError('Candidate count, IoU or object score differs')
            selected,index=select_raw_mask(raw_masks.float(),raw_ious,True)
            score=float(raw_score[0,0])
            if (score>0 and not torch.equal(selected,post[3])) or (score<=0 and not torch.all(post[3]==-1024)):
                raise ValueError('Native author selection or suppression differs')
            path=folder/f"{item['sequence']}_{item['frame']:06d}.pt"
            torch.save({'sequence':item['sequence'],'frame':item['frame'],'raw_candidates':raw_masks.float(),
                        'predicted_ious':raw_ious.float(),'object_score_logits':raw_score.float(),
                        'selected_index':index,'native_selected':post[3],'native_pointer':post[5]},path)
            rows.append({'split':item['split'],'sequence':item['sequence'],'frame':item['frame'],'capture_device':item['capture_device'],
                         'snapshot_sha256':item['snapshot_sha256'],'raw_file':str(path),'raw_sha256':digest(path),
                         'selected_index':int(index[0]),'object_score':score})
    if state_hash(model)!=before:raise ValueError('Model parameter or buffer changed')
    result={'passed':True,'scope':'Label-free frozen-head replay, not video inference or no-object bypass.',
            'device':a.device,'manifest_sha256':mhash,'checkpoint_sha256':digest(a.checkpoint),
            'reference_capture_sha256':digest(refpath),'fork_revision':rawid['revision'],'fork_patches':rawid['patch_sha256'],
            'decoder_source_sha256':digest(FORK/'sam2/modeling/sam2_base.py'),'runner_sha256':digest(Path(__file__)),
            'selector_source_sha256':digest(ROOT/'scripts/diagnose_lean_raw_decoder_train_reads.py'),
            'protocol_sha256':digest(ROOT/'protocols/RAW_MULTIMASK_REFINER_BRANCH_V1.md'),
            'model_state_sha256':before,'torch':str(torch.__version__),'fallback':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
            'fast_math':os.environ.get('PYTORCH_MPS_FAST_MATH'),'validation_labels_opened':False,'rows':rows}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'device':a.device,'samples':len(rows),
                      'training':sum(x['split']=='train' for x in rows),'validation':sum(x['split']=='validation' for x in rows)}),flush=True)


if __name__=='__main__':main()
