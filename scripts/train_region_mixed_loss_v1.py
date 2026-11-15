#!/usr/bin/env python3
"""Matched CPU/MPS-capture development study; not a full-video experiment.

MPS trains on a composite of previously captured CPU and MPS frozen features.
Loss alternatives are training-only; the inference graph is identical.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import PIL
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import save_binary_mask
from modeling.foreground_conditioned_refiner import ForegroundConditionedRefiner
from modeling.lean_channel_attention import author_sha256
from modeling.robust_segmentation_loss import robust_bce_dice_loss
from scripts.memory_token_io import digest,validate_output
from scripts.preflight_mixed_refiner_v1 import canonical_sha,validate_samples
from scripts.train_multilevel_refiner_seq4_empty_v2 import load_target,read_capture,to_video

ARMS=('region','global','shuffled','mask_only')
LOSSES=('bce','robust_bce')
SEED=20261114
EPOCHS=10
LOSS_FORMULAS={'bce':'mean(BCEWithLogits) + soft Dice',
               'robust_bce':'mean(10 * log1p(pixel_BCEWithLogits / 10)) + soft Dice'}


def dependency_hashes():
    """Bind helpers that define the actual fixed-read inputs and loss targets."""
    paths=('modeling/lean_channel_attention.py','modeling/robust_segmentation_loss.py','datasets/io.py','datasets/video.py',
           'scripts/memory_token_io.py','scripts/train_multilevel_refiner_seq4_empty_v2.py',
           'scripts/train_region_conditioned_hardcase_v1.py','scripts/preflight_mixed_refiner_v1.py',
           'scripts/preflight_expanded_train_capture_v1.py','scripts/score_region_mixed_loss_v1.py',
           'data/polypgen_sequence.jsonl')
    return {name:digest(ROOT/name) for name in paths}


def environment_identity():
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0':
        raise RuntimeError('MPS with fallback and fast math explicitly disabled is required')
    raw=subprocess.check_output(['system_profiler','SPDisplaysDataType','-json'],text=True)
    info=json.loads(raw)
    gpu=sorted(({'name':e.get('sppci_model',e.get('_name')),'cores':e.get('sppci_cores')}
                for e in info.get('SPDisplaysDataType',[]) if e.get('sppci_device_type')=='spdisplays_gpu'),
               key=lambda e:str(e['name']))
    if not gpu or any(not e['name'] for e in gpu):raise ValueError('GPU model unavailable')
    if not torch.are_deterministic_algorithms_enabled():raise ValueError('Deterministic algorithms must be enabled')
    return {'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,
            'torch':str(torch.__version__),'numpy':np.__version__,'pillow':PIL.__version__,
            'mps_device_count':torch.mps.device_count(),'gpu_inventory':gpu,
            'mps_fallback':'0','mps_fast_math':'0','deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}


def mixed_manifest(folder):
    path=folder/'manifest.json';m=json.loads(path.read_text());done=json.loads((folder/'completion.json').read_text())
    if (not done.get('passed') or done.get('manifest_sha256')!=digest(path) or not m.get('passed')
            or m.get('source_sha256')!=digest(ROOT/'scripts/preflight_mixed_refiner_v1.py')
            or canonical_sha(m['pipeline_identity'])!=m['feature_pipeline_sha256']):
        raise ValueError('Mixed manifest or pipeline identity changed')
    for name in ('cpu_manifest','expanded_manifest','split_manifest'):
        if digest(m[name+'_path'])!=m[name+'_sha256']:raise ValueError('Source manifest changed: '+name)
    for source in m['expanded_sources'].values():
        for name in ('samples','reference','capture','verification','prompt_audit'):
            if digest(source[name+'_path'])!=source[name+'_sha256']:raise ValueError('Expanded source changed: '+name)
    validate_samples(m['samples'])
    return m,digest(path)


def model_for(arm):
    if arm not in ARMS:raise ValueError('Unknown study arm')
    return ForegroundConditionedRefiner(gate_mode='global' if arm=='mask_only' else arm)


def state_sha256(model):
    """Hash names, dtype, shapes and bytes of the initialized trainable state."""
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        tensor=value.detach().cpu().contiguous()
        meta=json.dumps({'name':name,'dtype':str(tensor.dtype),'shape':list(tensor.shape)},sort_keys=True,separators=(',',':')).encode()
        h.update(len(meta).to_bytes(8,'big'));h.update(meta);h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def inputs(item,arm,device):
    early,deep,logits,shape=read_capture(item)
    if arm=='mask_only':early,deep=torch.zeros_like(early),torch.zeros_like(deep)
    return early.to(device),deep.to(device),logits.to(device),shape


def pixel_loss(prediction,target,kind):
    if kind not in LOSSES or prediction.shape!=target.shape:
        raise ValueError('Unknown loss or mask geometry')
    if kind=='robust_bce':
        return robust_bce_dice_loss(prediction,target,scale=10.0)
    b=F.binary_cross_entropy_with_logits(prediction,target,reduction='none')
    data=b.mean()
    prob=prediction.sigmoid()
    dice=1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
    return data+dice


def initial_models():
    torch.manual_seed(SEED)
    cpu=model_for('region').train()
    # Exercise gradients beyond the deliberately zero-initialized final head.
    torch.nn.init.normal_(cpu.head[-1].weight,mean=0,std=0.01)
    mps=model_for('region').train();mps.load_state_dict(cpu.state_dict(),strict=True);mps.to('mps')
    return cpu,mps


def parity(manifest,manifest_sha,output):
    cpu,mps=initial_models();checks=[]
    for seq,frame in (('seq4',24),('seq4',21),('seq6',90),('seq13',62)):
        item=next(x for x in manifest['samples'] if x['sequence']==seq and x['frame']==frame and x['split']=='train')
        for kind in LOSSES:
            results=[]
            for model,device in ((cpu,'cpu'),(mps,'mps')):
                model.zero_grad(set_to_none=True)
                early,deep,logits,shape=inputs(item,'region',device);target=load_target(item,shape).to(device)
                loss=pixel_loss(to_video(model(early,deep,logits),shape),target,kind)
                if not torch.isfinite(loss):raise ValueError('Nonfinite parity loss')
                loss.backward()
                gradients=[p.grad.detach().cpu().clone() for p in model.parameters()]
                if not all(torch.isfinite(x).all() for x in gradients):raise ValueError('Nonfinite parity gradient')
                results.append((loss.detach().cpu(),gradients))
            torch.testing.assert_close(results[0][0],results[1][0],rtol=1e-3,atol=1e-3)
            diff=0.0;maximum=0.0
            for a,b in zip(results[0][1],results[1][1],strict=True):
                torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-3)
                diff=max(diff,float((a-b).abs().max()));maximum=max(maximum,float(a.abs().max()))
            checks.append({'sequence':seq,'frame':frame,'capture_device':item['capture_device'],
                           'loss_kind':kind,'cpu_loss':float(results[0][0]),'mps_loss':float(results[1][0]),
                           'maximum_gradient_difference':diff,'maximum_gradient_magnitude':maximum})
    out=validate_output(output);out.mkdir(parents=True,exist_ok=False)
    receipt={'passed':True,'manifest_sha256':manifest_sha,'runner_sha256':digest(Path(__file__)),
             'model_sha256':digest(ROOT/'modeling/foreground_conditioned_refiner.py'),
             'base_model_sha256':digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
             'author_gate_sha256':author_sha256(),'torch':str(torch.__version__),
             'dependency_sha256':dependency_hashes(),'environment':environment_identity(),'checks':checks,
             'scope':'Fixed-read loss/gradient parity only; does not verify full video.'}
    (out/'parity.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)


def validate_parity_checks(checks):
    expected={(seq,frame,kind) for seq,frame in (('seq4',24),('seq4',21),('seq6',90),('seq13',62)) for kind in LOSSES}
    if (len(checks)!=len(expected) or {(x['sequence'],x['frame'],x['loss_kind']) for x in checks}!=expected
            or any(x['capture_device']!=('mps' if x['sequence'] in ('seq6','seq13') else 'cpu')
                   or any(not math.isfinite(x[k]) for k in ('cpu_loss','mps_loss','maximum_gradient_difference','maximum_gradient_magnitude'))
                   for x in checks)):
        raise ValueError('Parity check coverage or numerical diagnostics differ')


def verify_plan(path,manifest_sha,parity_path,arm,kind):
    p=json.loads(path.read_text());receipt=json.loads(parity_path.read_text())
    expected={'manifest_sha256':manifest_sha,'runner_sha256':digest(Path(__file__)),
              'model_sha256':digest(ROOT/'modeling/foreground_conditioned_refiner.py'),
              'base_model_sha256':digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
              'author_gate_sha256':author_sha256(),'torch':str(torch.__version__),
              'dependency_sha256':dependency_hashes(),'environment':environment_identity(),
              'seed':SEED,'epochs':EPOCHS,'arms':list(ARMS),'losses':list(LOSSES),
              'loss_formulas':LOSS_FORMULAS,'optimizer':'Adam','learning_rate':0.0003,'weight_decay':0.0001,
              'parameter_count':48005,'evaluation':'freeze all eight arms before independent label scoring'}
    if any(p.get(k)!=v for k,v in expected.items()) or p.get('parity_sha256')!=digest(parity_path):
        raise ValueError('Frozen plan or source identity differs')
    parity_expected={k:v for k,v in expected.items() if k in ('manifest_sha256','runner_sha256','model_sha256','base_model_sha256','author_gate_sha256','torch','dependency_sha256','environment')}
    if not receipt.get('passed') or any(receipt.get(k)!=v for k,v in parity_expected.items()):raise ValueError('Parity receipt differs')
    validate_parity_checks(receipt.get('checks',[]))
    if arm not in p['arms'] or kind not in p['losses']:raise ValueError('Arm/loss is outside plan')
    return p


def train(manifest,manifest_sha,arm,kind,output,parity_path,plan_path):
    verify_plan(plan_path,manifest_sha,parity_path,arm,kind)
    train_rows=sorted((x for x in manifest['samples'] if x['split']=='train'),key=lambda x:(x['sequence'],x['frame']))
    val_rows=sorted((x for x in manifest['samples'] if x['split']=='validation'),key=lambda x:(x['sequence'],x['frame']))
    for item in manifest['samples']:
        if digest(item['image'])!=item['image_sha256']:raise ValueError('Source image changed')
        read_capture(item)
    for item in train_rows:
        if digest(item['label'])!=item['label_sha256']:raise ValueError('Training label changed')
    out=validate_output(output);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(SEED);torch.mps.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    model=model_for(arm).to('mps').train();optimizer=torch.optim.Adam(model.parameters(),lr=0.0003,weight_decay=0.0001)
    initial_state_sha256=state_sha256(model)
    history=[];steps=0
    for epoch in range(EPOCHS):
        per_sample=[]
        for idx in torch.randperm(len(train_rows)).tolist():
            item=train_rows[idx];early,deep,logits,shape=inputs(item,arm,'mps');target=load_target(item,shape).to('mps')
            optimizer.zero_grad(set_to_none=True)
            loss=pixel_loss(to_video(model(early,deep,logits),shape),target,kind)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradient')
            grad=max(float(p.grad.detach().abs().max().cpu()) for p in model.parameters() if p.grad is not None)
            optimizer.step();steps+=1
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite parameter')
            per_sample.append({'sequence':item['sequence'],'frame':item['frame'],'capture_device':item['capture_device'],
                               'loss':float(loss.detach().cpu()),'maximum_gradient':grad})
        history.append({'epoch':epoch+1,'mean_loss':float(np.mean([x['loss'] for x in per_sample])),'samples':per_sample})
        (out/'progress.json').write_text(json.dumps({'steps':steps,'history':history},indent=2)+'\n')
        print(f'{arm}/{kind} epoch {epoch+1}/{EPOCHS}: {history[-1]["mean_loss"]:.6f}',flush=True)
    payload={'format':'foreground_conditioned_refiner_v1','research_only':True,'training_steps':steps,
             'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
             'backbone_sha256':manifest['checkpoint_sha256'],'feature_pipeline_sha256':manifest['feature_pipeline_sha256'],
             'train_manifest_sha256':manifest_sha,'author_sha256':author_sha256(),
             'conditioned_source_sha256':digest(ROOT/'modeling/foreground_conditioned_refiner.py'),
             'base_source_sha256':digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
             'runner_source_sha256':digest(Path(__file__)),'gate_mode':model.gate_mode,'contrast_head':True,
             'arm':'mask_only' if arm=='mask_only' else 'multilevel','study_arm':arm,'loss_kind':kind,
             'loss_formula':LOSS_FORMULAS[kind],'optimizer':'Adam','learning_rate':0.0003,'weight_decay':0.0001,
             'epochs':EPOCHS,'seed':SEED,'device':'mps','torch':str(torch.__version__),'environment':environment_identity(),
             'dependency_sha256':dependency_hashes(),
             'plan_sha256':digest(plan_path),'parity_sha256':digest(parity_path),
             'parameter_count':sum(p.numel() for p in model.parameters()),'initial_state_sha256':initial_state_sha256,
             'scope':manifest['scope']}
    checkpoint=out/'checkpoint.pt';torch.save(payload,checkpoint)
    fitted=model_for(arm);fitted.load_trained_state(torch.load(checkpoint,map_location='cpu',weights_only=True),
        backbone_sha256=manifest['checkpoint_sha256'],feature_pipeline_sha256=manifest['feature_pipeline_sha256'],train_manifest_sha256=manifest_sha)
    fitted.to('mps').eval();pred_dir=out/'frozen_predictions';pred_dir.mkdir()
    targets=val_rows+[next(x for x in train_rows if x['sequence']=='seq4' and x['frame']==i) for i in (21,24)]
    frozen=[]
    with torch.no_grad():
        for item in targets:
            early,deep,logits,shape=inputs(item,arm,'mps')
            outputs={'baseline':(to_video(logits,shape)>0).cpu().numpy()[0,0],
                     'refined':(to_video(fitted(early,deep,logits),shape)>0).cpu().numpy()[0,0]}
            entry={'sequence':item['sequence'],'frame':item['frame'],'split':item['split'],'shape':list(shape)}
            for name,mask in outputs.items():
                path=pred_dir/f"{item['sequence']}_{item['frame']:06d}_{name}.png";save_binary_mask(mask,path)
                entry[name]=str(path);entry[name+'_sha256']=digest(path)
            frozen.append(entry)
    frozen_path=out/'frozen_predictions.json';frozen_path.write_text(json.dumps({'checkpoint_sha256':digest(checkpoint),'rows':frozen},indent=2)+'\n')
    result={'passed':True,'scope':manifest['scope'],'arm':arm,'loss_kind':kind,'seed':SEED,'epochs':EPOCHS,
            'manifest_sha256':manifest_sha,'plan_sha256':digest(plan_path),'checkpoint_sha256':digest(checkpoint),
            'frozen_predictions_sha256':digest(frozen_path),'initial_state_sha256':initial_state_sha256,'history':history,
            'validation_labels_opened':False}
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'arm':arm,'loss_kind':kind,'steps':steps,'validation_labels_opened':False}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('parity','train'),required=True)
    p.add_argument('--data-preflight',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--loss',choices=LOSSES);p.add_argument('--plan',type=Path);p.add_argument('--parity-receipt',type=Path)
    a=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    environment_identity()
    m,h=mixed_manifest(a.data_preflight)
    if a.mode=='parity':parity(m,h,a.output)
    else:
        if None in (a.arm,a.loss,a.plan,a.parity_receipt):raise ValueError('Training requires arm, loss, plan and parity receipt')
        train(m,h,a.arm,a.loss,a.output,a.parity_receipt,a.plan)


if __name__=='__main__':main()
