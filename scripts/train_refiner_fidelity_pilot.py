#!/usr/bin/env python3
"""Exploratory fixed-read residual-fidelity control; no video propagation.

The original refiner and training data are reused.  This pilot adds a soft
training-only penalty for moving predicted probabilities away from the input
mask; it adds no inference threshold, detection, or memory-selection policy.
Validation frames have previously been inspected and are development data.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import save_binary_mask
from modeling.lean_channel_attention import author_sha256
from modeling.multilevel_contrast_refiner import MultilevelContrastRefiner
from scripts.memory_token_io import digest, validate_output
from scripts.train_multilevel_refiner_seq4_empty_v2 import (
    check_manifest, input_features, load_target, loss_fn, read_capture, to_video,
)

WEIGHT = 0.1
SEEDS = (20261114, 20261115, 20261116)


def fidelity_loss(prediction, target, original):
    """Segmentation loss plus a smooth, symmetric probability-change penalty."""
    if prediction.shape != target.shape or prediction.shape != original.shape:
        raise ValueError('Loss tensors must share output geometry')
    fidelity = F.mse_loss(prediction.sigmoid(), original.detach().sigmoid())
    return loss_fn(prediction, target) + WEIGHT * fidelity


def inputs(item, arm, device):
    early, deep, logits, shape = read_capture(item)
    early, deep = input_features(early, deep, arm)
    return early.to(device), deep.to(device), logits.to(device), shape


def parity(manifest, manifest_sha, checkpoint, output):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('arm') != 'multilevel' or payload.get('model_source_sha256') != digest(ROOT/'modeling/multilevel_contrast_refiner.py'):
        raise ValueError('Expected matching trained multilevel pilot checkpoint')
    models = []
    for device in ('cpu', 'mps'):
        model = MultilevelContrastRefiner()
        model.load_trained_state(payload, backbone_sha256=manifest['checkpoint_sha256'],
            feature_pipeline_sha256=manifest['feature_pipeline_sha256'],
            train_manifest_sha256=manifest['original_manifest_sha256'])
        models.append(model.to(device).train())
    samples = [next(x for x in manifest['samples'] if x['sequence']=='seq2' and x['split']=='train'),
               next(x for x in manifest['samples'] if x['sequence']=='seq4' and x['frame']==17)]
    checks=[]
    for item in samples:
        results=[]
        for device, model in zip(('cpu','mps'), models, strict=True):
            model.zero_grad(set_to_none=True)
            early, deep, logits, shape = inputs(item,'multilevel',device)
            target=load_target(item,shape).to(device)
            pred=to_video(model(early,deep,logits),shape)
            original=to_video(logits,shape)
            loss=fidelity_loss(pred,target,original)
            loss.backward()
            results.append((loss.detach().cpu(), [p.grad.detach().cpu() for p in model.parameters()]))
        torch.testing.assert_close(results[0][0],results[1][0],rtol=1e-3,atol=1e-3)
        maximum=0.0
        for c,g in zip(results[0][1],results[1][1],strict=True):
            torch.testing.assert_close(c,g,rtol=1e-3,atol=1e-3)
            maximum=max(maximum,float((c-g).abs().max()))
        checks.append({'sequence':item['sequence'],'frame':item['frame'],
                       'loss_cpu':float(results[0][0]),'loss_mps':float(results[1][0]),
                       'maximum_gradient_difference':maximum})
    out=validate_output(output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'manifest_sha256':manifest_sha,'checkpoint_sha256':digest(checkpoint),
            'runner_sha256':digest(Path(__file__)),'model_sha256':digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
            'base_runner_sha256':digest(ROOT/'scripts/train_multilevel_refiner_seq4_empty_v2.py'),
            'torch':str(torch.__version__),'weight':WEIGHT,'checks':checks,
            'scope':'Fixed-read CPU/MPS training loss and gradients only; not video equivalence.'}
    (out/'parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def train(manifest, manifest_sha, output, arm, seed, parity_path):
    receipt=json.loads(parity_path.read_text())
    expected={'passed':True,'manifest_sha256':manifest_sha,'runner_sha256':digest(Path(__file__)),
              'model_sha256':digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
              'base_runner_sha256':digest(ROOT/'scripts/train_multilevel_refiner_seq4_empty_v2.py'),
              'torch':str(torch.__version__),'weight':WEIGHT}
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('MPS parity receipt differs')
    if seed not in SEEDS or arm not in ('multilevel','mask_only'):raise ValueError('Unregistered pilot setting')
    samples=manifest['samples'];train_rows=sorted((x for x in samples if x['split']=='train'),key=lambda x:(x['sequence'],x['frame']))
    val_rows=sorted((x for x in samples if x['split']=='validation'),key=lambda x:(x['sequence'],x['frame']))
    for row in samples:
        if digest(row['image'])!=row['image_sha256']:raise ValueError('Image changed')
        read_capture(row)
    for row in train_rows:
        if digest(row['label'])!=row['label_sha256']:raise ValueError('Training label changed')
    out=validate_output(output);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);torch.mps.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model=MultilevelContrastRefiner().to('mps').train();optimizer=torch.optim.Adam(model.parameters(),lr=0.0003,weight_decay=0.0001)
    history=[];steps=0
    for epoch in range(30):
        losses=[]
        for index in torch.randperm(len(train_rows)).tolist():
            item=train_rows[index];early,deep,logits,shape=inputs(item,arm,'mps')
            target=load_target(item,shape).to('mps');original=to_video(logits,shape)
            optimizer.zero_grad(set_to_none=True)
            loss=fidelity_loss(to_video(model(early,deep,logits),shape),target,original)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradients')
            optimizer.step();steps+=1;losses.append(float(loss.detach().cpu()))
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
        (out/'progress.json').write_text(json.dumps({'steps':steps,'history':history},indent=2)+'\n')
        print(f'{arm} seed {seed} epoch {epoch+1}/30: {history[-1]["loss"]:.6f}',flush=True)
    payload={'format':'multilevel_contrast_refiner_v1','research_only':True,'training_steps':steps,
             'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'arm':arm,
             'backbone_sha256':manifest['checkpoint_sha256'],'feature_pipeline_sha256':manifest['feature_pipeline_sha256'],
             'train_manifest_sha256':manifest_sha,'author_sha256':author_sha256(),
             'model_source_sha256':digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
             'runner_source_sha256':digest(Path(__file__)),'base_runner_source_sha256':digest(ROOT/'scripts/train_multilevel_refiner_seq4_empty_v2.py'),
             'parity_sha256':digest(parity_path),'device':'mps','seed':seed,'epochs':30,
             'fidelity_weight':WEIGHT,'torch':str(torch.__version__),
             'scope':'Previously inspected fixed-read development data; not a video result or independent test.'}
    cp=out/'checkpoint.pt';torch.save(payload,cp)
    fitted=MultilevelContrastRefiner();fitted.load_trained_state(torch.load(cp,map_location='cpu',weights_only=True),
        backbone_sha256=manifest['checkpoint_sha256'],feature_pipeline_sha256=manifest['feature_pipeline_sha256'],train_manifest_sha256=manifest_sha)
    fitted.to('mps').eval();pred_dir=out/'validation_predictions';pred_dir.mkdir();frozen=[]
    with torch.no_grad():
        for item in val_rows:
            early,deep,logits,shape=inputs(item,arm,'mps')
            pred=(to_video(fitted(early,deep,logits),shape)>0).cpu().numpy()[0,0]
            base=(to_video(logits,shape)>0).cpu().numpy()[0,0]
            stem=f"{item['sequence']}_{item['frame']:06d}";paths={}
            for name,mask in (('refined',pred),('baseline',base)):
                path=pred_dir/f'{stem}_{name}.png';save_binary_mask(mask,path);paths[name]=str(path);paths[name+'_sha256']=digest(path)
            frozen.append({'sequence':item['sequence'],'frame':item['frame'],'shape':list(shape),**paths})
    (out/'frozen_predictions.json').write_text(json.dumps({'checkpoint_sha256':digest(cp),'rows':frozen},indent=2)+'\n')
    rows=[]
    for item,row in zip(val_rows,frozen,strict=True):
        target=load_target(item,tuple(row['shape']))[0,0].numpy().astype(bool);r={'sequence':row['sequence'],'frame':row['frame'],'labeled_present':bool(target.any())}
        for name in ('baseline','refined'):
            pred=np.asarray(Image.open(row[name]).convert('L'))>0;den=int(pred.sum()+target.sum())
            r[name]={'dice':2*int((pred&target).sum())/den if den else 1.0,'predicted_pixels':int(pred.sum())}
        rows.append(r)
    present=[r for r in rows if r['labeled_present']];empty=[r for r in rows if not r['labeled_present']]
    summary={name:{'present_frames':len(present),'mean_present_dice':float(np.mean([r[name]['dice'] for r in present])),
                   'empty_frames':len(empty),'empty_false_alarms':sum(r[name]['predicted_pixels']>0 for r in empty),
                   'mean_empty_predicted_pixels':float(np.mean([r[name]['predicted_pixels'] for r in empty]))} for name in ('baseline','refined')}
    result={'passed':True,'scope':payload['scope'],'manifest_sha256':manifest_sha,'checkpoint_sha256':digest(cp),
            'frozen_predictions_sha256':digest(out/'frozen_predictions.json'),'arm':arm,'seed':seed,'history':history,'rows':rows,'summary':summary}
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('parity','train'),required=True)
    p.add_argument('--data-preflight',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--parity-receipt',type=Path)
    p.add_argument('--arm',choices=('multilevel','mask_only'));p.add_argument('--seed',type=int)
    a=p.parse_args()
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0':raise RuntimeError('MPS without CPU fallback required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    manifest,manifest_sha=check_manifest(a.data_preflight)
    if a.mode=='parity':
        if a.checkpoint is None:raise ValueError('Parity requires existing frozen checkpoint')
        parity(manifest,manifest_sha,a.checkpoint,a.output)
    else:
        if a.parity_receipt is None:raise ValueError('Training requires parity receipt')
        train(manifest,manifest_sha,a.output,a.arm,a.seed,a.parity_receipt)


if __name__=='__main__':main()
