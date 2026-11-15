#!/usr/bin/env python3
"""Matched fixed-read development study of predicted-region channel weighting.

The extra hard cases are label-selected training samples; seq9/10 validation
was previously inspected. This is not a video or independent evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import save_binary_mask
from modeling.foreground_conditioned_refiner import ForegroundConditionedRefiner
from modeling.lean_channel_attention import author_sha256
from scripts.memory_token_io import digest, validate_output
from scripts.preflight_refiner_hardcase_v3 import validate_samples
from scripts.train_multilevel_refiner_seq4_empty_v2 import load_target, loss_fn, read_capture, to_video

ARMS = ('region', 'global', 'shuffled', 'mask_only')
SEED = 20261114
EPOCHS = 10


def load_manifest(folder):
    path = folder / 'manifest.json'
    manifest = json.loads(path.read_text())
    completion = json.loads((folder / 'completion.json').read_text())
    if (not manifest['passed'] or not completion['passed'] or completion['manifest_sha256'] != digest(path)
            or manifest['source_sha256'] != digest(ROOT / 'scripts/preflight_refiner_hardcase_v3.py')
            or manifest['previous_manifest_sha256'] != digest(manifest['previous_manifest_path'])
            or manifest['extra_receipt_sha256'] != digest(manifest['extra_receipt_path'])):
        raise ValueError('Hard-case manifest or provenance changed')
    validate_samples(manifest['samples'])
    previous = json.loads(Path(manifest['previous_manifest_path']).read_text())
    by_key = {(x['sequence'], x['frame']): x for x in manifest['samples']}
    if any(by_key.get((x['sequence'], x['frame'])) != x for x in previous['samples']):
        raise ValueError('Previous split or sample identity changed')
    extra = json.loads(Path(manifest['extra_receipt_path']).read_text())
    if not extra.get('passed') or extra.get('targets') != [19, 21, 24]:
        raise ValueError('Hard-case capture receipt differs')
    for frame in (21, 24):
        item = by_key[('seq4', frame)]
        entry = next(x for x in extra['frames'] if x['frame'] == frame)
        if item['feature_sha256'] != entry['feature_sha256'] or item['snapshot_sha256'] != entry['snapshot_sha256']:
            raise ValueError('Hard-case capture hashes differ')
    return manifest, digest(path)


def make_model(arm):
    if arm not in ARMS:
        raise ValueError('Unknown study arm')
    return ForegroundConditionedRefiner(gate_mode='global' if arm == 'mask_only' else arm)


def inputs(item, arm, device):
    early, deep, logits, shape = read_capture(item)
    if arm == 'mask_only':
        early, deep = torch.zeros_like(early), torch.zeros_like(deep)
    return early.to(device), deep.to(device), logits.to(device), shape


def parity(manifest, manifest_sha, output):
    torch.manual_seed(SEED)
    cpu = make_model('region').train()
    # Exercise nonzero paths through the gate and both projected features, not
    # only the zero-initialized residual head used at training start.
    torch.nn.init.normal_(cpu.head[-1].weight, mean=0, std=0.01)
    gpu = make_model('region').train()
    gpu.load_state_dict(cpu.state_dict(), strict=True)
    gpu.to('mps')
    checks = []
    for frame in (17, 21, 24):
        item = next(x for x in manifest['samples'] if x['sequence'] == 'seq4' and x['frame'] == frame)
        results = []
        for model, device in ((cpu, 'cpu'), (gpu, 'mps')):
            model.zero_grad(set_to_none=True)
            early, deep, logits, shape = inputs(item, 'region', device)
            target = load_target(item, shape).to(device)
            prediction = to_video(model(early, deep, logits), shape)
            loss = loss_fn(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite parity loss')
            loss.backward()
            gradients = [p.grad.detach().cpu().clone() for p in model.parameters()]
            if not all(torch.isfinite(x).all() for x in gradients):
                raise ValueError('Nonfinite parity gradient')
            results.append((loss.detach().cpu(), gradients))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-3, atol=1e-3)
        max_diff, max_grad = 0.0, 0.0
        for a, b in zip(results[0][1], results[1][1], strict=True):
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
            max_diff = max(max_diff, float((a-b).abs().max()))
            max_grad = max(max_grad, float(a.abs().max()))
        checks.append({'frame': frame, 'cpu_loss': float(results[0][0]), 'mps_loss': float(results[1][0]),
                       'max_gradient_difference': max_diff, 'max_gradient_magnitude': max_grad})
    out = validate_output(output)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'passed': True, 'manifest_sha256': manifest_sha, 'runner_sha256': digest(Path(__file__)),
               'model_sha256': digest(ROOT/'modeling/foreground_conditioned_refiner.py'),
               'base_sha256': digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
               'author_sha256': author_sha256(), 'torch': str(torch.__version__), 'checks': checks,
               'scope': 'Fixed-read loss/gradient engineering parity; not video equivalence.'}
    (out/'parity.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2), flush=True)


def train(manifest, manifest_sha, arm, output, parity_path):
    receipt = json.loads(parity_path.read_text())
    expected = {'passed': True, 'manifest_sha256': manifest_sha, 'runner_sha256': digest(Path(__file__)),
                'model_sha256': digest(ROOT/'modeling/foreground_conditioned_refiner.py'),
                'base_sha256': digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
                'author_sha256': author_sha256(), 'torch': str(torch.__version__)}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('MPS parity receipt changed')
    rows = manifest['samples']
    train_rows = sorted((x for x in rows if x['split'] == 'train'), key=lambda x:(x['sequence'],x['frame']))
    val_rows = sorted((x for x in rows if x['split'] == 'validation'), key=lambda x:(x['sequence'],x['frame']))
    for item in rows:
        if digest(item['image']) != item['image_sha256']:
            raise ValueError('Image changed')
        read_capture(item)
    for item in train_rows:
        if digest(item['label']) != item['label_sha256']:
            raise ValueError('Training label changed')
    out = validate_output(output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED); torch.mps.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    model = make_model(arm).to('mps').train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
    steps, history = 0, []
    for epoch in range(EPOCHS):
        epoch_rows = []
        for index in torch.randperm(len(train_rows)).tolist():
            item = train_rows[index]
            early, deep, logits, shape = inputs(item, arm, 'mps')
            target = load_target(item, shape).to('mps')
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(to_video(model(early, deep, logits), shape), target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite training gradient')
            max_grad = max(float(p.grad.detach().abs().max().cpu()) for p in model.parameters() if p.grad is not None)
            optimizer.step(); steps += 1
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite trained parameter')
            epoch_rows.append({'sequence': item['sequence'], 'frame': item['frame'],
                               'loss': float(loss.detach().cpu()), 'maximum_gradient': max_grad})
        history.append({'epoch': epoch+1, 'mean_loss': float(np.mean([x['loss'] for x in epoch_rows])), 'samples': epoch_rows})
        (out/'progress.json').write_text(json.dumps({'steps': steps, 'history': history}, indent=2)+'\n')
        print(f'{arm} epoch {epoch+1}/{EPOCHS}: {history[-1]["mean_loss"]:.6f}', flush=True)
    payload = {'format':'foreground_conditioned_refiner_v1', 'research_only':True, 'training_steps':steps,
               'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
               'backbone_sha256':manifest['checkpoint_sha256'], 'feature_pipeline_sha256':manifest['feature_pipeline_sha256'],
               'train_manifest_sha256':manifest_sha, 'author_sha256':author_sha256(),
               'conditioned_source_sha256':digest(ROOT/'modeling/foreground_conditioned_refiner.py'),
               'base_source_sha256':digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
               'runner_source_sha256':digest(Path(__file__)), 'gate_mode':model.gate_mode,
               'contrast_head':True, 'arm':'mask_only' if arm=='mask_only' else 'multilevel',
               'study_arm':arm, 'epochs':EPOCHS, 'seed':SEED, 'device':'mps', 'torch':str(torch.__version__),
               'parity_sha256':digest(parity_path), 'parameter_count':sum(p.numel() for p in model.parameters()),
               'scope':manifest['scope']}
    checkpoint = out/'checkpoint.pt'; torch.save(payload, checkpoint)
    fitted = make_model(arm)
    fitted.load_trained_state(torch.load(checkpoint, map_location='cpu', weights_only=True),
        backbone_sha256=manifest['checkpoint_sha256'], feature_pipeline_sha256=manifest['feature_pipeline_sha256'], train_manifest_sha256=manifest_sha)
    fitted.to('mps').eval()
    prediction_dir = out/'frozen_predictions'; prediction_dir.mkdir()
    # Validation and selected training-debug masks are frozen before scoring.
    to_score = val_rows + [next(x for x in train_rows if x['sequence']=='seq4' and x['frame']==i) for i in (21,24)]
    frozen = []
    with torch.no_grad():
        for item in to_score:
            early, deep, logits, shape = inputs(item, arm, 'mps')
            refined = (to_video(fitted(early, deep, logits), shape) > 0).cpu().numpy()[0,0]
            baseline = (to_video(logits, shape) > 0).cpu().numpy()[0,0]
            entry = {'sequence':item['sequence'], 'frame':item['frame'], 'split':item['split'], 'shape':list(shape)}
            for name, mask in (('baseline',baseline),('refined',refined)):
                path = prediction_dir/f"{item['sequence']}_{item['frame']:06d}_{name}.png"
                save_binary_mask(mask,path); entry[name]=str(path); entry[name+'_sha256']=digest(path)
            frozen.append(entry)
    (out/'frozen_predictions.json').write_text(json.dumps({'checkpoint_sha256':digest(checkpoint),'rows':frozen},indent=2)+'\n')
    scores = []
    for item, entry in zip(to_score, frozen, strict=True):
        target = load_target(item,tuple(entry['shape']))[0,0].numpy().astype(bool)
        row = {'sequence':item['sequence'], 'frame':item['frame'], 'split':item['split'], 'present':bool(target.any())}
        for name in ('baseline','refined'):
            pred=np.asarray(Image.open(entry[name]).convert('L'))>0
            den=int(pred.sum()+target.sum())
            row[name]={'dice':2*int((pred&target).sum())/den if den else 1.0,'predicted_pixels':int(pred.sum())}
        scores.append(row)
    val_scores=[x for x in scores if x['split']=='validation']
    present=[x for x in val_scores if x['present']]; empty=[x for x in val_scores if not x['present']]
    summary={name:{'mean_present_dice':float(np.mean([x[name]['dice'] for x in present])), 'present_frames':len(present),
                   'empty_false_alarms':sum(x[name]['predicted_pixels']>0 for x in empty), 'empty_frames':len(empty),
                   'mean_empty_predicted_pixels':float(np.mean([x[name]['predicted_pixels'] for x in empty]))} for name in ('baseline','refined')}
    result={'passed':True, 'scope':manifest['scope'], 'arm':arm, 'seed':SEED, 'epochs':EPOCHS,
            'manifest_sha256':manifest_sha, 'checkpoint_sha256':digest(checkpoint),
            'frozen_predictions_sha256':digest(out/'frozen_predictions.json'), 'history':history, 'rows':scores, 'summary':summary}
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('parity','train'),required=True)
    parser.add_argument('--data-preflight',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--parity-receipt',type=Path)
    args=parser.parse_args()
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0':
        raise RuntimeError('MPS without CPU fallback required')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    manifest, manifest_sha=load_manifest(args.data_preflight)
    if args.mode=='parity':
        parity(manifest,manifest_sha,args.output)
    else:
        if args.arm is None or args.parity_receipt is None:
            raise ValueError('Training needs arm and completed parity receipt')
        train(manifest,manifest_sha,args.arm,args.output,args.parity_receipt)


if __name__=='__main__':
    main()
