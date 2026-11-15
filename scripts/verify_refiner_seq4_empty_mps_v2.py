#!/usr/bin/env python3
"""CPU/MPS parity for the extended seq4-empty development manifest.

This is an engineering check on one frozen model and 13 captured reads, not
evidence of equivalent full training or propagated-video behavior. It writes a
new receipt only under the configured Google Drive outputs directory.
"""
import argparse
import json
import os
import platform
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.multilevel_contrast_refiner import MultilevelContrastRefiner
from modeling.lean_channel_attention import author_sha256
from scripts.memory_token_io import digest, validate_output
from scripts.train_multilevel_refiner_seq4_empty_v2 import check_manifest, load_target, loss_fn, read_capture, to_video

# Declared before checking results; mask threshold is the baseline's 0-logit
# convention. The rate is a parity acceptance bound, not a new inference gate.
ATOL = 1e-3
RTOL = 1e-3
MAX_MASK_DISAGREEMENT = 1e-4


def verify(data_preflight, checkpoint, output):
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise RuntimeError('Apple MPS must be available with fallback disabled')
    manifest, manifest_sha = check_manifest(data_preflight)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('arm') != 'multilevel' or payload.get('model_source_sha256') != digest(ROOT / 'modeling/multilevel_contrast_refiner.py'):
        raise ValueError('Expected matching multilevel pilot checkpoint and model source')
    cpu = MultilevelContrastRefiner()
    cpu.load_trained_state(payload, backbone_sha256=manifest['checkpoint_sha256'],
                           feature_pipeline_sha256=manifest['feature_pipeline_sha256'],
                           train_manifest_sha256=manifest['original_manifest_sha256'])
    mps = MultilevelContrastRefiner()
    mps.load_trained_state(payload, backbone_sha256=manifest['checkpoint_sha256'],
                           feature_pipeline_sha256=manifest['feature_pipeline_sha256'],
                           train_manifest_sha256=manifest['original_manifest_sha256'])
    mps.to('mps')
    cpu.eval(); mps.eval(); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    val = sorted((x for x in manifest['samples'] if x['split'] == 'validation' or (x['sequence'] == 'seq4' and x['frame'] in (17,18))), key=lambda x:(x['sequence'], x['frame']))
    rows = []
    for item in val:
        early, deep, logits, shape = read_capture(item)
        with torch.no_grad():
            c = to_video(cpu(early, deep, logits), shape)
            g = to_video(mps(early.to('mps'), deep.to('mps'), logits.to('mps')), shape).cpu()
        torch.testing.assert_close(g, c, atol=ATOL, rtol=RTOL)
        disagreement = (c.gt(0) != g.gt(0)).float().mean().item()
        if disagreement > MAX_MASK_DISAGREEMENT:
            raise ValueError('CPU/MPS hard-mask disagreement exceeds the predeclared bound')
        rows.append({'sequence':item['sequence'], 'frame':item['frame'],
                     'max_logit_abs_difference':float((c-g).abs().max()), 'mask_disagreement_fraction':disagreement})
    train = [next(x for x in manifest['samples'] if x['split'] == 'train' and x['sequence'] == 'seq2'),
             next(x for x in manifest['samples'] if x['sequence'] == 'seq4' and x['frame'] == 17)]
    cpu.train(); mps.train()
    training_checks = []
    for item in train:
        early, deep, logits, shape = read_capture(item)
        target = load_target(item, shape)
        losses = []
        for model, device in ((cpu, 'cpu'), (mps, 'mps')):
            model.zero_grad(set_to_none=True)
            inputs = [x.to(device) for x in (early,deep,logits,target)]
            prediction = to_video(model(*inputs[:3]), shape)
            loss = loss_fn(prediction, inputs[3]); loss.backward()
            losses.append(loss.detach().cpu())
        torch.testing.assert_close(losses[1], losses[0], atol=ATOL, rtol=RTOL)
        gradient_max = 0.0
        for (name, p), (other_name, q) in zip(cpu.named_parameters(), mps.named_parameters(), strict=True):
            if name != other_name or p.grad is None or q.grad is None:
                raise ValueError('Parameter gradients differ or are missing')
            qgrad = q.grad.detach().cpu()
            torch.testing.assert_close(qgrad, p.grad, atol=ATOL, rtol=RTOL)
            gradient_max = max(gradient_max, float((p.grad-qgrad).abs().max()))
        training_checks.append({'sequence':item['sequence'], 'frame':item['frame'], 'cpu_loss':float(losses[0]),
                                'mps_loss':float(losses[1]), 'max_gradient_abs_difference':gradient_max})
    out=validate_output(output)
    if out.exists(): raise FileExistsError('Parity output must be new')
    out.parent.mkdir(parents=True,exist_ok=True)
    result = {'passed':True,'manifest_sha256':manifest_sha,'checkpoint_sha256':digest(checkpoint),
              'runner_sha256':digest(ROOT/'scripts/train_multilevel_refiner_seq4_empty_v2.py'),
              'verifier_sha256':digest(Path(__file__)),'torch_version':str(torch.__version__),
              'model_source_sha256':digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
              'author_gate_sha256':author_sha256(), 'platform':platform.platform(), 'machine':platform.machine(),
              'atol':ATOL,'rtol':RTOL,'max_mask_disagreement_fraction':MAX_MASK_DISAGREEMENT,
              'fixed_read_rows':rows, 'training_checks':training_checks,
              'scope':'Engineering parity on fixed reads only; not training or video equivalence.'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-preflight',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    verify(args.data_preflight,args.checkpoint,args.output)
